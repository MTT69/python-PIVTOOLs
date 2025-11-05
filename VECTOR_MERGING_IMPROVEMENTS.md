# Vector Merging Improvements - Production Ready

## Summary
Successfully upgraded the vector merging system to production-ready status with multiprocessing support, proper error handling, and improved UI.

## Backend Changes (`src/vector_merging/app/views.py`)

### 1. Added Multiprocessing Support
- Added `concurrent.futures.ProcessPoolExecutor` for parallel frame processing
- Added `time` module for timing and progress estimation
- Created `_process_single_frame_merge()` helper function for parallel execution

### 2. New Endpoints

#### `/merge_vectors/merge_one` (POST)
- Merges vectors for a single frame
- Immediate response (no background job)
- Useful for testing and verification
- Parameters:
  - `base_path_idx`: Index of the base path
  - `cameras`: List of camera numbers to merge
  - `frame_idx`: Frame number to merge
  - `type_name`: Type of data (default: "instantaneous")
  - `endpoint`: Optional endpoint path
  - `image_count`: Total number of images

#### `/merge_vectors/merge_all` (POST)
- Merges all frames using multiprocessing
- Background job with progress tracking
- Uses `ProcessPoolExecutor` for parallel processing
- Default 4 workers (configurable via `max_workers` parameter)
- Parameters: Same as merge_one, but without `frame_idx`

### 3. Enhanced Status Endpoint
- `/merge_vectors/status/<job_id>` now includes:
  - `elapsed_time`: Time since job started
  - `estimated_remaining`: Estimated time to completion (when running)
  - Progress percentage
  - Frames processed vs total

### 4. Key Features
- **Automatic run detection**: Finds all non-empty passes in vector files
- **Smart coordinate handling**: Saves merged coordinates automatically
- **Error recovery**: Continues processing even if individual frames fail
- **Memory efficient**: Processes frames in parallel without loading all into memory
- **Production logging**: Comprehensive logging with loguru

## Frontend Changes (`VectorViewer.tsx`)

### 1. Created Inline Hook `useVectorMerging`
- Replaced missing external import with inline implementation
- Manages merging state and job polling
- Supports both single frame and batch merging
- Automatic camera selection (defaults to first 2 cameras)

### 2. Updated UI
- **Two-button approach**:
  - "Merge Frame X": Test on current frame only
  - "Merge All (N)": Process all frames with multiprocessing
- **Better visual feedback**:
  - Progress bar with frame count
  - Success/failure messages
  - Status indicators
  - Timing information
- **Camera selection checkboxes**: Select which cameras to merge
- **Proper error handling**: Shows errors with retry option

### 3. Fixed Type Issues
- Changed camera type from `string` to `number` throughout
- Added proper TypeScript types for all state variables
- Fixed array operations to work with numbers

## How It Works

### Merging Process
1. **Discovery**: Finds all non-empty passes in the first vector file
2. **Setup**: Creates output directory in "Merged" folder structure
3. **Parallel Processing**: 
   - Prepares arguments for all frames
   - Submits to `ProcessPoolExecutor`
   - Processes multiple frames simultaneously
4. **Data Loading**: For each frame:
   - Loads vectors from each camera
   - Loads coordinates for all runs
5. **Merging**: For each run in each frame:
   - Extracts valid (non-NaN) data points
   - Interpolates to common grid using `scipy.interpolate.griddata`
   - Creates distance-based weights (higher at center, lower at edges)
   - Blends overlapping regions using weighted average
6. **Saving**: Saves merged result in MATLAB format matching expected structure
7. **Coordinates**: Saves merged coordinate system once at the end

### Distance-Based Weighting
- Uses sine-based Hanning-like weights
- Gives more weight to data from camera center
- Smoothly blends in overlap regions
- Reduces edge artifacts

## Usage

### From UI:
1. Select multiple cameras (at least 2)
2. Choose "Merge Frame X" to test current frame, OR
3. Choose "Merge All" to process all frames
4. Monitor progress bar
5. When complete, enable "Use Merged Data" to view results

### From API:
```python
# Merge single frame
response = requests.post(f"{backend}/merge_vectors/merge_one", json={
    "base_path_idx": 0,
    "cameras": [1, 2],
    "frame_idx": 100,
    "image_count": 1000,
    "type_name": "instantaneous"
})

# Merge all frames
response = requests.post(f"{backend}/merge_vectors/merge_all", json={
    "base_path_idx": 0,
    "cameras": [1, 2],
    "image_count": 1000,
    "type_name": "instantaneous",
    "max_workers": 4  # Optional, default is 4
})

job_id = response.json()["job_id"]

# Check status
status = requests.get(f"{backend}/merge_vectors/status/{job_id}")
```

## Performance
- **Multiprocessing**: Processes 4 frames simultaneously by default
- **Scalable**: Adjust `max_workers` based on CPU cores
- **Progress tracking**: Real-time updates every second
- **Time estimation**: Shows estimated completion time

## Error Handling
- Validates camera count (need at least 2)
- Checks for valid runs in vector files
- Continues on individual frame failures
- Comprehensive error messages
- Retry functionality in UI

## File Structure
```
base_path/
  instantaneous/
    Merged/
      data/
        00001.mat  # Merged vectors
        00002.mat
        ...
        coordinates.mat  # Merged coordinate system
```

## Testing Recommendations
1. Start with "Merge Frame 1" to verify setup
2. Check merged coordinates are reasonable
3. Visualize merged result before running full batch
4. Use smaller max_workers if memory constrained
5. Monitor logs for any warnings

## Next Steps
- Consider adding merge quality metrics
- Add option to choose interpolation method
- Support for more than 2 cameras
- Adaptive grid spacing based on overlap
- Option to merge statistics (not just instantaneous)

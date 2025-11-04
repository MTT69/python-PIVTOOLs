# Coordinate System and Indexing Fixes

## Overview
This document describes the comprehensive fixes applied to resolve coordinate system mismatches and indexing confusion in the PIV pipeline. The goal was to match MATLAB's coordinate conventions while using C-contiguous (row-major) memory layout for performance.

## Coordinate System Convention

### Established Standard
- **Origin**: Bottom-left at (0,0) - matches how humans view images
- **X-axis**: Horizontal, increases to the right (width dimension, columns)
- **Y-axis**: Vertical, increases upward (height dimension, rows)
- **Array indexing**: `array[row, col]` or `array[y, x]` or `array[height_idx, width_idx]`

### Key Relationships
```
Nx = W = image_shape[1] = number of columns = width
Ny = H = image_shape[0] = number of rows = height

win_ctrs_x[i] = X-coordinate (column position, width dimension)
win_ctrs_y[j] = Y-coordinate (row position, height dimension)
```

## Memory Layout Change: Fortran to C-Contiguous

### Previous (MATLAB-like, Column-major)
- **Fortran/MATLAB order**: Column-major, first dimension varies fastest
- **Linear index**: `index = row + col * nRows`
- **Macro**: `SUB2IND_2D(i, j, M) = ((i) + (j)*(M))` where M = leading dimension

### New (C-contiguous, Row-major)
- **C/Python order**: Row-major, last dimension varies fastest
- **Linear index**: `index = row * nCols + col`
- **Macro**: `SUB2IND_2D(i, j, N) = ((i)*(N) + (j))` where N = number of columns
- **Benefits**:
  - Native Python/NumPy memory layout
  - No costly transpose operations
  - Better cache locality for row-wise access
  - Eliminates Fortran-contiguous conversion overhead

## Changes Made

### 1. Python Code (`cpu_instantaneous.py`)

#### Window Center Calculation
```python
# MATLAB reference:
# win_ctrs_x = 0.5 + wsize(1)/2 : win_spacing_x : Nx - wsize(1)/2 + 0.5
# win_ctrs_y = 0.5 + wsize(2)/2 : win_spacing_y : Ny - wsize(2)/2 + 0.5

# Fixed Python implementation:
H, W = config.image_shape  # (rows, cols) = (height, width)
Ny = H  # height dimension
Nx = W  # width dimension
win_height, win_width = config.window_sizes[pass_idx]

# Window centers span correct dimensions
first_ctr_x = 0.5 + win_width / 2   # Start in width dimension
last_ctr_x = Nx - win_width / 2 + 0.5
first_ctr_y = 0.5 + win_height / 2  # Start in height dimension
last_ctr_y = Ny - win_height / 2 + 0.5
```

#### Array Format Changes
- **Before**: `np.asfortranarray()` everywhere
- **After**: `np.ascontiguousarray()` everywhere
- **ctypes argtypes**: Changed from `F_CONTIGUOUS` to `C_CONTIGUOUS`

### 2. C Library (`common.h`)

#### Indexing Macro Update
```c
// Before (column-major):
#define SUB2IND_2D(i, j, M) ((i) + (j)*(M))

// After (row-major):
#define SUB2IND_2D(i, j, N) ((i)*(N) + (j))
// where N = number of columns (width)
// i = row index (Y-coordinate, height)
// j = column index (X-coordinate, width)
```

### 3. PIV Cross-Correlation (`PIV_2d_cross_correlate.c`)

#### Window Indexing Fix
```c
// Before (confused indexing):
ii = iWindowIdx % nWindows[0];
jj = ((iWindowIdx - ii) % (nWindows[0]*nWindows[1])) / nWindows[0];
xmin = (int)floor(fWinCtrsY[ii] - ...);  // WRONG: Y used for X
ymin = (int)floor(fWinCtrsX[jj] - ...);  // WRONG: X used for Y

// After (correct row-major indexing):
ii = iWindowIdx % nWindows[1];  // Column index (X)
jj = iWindowIdx / nWindows[1];  // Row index (Y)
row_min = (int)floor(fWinCtrsY[jj] - ...);  // Correct: Y for rows
col_min = (int)floor(fWinCtrsX[ii] - ...);  // Correct: X for cols
```

#### Window Extraction Fix
```c
// Extract window with proper row-major indexing
for(int row_win = 0; row_win < nWindowSize[0]; ++row_win) {
    int row_img = row_min + row_win;
    for(int col_win = 0; col_win < nWindowSize[1]; ++col_win) {
        int col_img = col_min + col_win;
        // Row-major: array[row, col] -> row*width + col
        fWindowA[SUB2IND_2D(row_win, col_win, nWindowSize[1])] = 
            fImageA[SUB2IND_2D(row_img, col_img, nImageSize[1])];
    }
}
```

#### Peak Location Storage Fix
```c
// Calculate proper row-major index for output
int out_idx = n * nWindows[0] * nWindows[1] + jj * nWindows[1] + ii;

// Correct displacement assignment
float peak_row = fPeakLoc[SUB2IND_2D(0, n, nPeaks)];  // Y-displacement
float peak_col = fPeakLoc[SUB2IND_2D(1, n, nPeaks)];  // X-displacement

fPkLocX[out_idx] = peak_col - nWindowSize[1]/2.0f;  // X is column
fPkLocY[out_idx] = peak_row - nWindowSize[0]/2.0f;  // Y is row
```

### 4. FFT Cross-Correlation (`xcorr.c`)

#### fftshift Update for Row-Major
```c
// Updated fftshift for row-major data
for(int row = 0; row < N[0]; ++row) {
    int row_swap = (row + N[0]/2) % N[0];
    // Copy left half of swapped row to right half of output
    memcpy(&c[SUB2IND_2D(row, N[1]/2, N[1])], 
           &c_copy[SUB2IND_2D(row_swap, 0, N[1])], 
           N[1]/2 * sizeof(float));
    // Copy right half of swapped row to left half of output
    memcpy(&c[SUB2IND_2D(row, 0, N[1])], 
           &c_copy[SUB2IND_2D(row_swap, N[1]/2, N[1])], 
           N[1]/2 * sizeof(float));
}
```

## Verification Checklist

### Coordinate System
- [x] X corresponds to width (columns) throughout
- [x] Y corresponds to height (rows) throughout
- [x] win_ctrs_x spans width dimension
- [x] win_ctrs_y spans height dimension
- [x] Window centers match MATLAB calculation
- [x] No image transposition occurs

### Memory Layout
- [x] All Python arrays are C-contiguous
- [x] C library uses row-major indexing
- [x] SUB2IND_2D macro updated for row-major
- [x] Window extraction uses correct dimensions
- [x] Peak locations stored with correct indexing
- [x] Mask indexing matches window grid

### Dimensional Consistency
- [x] nImageSize[0] = H (height/rows)
- [x] nImageSize[1] = W (width/cols)
- [x] nWindowSize[0] = win_height (rows)
- [x] nWindowSize[1] = win_width (cols)
- [x] nWindows[0] = n_win_y (rows)
- [x] nWindows[1] = n_win_x (cols)

## Testing Recommendations

1. **Unit Tests**: Verify window center calculation matches MATLAB
2. **Known Displacement Test**: Use synthetic images with known displacement
3. **Boundary Test**: Check vectors near image edges
4. **Mask Test**: Verify masked regions are properly excluded
5. **Multi-Pass Test**: Ensure predictor-corrector works correctly
6. **Performance Test**: Measure speedup from eliminating Fortran conversions

## Performance Impact

### Expected Improvements
- **No transpose overhead**: Eliminated costly array reordering
- **Better cache locality**: Row-major access patterns are cache-friendly
- **Reduced memory copies**: No format conversions needed
- **SIMD-friendly**: Contiguous memory enables vectorization

### Potential Speedup
- 10-20% reduction in memory operations
- 5-10% overall PIV computation speedup
- Better scaling with larger window sizes

## Backward Compatibility

### Breaking Changes
- **C library interface**: Now expects C-contiguous arrays
- **Array dimensions**: Must follow (rows, cols) convention
- **Window centers**: Must be calculated with correct Nx/Ny

### Migration Path
If you have existing code:
1. Recompile C library with updated code
2. Update Python code to use `np.ascontiguousarray()`
3. Verify window center calculations
4. Test with known reference data

## References

- MATLAB PIV implementation (original coordinate system)
- NumPy memory layout: https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray
- Row-major vs Column-major: https://en.wikipedia.org/wiki/Row-_and_column-major_order

## Authors
- Fix implemented: November 2025
- Based on MATLAB reference implementation

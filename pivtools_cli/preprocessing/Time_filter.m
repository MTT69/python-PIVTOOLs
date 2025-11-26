function images = Time_filter(images, setup, imRange, directory_path, filter_length)
% Image Processing with Time Minimum Filtering
%
% This script processes a set of images by computing the "time minimum" of each pixel 
% across multiple frames (images) and then subtracting this minimum from the original images.
%
% The filter_length parameter determines the batch size:
% - If filter_length is 50, images are processed in batches of 50
% - Each batch has its own minimum computed and subtracted
% 
% Example: With 200 images and filter_length=50:
%   - Images 1-50: processed with their own minimum
%   - Images 51-100: processed with a different minimum
%   - Images 101-150: processed with another minimum
%   - Images 151-200: processed with another minimum
%
% Inputs:
%   - `images`: A structure array containing image data, with fields `frame1` and `frame2`.
%   - `imRange`: A range of indices for the images to process.
%   - `setup`: A structure containing image properties, specifically `imageSize`.
%   - `directory_path`: The directory to save the filtered images.
%   - `filter_length`: The number of images to process in each batch.
%
% Outputs:
%   - The filtered frames (time minimum subtracted) are saved to disk as `filtered-time-min-...` 
%     and `filtered-time-...` for both frame1 and frame2.
%

    % Determine the total number of images and number of batches
    num_images = length(imRange);
    num_batches = ceil(num_images / filter_length);
    
    Setup_parpool(setup, 'Images')
    
    % Store the first batch's time minimum
    first_batch_min1 = [];
    first_batch_min2 = [];
    
    for batch = 1:num_batches
        % Determine the image range for this batch
        batch_start = (batch-1) * filter_length + 1;
        batch_end = min(batch * filter_length, num_images);
        batch_range = imRange(batch_start:batch_end);
        
        % Initialize minimum values for both images to infinity (single precision)
        timeminimum1 = single(inf(setup.imProperties.imageSize));  % For the first image
        timeminimum2 = single(inf(setup.imProperties.imageSize));  % For the second image
        
        % Compute the time minimum for this batch
        for i = 1:length(batch_range)
            ImNo = batch_range(i);
            timeminimum1 = min(timeminimum1, images(ImNo).frame1);
            timeminimum2 = min(timeminimum2, images(ImNo).frame2);
        end
        
        % Store the time minimum of the first batch
        if batch == 1
            first_batch_min1 = timeminimum1;
            first_batch_min2 = timeminimum2;
        end
        
        % Subtract the time minimum from each image in this batch
        for i = 1:length(batch_range)
            ImNo = batch_range(i);
            images(ImNo).frame1 = images(ImNo).frame1 - timeminimum1;
            images(ImNo).frame2 = images(ImNo).frame2 - timeminimum2;
        end
    end

    % Save only the time minimum and first frame pair of the first batch
    filename_frame1 = fullfile(directory_path, 'filtered-time-min-frame1');
    filename_frame2 = fullfile(directory_path, 'filtered-time-min-frame2');
    
    PARIMSAVE((double(first_batch_min1)), filename_frame1);  % Save filtered frame1
    PARIMSAVE((double(first_batch_min2)), filename_frame2);  % Save filtered frame2
    
    first_frame_idx = imRange(1); % Save only the first image
    filename_frame1 = fullfile(directory_path, 'filtered-time-frame1');
    filename_frame2 = fullfile(directory_path, 'filtered-time-frame2');
    
    PARIMSAVE((double(images(first_frame_idx).frame1)), filename_frame1);  % Save filtered frame1
    PARIMSAVE((double(images(first_frame_idx).frame2)), filename_frame2);  % Save filtered frame2
end



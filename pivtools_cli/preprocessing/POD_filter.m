function images = POD_filter(images, setup, imRange, directory_path)
% Image Processing using Proper Orthogonal Decomposition (POD) Filtering
%
% This script processes image data in chunks, applies Singular Value Decomposition 
% (SVD) to compute the covariance matrices, and uses Proper Orthogonal Decomposition 
% (POD) to filter out certain modes based on automatic mode selection criteria. 
% The filtered images are saved for further analysis. TODO - this could likely be altered 
% for parallel processing

% Inputs:
%   - `images`: A structure containing the frames (`frame1`, `frame2`) to be processed.
%   - `imRange`: A range of indices specifying the images to process.
%   - `setup`: A structure containing image properties (such as `imageSize`).
%   - `directory_path`: The directory to save the filtered images.
%
% The process is divided into several key steps:
% 1. **Initial Setup**: 
%    - The script initializes key parameters for automatic thresholding and image 
%      processing. It preallocates memory for the images and prepares the data for 
%      processing.
% 
% 2. **Covariance Matrix Calculation**:
%    - For each pair of images (frame1 and frame2), the covariance matrices are computed 
%      by reshaping the images into vectors, performing matrix multiplication, and applying 
%      Singular Value Decomposition (SVD) to extract eigenvectors and eigenvalues.
%
% 3. **Mode Selection**:
%    - Using a custom function `find_auto_mode`, the script automatically selects modes 
%      based on the eigenvalues and eigenvectors derived from the covariance matrices. 
%      The modes are identified based on the thresholds defined in `hand.eps_auto_psi` and 
%      `hand.eps_auto_sigma`.
%
% 4. **POD Evaluation**:
%    - Once the modes are selected, the script computes the corresponding "basis" vectors 
%      (PHI) and "time coefficients" (TCoeff) for both image blocks (frame1 and frame2). This 
%      is done using the function `evaluatePHITCoeff`.
%
% 5. **Image Processing Loop**:
%    - The images are processed by subtracting the modes selected from each image. The images 
%      are then reshaped back to their original form and stored.
%
% 6. **Saving Results**:
%    - After processing, the filtered frames are saved to disk using the `PARIMSAVE` function, 
%      allowing the user to store both the intermediate and final results.
%

    hand = [];            
    % Epsilon for the automatic threshold
    hand.eps_auto_psi = 0.01;
    hand.eps_auto_sigma = 0.01;


    % Define the number of images to process
    chunkLength = length(imRange);
    
    % Preallocate matrices for the block
    M_bloc1 = zeros(chunkLength, setup.imProperties.imageSize(1) * setup.imProperties.imageSize(2), 'single');
    M_bloc2 = zeros(chunkLength, setup.imProperties.imageSize(1) * setup.imProperties.imageSize(2), 'single');

    % Load the image data into M_bloc1 and M_bloc2
    for imNo = imRange
        idx = imNo - imRange(1) + 1;  % Adjust index based on imRange
        M_bloc1(idx, :) = reshape(images(imNo).frame1, 1, []);
        M_bloc2(idx, :) = reshape(images(imNo).frame2, 1, []);
    end

    % Compute the covariance matrices
    C1 = M_bloc1 * M_bloc1';
    C2 = M_bloc2 * M_bloc2';
     
    % Singular Value Decomposition (SVD)
    [PSI1, LAMBDA1] = svd(C1);
    eigVal1 = diag(LAMBDA1); clear LAMBDA1 C1;
 
    [PSI2, LAMBDA2] = svd(C2);
    eigVal2 = diag(LAMBDA2); clear LAMBDA2 C2;

    % Automatic mode selection for block 1
    N_auto1 = find_auto_mode(PSI1, eigVal1, hand, chunkLength);
    
    % Automatic mode selection for block 2
    N_auto2 = find_auto_mode(PSI2, eigVal2, hand, chunkLength);

    % Evaluate modes
    [PHI_bloc1, TCoeff_bloc1] = evaluatePHITCoeff(M_bloc1, PSI1, N_auto1);  % function in misc
    [PHI_bloc2, TCoeff_bloc2] = evaluatePHITCoeff(M_bloc2, PSI2, N_auto2);


    clear PSI1 eigVal1 PSI2 eigVal2;  % Clearing unnecessary variables

    % Image processing loop using spmd
   

    % Process images in the local range
    for ImNo = 1:numel(imRange)
        tmp_im1 = M_bloc1(ImNo, :);
        tmp_im2 = M_bloc2(ImNo, :);
        
        % Subtract modes from each image
        for mod_i = 1:N_auto1
            tmp_im1 = tmp_im1 - (TCoeff_bloc1{mod_i}(ImNo, 1) * PHI_bloc1{mod_i}(:, 1)');
        end
        
        for mod_i = 1:N_auto2
            tmp_im2 = tmp_im2 - (TCoeff_bloc2{mod_i}(ImNo, 1) * PHI_bloc2{mod_i}(:, 1)');
        end
        
        % Reshape back to image format and store in images
        images(imRange(ImNo)).frame1 = reshape(tmp_im1, setup.imProperties.imageSize(1), setup.imProperties.imageSize(2));
        images(imRange(ImNo)).frame2 = reshape(tmp_im2, setup.imProperties.imageSize(1), setup.imProperties.imageSize(2));
        if ImNo ==1 
            filename_frame1 = fullfile(directory_path, ['filtered-' 'POD-sub' '-frame1']);
            filename_frame2 = fullfile(directory_path, ['filtered-' 'POD-sub' '-frame2']);
           
            PARIMSAVE((double(reshape(M_bloc1(ImNo,:)-tmp_im1, setup.imProperties.imageSize(1), setup.imProperties.imageSize(2)))), filename_frame1);  % Save filtered frame1
            PARIMSAVE((double(reshape(M_bloc2(ImNo,:)-tmp_im2, setup.imProperties.imageSize(1), setup.imProperties.imageSize(2)))), filename_frame2);  % Save filtered frame1
        end
        
        
    end
    filename_frame1 = fullfile(directory_path, ['filtered-' 'POD' '-frame1']);
    filename_frame2 = fullfile(directory_path, ['filtered-' 'POD' '-frame2']);
   
    PARIMSAVE((double(images(imRange(1)).frame1)), filename_frame1);  % Save filtered frame1
    PARIMSAVE((double(images(imRange(1)).frame2)), filename_frame2);  % Save filtered frame2

    

      
end

function N_auto = find_auto_mode(PSI, eigVal, hand, chunkLength)
    % Automatic mode selection
    N_auto = 0;
    for i = 1:(chunkLength - 1)
        Mean_PSI = abs(mean(PSI(:, i)));
        Sig_Diff = (abs(eigVal(i) - eigVal(i + 1))) / eigVal(round(chunkLength / 2));
        
        if Mean_PSI < hand.eps_auto_psi && Sig_Diff < hand.eps_auto_sigma * eigVal(1)
            N_auto = i;
            break;
        end
    end
end
function [PHI, TCoeff] = evaluatePHITCoeff(M, PSI, N_auto)
    PHI = cell(N_auto, 1);
    TCoeff = cell(N_auto, 1);

    for mod_i = 1:N_auto
        % Evaluate PHI
        PHI{mod_i} = transpose(M) * PSI(:, mod_i);
        PHI{mod_i} = PHI{mod_i} / sqrt(sum(PHI{mod_i}.^2));

        % Evaluate T
        TCoeff{mod_i} = M * PHI{mod_i};
    end
end

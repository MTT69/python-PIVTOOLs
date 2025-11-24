function b_mask = compute_b_mask(setup, im_mask)
%{
    compute_b_mask.m
    ===============
    This function computes the binary mask `b_mask` for each pass based on the input image mask (`im_mask`) and the setup parameters defined in `setup`. 
    The mask is computed using a sliding window approach, where each window is convolved with a box filter to compute local statistics. The resulting filtered mask is then interpolated at the window center positions and thresholded to generate the final binary mask for each pass.

    Key operations:
    1. Loop through each pass (defined in `setup.instantaneous.passes`).
    2. For each pass, calculate the window size, overlap, and the positions of the window centers.
    3. Convolve the input mask (`im_mask`) with a box filter to obtain local smoothing.
    4. Interpolate the filtered mask at the window center positions.
    5. Apply a threshold to the interpolated mask to generate a binary mask.
    6. Store the resulting binary mask for each pass.

    @inputs:
    - setup: A structure containing setup parameters for the mask computation, including:
        - `setup.instantaneous.passes`: Number of passes to process.
        - `setup.instantaneous.windowSize`: The size of the window for each pass (in pixels).
        - `setup.instantaneous.overlap`: The overlap percentage between consecutive windows.
        - `setup.instantaneous.im_x` and `setup.instantaneous.im_y`: The coordinates of the image grid.
        - `setup.instantaneous.maskThreshold`: The threshold for the binary mask.
    - im_mask: The initial mask image that will be processed.

    @outputs:
    - b_mask: A cell array of binary masks (`b_mask{pass}`) corresponding to each pass.

    The function uses convolution with a box filter and interpolation to calculate the mask for each window. The mask is thresholded to create a binary mask that can be used in subsequent image processing tasks.
%}

    % Initialize b_mask cell array to store masks for each pass
    b_mask = cell(1, setup.instantaneous.passes);

    % Loop over each pass
    for pass = 1:setup.instantaneous.passes
        % Window size and overlap for this pass
        wsize = setup.instantaneous.windowSize(pass, :);
        overlap = setup.instantaneous.overlap(pass);
        
        % Get dimensions of the image grid
        Nx = length(setup.instantaneous.im_x);
        Ny = length(setup.instantaneous.im_y);
        
        % Calculate window spacing in x and y directions
        win_spacing_x = round((1 - overlap / 100) * wsize(1));
        win_spacing_y = round((1 - overlap / 100) * wsize(2));
        
        % Calculate window center positions in x and y directions
        win_ctrs_x = 0.5 + wsize(1)/2 : win_spacing_x : Nx - wsize(1)/2 + 0.5;
        win_ctrs_y = 0.5 + wsize(2)/2 : win_spacing_y : Ny - wsize(2)/2 + 0.5;
        
        % Create grid of window center coordinates
        [win_x, win_y] = ndgrid(win_ctrs_x, win_ctrs_y);
        
        % Perform 2D convolution on im_mask using box filter
        f_mask = conv2(im_mask, ones(wsize(1), 1) / wsize(1), 'same');
        f_mask = conv2(f_mask.', ones(wsize(2), 1) / wsize(2), 'same').';

        % Interpolate the filtered mask at the window center positions
        b_mask{pass} = interpn(setup.instantaneous.im_x, ...
                               setup.instantaneous.im_y, ...
                               f_mask, win_x, win_y, 'nearest') > setup.instantaneous.maskThreshold;
    end
end

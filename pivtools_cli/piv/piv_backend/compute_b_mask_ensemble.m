function b_mask = compute_b_mask_ensemble(setup, im_mask)
 %     * Function: compute_b_mask_ensemble
 % * ---------------------------------------------------------------------------
 % * Description:
 % * This function generates binary masks (`b_mask`) for each pass of an ensemble
 % * Particle Image Velocimetry (PIV) analysis. The masks define which regions of 
 % * the image grid are included in the PIV calculations for each pass, taking 
 % * into account image size, window size, and overlap settings for the ensemble 
 % * analysis.
 % 
 % * The binary mask is computed by filtering the input mask (`im_mask`) with 
 % * a box filter defined by the window size for each pass, and then interpolating 
 % * the result over the grid of window center positions. The resulting mask is 
 % * used to exclude unwanted regions in the velocity field calculation during 
 % * PIV processing.
 % * 
 % * ---------------------------------------------------------------------------
 % * Inputs:
 % *   struct setup   : A structure containing setup information and parameters 
 % *                    for the ensemble PIV analysis, including:
 % *       - setup.ensemble.passes        : Number of passes in the ensemble PIV.
 % *       - setup.ensemble.windowSize    : Size of the interrogation window for 
 % *                                        each pass.
 % *       - setup.ensemble.overlap       : Percentage overlap between adjacent 
 % *                                        windows for each pass.
 % *       - setup.ensemble.im_x          : x-coordinates of the image grid.
 % *       - setup.ensemble.im_y          : y-coordinates of the image grid.
 % *       - setup.ensemble.sumWindow     : Sum window size for correlation 
 % *                                        processing.
 % *       - setup.ensemble.maskThreshold : Threshold to binarize the filtered 
 % *                                        mask for each pass.
 % * 
 % *   matrix im_mask : A 2D matrix representing the initial mask of the image. 
 % *                    It defines the regions of interest in the image where 
 % *                    the PIV analysis will be performed. Typically, values 
 % *                    of 1 represent regions of interest, and 0 represents 
 % *                    masked-out regions.
 % * 
 % * ---------------------------------------------------------------------------
 % * Outputs:
 % *   cell b_mask    : A cell array where each element contains a binary mask 
 % *                    matrix for a specific pass of the ensemble PIV analysis. 
 % *                    These binary masks are used to limit the regions of the 
 % *                    PIV velocity field calculations.
 % * 
 % * ---------------------------------------------------------------------------
 % * Procedures:
 % *   1. **Initialization**:
 % *      - The `b_mask` cell array is initialized to store the mask for each pass 
 % *        of the ensemble analysis. The total number of passes is determined by 
 % *        `setup.ensemble.passes`.
 % * 
 % *   2. **Loop Over Each Pass**:
 % *      - For each pass, the function retrieves the window size, overlap, and 
 % *        calculates the spacing and center positions of the interrogation 
 % *        windows in both the x and y directions.
 % * 
 % *   3. **Window Position Calculation**:
 % *      - The center positions of the windows are determined based on the image 
 % *        grid size and the specified overlap percentage. The window centers are 
 % *        recalculated if the `runtype` is set to 'single', adjusting for padding 
 % *        based on the sum window size.
 % * 
 % *   4. **Mask Filtering**:
 % *      - The input mask (`im_mask`) is filtered using 2D convolution with a box 
 % *        filter defined by the window size. This smoothing step helps to ensure 
 % *        that the mask appropriately covers the interrogation window regions.
 % * 
 % *   5. **Interpolation and Binarization**:
 % *      - The filtered mask is interpolated at the window center positions using 
 % *        nearest-neighbor interpolation. The resulting mask is binarized by 
 % *        applying a threshold defined in `setup.ensemble.maskThreshold`.
 % * 
 % *   6. **Store Binary Mask**:
 % *      - The binary mask for the current pass is stored in the `b_mask` cell array.
 % * 
 % * ---------------------------------------------------------------------------
    b_mask = cell(1, setup.ensemble.passes);
    SumWindow = setup.ensemble.sumWindow;

    % Loop over each pass
    for pass = 1:setup.ensemble.passes
        runtype = setup.ensemble.type{pass};
        % Window size and overlap for this pass
        wsize = setup.ensemble.windowSize(pass, :);
        overlap = setup.ensemble.overlap(pass);
        
        % Get dimensions of the image grid
        Nx = length(setup.ensemble.im_x);
        Ny = length(setup.ensemble.im_y);
        
        % Calculate window spacing in x and y directions
        win_spacing_x = round((1 - overlap / 100) * wsize(1));
        win_spacing_y = round((1 - overlap / 100) * wsize(2));
        
        % Calculate window center positions in x and y directions
        win_ctrs_x = 0.5 + wsize(1)/2 : win_spacing_x : Nx - wsize(1)/2 + 0.5;
        win_ctrs_y = 0.5 + wsize(2)/2 : win_spacing_y : Ny - wsize(2)/2 + 0.5;
        
        % Create grid of window center coordinates - window centres/
        % padding needs to be computed differently for smaller windows 
        [win_x, win_y] = ndgrid(win_ctrs_x, win_ctrs_y);
        if strcmp('single', runtype)
            padtop=ceil((SumWindow(1)-wsize(1))/2);
            padbot=floor((SumWindow(1)-wsize(1))/2);
            padleft=ceil((SumWindow(2)-wsize(2))/2);
            padright=floor((SumWindow(2)-wsize(2))/2);
            Nx				= Nx+padleft+padright;
            Ny				= Ny+padtop+padbot;
            win_spacing_x	= round((1 - overlap/100) * wsize(1));
    
            win_spacing_y	= round((1 - overlap/100) * wsize(2));
            
            if wsize(1) ==1
                win_ctrs_x		= 1 + (SumWindow(1)/2) : win_spacing_x : Nx - (SumWindow(1)/2) + 1;
            else
                win_ctrs_x		= 0.5 + (SumWindow(1)/2) : win_spacing_x : Nx - (SumWindow(1)/2) + 0.5;
            end
    
    
    
            if wsize(2) ==1
                win_ctrs_y		= SumWindow(2)/2 : win_spacing_y : Ny - (SumWindow(2)/2);
            else
                win_ctrs_y		= 0.5 + SumWindow(2)/2 : win_spacing_y : Ny - (SumWindow(2)/2) + 0.5;
            end
    
            [win_x,win_y]	= ndgrid(win_ctrs_x, win_ctrs_y);
        end
        
        % Perform 2D convolution on im_mask using box filter
        f_mask = conv2(im_mask, ones(wsize(1), 1) / wsize(1), 'same');
        f_mask = conv2(f_mask.', ones(wsize(2), 1) / wsize(2), 'same').';

        % Interpolate the filtered mask at the window center positions
        b_mask{pass} = interpn(setup.ensemble.im_x, ...
                               setup.ensemble.im_y, ...
                               f_mask, win_x, win_y, 'nearest') > setup.ensemble.maskThreshold;
    end
end

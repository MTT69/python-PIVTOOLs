
function piv_result_cor=PIV_2D_wdef_ensemble(setup,predictorfield,pass,b_mask,cameraNo,im_mask,filters,images,masks)
% PIV_2D_wdef_ensemble: Performs 2D Particle Image Velocimetry (PIV) analysis 
% using window deformation, ensemble processing, and multiple correlation passes.
% This function tracks particle displacements between two images (A and B) over 
% multiple passes to enhance accuracy using deformation of interrogation windows 
% and temporal ensemble averaging.

% Inputs:
%   setup          - Struct containing various parameters for PIV analysis, 
%                    including window sizes, overlaps, image properties, and grid info. 
%                    Relevant fields in setup:
%                    * setup.ensemble.windowSize: Sizes of the interrogation windows (for each pass)
%                    * setup.ensemble.windowType: Type of windows (e.g., Gaussian, single-pixel)
%                    * setup.ensemble.overlap: Percentage overlap between windows
%                    * setup.instantaneous.im_x, im_y: Image grid size for x and y dimensions
%                    * setup.instantaneous.dt: Time interval between two image frames
%                    * setup.imProperties.imageSize: The size of the input image
%   predictorfield - The displacement field from the previous PIV pass, used as 
%                    a predictor for deforming the interrogation windows in the current pass.
%   pass           - Integer indicating the current pass number (in multi-pass processing).
%   b_mask         - Binary mask (or cell array of masks) specifying regions in the 
%                    images to be excluded from the PIV analysis (e.g., occlusions or invalid data).
%   cameraNo       - Integer representing the camera index being processed (in case of multi-camera setups).
%   im_mask        - Binary mask for excluding certain regions of the images from processing.
%                    Regions marked with 1 in the mask will be ignored.
%   filters        - Struct containing image processing filters (e.g., smoothing filters) 
%                    applied to the input images before correlation.
%
% Outputs:
%   piv_result_cor - Struct containing the results of the PIV analysis. Relevant fields:
%                    * piv_result_cor.n_windows: Number of interrogation windows in x and y directions
%                    * piv_result_cor.win_x, win_y: Center coordinates of interrogation windows
%                    * piv_result_cor.correlation_plane: Cross-correlation results for each window
%                    * piv_result_cor.Predictor_Field: The refined displacement field used for the next pass
%                    * piv_result_cor.b_mask: The binary mask used for each pass to exclude invalid areas
%                    * piv_result_cor.pointspreadA, pointspreadB: Point spread functions used for window weighting

% Procedure:
% 1. **Initialization and Setup**: 
%    - The function begins by initializing several key parameters, such as the size 
%      of the interrogation windows, window overlap, and image grid properties.
%    - The function checks if the overlap percentage is feasible given the window size, 
%      adjusting it if necessary to avoid excessive overlap that could reduce performance.

% 2. **Grid Generation**: 
%    - The center coordinates of the interrogation windows are computed based on the 
%      image size and window overlap. These windows define regions of the image to be 
%      analyzed for particle displacement.
%    - A grid of window centers (in both x and y directions) is created, which will 
%      be used in the subsequent cross-correlation steps.

% 3. **Window Deformation**: 
%    - The interrogation windows in the images are deformed using the predictor field 
%      (from the previous pass) to better align with the expected particle displacement.
%    - The deformation helps improve the accuracy of the cross-correlation by ensuring 
%      that the interrogation windows are aligned with the actual particle motion.

% 4. **Cross-correlation**: 
%    - Cross-correlation is performed between the deformed interrogation windows of the 
%      two images (A and B). This step calculates the displacement between particles in 
%      each window by finding the peak in the correlation plane, which represents the 
%      most likely displacement vector.
%    - The correlation plane for each window is stored in the output struct.

% 5. **Ensemble Averaging**: 
%    - The mean image deformation across the batch is subtracted from the images, helping 
%      to minimize systematic errors and noise.

% 6. **Multi-Pass Processing**: 
%    - The function supports multiple passes of the PIV analysis. In the first pass, 
%      the displacement field is initialized to zero, and in subsequent passes, the 
%      displacement field from the previous pass is used as a predictor for window deformation.
%    - Each pass refines the displacement field, improving the accuracy of the results.

% 7. **Masking and Invalid Region Handling**: 
%    - A binary mask is applied to exclude certain regions from the analysis (e.g., 
%      occlusions or areas without valid data).
%    - The mask is applied in each pass, ensuring that only valid regions of the image 
%      are included in the correlation and displacement calculations.



	% 
	Ensemble = true;
    runtype = setup.ensemble.type{pass};
    wtypeA  = setup.ensemble.windowType{pass};
    SumWindow = setup.ensemble.sumWindow;
    b_mask = b_mask{pass};
    if strcmp('single',runtype)
        wtypeA='singlepix';
    end
    % ensures feasability of setup
    if (setup.ensemble.windowSize(pass,1) * setup.ensemble.overlap(pass) <1 || setup.ensemble.windowSize(pass,2) * setup.ensemble.overlap(pass) <1) && setup.ensemble.overlap(pass)~=0
        setup.ensemble.overlap(k) =0;
        warning(' overlap is too large for desired windowsize, running with zero overlap')
    end

    
	
	% image interpolation for window deformation
	wdef_ksize		= 4;
	wdef_kernel		= 'lanczos';
	
	%% pre-allocate results structure
	empty			= cell(1,1);
	piv_result_cor		= struct(	'n_windows', empty, ...
								'win_x', empty, 'win_y', empty, ...
								'dt', empty,'correlation_plane',empty,...
                                'Predictor_Field',empty,...
                                'wsize_old',empty,'win_spacing_old',empty,'win_ctrs_x_old', ...
                                empty,'win_ctrs_y_old',empty,'b_mask',empty,'pointspreadA',empty,'pointspreadB',empty,'n_pre',empty,'n_post',empty);
	
	%% ITERATE OVER PASSES
    %% report
    
    
    %% calculate grid for window centres in images A and B
    %
    %     /---/     +---+      \---\
    %    /   /   -> |   |   ->  \   \
    %   /---/       +---+        \---\
    %     A          (0)           B
    %
    % the deformation between A and B is symmetric
    % one supposes a pair of identical deformations that maps
    % A forwards and B backwards onto the hypothetical (0), which is cuboidal
    % 
    % a window is defined by:
    % - the regular mesh of points in the window
    % - the deformation matrix applied to the window

    wsize			= setup.ensemble.windowSize(pass, :);
    wtype           = setup.ensemble.windowType{pass};     
    im_size			= setup.imProperties.imageSize;
    im_x			= setup.instantaneous.im_x;
    im_y			= setup.instantaneous.im_y;
    piv_dt			= setup.instantaneous.dt;
    Nx				= length(im_x);
    Ny				= length(im_y);       
    im_i			= 1 : im_size(1);
    im_j			= 1 : im_size(2);
    [im_imat,im_jmat]=ndgrid(im_i, im_j);
    im_mesh			= single(cat(3, im_imat, im_jmat));
    

    overlap			= setup.ensemble.overlap(pass);
    padtop=0;
    padbot=0;
    padleft=0;
    padright=0;

    win_spacing_x	= round((1 - overlap/100) * wsize(1));
    win_spacing_y	= round((1 - overlap/100) * wsize(2));
    win_ctrs_x		= 0.5 + wsize(1)/2 : win_spacing_x : Nx - wsize(1)/2 + 0.5;
    win_ctrs_y		= 0.5 + wsize(2)/2 : win_spacing_y : Ny - wsize(2)/2 + 0.5;
    n_windows		= [length(win_ctrs_x) length(win_ctrs_y)];
    [win_x,win_y]	= ndgrid(win_ctrs_x, win_ctrs_y);
    % padding is required so that the window centre occurs at the position of interest for inner window 
    if strcmp('single', runtype)
        padtop=ceil((SumWindow(1)-wsize(1))/2);
        padbot=floor((SumWindow(1)-wsize(1))/2);
        padleft=ceil((SumWindow(2)-wsize(2))/2);
        padright=floor((SumWindow(2)-wsize(2))/2);
        Nx				= length(im_x)+padleft+padright;
        Ny				= length(im_y)+padtop+padbot;
        win_spacing_x	= round((1 - overlap/100) * wsize(1));

        win_spacing_y	= round((1 - overlap/100) * wsize(2));
        
        if wsize(1) ==1
            win_ctrs_x		= 1 + (SumWindow(1)/2) : win_spacing_x : Nx - (SumWindow(1)/2) + 1;
            padright = padright+1;
            
        else
            win_ctrs_x		= 0.5 + (SumWindow(1)/2) : win_spacing_x : Nx - (SumWindow(1)/2) + 0.5;
        end



        if wsize(2) ==1
            win_ctrs_y		= 1 + SumWindow(2)/2 : win_spacing_y : Ny - (SumWindow(2)/2) + 1;
            padbot = padbot+1;
        else
            win_ctrs_y		= 0.5 + SumWindow(2)/2 : win_spacing_y : Ny - (SumWindow(2)/2) + 0.5;
        end

        [win_x,win_y]	= ndgrid(win_ctrs_x, win_ctrs_y);
        n_windows		= [length(win_ctrs_x) length(win_ctrs_y)];
    end
    
    % calculate window weighting function  
    
    window_weightA	   = window_weight_fun(wsize, wtypeA,SumWindow);

    if strcmp('single', runtype)        
        window_weightB	   = window_weight_fun(SumWindow, 'bsingle',SumWindow);

    else
        window_weightB	   = window_weight_fun(wsize, wtype,SumWindow);

    end
    



    %% build predictor displacement field and apply window deformation
    % we use a predictor-corrector heuristic to iteratively find
    % the displacement field. this is:
    % delta_ab, the displacement between A and B
    % DGT,      the displacement gradient tensor, analogous to the VGT
    % 
    % the predictor displacement field (delta_ab_pred, delta_bc_pred)
    % and the predictor DGT (dgt_pred) are based on:
    % pass 1)	a zero displacement field
    % 2:end)		an interpolation of the previous delta_ab
    %				the DGT comes from interpolating the estimate of the DGT from
    %				the previous pass

    % default zero displacement field
    % no a-priori guess of vector field, so use zero deformation and
    % zero displacement
    

    %% smooth the displacement field from the previous pass
    % cutoff wavelength for previous pass corresponds to about the
    % window size: we need to size this filter relative to the cutoff frequency  
    if pass == 1 
        ksize_filt = [1,1];
    else
        ksize_filt				= round(setup.ensemble.wsize_old ./ setup.ensemble.win_spacing_old) ;
        if mod(ksize_filt, 2) == 0
            ksize_filt = ksize_filt + 1; % Add 1 to make it odd
        end
    end
    sd						= sqrt(prod(ksize_filt))/3*0.65;
    % median filter to avoid errors due to spurious vectors in
    % result
    %delta_ab_old(:,:,1)		= medfilt2(delta_ab_old(:,:,1), ksize_filt);
    %delta_ab_old(:,:,2)		= medfilt2(delta_ab_old(:,:,2), ksize_filt);
    % smoothing filter
    % G_smooth				= fspecial('gaussian', ksize_filt, sd);
    % delta_ab_old			= convn(predictorfield, G_smooth, 'same');
    
    delta_ab_old(:,:,1) = imgaussfilt(predictorfield(:,:,1), sd, 'FilterSize', ksize_filt, 'Padding', 'replicate'); %handles edge effects better
    delta_ab_old(:,:,2)     =imgaussfilt(predictorfield(:,:,2),sd,'FilterSize',ksize_filt,'Padding', 'replicate'); %removes edge effects
    
    % Davis behaviour: filter with a fixed kernel size
    %G_smooth				= fspecial('average', ksize_filt);
                
    %% define dense velocity predictor field deformation field
    delta_ab_dense          = zeros([im_size 2]);
    if strcmp('spline', setup.ensemble.interpolator)
        delta_ab_dense(:,:,1)	= interpn(setup.ensemble.win_ctrs_x_old, setup.ensemble.win_ctrs_y_old, delta_ab_old(:,:,1), im_imat, im_jmat, 'spline', 0); 
        delta_ab_dense(:,:,2)	= interpn(setup.ensemble.win_ctrs_x_old, setup.ensemble.win_ctrs_y_old, delta_ab_old(:,:,2), im_imat, im_jmat, 'spline', 0);
    else
        delta_ab_dense(:,:,1)	= interpn(setup.ensemble.win_ctrs_x_old, setup.ensemble.win_ctrs_y_old, delta_ab_old(:,:,1), im_imat, im_jmat, 'linear', 0);  %%linear
        delta_ab_dense(:,:,2)	= interpn(setup.ensemble.win_ctrs_x_old, setup.ensemble.win_ctrs_y_old, delta_ab_old(:,:,2), im_imat, im_jmat, 'linear', 0);
    end
    delta_0b                = delta_ab_dense/2;
    delta_0a				=-delta_ab_dense/2;
    
    im_mesh_A				= im_mesh + delta_0a;
    im_mesh_B				= im_mesh + delta_0b;
    
    %% filter predictor field
    % extract weighted average of predictor field used for window
    % deformation
    %
    % see Astarita et al. (2007) for more details
    
    delta_ab_pred           = zeros([n_windows 2]);
    %G_smooth_predictor		= ones(ksize_filt)/prod(ksize_filt);
    G_smooth_predictor		= window_weight_fun(ksize_filt, 'gaussian',SumWindow); %
    G_smooth_predictor		= G_smooth_predictor / sum(sum(G_smooth_predictor));
    padding_amount          = floor(ksize_filt(1) / 2);
    padded_image            = padarray(delta_ab_old, [padding_amount, padding_amount], 'symmetric', 'both');
    delta_ab_filt           = convn(padded_image,G_smooth_predictor,'valid');
    % delta_ab_filt           = convn(delta_ab_old, G_smooth_predictor, 'same');
    if strcmp('spline', setup.ensemble.interpolator)

        delta_ab_pred(:,:,1)    = interpn(setup.ensemble.win_ctrs_x_old, setup.ensemble.win_ctrs_y_old, delta_ab_filt(:,:,1), win_x, win_y, 'spline', nan); %
        delta_ab_pred(:,:,2)    = interpn(setup.ensemble.win_ctrs_x_old, setup.ensemble.win_ctrs_y_old, delta_ab_filt(:,:,2), win_x, win_y, 'spline', nan);  %
    else
        delta_ab_pred(:,:,1)    = interpn(setup.ensemble.win_ctrs_x_old, setup.ensemble.win_ctrs_y_old, delta_ab_filt(:,:,1), win_x, win_y, 'linear', 0); %linear
        delta_ab_pred(:,:,2)    = interpn(setup.ensemble.win_ctrs_x_old, setup.ensemble.win_ctrs_y_old, delta_ab_filt(:,:,2), win_x, win_y, 'linear', 0);  %linear
    end

    
    %% initiaise key variables
    if strcmp('single', runtype)
        pointspread_plane_meanA = zeros(SumWindow(1)*SumWindow(2),n_windows(1),n_windows(2));

        pointspread_plane_meanB = zeros(SumWindow(1)*SumWindow(2),n_windows(1),n_windows(2));
    
        correlation_plane_mean = zeros(SumWindow(1)*SumWindow(2),n_windows(1),n_windows(2));

    else

        pointspread_plane_meanA = zeros(wsize(1)*wsize(2),n_windows(1),n_windows(2));

        pointspread_plane_meanB = zeros(wsize(1)*wsize(2),n_windows(1),n_windows(2));
    
        correlation_plane_mean = zeros(wsize(1)*wsize(2),n_windows(1),n_windows(2));
        
    end

    peak_finder = 6; %% todo these dont do anything but are needed for shared function definitons
    n_peaks =3 ;
    
    % TODO if speed becomes an issue - it would be interesting to batch the mean for an 'online' approach taking batch no of images as the mean 
    % this means you only have to load the data in once

    mean_A_warp = zeros(im_size);
    mean_B_warp = zeros(im_size);
    if strcmp('single', runtype)
        mean_A_warp = padarray(mean_A_warp, [padtop, padleft], 0, 'pre'); % Pad the top and left sides with zeros
        mean_A_warp = padarray(mean_A_warp, [padbot, padright], 0, 'post'); % Pad the bottom and right sides with zeros

        mean_B_warp = padarray(mean_B_warp, [padtop, padleft], 0, 'pre'); % Pad the top and left sides with zeros
        mean_B_warp = padarray(mean_B_warp, [padbot, padright], 0, 'post'); % Pad the bottom and right sides with zeros
    end
    %% calculate mean image after deformation
    batchCount = ceil(setup.imProperties.imageCount / setup.imProperties.batchSize);
    fprintf('Subtracting full temporal mean accross all batches at time %s\n', datetime('now'));
    % Initialize total accumulators
    count =0;
    nextMarker = 10;                % Next 10% marker to display
    for batchNo = 1:batchCount
        imRange = calculateImageRange(batchNo, batchCount, setup);
        if batchCount ~= 1
            images = Load_images(setup, cameraNo, imRange, batchNo,masks);
            images = Filter_images(images, setup, filters, imRange, cameraNo, batchNo);
        end
        fprintf('Applying window deformation at time %s\n', datetime('now'));
        numImages = numel(imRange);
        passSize = setup.imProperties.parforbatch;  % Size of each pass
        passes = ceil(numImages/passSize);
        Setup_parpool(setup, 'Processes')
        for batch = 1:passes
            % Determine the start and end indices for the current pass
            startIdx = (batch - 1) * passSize + 1;
            endIdx = min(batch * passSize, numImages);
            
            % Check to ensure we are within bounds
            if startIdx <= numImages
                % Get the relevant range for this pass
                currentRange = imRange(startIdx:endIdx);
                
                % Create a local copy of the data for the current pass
                local_data = images(currentRange);
                
                parfor imNoIndex = 1:numel(currentRange)
                   
                    A = single(local_data(imNoIndex).frame1);
                    B = single(local_data(imNoIndex).frame2);
        
                    %% Mask out invalid regions
                    A(im_mask) = 0;
                    B(im_mask) = 0;
        
                    %% Apply window deformation
                    A_prime = interp2custom(A, im_mesh_A(:,:,1)-1, im_mesh_A(:,:,2)-1, wdef_kernel, wdef_ksize);
                    B_prime = interp2custom(B, im_mesh_B(:,:,1)-1, im_mesh_B(:,:,2)-1, wdef_kernel, wdef_ksize);
        
                    if strcmp('single', runtype)
                        % Pad the top and left sides with zeros
                        A_prime = padarray(A_prime, [padtop, padleft], 0, 'pre');
                        A_prime = padarray(A_prime, [padbot, padright], 0, 'post');
                        B_prime = padarray(B_prime, [padtop, padleft], 0, 'pre');
                        B_prime = padarray(B_prime, [padbot, padright], 0, 'post');
                    end
        
                    % Accumulate results for this worker
                    mean_A_warp = mean_A_warp + A_prime;
                    mean_B_warp = mean_B_warp + B_prime;
                end
            
            % Update the main data structure with the loaded images
                images(currentRange) = local_data;
            end
            count = count + numel(currentRange);
            progress = (count / setup.imProperties.imageCount) * 100;
            if progress >= nextMarker
                currentTime = datetime('now', 'Format', 'HH:mm:ss');  % Get current time in HH:mm:ss format
                fprintf('Progress: %.0f%% completed. Current time: %s\n', nextMarker, char(currentTime));
                nextMarker = ceil(progress / 10) * 10;  % Update the next marker
            end


        end
    end
    % Normalize the result
    mean_A_warp = mean_A_warp / setup.imProperties.imageCount;
    mean_B_warp = mean_B_warp / setup.imProperties.imageCount;

%% perform correlation with mean field subtracted for signal strength increase
    nextMarker = 10;  
    for batchNo = 1:batchCount
        if batchCount ~= 1
            imRange = calculateImageRange(batchNo, batchCount, setup);  
            images = Load_images(setup, cameraNo, imRange, batchNo,masks);
            images = Filter_images(images, setup, filters, imRange, cameraNo, batchNo);
        else
            fprintf('Performing Cross correlation at time %s\n', datetime('now'));
        end
        fprintf('Performing Cross correlation for batch %d at time %s\n',batchNo, datetime('now'));
        for imNo = imRange
            A = single(images(imNo).frame1);
            B = single(images(imNo).frame2);
            A(im_mask)			= 0;
            B(im_mask)			= 0;
            A_prime					= interp2custom(A, im_mesh_A(:,:,1)-1, im_mesh_A(:,:,2)-1, wdef_kernel, wdef_ksize);
            B_prime					= interp2custom(B, im_mesh_B(:,:,1)-1, im_mesh_B(:,:,2)-1, wdef_kernel, wdef_ksize);
            if strcmp('single', runtype)

                A_prime = padarray(A_prime, [padtop, padleft], 0, 'pre'); % Pad the top and left sides with zeros
                A_prime = padarray(A_prime, [padbot, padright], 0, 'post'); % Pad the bottom and right sides with zeros

                B_prime = padarray(B_prime, [padtop, padleft], 0, 'pre'); % Pad the top and left sides with zeros
                B_prime = padarray(B_prime, [padbot, padright], 0, 'post'); % Pad the bottom and right sides with zeros
            end
            A_prime = A_prime - mean_A_warp;
            B_prime = B_prime - mean_B_warp;
            if strcmp('single', runtype)
                [correl_plane] =...
                    PIV_2d_cross_correlate( A_prime, B_prime, single(win_ctrs_x-1), single(win_ctrs_y-1), single(SumWindow), single(window_weightA), n_peaks, peak_finder,single(window_weightB),Ensemble,single(b_mask'));
                
                [pointspreadA] =...
                        PIV_2d_cross_correlate( A_prime, A_prime, single(win_ctrs_x-1), single(win_ctrs_y-1), single(SumWindow), single(window_weightA), n_peaks, peak_finder,single(window_weightB),Ensemble,single(b_mask'));
                
                [pointspreadB] =...
                        PIV_2d_cross_correlate( B_prime, B_prime, single(win_ctrs_x-1), single(win_ctrs_y-1), single(SumWindow), single(window_weightA), n_peaks, peak_finder,single(window_weightB),Ensemble,single(b_mask'));
    
    
            else

    
                [correl_plane] =...
                    PIV_2d_cross_correlate( A_prime, B_prime, single(win_ctrs_x-1), single(win_ctrs_y-1), single(wsize), single(window_weightA), n_peaks, peak_finder,single(window_weightB),Ensemble,single(b_mask'));
                
                [pointspreadA] =...
                        PIV_2d_cross_correlate( A_prime, A_prime, single(win_ctrs_x-1), single(win_ctrs_y-1), single(wsize), single(window_weightA), n_peaks, peak_finder,single(window_weightB),Ensemble,single(b_mask'));
                
                [pointspreadB] =...
                        PIV_2d_cross_correlate( B_prime, B_prime, single(win_ctrs_x-1), single(win_ctrs_y-1), single(wsize), single(window_weightA), n_peaks, peak_finder,single(window_weightB),Ensemble,single(b_mask'));
                
            end
            correlation_plane_mean=correlation_plane_mean+correl_plane;
            pointspread_plane_meanA=pointspread_plane_meanA+pointspreadA;
            pointspread_plane_meanB=pointspread_plane_meanB+pointspreadB;

            progress = (imNo / setup.imProperties.imageCount) * 100;
            if progress >= nextMarker
                currentTime = datetime('now', 'Format', 'HH:mm:ss');  % Get current time in HH:mm:ss format
                fprintf('Progress: %.0f%% completed. Current time: %s\n', nextMarker, char(currentTime));
                nextMarker = ceil(progress / 10) * 10; % Update the next marker
            end

        end
    end
    correlation_plane_mean=correlation_plane_mean/setup.imProperties.imageCount;
    pointspread_plane_meanA=pointspread_plane_meanA/setup.imProperties.imageCount;
    pointspread_plane_meanB=pointspread_plane_meanB/setup.imProperties.imageCount;

    
    %% prepare results structure for output
    
                            
    %% save displacement field from this pass
    % into "old" displacement field structure
    % at this juncture, pad the displacement field, as it will require
    % interpolation over the whole domain later
    
    % extend PIV grid
    win_ctrs_x_pre		= 1 : win_spacing_x : win_ctrs_x(1) - win_spacing_x/2;
    if isempty(win_ctrs_x_pre) || win_ctrs_x_pre(1) ~=1
        win_ctrs_x_pre = [1, win_ctrs_x_pre];
    end
    win_ctrs_y_pre		= 1 : win_spacing_y : win_ctrs_y(1) - win_spacing_y/2;
    if isempty(win_ctrs_y_pre) || win_ctrs_y_pre(1) ~=1
        win_ctrs_y_pre = [1, win_ctrs_y_pre];
    end
    win_ctrs_x_post		= im_size(1) : -win_spacing_x : win_ctrs_x(end) + win_spacing_x/2;
    if isempty(win_ctrs_x_post) || win_ctrs_x_post(1) < setup.imProperties.imageSize(1) + ceil(SumWindow(1)/2) % notation swap?! TODO
        win_ctrs_x_post = [setup.imProperties.imageSize(1) + ceil(SumWindow(1)/2) win_ctrs_x_post]; % this is jsut used to ensure the predictor field has scope to be interpolated onto
    end
    win_ctrs_y_post		= im_size(2) : -win_spacing_y : win_ctrs_y(end) + win_spacing_y/2;
    if isempty(win_ctrs_y_post) || win_ctrs_y_post(1) < setup.imProperties.imageSize(2) + ceil(SumWindow(2)/2)
        win_ctrs_y_post = [setup.imProperties.imageSize(2) + ceil(SumWindow(2)/2) win_ctrs_y_post];
    end
    win_ctrs_x_old		= [win_ctrs_x_pre win_ctrs_x win_ctrs_x_post(end:-1:1)];
    win_ctrs_y_old		= [win_ctrs_y_pre win_ctrs_y win_ctrs_y_post(end:-1:1)];
    % size of padding around edges
    n_pre				= [length(win_ctrs_x_pre ) length(win_ctrs_y_pre )];
    n_post				= [length(win_ctrs_x_post) length(win_ctrs_y_post)];
    % add padding to edges
    
    % make note of spacing
    win_spacing_old		= [win_spacing_x win_spacing_y];
    % save results

    
    piv_result_cor(pass)	= struct('n_windows', n_windows, ...
                                'win_x', (win_ctrs_y), ... %this might not always be true... im_x dependent
                                'win_y', win_ctrs_x, ... % interp1(1:Nx, im_x, win_ctrs_x), ...
                                'dt', piv_dt, 'correlation_plane',correlation_plane_mean,'Predictor_Field',delta_ab_pred,...
                                'wsize_old',wsize,'win_spacing_old',win_spacing_old,'win_ctrs_x_old',...
                                win_ctrs_x_old,'win_ctrs_y_old',win_ctrs_y_old,'b_mask',b_mask,'pointspreadA',pointspread_plane_meanA,'pointspreadB',pointspread_plane_meanB,...
                                'n_pre',n_pre,'n_post',n_post);


end
		


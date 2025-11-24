function piv_result = PIV_2D_wdef(A, B, setup, vector_mask, mask)
    % 
    % Perform 2D Particle Image Velocimetry (PIV) cross-correlation
    % between images A and B to determine the displacement field.
    % The function applies window deformation to account for movement between the frames.
    % The PIV process is performed in multiple passes to refine the displacement estimates.
    %
    % Images A and B should be pre-dewarped prior to applying this function.
    % This function applies both predictor-corrector techniques and a median filter for outlier rejection.
    %
    % Inputs:
    %   A           - Image A (2D matrix, single precision).
    %   B           - Image B (2D matrix, single precision).
    %   setup       - Struct containing PIV setup parameters (e.g. window size, overlap, etc.).
    %   vector_mask - Cell array of masks indicating invalid regions per pass.
    %   mask        - Matrix specifying regions of the image that should be excluded.
    %
    % Outputs:
    %   piv_result  - Struct containing the results of the PIV calculation.
    %                 The structure contains:
    %                 - n_windows    : [n_x, n_y] vector specifying the number of windows.
    %                 - win_x, win_y  : Coordinates of PIV window centers.
    %                 - ux            : x-component of displacement field.
    %                 - uy            : y-component of displacement field.
    %                 - nan_mask      : Boolean mask identifying vectors with large errors.
    %                 - peak_mag      : Magnitude of the chosen peak from cross-correlation.
    %                 - Q             : Q factor, the ratio of the peak to the next peak.
    %                 - peak_choice   : Index of the peak chosen for each window.
    %                 - dt            : Time difference between the images A and B.
    %                 - Predictor_Field: Predictor displacement field (used for refining results).
    %                 - wsize_old     : Old window size used in previous passes.
    %                 - win_spacing_old: Spacing of windows in the previous passes.
    %                 - win_ctrs_x_old: x-coordinates of window centers in the previous passes.
    %                 - win_ctrs_y_old: y-coordinates of window centers in the previous passes.
    %                 - b_mask        : Mask for the bad regions in each pass.
    %
    % Procedure:
    % 1. Pre-process images by applying masks and preparing for PIV calculations.
    % 2. Iteratively apply PIV cross-correlation over the image using different window sizes, 
    %    overlaps, and interpolation methods.
    % 3. Perform window deformation based on the predicted displacement field (predictor-corrector method).
    % 4. Apply a median filter to reject outliers based on the estimated displacement field.
    % 5. Calculate the displacement field (ux, uy) and other related metrics (peak magnitude, Q-factor).
    % 6. Store the results in the output structure piv_result.
    %
    % Notes:
    % - This function uses a predictor-corrector method to iteratively refine displacement estimates.
    % - The algorithm supports different peak-finding methods and interpolation schemes (linear or spline).
    % - A filtering step is applied to reduce outlier vectors, especially after each pass.
    % - The final displacement field is produced after multiple passes, with better accuracy and fewer spurious vectors.
    %
	
	%% convert images to single precision
	A				= single(A);
	B				= single(B);
    Ensemble        = false;
	
	%% mask out invalid regions
	A(mask)			= 0;
	B(mask)			= 0;
	
	%% extract main parameters
	n_passes		= setup.instantaneous.passes;
	n_peaks			= setup.instantaneous.peaks;
	im_x			= setup.instantaneous.im_x;
	im_y			= setup.instantaneous.im_y;
	im_dx			= im_x(2) - im_x(1);
	im_dy			= im_y(2) - im_y(1);
	piv_dt			= setup.instantaneous.dt;
	Nx				= length(im_x);
	Ny				= length(im_y);
	im_size			= size(A);	
	im_i			= 1 : im_size(1);
	im_j			= 1 : im_size(2);
	[im_imat,im_jmat]=ndgrid(im_i, im_j);
	im_mesh			= single(cat(3, im_imat, im_jmat));
	
	% image interpolation for window deformation
	wdef_ksize		= 4;
	wdef_kernel		= 'lanczos';
	
	%% pre-allocate results structure
	empty			= cell(n_passes,1);
	piv_result		= struct(	'n_windows', empty, ...
								'win_x', empty, 'win_y', empty, ...
								'ux', empty, 'uy', empty, ...
								'nan_mask', empty, 'Q', empty, ...
								'peak_mag', empty, 'peak_choice', empty, ...
								'dt', empty,'Predictor_Field',empty,...
                                'wsize_old',empty,'win_spacing_old',empty,'win_ctrs_x_old', ...
                                empty,'win_ctrs_y_old',empty,'b_mask',empty);
	
	%% ITERATE OVER PASSES
	for pass = 1 : n_passes
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

		wsize			= setup.instantaneous.windowSize(pass, :);
        wtype           = setup.instantaneous.windowType{pass};
		overlap			= setup.instantaneous.overlap(pass);
		
		%win_axis_x		= -wsize(1)/2+0.5 : +wsize(1)/2-0.5;
		%win_axis_y		= -wsize(2)/2+0.5 : +wsize(2)/2-0.5;
		win_spacing_x	= round((1 - overlap/100) * wsize(1));
		win_spacing_y	= round((1 - overlap/100) * wsize(2));
		win_ctrs_x		= 0.5 + wsize(1)/2 : win_spacing_x : Nx - wsize(1)/2 + 0.5;
		win_ctrs_y		= 0.5 + wsize(2)/2 : win_spacing_y : Ny - wsize(2)/2 + 0.5;
		[win_x,win_y]	= ndgrid(win_ctrs_x, win_ctrs_y);
		n_windows		= [length(win_ctrs_x) length(win_ctrs_y)];
        
        % calculate window weighting function

	    window_weightA	   = window_weight_fun(wsize, wtype, 'A'); % A is not needed here
        
		window_weightB	   = window_weight_fun(wsize, wtype, 'B');% B is not needed here

        b_mask =vector_mask{pass};


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
		
		if pass > 1
			%% smooth the displacement field from the previous pass
			% cutoff wavelength for previous pass corresponds to about the
			% window size: we need to size this filter relative to the cutoff frequency  
			ksize_filt				= round(wsize_old ./ win_spacing_old) + 1;
            if mod(ksize_filt, 2) == 0
                % If it's even, increment it to make it odd
                ksize_filt = ksize_filt + 1;
            end
			sd						= sqrt(prod(ksize_filt))/3*0.65;
			% median filter to avoid errors due to spurious vectors in
			% result
			%delta_ab_old(:,:,1)		= medfilt2(delta_ab_old(:,:,1), ksize_filt);
			%delta_ab_old(:,:,2)		= medfilt2(delta_ab_old(:,:,2), ksize_filt);
			% smoothing filter
			% G_smooth 				= fspecial('gaussian', ksize_filt, sd);
			% delta_ab_old			= convn(delta_ab_old, G_smooth, 'same'); %% use imgaussfilt here
            delta_ab_old(:,:,1)     =imgaussfilt(delta_ab_old(:,:,1),sd,'FilterSize',ksize_filt); %removes edge effects
            delta_ab_old(:,:,2)     =imgaussfilt(delta_ab_old(:,:,2),sd,'FilterSize',ksize_filt); %removes edge effects
			
			% Davis behaviour: filter with a fixed kernel size
			%G_smooth				= fspecial('average', ksize_filt);
						
			%% define dense velocity predictor field deformation field
			delta_ab_dense          = zeros([im_size 2]);
            if strcmp('spline', setup.instantaneous.interpolator)
		        delta_ab_dense(:,:,1)	= interpn(win_ctrs_x_old, win_ctrs_y_old, delta_ab_old(:,:,1), im_imat, im_jmat, 'spline', 0);  %%linear
		        delta_ab_dense(:,:,2)	= interpn(win_ctrs_x_old, win_ctrs_y_old, delta_ab_old(:,:,2), im_imat, im_jmat, 'spline', 0);
            else
                delta_ab_dense(:,:,1)	= interpn(win_ctrs_x_old, win_ctrs_y_old, delta_ab_old(:,:,1), im_imat, im_jmat, 'linear', 0);  %%linear
		        delta_ab_dense(:,:,2)	= interpn(win_ctrs_x_old, win_ctrs_y_old, delta_ab_old(:,:,2), im_imat, im_jmat, 'linear', 0);
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
			G_smooth_predictor		= window_weight_fun(ksize_filt, wtype,'A');
	        G_smooth_predictor		= G_smooth_predictor / sum(sum(G_smooth_predictor));
            padding_amount          = floor(ksize_filt(1) / 2);
            padded_image            = padarray(delta_ab_old, [padding_amount, padding_amount], 'symmetric', 'both');
            delta_ab_filt           = convn(padded_image,G_smooth_predictor,'valid');
	        % delta_ab_filt           = convn(delta_ab_old, G_smooth_predictor, 'same');
            if strcmp('spline', setup.instantaneous.interpolator)
                delta_ab_pred(:,:,1)    = interpn(win_ctrs_x_old, win_ctrs_y_old, delta_ab_filt(:,:,1), win_x, win_y, 'spline', 0); %linear
                delta_ab_pred(:,:,2)    = interpn(win_ctrs_x_old, win_ctrs_y_old, delta_ab_filt(:,:,2), win_x, win_y, 'spline', 0);  %linear
            else
                delta_ab_pred(:,:,1)    = interpn(win_ctrs_x_old, win_ctrs_y_old, delta_ab_filt(:,:,1), win_x, win_y, 'linear', 0); %linear
                delta_ab_pred(:,:,2)    = interpn(win_ctrs_x_old, win_ctrs_y_old, delta_ab_filt(:,:,2), win_x, win_y, 'linear', 0);  %linear
            end		
			% Davis behaviour: do not filter 
			%delta_ab_filt			= delta_ab_old;

			%% apply window deformation
			%A_prime					= interpn(im_i, im_j, A, im_mesh_A(:,:,1), im_mesh_A(:,:,2), 'linear', 0);
			%B_prime					= interpn(im_i, im_j, B, im_mesh_B(:,:,1), im_mesh_B(:,:,2), 'linear', 0);
			A_prime					= interp2custom(A, im_mesh_A(:,:,1)-1, im_mesh_A(:,:,2)-1, wdef_kernel, wdef_ksize);
			B_prime					= interp2custom(B, im_mesh_B(:,:,1)-1, im_mesh_B(:,:,2)-1, wdef_kernel, wdef_ksize);
		else
			%% do not apply window deformation as there is no A priori information on first pass
			delta_ab_pred			= zeros([n_windows, 2]);
			A_prime					= A;
			B_prime					= B;
		end
		
		
		%% PIV cross-correlation
		% translate integer codes for peak finder
		peak_finder		= 3;
		if		strcmp('gauss3', setup.instantaneous.peakFinder)
			peak_finder = 3;
		elseif	strcmp('gauss4',  setup.instantaneous.peakFinder)
			peak_finder	= 4;
		elseif	strcmp('gauss5',  setup.instantaneous.peakFinder)
			peak_finder = 5;
		elseif	strcmp('gauss6',  setup.instantaneous.peakFinder)
			peak_finder = 6;
		end
        % C accelerated code for peak detection foundn in : SOURCE_FILES\core\xcorr2d
        [peak_loc_x, peak_loc_y, peak_height,~,~,~,~] = ...
	    PIV_2d_cross_correlate( A_prime, B_prime, single(win_ctrs_x-1), single(win_ctrs_y-1), single(wsize), single(window_weightA), n_peaks, peak_finder,single(window_weightB),Ensemble,single(b_mask'));
        
        %% reject masked and unphysical vectors
        peak_choice		= ones(n_windows);
		
		% apply mask
		peak_loc_x(:,b_mask)	= nan;
		peak_loc_y(:,b_mask)	= nan;
		peak_height(:,b_mask)	= nan;


        
        % mask vectors with excessively large displacements
        b_large_disp    = abs(peak_loc_x) > wsize(1) / 4 ...
                        | abs(peak_loc_y) > wsize(2) / 4;
        peak_loc_x(b_large_disp) = nan;
        peak_loc_y(b_large_disp) = nan;
        peak_height(b_large_disp)= nan;

        
		
		% reformat
		peak_loc_x		= double(permute(peak_loc_x, [2 3 1]));
		peak_loc_y		= double(permute(peak_loc_y, [2 3 1]));
		peak_height		= double(permute(peak_height, [2 3 1]));

		
		% add predictor displacement
		peak_loc_x		= bsxfun(@plus, peak_loc_x, delta_ab_pred(:,:,1));
		peak_loc_y		= bsxfun(@plus, peak_loc_y, delta_ab_pred(:,:,2));

        

		% choose peak
		[ii,jj]			= ndgrid(1:n_windows(1), 1:n_windows(2));
		idx_choice		= sub2ind([n_windows n_peaks], ii, jj, peak_choice);
        
		ux_mat			= peak_loc_x(idx_choice);
		uy_mat			= peak_loc_y(idx_choice);

		nan_mask		= isnan(ux_mat) | isnan(uy_mat);

		%% normalised median test filter
		% apply universal outlier detection filter
        
		nan_mask				= nan_mask | PIV_2D_outlier(ux_mat, uy_mat, setup.instantaneous.epsilon, setup.instantaneous.epsilonThreshold);
        
		
		% select a better peak
		if setup.instantaneous.secondaryPeaks
			for pk = 2 : n_peaks
				% select next best peak
				peak_choice(nan_mask)	= peak_choice(nan_mask)+1;
				idx_choice				= sub2ind([n_windows n_peaks], ii, jj, peak_choice);
				ux_mat					= peak_loc_x(idx_choice);
				uy_mat					= peak_loc_y(idx_choice);

				% re-apply filter
				nan_mask				= nan_mask | PIV_2D_outlier(ux_mat, uy_mat, setup.instantaneous.epsilon, setup.instantaneous.epsilonThreshold);
				if ~any(any(nan_mask))
					break;
				end
			end
		end

		
		%% update peak height and Q factor
		peak_mag		= peak_height(idx_choice);
        nan_mask = nan_mask |  peak_mag < 0.2;
		% Q factor is defined as the chosen peak height, relative to next smallest peak height
		Q_mat			= peak_height ./ peak_height(:,:,[2:end,end]);
		Q				= Q_mat(idx_choice);
		
		%% inpaint any remaining spurious vectors
		if any(any(nan_mask))
			% mark bad vectors as NaN
			ux_mat(nan_mask)	= nan;
			uy_mat(nan_mask)	= nan;

  
            ux_mat(b_mask)	= 0;
	        uy_mat(b_mask)	= 0;
%             if pass == 5
%                 % Close any previous figures
%                 close all;
%                 
%                 % Create a new figure and maximize it
%                 figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);
%                 
%                 % Initialize a cell array to store method names
%                 methods = {'Method 0', 'Method 1', 'Method 2', 'Method 3', 'Method 4'};
%                 
%                 % Create subplots for uy_mat
%                 for method = 0:4
%                     tic
%                     uy_inpainted = inpaint_nans(uy_mat, method); % Apply method to uy_mat
%                     toc
%                     
%                     subplot(5, 1, method+1 ); % Create subplots for uy_mat (bottom row)
%                     imagesc(uy_inpainted, [-10, 10]); % Adjust colormap as needed
%                     colorbar;
%                     title(['uy\_mat: ', methods{method + 1}]);
%                     daspect([1 1 1])
%                 end
%                 
%                 % Add a supertitle for the entire figure
%                 sgtitle('Comparison of inpaint\_nans Methods (0-4)');
%             end


            ux_mat			= inpaint_nans(ux_mat,3); % 3 is slowest but prelim analysis says will lose ~5 mins of 18000 images
            uy_mat			= inpaint_nans(uy_mat,3);
            
           
			peak_choice(nan_mask)= 0;
			Q(nan_mask)			= 0;
			peak_mag(nan_mask)	= 0;
		end
		
		%% zero out masked vectors
		% zero out masked vectors
		ux_mat(b_mask)	= 0;
		uy_mat(b_mask)	= 0;

		
								
		%% save displacement field from this pass
		% into "old" displacement field structure
		% at this juncture, pad the displacement field, as it will require
		% interpolation over the whole domain later
		
		% extend PIV grid
		win_ctrs_x_pre		= 1 : win_spacing_x : win_ctrs_x(1) - win_spacing_x/2;
        if isempty(win_ctrs_x_pre)
            win_ctrs_x_pre = 1;
        end
		win_ctrs_y_pre		= 1 : win_spacing_y : win_ctrs_y(1) - win_spacing_y/2;
        if isempty(win_ctrs_y_pre)
            win_ctrs_y_pre = 1;
        end
		win_ctrs_x_post		= im_size(1) : -win_spacing_x : win_ctrs_x(end) + win_spacing_x/2;
        if isempty(win_ctrs_x_post)
            win_ctrs_x_post = im_size(1);
        end
		win_ctrs_y_post		= im_size(2) : -win_spacing_y : win_ctrs_y(end) + win_spacing_y/2;
        if isempty(win_ctrs_y_post)
            win_ctrs_y_post = im_size(2);
        end
		win_ctrs_x_old		= [win_ctrs_x_pre win_ctrs_x win_ctrs_x_post(end:-1:1)];
		win_ctrs_y_old		= [win_ctrs_y_pre win_ctrs_y win_ctrs_y_post(end:-1:1)];
		% size of padding around edges
		n_pre				= [length(win_ctrs_x_pre ) length(win_ctrs_y_pre )];
		n_post				= [length(win_ctrs_x_post) length(win_ctrs_y_post)];
		% add padding to edges
		delta_ab_old		= cat(3, ux_mat, uy_mat);
		delta_ab_old		= padarray(delta_ab_old, [n_pre  0], 'replicate', 'pre' );
		delta_ab_old		= padarray(delta_ab_old, [n_post 0], 'replicate', 'post');
		% make note of spacing
		win_spacing_old		= [win_spacing_x win_spacing_y];
		wsize_old			= wsize;
        % store different information depending on flags to be space
        % efficeiint
        if pass==n_passes
            piv_result(pass)	= struct('n_windows', n_windows, ...
									    'win_x', interp1(1:Ny, im_y, win_ctrs_y),...
									    'win_y', (interp1(1:Nx, im_x, win_ctrs_x)), ...
									    'ux', (uy_mat * im_dy / piv_dt), ...
									    'uy', -(ux_mat * im_dx / piv_dt), ...
									    'nan_mask', nan_mask, ...
									    'Q', Q, ...
									    'peak_mag', peak_mag, ...
									    'peak_choice', peak_choice, ...
									    'dt', piv_dt, ...
                                        'Predictor_Field',delta_ab_old,...
                                        'wsize_old',wsize,'win_spacing_old',win_spacing_old,'win_ctrs_x_old',...
                                        win_ctrs_x_old,'win_ctrs_y_old',win_ctrs_y_old,'b_mask',b_mask);
        else
            if ismember(pass,setup.instantaneous.runs)
                piv_result(pass)	= struct('n_windows', n_windows, ...
									        'win_x', interp1(1:Ny, im_y, win_ctrs_y), ...
									        'win_y', (interp1(1:Nx, im_x, win_ctrs_x)), ...
									        'ux', (uy_mat * im_dy / piv_dt), ...
									        'uy', -(ux_mat * im_dx / piv_dt), ...
									        'nan_mask', nan_mask, ...
									        'Q', Q, ...
									        'peak_mag', peak_mag, ...
									        'peak_choice', peak_choice, ...
									        'dt', piv_dt, ...
                                            'Predictor_Field',[],...
                                            'wsize_old',[],'win_spacing_old',[],'win_ctrs_x_old',...
                                            [],'win_ctrs_y_old',[],'b_mask',b_mask);
            else
                piv_result(pass)	=struct(	'n_windows', [], ...
								'win_x', [], 'win_y', [], ...
								'ux', [], 'uy', [], ...
								'nan_mask', [], 'Q', [], ...
								'peak_mag', [], 'peak_choice', [], ...
								'dt', [],'Predictor_Field',[],...
                                'wsize_old',[],'win_spacing_old',[],'win_ctrs_x_old', ...
                                [],'win_ctrs_y_old',[],'b_mask',[]);
            end

        end

	end
end
		


function [gaussResult_combined, status_combined] = evaluate_correlation_planes(Ensemble_guess_setup, piv_result, pass, mask, setup)

    % evaluate_correlation_planes: Processes correlation planes in parallel to compute
    % Gaussian fit results for velocity probability distribution (PD) estimates.
    %
    % This function efficiently evaluates the Gaussian fits across correlation planes 
    % by leveraging parallel processing. The data is processed in batches, and each 
    % worker handles a subset of the data, significantly speeding up the computation.
    %
    % Inputs:
    %   Ensemble_guess_setup - Structure containing initial guess data for ensemble
    %                          processing. Relevant fields include:
    %                          * Ensemble_guess_setup.PD_guess_x: Initial guess for 
    %                            the x-component of the velocity PD.
    %                          * Ensemble_guess_setup.PD_guess_y: Initial guess for 
    %                            the y-component of the velocity PD.
    %                          * Ensemble_guess_setup.X1, Ensemble_guess_setup.X2: 
    %                            Grids for the correlation analysis.
    %                          * Ensemble_guess_setup.centralIndex: Index for central 
    %                            point in the window.
    %                          * Ensemble_guess_setup.x_guess, Ensemble_guess_setup.y_guess:
    %                            Initial guesses for the x and y locations in the grid.
    %   piv_result - Structure containing PIV results from previous passes.
    %                For the current pass, piv_result(pass) contains:
    %                * piv_result(pass).pointspreadA: Array for the point spread function 
    %                  A used in the correlation.
    %                * piv_result(pass).pointspreadB: Array for the point spread function 
    %                  B used in the correlation.
    %                * piv_result(pass).correlation_plane: The correlation plane 
    %                  between pointspread A and B.
    %                * piv_result(pass).wsize_old: The size of the old windows used in PIV.
    %   pass - Integer representing the current pass number in the PIV process.
    %   mask - Logical matrix indicating valid locations for processing, with NaN or
    %          zero entries representing locations that should be skipped.
    %
    % Outputs:
    %   gaussResult_combined - 3D matrix containing the results from fitting a Gaussian 
    %                           to the correlation plane for each point in the grid.
    %   status_combined - 3D matrix containing the status of each fit. A value of 0 
    %                     indicates a failed fit, while 1 means success.
    %
    % Procedure:
    % 1. **Extract the relevant data**:
    %    - The function retrieves the point spread functions (AA, BB), the correlation 
    %      plane (AB), and the initial PD guesses from `Ensemble_guess_setup` and `piv_result`.
    % 
    % 2. **Parallelization**:
    %    - The function employs `parfor` to process each point in the correlation plane 
    %      in parallel. Each worker processes a subset of rows, extracting the necessary 
    %      data and computing Gaussian fits. This is a low memory process so can be split accross max cores.
    %      The processing is done inside the mex file mex_marquadt_gaussian which jointly evaluates the auto and cross correlation functions.
    %
    % 3. **Collect Results**:
    %    - The results from each worker are concatenated back into the original 3D 
    %      matrices for Gaussian fit results and status indicators.
    %
    % Speed improvements are achieved by parallelizing the computationally intensive 
    % Gaussian fitting process, allowing for better utilization of available CPU resources.

    % Extract key variables
    AA = piv_result(pass).pointspreadA;  % Size: [m, n_rows, n_cols]
    [~, n_rows, n_cols] = size(AA);
    BB = piv_result(pass).pointspreadB;   % Size: [m, n_rows, n_cols]
    AB = piv_result(pass).correlation_plane; % Size: [m, n_rows, n_cols]
    guess_sx_PD = Ensemble_guess_setup.PD_guess_x;  % Size: [n_rows, n_cols]
    guess_sy_PD = Ensemble_guess_setup.PD_guess_y;  % Size: [n_rows, n_cols]

    % Initialize output variables outside the loop
    gauss_result = zeros(n_rows, n_cols, 13);
    status = zeros(n_rows, n_cols);

    fprintf('Evaluating Correlation planes at time %s\n', datetime('now'));
    wsize = piv_result(pass).wsize_old;

    % Setup parallel pool if needed
    Setup_parpool(setup, 'Max');
    rowbatch = 1;
    nextMarker=10;


    for batch = 1:rowbatch:n_rows
        % Compute the end row for the current batch
        endRow = min(batch + rowbatch - 1, n_rows);  % Ensure we don't exceed the row limit
        relevantrows = batch:endRow;                % Get relevant rows for this batch
        
        % Extract slices for the current batch and reshape them into 2D arrays
        AA_local = reshape(AA(:, relevantrows, :), size(AA, 1), []);  % Size: [m, n_rows * n_cols]
        BB_local = reshape(BB(:, relevantrows, :), size(BB, 1), []);  % Size: [m, n_rows * n_cols]
        AB_local = reshape(AB(:, relevantrows, :), size(AB, 1), []);  % Size: [m, n_rows * n_cols]
        
        % Flatten local guess and mask arrays
        local_guess_sx_PD = reshape(guess_sx_PD(relevantrows, :), [], 1);  % Size: [batch_size * n_cols, 1]
        local_guess_sy_PD = reshape(guess_sy_PD(relevantrows, :), [], 1);  % Size: [batch_size * n_cols, 1]
        local_mask = reshape(mask(relevantrows, :), [], 1);                 % Size: [batch_size * n_cols, 1]
        
        % Flatten result arrays
        gauss_result_local = zeros(length(relevantrows) * n_cols, 13);  % Flattened: Size: [(batch_size * n_cols), 13]
        status_local = zeros(length(relevantrows) * n_cols, 1);          % Flattened: Size: [(batch_size * n_cols), 1]

        % Use parfor to iterate over all elements in the current batch
        parfor idx = 1:(length(relevantrows) * n_cols)
            % Check the mask for the current index
            if local_mask(idx)
                % Skip masked areas
                continue;
            end
            
            % Extract local AA, BB, and AB for the current index
            BB_local_rc = BB_local(:, idx);
            AB_local_rc = AB_local(:, idx);
            AA_local_rc = AA_local(:, idx);
            
            % Validate data before proceeding
            if any(isnan(BB_local_rc)) || any(isnan(AB_local_rc)) || any(isnan(AA_local_rc))
                % Set default values for NaN input
                gauss_result_local(idx, :) = zeros(1, 13);
                status_local(idx) = 0;
                continue;
            end
            
            % Get local PD guesses
            guess_sx_PD_local = local_guess_sx_PD(idx);
            guess_sy_PD_local = local_guess_sy_PD(idx);
            
            % Prepare the real correlation data
            real_corr = double([AA_local_rc; BB_local_rc; AB_local_rc]);  % [3 * m x 1]
            
            % Construct the initial guess
            if pass ~= 1
                initialGuess = double([
                    AA_local_rc(Ensemble_guess_setup.centralIndex), BB_local_rc(Ensemble_guess_setup.centralIndex), AB_local_rc(Ensemble_guess_setup.centralIndex), ...
                    1, 1, 0, ...
                    guess_sx_PD_local, guess_sy_PD_local, 0, ...
                    Ensemble_guess_setup.x_guess, Ensemble_guess_setup.y_guess, Ensemble_guess_setup.x_guess, Ensemble_guess_setup.y_guess
                ]);
            else
                % If it's the first pass, compute the maximum location for AB
                [~, max_idx] = max(AB_local_rc);
                [guess_y_AB, guess_x_AB] = ind2sub(wsize, max_idx);
                center_idx_AA = Ensemble_guess_setup.centralIndex;
                center_idx_BB = Ensemble_guess_setup.centralIndex;

                if setup.ensemble.noisy
                    gaussian_radius = 8;
    
                    [cy_AA, cx_AA] = ind2sub(wsize, center_idx_AA);
                    [cy_BB, cx_BB] = ind2sub(wsize, center_idx_BB);
                    [cy_AB, cx_AB] = ind2sub(wsize, max_idx);
                    
                    % --- Build grid of 2D coordinates for each element ---
                    [XX, YY] = meshgrid(1:wsize(2), 1:wsize(1));  % Note: meshgrid uses (cols, rows)
                    
                    % --- Compute Gaussian window centered at each correlation center ---
                    gaussian_window = @(cx, cy) exp(-((XX - cx).^2 + (YY - cy).^2) / (2 * gaussian_radius^2));
                    
                    w_AA = gaussian_window(cx_AA, cy_AA);
                    w_BB = gaussian_window(cx_BB, cy_BB);
                    w_AB = gaussian_window(cx_AB, cy_AB);
                    w_AA = w_AA(:);
                    w_BB = w_BB(:);
                    w_AB = w_AB(:);
                    
                    % --- Apply window to original 1D arrays ---
                    AA_weighted = AA_local_rc .* w_AA;
                    BB_weighted = BB_local_rc .* w_BB;
                    AB_weighted = AB_local_rc .* w_AB;
                    real_corr = double([AA_weighted; BB_weighted; AB_weighted]);
                    
                end                 
                initialGuess = double([
                    AA_local_rc(Ensemble_guess_setup.centralIndex), BB_local_rc(Ensemble_guess_setup.centralIndex), AB_local_rc(max_idx), ...
                    1, 1, 0, ...
                    guess_sx_PD_local, guess_sy_PD_local, 0, ...
                    Ensemble_guess_setup.x_guess, Ensemble_guess_setup.y_guess, guess_x_AB, guess_y_AB
                ]);
            end
            

            
            % Add safeguards to ensure inputs are valid
            if any(~isfinite(Ensemble_guess_setup.X1(:))) || any(~isfinite(Ensemble_guess_setup.X2(:))) || ...
               any(~isfinite(real_corr)) || any(~isfinite(initialGuess))
                error('Non-finite values detected in inputs');
            end
            
            [gaussResult, solver_status] = mex_marquadt_gaussian(Ensemble_guess_setup.X1(:), Ensemble_guess_setup.X2(:), real_corr, initialGuess);
            
            % Check output validity
            if any(~isfinite(gaussResult))
                warning('Non-finite values detected in output at index %d', idx);
                gaussResult = zeros(1, 13);
                solver_status = 0;
            end
       
            
            % Store results directly in the local result arrays
            gauss_result_local(idx, :) = gaussResult;
            status_local(idx) = solver_status;
        end
        
        % Append local results back to the global arrays
        gauss_result(relevantrows, :, :) = reshape(gauss_result_local, [length(relevantrows), n_cols, 13]);
        status(relevantrows, :) = reshape(status_local, [length(relevantrows), n_cols]);
        progress = (batch/n_rows)*100;
        if progress >= nextMarker
            currentTime = datetime('now', 'Format', 'HH:mm:ss');  % Get current time in HH:mm:ss format
            fprintf('Progress: %.0f%% completed. Current time: %s\n', nextMarker, char(currentTime));
            nextMarker = ceil(progress / 10) * 10;  % Update the next marker
        end

    end

    fprintf('Processing complete at time %s\n', datetime('now'));


    % Assign combined results to output
    gaussResult_combined = gauss_result;
    status_combined = status;

end

function [piv_result] = process_correlation_planes(piv_result, gaussResult_full, status_full,setup, pass, mask,predu,predv, guess_setup);
    % process_correlation_planes.m
    % ============================
    % This function analyzes the correlation planes from Particle Image Velocimetry (PIV) data 
    % and updates the PIV results in a structured format. It processes the 
    % Gaussian fit results of the cross-correlation data, applying a series of checks and 
    % validation rules based on the physical setup, and then calculates various 
    % parameters such as peak heights, turbulent velocities, and covariance values. 
    % The function operates in parallel to enhance performance on large datasets.
    % 
    % @inputs:
    % - piv_result: A structure containing the PIV results for the current pass.
    % - gaussResult_full: A 3D matrix of Gaussian fit results with dimensions (n_rows x n_cols x 16).
    % - status_full: A 2D matrix of fit statuses (n_rows x n_cols), where each element indicates 
    %   the success (or failure) of the Gaussian fit for a corresponding
    %   pixel. 
    %   1 = solver didn't converge
    %   2 = means AB peak height is non-valid
    %   3 = breaks 1/4 displacement rule
    %   4 = Gaussian spread too large for window
    %   5 = AB smaller than AA
    %   6 = Inside mask
    % - setup: A structure containing the setup parameters for the ensemble, including 
    %   window size and run type (e.g., 'single' or 'double').
    % - pass: The current pass index for updating `piv_result`.
    % - mask: The mask array used for identifying valid pixels.
    % - predu: A vector of pre-processed x-displacement corrections.
    % - predv: A vector of pre-processed y-displacement corrections.
    % - guess_setup: A structure containing the initial guesses for the central index and other parameters.
    % 
    % @outputs:
    % - piv_result: Updated structure containing the processed PIV results for the current pass, 
    %   including peak heights, velocities, and uncertainty estimates (standard deviations) for each point.
    % 
    % The function performs the following key operations:
    % - Validates Gaussian fit results based on physical constraints, including checking the 
    %   centrality of peaks and whether the fitted Gaussians fit within the window size.
    % - Computes key parameters such as peak heights, displacement, turbulent stresses, 
    %   eigenvalues, and standard deviations (for both x and y components).
    % - Stores computed results in the `piv_result` structure for later analysis.
    centralIndex = guess_setup.centralIndex;
    runtype = setup.ensemble.type{pass};
    wsize = setup.ensemble.windowSize(pass, :);
    SumWindow = setup.ensemble.sumWindow;
    [n_rows,n_cols] = size(status_full);
    AA = piv_result(pass).pointspreadA(centralIndex,:,:);  % Size: [m, n_rows, n_cols]
    BB = piv_result(pass).pointspreadB(centralIndex,:,:);   % Size: [m, n_rows, n_cols]
    AA = reshape(AA, [], size(AA, 3));
    BB = reshape(BB, [], size(BB, 3));
    NanReason = zeros([n_rows,n_cols]);
    peakheights_A = nan([n_rows,n_cols]);
    peakheights_B = nan([n_rows,n_cols]);
    peakheights_AB = nan([n_rows,n_cols]);
    uxa = nan([n_rows,n_cols]);
    uya = nan([n_rows,n_cols]);
    ux = nan([n_rows,n_cols]);
    uy = nan([n_rows,n_cols]);
    eigenval_x = nan([n_rows,n_cols]);
    eigenval_y = nan([n_rows,n_cols]);
    rotation_PD = nan([n_rows,n_cols]);
    UU_stress = nan([n_rows,n_cols]);
    VV_stress = nan([n_rows,n_cols]);
    UV_stress = nan([n_rows,n_cols]);
    sig_AB_x = nan([n_rows,n_cols]);
    sig_PD_x = nan([n_rows,n_cols]);
    sig_A_x = nan([n_rows,n_cols]);
    sig_AB_y = nan([n_rows,n_cols]);
    sig_PD_y = nan([n_rows,n_cols]);
    sig_A_y = nan([n_rows,n_cols]);
    sig_AB_xy = nan([n_rows,n_cols]);
    sig_PD_xy = nan([n_rows,n_cols]);
    sig_A_xy = nan([n_rows,n_cols]);

    rowbatch =1;  % computes 4 rows at a time
    fprintf('Processing correlation planes at time %s\n', datetime('now'));
    Setup_parpool(setup, 'Max')
    nextMarker =10;
    for batch = 1:rowbatch:n_rows
        % Compute the end row for the current batch
        endRow = min(batch + rowbatch - 1, n_rows);  % Ensure we don't exceed the row limit
        relevantrows = batch:endRow;                % Get relevant rows for this batch        
        % Flatten result arrays
        gauss_result_local = reshape(gaussResult_full(relevantrows,:,:),[],13);
        local_status = reshape(status_full(relevantrows, :), [], 1);
        local_NanReason = reshape(NanReason(relevantrows, :), [], 1);
        local_peakheights_A = reshape(peakheights_A(relevantrows, :), [], 1);
        local_peakheights_B = reshape(peakheights_B(relevantrows, :), [], 1);
        local_peakheights_AB = reshape(peakheights_AB(relevantrows, :), [], 1);
        local_uxa = reshape(uxa(relevantrows, :), [], 1);
        local_uya = reshape(uya(relevantrows, :), [], 1);
        local_ux = reshape(ux(relevantrows, :), [], 1);
        local_uy = reshape(uy(relevantrows, :), [], 1);
        local_Uturb = reshape(UU_stress(relevantrows, :), [], 1);
        local_Vturb = reshape(VV_stress(relevantrows, :), [], 1);
        local_UturbVturb = reshape(UV_stress(relevantrows, :), [], 1);
        local_sig_AB_x = reshape(sig_AB_x(relevantrows, :), [], 1);
        local_sig_PD_x = reshape(sig_PD_x(relevantrows, :), [], 1);
        local_sig_A_x = reshape(sig_A_x(relevantrows, :), [], 1);
        local_sig_AB_y = reshape(sig_AB_y(relevantrows, :), [], 1);
        local_sig_PD_y = reshape(sig_PD_y(relevantrows, :), [], 1);
        local_sig_A_y = reshape(sig_A_y(relevantrows, :), [], 1);
        local_sig_AB_xy = reshape(sig_AB_xy(relevantrows, :), [], 1);
        local_sig_PD_xy = reshape(sig_PD_xy(relevantrows, :), [], 1);
        local_sig_A_xy = reshape(sig_A_xy(relevantrows, :), [], 1);
        local_AA = reshape(AA(relevantrows, :), [], 1);
        local_BB = reshape(BB(relevantrows, :), [], 1);
        local_mask = reshape(mask(relevantrows, :), [], 1);

        
        % Iterate over local rows and columns
        parfor idx = 1:(length(relevantrows) * n_cols)
            
            if local_mask(idx)
                local_NanReason(idx) = 6;
                continue
            end
% 
            status = local_status(idx);
            if status ~= 0
                local_NanReason(idx) = 1;  %% status 1 means solver didnt converge
                continue;
            end

            % Extract the correlation planes and gauss result for the current (idx)
            gaussresult = gauss_result_local(idx, :);
            AA_local = local_AA( idx);
            BB_local = local_BB(idx);

            % Compute AB and check its validity
            AB_value = gaussresult(3) / (sqrt(AA_local * BB_local));
            if ~isreal(AB_value) || AB_value < 0 || AB_value > 1
                local_NanReason(idx) = 2; % status 2 means AB peak height is non valid
                continue;
            end
% 
            % Check bounds based on runtype (single or double window)
            if strcmp(runtype, 'single')
                % Define the central range for SumWindow
                centerStart1 = (SumWindow(1) / 2) - (SumWindow(1) / 4);
                centerEnd1 = (SumWindow(1) / 2) + (SumWindow(1) / 4);
                
                centerStart2 = (SumWindow(2) / 2) - (SumWindow(2) / 4);
                centerEnd2 = (SumWindow(2) / 2) + (SumWindow(2) / 4);
            
                % Check if gaussresult(12) and gaussresult(13) are within the central range
                if gaussresult(12) < centerStart1 || gaussresult(12) > centerEnd1 || ...
                   gaussresult(13) < centerStart2 || gaussresult(13) > centerEnd2
                    local_NanReason(idx) = 3;  % status 3 breaks 1/4 rule 
                    continue;
                    
                end
% %                 % checks whether gaussian can fit on the grid 
%                 if gaussresult(4) + gaussresult(7) > (SumWindow(1) / 3.5) || gaussresult(5) + gaussresult(8) > (SumWindow(1) / 3.5)
%                     local_NanReason(idx) = 4; % gaussian spread too large for window
%                     continue;
%                     
%                 end
            else
                % Define the central range for wsize
                centerStart1 = (wsize(1) / 2) - (wsize(1) / 4);
                centerEnd1 = (wsize(1) / 2) + (wsize(1) / 4);
                
                centerStart2 = (wsize(2) / 2) - (wsize(2) / 4);
                centerEnd2 = (wsize(2) / 2) + (wsize(2) / 4);
                if pass ~= 1
                    % Check if gaussresult(12) and gaussresult(13) are within the central range
                    if gaussresult(12) < centerStart1 || gaussresult(12) > centerEnd1 || ...
                       gaussresult(13) < centerStart2 || gaussresult(13) > centerEnd2
                       local_NanReason(idx) = 3;  % status 3 breaks 1/4 rule 
                       continue;
                    end
                   
                end
%                 if (gaussresult(4) + gaussresult(7) > (wsize(1) / 3.5) || gaussresult(5) + gaussresult(8) > (wsize(1) / 3.5)) && pass> setup.ensemble.convergedRun
%                     local_NanReason(idx) = 4; % gaussian spread too large for window
%                     continue;
%                     
%                 end
            end
            if (gaussresult(7) || gaussresult(8)) <0 
                local_NanReason(idx) = 5; % AB smaller than AA
                continue
            end

            

            % Compute the peak heights
            local_peakheights_A(idx) = gaussresult(1) / AA_local;
            local_peakheights_B(idx) = gaussresult(2) / BB_local;
            local_peakheights_AB(idx) = gaussresult(3) / sqrt(AA_local * BB_local);

            % Extract other parameters from the gauss result
            sx_AA = gaussresult(4);
            sy_AA = gaussresult(5);
            sxy_AA = gaussresult(6);
            sx_PD = gaussresult(7);
            sy_PD = gaussresult(8);
            sxy_PD = gaussresult(9);

            % Compute displacement based on runtype
            if strcmp(runtype, 'single')
                local_uxa(idx) = gaussresult(10) - (SumWindow(1) / 2 + 1);
                local_uya(idx) = gaussresult(11) - (SumWindow(2) / 2 + 1);
                local_ux(idx) = gaussresult(12) - (SumWindow(1) / 2 + 1);
                local_uy(idx) = gaussresult(13) - (SumWindow(2) / 2 + 1);
            else
                local_uxa(idx) = gaussresult(10) - (wsize(1) / 2 + 1);
                local_uya(idx) = gaussresult(11) - (wsize(2) / 2 + 1);
                local_ux(idx) = gaussresult(12) - (wsize(1) / 2 + 1);
                local_uy(idx) = gaussresult(13) - (wsize(2) / 2 + 1);
            end


            % Compute turbulent velocities
            local_Uturb(idx) = sx_PD
            local_Vturb(idx) = sy_PD
            local_UturbVturb(idx) = sxy_PD
            local_sig_AB_x(idx) = (sx_PD + sx_AA);
            local_sig_PD_x(idx) = (sx_PD);
            local_sig_A_x(idx) = (sx_AA);
            local_sig_AB_y(idx) = (sy_PD + sy_AA);
            local_sig_PD_y(idx) = (sy_PD);
            local_sig_A_y(idx) = (sy_AA);
            local_sig_AB_xy(idx) =(sxy_PD + sxy_AA);
            local_sig_PD_xy(idx) = (sxy_PD);
            local_sig_A_xy(idx) = (sxy_AA);
        end
        piv_result(pass).peakheights_A(relevantrows, :) = reshape(local_peakheights_A, length(relevantrows), n_cols);
        piv_result(pass).peakheights_B(relevantrows, :) = reshape(local_peakheights_B, length(relevantrows), n_cols);
        piv_result(pass).peakheights_AB(relevantrows, :) = reshape(local_peakheights_AB, length(relevantrows), n_cols);
        piv_result(pass).NanReason(relevantrows, :) = reshape(local_NanReason, length(relevantrows), n_cols);
        piv_result(pass).ux(relevantrows, :) = reshape(local_ux, length(relevantrows), n_cols);
        piv_result(pass).uy(relevantrows, :) = reshape(local_uy, length(relevantrows), n_cols);
        piv_result(pass).UU_stress(relevantrows, :) = reshape(local_Uturb, length(relevantrows), n_cols);
        piv_result(pass).VV_stress(relevantrows, :) = reshape(local_Vturb, length(relevantrows), n_cols);
        piv_result(pass).UV_stress(relevantrows, :) = reshape(local_UturbVturb, length(relevantrows), n_cols);
        piv_result(pass).sig_AB_x(relevantrows, :) = reshape(local_sig_AB_x, length(relevantrows), n_cols);
        piv_result(pass).sig_A_x(relevantrows, :) = reshape(local_sig_A_x, length(relevantrows), n_cols);
        piv_result(pass).sig_AB_y(relevantrows, :) = reshape(local_sig_AB_y, length(relevantrows), n_cols);
        piv_result(pass).sig_A_y(relevantrows, :) = reshape(local_sig_A_y, length(relevantrows), n_cols);
        piv_result(pass).sig_AB_xy(relevantrows, :) = reshape(local_sig_AB_xy, length(relevantrows), n_cols);
        piv_result(pass).sig_A_xy(relevantrows, :) = reshape(local_sig_A_xy, length(relevantrows), n_cols);
        piv_result(pass).uxa(relevantrows, :) = reshape(local_uxa, length(relevantrows), n_cols);
        piv_result(pass).uya(relevantrows, :) = reshape(local_uya, length(relevantrows), n_cols);
        piv_result(pass).sig_PD_x(relevantrows, :) = reshape(local_sig_PD_x, length(relevantrows), n_cols);
        piv_result(pass).sig_PD_y(relevantrows, :) = reshape(local_sig_PD_y, length(relevantrows), n_cols);
        piv_result(pass).sig_PD_xy(relevantrows, :) = reshape(local_sig_PD_xy, length(relevantrows), n_cols);
        progress = (batch/n_rows)*100;
        if progress >= nextMarker
            currentTime = datetime('now', 'Format', 'HH:mm:ss');  % Get current time in HH:mm:ss format
            fprintf('Progress: %.0f%% completed. Current time: %s\n', nextMarker, char(currentTime));
            nextMarker = ceil(progress / 10) * 10;  % Update the next marker
        end


    end
    

    piv_result(pass).peakheights_A(mask) = NaN;  
    piv_result(pass).peakheights_B(mask) = NaN;
    piv_result(pass).peakheights_AB(mask) = NaN;   
    piv_result(pass).uxa(mask) = NaN;
    piv_result(pass).uya(mask) = NaN;
    piv_result(pass).ux = piv_result(pass).ux + predu;
    piv_result(pass).ux(mask) = NaN;
    piv_result(pass).uy = -(piv_result(pass).uy + predv);
    piv_result(pass).uy(mask) = NaN;
    piv_result(pass).UU_stress(mask) = NaN;
    piv_result(pass).VV_stress(mask) = NaN;
    piv_result(pass).UV_stress(mask) = NaN;
    piv_result(pass).sig_AB_x(mask) = NaN;
    piv_result(pass).sig_PD_x(mask) = NaN;
    piv_result(pass).sig_A_x(mask) = NaN;
    piv_result(pass).sig_AB_y(mask) = NaN;
    piv_result(pass).sig_PD_y(mask) = NaN;
    piv_result(pass).sig_A_y(mask) = NaN;
    piv_result(pass).sig_AB_xy(mask) = NaN;
    piv_result(pass).sig_PD_xy(mask) = NaN;
    piv_result(pass).sig_A_xy(mask) = NaN;

end

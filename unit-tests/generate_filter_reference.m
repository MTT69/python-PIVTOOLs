function generate_filter_reference()
% GENERATE_FILTER_REFERENCE  Create reference .mat files for Python test comparison.
%
% Generates deterministic synthetic image batches (no random seed issues
% across languages), runs POD_filter logic and Time_filter logic on them,
% and saves inputs + outputs + intermediates to .mat files that the Python
% pytest can load.
%
% Usage:
%   >> generate_filter_reference
%
% Output files (saved to unit-tests/test_output/):
%   pod_reference.mat   - POD filter inputs, intermediates, outputs
%   time_reference.mat  - Time filter inputs and outputs

    output_dir = fullfile(fileparts(mfilename('fullpath')), 'test_output');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    %% ===== POD Filter Reference =====
    fprintf('Generating POD filter reference data...\n');
    generate_pod_reference(output_dir);

    %% ===== Time Filter Reference =====
    fprintf('Generating Time filter reference data...\n');
    generate_time_reference(output_dir);

    fprintf('Done! Reference files saved to: %s\n', output_dir);
end


function generate_pod_reference(output_dir)
    % Create deterministic synthetic images:
    % - Strong background (mode 1-2) + weaker structure (mode 3) + noise
    % This ensures find_auto_mode has something meaningful to detect.

    n_images = 20;
    H = 64;
    W = 48;
    n_pixels = H * W;

    % Build images from known modes so behaviour is deterministic
    % Use simple trig functions instead of random numbers
    [yy, xx] = meshgrid(1:W, 1:H);  % xx is H x W, yy is H x W

    % Mode 1: strong uniform background gradient
    bg1 = single(100 + 30 * xx / H);  % varies 100-130 across rows

    % Mode 2: vertical stripe pattern
    bg2 = single(20 * sin(2 * pi * yy / W));

    % Mode 3: weaker diagonal pattern
    bg3 = single(5 * cos(2 * pi * (xx + yy) / (H + W)));

    % Build frame1 and frame2 image stacks
    M_bloc1 = zeros(n_images, n_pixels, 'single');
    M_bloc2 = zeros(n_images, n_pixels, 'single');

    for k = 1:n_images
        % Temporal modulation of each mode
        t1 = 1.0 + 0.1 * sin(2 * pi * k / n_images);         % mode 1: slow variation
        t2 = 0.8 + 0.3 * cos(2 * pi * 2 * k / n_images);     % mode 2: faster
        t3 = 0.5 + 0.2 * sin(2 * pi * 3 * k / n_images);     % mode 3: even faster

        % "Noise" - deterministic high-frequency pattern unique per frame
        noise1 = single(2.0 * sin(xx * k * 0.7 + yy * k * 0.3));
        noise2 = single(2.0 * cos(xx * k * 0.5 + yy * k * 0.9));

        img1 = t1 * bg1 + t2 * bg2 + t3 * bg3 + noise1;
        img2 = t1 * bg1 + t2 * bg2 + t3 * bg3 + noise2;

        M_bloc1(k, :) = reshape(img1, 1, []);
        M_bloc2(k, :) = reshape(img2, 1, []);
    end

    % --- Run POD (same logic as POD_filter.m) ---
    hand.eps_auto_psi = 0.01;
    hand.eps_auto_sigma = 0.01;
    chunkLength = n_images;

    % Covariance
    C1 = M_bloc1 * M_bloc1';
    C2 = M_bloc2 * M_bloc2';

    % SVD
    [PSI1, LAMBDA1] = svd(C1);
    eigVal1 = diag(LAMBDA1);

    [PSI2, LAMBDA2] = svd(C2);
    eigVal2 = diag(LAMBDA2);

    % Auto mode selection
    N_auto1 = find_auto_mode(PSI1, eigVal1, hand, chunkLength);
    N_auto2 = find_auto_mode(PSI2, eigVal2, hand, chunkLength);

    fprintf('  Frame1: removing %d modes\n', N_auto1);
    fprintf('  Frame2: removing %d modes\n', N_auto2);

    % Evaluate PHI and TCoeff
    [PHI_bloc1, TCoeff_bloc1] = evaluatePHITCoeff(M_bloc1, PSI1, N_auto1);
    [PHI_bloc2, TCoeff_bloc2] = evaluatePHITCoeff(M_bloc2, PSI2, N_auto2);

    % Apply filter (subtract modes)
    M_filtered1 = M_bloc1;
    M_filtered2 = M_bloc2;

    for ImNo = 1:n_images
        tmp1 = M_bloc1(ImNo, :);
        tmp2 = M_bloc2(ImNo, :);

        for mod_i = 1:N_auto1
            tmp1 = tmp1 - (TCoeff_bloc1{mod_i}(ImNo, 1) * PHI_bloc1{mod_i}(:, 1)');
        end
        for mod_i = 1:N_auto2
            tmp2 = tmp2 - (TCoeff_bloc2{mod_i}(ImNo, 1) * PHI_bloc2{mod_i}(:, 1)');
        end

        M_filtered1(ImNo, :) = tmp1;
        M_filtered2(ImNo, :) = tmp2;
    end

    % Save everything Python needs
    save(fullfile(output_dir, 'pod_reference.mat'), ...
        'M_bloc1', 'M_bloc2', ...
        'eigVal1', 'eigVal2', ...
        'PSI1', 'PSI2', ...
        'N_auto1', 'N_auto2', ...
        'M_filtered1', 'M_filtered2', ...
        'n_images', 'H', 'W', ...
        '-v7');

    fprintf('  Saved pod_reference.mat\n');
end


function generate_time_reference(output_dir)
    % Time filter: subtract per-pixel minimum across temporal batch.

    n_images = 20;
    H = 64;
    W = 48;

    [yy, xx] = meshgrid(1:W, 1:H);

    bg1 = single(100 + 30 * xx / H);
    bg2 = single(20 * sin(2 * pi * yy / W));

    % Build raw images as (n_images, H, W) - 3D array
    images_frame1 = zeros(n_images, H, W, 'single');
    images_frame2 = zeros(n_images, H, W, 'single');

    for k = 1:n_images
        t1 = 1.0 + 0.1 * sin(2 * pi * k / n_images);
        t2 = 0.8 + 0.3 * cos(2 * pi * 2 * k / n_images);

        noise1 = single(2.0 * sin(xx * k * 0.7 + yy * k * 0.3));
        noise2 = single(2.0 * cos(xx * k * 0.5 + yy * k * 0.9));

        images_frame1(k, :, :) = t1 * bg1 + t2 * bg2 + noise1;
        images_frame2(k, :, :) = t1 * bg1 + t2 * bg2 + noise2;
    end

    % Compute time minimum per channel
    time_min1 = squeeze(min(images_frame1, [], 1));  % H x W
    time_min2 = squeeze(min(images_frame2, [], 1));

    % Subtract
    filtered_frame1 = zeros(size(images_frame1), 'single');
    filtered_frame2 = zeros(size(images_frame2), 'single');

    for k = 1:n_images
        filtered_frame1(k, :, :) = squeeze(images_frame1(k, :, :)) - time_min1;
        filtered_frame2(k, :, :) = squeeze(images_frame2(k, :, :)) - time_min2;
    end

    save(fullfile(output_dir, 'time_reference.mat'), ...
        'images_frame1', 'images_frame2', ...
        'time_min1', 'time_min2', ...
        'filtered_frame1', 'filtered_frame2', ...
        'n_images', 'H', 'W', ...
        '-v7');

    fprintf('  Saved time_reference.mat\n');
end


function N_auto = find_auto_mode(PSI, eigVal, hand, chunkLength)
    % Automatic mode selection (identical to POD_filter.m)
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
        PHI{mod_i} = transpose(M) * PSI(:, mod_i);
        PHI{mod_i} = PHI{mod_i} / sqrt(sum(PHI{mod_i}.^2));
        TCoeff{mod_i} = M * PHI{mod_i};
    end
end

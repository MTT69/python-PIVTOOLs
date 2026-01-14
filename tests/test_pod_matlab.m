%% POD Filter Test Script - MATLAB Reference
% Load TIFFs directly, run POD, save results for comparison
% This is a standalone test that matches the Python implementation
%
% Usage:
%   Set data_path below to your test data directory containing:
%   B00001_A.tif, B00001_B.tif, B00002_A.tif, etc.

clear; clc;

%% Configuration - EDIT THESE PATHS AS NEEDED
data_path = '/Users/morgan/Documents/PIV_test';  % Path to test TIFF images
num_images = 50;  % Number of image pairs to load

% POD parameters (must match Python exactly)
eps_auto_psi = 0.01;
eps_auto_sigma = 0.01;

%% Load Images Directly
fprintf('Loading %d image pairs from %s\n', num_images, data_path);

% Get image size from first file
first_img = imread(fullfile(data_path, 'B00001_A.tif'));
[H, W] = size(first_img);
fprintf('Image size: %d x %d\n', H, W);
fprintf('First image dtype: %s, range: [%d, %d]\n', class(first_img), min(first_img(:)), max(first_img(:)));

% Preallocate as single (float32)
M_A = zeros(num_images, H * W, 'single');
M_B = zeros(num_images, H * W, 'single');

% Load all images
for i = 1:num_images
    fname_A = fullfile(data_path, sprintf('B%05d_A.tif', i));
    fname_B = fullfile(data_path, sprintf('B%05d_B.tif', i));

    img_A = single(imread(fname_A));
    img_B = single(imread(fname_B));

    M_A(i, :) = reshape(img_A, 1, []);
    M_B(i, :) = reshape(img_B, 1, []);
end
fprintf('Loaded %d pairs\n', num_images);
fprintf('Data range A: [%.2f, %.2f]\n', min(M_A(:)), max(M_A(:)));
fprintf('Data range B: [%.2f, %.2f]\n', min(M_B(:)), max(M_B(:)));

%% ========== FRAME A ==========
fprintf('\n========== Processing Frame A ==========\n');

% Covariance matrix
C_A = M_A * M_A';
fprintf('Covariance matrix: min=%.2e, max=%.2e\n', min(C_A(:)), max(C_A(:)));
fprintf('Covariance has NaN: %d, Inf: %d\n', any(isnan(C_A(:))), any(isinf(C_A(:))));

% SVD
[PSI_A, LAMBDA_A, ~] = svd(C_A);
eigVal_A = diag(LAMBDA_A);
fprintf('Eigenvalues: max=%.2e, min=%.2e\n', max(eigVal_A), min(eigVal_A));
fprintf('PSI has NaN: %d\n', any(isnan(PSI_A(:))));

% Auto mode selection (matching Python logic exactly)
N_auto_A = 0;
norm_factor_A = eigVal_A(round(num_images/2));
fprintf('\nMode selection (norm_factor=%.2e, threshold=%.2e):\n', norm_factor_A, eps_auto_sigma * eigVal_A(1));

for i = 1:(num_images - 1)
    Mean_PSI = abs(mean(PSI_A(:, i)));
    Sig_Diff = abs(eigVal_A(i) - eigVal_A(i + 1)) / norm_factor_A;

    if Mean_PSI < eps_auto_psi && Sig_Diff < eps_auto_sigma * eigVal_A(1)
        N_auto_A = i;
        fprintf('  Mode %2d: Mean_PSI=%.6f, Sig_Diff=%.4e -> NOISE FLOOR DETECTED\n', i, Mean_PSI, Sig_Diff);
        break;
    else
        if i <= 10  % Only print first 10 modes
            fprintf('  Mode %2d: Mean_PSI=%.6f, Sig_Diff=%.4e\n', i, Mean_PSI, Sig_Diff);
        end
    end
end
fprintf('Frame A: Removing %d modes\n', N_auto_A);

%% ========== FRAME B ==========
fprintf('\n========== Processing Frame B ==========\n');

C_B = M_B * M_B';
fprintf('Covariance matrix: min=%.2e, max=%.2e\n', min(C_B(:)), max(C_B(:)));

[PSI_B, LAMBDA_B, ~] = svd(C_B);
eigVal_B = diag(LAMBDA_B);
fprintf('Eigenvalues: max=%.2e, min=%.2e\n', max(eigVal_B), min(eigVal_B));

N_auto_B = 0;
norm_factor_B = eigVal_B(round(num_images/2));
fprintf('\nMode selection (norm_factor=%.2e, threshold=%.2e):\n', norm_factor_B, eps_auto_sigma * eigVal_B(1));

for i = 1:(num_images - 1)
    Mean_PSI = abs(mean(PSI_B(:, i)));
    Sig_Diff = abs(eigVal_B(i) - eigVal_B(i + 1)) / norm_factor_B;

    if Mean_PSI < eps_auto_psi && Sig_Diff < eps_auto_sigma * eigVal_B(1)
        N_auto_B = i;
        fprintf('  Mode %2d: Mean_PSI=%.6f, Sig_Diff=%.4e -> NOISE FLOOR DETECTED\n', i, Mean_PSI, Sig_Diff);
        break;
    else
        if i <= 10
            fprintf('  Mode %2d: Mean_PSI=%.6f, Sig_Diff=%.4e\n', i, Mean_PSI, Sig_Diff);
        end
    end
end
fprintf('Frame B: Removing %d modes\n', N_auto_B);

%% ========== APPLY FILTER ==========
fprintf('\n========== Applying POD Filter ==========\n');

% Frame A - using PHI/TCoeff method
if N_auto_A > 0
    PHI_A = cell(N_auto_A, 1);
    TCoeff_A = cell(N_auto_A, 1);

    for mod_i = 1:N_auto_A
        % PHI = M' * PSI(:,i), then normalize
        PHI_A{mod_i} = M_A' * PSI_A(:, mod_i);
        phi_norm = sqrt(sum(PHI_A{mod_i}.^2));
        PHI_A{mod_i} = PHI_A{mod_i} / phi_norm;

        % TCoeff = M * PHI
        TCoeff_A{mod_i} = M_A * PHI_A{mod_i};
    end

    % Subtract modes from each image
    M_A_filtered = M_A;
    for i = 1:num_images
        for mod_i = 1:N_auto_A
            M_A_filtered(i, :) = M_A_filtered(i, :) - TCoeff_A{mod_i}(i) * PHI_A{mod_i}';
        end
    end
else
    M_A_filtered = M_A;
end

% Frame B - using PHI/TCoeff method
if N_auto_B > 0
    PHI_B = cell(N_auto_B, 1);
    TCoeff_B = cell(N_auto_B, 1);

    for mod_i = 1:N_auto_B
        PHI_B{mod_i} = M_B' * PSI_B(:, mod_i);
        phi_norm = sqrt(sum(PHI_B{mod_i}.^2));
        PHI_B{mod_i} = PHI_B{mod_i} / phi_norm;
        TCoeff_B{mod_i} = M_B * PHI_B{mod_i};
    end

    M_B_filtered = M_B;
    for i = 1:num_images
        for mod_i = 1:N_auto_B
            M_B_filtered(i, :) = M_B_filtered(i, :) - TCoeff_B{mod_i}(i) * PHI_B{mod_i}';
        end
    end
else
    M_B_filtered = M_B;
end

%% ========== SAVE RESULTS ==========
output_path = fullfile(data_path, 'matlab_pod_results');
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

% Save first filtered pair as TIFF for visual comparison
img_A_out = reshape(M_A_filtered(1, :), H, W);
img_B_out = reshape(M_B_filtered(1, :), H, W);

% Clip to valid range and save
imwrite(uint16(max(0, img_A_out)), fullfile(output_path, 'filtered_A_001.tif'));
imwrite(uint16(max(0, img_B_out)), fullfile(output_path, 'filtered_B_001.tif'));

% Save numerical data for precise comparison with Python
save(fullfile(output_path, 'pod_results.mat'), ...
    'N_auto_A', 'N_auto_B', ...
    'eigVal_A', 'eigVal_B', ...
    'M_A_filtered', 'M_B_filtered', ...
    'H', 'W', 'num_images', ...
    'eps_auto_psi', 'eps_auto_sigma');

%% ========== FINAL SUMMARY ==========
fprintf('\n========================================\n');
fprintf('              RESULTS SUMMARY\n');
fprintf('========================================\n');
fprintf('Modes removed:  Frame A = %d, Frame B = %d\n', N_auto_A, N_auto_B);
fprintf('Output range A: [%.2f, %.2f]\n', min(M_A_filtered(:)), max(M_A_filtered(:)));
fprintf('Output range B: [%.2f, %.2f]\n', min(M_B_filtered(:)), max(M_B_filtered(:)));
fprintf('Has NaN A: %d\n', any(isnan(M_A_filtered(:))));
fprintf('Has NaN B: %d\n', any(isnan(M_B_filtered(:))));
fprintf('Has negative A: %d (min=%.2f)\n', any(M_A_filtered(:) < 0), min(M_A_filtered(:)));
fprintf('Has negative B: %d (min=%.2f)\n', any(M_B_filtered(:) < 0), min(M_B_filtered(:)));
fprintf('\nResults saved to: %s\n', output_path);
fprintf('========================================\n');

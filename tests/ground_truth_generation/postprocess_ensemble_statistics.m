%% Post-Processing of JHTDB Particle Position Data
% Computes mean velocities and full Reynolds stress tensor from particle pairs
% Outputs statistics at multiple window sizes for benchmarking
%
% Output files:
%   - ensemble_statistics.mat: Statistics for all window sizes
%   - coordinates.mat: Grid coordinates for each window size
%   - Validation plots: U vs y, U+ vs y+, Reynolds stresses vs y+

clear variables;
close all;

%% ========== CONFIGURATION ==========
% Load parameters from download script
params_file = fullfile('C:\Users\mtt1e23\OneDrive - University of Southampton\Documents\#current_processing\query_JHTDB\download_from_jhtdb\particle_positions', 'download_parameters.mat');
load(params_file, 'params');

% Data location
data_folder = 'C:\Users\mtt1e23\OneDrive - University of Southampton\Documents\#current_processing\query_JHTDB\download_from_jhtdb\particle_positions';
n_frames = params.n_frames;
n_points = params.n_points;
dt = params.dt;

% Physical parameters
h_mm = params.h_mm;              % 150 mm (channel half-height)
Nx = params.Nx;                  % 2048 pixels
Ny = params.Ny;                  % 2048 pixels
mm_per_pixel = params.mm_per_pixel;

% Domain bounds (original)
min_xyz = params.min_xyz;        % [0, -1, -1/256] non-dim
max_xyz = params.max_xyz;        % [1, 0, 1/256] non-dim

% DNS wall-unit parameters (from JHTDB channel flow documentation)
u_tau_nondim = 0.0499;           % Friction velocity (non-dimensional)
nu_nondim = 5e-5;                % Kinematic viscosity (non-dimensional)

% Convert to physical units
velocity_conv = params.velocity_conv;  % 168.75 mm/s per non-dim velocity
length_conv = params.length_conv;      % 150 mm per non-dim length

u_tau = u_tau_nondim * velocity_conv;  % Friction velocity in mm/s
nu = nu_nondim * length_conv * velocity_conv;  % Viscosity in mm²/s
delta_nu = nu / u_tau;                 % Viscous length scale in mm

% Output folder
output_folder = 'ensemble_statistics';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% ========== PRINT CONFIGURATION ==========
fprintf('=== Ensemble Statistics Post-Processing ===\n\n');
fprintf('Input data:\n');
fprintf('  Data folder: %s\n', data_folder);
fprintf('  Number of frames: %d\n', n_frames);
fprintf('  Particles per frame: %d\n', n_points);
fprintf('  Total particles: %d million\n', n_frames * n_points / 1e6);
fprintf('\n');
fprintf('Physical parameters:\n');
fprintf('  Channel half-height h = %.0f mm\n', h_mm);
fprintf('  PIV dt = %.4f ms\n', dt * 1000);
fprintf('  Resolution: %.4f mm/pixel\n', mm_per_pixel);
fprintf('\n');
fprintf('Wall units:\n');
fprintf('  Friction velocity u_tau = %.4f mm/s\n', u_tau);
fprintf('  Kinematic viscosity nu = %.4f mm^2/s\n', nu);
fprintf('  Viscous length scale delta_nu = %.4f mm\n', delta_nu);
fprintf('  Re_tau = h/delta_nu = %.0f\n', h_mm / delta_nu);
fprintf('\n');

%% ========== WINDOW DEFINITIONS ==========
% Window sizes in pixels: 16, 8, 6, 4, 2, 1
window_sizes = [16, 8, 6, 4, 2, 1];
n_window_sizes = length(window_sizes);

% Overlaps: 50% for 16px, 0% for all others
overlaps = [0.5, 0, 0, 0, 0, 0];

fprintf('Window configurations:\n');
for i = 1:n_window_sizes
    spacing = window_sizes(i) * (1 - overlaps(i));
    n_win_x = floor((Nx - window_sizes(i)) / spacing) + 1;
    n_win_y = floor((Ny - window_sizes(i)) / spacing) + 1;
    fprintf('  %2d px: overlap %.0f%%, spacing %d px, grid %d x %d\n', ...
            window_sizes(i), overlaps(i)*100, spacing, n_win_x, n_win_y);
end
fprintf('\n');

%% ========== SCAN FOR AVAILABLE FRAMES ==========
fprintf('Scanning for available frame pairs...\n');

% Scan directory for all B*****_A.data files to find available frames
% This ignores n_frames from params and finds ALL available data
file_list = dir(fullfile(data_folder, 'B*_A.data'));
available_frames = [];

for k = 1:length(file_list)
    % Extract frame number from filename (e.g., B00001_A.data -> 1)
    filename = file_list(k).name;
    frame_num = str2double(filename(2:6));  % Extract 5-digit number after 'B'

    % Check if corresponding _B file exists
    file_B = fullfile(data_folder, ['B', sprintf('%05d', frame_num), '_B.data']);
    if isfile(file_B)
        available_frames = [available_frames, frame_num];
    end
end

% Sort frame numbers
available_frames = sort(available_frames);
n_available = length(available_frames);

fprintf('  Found %d complete frame pairs\n', n_available);
if n_available > 0
    fprintf('  Frame range: %d to %d\n', min(available_frames), max(available_frames));
end

if n_available == 0
    error('No complete frame pairs found in %s', data_folder);
end

% Check for gaps in sequence
expected_frames = min(available_frames):max(available_frames);
missing_frames = setdiff(expected_frames, available_frames);
n_missing = length(missing_frames);

if n_missing > 0
    fprintf('  Missing %d frames in sequence: ', n_missing);
    if n_missing <= 20
        fprintf('%d ', missing_frames);
    else
        fprintf('%d %d %d ... %d %d %d', missing_frames(1), missing_frames(2), missing_frames(3), ...
                missing_frames(end-2), missing_frames(end-1), missing_frames(end));
    end
    fprintf('\n');
end
fprintf('\n');

%% ========== LOAD ALL PARTICLE DATA ==========
fprintf('Loading %d frame pairs into memory...\n', n_available);
tic;

% Estimate memory requirement
mem_per_particle = 5 * 4;  % 5 floats (x, y, u, v, w) × 4 bytes
total_particles = n_available * n_points;
mem_gb = total_particles * mem_per_particle / 1e9;
fprintf('  Estimated memory: %.2f GB\n', mem_gb);

% Preallocate arrays (single precision to save memory)
all_x = zeros(total_particles, 1, 'single');
all_y = zeros(total_particles, 1, 'single');
all_u = zeros(total_particles, 1, 'single');
all_v = zeros(total_particles, 1, 'single');
all_w = zeros(total_particles, 1, 'single');

% Progress tracking
load_start = tic;
progress_interval = max(1, floor(n_available / 20));  % Update ~20 times

for i = 1:n_available
    frame_idx = available_frames(i);
    frame_str = sprintf('%05d', frame_idx);

    % Load position files
    file_A = fullfile(data_folder, ['B', frame_str, '_A.data']);
    file_B = fullfile(data_folder, ['B', frame_str, '_B.data']);

    pos_A = readmatrix(file_A, 'FileType', 'text');
    pos_B = readmatrix(file_B, 'FileType', 'text');

    % Compute velocities from displacement: v = (pos_B - pos_A) / dt
    velocity = (pos_B - pos_A) / dt;  % mm/s

    % Mid-point positions (where velocity is evaluated)
    pos_mid = (pos_A + pos_B) / 2;

    % Transform y coordinate: from [-150, 0] to [0, 150]
    % Wall at y=0, centerline at y=150
    pos_mid(:, 2) = pos_mid(:, 2) + h_mm;

    % Store data (using sequential index i, not frame_idx)
    idx_start = (i - 1) * n_points + 1;
    idx_end = i * n_points;

    all_x(idx_start:idx_end) = single(pos_mid(:, 1));
    all_y(idx_start:idx_end) = single(pos_mid(:, 2));
    all_u(idx_start:idx_end) = single(velocity(:, 1));
    all_v(idx_start:idx_end) = single(velocity(:, 2));
    all_w(idx_start:idx_end) = single(velocity(:, 3));

    % Progress indicator
    if mod(i, progress_interval) == 0 || i == n_available
        elapsed = toc(load_start);
        rate = i / elapsed;
        remaining = (n_available - i) / rate;
        fprintf('  Loading: %d/%d (%.0f%%) - %.1f frames/s - ETA: %.0fs\n', ...
                i, n_available, 100*i/n_available, rate, remaining);
    end
end

load_time = toc(load_start);
fprintf('Data loaded in %.1f seconds (%.1f min)\n', load_time, load_time/60);
fprintf('Total particles: %.2f million\n\n', total_particles / 1e6);

%% ========== CONVERT POSITIONS TO PIXEL COORDINATES ==========
% Positions are in mm, domain is [0, h_mm] x [0, h_mm] in (x, y)
% Map to [0, Nx] x [0, Ny] in pixels

x_pixels = all_x / mm_per_pixel;
y_pixels = all_y / mm_per_pixel;

% Verify ranges
fprintf('Position ranges (pixels):\n');
fprintf('  x: [%.1f, %.1f] (expected [0, %d])\n', min(x_pixels), max(x_pixels), Nx);
fprintf('  y: [%.1f, %.1f] (expected [0, %d])\n', min(y_pixels), max(y_pixels), Ny);
fprintf('\n');

%% ========== COMPUTE STATISTICS FOR EACH WINDOW SIZE ==========
% Initialize output structures
ensemble_stats = struct();
coordinates = struct();

for win_idx = 1:n_window_sizes
    W = window_sizes(win_idx);
    overlap = overlaps(win_idx);
    spacing = W * (1 - overlap);

    fprintf('Processing %d px window (%.0f%% overlap)...\n', W, overlap*100);
    tic;

    % Compute window centers (zero-indexed, in pixels)
    % First center at (W-1)/2, then every 'spacing' pixels
    % e.g., W=16, overlap=0.5: centers at 7.5, 15.5, 23.5, ...
    first_center = (W - 1) / 2;
    last_center = Nx - 1 - (W - 1) / 2;

    win_ctrs_x = first_center : spacing : last_center;
    win_ctrs_y = first_center : spacing : last_center;

    n_win_x = length(win_ctrs_x);
    n_win_y = length(win_ctrs_y);

    fprintf('  Grid: %d x %d windows\n', n_win_x, n_win_y);

    % Initialize accumulator grids
    % Using Welford's online algorithm would be better for numerical stability,
    % but for simplicity we use sum accumulators: <u'v'> = <uv> - <u><v>
    count = zeros(n_win_x, n_win_y, 'single');
    sum_u = zeros(n_win_x, n_win_y, 'single');
    sum_v = zeros(n_win_x, n_win_y, 'single');
    sum_w = zeros(n_win_x, n_win_y, 'single');
    sum_uu = zeros(n_win_x, n_win_y, 'single');
    sum_vv = zeros(n_win_x, n_win_y, 'single');
    sum_ww = zeros(n_win_x, n_win_y, 'single');
    sum_uv = zeros(n_win_x, n_win_y, 'single');
    sum_uw = zeros(n_win_x, n_win_y, 'single');
    sum_vw = zeros(n_win_x, n_win_y, 'single');

    % Bin each particle into its window
    % Window boundaries: [center - W/2, center + W/2)
    half_W = W / 2;

    % Compute bin indices for each particle
    % For x: find which window center this particle belongs to
    % bin_idx = floor((x - first_center + spacing/2) / spacing) + 1
    % But we need to handle edge cases

    % Alternative: for each particle, compute which bin it's in
    % This is vectorized for speed

    % Relative position from first window edge
    x_rel = x_pixels - (first_center - half_W);
    y_rel = y_pixels - (first_center - half_W);

    % Bin indices (1-based)
    bin_x = floor(x_rel / spacing) + 1;
    bin_y = floor(y_rel / spacing) + 1;

    % Check if particle is within a valid window
    % A particle at position p is in window i if:
    % win_ctrs(i) - half_W <= p < win_ctrs(i) + half_W

    % For overlapping windows, a particle can be in multiple windows
    % For simplicity with 50% overlap, we assign to the nearest center
    % This is an approximation but works well for statistics

    % Clamp to valid range
    valid = (bin_x >= 1) & (bin_x <= n_win_x) & ...
            (bin_y >= 1) & (bin_y <= n_win_y);

    % For overlapping windows, use fast pixel-level binning + convolution approach
    if overlap > 0
        fprintf('    Using fast convolution method for overlapping windows...\n');

        % Step 1: Bin all particles into 1-pixel bins (full resolution)
        % Floor to get pixel indices (0-based, then +1 for MATLAB)
        px_x = floor(x_pixels);
        px_y = floor(y_pixels);

        % Clamp to valid pixel range [0, Nx-1] -> [1, Nx] for MATLAB indexing
        px_x = max(0, min(Nx-1, px_x)) + 1;
        px_y = max(0, min(Ny-1, px_y)) + 1;

        % Create 1-pixel resolution accumulator grids
        pixel_count = zeros(Nx, Ny);
        pixel_sum_u = zeros(Nx, Ny);
        pixel_sum_v = zeros(Nx, Ny);
        pixel_sum_w = zeros(Nx, Ny);
        pixel_sum_uu = zeros(Nx, Ny);
        pixel_sum_vv = zeros(Nx, Ny);
        pixel_sum_ww = zeros(Nx, Ny);
        pixel_sum_uv = zeros(Nx, Ny);
        pixel_sum_uw = zeros(Nx, Ny);
        pixel_sum_vw = zeros(Nx, Ny);

        % Accumulate into pixel bins using linear indexing
        linear_px = sub2ind([Nx, Ny], px_x, px_y);
        pixel_count = reshape(accumarray(linear_px, 1, [Nx*Ny, 1]), Nx, Ny);
        pixel_sum_u = reshape(accumarray(linear_px, double(all_u), [Nx*Ny, 1]), Nx, Ny);
        pixel_sum_v = reshape(accumarray(linear_px, double(all_v), [Nx*Ny, 1]), Nx, Ny);
        pixel_sum_w = reshape(accumarray(linear_px, double(all_w), [Nx*Ny, 1]), Nx, Ny);
        pixel_sum_uu = reshape(accumarray(linear_px, double(all_u).^2, [Nx*Ny, 1]), Nx, Ny);
        pixel_sum_vv = reshape(accumarray(linear_px, double(all_v).^2, [Nx*Ny, 1]), Nx, Ny);
        pixel_sum_ww = reshape(accumarray(linear_px, double(all_w).^2, [Nx*Ny, 1]), Nx, Ny);
        pixel_sum_uv = reshape(accumarray(linear_px, double(all_u).*double(all_v), [Nx*Ny, 1]), Nx, Ny);
        pixel_sum_uw = reshape(accumarray(linear_px, double(all_u).*double(all_w), [Nx*Ny, 1]), Nx, Ny);
        pixel_sum_vw = reshape(accumarray(linear_px, double(all_v).*double(all_w), [Nx*Ny, 1]), Nx, Ny);

        fprintf('    Pixel binning complete, applying window convolution...\n');

        % Step 2: Use convolution with a box kernel to sum over WxW windows
        % The kernel is just ones(W, W)
        box_kernel = ones(W, W);

        % Convolve to get sums over each WxW region
        count_conv = conv2(pixel_count, box_kernel, 'valid');
        sum_u_conv = conv2(pixel_sum_u, box_kernel, 'valid');
        sum_v_conv = conv2(pixel_sum_v, box_kernel, 'valid');
        sum_w_conv = conv2(pixel_sum_w, box_kernel, 'valid');
        sum_uu_conv = conv2(pixel_sum_uu, box_kernel, 'valid');
        sum_vv_conv = conv2(pixel_sum_vv, box_kernel, 'valid');
        sum_ww_conv = conv2(pixel_sum_ww, box_kernel, 'valid');
        sum_uv_conv = conv2(pixel_sum_uv, box_kernel, 'valid');
        sum_uw_conv = conv2(pixel_sum_uw, box_kernel, 'valid');
        sum_vw_conv = conv2(pixel_sum_vw, box_kernel, 'valid');

        % Step 3: Sample at the window center locations (with spacing)
        % conv2 'valid' output has size (Nx-W+1, Ny-W+1)
        % Window centers in the convolved output correspond to indices:
        % Original center at (W-1)/2 + 0.5 maps to conv output index 1
        % Spacing in original pixels = spacing in conv output indices

        % Generate indices into convolved output
        % First window center at (W-1)/2, which is index 1 in conv output (0-based: 0)
        % With 50% overlap, spacing = W/2, so indices are 1, 1+W/2, 1+W, ...
        sample_indices_x = 1 : spacing : size(count_conv, 1);
        sample_indices_y = 1 : spacing : size(count_conv, 2);

        % Extract at sample points
        count = count_conv(sample_indices_x, sample_indices_y);
        sum_u = sum_u_conv(sample_indices_x, sample_indices_y);
        sum_v = sum_v_conv(sample_indices_x, sample_indices_y);
        sum_w = sum_w_conv(sample_indices_x, sample_indices_y);
        sum_uu = sum_uu_conv(sample_indices_x, sample_indices_y);
        sum_vv = sum_vv_conv(sample_indices_x, sample_indices_y);
        sum_ww = sum_ww_conv(sample_indices_x, sample_indices_y);
        sum_uv = sum_uv_conv(sample_indices_x, sample_indices_y);
        sum_uw = sum_uw_conv(sample_indices_x, sample_indices_y);
        sum_vw = sum_vw_conv(sample_indices_x, sample_indices_y);

        % Update grid size to match sampled output
        n_win_x = length(sample_indices_x);
        n_win_y = length(sample_indices_y);

        % Recompute window centers to match
        win_ctrs_x = first_center : spacing : (first_center + (n_win_x-1)*spacing);
        win_ctrs_y = first_center : spacing : (first_center + (n_win_y-1)*spacing);

        fprintf('    Convolution complete. Grid: %d x %d\n', n_win_x, n_win_y);
    else
        % No overlap: use fast vectorized accumulation
        bin_x_valid = bin_x(valid);
        bin_y_valid = bin_y(valid);
        u_valid = all_u(valid);
        v_valid = all_v(valid);
        w_valid = all_w(valid);

        % Use accumarray for fast binning
        linear_idx = sub2ind([n_win_x, n_win_y], bin_x_valid, bin_y_valid);

        count = reshape(accumarray(linear_idx, 1, [n_win_x * n_win_y, 1]), n_win_x, n_win_y);
        sum_u = reshape(accumarray(linear_idx, double(u_valid), [n_win_x * n_win_y, 1]), n_win_x, n_win_y);
        sum_v = reshape(accumarray(linear_idx, double(v_valid), [n_win_x * n_win_y, 1]), n_win_x, n_win_y);
        sum_w = reshape(accumarray(linear_idx, double(w_valid), [n_win_x * n_win_y, 1]), n_win_x, n_win_y);
        sum_uu = reshape(accumarray(linear_idx, double(u_valid.^2), [n_win_x * n_win_y, 1]), n_win_x, n_win_y);
        sum_vv = reshape(accumarray(linear_idx, double(v_valid.^2), [n_win_x * n_win_y, 1]), n_win_x, n_win_y);
        sum_ww = reshape(accumarray(linear_idx, double(w_valid.^2), [n_win_x * n_win_y, 1]), n_win_x, n_win_y);
        sum_uv = reshape(accumarray(linear_idx, double(u_valid .* v_valid), [n_win_x * n_win_y, 1]), n_win_x, n_win_y);
        sum_uw = reshape(accumarray(linear_idx, double(u_valid .* w_valid), [n_win_x * n_win_y, 1]), n_win_x, n_win_y);
        sum_vw = reshape(accumarray(linear_idx, double(v_valid .* w_valid), [n_win_x * n_win_y, 1]), n_win_x, n_win_y);
    end

    % Compute statistics
    % Avoid division by zero
    count(count == 0) = NaN;

    % Mean velocities [mm/s]
    U_mean = sum_u ./ count;
    V_mean = sum_v ./ count;
    W_mean = sum_w ./ count;

    % Reynolds stresses [mm²/s²]: <u'u'> = <uu> - <u>²
    UU_stress = sum_uu ./ count - U_mean.^2;
    VV_stress = sum_vv ./ count - V_mean.^2;
    WW_stress = sum_ww ./ count - W_mean.^2;
    UV_stress = sum_uv ./ count - U_mean .* V_mean;
    UW_stress = sum_uw ./ count - U_mean .* W_mean;
    VW_stress = sum_vw ./ count - V_mean .* W_mean;

    % Restore count for output
    count(isnan(count)) = 0;

    % Store in output structure
    ensemble_stats(win_idx).window_size_px = W;
    ensemble_stats(win_idx).window_size_mm = W * mm_per_pixel;
    ensemble_stats(win_idx).overlap = overlap;
    ensemble_stats(win_idx).n_windows = [n_win_x, n_win_y];

    % ===== 2D SPATIAL FIELDS =====
    % Mean velocities
    ensemble_stats(win_idx).U_mean = single(U_mean);   % [mm/s]
    ensemble_stats(win_idx).V_mean = single(V_mean);   % [mm/s]
    ensemble_stats(win_idx).W_mean = single(W_mean);   % [mm/s]

    % Reynolds stress tensor (symmetric, 6 unique components)
    ensemble_stats(win_idx).UU_stress = single(UU_stress);  % [mm²/s²]
    ensemble_stats(win_idx).VV_stress = single(VV_stress);  % [mm²/s²]
    ensemble_stats(win_idx).WW_stress = single(WW_stress);  % [mm²/s²]
    ensemble_stats(win_idx).UV_stress = single(UV_stress);  % [mm²/s²]
    ensemble_stats(win_idx).UW_stress = single(UW_stress);  % [mm²/s²]
    ensemble_stats(win_idx).VW_stress = single(VW_stress);  % [mm²/s²]

    % Particle counts
    ensemble_stats(win_idx).count = single(count);
    ensemble_stats(win_idx).total_particles = sum(count(:), 'omitnan');

    % ===== 1D PROFILES (spatially averaged over x) =====
    % These are the benchmark profiles - averaged over all x columns
    ensemble_stats(win_idx).y_mm = single(win_ctrs_y * mm_per_pixel);  % [mm]
    ensemble_stats(win_idx).y_plus = single(win_ctrs_y * mm_per_pixel / delta_nu);  % [wall units]

    % Mean velocity profiles (averaged over x)
    ensemble_stats(win_idx).U_profile = single(mean(U_mean, 1, 'omitnan'));  % [mm/s]
    ensemble_stats(win_idx).V_profile = single(mean(V_mean, 1, 'omitnan'));  % [mm/s]
    ensemble_stats(win_idx).W_profile = single(mean(W_mean, 1, 'omitnan'));  % [mm/s]

    % Reynolds stress profiles (averaged over x)
    ensemble_stats(win_idx).UU_profile = single(mean(UU_stress, 1, 'omitnan'));  % [mm²/s²]
    ensemble_stats(win_idx).VV_profile = single(mean(VV_stress, 1, 'omitnan'));  % [mm²/s²]
    ensemble_stats(win_idx).WW_profile = single(mean(WW_stress, 1, 'omitnan'));  % [mm²/s²]
    ensemble_stats(win_idx).UV_profile = single(mean(UV_stress, 1, 'omitnan'));  % [mm²/s²]
    ensemble_stats(win_idx).UW_profile = single(mean(UW_stress, 1, 'omitnan'));  % [mm²/s²]
    ensemble_stats(win_idx).VW_profile = single(mean(VW_stress, 1, 'omitnan'));  % [mm²/s²]

    % Profiles in wall units (for convenience)
    ensemble_stats(win_idx).U_plus_profile = single(mean(U_mean, 1, 'omitnan') / u_tau);
    ensemble_stats(win_idx).UU_plus_profile = single(mean(UU_stress, 1, 'omitnan') / u_tau^2);
    ensemble_stats(win_idx).VV_plus_profile = single(mean(VV_stress, 1, 'omitnan') / u_tau^2);
    ensemble_stats(win_idx).WW_plus_profile = single(mean(WW_stress, 1, 'omitnan') / u_tau^2);
    ensemble_stats(win_idx).UV_plus_profile = single(mean(UV_stress, 1, 'omitnan') / u_tau^2);

    % Particle count profile (sum over x gives total particles at each y)
    ensemble_stats(win_idx).count_profile = single(sum(count, 1, 'omitnan'));

    % Coordinates (zero-indexed, in pixels and mm)
    % Create 2D grids
    [X_px, Y_px] = ndgrid(win_ctrs_x, win_ctrs_y);
    X_mm = X_px * mm_per_pixel;
    Y_mm = Y_px * mm_per_pixel;

    % Wall units
    Y_plus = Y_mm / delta_nu;

    coordinates(win_idx).window_size_px = W;
    coordinates(win_idx).window_size_mm = W * mm_per_pixel;
    coordinates(win_idx).overlap = overlap;
    coordinates(win_idx).n_windows = [n_win_x, n_win_y];

    % 1D center arrays (zero-indexed pixels)
    coordinates(win_idx).win_ctrs_x_px = single(win_ctrs_x);  % [pixels]
    coordinates(win_idx).win_ctrs_y_px = single(win_ctrs_y);  % [pixels]

    % 1D center arrays (mm)
    coordinates(win_idx).win_ctrs_x_mm = single(win_ctrs_x * mm_per_pixel);  % [mm]
    coordinates(win_idx).win_ctrs_y_mm = single(win_ctrs_y * mm_per_pixel);  % [mm]

    % 2D grids
    coordinates(win_idx).X_px = single(X_px);    % [pixels]
    coordinates(win_idx).Y_px = single(Y_px);    % [pixels]
    coordinates(win_idx).X_mm = single(X_mm);    % [mm]
    coordinates(win_idx).Y_mm = single(Y_mm);    % [mm]
    coordinates(win_idx).Y_plus = single(Y_plus); % [wall units]

    proc_time = toc;
    fprintf('  Completed in %.1f s, %.0f particles binned\n', proc_time, ensemble_stats(win_idx).total_particles);
end

fprintf('\nStatistics computation complete.\n\n');

%% ========== SAVE RESULTS ==========
% Save ensemble statistics
stats_file = fullfile(output_folder, 'ensemble_statistics.mat');
save(stats_file, 'ensemble_stats', 'u_tau', 'nu', 'delta_nu', 'h_mm', 'mm_per_pixel', '-v7.3');
fprintf('Saved: %s\n', stats_file);

% Save coordinates
coords_file = fullfile(output_folder, 'coordinates.mat');
save(coords_file, 'coordinates', '-v7.3');
fprintf('Saved: %s\n', coords_file);

% Save wall unit parameters separately for convenience
wall_units = struct();
wall_units.u_tau = u_tau;           % [mm/s]
wall_units.nu = nu;                  % [mm²/s²]
wall_units.delta_nu = delta_nu;      % [mm]
wall_units.h_mm = h_mm;              % [mm]
wall_units.Re_tau = h_mm / delta_nu;
wall_units_file = fullfile(output_folder, 'wall_units.mat');
save(wall_units_file, 'wall_units');
fprintf('Saved: %s\n', wall_units_file);

% Save 1D profiles separately for easy benchmarking
profiles = struct();
for win_idx = 1:n_window_sizes
    W = window_sizes(win_idx);
    field_name = sprintf('win_%dpx', W);

    profiles.(field_name).window_size_px = W;
    profiles.(field_name).window_size_mm = ensemble_stats(win_idx).window_size_mm;
    profiles.(field_name).overlap = ensemble_stats(win_idx).overlap;

    % Coordinates
    profiles.(field_name).y_mm = ensemble_stats(win_idx).y_mm;
    profiles.(field_name).y_plus = ensemble_stats(win_idx).y_plus;

    % Mean velocities (dimensional)
    profiles.(field_name).U = ensemble_stats(win_idx).U_profile;  % [mm/s]
    profiles.(field_name).V = ensemble_stats(win_idx).V_profile;  % [mm/s]
    profiles.(field_name).W = ensemble_stats(win_idx).W_profile;  % [mm/s]

    % Reynolds stresses (dimensional)
    profiles.(field_name).uu = ensemble_stats(win_idx).UU_profile;  % [mm²/s²]
    profiles.(field_name).vv = ensemble_stats(win_idx).VV_profile;  % [mm²/s²]
    profiles.(field_name).ww = ensemble_stats(win_idx).WW_profile;  % [mm²/s²]
    profiles.(field_name).uv = ensemble_stats(win_idx).UV_profile;  % [mm²/s²]
    profiles.(field_name).uw = ensemble_stats(win_idx).UW_profile;  % [mm²/s²]
    profiles.(field_name).vw = ensemble_stats(win_idx).VW_profile;  % [mm²/s²]

    % Wall units
    profiles.(field_name).U_plus = ensemble_stats(win_idx).U_plus_profile;
    profiles.(field_name).uu_plus = ensemble_stats(win_idx).UU_plus_profile;
    profiles.(field_name).vv_plus = ensemble_stats(win_idx).VV_plus_profile;
    profiles.(field_name).ww_plus = ensemble_stats(win_idx).WW_plus_profile;
    profiles.(field_name).uv_plus = ensemble_stats(win_idx).UV_plus_profile;

    % Particle count at each y
    profiles.(field_name).count = ensemble_stats(win_idx).count_profile;
end
profiles.wall_units = wall_units;
profiles_file = fullfile(output_folder, 'profiles.mat');
save(profiles_file, 'profiles');
fprintf('Saved: %s\n\n', profiles_file);

%% ========== GENERATE VALIDATION PLOTS ==========
fprintf('Generating validation plots...\n');

% Color scheme for different window sizes
colors = lines(n_window_sizes);

%% --- Plot 1: Mean streamwise velocity U vs y ---
figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
hold on;
for win_idx = 1:n_window_sizes
    W = window_sizes(win_idx);

    % Use pre-computed profiles (spatially averaged over x)
    U_profile = ensemble_stats(win_idx).U_profile;
    y_mm = ensemble_stats(win_idx).y_mm;

    plot(U_profile, y_mm, '-', 'LineWidth', 1.5, 'Color', colors(win_idx, :), ...
         'DisplayName', sprintf('%d px', W));
end
hold off;
xlabel('U [mm/s]', 'FontSize', 12);
ylabel('y [mm]', 'FontSize', 12);
title('Mean Streamwise Velocity Profile', 'FontSize', 14);
legend('Location', 'southeast', 'FontSize', 10);
grid on;
xlim([0, inf]);
ylim([0, h_mm]);

subplot(1, 2, 2);
hold on;
for win_idx = 1:n_window_sizes
    W = window_sizes(win_idx);

    % Use pre-computed wall unit profiles
    U_plus = ensemble_stats(win_idx).U_plus_profile;
    y_plus = ensemble_stats(win_idx).y_plus;

    plot(y_plus, U_plus, '-', 'LineWidth', 1.5, 'Color', colors(win_idx, :), ...
         'DisplayName', sprintf('%d px', W));
end

% Add reference lines
y_plus_ref = logspace(0, 3, 100);
% Viscous sublayer: U+ = y+
plot(y_plus_ref(y_plus_ref < 5), y_plus_ref(y_plus_ref < 5), 'k--', 'LineWidth', 1, ...
     'DisplayName', 'U^+ = y^+');
% Log law: U+ = (1/kappa)*ln(y+) + B, kappa=0.41, B=5.2
kappa = 0.41;
B = 5.2;
y_log = y_plus_ref(y_plus_ref > 30 & y_plus_ref < 500);
plot(y_log, (1/kappa)*log(y_log) + B, 'k-.', 'LineWidth', 1, ...
     'DisplayName', sprintf('U^+ = %.2f ln(y^+) + %.1f', 1/kappa, B));

hold off;
xlabel('y^+', 'FontSize', 12);
ylabel('U^+', 'FontSize', 12);
title('Mean Velocity in Wall Units', 'FontSize', 14);
legend('Location', 'northwest', 'FontSize', 9);
grid on;
set(gca, 'XScale', 'log');
xlim([1, 1000]);
ylim([0, 25]);

sgtitle('Streamwise Velocity Profiles', 'FontSize', 16);
saveas(gcf, fullfile(output_folder, 'velocity_profiles.png'));
saveas(gcf, fullfile(output_folder, 'velocity_profiles.fig'));
fprintf('  Saved velocity_profiles.png/fig\n');

%% --- Plot 2: Reynolds normal stresses vs y+ ---
figure('Position', [100, 100, 1600, 500]);

stress_names = {'u''u''', 'v''v''', 'w''w'''};
stress_profile_fields = {'UU_plus_profile', 'VV_plus_profile', 'WW_plus_profile'};

for s = 1:3
    subplot(1, 3, s);
    hold on;

    for win_idx = 1:n_window_sizes
        W = window_sizes(win_idx);

        % Use pre-computed wall unit profiles
        stress_plus = ensemble_stats(win_idx).(stress_profile_fields{s});
        y_plus = ensemble_stats(win_idx).y_plus;

        plot(y_plus, stress_plus, '-', 'LineWidth', 1.5, 'Color', colors(win_idx, :), ...
             'DisplayName', sprintf('%d px', W));
    end

    hold off;
    xlabel('y^+', 'FontSize', 12);
    ylabel(sprintf('<%s>^+', stress_names{s}), 'FontSize', 12);
    title(sprintf('<%s> / u_\\tau^2', stress_names{s}), 'FontSize', 14);
    legend('Location', 'northeast', 'FontSize', 9);
    grid on;
    set(gca, 'XScale', 'log');
    xlim([1, 1000]);
    ylim([0, inf]);
end

sgtitle('Reynolds Normal Stresses in Wall Units', 'FontSize', 16);
saveas(gcf, fullfile(output_folder, 'reynolds_normal_stresses.png'));
saveas(gcf, fullfile(output_folder, 'reynolds_normal_stresses.fig'));
fprintf('  Saved reynolds_normal_stresses.png/fig\n');

%% --- Plot 3: Reynolds shear stress -<u'v'> vs y+ ---
figure('Position', [100, 100, 800, 600]);

hold on;
for win_idx = 1:n_window_sizes
    W = window_sizes(win_idx);

    % Use pre-computed profile (note: UV_plus_profile stores <u'v'>, we want -<u'v'>)
    UV_plus = -ensemble_stats(win_idx).UV_plus_profile;
    y_plus = ensemble_stats(win_idx).y_plus;

    plot(y_plus, UV_plus, '-', 'LineWidth', 1.5, 'Color', colors(win_idx, :), ...
         'DisplayName', sprintf('%d px', W));
end

% Add theoretical line: -<u'v'>+ = 1 - y/h for channel flow (approximate)
y_plus_theory = linspace(0, 1000, 100);
y_over_h = y_plus_theory * delta_nu / h_mm;
UV_plus_theory = 1 - y_over_h;
UV_plus_theory(UV_plus_theory < 0) = 0;
plot(y_plus_theory, UV_plus_theory, 'k--', 'LineWidth', 1, ...
     'DisplayName', '-<u''v''>^+ = 1 - y/h');

hold off;
xlabel('y^+', 'FontSize', 12);
ylabel('-<u''v''>^+', 'FontSize', 12);
title('Reynolds Shear Stress in Wall Units', 'FontSize', 14);
legend('Location', 'northeast', 'FontSize', 10);
grid on;
set(gca, 'XScale', 'log');
xlim([1, 1000]);
ylim([0, 1.2]);

saveas(gcf, fullfile(output_folder, 'reynolds_shear_stress.png'));
saveas(gcf, fullfile(output_folder, 'reynolds_shear_stress.fig'));
fprintf('  Saved reynolds_shear_stress.png/fig\n');

%% --- Plot 4: Turbulent kinetic energy vs y+ ---
figure('Position', [100, 100, 800, 600]);

hold on;
for win_idx = 1:n_window_sizes
    W = window_sizes(win_idx);

    % TKE+ = 0.5 * (<u'u'>+ + <v'v'>+ + <w'w'>+)
    TKE_plus = 0.5 * (ensemble_stats(win_idx).UU_plus_profile + ...
                      ensemble_stats(win_idx).VV_plus_profile + ...
                      ensemble_stats(win_idx).WW_plus_profile);
    y_plus = ensemble_stats(win_idx).y_plus;

    plot(y_plus, TKE_plus, '-', 'LineWidth', 1.5, 'Color', colors(win_idx, :), ...
         'DisplayName', sprintf('%d px', W));
end

hold off;
xlabel('y^+', 'FontSize', 12);
ylabel('k^+ = 0.5(<u''u''> + <v''v''> + <w''w''>)/u_\tau^2', 'FontSize', 12);
title('Turbulent Kinetic Energy in Wall Units', 'FontSize', 14);
legend('Location', 'northeast', 'FontSize', 10);
grid on;
set(gca, 'XScale', 'log');
xlim([1, 1000]);
ylim([0, inf]);

saveas(gcf, fullfile(output_folder, 'turbulent_kinetic_energy.png'));
saveas(gcf, fullfile(output_folder, 'turbulent_kinetic_energy.fig'));
fprintf('  Saved turbulent_kinetic_energy.png/fig\n');

%% --- Plot 5: Particle count distribution ---
figure('Position', [100, 100, 1200, 400]);

for win_idx = 1:min(3, n_window_sizes)  % Show first 3 window sizes
    W = window_sizes(win_idx);

    subplot(1, 3, win_idx);

    count_data = ensemble_stats(win_idx).count;
    imagesc(coordinates(win_idx).win_ctrs_x_mm, coordinates(win_idx).win_ctrs_y_mm, count_data');
    colorbar;
    axis xy;
    xlabel('x [mm]', 'FontSize', 12);
    ylabel('y [mm]', 'FontSize', 12);
    title(sprintf('%d px window: particle count', W), 'FontSize', 12);
end

sgtitle('Particle Count Distribution', 'FontSize', 14);
saveas(gcf, fullfile(output_folder, 'particle_count_distribution.png'));
fprintf('  Saved particle_count_distribution.png\n');

%% ========== SUMMARY ==========
fprintf('\n=== Processing Complete ===\n\n');
fprintf('Output folder: %s\n', fullfile(pwd, output_folder));
fprintf('\nFiles created:\n');
fprintf('  - ensemble_statistics.mat  (2D spatial fields + 1D profiles)\n');
fprintf('  - coordinates.mat          (grid coordinates for each window size)\n');
fprintf('  - profiles.mat             (1D benchmark profiles, spatially averaged)\n');
fprintf('  - wall_units.mat           (u_tau, nu, delta_nu, Re_tau)\n');
fprintf('  - velocity_profiles.png/fig\n');
fprintf('  - reynolds_normal_stresses.png/fig\n');
fprintf('  - reynolds_shear_stress.png/fig\n');
fprintf('  - turbulent_kinetic_energy.png/fig\n');
fprintf('  - particle_count_distribution.png\n');
fprintf('\nWall unit parameters:\n');
fprintf('  u_tau = %.4f mm/s\n', u_tau);
fprintf('  delta_nu = %.4f mm\n', delta_nu);
fprintf('  Re_tau = %.0f\n', h_mm / delta_nu);
fprintf('\nStatistics computed for window sizes: ');
fprintf('%d ', window_sizes);
fprintf('pixels\n');
fprintf('\nProfile data structure (profiles.mat):\n');
fprintf('  profiles.win_16px, profiles.win_8px, etc.\n');
fprintf('  Each contains: y_mm, y_plus, U, V, W, uu, vv, ww, uv, uw, vw\n');
fprintf('  Plus wall unit versions: U_plus, uu_plus, vv_plus, ww_plus, uv_plus\n');

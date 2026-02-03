%% Post-Processing of JHTDB Particle Position Data - DNS Ground Truth
% MEMORY-EFFICIENT VERSION
%
% Computes mean velocities and full Reynolds stress tensor from particle pairs
% using the Two-Pass Global Method for accurate DNS ground truth statistics.
%
% KEY METHODOLOGY:
%   - Pass 1: Compute global mean velocity profile U_ref(y) using ALL particles
%   - Pass 2: Compute fluctuations u' = u - U_ref(y), then Reynolds stresses
%   - This exploits streamwise homogeneity of fully developed channel flow
%   - Avoids numerical catastrophic cancellation in variance computation
%   - Preserves large-scale turbulent structures (no high-pass filtering)
%
% MEMORY OPTIMIZATION:
%   - Pass 2 processes data in serial chunks to avoid parfor broadcast overhead
%   - Only Pass 3 (window convolution) uses parfor
%
% Output files:
%   - ensemble_statistics.mat: Statistics for all window sizes
%   - coordinates.mat: Grid coordinates for each window size
%   - profiles.mat: 1D benchmark profiles
%   - wall_units.mat: Wall unit parameters
%   - Validation plots: U vs y, U+ vs y+, Reynolds stresses vs y+

clear variables;
close all;

%% ========== CONFIGURATION ==========
% Load parameters from download script
params_file = fullfile('C:\Users\mtt1e23\OneDrive - University of Southampton\Documents\#current_processing\query_JHTDB\download_from_jhtdb\bottom_channel\particle_positions', 'download_parameters.mat');
load(params_file, 'params');

% Data location
data_folder = 'C:\Users\mtt1e23\OneDrive - University of Southampton\Documents\#current_processing\query_JHTDB\download_from_jhtdb\bottom_channel\particle_positions_reduced';
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
output_folder = 'ensemble_statistics_reduced';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% ========== PRINT CONFIGURATION ==========
fprintf('=== DNS Ground Truth Statistics Post-Processing ===\n');
fprintf('=== MEMORY-EFFICIENT VERSION ===\n\n');
fprintf('Method: Two-Pass Global Mean (Numerically Stable)\n\n');
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

% Load all data into cell arrays first (simpler, handles variable sizes)
all_x_cell = cell(n_available, 1);
all_y_cell = cell(n_available, 1);
all_u_cell = cell(n_available, 1);
all_v_cell = cell(n_available, 1);
all_w_cell = cell(n_available, 1);

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

    % Store in cell arrays
    all_x_cell{i} = single(pos_mid(:, 1));
    all_y_cell{i} = single(pos_mid(:, 2));
    all_u_cell{i} = single(velocity(:, 1));
    all_v_cell{i} = single(velocity(:, 2));
    all_w_cell{i} = single(velocity(:, 3));

    % Progress indicator
    if mod(i, progress_interval) == 0 || i == n_available
        elapsed = toc(load_start);
        rate = i / elapsed;
        remaining = (n_available - i) / rate;
        fprintf('  Loading: %d/%d (%.0f%%) - %.1f frames/s - ETA: %.0fs\n', ...
                i, n_available, 100*i/n_available, rate, remaining);
    end
end

% Concatenate into single arrays
fprintf('  Concatenating data...\n');
all_x = vertcat(all_x_cell{:});
all_y = vertcat(all_y_cell{:});
all_u = vertcat(all_u_cell{:});
all_v = vertcat(all_v_cell{:});
all_w = vertcat(all_w_cell{:});
clear all_x_cell all_y_cell all_u_cell all_v_cell all_w_cell;

total_particles = length(all_x);

load_time = toc(load_start);
fprintf('Data loaded in %.1f seconds (%.1f min)\n', load_time, load_time/60);
fprintf('Total particles: %.2f million\n\n', total_particles / 1e6);

%% ========== PASS 1: COMPUTE GLOBAL REFERENCE PROFILE ==========
% This is the key to accurate DNS ground truth:
% Exploit streamwise homogeneity to compute the TRUE mean velocity U(y)
% by averaging over ALL particles at each y-location.

fprintf('=== PASS 1: Computing Global Reference Profile ===\n');
fprintf('  (Exploiting streamwise homogeneity for true population mean)\n\n');

% Define fine bins for reference profile (1-pixel resolution)
dy_ref = mm_per_pixel;
ref_y_edges = 0 : dy_ref : (Ny * mm_per_pixel);
ref_y_centers = ref_y_edges(1:end-1) + dy_ref/2;
n_ref_bins = length(ref_y_centers);

% 1D Bin indices for all particles
ref_bin_idx = floor(all_y / dy_ref) + 1;
ref_bin_idx = max(1, min(n_ref_bins, ref_bin_idx));

% Compute reference sums using accumarray (vectorized, fast)
fprintf('  Binning particles into %d y-bins...\n', n_ref_bins);
tic;

ref_count = accumarray(ref_bin_idx, 1, [n_ref_bins, 1]);
ref_sum_u = accumarray(ref_bin_idx, double(all_u), [n_ref_bins, 1]);
ref_sum_v = accumarray(ref_bin_idx, double(all_v), [n_ref_bins, 1]);
ref_sum_w = accumarray(ref_bin_idx, double(all_w), [n_ref_bins, 1]);

% Compute global mean profiles (the "True" Population Mean)
valid_bins = ref_count > 0;
U_ref_profile = NaN(size(ref_y_centers));
V_ref_profile = NaN(size(ref_y_centers));
W_ref_profile = NaN(size(ref_y_centers));

U_ref_profile(valid_bins) = ref_sum_u(valid_bins) ./ ref_count(valid_bins);
V_ref_profile(valid_bins) = ref_sum_v(valid_bins) ./ ref_count(valid_bins);
W_ref_profile(valid_bins) = ref_sum_w(valid_bins) ./ ref_count(valid_bins);

% Fill gaps using linear interpolation (for any empty bins)
U_ref_profile = fillmissing(U_ref_profile, 'linear');
V_ref_profile = fillmissing(V_ref_profile, 'linear');
W_ref_profile = fillmissing(W_ref_profile, 'linear');

pass1_time = toc;
fprintf('  Global reference profile computed in %.1f s\n', pass1_time);
fprintf('  Mean particles per y-bin: %.0f\n', mean(ref_count(valid_bins)));
fprintf('  Min/Max particles per bin: %d / %d\n', min(ref_count(valid_bins)), max(ref_count(valid_bins)));

% Create interpolants for fluctuation calculation
F_U = griddedInterpolant(ref_y_centers, U_ref_profile, 'linear', 'nearest');
F_V = griddedInterpolant(ref_y_centers, V_ref_profile, 'linear', 'nearest');
F_W = griddedInterpolant(ref_y_centers, W_ref_profile, 'linear', 'nearest');

fprintf('\n');

%% ========== PASS 2: COMPUTE FLUCTUATIONS AND BIN (SERIAL, CHUNKED) ==========
% Key insight: Compute u' = u - U_ref(y) FIRST, then square.
% This avoids catastrophic cancellation in variance computation.
%
% MEMORY OPTIMIZATION: Process in serial chunks to avoid parfor broadcast overhead

fprintf('=== PASS 2: Serial Chunked Binning of Fluctuations ===\n');
fprintf('  (Memory-efficient: no parfor broadcast of large arrays)\n\n');

% Initialize pixel-level accumulator grids
fprintf('  Initializing %d x %d pixel grids...\n', Nx, Ny);
grid_count  = zeros(Nx, Ny);
grid_sum_u  = zeros(Nx, Ny);
grid_sum_v  = zeros(Nx, Ny);
grid_sum_w  = zeros(Nx, Ny);
grid_sum_uu = zeros(Nx, Ny);  % This is sum(u'^2), NOT sum(u^2)
grid_sum_vv = zeros(Nx, Ny);
grid_sum_ww = zeros(Nx, Ny);
grid_sum_uv = zeros(Nx, Ny);
grid_sum_uw = zeros(Nx, Ny);
grid_sum_vw = zeros(Nx, Ny);

% Process in chunks to show progress and manage memory
n_chunks = 20;  % Process in 20 chunks
chunk_size = ceil(total_particles / n_chunks);

fprintf('  Processing %d chunks of ~%.1f million particles each...\n', n_chunks, chunk_size/1e6);
pass2_start = tic;

for chunk = 1:n_chunks
    % Define chunk indices
    idx_start = (chunk-1)*chunk_size + 1;
    idx_end = min(chunk*chunk_size, total_particles);
    
    if idx_start > total_particles
        break;
    end
    
    % Extract chunk data
    x_chunk = double(all_x(idx_start:idx_end));
    y_chunk = double(all_y(idx_start:idx_end));
    u_chunk = double(all_u(idx_start:idx_end));
    v_chunk = double(all_v(idx_start:idx_end));
    w_chunk = double(all_w(idx_start:idx_end));
    
    % Convert to pixel indices (1-based for MATLAB)
    px_x = floor(x_chunk / mm_per_pixel) + 1;
    px_y = floor(y_chunk / mm_per_pixel) + 1;
    
    % Clamp to valid range
    px_x = max(1, min(Nx, px_x));
    px_y = max(1, min(Ny, px_y));
    
    % Linear index for 2D grid
    lin_idx = sub2ind([Nx, Ny], px_x, px_y);
    
    % *** KEY STEP: Compute fluctuations using GLOBAL reference ***
    % This is the numerically stable two-pass method
    u_prime = u_chunk - F_U(y_chunk);
    v_prime = v_chunk - F_V(y_chunk);
    w_prime = w_chunk - F_W(y_chunk);
    
    % Accumulate into grids
    sz = [Nx*Ny, 1];
    
    grid_count  = grid_count  + reshape(accumarray(lin_idx, 1, sz), Nx, Ny);
    grid_sum_u  = grid_sum_u  + reshape(accumarray(lin_idx, u_chunk, sz), Nx, Ny);
    grid_sum_v  = grid_sum_v  + reshape(accumarray(lin_idx, v_chunk, sz), Nx, Ny);
    grid_sum_w  = grid_sum_w  + reshape(accumarray(lin_idx, w_chunk, sz), Nx, Ny);
    grid_sum_uu = grid_sum_uu + reshape(accumarray(lin_idx, u_prime.^2, sz), Nx, Ny);
    grid_sum_vv = grid_sum_vv + reshape(accumarray(lin_idx, v_prime.^2, sz), Nx, Ny);
    grid_sum_ww = grid_sum_ww + reshape(accumarray(lin_idx, w_prime.^2, sz), Nx, Ny);
    grid_sum_uv = grid_sum_uv + reshape(accumarray(lin_idx, u_prime.*v_prime, sz), Nx, Ny);
    grid_sum_uw = grid_sum_uw + reshape(accumarray(lin_idx, u_prime.*w_prime, sz), Nx, Ny);
    grid_sum_vw = grid_sum_vw + reshape(accumarray(lin_idx, v_prime.*w_prime, sz), Nx, Ny);
    
    % Progress
    elapsed = toc(pass2_start);
    rate = idx_end / elapsed;
    remaining = (total_particles - idx_end) / rate;
    fprintf('  Chunk %2d/%d: particles %d-%d (%.0f%%) - ETA: %.0fs\n', ...
            chunk, n_chunks, idx_start, idx_end, 100*idx_end/total_particles, remaining);
end

pass2_time = toc(pass2_start);
fprintf('  Pass 2 completed in %.1f s (%.1f min)\n\n', pass2_time, pass2_time/60);

% Clear large particle arrays to free memory before Pass 3
clear all_x all_y all_u all_v all_w;
fprintf('  Cleared particle arrays to free memory.\n\n');

%% ========== PASS 3: WINDOW CONVOLUTION FOR SPATIAL AVERAGING ==========
fprintf('=== PASS 3: Window Convolution (Parallel) ===\n\n');

% Start parallel pool if not already running
currPool = gcp('nocreate');
if isempty(currPool)
    parpool;
end
n_workers = gcp().NumWorkers;
fprintf('  Using Parallel Pool with %d workers.\n\n', n_workers);

% Initialize output structures
ensemble_stats = struct();
coordinates = struct();

% Temporary cell arrays for parfor results
stats_cell = cell(n_window_sizes, 1);
coords_cell = cell(n_window_sizes, 1);

% Note: parfor here is safe because we're only passing the grid arrays,
% not the massive particle arrays. Grid arrays are Nx x Ny (~33 MB each).

parfor win_idx = 1:n_window_sizes
    W = window_sizes(win_idx);
    overlap = overlaps(win_idx);
    spacing = W * (1 - overlap);
    
    % Convolution kernel (box filter)
    box_kernel = ones(W, W);
    
    % Perform convolutions to sum over WxW windows
    c_count  = conv2(grid_count, box_kernel, 'valid');
    c_sum_u  = conv2(grid_sum_u, box_kernel, 'valid');
    c_sum_v  = conv2(grid_sum_v, box_kernel, 'valid');
    c_sum_w  = conv2(grid_sum_w, box_kernel, 'valid');
    c_sum_uu = conv2(grid_sum_uu, box_kernel, 'valid');
    c_sum_vv = conv2(grid_sum_vv, box_kernel, 'valid');
    c_sum_ww = conv2(grid_sum_ww, box_kernel, 'valid');
    c_sum_uv = conv2(grid_sum_uv, box_kernel, 'valid');
    c_sum_uw = conv2(grid_sum_uw, box_kernel, 'valid');
    c_sum_vw = conv2(grid_sum_vw, box_kernel, 'valid');
    
    % Sample at window center locations (with spacing for overlap)
    sample_indices_x = 1 : spacing : size(c_count, 1);
    sample_indices_y = 1 : spacing : size(c_count, 2);
    
    % Extract sampled data
    count_sampled = c_count(sample_indices_x, sample_indices_y);
    
    % Avoid division by zero
    div_count = count_sampled;
    div_count(div_count == 0) = NaN;
    
    % Compute window center coordinates
    first_center = (W - 1) / 2;
    n_win_x = length(sample_indices_x);
    n_win_y = length(sample_indices_y);
    win_ctrs_x = first_center + (0:(n_win_x-1)) * spacing;
    win_ctrs_y = first_center + (0:(n_win_y-1)) * spacing;
    
    % === Build output structure ===
    s = struct();
    
    s.window_size_px = W;
    s.window_size_mm = W * mm_per_pixel;
    s.overlap = overlap;
    s.n_windows = [n_win_x, n_win_y];
    
    % Mean velocities [mm/s]
    s.U_mean = single(c_sum_u(sample_indices_x, sample_indices_y) ./ div_count);
    s.V_mean = single(c_sum_v(sample_indices_x, sample_indices_y) ./ div_count);
    s.W_mean = single(c_sum_w(sample_indices_x, sample_indices_y) ./ div_count);
    
    % Reynolds stresses [mm²/s²]
    % NOTE: These are <u'^2>, NOT <u^2> - <u>^2
    % The fluctuations were computed in Pass 2 using the global reference
    s.UU_stress = single(c_sum_uu(sample_indices_x, sample_indices_y) ./ div_count);
    s.VV_stress = single(c_sum_vv(sample_indices_x, sample_indices_y) ./ div_count);
    s.WW_stress = single(c_sum_ww(sample_indices_x, sample_indices_y) ./ div_count);
    s.UV_stress = single(c_sum_uv(sample_indices_x, sample_indices_y) ./ div_count);
    s.UW_stress = single(c_sum_uw(sample_indices_x, sample_indices_y) ./ div_count);
    s.VW_stress = single(c_sum_vw(sample_indices_x, sample_indices_y) ./ div_count);
    
    % Particle counts
    s.count = single(count_sampled);
    s.total_particles = sum(count_sampled(:), 'omitnan');
    
    % === 1D Profiles (spatially averaged over x) ===
    s.y_mm = single(win_ctrs_y * mm_per_pixel);
    s.y_plus = single(win_ctrs_y * mm_per_pixel / delta_nu);
    
    % Mean velocity profiles
    s.U_profile = single(mean(s.U_mean, 1, 'omitnan'));
    s.V_profile = single(mean(s.V_mean, 1, 'omitnan'));
    s.W_profile = single(mean(s.W_mean, 1, 'omitnan'));
    
    % Reynolds stress profiles
    s.UU_profile = single(mean(s.UU_stress, 1, 'omitnan'));
    s.VV_profile = single(mean(s.VV_stress, 1, 'omitnan'));
    s.WW_profile = single(mean(s.WW_stress, 1, 'omitnan'));
    s.UV_profile = single(mean(s.UV_stress, 1, 'omitnan'));
    s.UW_profile = single(mean(s.UW_stress, 1, 'omitnan'));
    s.VW_profile = single(mean(s.VW_stress, 1, 'omitnan'));
    
    % Profiles in wall units
    s.U_plus_profile = single(s.U_profile / u_tau);
    s.UU_plus_profile = single(s.UU_profile / u_tau^2);
    s.VV_plus_profile = single(s.VV_profile / u_tau^2);
    s.WW_plus_profile = single(s.WW_profile / u_tau^2);
    s.UV_plus_profile = single(s.UV_profile / u_tau^2);
    
    % Particle count profile
    s.count_profile = single(sum(s.count, 1, 'omitnan'));
    
    stats_cell{win_idx} = s;
    
    % === Coordinates structure ===
    c = struct();
    
    [X_px, Y_px] = ndgrid(win_ctrs_x, win_ctrs_y);
    X_mm = X_px * mm_per_pixel;
    Y_mm = Y_px * mm_per_pixel;
    Y_plus = Y_mm / delta_nu;
    
    c.window_size_px = W;
    c.window_size_mm = W * mm_per_pixel;
    c.overlap = overlap;
    c.n_windows = [n_win_x, n_win_y];
    c.win_ctrs_x_px = single(win_ctrs_x);
    c.win_ctrs_y_px = single(win_ctrs_y);
    c.win_ctrs_x_mm = single(win_ctrs_x * mm_per_pixel);
    c.win_ctrs_y_mm = single(win_ctrs_y * mm_per_pixel);
    c.X_px = single(X_px);
    c.Y_px = single(Y_px);
    c.X_mm = single(X_mm);
    c.Y_mm = single(Y_mm);
    c.Y_plus = single(Y_plus);
    
    coords_cell{win_idx} = c;
    
    fprintf('  Completed %d px window: %d x %d grid, %.0f particles\n', ...
            W, n_win_x, n_win_y, s.total_particles);
end

% Unpack cell arrays to struct arrays
ensemble_stats = [stats_cell{:}];
coordinates = [coords_cell{:}];

fprintf('\nStatistics computation complete.\n\n');

%% ========== SAVE RESULTS ==========
fprintf('Saving results...\n');

% Save ensemble statistics
stats_file = fullfile(output_folder, 'ensemble_statistics.mat');
save(stats_file, 'ensemble_stats', 'u_tau', 'nu', 'delta_nu', 'h_mm', 'mm_per_pixel', ...
     'U_ref_profile', 'V_ref_profile', 'W_ref_profile', 'ref_y_centers', '-v7.3');
fprintf('  Saved: %s\n', stats_file);

% Save coordinates
coords_file = fullfile(output_folder, 'coordinates.mat');
save(coords_file, 'coordinates', '-v7.3');
fprintf('  Saved: %s\n', coords_file);

% Save wall unit parameters
wall_units = struct();
wall_units.u_tau = u_tau;            % [mm/s]
wall_units.nu = nu;                  % [mm²/s]
wall_units.delta_nu = delta_nu;      % [mm]
wall_units.h_mm = h_mm;              % [mm]
wall_units.Re_tau = h_mm / delta_nu;
wall_units_file = fullfile(output_folder, 'wall_units.mat');
save(wall_units_file, 'wall_units');
fprintf('  Saved: %s\n', wall_units_file);

% Save global reference profile (the "true" mean)
reference = struct();
reference.y_mm = ref_y_centers;
reference.y_plus = ref_y_centers / delta_nu;
reference.U = U_ref_profile;
reference.V = V_ref_profile;
reference.W = W_ref_profile;
reference.U_plus = U_ref_profile / u_tau;
reference.particle_count = ref_count;
reference_file = fullfile(output_folder, 'global_reference_profile.mat');
save(reference_file, 'reference');
fprintf('  Saved: %s\n', reference_file);

% Save 1D profiles for easy benchmarking
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
    profiles.(field_name).U = ensemble_stats(win_idx).U_profile;
    profiles.(field_name).V = ensemble_stats(win_idx).V_profile;
    profiles.(field_name).W = ensemble_stats(win_idx).W_profile;
    
    % Reynolds stresses (dimensional)
    profiles.(field_name).uu = ensemble_stats(win_idx).UU_profile;
    profiles.(field_name).vv = ensemble_stats(win_idx).VV_profile;
    profiles.(field_name).ww = ensemble_stats(win_idx).WW_profile;
    profiles.(field_name).uv = ensemble_stats(win_idx).UV_profile;
    profiles.(field_name).uw = ensemble_stats(win_idx).UW_profile;
    profiles.(field_name).vw = ensemble_stats(win_idx).VW_profile;
    
    % Wall units
    profiles.(field_name).U_plus = ensemble_stats(win_idx).U_plus_profile;
    profiles.(field_name).uu_plus = ensemble_stats(win_idx).UU_plus_profile;
    profiles.(field_name).vv_plus = ensemble_stats(win_idx).VV_plus_profile;
    profiles.(field_name).ww_plus = ensemble_stats(win_idx).WW_plus_profile;
    profiles.(field_name).uv_plus = ensemble_stats(win_idx).UV_plus_profile;
    
    % Particle count
    profiles.(field_name).count = ensemble_stats(win_idx).count_profile;
end
profiles.wall_units = wall_units;
profiles.global_reference = reference;
profiles_file = fullfile(output_folder, 'profiles.mat');
save(profiles_file, 'profiles');
fprintf('  Saved: %s\n\n', profiles_file);

%% ========== GENERATE VALIDATION PLOTS ==========
fprintf('Generating validation plots...\n');

% Color scheme for different window sizes
colors = lines(n_window_sizes);

%% --- Plot 1: Mean streamwise velocity U vs y ---
figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
hold on;
% Plot global reference profile (black, thick)
plot(U_ref_profile, ref_y_centers, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Global Reference');
for win_idx = 1:n_window_sizes
    W = window_sizes(win_idx);
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
% Plot global reference in wall units
plot(ref_y_centers / delta_nu, U_ref_profile / u_tau, 'k-', 'LineWidth', 2.5, ...
     'DisplayName', 'Global Reference');
for win_idx = 1:n_window_sizes
    W = window_sizes(win_idx);
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
% Log law: U+ = (1/kappa)*ln(y+) + B
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

sgtitle('Streamwise Velocity Profiles (DNS Ground Truth)', 'FontSize', 16);
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

sgtitle('Reynolds Normal Stresses in Wall Units (DNS Ground Truth)', 'FontSize', 16);
saveas(gcf, fullfile(output_folder, 'reynolds_normal_stresses.png'));
saveas(gcf, fullfile(output_folder, 'reynolds_normal_stresses.fig'));
fprintf('  Saved reynolds_normal_stresses.png/fig\n');

%% --- Plot 3: Reynolds shear stress -<u'v'> vs y+ ---
figure('Position', [100, 100, 800, 600]);

hold on;
for win_idx = 1:n_window_sizes
    W = window_sizes(win_idx);
    % Note: UV_plus_profile stores <u'v'>, we want -<u'v'>
    UV_plus = -ensemble_stats(win_idx).UV_plus_profile;
    y_plus = ensemble_stats(win_idx).y_plus;
    plot(y_plus, UV_plus, '-', 'LineWidth', 1.5, 'Color', colors(win_idx, :), ...
         'DisplayName', sprintf('%d px', W));
end

% Add theoretical line: -<u'v'>+ = 1 - y/h for channel flow
y_plus_theory = linspace(0, 1000, 100);
y_over_h = y_plus_theory * delta_nu / h_mm;
UV_plus_theory = 1 - y_over_h;
UV_plus_theory(UV_plus_theory < 0) = 0;
plot(y_plus_theory, UV_plus_theory, 'k--', 'LineWidth', 1, ...
     'DisplayName', '-<u''v''>^+ = 1 - y/h');

hold off;
xlabel('y^+', 'FontSize', 12);
ylabel('-<u''v''>^+', 'FontSize', 12);
title('Reynolds Shear Stress in Wall Units (DNS Ground Truth)', 'FontSize', 14);
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
title('Turbulent Kinetic Energy in Wall Units (DNS Ground Truth)', 'FontSize', 14);
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

for win_idx = 1:min(3, n_window_sizes)
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

%% --- Plot 6: Global reference profile quality check ---
figure('Position', [100, 100, 1000, 400]);

subplot(1, 2, 1);
plot(ref_y_centers, ref_count, 'b-', 'LineWidth', 1);
xlabel('y [mm]', 'FontSize', 12);
ylabel('Particle count per bin', 'FontSize', 12);
title('Global Reference: Particles per y-bin', 'FontSize', 14);
grid on;

subplot(1, 2, 2);
semilogy(ref_y_centers / delta_nu, ref_count, 'b-', 'LineWidth', 1);
xlabel('y^+', 'FontSize', 12);
ylabel('Particle count per bin (log)', 'FontSize', 12);
title('Global Reference: Particles per y-bin (Wall Units)', 'FontSize', 14);
grid on;

sgtitle('Global Reference Profile Quality', 'FontSize', 14);
saveas(gcf, fullfile(output_folder, 'reference_profile_quality.png'));
fprintf('  Saved reference_profile_quality.png\n');

%% ========== SUMMARY ==========
fprintf('\n=== Processing Complete ===\n\n');
fprintf('Method: Two-Pass Global Mean (DNS Ground Truth)\n');
fprintf('  - Pass 1: Global reference profile U_ref(y) from ALL particles\n');
fprintf('  - Pass 2: Fluctuations u'' = u - U_ref(y), then Reynolds stresses (SERIAL)\n');
fprintf('  - Pass 3: Window convolution for spatial averaging (PARALLEL)\n');
fprintf('  - This preserves large-scale turbulence and avoids numerical errors\n\n');

fprintf('Output folder: %s\n', fullfile(pwd, output_folder));
fprintf('\nFiles created:\n');
fprintf('  - ensemble_statistics.mat     (2D fields + 1D profiles + global reference)\n');
fprintf('  - coordinates.mat             (grid coordinates for each window size)\n');
fprintf('  - profiles.mat                (1D benchmark profiles)\n');
fprintf('  - global_reference_profile.mat (true mean U(y) from all particles)\n');
fprintf('  - wall_units.mat              (u_tau, nu, delta_nu, Re_tau)\n');
fprintf('  - velocity_profiles.png/fig\n');
fprintf('  - reynolds_normal_stresses.png/fig\n');
fprintf('  - reynolds_shear_stress.png/fig\n');
fprintf('  - turbulent_kinetic_energy.png/fig\n');
fprintf('  - particle_count_distribution.png\n');
fprintf('  - reference_profile_quality.png\n');

fprintf('\nWall unit parameters:\n');
fprintf('  u_tau = %.4f mm/s\n', u_tau);
fprintf('  delta_nu = %.4f mm\n', delta_nu);
fprintf('  Re_tau = %.0f\n', h_mm / delta_nu);

fprintf('\nStatistics computed for window sizes: ');
fprintf('%d ', window_sizes);
fprintf('pixels\n');

fprintf('\nKey difference from naive method:\n');
fprintf('  - Reynolds stresses are <(u - U_ref)^2>, NOT <u^2> - <u>^2\n');
fprintf('  - This avoids catastrophic cancellation and high-pass filtering\n');
fprintf('  - Large-scale turbulent structures are preserved in the statistics\n');
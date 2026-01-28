%% Download Particle Position Pairs from JHTDB Channel Flow
% Downloads velocity data and creates synthetic PIV particle position files
% Uses parallel processing with 10 workers for faster downloads
%
% Output: Frame pairs B00001_A.data and B00001_B.data (positions in mm)
%
% Domain: Bottom half of channel (y: -1 to 0), streamwise 0 to 1h
% Image: 2048x2048 pixels, minimum window 16x16 = 1/128 h
% Target: ~15 pixel displacement at centerline velocity

clear variables;
close all;

%% ========== USER CONFIGURATION ==========
authkey = 'uk.ac.cam.dm529-ca4ff15b';
dataset = 'channel';

% getData parameters
variable = 'velocity';
temporal_method = 'none';
spatial_method = 'lag4';
spatial_operator = 'field';

% Parallel workers
n_workers = 10;

% Number of frames to download
n_frames = 1000;

% Output folder
data_folder = 'C:\Users\mtt1e23\OneDrive - University of Southampton\Documents\#current_processing\query_JHTDB\download_from_jhtdb\particle_positions';

%% ========== PHYSICAL PARAMETERS ==========
% Channel half-height
h_mm = 150;  % mm

% Unit conversions (from DNS to physical)
length_conv = h_mm;        % mm per non-dimensional length unit (h)
velocity_conv = 168.75;    % mm/s per non-dimensional velocity unit

% Centerline velocity (from DNS data)
Uc_nondim = 1.1312;
Uc_mm_s = Uc_nondim * velocity_conv;  % = 190.89 mm/s

%% ========== DOMAIN SETUP ==========
% Domain in non-dimensional units (h = 1)
% x: streamwise (0 to 1h)
% y: wall-normal (-1 = wall, 0 = centerline) - bottom half only
% z: spanwise (thin sheet, 1 window width = 1/128 h)
min_xyz = [0, -1, -1/256];
max_xyz = [1,  0,  1/256];

%% ========== IMAGE & PIV PARAMETERS ==========
% Image size
Nx = 2048;
Ny = 2048;

% Domain extent mapped to image
domain_x = max_xyz(1) - min_xyz(1);  % 1h in x
domain_y = max_xyz(2) - min_xyz(2);  % 1h in y

% Spatial resolution
mm_per_pixel = (domain_y * length_conv) / Ny;  % 150/2048 ≈ 0.0732 mm/pixel

% Target displacement and dt calculation
target_displacement_pixels = 15;
target_displacement_mm = target_displacement_pixels * mm_per_pixel;
dt = target_displacement_mm / Uc_mm_s;  % PIV time step in seconds
disp(dt)

%% ========== PARTICLE SEEDING ==========
% For 8 particles per 16x16 window:
% Number of 16x16 windows = (2048/16)^2 = 16384
% Total particles = 16384 * 8 ≈ 131,072
n_points = 130000;

%% ========== DNS TIME STEPS ==========
% JHTDB channel flow: 4000 frames from t=0 to ~26 (non-dimensional)
final_t = 25.9740;
time_steps = linspace(0, final_t, n_frames);

%% ========== PRINT CONFIGURATION ==========
fprintf('=== JHTDB Particle Position Download ===\n\n');
fprintf('Domain (non-dim):\n');
fprintf('  x: [%.3f, %.3f] (streamwise)\n', min_xyz(1), max_xyz(1));
fprintf('  y: [%.3f, %.3f] (wall-normal, bottom half)\n', min_xyz(2), max_xyz(2));
fprintf('  z: [%.6f, %.6f] (spanwise, thin sheet)\n', min_xyz(3), max_xyz(3));
fprintf('\n');
fprintf('Physical parameters:\n');
fprintf('  Channel half-height h = %.0f mm\n', h_mm);
fprintf('  Centerline velocity Uc = %.2f mm/s\n', Uc_mm_s);
fprintf('\n');
fprintf('Image parameters:\n');
fprintf('  Image size: %d x %d pixels\n', Nx, Ny);
fprintf('  Resolution: %.4f mm/pixel\n', mm_per_pixel);
fprintf('  Min window: 16x16 pixels = %.4f mm\n', 16 * mm_per_pixel);
fprintf('\n');
fprintf('PIV parameters:\n');
fprintf('  Target displacement: %d pixels = %.4f mm\n', target_displacement_pixels, target_displacement_mm);
fprintf('  Calculated dt: %.6f s (%.3f ms)\n', dt, dt*1000);
fprintf('  Particles per frame: %d\n', n_points);
fprintf('\n');
fprintf('Download parameters:\n');
fprintf('  Total frames: %d\n', n_frames);
fprintf('  DNS time range: [0, %.4f]\n', final_t);
fprintf('  Parallel workers: %d\n', n_workers);
fprintf('  Output folder: %s\n', data_folder);
fprintf('\n');

%% ========== CREATE OUTPUT FOLDER ==========
if ~exist(data_folder, 'dir')
    mkdir(data_folder);
    fprintf('Created output folder: %s\n', data_folder);
end

%% ========== SAVE PARAMETERS ==========
params = struct();
params.authkey = authkey;
params.dataset = dataset;
params.n_frames = n_frames;
params.n_points = n_points;
params.dt = dt;
params.dt_ms = dt * 1000;
params.min_xyz = min_xyz;
params.max_xyz = max_xyz;
params.h_mm = h_mm;
params.length_conv = length_conv;
params.velocity_conv = velocity_conv;
params.target_displacement_pixels = target_displacement_pixels;
params.target_displacement_mm = target_displacement_mm;
params.Uc_nondim = Uc_nondim;
params.Uc_mm_s = Uc_mm_s;
params.mm_per_pixel = mm_per_pixel;
params.Nx = Nx;
params.Ny = Ny;
params.time_steps = time_steps;
params.final_t = final_t;
save(fullfile(data_folder, 'download_parameters.mat'), 'params');
fprintf('Saved parameters to: %s\n\n', fullfile(data_folder, 'download_parameters.mat'));

%% ========== INITIALIZE PARALLEL POOL ==========

target_workers = 10; 

fprintf('Configuring cluster for %d workers...\n', target_workers);

% 1. Get the local cluster object
c = parcluster('local');

% 2. Force the cluster to accept your desired number of workers
c.NumWorkers = target_workers;

% 5. Start the pool
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool(c, target_workers);
elseif pool.NumWorkers ~= target_workers
    delete(pool);
    pool = parpool(c, target_workers);
end

fprintf('Parallel pool ready with %d workers.\n\n', pool.NumWorkers);

%% ========== MAIN DOWNLOAD LOOP ==========
fprintf('Starting download of %d frame pairs...\n', n_frames);
fprintf('============================================\n');
tic;

% Track failures
failed_frames = zeros(1, n_frames);

parfor frame_idx = 1:n_frames
    % Get DNS time for this frame
    t = time_steps(frame_idx);
    frame_str = sprintf('%05d', frame_idx);

    % Generate random points in the domain
    % Using frame_idx as seed for reproducibility
    rng_state = rng(frame_idx, 'twister');
    points = zeros(n_points, 3);
    points(:, 1) = rand(n_points, 1) * (max_xyz(1) - min_xyz(1)) + min_xyz(1);
    points(:, 2) = rand(n_points, 1) * (max_xyz(2) - min_xyz(2)) + min_xyz(2);
    points(:, 3) = rand(n_points, 1) * (max_xyz(3) - min_xyz(3)) + min_xyz(3);

    % Retry logic for getData
    success = false;
    retry_count = 0;
    max_retries = 10;
    result = [];

    while ~success && retry_count < max_retries
        try
            result = getData(authkey, dataset, variable, t, ...
                            temporal_method, spatial_method, spatial_operator, points);

            % Check for error response
            if isstring(result) || ischar(result)
                error('getData returned error string');
            end
            if isempty(result) || size(result, 2) ~= 3
                error('getData returned invalid result');
            end

            success = true;
        catch ME
            retry_count = retry_count + 1;
            if retry_count < max_retries
                pause(1 + rand());  % Random backoff
            end
        end
    end

    if ~success
        fprintf('Frame %s: FAILED after %d retries\n', frame_str, max_retries);
        failed_frames(frame_idx) = 1;
        continue;
    end

    % Convert positions to physical units (mm)
    x_mm = points(:, 1) * length_conv;
    y_mm = points(:, 2) * length_conv;
    z_mm = points(:, 3) * length_conv;

    % Convert velocities to physical units (mm/s)
    u_mm_s = result(:, 1) * velocity_conv;
    v_mm_s = result(:, 2) * velocity_conv;
    w_mm_s = result(:, 3) * velocity_conv;

    % Compute particle positions using symmetric half-dt scheme
    % pos_A = position at t - dt/2
    % pos_B = position at t + dt/2
    half_dt = 0.5 * dt;
    pos_A = single([x_mm - half_dt * u_mm_s, ...
                    y_mm - half_dt * v_mm_s, ...
                    z_mm - half_dt * w_mm_s]);
    pos_B = single([x_mm + half_dt * u_mm_s, ...
                    y_mm + half_dt * v_mm_s, ...
                    z_mm + half_dt * w_mm_s]);

    % Save to files (space-delimited text)
    file_A = fullfile(data_folder, ['B', frame_str, '_A.data']);
    file_B = fullfile(data_folder, ['B', frame_str, '_B.data']);
    writematrix(pos_A, file_A, 'Delimiter', ' ', 'FileType', 'text');
    writematrix(pos_B, file_B, 'Delimiter', ' ', 'FileType', 'text');

    % Progress indicator (will be somewhat out of order due to parfor)
    if mod(frame_idx, 50) == 0
        fprintf('Frame %s completed (t = %.4f)\n', frame_str, t);
    end
end

elapsed_time = toc;

%% ========== SUMMARY ==========
fprintf('\n============================================\n');
fprintf('Download complete!\n\n');
fprintf('Summary:\n');
fprintf('  Total frames requested: %d\n', n_frames);
fprintf('  Successful downloads: %d\n', n_frames - sum(failed_frames));
fprintf('  Failed downloads: %d\n', sum(failed_frames));
fprintf('  Elapsed time: %.1f seconds (%.1f min)\n', elapsed_time, elapsed_time/60);
fprintf('  Average time per frame: %.2f seconds\n', elapsed_time / n_frames);
fprintf('\n');
fprintf('Output files:\n');
fprintf('  Location: %s\n', fullfile(pwd, data_folder));
fprintf('  Format: B#####_A.data and B#####_B.data\n');
fprintf('  Coordinates: mm (x, y, z)\n');
fprintf('\n');

% Report any failures
if sum(failed_frames) > 0
    fprintf('Failed frames:\n');
    failed_indices = find(failed_frames);
    fprintf('  %s\n', mat2str(failed_indices));

    % Save failed frames list
    save(fullfile(data_folder, 'failed_frames.mat'), 'failed_indices');
    fprintf('  Saved to: %s\n', fullfile(data_folder, 'failed_frames.mat'));
end

fprintf('\nDone.\n');

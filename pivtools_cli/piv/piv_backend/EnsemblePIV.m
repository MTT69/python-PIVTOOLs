function EnsemblePIV(setup, masks,filters)
 % * Function: EnsemblePIV
 % * ---------------------------------------------------------------------------
 % * Description:
 % * This function performs ensemble PIV (Particle Image Velocimetry) analysis 
 % * across multiple cameras and passes, processing the PIV data in an iterative 
 % * manner to improve accuracy. The ensemble PIV approach allows the computation 
 % * of average flow fields by analyzing a collection of instantaneous velocity 
 % * measurements over a specified number of passes.
 % * 
 % * The function performs the following operations:
 % *   1. Initializes the necessary directories for storing uncalibrated and 
 % *      calibrated PIV results.
 % *   2. Generates masks and predictor fields for each pass.
 % *   3. Executes PIV processing using the `PIV_2D_wdef_ensemble` function.
 % *   4. Processes correlation planes, refines PIV results, and generates 
 % *      ensemble velocity fields.
 % *   5. Calculates coordinates and stores PIV results, such as velocity and 
 % *      correlation data, across all passes.
 % * 
 % * The ensemble approach refines the velocity field predictions across 
 % * successive passes to ensure higher accuracy. Results are saved in specified 
 % * directories for further analysis.
 % * 
 % * ---------------------------------------------------------------------------
 % * Inputs:
 % *   struct setup    : A structure containing setup and configuration 
 % *                     parameters for the PIV analysis, including:
 % *       - setup.imProperties.cameraCount   : Number of cameras in the analysis.
 % *       - setup.imProperties.imageCount    : Number of images per camera.
 % *       - setup.ensemble.passes            : Number of ensemble passes to 
 % *                                            refine the flow field.
 % *       - setup.ensemble.windowSize        : Size of the interrogation window 
 % *                                            used in each pass.
 % *       - setup.directory.base             : Base directory for loading and 
 % *                                            saving data.
 % * 
 % *   struct masks    : A structure containing masking information for each camera.
 % *                     Each mask is used to limit the regions in the PIV images 
 % *                     to those of interest.
 % * 
 % *   struct filters  : A structure containing filters to process the PIV 
 % *                     correlation planes, applied to refine the velocity 
 % *                     predictions.
 % * 
 % *   Example:
 % *   setup = struct('imProperties', struct('cameraCount', 2, 'imageCount', 100),
 % *                  'ensemble', struct('passes', 3, 'windowSize', [32, 32]),
 % *                  'directory', struct('base', '/path/to/data'));
 % *   masks = struct(...); % Define masks for the cameras.
 % *   filters = struct(...); % Define filters to refine PIV data.
 % *   
 % * ---------------------------------------------------------------------------
 % * Outputs:
 % *   - The function saves PIV results (uncalibrated and calibrated) for each 
 % *     camera in directories under:
 % *       - Uncalibrated: `setup.directory.base/UncalibratedPIV/<imageCount>/Cam<cameraNo>/Ensemble/`
 % *       - Calibrated: `setup.directory.base/CalibratedPIV/<imageCount>/Cam<cameraNo>/Ensemble/`
 % *     
 % *     The saved `.mat` files contain the following variables:
 % *       - piv_result: Struct containing velocity fields, correlation planes, 
 % *                     predictor fields, and other PIV-related data for each pass.
 % *       - Co_ords   : Struct containing coordinate data for the PIV results 
 % *                     (grid points and spacing).
 % *   
 % * ---------------------------------------------------------------------------
 % * Procedures:
 % *   1. **Initialize Directories**:
 % *      - Create directories for storing uncalibrated and calibrated PIV data 
 % *        for each camera and pass.
 % * 
 % *   2. **Mask and Predictor Field Generation**:
 % *      - For each camera, generate masks for the PIV processing and initialize 
 % *        the predictor field for the first pass.
 % * 
 % *   3. **PIV Processing**:
 % *      - Perform PIV analysis using the `PIV_2D_wdef_ensemble` function, which 
 % *        computes the displacement field across image pairs.
 % * 
 % *   4. **Correlation Plane Processing**:
 % *      - Refine PIV results by processing the correlation planes using the 
 % *        `evaluate_correlation_planes` and `process_correlation_planes` 
 % *        functions. These functions extract the most likely velocity vectors 
 % *        from the correlation peaks.
 % * 
 % *   5. **Coordinate Generation**:
 % *      - For each pass, generate coordinates (grid points) using the 
 % *        `generateCoordinates` function. These coordinates are used to store 
 % *        the spatial positions of the PIV vectors.
 % * 
 % *   6. **Save Results**:
 % *      - Save the computed PIV results (velocity fields, predictor fields, 
 % *        etc.) and coordinates in `.mat` files for further analysis.
 % * 
 % * ---------------------------------------------------------------------------

if setup.pipeline.ensemble
    % Loop over all cameras specified in the setup
    for cameraNo = 1:setup.imProperties.cameraCount
        
        % Initialize structures to store coordinate and PIV results for this camera
        Co_ords = struct();  
        piv_result = struct();
        
        % Load the mask for the current camera
        im_mask = masks(cameraNo).camera{1,1};
        
        % Define the directory path for storing uncalibrated PIV data for the current camera
        directory_path = fullfile(setup.directory.base, 'UncalibratedPIV', ...
            num2str(setup.imProperties.imageCount), ['Cam', num2str(cameraNo)], 'Ensemble');
        
        % Check if the directory exists; if not, create it
        if ~exist(directory_path, 'dir')
            mkdir(directory_path);  % Create the directory
        end
        
        % Define the directory path for storing calibrated PIV data for the current camera
        calibrated_directory_path = fullfile(setup.directory.base, 'CalibratedPIV', ...
            num2str(setup.imProperties.imageCount), ['Cam', num2str(cameraNo)], 'Ensemble');
        
        % Check if the calibrated directory exists; if not, create it
        if ~exist(calibrated_directory_path, 'dir')
            mkdir(calibrated_directory_path);  % Create the directory
        end
        
        % Compute the mask for the ensemble processing for the current camera
        b_mask = compute_b_mask_ensemble(setup, im_mask);

        images=struct();
        batchCount = ceil(setup.imProperties.imageCount / setup.imProperties.batchSize);
        % if single batch count used - load images outside of loop and
        % retain
        if batchCount == 1
            batchNo = 1;
            imRange = calculateImageRange(batchNo, batchCount, setup);    
            % Load and filter images outside of spmd to avoid redundant loading
            images = Load_images(setup, cameraNo, imRange, batchNo,masks);
            images = Filter_images(images, setup, filters, imRange, cameraNo, batchNo);
        end
        
        % Loop over the number of ensemble passes defined in the setup
        for pass = 1: setup.ensemble.passes
            if pass < setup.ensemble.resumeCase
                continue
            end

            if pass == setup.ensemble.resumeCase
                folderPath = fullfile(setup.directory.base, 'UncalibratedPIV', num2str(setup.imProperties.imageCount), ['Cam', num2str(cameraNo)], 'Ensemble');
                filePath = fullfile(folderPath, sprintf(setup.ensemble.nameConvention{1}, 1));
                if exist(filePath, 'file')
                    % Load the file if it exists
                    load(filePath);
                    piv_result(pass:end)=[];
                    fprintf('File loaded successfully: %s\n', filePath);
                else
                    % Throw an error if the file does not exist
                    error('File not found: %s\nPlease ensure the file exists at the specified location.', filePath);
                end
            end
            
            % Display progress for the current pass, window size, and camera number
            fprintf('Running Ensemble analysis for window x %d x y %d of camera %d at time %s\n', ...
                setup.ensemble.windowSize(pass,1), setup.ensemble.windowSize(pass,2), cameraNo, datetime('now'));
            
            % Initialize the predictor field for the current pass and camera
            [setup, predictorfield] = initialisePredictorField(setup, pass, piv_result, cameraNo);
            
            
            % Perform the PIV ensemble analysis for the current pass
            result = PIV_2D_wdef_ensemble(setup, predictorfield, pass, b_mask, cameraNo, im_mask, filters,images,masks);
            
            % Store the results from this pass in the piv_result structure
            fields = fieldnames(result);
            for i = 1:numel(fields)
                piv_result(pass).(fields{i}) = result(pass).(fields{i});
            end
            clear result  % Clear temporary result variable to free memory
            
            % Set up additional variables for ensemble analysis
            [Ensemble_guess_setup] = setup_variables(setup, piv_result, pass);
            
            % Extract predictor field data (u and v velocity components)
            predfield = piv_result(pass).Predictor_Field;
            predu = predfield(:,:,2);
            predv = predfield(:,:,1);
            
            % Apply the mask for the current pass
            mask = b_mask{pass};
            
            % Generate coordinate data for the current pass
            gen_co_ords = generateCoordinates(piv_result, pass, setup);
            fields = fieldnames(gen_co_ords);
            
            % Store generated coordinates in the Co_ords structure
            for i = 1:numel(fields)
                Co_ords(pass).(fields{i}) = gen_co_ords.(fields{i});
            end
            clear gen_co_ords  % Clear temporary variable to free memory
            
            % Ensure the directory for saving PIV data exists
            folderPath = fullfile(setup.directory.base, 'UncalibratedPIV', ...
                num2str(setup.imProperties.imageCount), ['Cam', num2str(cameraNo)], 'Ensemble');
            
            if ~exist(folderPath, 'dir')
                mkdir(folderPath);  % Create the folder if it doesn't exist
            end
            
            % Evaluate the correlation planes based on ensemble guess setup
            [gaussResult_full, status_full] = evaluate_correlation_planes(Ensemble_guess_setup, piv_result, pass, mask,setup);
            
            % Process the correlation planes to get the final results for this pass
            result = process_correlation_planes(piv_result, gaussResult_full, status_full, setup, pass, mask, predu, predv, Ensemble_guess_setup);
            fields = fieldnames(result);
            
            % Store the results from correlation processing in the piv_result structure
            for i = 1:numel(fields)
                piv_result(pass).(fields{i}) = result(pass).(fields{i});
            end
            clear result  % Clear temporary result variable to free memory
            
            % Calculate and store sum windows based on PIV results
            setup = calculate_SumWindow(setup, piv_result,pass);

            
            % Process the PIV results and save them using a separate function
            processPIVResults(setup, cameraNo, piv_result, pass, Co_ords);
            piv_result(pass).pointspreadA = [];
            piv_result(pass).pointspreadB = [];
            piv_result(pass).correlation_plane = [];
            
            % Clear unnecessary data fields to save memory
           
        end
    end
end

end
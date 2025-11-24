function setup = calculate_SumWindow(setup,piv_result,pass)
% * Function: calculate_SumWindow
%  * ---------------------------------------------------------------------------
%  * Description:
%  * This function calculates the sum window size (`SumWindow`) for ensemble 
%  * Particle Image Velocimetry (PIV) analysis. The sum window is computed based on 
%  * the spread (standard deviation) of the displacement fields (`sig_AB_x` and 
%  * `sig_AB_y`) from the PIV results of the converged run. The sum window size is 
%  * used to define the correlation processing area in subsequent PIV analysis passes.
%  * 
%  * The function determines the 95th percentile of the displacement field spread in 
%  * both the x and y directions and computes the sum window size as 12 times the 
%  * maximum of these values. It ensures the window size is a multiple of 2 and has 
%  * a minimum value of 16 to prevent excessively small windows.
%  * 
%  * ---------------------------------------------------------------------------
%  * Inputs:
%  *   struct setup       : A structure containing setup information and parameters 
%  *                        for the ensemble PIV analysis, including:
%  *       - setup.ensemble.convergedRun  : The pass number of the converged run in 
%  *                                        the ensemble PIV analysis.
%  *       - setup.ensemble.sumWindow     : The sum window size used for correlation 
%  *                                        processing, to be updated by this function.
%  * 
%  *   struct piv_result  : A structure containing the results from the PIV analysis 
%  *                        for each pass, including:
%  *       - piv_result.sig_AB_x          : The displacement field spread in the x 
%  *                                        direction for the converged run.
%  *       - piv_result.sig_AB_y          : The displacement field spread in the y 
%  *                                        direction for the converged run.
%  * 
%  * ---------------------------------------------------------------------------
%  * Outputs:
%  *   struct setup       : The updated `setup` structure with the newly calculated 
%  *                        sum window size (`setup.ensemble.sumWindow`) for ensemble 
%  *                        PIV analysis.
%  * 
%  * ---------------------------------------------------------------------------
%  * Procedures:
%  *   1. **Check Converged Run**:
%  *      - The function checks if the current pass is the converged run (defined in 
%  *        `setup.ensemble.convergedRun`). Only if this condition is met, the sum 
%  *        window calculation proceeds.
%  * 
%  *   2. **Filter Non-Empty Displacement Fields**:
%  *      - The function identifies non-NaN and non-zero displacement values in both 
%  *        the x (`sig_AB_x`) and y (`sig_AB_y`) displacement fields for the 
%  *        converged run.
%  * 
%  *   3. **Compute 95th Percentile**:
%  *      - The 95th percentile of the filtered displacement values in both x and y 
%  *        directions is computed to capture the significant spread in the 
%  *        displacement field.
%  * 
%  *   4. **Calculate Sum Window Size**:
%  *      - The sum window size is calculated as 12 times the maximum of the 95th 
%  *        percentiles in both directions (`sig_AB_x` and `sig_AB_y`).
%  *      - The window size is then rounded up to the nearest multiple of 2, ensuring 
%  *        it is even, and a minimum value of 16 is enforced to avoid very small 
%  *        windows.
%  * 
%  *   5. **Update Setup Structure**:
%  *      - The computed sum window size is saved back into the `setup` structure as 
%  *        `setup.ensemble.sumWindow`.
%  * 
%  * ---------------------------------------------------------------------------
    if pass==setup.ensemble.convergedRun && setup.pipeline.calculateSumWindow
        non_empty_idx = ~isnan(piv_result(setup.ensemble.convergedRun).sig_AB_x) & (piv_result(setup.ensemble.convergedRun).sig_AB_x ~= 0);
        percentile_95_sigx = prctile(piv_result(setup.ensemble.convergedRun).sig_AB_x(non_empty_idx), 95, 'all');    
        percentile_95_sigy = prctile(piv_result(setup.ensemble.convergedRun).sig_AB_y(non_empty_idx), 95, 'all');   
        Sumwindowvalx = 12* max(percentile_95_sigx);
        Sumwindowvaly = 12* max(percentile_95_sigy);
        Sumwindowval = max(Sumwindowvalx, Sumwindowvaly);

        % Ensure Sumwindowval is a multiple of 2
        Sumwindowval = ceil(Sumwindowval / 2) * 2;
        
        % Limit Sumwindowval to a maximum of 16
        Sumwindowval = max(Sumwindowval, 16);  % debug

        SumWindow = [Sumwindowval, Sumwindowval];
        disp('Calculated single window for ensemble analysis')
        setup.ensemble.sumWindow = SumWindow;
        disp(SumWindow)
    end
end
#include "PIV_2d_cross_correlate.h"
#include "common.h"
#include "xcorr.h"
#include "xcorr_cache.h"      /* FFTW wisdom caching */
#include "peak_locate_lm.h"   /* Fast LM solver instead of GSL */
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

unsigned char bulkxcorr2d(const float *fImageA, const float *fImageB, const float *fMask, const int *nImageSize,
                          const float *fWinCtrsX, const float *fWinCtrsY, const int *nWindows, float *fWindowWeightA, bool bEnsemble,
                          const float *fWindowWeightB, const int *nWindowSize, int nPeaks, int iPeakFinder,
                          float *fPkLocX, float *fPkLocY, float *fPkHeight, float *fSx, float *fSy, float *fSxy, float *fCorrelPlane_Out)
{
	int i, j, ii, jj, x, y;
	int xmin, ymin;
	int iWindowIdx, nWindowsTotal;
	float *fWindowA, *fWindowB;
	float *fCorrelPlane, *fStd, *fCorrelWeight;
	float *fPeakLoc;
	float fMeanA, fMeanB, fEnergyA, fEnergyB, fEnergyNorm;
	int nPxPerWindow, n;
	unsigned uError;
	sPlan sCCPlan;
	/* Removed peak_finder_lock - LM solver is thread-safe without locks */
	
	/* calculate correlation plane weighting matrix
	 * according to Raffel et al., the weight factors can be obtained
	 * by convolving the image weighting function with itself
	 */
	nPxPerWindow		= nWindowSize[0] * nWindowSize[1];
	fCorrelWeight = (float*)malloc(nPxPerWindow * sizeof(float));
	if (!fCorrelWeight) { return ERROR_NOMEM; }
	uError = convolve(fWindowWeightB, fWindowWeightB, fCorrelWeight, nWindowSize);
	if (uError)
	{
		free(fCorrelWeight);
		return uError;
	}
	
	for(n = 0; n < nPxPerWindow; ++n)
		fCorrelWeight[n] = nPxPerWindow / fCorrelWeight[n];
	

	nWindowsTotal = nWindows[0] * nWindows[1];
	
	/* Load FFTW wisdom for optimized plans */
	char wisdom_path[512];
	xcorr_cache_get_default_wisdom_path(wisdom_path, sizeof(wisdom_path));
	xcorr_cache_init(wisdom_path);

	/* fork here, parallelise */
	
	/* fork here, parallelise */
	uError = ERROR_NONE;
	// printf("  Max threads: %d\n", omp_get_max_threads());
	
	#pragma omp parallel \
        private(i, j, n, ii, jj, x, y, \
                xmin, ymin, \
                iWindowIdx, \
                fWindowA, fWindowB, fCorrelPlane, fStd, fPeakLoc, \
                fMeanA, fMeanB, fEnergyA, fEnergyB, fEnergyNorm, \
                sCCPlan) \
        shared(fImageA, fImageB, fMask, nImageSize, \
               fWinCtrsX, fWinCtrsY, nWindows, bEnsemble, \
               fCorrelWeight, fWindowWeightA, fWindowWeightB, nWindowSize, nPeaks, iPeakFinder, \
               fPkLocX, fPkLocY, fPkHeight, fSx, fSy, fSxy, \
               nWindowsTotal, nPxPerWindow, fCorrelPlane_Out) \
        default(none) \
        reduction(|:uError) \
        num_threads(omp_get_max_threads())
	{
		/* Allocate memory for correlation windows
		 * Use aligned allocation for better cache performance
		 */
		uError			= ERROR_NONE;
		fCorrelPlane	= (float*)fftwf_malloc(nPxPerWindow * sizeof(float));       
		fWindowA			= (float*)fftwf_malloc(nPxPerWindow * sizeof(float));
		fWindowB			= (float*)fftwf_malloc(nPxPerWindow * sizeof(float));
        fStd = (float*)malloc(3 * nPeaks * sizeof(float));
	    fPeakLoc			= (float*)malloc(3 * nPeaks * sizeof(float));
		    
        
		if(!fWindowA || !fWindowB || !fCorrelPlane || !fPeakLoc || !fStd)
		{
			uError		= ERROR_NOMEM;
			goto thread_cleanup;
		}
		
		/* create cross-correlation plan for this thread */
		memset(&sCCPlan, 0, sizeof(sCCPlan));
		#pragma omp critical
		{
			uError			= xcorr_create_plan(nWindowSize, &sCCPlan);
		}
		if(uError)
			goto thread_cleanup;

		/* condense to one loop to make parallelisation easier */			
		#pragma omp for schedule(static, CHUNKSIZE) nowait
		for(iWindowIdx = 0; iWindowIdx < nWindowsTotal; ++iWindowIdx)
		{
            /* Coordinate system (matching MATLAB and Python):
             * - nWindows[0] = number of windows in Y (height) direction
             * - nWindows[1] = number of windows in X (width) direction
             * - fWinCtrsY[jj] = Y-coordinate of window center (row index)
             * - fWinCtrsX[ii] = X-coordinate of window center (column index)
             * - Row-major linearization: index = jj * nWindows[1] + ii
             */
			
			/* get index in window center arrays
			 * For row-major: linearIdx = row*nCols + col
			 * Here: iWindowIdx = jj*nWindows[1] + ii
			 */
			ii			= iWindowIdx % nWindows[1];  // Column index (X)
			jj			= iWindowIdx / nWindows[1];  // Row index (Y)
            
			/* Mask uses same row-major indexing */
			int mask_idx = jj * nWindows[1] + ii;
            
            if (mask_idx < 0 || mask_idx >= nWindows[0] * nWindows[1])
            {
				uError = ERROR_OUT_OF_BOUNDS;
				//goto thread_cleanup;
            }
            // Check if the mask value at this index is 1
            if (fMask[mask_idx] == 1)
            {
                continue;  // Skip this window if the mask value is 1
            }
			
			/* Extract correlation window from images
			 * Window center coordinates (0-based array indices):
			 * - fWinCtrsX[ii] is the X-coordinate (column) of the window center
			 *   For a 128-pixel window at the left edge: center = 63.5
			 * - fWinCtrsY[jj] is the Y-coordinate (row) of the window center
			 * Window size:
			 * - nWindowSize[0] = window height (number of rows)
			 * - nWindowSize[1] = window width (number of columns)
			 * Image dimensions:
			 * - nImageSize[0] = image height (number of rows)
			 * - nImageSize[1] = image width (number of columns)
			 * 
			 * Window extraction:
			 * - For a window of size N centered at position C:
			 *   window covers pixels from floor(C - (N-1)/2 + 0.5) to floor(C + (N-1)/2 + 0.5)
			 * - Example: N=128, C=63.5 -> floor(63.5 - 63.5 + 0.5) = 0 to floor(63.5 + 63.5 + 0.5) = 127
			 */
			
			/* Calculate top-left corner of window in image coordinates */
			int row_min = (int)floor(fWinCtrsY[jj] - ((float)nWindowSize[0]-1.0)/2.0 + 0.5);
			int col_min = (int)floor(fWinCtrsX[ii] - ((float)nWindowSize[1]-1.0)/2.0 + 0.5);
			
			/* Bounds check to prevent segfault */
			if (row_min < 0 || col_min < 0 || 
			    row_min + nWindowSize[0] > nImageSize[0] || 
			    col_min + nWindowSize[1] > nImageSize[1]) {
				uError = ERROR_OUT_OF_BOUNDS;
				continue;  /* Skip this window */
			}
			
			/* Extract window pixels: iterate over window rows and columns */
			for(int row_win = 0; row_win < nWindowSize[0]; ++row_win)
			{
				int row_img = row_min + row_win;
				for(int col_win = 0; col_win < nWindowSize[1]; ++col_win)
				{
					int col_img = col_min + col_win;
					/* Row-major indexing: array[row, col] -> row*width + col */
					fWindowA[SUB2IND_2D(row_win, col_win, nWindowSize[1])] = 
						fImageA[SUB2IND_2D(row_img, col_img, nImageSize[1])];
					fWindowB[SUB2IND_2D(row_win, col_win, nWindowSize[1])] = 
						fImageB[SUB2IND_2D(row_img, col_img, nImageSize[1])];
				}
			}
			/* Pre-multiply by weighting window and compute mean 
			 * Using SIMD hints for vectorization
			 */
			fMeanA		= 0;
			fMeanB		= 0;
			#pragma omp simd reduction(+:fMeanA,fMeanB)
			for(n = 0; n < nPxPerWindow; ++n)
			{
				fWindowA[n] *= fWindowWeightA[n];
				fWindowB[n] *= fWindowWeightB[n];
				fMeanA		+= fWindowA[n];
				fMeanB		+= fWindowB[n];
			}
			fMeanA		= fMeanA / (float)nPxPerWindow;
			fMeanB		= fMeanB / (float)nPxPerWindow;
			
			/* Subtract mean and calculate signal energy for peak normalisation
			 * Using SIMD hints for vectorization
			 */
			fEnergyA		= 0;
			fEnergyB		= 0;
			if (!bEnsemble) {
				#pragma omp simd reduction(+:fEnergyA,fEnergyB)
				for(n = 0; n < nPxPerWindow; ++n)
				{
					fWindowA[n] -= fMeanA;
					fWindowB[n] -= fMeanB;
					fEnergyA 	+= fWindowA[n]*fWindowA[n];
					fEnergyB 	+= fWindowB[n]*fWindowB[n];
				}
			} else {
				#pragma omp simd reduction(+:fEnergyA,fEnergyB)
				for(n = 0; n < nPxPerWindow; ++n)
				{
					fEnergyA 	+= fWindowA[n]*fWindowA[n];
					fEnergyB 	+= fWindowB[n]*fWindowB[n];
				}
			}
			fEnergyNorm = 1 / (float)sqrt(fEnergyA * fEnergyB);

			/* Cross-correlate */
			xcorr_preplanned(fWindowB, fWindowA, fCorrelPlane, &sCCPlan);

			/* Apply correlation plane weighting with SIMD vectorization */
			if (!bEnsemble) {
				#pragma omp simd
				for (n = 0; n < nPxPerWindow; ++n)
				{
					fCorrelPlane[n] *= fCorrelWeight[n];
				}
			}
			
			
            

            
            memcpy(&fCorrelPlane_Out[nPxPerWindow * iWindowIdx], fCorrelPlane, nPxPerWindow * sizeof(float));
            
                   

			/* Call peak finder - LM solver is fully thread-safe, no locks needed */
            if (!bEnsemble) {
			    lsqpeaklocate_lm(fCorrelPlane, nWindowSize, fPeakLoc, nPeaks, iPeakFinder, fStd);
                
    
			    /* Save displacement and peak height
				 * Output arrays have shape [nPeaks, nWindows[0], nWindows[1]]
				 * where nWindows[0] = Y-windows, nWindows[1] = X-windows
				 * Peak locations from lsqpeaklocate_lm:
				 * - fPeakLoc[0, n, 3] = row offset (Y) from window center
				 * - fPeakLoc[1, n, 3] = column offset (X) from window center
				 * - fPeakLoc[2, n, 3] = peak magnitude
				 */
			    for(n = 0; n < nPeaks; ++n)
			    {
				    /* Calculate linear index for this peak and window
					 * For 3D array [nPeaks, nRows, nCols] in row-major:
					 * index = peak*nRows*nCols + row*nCols + col
					 */
				    int out_idx = n * nWindows[0] * nWindows[1] + jj * nWindows[1] + ii;
				    
				    /* Peak location in correlation plane (centered at window size/2)
					 * Subtract window center to get displacement from window center
					 * fPeakLoc dimensions: [3, nPeaks] in row-major
					 */
				    float peak_row = fPeakLoc[SUB2IND_2D(0, n, nPeaks)];  // Y-displacement
				    float peak_col = fPeakLoc[SUB2IND_2D(1, n, nPeaks)];  // X-displacement
				    float peak_mag = fPeakLoc[SUB2IND_2D(2, n, nPeaks)];
				    
				    /* Store displacements (subtract window center to get offset) */
				    fPkLocX[out_idx] = peak_col - nWindowSize[1]/2.0f;  // X is column
				    fPkLocY[out_idx] = peak_row - nWindowSize[0]/2.0f;  // Y is row
				    
				    /* Store standard deviations */
				    fSx[out_idx] = fStd[SUB2IND_2D(0, n, nPeaks)];
				    fSy[out_idx] = fStd[SUB2IND_2D(1, n, nPeaks)];
				    fSxy[out_idx] = fStd[SUB2IND_2D(2, n, nPeaks)];
    
				    /* Normalize peak height by window weight and energy content */
				    int pk_row = MIN(MAX(0, (int)peak_row), nWindowSize[0]-1);
				    int pk_col = MIN(MAX(0, (int)peak_col), nWindowSize[1]-1);
				    fPkHeight[out_idx] = peak_mag * fEnergyNorm / 
				        fCorrelWeight[SUB2IND_2D(pk_row, pk_col, nWindowSize[1])];
			    }
            }
		}
		
		/* Cleanup memory and other resources before leaving the thread */
thread_cleanup:
		#pragma omp critical
		{
			xcorr_destroy_plan(&sCCPlan);
		}
		if(fWindowA) fftwf_free(fWindowA);
		if(fStd) free(fStd);
		if(fWindowB) fftwf_free(fWindowB);
		if(fCorrelPlane) fftwf_free(fCorrelPlane);
		if(fPeakLoc) free(fPeakLoc);

	} /* end parallelised section */

	/* Save wisdom for future runs */
	xcorr_cache_save_wisdom(wisdom_path);

	return uError;
}

/* fminvec, find minimum element in vector */
float fminvec(const float *fVec, int n)
{
	int i;
	float ret;

	ret = fVec[0];
	for(i = 1; i < n; ++i)
		ret = MIN(ret, fVec[i]);

	return ret;
}

/* fmaxvec, find maximum element in vector */
float fmaxvec(const float *fVec, int n)
{
	int i;
	float ret;

	ret = fVec[0];
	for(i = 1; i < n; ++i)
		ret = MAX(ret, fVec[i]);

	return ret;
}

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

	/* Initialize FFTW threading and wisdom cache */
	fftwf_init_threads();
	fftwf_plan_with_nthreads(1);
	
	/* Load FFTW wisdom for optimized plans */
	char wisdom_path[512];
	xcorr_cache_get_default_wisdom_path(wisdom_path, sizeof(wisdom_path));
	xcorr_cache_init(wisdom_path);

	/* fork here, parallelise */
	uError = ERROR_NONE;
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
			fftwf_plan_with_nthreads(1);
			uError			= xcorr_create_plan(nWindowSize, &sCCPlan);
		}
		if(uError)
			goto thread_cleanup;

		/* condense to one loop to make parallelisation easier */			
		#pragma omp for schedule(static, CHUNKSIZE) nowait
		for(iWindowIdx = 0; iWindowIdx < nWindowsTotal; ++iWindowIdx)
		{
            
			/* get index in fWinCtrsX/Y/Z */
			ii			= iWindowIdx % nWindows[0];
			jj			= ((iWindowIdx - ii) % (nWindows[0]*nWindows[1])) / nWindows[0];
            
			int mask_idx = ii * nWindows[1] + jj;
            
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
			/* get points in correlation window 
			 * round limits to nearest integer
			 */
			xmin		= (int)floor(fWinCtrsY[ii] - ((float)nWindowSize[0]-1.0)/2 + 0.5);		
			ymin		= (int)floor(fWinCtrsX[jj] - ((float)nWindowSize[1]-1.0)/2 + 0.5);
			// printf("Window %d: ii=%d jj=%d xmin=%d, ymin=%d\n", iWindowIdx, ii, jj, xmin, ymin);
			for(j = 0, y = ymin; j < nWindowSize[1]; ++j, ++y)
			{
				for(i = 0, x = xmin; i < nWindowSize[0]; ++i, ++x)
				{
					fWindowA[SUB2IND_2D(i, j, nWindowSize[0])] = 
						fImageA[SUB2IND_2D(x, y, nImageSize[0])];
					fWindowB[SUB2IND_2D(i, j, nWindowSize[0])] = 
						fImageB[SUB2IND_2D(x, y, nImageSize[0])];
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
                
    
			    /* save displacement and peak height */
			    for(n = 0; n < nPeaks; ++n)
			    {
				    /* save peak location */
				    fPkLocX[SUB2IND_2D(n, iWindowIdx, nPeaks)] = 
					    fPeakLoc[SUB2IND_2D(0, n, 3)] - nWindowSize[0]/2;
				    fPkLocY[SUB2IND_2D(n, iWindowIdx, nPeaks)] = 
					    fPeakLoc[SUB2IND_2D(1, n, 3)] - nWindowSize[1]/2;
				    fSx[SUB2IND_2D(n, iWindowIdx, nPeaks)] = fStd[SUB2IND_2D(0, n, 3)];
				    fSy[SUB2IND_2D(n, iWindowIdx, nPeaks)] = fStd[SUB2IND_2D(1, n, 3)];
				    fSxy[SUB2IND_2D(n, iWindowIdx, nPeaks)] = fStd[SUB2IND_2D(2, n, 3)];
    
				    /* normalise peak height by window weight and energy content */
				    ii 		= MIN(MAX(0,(int)fPeakLoc[SUB2IND_2D(0,n,3)]),nWindowSize[0]-1);
				    jj 		= MIN(MAX(0,(int)fPeakLoc[SUB2IND_2D(1,n,3)]),nWindowSize[1]-1);
				    fPkHeight[SUB2IND_2D(n, iWindowIdx, nPeaks)] = 
					    fPeakLoc[SUB2IND_2D(2, n, 3)] * fEnergyNorm / fCorrelWeight[SUB2IND_2D(ii,jj,nWindowSize[0])];
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

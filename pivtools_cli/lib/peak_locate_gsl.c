#include "peak_locate_gsl.h"
#include "common.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/******************************************************************************
 * Function: lsqpeaklocate
 * ---------------------------------------------------------------------------
 * Description:
 * This function locates the peaks in a cross-correlation matrix using a 
 * least-squares fitting approach. The function finds the specified number 
 * of peak locations (`nPeaks`) and estimates the peak's position using 
 * different fitting types, depending on the specified `iFitType`. The function 
 * returns the locations of the peaks in the `peak_loc` array and optionally 
 * provides the estimated standard deviation of the peak using `std_dev`.
 *
 * The function performs a 2D search to locate the maximum points in the 
 * cross-correlation matrix (`xcorr`). For each peak, the function applies 
 * least-squares fitting to refine the location estimate.
 *
 * The method of fitting is determined by the `iFitType` parameter, which 
 * specifies different fitting models based on the number of degrees of freedom 
 * (DOF).
 *
 * ---------------------------------------------------------------------------
 * Inputs:
 *   float *xcorr      : Pointer to the 2D array representing the cross-correlation 
 *                       matrix. The matrix should be of size [N[0], N[1]] where
 *                       N[0] is the number of rows and N[1] is the number of columns.
 *
 *   int *N            : Pointer to an array of size 2 where:
 *                       - N[0] : The number of rows in the xcorr matrix.
 *                       - N[1] : The number of columns in the xcorr matrix.
 *   
 *   int nPeaks        : The number of peaks to locate in the cross-correlation matrix.
 *                       Should be a positive integer (e.g., 1 for a single peak).
 *
 *   int iFitType      : Specifies the type of least-squares fit to use for peak 
 *                       localization. Valid options include:
 *                       - 3 : Use a 3-point least-squares estimator.
 *                       - 4 : Use a 4-DOF Gaussian fit.
 *                       - 5 : Use a 5-DOF Gaussian fit.
 *                       - 6 : Use a 6-DOF Gaussian fit.
 *
 * ---------------------------------------------------------------------------
 * Outputs:
 *   float *peak_loc   : Pointer to an array where the locations of the peaks 
 *                       will be stored. It must be able to hold `nPeaks * 2` elements 
 *                       (x, y coordinates for each peak).
 *
 *   float *std_dev    : Pointer to a variable where the standard deviation of 
 *                       the peak location will be stored (optional, can be NULL).
 *
 * ---------------------------------------------------------------------------
 * Procedures:
 *   1. Parse the cross-correlation matrix and search for the highest values (peaks).
 *   2. For each identified peak, refine the location using the least-squares 
 *      fitting method based on the `iFitType` parameter.
 *   3. Store the estimated peak locations in the `peak_loc` array.
 *   4. Optionally, compute the standard deviation of the peak location and 
 *      store it in `std_dev`.
 *
 * ---------------------------------------------------------------------------
 * Example Usage:
 *   float xcorr[10][10];     // Cross-correlation matrix of size 10x10
 *   int N[2] = {10, 10};      // N[0] = 10 (rows), N[1] = 10 (columns)
 *   float peak_loc[2];        // Array to store peak locations (x, y)
 *   int nPeaks = 1;           // Looking for 1 peak
 *   int iFitType = 4;         // Use 4-point Gaussian fitting
 *   float std_dev;            // Variable to store the peak standard deviation
 *   
 *   lsqpeaklocate(&xcorr[0][0], N, peak_loc, nPeaks, iFitType, &std_dev);
 *   printf("Peak at location: (%f, %f)\n", peak_loc[0], peak_loc[1]);
 *   printf("Peak standard deviation: %f\n", std_dev);
 * 
 *****************************************************************************/

void lsqpeaklocate(const float *xcorr, const int *N, float *peak_loc, int nPeaks, int iFitType, float *std_dev)
{
	int i, j, iPeak, idx;
	int i0, j0;
	float *xcorr_copy;
	float fPeakHeight;
	float subxcorr[PKSIZE_X][PKSIZE_Y];
	float fitval[PKSIZE_X][PKSIZE_Y];
	int Nsub[2];
	float peak[2];
    float sig[3];
   
	/* make a copy of xcorr that we can manipulate */
	xcorr_copy		= (float*)malloc(sizeof(float) * N[0]*N[1]);
	memcpy(xcorr_copy, xcorr, N[0]*N[1]*sizeof(float));
	Nsub[0]			= PKSIZE_X;
	Nsub[1]			= PKSIZE_Y;
	
	/* iterate over peaks */
	for(iPeak = 0; iPeak < nPeaks; ++iPeak)
	{
		/* find maximum value in array
		 * only search for peaks in range
		 * N[0]/8 <= i < N[0]*7/8
		 * N[1]/8 <= j < N[1]*7/8
		 */
		i0 			= j0 = 0;
		fPeakHeight = 0;
		for(i = N[0]/8; i < N[0]*7/8; ++i)
		{
			for(j = N[1]/8; j < N[1]*7/8; ++j)
			{
				if(xcorr_copy[SUB2IND_2D(i, j, N[0])] > fPeakHeight)
				{
					fPeakHeight	= xcorr_copy[SUB2IND_2D(i, j, N[0])];
					i0				= i;
					j0				= j;
				}
			}
		}

		/* error out if peak height is not positive */
		if(fPeakHeight <= 0)
		{
			peak_loc[SUB2IND_2D(0, iPeak, 3)] = NAN;
			peak_loc[SUB2IND_2D(1, iPeak, 3)] = NAN;
			peak_loc[SUB2IND_2D(2, iPeak, 3)] = 0;
			continue;
		}

		/* error out if too close to the edges, or is not a local maximum */
		if(	i0 < (PKSIZE_X-1)/2 || i0 >= N[0]-(PKSIZE_X-1)/2  
			|| j0 < (PKSIZE_Y-1)/2 || j0 >= N[1]-(PKSIZE_Y-1)/2 
			|| fPeakHeight <= xcorr_copy[SUB2IND_2D(i0-1, j0  , N[0])] 
			|| fPeakHeight <= xcorr_copy[SUB2IND_2D(i0+1, j0  , N[0])] 
			|| fPeakHeight <= xcorr_copy[SUB2IND_2D(i0  , j0-1, N[0])] 
			|| fPeakHeight <= xcorr_copy[SUB2IND_2D(i0  , j0+1, N[0])] )
		{
			peak_loc[SUB2IND_2D(0, iPeak, 3)] = NAN;
			peak_loc[SUB2IND_2D(1, iPeak, 3)] = NAN;
			peak_loc[SUB2IND_2D(2, iPeak, 3)] = 0;
			continue;
		}

		/* copy into C-style matrix */
		for(i = 0; i < PKSIZE_X; ++i)
		{
			for(j = 0; j < PKSIZE_Y; ++j)
			{
				subxcorr[i][j] = xcorr_copy[SUB2IND_2D(i0 + i - (PKSIZE_X-1)/2, j0 + j - (PKSIZE_Y-1)/2, N[0])];
			}
		}

		/* perform least-squares fit to get peak location
		 * only bother using high-order scheme for first peak
		 */
		peakfit(&subxcorr[0][0], Nsub, peak, &fitval[0][0], iPeak ? 3 : iFitType, sig);

		/* save peak location and subtract fit from correlation plane 
		 * note this could mean that the correlation plane is not strictly positive-valued everywhere
		 */

		peak_loc[SUB2IND_2D(0, iPeak, 3)] = peak[0] + i0;
		peak_loc[SUB2IND_2D(1, iPeak, 3)] = peak[1] + j0;
		peak_loc[SUB2IND_2D(2, iPeak, 3)] = fPeakHeight;
		std_dev[SUB2IND_2D(0, iPeak, 3)] = sig[0];
		std_dev[SUB2IND_2D(1, iPeak, 3)] = sig[1];
		std_dev[SUB2IND_2D(2, iPeak, 3)] = sig[2];
		for(i = 0; i < PKSIZE_X; ++i)
		{
			for(j = 0; j < PKSIZE_Y; ++j)
			{
				idx 			= SUB2IND_2D( i0 + i - (PKSIZE_X-1)/2, j0 + j - (PKSIZE_Y-1)/2, N[0]);
				xcorr_copy[idx] = MAX(0, xcorr_copy[idx] - fitval[i][j]);
			}
		}
	}

	/* clean up and exit */
	free(xcorr_copy);
	return;
}

/****************************************************
 * void peakfit(const float *xcorr, const int *N, float *peak_loc, float *fitval, int iFitType)
 * 
 * finds the sub-grid location of the nPeaks largest correlation-peaks in 
 * the two-dimensional array xcorr
 *
 * xcorr is a two-dimensional array with an odd number of elements along each dimension
 * typically five
 * such that the element at the centre is at i = j = 0
 * 
 * the sub-grid location is returned in peak_loc in this coordinate system
 * the best fit function is returned in fitval for the fit parameters which best match the correlation peak 
 *
 * the fit type to be used (3, 4, 5 or 6 degree of freedom) is specified by iFitType
 *
 */

void peakfit(const float *xcorr, const int *N, float *peak_loc, float *fitval, int iFitType, float *sig)
{
	float x_fit[3];
	float y_fit[3];
	int i;
	int info, status;
	float A, sx, sy, sx_lb, sy_lb, sx_ub, sy_ub, A_lb, A_ub;
	
	size_t n_data;
	double X[6];
	pkdata pd;
	gsl_vector 								*gsl_F, *gsl_X;
	gsl_vector_view 						gsl_X0;

	/* make initial guess of sub-pixel location and gaussian peak size
	 * using three-point estimator
	 */
	for(i = 0; i < 3; ++i)
	{
		x_fit[i] = xcorr[	  (i - 1 + (N[0]-1)/2)	*N[1]
								+ (N[1]-1)/2                 ];
		y_fit[i] = xcorr[	  (N[0]-1)/2				*N[1] 
								+ (i - 1 + (N[1]-1)/2)	     ];
		x_fit[i] = (float)log((x_fit[i] < FLT_EPSILON) ? FLT_EPSILON : x_fit[i]);
		y_fit[i] = (float)log((y_fit[i] < FLT_EPSILON) ? FLT_EPSILON : y_fit[i]);
	}

	/* peak location in i direction is at i0 = numer/denom
	 * numer =   ln(R(i-1), j)                  - ln(R(i+1), j)
	 *      ~= -4 (i0 - i) / sigma_x^2
	 * denom =  2ln(R(i-1, j)) - 4ln(R(i,j)) + 2ln(R(i+1,j))
	 *      ~= -4 / sigma_x^2
	 * when R is modelled as a gaussian R ~ exp(-(i-i0)/sigma_x^2)
	 */
	peak_loc[0] = (x_fit[0] - x_fit[2]) / (2*x_fit[0] - 4*x_fit[1] + 2*x_fit[2]);
	peak_loc[1] = (y_fit[0] - y_fit[2]) / (2*y_fit[0] - 4*y_fit[1] + 2*y_fit[2]);
	A				= xcorr[(N[0]-1)/2*N[1] + (N[1]-1)/2];
	sx				= (float)sqrt(-4 / (2*x_fit[0] - 4*x_fit[1] + 2*x_fit[2]));
	sy				= (float)sqrt(-4 / (2*y_fit[0] - 4*y_fit[1] + 2*y_fit[2]));
	sx_lb			= 0.25;		sy_lb			= 0.25;
	sx_ub			= 2*sx;		sy_ub			= 2*sy;
	A_lb			= A / 2;
	A_ub			= A * 2;

	/***** allocate default result for 3-point fit *****/
	X[0] 		= A;
	X[1] 		= peak_loc[0];
	X[2] 		= peak_loc[1];
	X[3]		= sqrt(sx*sx+sy*sy);
	n_data 	= N[0] * N[1];

	// evaluate fit function and copy
	memset(fitval, 0, sizeof(float)*n_data);
	pd.xcorr = fitval; // hand gauss4_f a zero array to evaluate fit function only
	pd.N 		= N;
	gsl_F 	= gsl_vector_alloc(n_data);
	gsl_X0	= gsl_vector_view_array(X, 4);
	gauss4_f(&(gsl_X0.vector), (void*)&pd, gsl_F);
	for(i = 0; i < n_data; ++i)
		fitval[i] = (float) gsl_vector_get(gsl_F, i);
	gsl_vector_free(gsl_F);

	/* if using three point estimator fit, stop here */
	if(iFitType == 3)
		return;

	/***** set-up for non-linear least squares fitting *****/
	const gsl_multifit_nlinear_type 	*gsl_fittype = gsl_multifit_nlinear_trust;
	gsl_multifit_nlinear_workspace 	*gsl_ws;
	gsl_multifit_nlinear_fdf 			gsl_fdf;
	gsl_multifit_nlinear_parameters 	gsl_fdf_params = gsl_multifit_nlinear_default_parameters();

	const double 							xtol = 1e-6;
	const double 							gtol = 1e-6;
	const double 							ftol = 0.0;

	/* set common parameters for all types of fit */
	pd.N 					= N;
	pd.xcorr 			= xcorr;
	gsl_fdf.fvv 		= NULL; 			// not using geodesic acceleration
	gsl_fdf.n 			= N[0]*N[1]; 	// number of data
	gsl_fdf.params 	=&pd; 			// data for optimisation
	gsl_fdf_params.trs= gsl_multifit_nlinear_trs_lm;
	
	/* specific settings for different fit types */
	switch(iFitType)
	{
	case 4:
		/* four DOF fit */
		/* initial guess */
		X[0]		= A;
		X[1]		= peak_loc[0];
		X[2]		= peak_loc[1];
		X[3]		= sqrt(sx*sx+sy*sy);
		gsl_X0	= gsl_vector_view_array(X, 4);

		/* define the function to be minimized */
		gsl_fdf.f 	= gauss4_f; 	// objective function to minimise
		gsl_fdf.df 	= gauss4_jac; 	// jacobian
		gsl_fdf.p 	= 4; 				// number of parameters
		break;
	case 5:
		/* five DOF fit */
		/* initial guess */
		X[0] 		= A;
		X[1] 		= peak_loc[0];
		X[2] 		= peak_loc[1];
		X[3] 		= sx;
		X[4] 		= sy;
		gsl_X0 	= gsl_vector_view_array(X, 5);

		/* define the function to be minimized */
		gsl_fdf.f 	= gauss5_f; 	// objective function to minimise
		gsl_fdf.df 	= gauss5_jac; 	// jacobian
		gsl_fdf.p 	= 5; 				// number of parameters
		break;
	default:
	case 6:
		/* six DOF fit */
		/* initial guess */
        
		X[0] 		= A;
		X[1] 		= peak_loc[0];
		X[2] 		= peak_loc[1];
		X[3] 		= sx*sx;
		X[4] 		= sy*sy;
		X[5] 		= 0;
		gsl_X0 	= gsl_vector_view_array(X, 6);

		/* define the function to be minimized */
		gsl_fdf.f 	= gauss6_f; 	// objective function to minimise
		gsl_fdf.df 	= gauss6_jac; 	// jacobian
		gsl_fdf.p 	= 6; 				// number of parameters
		break;
	}

	/**** run solver ****/
	// allocate workspace with default parameters
	gsl_ws = gsl_multifit_nlinear_alloc(gsl_fittype, &gsl_fdf_params, gsl_fdf.n, gsl_fdf.p);
	// initialize solver with starting point and weights
	gsl_multifit_nlinear_init(&(gsl_X0.vector), &gsl_fdf, gsl_ws);
	// solve the system with a maximum of MAXITER iterations
	gsl_X = gsl_multifit_nlinear_position(gsl_ws);

	// note: fitting can crash with a memory access error if nan or bad values are returned by any of the fitting functions!
	status = gsl_multifit_nlinear_driver(MAXITER, xtol, gtol, ftol, NULL, NULL, &info, gsl_ws);

	/**** allocate result if successful ****/
	if(status == GSL_SUCCESS)
    //extract std 210224a
	
	{
		gsl_X = gsl_multifit_nlinear_position(gsl_ws);
		if (gsl_fdf.p == 6) {
            sig[0] = (float)gsl_vector_get(gsl_X,4);
            sig[1] = (float)gsl_vector_get(gsl_X,3);
			sig[2] = -(float)gsl_vector_get(gsl_X, 5)*(sig[0]*sig[1]-((float)gsl_vector_get(gsl_X, 5)*(float)gsl_vector_get(gsl_X, 5)));
        } else {
            sig[0] = 0.0;
            sig[1] = 0.0;
            sig[2] = 0.0;
        }
		
		peak_loc[0]		= (float)gsl_vector_get(gsl_X, 1);
		peak_loc[1]		= (float)gsl_vector_get(gsl_X, 2);
		// evaluate fit function and copy
		memset(fitval, 0, sizeof(float)*n_data);
		pd.xcorr 		= fitval; // necessary
		pd.N 				= N;
		gsl_F 			= gsl_vector_alloc(n_data);
		(*gsl_fdf.f)(gsl_X, (void*)&pd, gsl_F);
		for(i = 0; i < n_data; ++i)
			fitval[i] = (float) gsl_vector_get(gsl_F, i);
		gsl_vector_free(gsl_F);
	}

	/**** free ****/
	gsl_multifit_nlinear_free(gsl_ws);

	return;
}

/****************************************************
 * fitting functions
 *
 * we provide the function which predicts 
 * the measurement vector x of dimension n x 1 that
 * x = F(i, j; p)
 * where p is the vector to be predicted with dimension m x 1 
 *
 * the jacobian jac = \partial x / \partial p is organised in a C-style
 * n x m matrix, such that 
 * jac[0]			corresponds to \partial x_0 / \partial p_0
 * jac[1]				"        "	\partial x_0 / \partial p_1
 * jac[i*m + j]		"			"	\partial x_i / \partial p_j
 *
 * x represents a flattened C-style N[0] x N[1] array indexed by
 * x[i][j], i = -(N[0]-1)/2 .. (N[0]-1)/2, j = -(N[1]-1)/2 ...
 * and should contain F(i, j; p) evaluated at each i, j
 *
 * data is a pointer to an int array (int*) which contains the
 * dimensions that the flattened array x corresponds to
 */

/********
 * gauss4 and gauss4jac
 * four-DOF gaussian fit
 *
 * F(i, j) = A * exp( -((i - i0)^2 + (j - j0)^2) / sigma^2)
 * 
 * p = [A i0 j0 sigma]
 */
int gauss4_f(const gsl_vector *p, void *data, gsl_vector *res)
{
	const int *N;
	int ii, jj, xidx;
	float i0, j0, A, s, i, j, ihat, jhat, F;
	const float *f;

	/* extract parameters */
	N 		= ((pkdata*)data)->N;
	f 		= ((pkdata*)data)->xcorr;
	A		= (float)gsl_vector_get(p, 0);
	i0		= (float)gsl_vector_get(p, 1);
	j0		= (float)gsl_vector_get(p, 2);
	s		= (float)gsl_vector_get(p, 3);

	/* F(i, j) = A * exp( -((i - i0)^2 + (j - j0)^2) / s^2 ) */
	for(ii = 0; ii < N[0]; ++ii)
	{
		i = (float)(ii - (N[0]-1)/2);
		for(jj = 0; jj < N[1]; ++jj)
		{
			j = (float)(jj - (N[1]-1)/2);
			xidx	= ii*N[1] + jj;
			ihat  = (i-i0)/s;
			jhat  = (j-j0)/s;
			F		= A * expf( -ihat*ihat - jhat*jhat );
			gsl_vector_set(res, xidx, F - f[xidx]);
		}
	}

	return GSL_SUCCESS;
}

int gauss4_jac(const gsl_vector *p, void *data, gsl_matrix *jac)
{
	const int *N;
	int ii, jj, xidx;
	float i0, j0, A, s, i, j, ihat, jhat, F;

	/* extract parameters */
	N 		= ((pkdata*)data)->N;
	A		= (float)gsl_vector_get(p, 0);
	i0		= (float)gsl_vector_get(p, 1);
	j0		= (float)gsl_vector_get(p, 2);
	s		= (float)gsl_vector_get(p, 3);

	/* F(i, j)	= A * exp( -((i - i0)^2 + (j - j0)^2) / (s*s)) 
	 * dF / dA		= F / A
	 * dF / di0		= 2*F*(i - i0)/s^2
	 * dF / dj0		= 2*F*(j - j0)/s^2
	 * dF / ds		= 2*((i - i0)^2 + (j - j0)^2)/s^3
	 */
	for(ii = 0; ii < N[0]; ++ii)
	{
		i = (float)(ii - (N[0]-1)/2);
		for(jj = 0; jj < N[1]; ++jj)
		{
			j		= (float)(jj - (N[1]-1)/2);
			xidx	= ii*N[1] + jj;
			ihat  = (i-i0)/s;
			jhat  = (j-j0)/s;
			F		= A * expf( -ihat*ihat - jhat*jhat );

			gsl_matrix_set(jac, xidx, 0, F / A); 								/* dF / dA	*/
			gsl_matrix_set(jac, xidx, 1, 2*F*ihat/s);							/* dF / di0 */
			gsl_matrix_set(jac, xidx, 2, 2*F*jhat/s);							/* dF / dj0 */
			gsl_matrix_set(jac, xidx, 3, 2*F*(ihat*ihat+jhat*jhat)/s);	/* dF / ds	*/
		}
	}

	return GSL_SUCCESS;
}

int gauss4_fvv(const gsl_vector *p, const gsl_vector *v, void *data, gsl_vector *fvv)
{
	const int *N;
	int ii, jj, xidx;
	float i0, j0, A, s, i, j, F, ss, ihat, jhat;
	float vA, vi, vj, vs;
	float DAi, DAj, DAs, Dii, Dij, Dis, Djj, Djs, Dss, D2v;
	const float *f;

	/* extract parameters */
	N 		= ((pkdata*)data)->N;
	A		= (float)gsl_vector_get(p, 0);
	i0		= (float)gsl_vector_get(p, 1);
	j0		= (float)gsl_vector_get(p, 2);
	s		= (float)gsl_vector_get(p, 3);
	/* velocity */
	vA 	= (float)gsl_vector_get(v, 0);
	vi 	= (float)gsl_vector_get(v, 1);
	vj 	= (float)gsl_vector_get(v, 2);
	vs 	= (float)gsl_vector_get(v, 3);

	/* F(i, j) = A * exp( -((i - i0)^2 + (j - j0)^2) / s^2 ) */
	for(ii = 0; ii < N[0]; ++ii)
	{
		i = (float)(ii - (N[0]-1)/2);
		for(jj = 0; jj < N[1]; ++jj)
		{
			j 		= (float)(jj - (N[1]-1)/2);
			xidx	= ii*N[1] + jj;

			/* calculate second derivatives */
			ihat 	= (i - i0)/s;
			jhat 	= (j - j0)/s;
			

			DAi 	= 2.0*ihat/(s*A);
			DAj 	= 2.0*jhat/(s*A);
			DAs 	= 2.0*(ihat*ihat + jhat*jhat)/(s*A);
			Dii 	= 2.0*(ihat*ihat - 1.0)/(s*s);
			Dij 	= 4.0*ihat*jhat/(s*s);
			Dis 	= 4.0*ihat*(ihat*ihat + jhat*jhat - 1.0)/(s*s);
			Djj 	= 2.0*(jhat*jhat - 1.0);
			Djs 	= 4.0*jhat*(ihat*ihat + jhat*jhat - 1.0)/(s*s);
			Dss 	= 4.0*(ihat*ihat + jhat*jhat)*(ihat*ihat + jhat*jhat - 1.5)/(s*s);

			F 		= A * expf(-ihat*ihat - jhat*jhat);
			D2v 	= F*(0 + 2*DAi*vA*vi + 2*DAj*vA*vj + 2*DAs*vA*vs 
					+ Dii*vi*vi + 2*Dij*vi*vj + 2*Dis*vi*vs 
					+ Djj*vj*vj + 2*Djs*vj*vs + Dss*vs*vs);
			
			gsl_vector_set(fvv, xidx, D2v);
		}
	}

	return GSL_SUCCESS;
}

/********
 * gauss5 and gauss5jac
 * five-DOF gaussian fit
 *
 * F(i, j) = A * exp( -(i - i0)^2*sxx - (j - j0)^2*syy)
 * 
 * p = [A i0 j0 sxx syy]
 * (sxx = 1/sigma_x^2)
 */
int gauss5_f(const gsl_vector *p, void *data, gsl_vector *res)
{
	const int *N;
	int ii, jj, xidx;
	float i0, j0, A, sx, sy, i, j, ihat, jhat, F;
	const float *f;

	/* extract parameters */
	N 		= ((pkdata*)data)->N;
	f 		= ((pkdata*)data)->xcorr;
	A		= (float)gsl_vector_get(p, 0);
	i0		= (float)gsl_vector_get(p, 1);
	j0		= (float)gsl_vector_get(p, 2);
	sx		= (float)gsl_vector_get(p, 3);
	sy 	= (float)gsl_vector_get(p, 4);

	/* F(i, j) = A * exp( -(i - i0)^2/sx^2 - (j - j0)^2/sy^2) */
	for(ii = 0; ii < N[0]; ++ii)
	{
		i = (float)(ii - (N[0]-1)/2);
		for(jj = 0; jj < N[1]; ++jj)
		{
			j 		= (float)(jj - (N[1]-1)/2);
			xidx	= ii*N[1] + jj;
			ihat  = (i-i0)/sx;
			jhat  = (j-j0)/sy;
			F		= A * expf( -ihat*ihat - jhat*jhat );
			gsl_vector_set(res, xidx, F - f[xidx]);
		}
	}

	return GSL_SUCCESS;
}

int gauss5_jac(const gsl_vector *p, void *data, gsl_matrix *jac)
{
	const int *N;
	int ii, jj, xidx;
	float i0, j0, A, sx, sy, i, j, ihat, jhat, F;

	/* extract parameters */
	N 		= ((pkdata*)data)->N;
	A		= (float)gsl_vector_get(p, 0);
	i0		= (float)gsl_vector_get(p, 1);
	j0		= (float)gsl_vector_get(p, 2);
	sx		= (float)gsl_vector_get(p, 3);
	sy 	= (float)gsl_vector_get(p, 4);

	/* F(i, j)	= A * exp( -(i - i0)^2*sxx - (j - j0)^2*syy)
	 * dF / dA		= F / A
	 * dF / di0		= 2*F*(i - i0)/sx^2
	 * dF / dj0		= 2*F*(j - j0)/sy^2
	 * dF / dsx		= 2*F*(i - i0)^2/sx^3
	 * dF / dsy		= 2*F*(j - j0)^2/sy^3
	 */
	for(ii = 0; ii < N[0]; ++ii)
	{
		i = (float)(ii - (N[0]-1)/2);
		for(jj = 0; jj < N[1]; ++jj)
		{
			j 		= (float)(jj - (N[1]-1)/2);
			xidx 	= ii*N[1] + jj;
			ihat  = (i-i0)/sx;
			jhat  = (j-j0)/sy;
			F		= A * expf( -ihat*ihat - jhat*jhat );
			
			gsl_matrix_set(jac, xidx, 0, F / A);					/* dF / dA		*/
			gsl_matrix_set(jac, xidx, 1, 2*F*ihat/sx);			/* dF / di0		*/
			gsl_matrix_set(jac, xidx, 2, 2*F*jhat/sy);			/* dF / dj0		*/
			gsl_matrix_set(jac, xidx, 3, 2*F*ihat*ihat/sx);		/* dF / dsx	*/
			gsl_matrix_set(jac, xidx, 4, 2*F*jhat*jhat/sy);		/* dF / dsy	*/
		}
	}

	return GSL_SUCCESS;
}


/********
 * gauss6 and gauss6jac
 * six-DOF gaussian fit
 *
 * F(i, j) = A * exp( -(i - i0)*(i - i0)/sx^2 -(j - j0)*(j - j0)/sy^2 
 *							 -(i - i0)*(j - j0)*sxy)
 * 
 * p = [A i0 j0 sxx syy sxy]
 * (sxx = 1/sigma_x^2 etc.)
 */
int gauss6_f(const gsl_vector *p, void *data, gsl_vector *res)
{
	const int *N;
	int ii, jj, xidx;
	float i0, j0, A, sx, sy, sxy, i, j, F;
	const float *f;

	/* extract parameters */
	N 		= ((pkdata*)data)->N;
	f 		= ((pkdata*)data)->xcorr;
	A		= (float)gsl_vector_get(p, 0);
	i0		= (float)gsl_vector_get(p, 1);
	j0		= (float)gsl_vector_get(p, 2);
	sx		= (float)gsl_vector_get(p, 3);
	sy 	= (float)gsl_vector_get(p, 4);
	sxy 	= (float)gsl_vector_get(p, 5);

	/*  F(i, j) =A*exp(-0.5*((((j-xbar)^2*sy)-(2*(j-xbar)*(i-ybar)*sxy)+(i-ybar)^2*sx))/(sx*sy-sxy^2));
	 */
	for(ii = 0; ii < N[0]; ++ii)
	{
		i = (float)(ii - (N[0]-1)/2); /* goes from -N/2 to N/2 */
		for(jj = 0; jj < N[1]; ++jj)
		{
			j 		= (float)(jj - (N[1]-1)/2);
			xidx 	= ii*N[1] + jj;
            F 		= A * expf(0.5*( -(i-i0)*(i-i0)/(sx) 
									-(j-j0)*(j-j0)/(sy) 
									-2*(i-i0)*(j-j0)* sxy));
			gsl_vector_set(res, xidx, F - f[xidx]);
		}
	}

	return GSL_SUCCESS;
}

int gauss6_jac(const gsl_vector *p, void *data, gsl_matrix *jac)
{
	const int *N;
	int ii, jj, xidx;
	float i0, j0, A, sx, sy, sxy, i, j, F, ihat, jhat;

	/* extract parameters */
	N 		= ((pkdata*)data)->N;
	A		= (float)gsl_vector_get(p, 0);
	i0		= (float)gsl_vector_get(p, 1);
	j0		= (float)gsl_vector_get(p, 2);
	sx		= (float)gsl_vector_get(p, 3);
	sy 	= (float)gsl_vector_get(p, 4);
	sxy 	= (float)gsl_vector_get(p, 5);

	for(ii = 0; ii < N[0]; ++ii)
	{
		i = (float)(ii - (N[0]-1)/2);
		for(jj = 0; jj < N[1]; ++jj)
		{
			j 		= (float)(jj - (N[1]-1)/2);
			xidx 	= ii*N[1] + jj;
            F 		= A * expf(0.5*( -(i-i0)*(i-i0)/(sx) 
									-(j-j0)*(j-j0)/(sy) 
									-2*(i-i0)*(j-j0)* sxy));
            
            gsl_matrix_set(jac, xidx, 0, F / A);											/* dF / dA		*/
			gsl_matrix_set(jac, xidx, 1, -(F*(i0 + (j0 - j) * sx * sxy - i))/(sx));	/* dF / di0		*/
			gsl_matrix_set(jac, xidx, 2, -(F*(j0 + (i0 - i) * sy * sxy - j))/(sy));	/* dF / dj0		*/	
			gsl_matrix_set(jac, xidx, 3, (F*(i-i0)*(i-i0))/(2*sx*sx));				/* dF / dsx	*/
			gsl_matrix_set(jac, xidx, 4, (F*(j-j0)*(j-j0))/(2*sy*sy));				/* dF / dsy	*/
			gsl_matrix_set(jac, xidx, 5, -(i-i0)*(j-j0)*F);								/* dF / dsxy	*/
		}
	}

	return GSL_SUCCESS;
}

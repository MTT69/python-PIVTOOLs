#ifndef PEAK_LOCATE_H
#define PEAK_LOCATE_H

// Gnu Science Library includes
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>

/**** defines ****/
#define PKSIZE_X						5
#define PKSIZE_Y						5
#define PKSIZE_Z						5
#define MAXITER							100

/*** structures ***/
typedef struct _pkdata
{
    const int *N;
    const float *xcorr;
} pkdata;

/**** function prototypes ****/
//add std 210224
void lsqpeaklocate(const float *xcorr, const int *N, float *peak_loc, int nPeaks, int iFitType, float *std_dev);

void peakfit(const float *xcorr, const int *N, float *peak_loc, float *fitval, int iFitType, float *sig);

int gauss4_f(const gsl_vector *x, void *data, gsl_vector *res);
int gauss4_jac(const gsl_vector *x, void *data, gsl_matrix *jac);
int gauss4_fvv(const gsl_vector *p, const gsl_vector *v, void *data, gsl_vector *fvv);

int gauss5_f(const gsl_vector *x, void *data, gsl_vector *res);
int gauss5_jac(const gsl_vector *x, void *data, gsl_matrix *jac);
int gauss6_f(const gsl_vector *x, void *data, gsl_vector *res);
int gauss6_jac(const gsl_vector *x, void *data, gsl_matrix *jac);

#endif

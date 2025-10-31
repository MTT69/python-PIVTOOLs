#ifndef _INTERP2_CUSTOM_H
#define _INTERP2_CUSTOM_H

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

/**** defines ****/
#define PI 3.14159265f
#define CHUNKSIZE 1024
#define NUMTHREADS_MAX omp_get_max_threads()

#define DEFAULT_KERNEL_HALF_SIZE 10
#define DEFAULT_KERNEL_SIZE (2*DEFAULT_KERNEL_HALF_SIZE)
#define KERNEL_SIZE (2*g_iKernelHalfSize)
#define KERNEL_HALF_SIZE g_iKernelHalfSize
#define NLUT 32768

#define KERNEL_LANCZOS 0 
#define KERNEL_GAUSSIAN 1

#define SUB2IND_2D(i, j, M) ((j)*M + (i))
#define MIN(A, B) (((A)<(B))?(A):(B))
#define MAX(A, B) (((A)>(B))?(A):(B))

/**** global variables ****/
extern int g_iLUTIn_interptialised;
extern int g_iKernelHalfSize;
extern int g_iKernelType;
extern float g_fGaussKernelStd;
extern float *g_fLUT;

/**** function declarations ****/
EXPORT void interp2custom(const float *y, size_t *N, const float *f_i, const float *f_j, float *yi, int n_interp);
EXPORT float interp1custom(float *y, float *fFilterCoeffs);
EXPORT void interp1custom_vec(float *y, float *yi, int N, float *fFilterCoeffs);
EXPORT void interp1custom_generatelut(int iKernelType, int iKernelSize, float fOptions);
EXPORT void interp1custom_destroylut(void);
EXPORT float sinc(float x);

#endif

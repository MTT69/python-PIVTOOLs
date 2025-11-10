#include "interp2custom.h"

/**** global variables ****/
int g_iLUTIn_interptialised = 0;
int g_iKernelHalfSize = 5;
int g_iKernelType = 0;
float g_fGaussKernelStd = 0.65f;
float *g_fLUT = NULL;

/**** sinc helper ****/
float sinc(float x) {
    return x == 0.0f ? 1.0f : sinf(PI*x)/(PI*x);
}

/**** 1D interpolation ****/
float interp1custom(float *y, float *fFilterCoefficients) {
    int m;
    float yi = 0;
    #pragma omp simd
    for(m = 0; m < KERNEL_SIZE; ++m) {
        yi += y[m] * fFilterCoefficients[m];
    }
    return yi;
}

/**** vectorized 1D interpolation ****/
void interp1custom_vec(float *y, float *yi, int N, float *fFilterCoefficients) {
    int m, n;
    memset(yi, 0, sizeof(float) * N);
    for(m = 0; m < KERNEL_SIZE; ++m) {
        #pragma omp simd
        for(n = 0; n < N; ++n) {
            yi[n] += y[m*N + n] * fFilterCoefficients[m];
        }
    }
}

/**** generate LUT ****/
void interp1custom_generatelut(int iKernelType, int iKernelSize, float fOptions) {
    int m, n;
    float delta, A;
    int iKernelHalfSize = iKernelSize / 2;
    int bRegenerate = !g_iLUTIn_interptialised || iKernelType != g_iKernelType || iKernelHalfSize != g_iKernelHalfSize;
    if(iKernelType == KERNEL_GAUSSIAN)
        bRegenerate = bRegenerate || fOptions != g_fGaussKernelStd;

    if(!bRegenerate) return;

    if(g_iLUTIn_interptialised && g_fLUT) free(g_fLUT);
    g_fLUT = (float*)malloc(NLUT * iKernelSize * sizeof(float));

    switch(iKernelType) {
        case KERNEL_LANCZOS:
            for(n = 1; n < NLUT; ++n) {
                delta = ((float)n)/((float)NLUT);
                for(m = -iKernelHalfSize+1; m <= iKernelHalfSize; ++m) {
                    g_fLUT[n * iKernelSize + m + iKernelHalfSize - 1] = sinc(delta - m) * sinc((delta - m)/(float)iKernelHalfSize);
                }
            }
            memset(g_fLUT, 0, sizeof(float)*iKernelSize);
            g_fLUT[iKernelHalfSize-1] = 1;
            break;
        case KERNEL_GAUSSIAN:
            for(n = 0; n < NLUT; ++n) {
                delta = ((float)n)/((float)NLUT);
                A = 0;
                for(m = -iKernelHalfSize+1; m <= iKernelHalfSize; ++m) {
                    g_fLUT[n * iKernelSize + m + iKernelHalfSize - 1] = expf(-powf((m-delta)/fOptions, 2));
                    A += g_fLUT[n * iKernelSize + m + iKernelHalfSize - 1];
                }
                A = 1/A;
                for(m = -iKernelHalfSize+1; m <= iKernelHalfSize; ++m)
                    g_fLUT[n * iKernelSize + m + iKernelHalfSize - 1] *= A;
            }
            break;
    }
    g_iKernelType = iKernelType;
    g_fGaussKernelStd = fOptions;
    g_iKernelHalfSize = iKernelHalfSize;
    g_iLUTIn_interptialised = 1;
}

/**** destroy LUT ****/
void interp1custom_destroylut(void) {
    if(g_iLUTIn_interptialised && g_fLUT) {
        free(g_fLUT);
        g_fLUT = NULL;
    }
}

/**** 2D interpolation ****/
void interp2custom(const float *y, size_t *N, const float *f_i, const float *f_j, float *yi, int n_interp) {
    int n, i, j;
    float delta[2];
    int m[2];
    float *fFilterCoefficients[2];
    float *yi_stage0, *yi_stage1;
    int i_min, i_max, j_min, j_max;

    #pragma omp parallel default(none) \
        private(n,i,j,i_min,i_max,j_min,j_max,delta,m,fFilterCoefficients,yi_stage0,yi_stage1) \
        shared(y,N,f_i,f_j,yi,n_interp,g_fLUT,g_iKernelHalfSize) \
        num_threads(NUMTHREADS_MAX)
    {
        yi_stage0 = (float*)malloc(KERNEL_SIZE*KERNEL_SIZE*sizeof(float));
        yi_stage1 = (float*)malloc(KERNEL_SIZE*sizeof(float));

        #pragma omp for schedule(static, CHUNKSIZE)
        for(n = 0; n < n_interp; ++n) {
            if(f_i[n] < 0 || f_i[n] > N[0]-1 || f_j[n] < 0 || f_j[n] > N[1]-1) {
                yi[n] = 0;
                continue;
            }

            m[0] = (int)f_i[n];
            m[1] = (int)f_j[n];
            delta[0] = f_i[n] - m[0];
            delta[1] = f_j[n] - m[1];

            for(i = 0; i < 2; ++i) {
                j = (int)((float)NLUT*delta[i]);
                j = MIN(MAX(j,0),NLUT-1);
                fFilterCoefficients[i] = &g_fLUT[j * KERNEL_SIZE];
            }

            i_min = MAX(0, m[0]-KERNEL_HALF_SIZE+1);
            j_min = MAX(0, m[1]-KERNEL_HALF_SIZE+1);
            i_max = MIN(N[0]-1, m[0]+KERNEL_HALF_SIZE);
            j_max = MIN(N[1]-1, m[1]+KERNEL_HALF_SIZE);

            memset(yi_stage0, 0, KERNEL_SIZE*KERNEL_SIZE*sizeof(float));

            for(j = j_min; j <= j_max; ++j) {
                memcpy(&yi_stage0[SUB2IND_2D(i_min-m[0]+KERNEL_HALF_SIZE-1, j-m[1]+KERNEL_HALF_SIZE-1, KERNEL_SIZE)],
                       &y[SUB2IND_2D(i_min,j,N[0])],
                       (i_max-i_min+1)*sizeof(float));
            }

            interp1custom_vec(yi_stage0, yi_stage1, KERNEL_SIZE, fFilterCoefficients[1]);
            yi[n] = interp1custom(yi_stage1, fFilterCoefficients[0]);
        }

        free(yi_stage0);
        free(yi_stage1);
    }
}

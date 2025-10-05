#ifndef PIV_2D_XCORR_H
#define PIV_2D_XCORR_H
#include <stdbool.h>
/**** function declarations ****/

unsigned char bulkxcorr2d(const float *fImageA, const float *fImageB,const float *fMask, const int *nImageSize, 
                           const float *fWinCtrsX, const float *fWinCtrsY, const int *nWindows, float * fWindowWeightA, bool bEnsemble,
                           const float *fWindowWeightB, const int *nWindowSize, int nPeaks, int iPeakFinder, 
                           float *fPkLocX, float *fPkLocY, float *fPkHeight, float *fSx, float *fSy, float *fSxy, float *fCorrelPlane_Out);


float fminvec(const float *fVec, int n);
float fmaxvec(const float *fVec, int n);

#endif
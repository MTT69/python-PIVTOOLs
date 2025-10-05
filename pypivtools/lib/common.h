#ifndef COMMON_H
#define COMMON_H

/**** useful macros ****/
#define SUB2IND_2D(i, j, M)         ((i) + (j)*(M))
#define SUB2IND_3D(i, j, k, M, N)   ((i) + (j)*(M) + (k)*(M)*(N))

#define MIN(A,B)							((A)<(B)?(A):(B))
#define MAX(A,B)							((A)>(B)?(A):(B))

/**** defines ****/
#define PI									3.14159265f
#define SQRT_PI							1.77245385f
#define TRUE								1
#define FALSE								0
#ifndef NAN 
#define NAN (0.0f/0.0f) 
#endif 

#define CHUNKSIZE							256
#define NUMTHREADS_MAX					omp_get_max_threads()

#define ERROR_NONE						0
#define ERROR_NOMEM						1
#define ERROR_NOPLAN_FWD				2
#define ERROR_NOPLAN_BWD				4
#define ERROR_NOPLAN						8
#define ERROR_OUT_OF_BOUNDS             9 

#endif

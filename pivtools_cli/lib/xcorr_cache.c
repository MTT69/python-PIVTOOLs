#include "xcorr_cache.h"
#include "common.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <errno.h>
#include <omp.h>

/******************************************************************************
 * FFTW Wisdom Cache and Plan Reuse
 * 
 * Implements persistent caching of FFTW plans to avoid expensive planning
 * on repeated runs. Key optimizations:
 * 
 * - Save/load FFTW wisdom to disk for persistent optimization
 * - Cache plans by window size to avoid recreating identical plans
 * - Thread-safe plan cache with minimal locking
 *****************************************************************************/

/* Global wisdom management */
static omp_lock_t wisdom_lock;
static int wisdom_initialized = 0;

/* Initialize wisdom system */
void xcorr_cache_init(const char *wisdom_file)
{
	if(!wisdom_initialized) {
		omp_init_lock(&wisdom_lock);
		wisdom_initialized = 1;
		
		/* Try to load existing wisdom */
		if(wisdom_file && wisdom_file[0] != '\0') {
			omp_set_lock(&wisdom_lock);
			FILE *f = fopen(wisdom_file, "r");
			if(f) {
				fftwf_import_wisdom_from_file(f);
				fclose(f);
			}
			omp_unset_lock(&wisdom_lock);
		}
	}
}

/* Save wisdom to disk */
void xcorr_cache_save_wisdom(const char *wisdom_file)
{
	if(!wisdom_initialized || !wisdom_file || wisdom_file[0] == '\0')
		return;
		
	omp_set_lock(&wisdom_lock);
	FILE *f = fopen(wisdom_file, "w");
	if(f) {
		fftwf_export_wisdom_to_file(f);
		fclose(f);
	}
	omp_unset_lock(&wisdom_lock);
}

/* Cleanup wisdom system */
void xcorr_cache_cleanup(void)
{
	if(wisdom_initialized) {
		omp_destroy_lock(&wisdom_lock);
		wisdom_initialized = 0;
	}
}

/* Get default wisdom file path (in user's home or temp directory) */
void xcorr_cache_get_default_wisdom_path(char *path, size_t max_len)
{
	const char *home = getenv("HOME");
	if(home) {
		snprintf(path, max_len, "%s/.pypivtools_fftw_wisdom", home);
	} else {
		snprintf(path, max_len, "/tmp/.pypivtools_fftw_wisdom");
	}
}

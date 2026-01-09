#include "xcorr_cache.h"
#include "common.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <errno.h>
#include <omp.h>
#include <stdatomic.h>

/******************************************************************************
 * FFTW Wisdom Cache and Plan Reuse
 *
 * Implements persistent caching of FFTW plans to avoid expensive planning
 * on repeated runs. Key optimizations:
 *
 * - Save/load FFTW wisdom to disk for persistent optimization
 * - Cache plans by window size to avoid recreating identical plans
 * - Thread-safe plan cache with minimal locking
 * - Thread-safe one-time initialization using C11 atomics
 *****************************************************************************/

/* Global state using atomics for thread-safe initialization */
static omp_lock_t wisdom_lock;
static atomic_int wisdom_initialized = 0;
static atomic_int fftw_initialized = 0;

/* Thread-safe one-time FFTW library initialization */
void fftw_library_init(void)
{
	int expected = 0;
	if (atomic_compare_exchange_strong(&fftw_initialized, &expected, 1)) {
		/* FFTW initialized - we use OpenMP for parallelism at the PIV window level,
		   so no need for FFTW's internal threading (fftwf_init_threads). */
	}
	/* If we didn't win, another thread already initialized */
}

/* Initialize wisdom system (thread-safe) */
void xcorr_cache_init(const char *wisdom_file)
{
	int expected = 0;
	if (atomic_compare_exchange_strong(&wisdom_initialized, &expected, 1)) {
		/* We won the race - initialize the lock and load wisdom */
		omp_init_lock(&wisdom_lock);

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
	/* If we didn't win, another thread already initialized */
}

/* Save wisdom to disk */
void xcorr_cache_save_wisdom(const char *wisdom_file)
{
	if(!atomic_load(&wisdom_initialized) || !wisdom_file || wisdom_file[0] == '\0')
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
	if(atomic_load(&wisdom_initialized)) {
		omp_destroy_lock(&wisdom_lock);
		atomic_store(&wisdom_initialized, 0);
	}
}

/* Comprehensive cleanup of FFTW and wisdom cache (call at program exit) */
void fftw_library_cleanup(void)
{
	/* Cleanup FFTW state if it was initialized */
	if (atomic_load(&fftw_initialized)) {
		fftwf_cleanup();
		atomic_store(&fftw_initialized, 0);
	}

	/* Cleanup wisdom cache */
	xcorr_cache_cleanup();
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

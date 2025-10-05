#ifndef XCORR_CACHE_H
#define XCORR_CACHE_H

#include <fftw3.h>
#include <stddef.h>

/******************************************************************************
 * FFTW Wisdom Cache
 * 
 * Provides persistent caching of FFTW optimization data (wisdom) to avoid
 * expensive plan creation on repeated runs.
 *****************************************************************************/

/* Initialize the wisdom cache system */
void xcorr_cache_init(const char *wisdom_file);

/* Save current wisdom to disk */
void xcorr_cache_save_wisdom(const char *wisdom_file);

/* Cleanup wisdom cache */
void xcorr_cache_cleanup(void);

/* Get default wisdom file path */
void xcorr_cache_get_default_wisdom_path(char *path, size_t max_len);

#endif

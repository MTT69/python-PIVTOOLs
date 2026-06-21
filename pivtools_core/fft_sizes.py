"""Single source of truth for the window/FFT sizes the codelet engine supports.

The cross-correlation FFT is built from fixed-size unrolled codelets (see
``pivtools_cli/lib/codelet_fft.c`` and ``codelet_gen/gen_codelet.py``). Only
these axis lengths have a codelet, so window sizes are restricted to them.

Consumed by:
  * the build (``setup.py`` -> ``gen_codelet.py --sizes``),
  * config validation (``pivtools_core/validation.py``),
  * the GUI (served via the backend so the frontend need not hardcode a copy).

Each size is 2^k or 2^k*3, which the mixed-radix codelet generator handles.
Kept dependency-free so ``setup.py`` can import it without pulling in numpy etc.
"""

BUILT_FFT_SIZES = (8, 12, 16, 24, 32, 48, 64, 96, 128)
"""Supported window axis lengths (rows and columns, independently)."""

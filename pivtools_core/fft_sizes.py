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

SINGLE_MODE_A_SIZES = (4, 6)
"""Extra Frame-A window lengths legal ONLY for ensemble ``single``-mode passes.

These are not FFT sizes and need no codelet. In single mode the correlation FFT
runs at ``sum_window`` (``cpu_ensemble.py`` -> ``window_sizes_for_computation``);
``window_size`` only sets the support of the ``singlepix`` Frame-A weight mask
inside that sum window, plus the grid spacing. The C correlator never receives
it as a length (``PIV_2d_cross_correlate.c`` loops over ``nWindowSize`` =
``sum_window`` and multiplies by the weight arrays it is handed).

2 and 1 are deliberately absent — on cost and signal, NOT on any known
mathematical degeneracy (see the note below), and neither has been vetted:
  * grid spacing follows this size, so plane memory grows as 1/spacing^2:
    2 costs 64x and 1 costs 256x the correlation-plane memory of a 16 px
    window at equal overlap.
  * the AB peak's correlated support is the A window, so at 2x2 (4 px) or
    1x1 (1 px) it is built from a small fraction of a particle per pair.

Note on the AA plane (corrected 2026-07-26 after review): a 1x1 A window does
NOT collapse the autocorrelation. ``cpu_ensemble.py`` passes ``win_weights_B``
(the full ``sum_window`` "bsingle" mask) as BOTH auto weights, and the C
signature labels them "AA/BB weights (always full/symmetric)", so AA/BB never
depend on this size at all. The matched-mean subtraction removes the mean over
that same full sum-window support, not over the m x m support. An earlier
version of this docstring claimed the opposite; it was wrong.
"""

SINGLE_MODE_WINDOW_SIZES = tuple(sorted(SINGLE_MODE_A_SIZES + BUILT_FFT_SIZES))
"""Window axis lengths accepted for an ensemble ``single``-mode pass.

Superset of :data:`BUILT_FFT_SIZES`, so every pre-existing single-mode config
validates exactly as before.
"""

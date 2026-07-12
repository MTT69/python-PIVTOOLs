"""
NaN reason codes for instantaneous PIV.

Per-window int8 codes recording WHY a vector was invalidated, stored on
``PIVPassResult.nan_reason`` and written to the ``full`` save-mode .mat
struct. Codes are aligned with the ensemble taxonomy in
``single_pass_accumulator.py`` where the meanings match (-1 masked,
1 fit failure, 6 displacement check, 10 outlier); instantaneous-only
stages use fresh numbers that do not collide with ensemble meanings.

This module is deliberately matplotlib-free so the interactive manual
tool (``manual_tools/inspect_corr_planes.py``) can import it without
inheriting the Agg backend forced by ``cpu_instantaneous``.

Invariant maintained by the correlator: ``(nan_reason != 0) == nan_mask``.
"""

NAN_MASKED = -1
NAN_VALID = 0
NAN_FIT_FAIL = 1
NAN_LARGE_DISP = 6
NAN_UNCLASSIFIED = 9
NAN_OUTLIER = 10
NAN_SECONDARY_OUTLIER = 12
NAN_PEAK_MAG_NAN = 13

NAN_REASON_LABELS: dict[int, str] = {
    NAN_MASKED: "masked window",
    NAN_VALID: "valid",
    NAN_FIT_FAIL: "peak-fit failure (no peak / LM did not converge)",
    NAN_LARGE_DISP: "large-displacement rejection",
    NAN_UNCLASSIFIED: "unclassified (in nan_mask, no stage claimed it)",
    NAN_OUTLIER: "outlier detection (primary peak)",
    NAN_SECONDARY_OUTLIER: "outlier detection (substituted secondary peak)",
    NAN_PEAK_MAG_NAN: "chosen peak magnitude NaN (peaks exhausted)",
}

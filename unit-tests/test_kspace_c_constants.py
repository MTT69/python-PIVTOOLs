"""Guard against constant drift between kspace_lm_fit.c and its Python oracle.

The C fitter hard-codes its own copy of every tuning constant and status code in
``kspace_lm_fitting.py``. Nothing in the compiler or the type system connects the
two, so a change applied to one and not the other would ship silently — the fits
would simply be slightly different, in a way no parity gate run *before* the edit
could catch. This test makes that failure loud and immediate.

Three C constants (LAM_INIT, LAM_MIN, LAM_ENOPROG) are literals rather than named
constants on the Python side, at ``_batched_lm``; they are pinned here against
explicit expected values. If you change one deliberately, change the Python site
and this table together.
"""

import re
from pathlib import Path

import pytest

import pivtools_cli.piv.piv_backend.kspace_lm_fitting as klm

C_SOURCE = (
    Path(__file__).parent.parent / "pivtools_cli" / "lib" / "kspace_lm_fit.c"
)

# C #define -> the Python value it must equal.
NAME_MAPPED = {
    "MAIN_MAX_ITER": klm.MAIN_MAX_ITER,
    "LM_XTOL": klm.LM_XTOL,
    "LM_FTOL": klm.LM_FTOL,
    "MIN_VALID_PTS": klm.MIN_VALID_PTS,
    "MAX_DISP_FRAC": klm.MAX_DISP_FRAC,
    "COST_PER_PT_ACCEPT": klm.COST_PER_PT_ACCEPT,
    "EXP_ARG_MAX": klm._EXP_ARG_MAX,
    "COLOURED_N0_HI": klm.COLOURED_N0_HI,
    "COLOURED_SEED_KR_MIN": klm.COLOURED_SEED_KR_MIN,
    "COLOURED_SEED_CLIP_LO": klm.COLOURED_SEED_CLIP[0],
    "COLOURED_SEED_CLIP_HI": klm.COLOURED_SEED_CLIP[1],
    "SIGMA_SEED_XX": klm.SIGMA_SEED[0],
    "SIGMA_SEED_YY": klm.SIGMA_SEED[1],
    # gain bounds live in the Python bound vectors, at parameter index 5
    "GAIN_LO": klm.MAIN_LO[5],
    "GAIN_HI": klm.MAIN_HI[5],
    "STATUS_MASKED": klm.STATUS_MASKED,
    "STATUS_SUCCESS": klm.STATUS_SUCCESS,
    "STATUS_NO_CONVERGE": klm.STATUS_NO_CONVERGE,
    "STATUS_LOW_SNR": klm.STATUS_LOW_SNR,
    "STATUS_BIG_DISP": klm.STATUS_BIG_DISP,
}

# Literals on the Python side (see module docstring).
LITERAL_PINNED = {
    "LAM_INIT": 1e-3,  # _batched_lm: lam = np.full(N, 1e-3)
    "LAM_ENOPROG": 1e12,  # _batched_lm: active[rej[lam[rej] > 1e12]] = False
    "LAM_MIN": 1e-12,
    "KMAX": 9,  # 7 base parameters + b4x + b4y
}


def _c_defines():
    """Parse `#define NAME VALUE` from the C source, ignoring function macros."""
    text = C_SOURCE.read_text(encoding="utf-8", errors="replace")
    out = {}
    pattern = re.compile(
        r"^#define\s+([A-Z_][A-Z0-9_]*)\s+(\(?-?[0-9][0-9eE.+-]*\)?)\s*(?:/\*|$)",
        re.MULTILINE,
    )
    for name, raw in pattern.findall(text):
        out[name] = float(raw.strip("()"))
    return out


@pytest.fixture(scope="module")
def defines():
    assert C_SOURCE.is_file(), f"C source not found: {C_SOURCE}"
    d = _c_defines()
    assert d, f"no #define constants parsed from {C_SOURCE}"
    return d


@pytest.mark.parametrize("name,py_value", sorted(NAME_MAPPED.items()))
def test_c_constant_matches_python(defines, name, py_value):
    assert name in defines, f"{name} missing from {C_SOURCE.name}"
    assert defines[name] == pytest.approx(float(py_value), rel=0, abs=0), (
        f"{name}: C has {defines[name]}, Python has {float(py_value)}. "
        "The C fitter mirrors these by hand — update both."
    )


@pytest.mark.parametrize("name,expected", sorted(LITERAL_PINNED.items()))
def test_c_constant_matches_pinned_literal(defines, name, expected):
    assert name in defines, f"{name} missing from {C_SOURCE.name}"
    assert defines[name] == pytest.approx(expected, rel=0, abs=0), (
        f"{name}: C has {defines[name]}, expected {expected}. This one is a "
        "literal in the Python (_batched_lm) — change both, then this table."
    )


def test_status_neg_var_absent_from_c():
    """STATUS_NEG_VAR is defined in Python for contract parity but never set.

    The C does not define it at all. If it ever appears there, the two status
    vocabularies have diverged.
    """
    assert klm.STATUS_NEG_VAR == 5
    assert "STATUS_NEG_VAR" not in _c_defines()

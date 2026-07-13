"""
Tests for the instantaneous per-vector nan_reason codes.

Codes live in pivtools_cli/piv/piv_backend/nan_reason_codes.py; the
correlator maintains the invariant (nan_reason != 0) == nan_mask.
"""

import numpy as np
import pytest
import scipy.io
from corrplane_debug_utils import make_config, run_correlator
from synthetic_piv import generate_displaced_pair

from pivtools_cli.piv.piv_backend import cpu_instantaneous
from pivtools_cli.piv.piv_backend.nan_reason_codes import (
    NAN_FIT_FAIL,
    NAN_LARGE_DISP,
    NAN_MASKED,
    NAN_OUTLIER,
    NAN_PEAK_MAG_NAN,
    NAN_SECONDARY_OUTLIER,
    NAN_UNCLASSIFIED,
    NAN_VALID,
)
from pivtools_cli.piv.save_results import save_piv_result_distributed

ALLOWED_NONZERO = {
    NAN_MASKED,
    NAN_FIT_FAIL,
    NAN_LARGE_DISP,
    NAN_UNCLASSIFIED,
    NAN_OUTLIER,
    NAN_SECONDARY_OUTLIER,
    NAN_PEAK_MAG_NAN,
}


@pytest.fixture()
def clean_pair():
    img_a, img_b = generate_displaced_pair(
        (256, 256),
        num_particles=2000,
        particle_diameter=2.5,
        dx=3.4,
        dy=-2.1,
        seed=7,
    )
    return np.stack([img_a, img_b])


def _assert_invariant(pass_result):
    assert pass_result.nan_reason is not None
    assert pass_result.nan_reason.dtype == np.int8
    assert np.array_equal(
        pass_result.nan_reason != NAN_VALID, pass_result.nan_mask
    ), "(nan_reason != 0) must equal nan_mask"


class TestNanReasonCodes:
    def test_clean_pair(self, tmp_path, clean_pair):
        config = make_config(tmp_path, clean_pair)
        result = run_correlator(config, clean_pair)
        pr = result.passes[0]
        _assert_invariant(pr)
        # valid vectors carry code 0; the clean pair is overwhelmingly valid
        assert np.all(pr.nan_reason[~pr.nan_mask] == NAN_VALID)
        assert (pr.nan_reason == NAN_VALID).mean() > 0.8
        assert set(np.unique(pr.nan_reason)) <= ALLOWED_NONZERO | {NAN_VALID}

    def test_masked_block(self, tmp_path, clean_pair):
        config = make_config(tmp_path, clean_pair)
        # first run to learn the window grid shape
        shape = run_correlator(config, clean_pair).passes[0].ux_mat.shape
        mask = np.zeros(shape, dtype=np.float32)
        mask[1:3, 1:4] = 1.0
        pr = run_correlator(config, clean_pair, vector_masks=[mask]).passes[0]
        _assert_invariant(pr)
        assert np.all(pr.nan_reason[mask.astype(bool)] == NAN_MASKED)
        assert not np.any(pr.nan_reason[~mask.astype(bool)] == NAN_MASKED)

    def test_corrupted_region(self, tmp_path, clean_pair):
        # Decorrelate a block of image B: those windows must fail with codes
        # from the allowed set while the rest of the field stays valid
        # (100% invalid would make infilling abort by design).
        rng = np.random.default_rng(3)
        images = clean_pair.copy()
        images[1, 64:160, 64:160] = rng.random((96, 96)) * 255
        config = make_config(tmp_path, images)
        pr = run_correlator(config, images).passes[0]
        _assert_invariant(pr)
        nonzero = set(np.unique(pr.nan_reason)) - {NAN_VALID}
        assert nonzero <= ALLOWED_NONZERO
        assert len(nonzero) > 0, "decorrelated windows must produce invalid vectors"

    def test_save_roundtrip_full_mode(self, tmp_path, clean_pair):
        config = make_config(tmp_path, clean_pair)
        result = run_correlator(config, clean_pair)
        out = save_piv_result_distributed(
            result, tmp_path / "out", 1, None, "B%05d.mat", save_mode="full"
        )
        mat = scipy.io.loadmat(out, struct_as_record=False, squeeze_me=True)
        saved = mat["piv_result"].nan_reason
        assert np.array_equal(saved.astype(np.int8), result.passes[0].nan_reason)

    def test_minimal_mode_excludes_nan_reason(self, tmp_path, clean_pair):
        config = make_config(tmp_path, clean_pair)
        result = run_correlator(config, clean_pair)
        out = save_piv_result_distributed(
            result, tmp_path / "out", 1, None, "B%05d.mat", save_mode="minimal"
        )
        mat = scipy.io.loadmat(out, struct_as_record=False, squeeze_me=True)
        assert not hasattr(mat["piv_result"], "nan_reason")


class TestSecondaryPeakRescue:
    def test_rescued_vector_survives(self, tmp_path, clean_pair, monkeypatch):
        """A vector flagged as an outlier on the primary peak must be rescued
        when the substituted secondary peak passes outlier detection —
        previously nan_mask was only ever OR'd, so the substitution was
        always NaN'd again."""
        config = make_config(
            tmp_path,
            clean_pair,
            piv_overrides={"num_peaks": 2, "secondary_peak": True},
        )
        target = (3, 3)  # interior window
        calls = {"n": 0}
        real_detect = cpu_instantaneous.apply_outlier_detection

        def fake_detect(ux, uy, methods, peak_mag=None):
            calls["n"] += 1
            mask = np.zeros(ux.shape, dtype=bool)
            if calls["n"] == 1:  # primary-peak stage: flag the target only
                mask[target] = True
            return mask

        monkeypatch.setattr(cpu_instantaneous, "apply_outlier_detection", fake_detect)
        pr = run_correlator(config, clean_pair).passes[0]
        monkeypatch.setattr(cpu_instantaneous, "apply_outlier_detection", real_detect)

        assert calls["n"] >= 2, "secondary-peak outlier re-check did not run"
        _assert_invariant(pr)
        assert not pr.nan_mask[target], "secondary peak was not rescued"
        assert pr.nan_reason[target] == NAN_VALID
        assert pr.peak_choice[target] == 2
        assert np.isfinite(pr.ux_mat[target]) and np.isfinite(pr.uy_mat[target])

    def test_reflagged_secondary_stays_invalid(self, tmp_path, clean_pair, monkeypatch):
        """If outlier detection also rejects the substituted peak, the vector
        stays invalid with the secondary-outlier code."""
        config = make_config(
            tmp_path,
            clean_pair,
            piv_overrides={"num_peaks": 2, "secondary_peak": True},
        )
        target = (3, 3)

        def fake_detect(ux, uy, methods, peak_mag=None):
            mask = np.zeros(ux.shape, dtype=bool)
            mask[target] = True  # reject on every stage
            return mask

        monkeypatch.setattr(cpu_instantaneous, "apply_outlier_detection", fake_detect)
        pr = run_correlator(config, clean_pair).passes[0]

        _assert_invariant(pr)
        assert pr.nan_mask[target]
        # either the substituted peak was itself NaN (13) or re-flagged (12)
        assert pr.nan_reason[target] in (
            NAN_SECONDARY_OUTLIER,
            NAN_PEAK_MAG_NAN,
        )
        assert np.isnan(pr.ux_mat[target]) or pr.nan_mask[target]

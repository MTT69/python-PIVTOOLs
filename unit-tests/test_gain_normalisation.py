"""
Tests for the per-frame laser-gain normalisation preprocessing option.

Covers the config surface (preprocessing.gain_normalisation, default off), the
two-pass regression estimator (pivtools_cli.preprocessing.gain_normalisation)
against an inline numpy oracle transcribing
manual_tools/normalise_frame_gains.py::compute_gains, the masked-regression
restriction, and the application point in the filter pipeline (gain divide
before pixel mask and all filters, ragged-chunk absolute indexing).
"""

import logging

import dask.array as da
import numpy as np
import pytest

from pivtools_cli.preprocessing.gain_normalisation import compute_frame_gains
from pivtools_cli.processing.dask_pipeline import (
    apply_all_filters_slim,
    create_filter_pipeline,
)
from pivtools_core.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cfg(**data) -> Config:
    cfg = Config.__new__(Config)
    cfg.data = data
    return cfg


def _oracle_gains(frames: np.ndarray, mask) -> np.ndarray:
    """Inline transcription of manual_tools/normalise_frame_gains.compute_gains
    (float64 mean image -> per-frame regression), with the illuminated region
    expressed as unmasked pixels instead of a y_max row limit."""
    refs = frames.astype(np.float64).mean(axis=0)  # (2, H, W)
    if mask is not None:
        refs[:, mask] = 0.0
    denoms = (refs * refs).sum(axis=(1, 2))
    numer = np.einsum("nchw,chw->nc", frames.astype(np.float64), refs)
    return numer / denoms[np.newaxis, :]


def _synthetic_series(n_pairs=10, h=24, w=20, seed=7, jitter=0.3):
    """uint16 A/B series I_i = g_i * P with known per-frame gains."""
    rng = np.random.default_rng(seed)
    pattern = 200.0 + 800.0 * rng.random((2, h, w))
    gains = 1.0 + jitter * (rng.random((n_pairs, 2)) - 0.5)
    frames = np.rint(gains[:, :, None, None] * pattern[None]).astype(np.uint16)
    return frames, gains


def _as_dask(frames: np.ndarray, chunks=(4,)) -> da.Array:
    _, c, h, w = frames.shape
    return da.from_array(frames.astype(np.float32), chunks=(chunks[0], c, h, w))


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------
class TestGainNormalisationConfig:
    def test_default_is_off(self):
        assert _cfg().gain_normalisation is False
        assert _cfg(preprocessing={}).gain_normalisation is False

    def test_explicit_values(self):
        on = _cfg(preprocessing={"gain_normalisation": True})
        off = _cfg(preprocessing={"gain_normalisation": False})
        assert on.gain_normalisation is True
        assert off.gain_normalisation is False


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------
class TestComputeFrameGains:
    def test_recovers_injected_gains(self):
        frames, injected = _synthetic_series()
        gains, _ = compute_frame_gains(_as_dask(frames), None)
        # Regression against the ensemble mean recovers gains relative to
        # their per-channel mean (ref = mean(g) * P).
        expected = injected / injected.mean(axis=0, keepdims=True)
        np.testing.assert_allclose(gains, expected, rtol=2e-3)

    def test_matches_numpy_oracle(self):
        frames, _ = _synthetic_series()
        gains, _ = compute_frame_gains(_as_dask(frames), None)
        np.testing.assert_allclose(gains, _oracle_gains(frames, None), rtol=1e-6)

    def test_ragged_chunks_match_single_chunk(self):
        frames, _ = _synthetic_series(n_pairs=10)
        ragged, _ = compute_frame_gains(_as_dask(frames, chunks=(4,)), None)
        single, _ = compute_frame_gains(_as_dask(frames, chunks=(10,)), None)
        np.testing.assert_allclose(ragged, single, rtol=1e-12)

    def test_masked_regression_ignores_masked_pixels(self):
        frames, _ = _synthetic_series()
        mask = np.zeros(frames.shape[2:], dtype=bool)
        mask[:8, :] = True
        # Garbage in the masked region must not touch the gains
        contaminated = frames.copy()
        contaminated[:, :, :8, :] = np.iinfo(np.uint16).max
        gains, prov = compute_frame_gains(_as_dask(contaminated), mask)
        np.testing.assert_allclose(
            gains, _oracle_gains(contaminated, mask), rtol=1e-6
        )
        np.testing.assert_allclose(gains, _oracle_gains(frames, mask), rtol=1e-6)
        assert prov["mask_applied"] is True
        assert prov["n_masked_pixels"] == int(mask.sum())

    def test_no_mask_is_loud(self, caplog):
        frames, _ = _synthetic_series()
        with caplog.at_level(logging.WARNING):
            _, prov = compute_frame_gains(_as_dask(frames), None)
        assert any("FULL FRAME" in r.message for r in caplog.records)
        assert prov["mask_applied"] is False

    def test_mask_shape_mismatch_raises(self):
        frames, _ = _synthetic_series()
        bad_mask = np.zeros((3, 3), dtype=bool)
        with pytest.raises(ValueError, match="mask shape"):
            compute_frame_gains(_as_dask(frames), bad_mask)

    def test_all_masked_raises(self):
        frames, _ = _synthetic_series()
        mask = np.ones(frames.shape[2:], dtype=bool)
        with pytest.raises(ValueError, match="denominator"):
            compute_frame_gains(_as_dask(frames), mask)

    def test_black_frame_raises(self):
        frames, _ = _synthetic_series()
        frames[3] = 0
        with pytest.raises(ValueError, match="non-positive or non-finite"):
            compute_frame_gains(_as_dask(frames), None)


# ---------------------------------------------------------------------------
# Application in the filter pipeline
# ---------------------------------------------------------------------------
class TestPipelineApplication:
    def test_ragged_absolute_index_end_to_end(self):
        frames, _ = _synthetic_series(n_pairs=10)
        gains = np.linspace(0.7, 1.3, 20).reshape(10, 2)
        images = _as_dask(frames, chunks=(4,))  # chunks 4, 4, 2 — ragged
        out = create_filter_pipeline(
            images, _cfg(filters=[]), None, frame_gains=gains
        ).compute()
        expected = frames.astype(np.float32) / gains.astype(np.float32)[
            :, :, None, None
        ]
        np.testing.assert_allclose(out, expected, rtol=1e-6)
        assert out.dtype == np.float32

    def test_gain_divide_happens_before_filters(self):
        frames, _ = _synthetic_series(n_pairs=4)
        block = frames.astype(np.float32)
        gains = np.linspace(0.8, 1.2, 8).reshape(4, 2)
        # 'norm' has an absolute gain floor (max_gain=1.0), so it does NOT
        # commute with a per-frame scalar — ordering is observable.
        specs = [{"type": "norm", "size": [5, 5], "max_gain": 4.0}]
        actual = apply_all_filters_slim(
            block.copy(),
            filter_specs=specs,
            frame_gains=gains,
            chunk_starts=(0,),
            block_id=(0, 0, 0, 0),
        )
        divided_first = apply_all_filters_slim(
            (block / gains.astype(np.float32)[:, :, None, None]).copy(),
            filter_specs=specs,
        )
        filtered_first = (
            apply_all_filters_slim(block.copy(), filter_specs=specs)
            / gains.astype(np.float32)[:, :, None, None]
        )
        np.testing.assert_allclose(actual, divided_first, rtol=1e-6)
        assert not np.allclose(actual, filtered_first, rtol=1e-3)

    def test_masked_pixels_still_zero_after_gain(self):
        frames, _ = _synthetic_series(n_pairs=4)
        mask = np.zeros(frames.shape[2:], dtype=bool)
        mask[:5, :] = True
        gains = np.full((4, 2), 1.5)
        out = apply_all_filters_slim(
            frames.astype(np.float32).copy(),
            pixel_mask=mask,
            frame_gains=gains,
            chunk_starts=(0,),
            block_id=(0, 0, 0, 0),
        )
        assert np.all(out[:, :, mask] == 0.0)
        np.testing.assert_allclose(
            out[:, :, ~mask], frames.astype(np.float32)[:, :, ~mask] / 1.5, rtol=1e-6
        )

    def test_length_mismatch_raises(self):
        frames, _ = _synthetic_series(n_pairs=6)
        gains = np.ones((4, 2))
        with pytest.raises(ValueError, match="frame_gains"):
            create_filter_pipeline(
                _as_dask(frames), _cfg(filters=[]), None, frame_gains=gains
            )

    def test_gains_without_chunk_starts_raises(self):
        frames, _ = _synthetic_series(n_pairs=4)
        with pytest.raises(ValueError, match="chunk_starts"):
            apply_all_filters_slim(
                frames.astype(np.float32).copy(),
                frame_gains=np.ones((4, 2)),
                block_id=(0, 0, 0, 0),
            )

    def test_non_positive_gain_in_block_raises(self):
        frames, _ = _synthetic_series(n_pairs=4)
        gains = np.ones((4, 2))
        gains[2, 1] = 0.0
        with pytest.raises(ValueError, match="non-positive"):
            apply_all_filters_slim(
                frames.astype(np.float32).copy(),
                frame_gains=gains,
                chunk_starts=(0,),
                block_id=(0, 0, 0, 0),
            )

    def test_disabled_is_noop(self):
        frames, _ = _synthetic_series(n_pairs=4)
        images = _as_dask(frames)
        # Nothing configured: the pipeline must hand back the array unchanged
        assert create_filter_pipeline(images, _cfg(filters=[]), None) is images
        # frame_gains=None leaves the existing filter path bit-identical
        specs = [{"type": "gaussian", "sigma": 1.0}]
        a = apply_all_filters_slim(frames.astype(np.float32).copy(), filter_specs=specs)
        b = apply_all_filters_slim(frames.astype(np.float32).copy(), filter_specs=specs)
        np.testing.assert_array_equal(a, b)

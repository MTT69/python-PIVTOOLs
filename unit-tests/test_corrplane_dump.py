"""
Tests for the instantaneous correlation-plane debug dump
(instantaneous_piv.dump_correlation_planes).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from corrplane_debug_utils import make_config, run_correlator
from synthetic_piv import generate_displaced_pair

from pivtools_cli.piv.save_results import save_corr_plane_dump

MANUAL_TOOLS = Path(__file__).resolve().parents[1] / "manual_tools"

DUMP_KEYS = (
    "planes",
    "pk_loc_x_raw",
    "pk_loc_y_raw",
    "pk_height_raw",
    "sx",
    "sy",
    "sxy",
    "peak_finder",
)


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


def _run_with_dump(tmp_path, images, piv_overrides=None):
    overrides = {"dump_correlation_planes": True}
    overrides.update(piv_overrides or {})
    config = make_config(tmp_path, images, piv_overrides=overrides)
    return run_correlator(config, images)


class TestDumpPayload:
    def test_shapes_and_peak_consistency(self, tmp_path, clean_pair):
        result = _run_with_dump(tmp_path, clean_pair)
        pr = result.passes[0]
        dd = pr.debug_dump
        assert dd is not None
        assert set(DUMP_KEYS) <= set(dd.keys())

        planes = dd["planes"]
        ny, nx = pr.ux_mat.shape
        wh, ww = planes.shape[2], planes.shape[3]
        assert planes.shape == (ny, nx, wh, ww)
        assert planes.dtype == np.float32

        # argmax of each valid window's plane must sit on the raw peak
        for j in range(ny):
            for i in range(nx):
                if pr.nan_mask[j, i]:
                    continue
                am_r, am_c = np.unravel_index(np.argmax(planes[j, i]), (wh, ww))
                pk_r = dd["pk_loc_y_raw"][0, j, i] + wh // 2
                pk_c = dd["pk_loc_x_raw"][0, j, i] + ww // 2
                assert abs(am_r - pk_r) <= 1.0 and abs(am_c - pk_c) <= 1.0, (
                    f"window ({j},{i}): plane argmax ({am_r},{am_c}) vs "
                    f"raw peak ({pk_r:.2f},{pk_c:.2f})"
                )

    def test_masked_windows_dump_zero_planes(self, tmp_path, clean_pair):
        shape = _run_with_dump(tmp_path, clean_pair).passes[0].ux_mat.shape
        mask = np.zeros(shape, dtype=np.float32)
        mask[0:2, 0:3] = 1.0
        config = make_config(
            tmp_path,
            clean_pair,
            piv_overrides={"dump_correlation_planes": True},
        )
        pr = run_correlator(config, clean_pair, vector_masks=[mask]).passes[0]
        planes = pr.debug_dump["planes"]
        for j, i in np.argwhere(mask > 0):
            assert not planes[j, i].any(), f"masked window ({j},{i}) not zero"
        # at least one unmasked window must have real data
        assert planes[shape[0] - 1, shape[1] - 1].any()

    def test_flag_off_no_dump(self, tmp_path, clean_pair):
        config = make_config(tmp_path, clean_pair)
        result = run_correlator(config, clean_pair)
        assert all(p.debug_dump is None for p in result.passes)

    def test_scalar_batch_plane_parity(self, tmp_path, clean_pair):
        """The scalar and lockstep-batch peak-fit paths export through
        different memcpy sites — the dumped planes must be identical."""
        r_batch = _run_with_dump(tmp_path / "b", clean_pair, {"peak_fit_impl": "batch"})
        r_scalar = _run_with_dump(
            tmp_path / "s", clean_pair, {"peak_fit_impl": "scalar"}
        )
        assert np.array_equal(
            r_batch.passes[0].debug_dump["planes"],
            r_scalar.passes[0].debug_dump["planes"],
        )


class TestNpzWriter:
    def test_write_and_tool_load(self, tmp_path, clean_pair):
        result = _run_with_dump(tmp_path, clean_pair)
        out = save_corr_plane_dump(result, tmp_path / "res", 1, "B%05d.mat")
        assert out is not None
        assert Path(out).name == "B00001_corrplanes.npz"

        sys.path.insert(0, str(MANUAL_TOOLS))
        try:
            from inspect_corr_planes import load_dump
        finally:
            sys.path.remove(str(MANUAL_TOOLS))

        dump = load_dump(Path(out))
        assert dump["frame_number"] == 1
        assert dump["n_passes"] == 1
        assert 0 in dump["labels"] and -1 in dump["labels"]
        p = dump["passes"][0]
        assert np.array_equal(p["planes"], result.passes[0].debug_dump["planes"])
        assert np.array_equal(p["nan_reason"], result.passes[0].nan_reason)

    def test_no_payload_writes_nothing(self, tmp_path, clean_pair):
        config = make_config(tmp_path, clean_pair)
        result = run_correlator(config, clean_pair)  # flag off
        out = save_corr_plane_dump(result, tmp_path / "res", 1, "B%05d.mat")
        assert out is None
        assert not (tmp_path / "res" / "debug_corr_planes").exists()

"""Unit tests for the pure (no-Dask, no-figure) logic of the bench harness.

Covers the algorithmic bits that are easy to get subtly wrong: sweep generation
and its caps, the fixed-vs-per-worker workload split, and the A/B provenance
guard. Run with ``pytest profile/test_bench.py`` from the worktree root.
"""

from __future__ import annotations

import bench_common as bc
import bench_scaling as bs


# --- sweep generation ------------------------------------------------------


def test_progression_includes_cap_and_powers():
    assert bs._progression(20) == [1, 2, 4, 8, 16, 20]
    assert bs._progression(16) == [1, 2, 4, 8, 16]
    assert bs._progression(192) == [1, 2, 4, 8, 16, 32, 64, 128, 192]
    assert bs._progression(1) == [1]


def test_matrix_sweep_respects_core_and_worker_caps():
    cfgs = bs.matrix_sweep(total_cores=8, max_workers=2)
    assert cfgs, "expected at least one matrix config"
    for c in cfgs:
        assert c["workers"] * c["threads"] <= 8
        assert c["workers"] <= 2


def test_oversub_sweep_is_actually_oversubscribed():
    cfgs = bs.oversub_sweep(total_cores=8, max_workers=8)
    assert cfgs, "expected oversub configs on an 8-core budget"
    for c in cfgs:
        assert c["workers"] * c["threads"] > 8


def test_build_config_list_dedups_across_sweeps():
    cfgs = bs.build_config_list("all", total_cores=8, max_workers=4)
    keys = [(c["workers"], c["threads"]) for c in cfgs]
    assert len(keys) == len(set(keys)), "duplicate (workers, threads) leaked through"


def test_workload_fixed_vs_per_worker():
    n, mode = bs.workload_for(workers=4, batch_size=10, n_images=200)
    assert (n, mode) == (200, bs.WORKLOAD_FIXED)
    n, mode = bs.workload_for(workers=4, batch_size=10, n_images=None)
    assert mode == bs.WORKLOAD_PER_WORKER
    assert n == 4 * bs._BATCHES_PER_WORKER * 10


# --- FFTW wisdom policy ----------------------------------------------------


def test_wisdom_shared_is_noop_and_per_worker_refuses():
    bs._apply_wisdom_policy(bs.WISDOM_SHARED)  # must not raise
    try:
        bs._apply_wisdom_policy(bs.WISDOM_PER_WORKER)
    except NotImplementedError as e:
        assert "PIV_FFTW_WISDOM_PATH" in str(e)
    else:
        raise AssertionError("per-worker wisdom should refuse until the C hook exists")


# --- provenance guard ------------------------------------------------------


def _prov(backend, **over):
    base = {
        "fft_backend": backend, "git_sha": "abc", "git_dirty": False,
        "hostname": "node1", "cpu_count": 192, "cpu_model": "EPYC",
        "filesystem": "lustre", "cache_policy": "warm",
        "pivtools_version": "1.0", "platform": "linux",
    }
    base.update(over)
    return base


def test_guard_ok_when_only_backend_differs():
    g = bc.compare_provenance(_prov("fftw"), _prov("codelet"))
    assert g["ok"] is True
    assert g["backends"] == ("fftw", "codelet")
    assert not g["warnings"]


def test_guard_warns_on_cross_machine():
    g = bc.compare_provenance(_prov("fftw"), _prov("codelet", cpu_model="Broadwell"))
    assert g["ok"] is False
    assert any("cpu_model" in w for w in g["warnings"])


def test_guard_warns_when_same_backend():
    g = bc.compare_provenance(_prov("fftw"), _prov("fftw"))
    assert g["ok"] is False
    assert any("same FFT backend" in w for w in g["warnings"])

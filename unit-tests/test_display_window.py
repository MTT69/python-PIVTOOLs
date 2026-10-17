#!/usr/bin/env python3
"""
test_display_window.py

Particle-aware display window for the image-pair viewer
(``pivtools_gui.utils.particle_display_window``).

Area percentiles put the display window where most pixels are, so glare and
walls decide it and PIV particles end up dim. The particle window measures the
background level and the median particle-peak intensity per tile of an 8x8
grid and takes medians over tiles, so a glare patch covering a minority of
tiles is ignored. These tests check that on a synthetic frame with a known
background, a known particle peak and a saturated glare patch, plus the
fallback, the linear counts encode, manual-window parsing, the route wiring
and the speed budget.

Usage:
    pytest unit-tests/test_display_window.py -v [--make-figures]
"""

import base64
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import yaml
from flask import Flask
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pivtools_gui.app as gui_app  # noqa: E402
from pivtools_core.config import Config  # noqa: E402
from pivtools_gui.utils import (  # noqa: E402
    ENCODE_HEADROOM,
    MIN_PEAK_TILES,
    TILE_GRID,
    encode_display_frame,
    numpy_to_base64,
    parse_display_bounds,
    particle_display_window,
)

# Synthetic frame (12-bit camera on a uint16 container).
_SHAPE = (1024, 1024)
_BG = 100.0  # background level (counts)
_NOISE = 5.0  # background Gaussian noise sigma (counts)
_PEAK = 250.0  # particle peak intensity, background included (counts)
_PARTICLE_SIGMA = 1.0  # particle image Gaussian sigma (px), ~2.8 px diameter at e^-2
_PARTICLES_PER_PX = 0.003  # sparse seeding, so particles rarely overlap
_SATURATION = 4095  # 12-bit full well; the glare patch sits here
# Glare covers a 2 x 3 block of tiles = 6/64 ~ 9 % < 25 % of tiles.
_GLARE_TILES = (2, 3)
# 16 MP timing case: the BFS_1500 sensor shape.
_TIMING_SHAPE = (3248, 4872)
_TIME_BUDGET_S = 0.050  # plan target per frame


def _synthetic_frame(rng: np.random.Generator, glare: bool = True) -> np.ndarray:
    """Noisy background + Gaussian particles of peak _PEAK + optional glare."""
    h, w = _SHAPE
    frame = rng.normal(_BG, _NOISE, size=_SHAPE)
    n = int(_PARTICLES_PER_PX * h * w)
    ys = rng.integers(3, h - 3, n)
    xs = rng.integers(3, w - 3, n)
    r = np.arange(-3, 4)
    kernel = (_PEAK - _BG) * np.exp(
        -(r[:, None] ** 2 + r[None, :] ** 2) / (2 * _PARTICLE_SIGMA**2)
    )
    for y, x in zip(ys, xs):
        # max, not add: overlapping particles must not create brighter peaks.
        win = frame[y - 3 : y + 4, x - 3 : x + 4]
        np.maximum(win, _BG + kernel, out=win)
    if glare:
        gh, gw = _GLARE_TILES
        frame[: gh * h // TILE_GRID, : gw * w // TILE_GRID] = _SATURATION
    return np.clip(np.rint(frame), 0, _SATURATION).astype(np.uint16)


def _decode(b64: str) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(base64.b64decode(b64))).convert("L"))


@pytest.fixture
def rng():
    return np.random.default_rng(20261017)


# --- Algorithm -----------------------------------------------------------------


def test_window_recovers_background_and_particle_peak_despite_glare(
    rng, make_figures, output_dir
):
    """Black ~ background, white ~ particle peak; the glare tiles are ignored."""
    frame = _synthetic_frame(rng)
    win = particle_display_window(frame)

    assert win["method"] == "particle"
    # Particles cover ~2 % of a tile so the tile median sits within ~1 count
    # of the true background; the peak median sees the +-sigma noise.
    assert win["black"] == pytest.approx(_BG, abs=2.0)
    assert win["white"] == pytest.approx(_PEAK, rel=0.05)

    # The old area window is pulled up by the glare: it is the failure mode.
    old = np.percentile(frame, 99.5)
    assert old == _SATURATION

    if make_figures:
        _figure(frame, win, output_dir / "display_window")


def test_full_tile_and_sub_block_agree(rng):
    """TILE_SAMPLE is a speed knob only: it must not move the window."""
    frame = _synthetic_frame(rng)
    sub = particle_display_window(frame, tile_sample=64)
    full = particle_display_window(frame, tile_sample=None)
    assert sub["black"] == pytest.approx(full["black"], rel=0.05)
    assert sub["white"] == pytest.approx(full["white"], rel=0.05)


@pytest.mark.parametrize(
    "frame",
    [
        np.full(_SHAPE, 500, dtype=np.uint16),  # flat
        np.zeros(_SHAPE, dtype=np.float32),  # empty processed frame
    ],
    ids=["flat", "zeros"],
)
def test_frame_without_particles_falls_back_to_percentile(frame):
    win = particle_display_window(frame)
    assert win["method"] == "percentile_fallback"
    assert win["black"] == win["white"] == float(frame.flat[0])


def test_fallback_is_the_5_95_count_percentiles(rng):
    """The old auto rule took 5/95 percentiles in the sqrt domain; percentiles
    commute with sqrt, so in counts they are the plain 5/95 percentiles."""
    frame = rng.uniform(0.0, 1000.0, size=(256, 256)).astype(np.float32)
    frame[rng.random(frame.shape) < 0.7] = 0.0  # sigma = 0 everywhere -> fallback
    win = particle_display_window(frame)
    assert win["method"] == "percentile_fallback"
    lo, hi = np.percentile(frame, [5, 95])
    assert win["black"] == pytest.approx(lo)
    assert win["white"] == pytest.approx(hi)
    sq_lo, sq_hi = np.percentile(np.sqrt(frame), [5, 95])
    assert win["white"] == pytest.approx(sq_hi**2, rel=1e-4)
    assert win["black"] == pytest.approx(sq_lo**2, abs=1e-6)


def test_mostly_zero_residual_is_not_mistaken_for_particles(rng):
    """>50 % exact zeros gives MAD = 0 in every tile. With sigma = 0 the
    5-sigma test would count every positive noise value as a particle; such
    tiles are excluded, so the frame falls back instead of reporting noise."""
    frame = rng.uniform(0.0, 3.0, size=_SHAPE).astype(np.float32)
    frame[rng.random(_SHAPE) < 0.7] = 0.0
    win = particle_display_window(frame)
    assert win["method"] == "percentile_fallback"


def test_quorum_counts_tiles_not_peaks(rng):
    """Particles confined to fewer than MIN_PEAK_TILES tiles -> fallback.

    Bounded (uniform) noise: Gaussian tails put an occasional 5-sigma maximum
    in some tile, which would make the planted-tile count seed-dependent."""
    frame = rng.uniform(_BG - _NOISE, _BG + _NOISE, size=_SHAPE)
    th, tw = _SHAPE[0] // TILE_GRID, _SHAPE[1] // TILE_GRID
    for k in range(MIN_PEAK_TILES - 1):
        cy, cx = th // 2, k * tw + tw // 2
        frame[cy - 1 : cy + 2, cx - 1 : cx + 2] = _PEAK
    win = particle_display_window(frame.astype(np.float32))
    assert win["method"] == "percentile_fallback"


def test_read_only_frame_is_not_written(rng):
    """Readers hand back cached read-only arrays; the window must only read."""
    frame = _synthetic_frame(rng)
    frame.flags.writeable = False
    assert particle_display_window(frame)["method"] == "particle"
    encode_display_frame(frame, "png")


def test_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        particle_display_window(np.zeros((2, 64, 64), dtype=np.uint16))


# --- Manual window parsing ---------------------------------------------------------


def test_parse_display_bounds():
    assert parse_display_bounds(None, None) is None
    assert parse_display_bounds("120", "400.5") == (120.0, 400.5)
    assert parse_display_bounds(7.4, 80) == (7.4, 80.0)


@pytest.mark.parametrize(
    "vmin, vmax, match",
    [
        ("120", None, "both"),
        (None, 400, "both"),
        ("abc", "400", "numbers"),
        ("120", "inf", "finite"),
        ("400", "120", "vmax > vmin"),
        ("120", "120", "vmax > vmin"),
    ],
)
def test_parse_display_bounds_rejects_bad_input(vmin, vmax, match):
    with pytest.raises(ValueError, match=match):
        parse_display_bounds(vmin, vmax)


# --- Encoding --------------------------------------------------------------------


def test_manual_window_encodes_linearly_in_counts():
    """[80, 190] -> encode span 80 .. min(80 + 2*110, max(190, data max 250)) =
    80 .. 250, so a level is (x - 80) * 255 / 170 = 1.5 * (x - 80), rounded."""
    arr = np.array([[30, 80, 100, 190, 250]], dtype=np.uint8)
    b64, stats = encode_display_frame(arr, "png", (80.0, 190.0))
    img = _decode(b64)
    assert img.tolist() == [[0, 0, 30, 165, 255]]
    assert stats["enc_lo"] == 80.0
    assert stats["enc_hi"] == 250.0
    assert (stats["data_min"], stats["data_max"]) == (30.0, 250.0)
    # Without bounds, numpy_to_base64 still passes uint8 through (calibration).
    assert np.array_equal(_decode(numpy_to_base64(arr, format="png")), arr)


def test_auto_encode_span_and_stats(rng):
    frame = _synthetic_frame(rng)
    b64, stats = encode_display_frame(frame, "png")
    win = particle_display_window(frame)
    assert set(stats) == {
        "black",
        "white",
        "method",
        "enc_lo",
        "enc_hi",
        "data_min",
        "data_max",
    }
    assert stats["method"] == "particle"
    assert (stats["black"], stats["white"]) == (win["black"], win["white"])
    assert stats["enc_lo"] == win["black"]
    assert stats["enc_hi"] == pytest.approx(
        win["black"] + ENCODE_HEADROOM * (win["white"] - win["black"])
    )
    img = _decode(b64)
    assert img[frame <= win["black"]].max() == 0
    assert img[frame == _SATURATION].min() == 255  # rint: the top is exactly 255


def test_manual_encode_keeps_auto_stats(rng):
    """A manual window changes the encode span, never the reported auto window."""
    frame = _synthetic_frame(rng)
    _, auto = encode_display_frame(frame, "png")
    _, manual = encode_display_frame(frame, "png", (0.0, 4095.0))
    assert (manual["black"], manual["white"]) == (auto["black"], auto["white"])
    assert manual["enc_lo"] == 0.0
    assert manual["enc_hi"] == 4095.0  # capped at the data max (glare = 4095)


# --- Routes ------------------------------------------------------------------------


def _make_config(tmp_path) -> Config:
    cfg = {
        "paths": {
            "base_paths": [str(tmp_path / "out")],
            "source_paths": [str(tmp_path / "src")],
            "camera_count": 1,
            "camera_numbers": [1],
            "camera_subfolders": [],
        },
        "images": {
            "image_type": "standard",
            "image_format": ["B%05d_A.tif", "B%05d_B.tif"],
            "num_images": 10,
            "start_index": 1,
            "pairing_preset": "pre_paired",
            "num_loops": 1,
            "use_camera_subfolders": False,
        },
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return Config(path=str(path))


@pytest.fixture
def client(tmp_path, monkeypatch, rng):
    """Real api_bp routes; read_pair is stubbed with the synthetic pair."""
    config = _make_config(tmp_path)
    pair = np.stack([_synthetic_frame(rng), _synthetic_frame(rng)])
    monkeypatch.setattr(gui_app, "get_config", lambda: config)
    monkeypatch.setattr(gui_app, "read_pair", lambda *a, **k: pair)
    monkeypatch.setattr(gui_app, "raw_image_cache", {})
    monkeypatch.setattr(gui_app, "FRAME_1_KEYS", set())
    app = Flask(__name__)
    app.register_blueprint(gui_app.api_bp)
    return app.test_client()


def test_get_frame_pair_manual_window_and_cache_separation(client):
    q = "/backend/get_frame_pair?camera=1&idx=2&format=png"
    auto = client.get(q).get_json()
    manual = client.get(q + "&vmin=0&vmax=4095").get_json()

    assert auto["stats"]["A"]["method"] == "particle"
    assert manual["stats"]["A"]["enc_lo"] == 0.0
    assert manual["stats"]["A"]["black"] == auto["stats"]["A"]["black"]
    # Distinct cache entries: the second request must not get the first's PNG.
    assert auto["A"] != manual["A"]
    assert len(gui_app.raw_image_cache) == 2

    # Cache hit returns the same stats as the miss.
    assert client.get(q).get_json()["stats"] == auto["stats"]


def test_only_auto_window_frame_1_is_pinned(client):
    """Frame 1 is pinned in the raw cache (never evicted). Each manual window is
    a new cache key, so pinning those would let window tuning on frame 1 grow
    the pinned set without bound."""
    q = "/backend/get_frame_pair?camera=1&idx=1&format=png"
    client.get(q)
    for hi in (300, 400, 500):
        client.get(q + f"&vmax={hi}&vmin=100")
    assert len(gui_app.raw_image_cache) == 4
    assert len(gui_app.FRAME_1_KEYS) == 1
    (pinned,) = gui_app.FRAME_1_KEYS
    assert pinned[-1] is None  # the auto-window key


@pytest.mark.parametrize("params", ["&vmin=10", "&vmin=abc&vmax=5", "&vmin=50&vmax=5"])
def test_bad_manual_window_is_rejected(client, params):
    r = client.get("/backend/get_frame_pair?camera=1&idx=1" + params)
    assert r.status_code == 400
    assert "Display window" in r.get_json()["error"]


def test_preload_rejects_bad_manual_window(client):
    r = client.post("/backend/preload_images", json={"camera": 1, "vmin": 5})
    assert r.status_code == 400


# --- Speed -------------------------------------------------------------------------


@pytest.mark.slow
def test_16mp_frame_meets_time_budget():
    """Best of 5 on a 16 MP uint16 frame must beat the 50 ms budget."""
    rng = np.random.default_rng(0)
    frame = rng.normal(140, 10, size=_TIMING_SHAPE).astype(np.uint16)
    peaks = rng.random(_TIMING_SHAPE) < 0.003
    frame[peaks] = 220
    particle_display_window(frame)  # warm-up (import / allocator)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        win = particle_display_window(frame)
        times.append(time.perf_counter() - t0)
    assert win["method"] == "particle"
    assert min(times) < _TIME_BUDGET_S, f"best {min(times) * 1e3:.1f} ms"


# --- Figure ------------------------------------------------------------------------


def _figure(frame: np.ndarray, win: dict, out_dir: Path) -> None:
    """Histogram with bounds + rendered crop across the glare edge."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_h, ax_old, ax_new) = plt.subplots(1, 3, figsize=(16, 5))
    ax_h.hist(frame.ravel(), bins=np.arange(0, 400, 2), color="0.5", log=True)
    ax_h.axvline(_BG, color="k", ls=":", label=f"true background {_BG:.0f}")
    ax_h.axvline(_PEAK, color="k", ls="--", label=f"true particle peak {_PEAK:.0f}")
    ax_h.axvline(win["black"], color="tab:blue", label=f"black {win['black']:.1f}")
    ax_h.axvline(win["white"], color="tab:orange", label=f"white {win['white']:.1f}")
    ax_h.set_xlabel("intensity (counts)")
    ax_h.set_ylabel("pixel count (log)")
    ax_h.set_title(f"synthetic frame (glare at {_SATURATION} off-axis)")
    ax_h.legend(fontsize=8)

    h, w = _SHAPE
    gh, gw = _GLARE_TILES
    y0, x0 = gh * h // TILE_GRID - 128, gw * w // TILE_GRID - 128
    crop = (slice(y0, y0 + 256), slice(x0, x0 + 256))
    old = _decode(numpy_to_base64(frame, format="png"))  # old sqrt 0.5-99.5 encode
    b64, stats = encode_display_frame(frame, "png")
    new = _decode(b64)
    # Client remap of the auto window inside the encode span (as the viewer does).
    white_level = (
        255 * (win["white"] - stats["enc_lo"]) / (stats["enc_hi"] - stats["enc_lo"])
    )
    ax_old.imshow(old[crop], cmap="gray", vmin=0, vmax=255)
    ax_old.set_title("old sqrt full-range encode")
    ax_new.imshow(new[crop], cmap="gray", vmin=0, vmax=white_level)
    ax_new.set_title(
        f"linear auto window [{win['black']:.0f}, {win['white']:.0f}] counts"
    )
    for ax in (ax_old, ax_new):
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
    fig.tight_layout()
    fig.savefig(out_dir / "synthetic_window.png", dpi=110)
    plt.close(fig)

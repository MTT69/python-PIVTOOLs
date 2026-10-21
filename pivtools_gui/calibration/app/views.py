"""calibration Flask blueprint — backend for the unified Calibration GUI.

One backend, two front-ends: these routes call the SAME ``Calibrator`` /
``StereoCalibrator`` / ``record`` code as the CLI. Image loading + validation reuse
the app-wide all-format reader (``calibration_loader.read_calibration_image`` /
``validate_calibration_images``), so standard images AND LaVision .im7/.set +
Phantom .cine all work, with per-camera-subfolder + indexing resolved from config —
exactly like the rest of the app.

Source model: the GUI persists the image-source config under ``config.calibration``
(``calibration_sources``, ``image_format``, ``image_type``, ``num_images``,
``use_camera_subfolders``, ``camera_subfolders``, ``zero_based_indexing``) and passes
a ``source_path_idx``; the backend reads that config via ``get_config()``. The fitted
model is the DaVis-matching pinhole (k1,k2,p1,p2) — there is no distortion choice.

Routes (prefix ``/calibration`` under the app's ``/backend``):
- POST /calibration/validate        -> validate the image source (found-N, preview, suggested pattern)
- POST /calibration/detect_datum    -> detect the datum view, cache it, return dot pixels
- GET  /calibration/datum_image     -> datum-view PNG for the click overlay
- GET  /calibration/frame_image     -> any frame as PNG (frame navigation)
- POST /calibration/detect_frame    -> detect an arbitrary frame + report frame count
- POST /calibration/detect_views    -> detect every view, write through to inputs.mat
- POST /calibration/restore_detections -> stored detections from inputs.mat, detecting nothing
- POST /calibration/snap_fiducial   -> snap a click to the nearest detected dot
- POST /calibration/generate_model  -> run the full calibration (mono/stereo), save + figures
- GET  /calibration/model           -> summary of a saved model
- GET  /calibration/figures|figure  -> list / serve the proof figures for a model
- POST /calibration/global/compute  -> planar N-camera datum-chain shifts
- POST /calibration/set_datum       -> shift a type's coordinates.mat (viewer datum/offset)
- POST /calibration/apply (+ /apply/status/<id>) -> apply the model to PIV output (job)
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
import scipy.io
from flask import Blueprint, Response, jsonify, request

from pivtools_cli import calibration_cli as c2
from pivtools_core import calibration_settings as cs
from pivtools_core.config import get_config
from pivtools_core.coordinate_utils import extract_coordinates, get_num_coordinate_runs
from pivtools_core.image_handling.calibration_loader import (
    get_calibration_frame_count,
    read_calibration_image,
    validate_calibration_images,
)
from pivtools_core.image_handling.path_utils import infer_image_type
from pivtools_core.paths import get_data_paths, vector_glob_from_format
from pivtools_gui.calibration import global_coords as gc2
from pivtools_gui.calibration import record as rec
from pivtools_gui.calibration import runio as c2runio
from pivtools_gui.calibration import self_cal as c2sc
from pivtools_gui.calibration import world_frame as WF
from pivtools_gui.calibration.camera_model import (
    CameraModel,
    DistortionModel,
    Polynomial3DModel,
    PolynomialModel,
    ScaleFactorModel,
)
from pivtools_gui.calibration.detection.charuco import CharucoBoardDetector
from pivtools_gui.calibration.global_grid import (
    first_view_orientation_candidates,
    resolve_global_grid_partial,
)
from pivtools_gui.calibration.inputs_store import (
    joint_det_key,
    save_inputs,
    try_load_inputs,
    update_inputs,
)
from pivtools_gui.calibration.joint import (
    PoseDiversity,
    format_pose_diversity,
    pose_diversity_from_meta,
)
from pivtools_gui.calibration.joint_driver import run_joint_from_spec
from pivtools_gui.calibration.pipeline import Calibrator, build_scale_factor_record
from pivtools_gui.calibration.settings_seed import seed_settings
from pivtools_gui.calibration.stereo_model import StereoCalibrator
from pivtools_gui.services.job_manager import job_manager
from pivtools_gui.utils import (
    camera_number,
    get_display_contrast_stats,
    numpy_to_base64,
)
from pivtools_gui.utils.worker_pool import get_max_workers

calibration_bp = Blueprint("calibration", __name__)
logger = logging.getLogger(__name__)


def _detect_parallel(board, params, imgs, spacing_mm=None, on_done=None):
    """Detect calibration targets in many views concurrently.

    Detection is OpenCV-bound (the GIL is released inside cv2), so threads scale and
    the pool honours the ``processing.post_processing_workers`` knob. Each task builds
    its own detector — cv2 detector objects are not documented thread-safe. Results
    return in image order; ``on_done()`` fires once per completed view (any order).
    """
    dets: List[Any] = [None] * len(imgs)
    if not imgs:
        return dets
    max_workers = get_max_workers(len(imgs))

    def _one(k):
        d = c2._build_detector(board, params).detect(imgs[k])
        if spacing_mm is not None:
            d.spacing_mm = spacing_mm
        return k, d

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_one, k) for k in range(len(imgs))]
        for future in as_completed(futures):
            k, d = future.result()
            dets[k] = d
            if on_done is not None:
                on_done()
    return dets


def _board_not_detected(detector: Any, img: np.ndarray, det: Any) -> Tuple[str, dict]:
    """Error text and diagnostics for one frame whose board was not detected.

    ChArUco failures are diagnosed (wrong dictionary, wrong squares count, no
    markers) and the hint is folded into the text: "board not detected — <hint>".
    The diagnosis re-runs marker detection, so it belongs on single-frame failure
    paths only.
    """
    if isinstance(detector, CharucoBoardDetector):
        diag = detector.diagnose_failure(img, det)
        return f"board not detected — {diag['hint']}", diag
    return "board not detected", det.diagnostics


# Datum-view detections cached for the snap workflow: camera number ->
# ``((camera, source_idx, board, datum_frame), DetectionResult)``. One entry per camera, so
# both cameras of a stereo pair can be picked at once; snap_fiducial refuses a detection whose
# key (source, board or datum frame) differs from the snap request's.
_datum_cache: dict = {}
_datum_lock = threading.Lock()

# Joint multi-camera: detections cached per (source path, board, n_views, format, params) so the
# live resolve-grid loop and the generate job do not re-detect every view on every call. The
# in-memory layer is the fast path; the persistent layer is the sidecar inputs.mat (see
# _joint_detect), keyed by _joint_det_key so a param change forces a re-detect.
_joint_detect_cache: dict = {}
_joint_detect_lock = threading.Lock()

# Apply jobs: cooperative cancel events keyed by job id (the video_maker pattern), and the
# lock that makes "no running apply overlaps my output dirs" + create_job + registering
# the event one atomic step, so two near-simultaneous Apply clicks cannot both pass the
# overlap check before either job exists.
_apply_cancel_events: dict[str, threading.Event] = {}
_apply_lock = threading.Lock()
# Job statuses during which an apply job may still write calibrated frames.
_APPLY_ACTIVE_STATUSES = ("starting", "running")


def _joint_inputs_dir(source_idx: int, board: str) -> Optional[Path]:
    """Model dir holding this rig's sidecar ``inputs.mat``, or None if the source is unknown.

    The sidecar lives beside the solved joint model (``joint_{board}/model/``); the detections
    persist there so they survive a server restart and a model delete — detecting every view of
    every camera is the slow part of the live overlay.
    """
    try:
        return rec.joint_model_dir_for_source(_source_path(source_idx), board)
    except Exception:
        return None


# The GUI offers exactly one model — the DaVis PinholeOpenCV (k1,k2,p1,p2).
_MODEL = DistortionModel.STANDARD

# frame_total_source labels for a view count the user set rather than one counted
# from the images (those carry the loader's FrameCount.source).
FRAME_TOTAL_FROM_REQUEST = "request"
FRAME_TOTAL_FROM_SETTING = "n_views setting"
# Mono generate_model only: the auto-count failed (images gone) and the view count is
# the length of the detection set cached in the inputs.mat sidecar.
FRAME_TOTAL_FROM_SIDECAR = "inputs.mat cache"


def _cfg() -> dict:
    return c2._cfg2(get_config())


def _request_board(get: Callable[[str], Any]) -> Tuple[str, Optional[bool]]:
    """``(board, active_stereo)`` for a request.

    An explicit request ``board`` wins and ``active_stereo`` is None (the route's own
    ``stereo`` flag decides). Otherwise both come from ``calibration.active`` via
    ``parse_active_method``, which raises when the key is missing or unknown.
    """
    board = get("board")
    if board:
        return board, None
    return c2runio.parse_active_method(_cfg().get("active"))


def _source_idx(get: Callable[[str], Any]) -> int:
    v = get("source_path_idx")
    return int(v) if v not in (None, "") else int(_cfg().get("source_idx", 0))


def _source_path(source_idx: int) -> Path:
    """Resolved calibration source directory for this index (where the model is saved)."""
    return get_config().get_calibration_source(int(source_idx))


def _settings_idx(source_idx: int) -> dict:
    """Settings sidecar for a source index; the defaults template when absent.

    An unconfigured source list also yields the template — helpers that only
    need a knob default (datum frame, camera number) must not fail before the
    route's own source handling does. A sidecar that EXISTS but fails
    validation still raises: silently reverting a corrupt file to the defaults
    template would be exactly the stale-state class this store eliminates.
    """
    try:
        source = _source_path(source_idx)
    except (ValueError, IndexError):
        return cs.default_settings()
    settings = cs.try_load_settings(source)
    return settings if settings is not None else cs.default_settings()


def _zero_based(source_idx: int) -> bool:
    """The source's ``image.zero_based_indexing`` — what the frame reader applies.

    Folded into every detection cache key (``joint_det_key`` and the in-memory joint
    cache): flipping it maps each frame number to a different file.
    """
    return bool(_settings_idx(source_idx)["image"]["zero_based_indexing"])


def _settings_for(get: Callable[[str], Any]) -> dict:
    return _settings_idx(_source_idx(get))


def _rig(get: Callable[[str], Any]) -> dict:
    return _settings_for(get).get("rig") or {}


def _sidecar_rig(get: Callable[[str], Any]) -> dict:
    """The ``rig`` block of the source's settings SIDECAR, ``{}`` when no sidecar exists.

    Unlike :func:`_rig` this never falls through to the defaults template, so a caller
    that needs a user-chosen value (the stereo camera pair) cannot pick up the
    template's ``[1, 2]`` by accident. A present-but-invalid sidecar still raises.
    """
    try:
        source = _source_path(_source_idx(get))
    except (ValueError, IndexError):
        return {}
    settings = cs.try_load_settings(source)
    return (settings or {}).get("rig") or {}


def _methods(get: Callable[[str], Any]) -> dict:
    return _settings_for(get).get("methods") or {}


def _model_type_arg(
    get: Callable[[str], Any], board: Optional[str] = None, stereo: bool = False
) -> Optional[str]:
    """Requested record type for a model load, now that types have per-type files.

    Explicit ``model_type`` request param first, else the board's configured
    ``model_type`` (the GUI persists its selector there). For stereo loads a
    mono-only configured type (``polynomial`` / ``scale_factor``) is dropped —
    it names the mono fit, not the stereo record. ``None`` lets the resolver
    pick the single record present (ambiguity raises, listing the types).
    """
    mt = get("model_type")
    if not mt and board:
        mt = _methods(get).get(board, {}).get("model_type")
    if not mt:
        return None
    mt = str(mt)
    if stereo and mt not in rec.STEREO_MODEL_TYPES:
        return None
    return mt


def _generate_dt(get: Callable[[str], Any], source: Path) -> float:
    """dt to stamp into a generated record: request > settings ``rig.dt`` > error.

    Velocity scales linearly with dt, so it has no safe default; generation is
    the moment the user is present to supply it, and every record it produces
    carries it (apply then resolves request > model-stamped > error, with no
    config source).
    """
    v = get("dt")
    if v not in (None, ""):
        return c2runio.require_positive_dt(v, "the request 'dt'")
    settings = cs.try_load_settings(source)
    dt = ((settings or {}).get("rig") or {}).get("dt")
    if dt is None:
        raise ValueError(
            "dt is required to generate a calibration model — velocity has no "
            "safe default. Set dt in the Calibration tab (saved to the "
            "source's calibration/settings.yaml) or pass 'dt' in the request."
        )
    return c2runio.require_positive_dt(dt, "settings rig.dt")


def _resolve_board(get: Callable[[str], Any], overrides: Optional[dict] = None):
    """Resolve (cfg, board, params, detector). Board params may be overridden per-call.

    ``cfg`` (the YAML calibration block) supplies only the ``active`` pointer;
    board geometry comes from the source's settings sidecar + request overrides.
    """
    cfg = _cfg()
    board = _request_board(get)[0]
    params = c2._board_params(_methods(get), board, overrides)
    detector = c2._build_detector(board, params)
    return cfg, board, params, detector


@calibration_bp.route("/calibration/settings", methods=["GET"])
def get_calibration_settings():
    """Per-source settings sidecar for the requested source.

    ``exists: false`` ships a seed built from the model records on disk (the
    persisted YAML is presumed stale and never consulted); nothing is written
    until the client saves. A present-but-corrupt sidecar is an error, never
    silently reseeded.
    """
    try:
        source = _source_path(_source_idx(request.args.get))
    except (ValueError, IndexError) as e:
        return jsonify({"error": str(e)}), 400
    try:
        existing = cs.try_load_settings(source)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if existing is not None:
        return jsonify({"exists": True, "settings": existing})
    return jsonify({"exists": False, "settings": seed_settings(source)})


@calibration_bp.route("/calibration/settings", methods=["POST"])
def save_calibration_settings():
    """Deep-merge a partial settings payload into the source's sidecar.

    Unknown TOP-LEVEL blocks are rejected — a typo'd block name would be
    written and then ignored forever, the silent-staleness class this store
    exists to kill. Keys nested inside known blocks stay open: the deliberately
    untemplated knobs (``fit.use_release_object``, ``methods.<board>.fix_k2``)
    are legal precisely because they are absent from the template.

    Every save reconciles ``image.camera_subfolders`` with the configured camera
    count (empty at one camera, at most one entry per camera otherwise), so the
    returned ``settings`` carry the list actually stored.
    """
    data = request.get_json(force=True) or {}
    partial = data.get("settings")
    if not isinstance(partial, dict):
        return jsonify({"error": "settings must be a mapping"}), 400
    unknown = set(partial) - set(cs.default_settings())
    if unknown:
        return (
            jsonify(
                {"error": "unknown settings block(s): " + ", ".join(sorted(unknown))}
            ),
            400,
        )
    try:
        source = _source_path(_source_idx(data.get))
    except (ValueError, IndexError) as e:
        return jsonify({"error": str(e)}), 400
    try:
        if not cs.settings_path(source).exists():
            # First save for this source: persist the record-recovered seed as
            # the base, THEN merge the client partial over it. Without this the
            # seed shown by GET (dt, geometry, camera_pair from the newest
            # records) would be discarded whenever the first POST is partial —
            # the whole migration path for existing datasets rides on it.
            cs.save_settings(source, seed_settings(source))
        merged = cs.save_settings(source, partial)
        # Camera-subfolder invariant (mirrors /update_config for config.yaml): a
        # single camera resolves at the source root and a multi-camera rig has at
        # most one entry per camera. Stale entries from a previous session must not
        # survive, so prune on every persist. This route is the only writer that can
        # put entries in the list (the seed and init-settings write the empty default).
        camera_count = get_config().camera_count
        subfolders = list((merged.get("image") or {}).get("camera_subfolders") or [])
        pruned = [] if camera_count == 1 else subfolders[:camera_count]
        if pruned != subfolders:
            logger.info(
                "Pruned calibration camera_subfolders %r -> %r for camera_count=%d",
                subfolders,
                pruned,
                camera_count,
            )
            merged = cs.save_settings(source, {"image": {"camera_subfolders": pruned}})
    except (ValueError, PermissionError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"status": "success", "settings": merged})


def _load_one(
    camera: int,
    frame: int,
    source_idx: int,
    image_format: Optional[str],
    image_type: Optional[str],
    normalize_uint8: bool = True,
) -> np.ndarray:
    """Load one calibration frame (any format) via the app-wide reader.

    Detection callers pass ``normalize_uint8=False``: the dotboard detector works in
    float32 and the ChArUco / stepped detectors scale to uint8 themselves, so the
    native 12/16-bit frame is what they must see (the CLI already does this; an
    8-bit min-max collapse of a retro-reflective board lost 89 of 129 dots,
    2026-10-20). The display endpoints that drive the auto-contrast slider also
    need the native depth so ``get_display_contrast_stats`` sees the real range.
    The uint8 default remains for the 8-bit preview routes.
    """
    return read_calibration_image(
        int(frame),
        int(camera),
        get_config(),
        int(source_idx),
        image_format=image_format,
        image_type=image_type,
        normalize_uint8=normalize_uint8,
    )


def _load_views(
    camera: int,
    frame_total: int,
    source_idx: int,
    image_format: Optional[str],
    image_type: Optional[str],
) -> List[np.ndarray]:
    return [
        _load_one(
            camera, k + 1, source_idx, image_format, image_type, normalize_uint8=False
        )
        for k in range(int(frame_total))
    ]


def _frame_total(
    get: Callable[[str], Any], camera: int, source_idx: int
) -> Tuple[int, str]:
    """Resolve ``(frame total, source)``: request value, else settings ``image.n_views``,
    else the auto-count from the images.

    ``source`` is ``"request"``, ``"n_views setting"`` or the auto-count's
    ``FrameCount.source`` (``"files"``, ``".set index"``, ``".cine header"``), so the
    GUI can say where the number came from.

    Raises ``ValueError`` for a non-positive request value, and naming the camera and
    source when the auto-count fails: a zero or guessed view count would silently
    detect, persist or solve the wrong set.
    """
    v = get("frame_total")
    if v not in (None, ""):
        n = int(v)
        if n <= 0:
            raise ValueError(f"frame_total must be a positive view count, got {v!r}")
        return n, FRAME_TOTAL_FROM_REQUEST
    n = int((_settings_idx(source_idx).get("image") or {}).get("n_views") or 0)
    if n > 0:
        return n, FRAME_TOTAL_FROM_SETTING
    try:
        count = get_calibration_frame_count(int(camera), get_config(), int(source_idx))
    except Exception as exc:
        raise ValueError(
            f"could not auto-detect the number of calibration views for camera "
            f"{camera}, source {source_idx}: {exc} -- set n_views explicitly"
        ) from exc
    return count.n, count.source


def _pair_frame_total(
    get: Callable[[str], Any], cameras: List[int], source_idx: int
) -> Tuple[int, str]:
    """Resolve ``(frame total, source)`` for a stereo pair.

    An explicit request ``frame_total`` or settings ``n_views`` applies to every camera
    and is not compared. Otherwise each camera is auto-counted and a mismatch raises
    ``ValueError``: calibration pairs views by index, so unequal counts mean the pairs
    are misaligned or a view is missing.
    """
    n, source = _frame_total(get, cameras[0], source_idx)
    if source in (FRAME_TOTAL_FROM_REQUEST, FRAME_TOTAL_FROM_SETTING):
        return n, source
    for other in cameras[1:]:
        m, _ = _frame_total(get, other, source_idx)
        if m != n:
            raise ValueError(
                f"Camera {cameras[0]} has {n} views, camera {other} has {m} -- "
                "calibration pairs views by index, so the counts must match"
            )
    return n, source


def _frame_count_or_error(
    camera: int, source_idx: int
) -> Tuple[Optional[int], Optional[str]]:
    """``(frame_count, None)`` from the auto-count, or ``(None, reason)`` when it fails.

    For the browsing routes only: they still serve a frame that loads when the count
    cannot be read, and report the reason instead of a fake ``0`` so the GUI can show
    why the frame range is unknown.
    """
    try:
        return get_calibration_frame_count(camera, get_config(), source_idx).n, None
    except Exception as exc:
        logger.warning(
            "calibration frame count failed for camera %s, source %s: %s",
            camera,
            source_idx,
            exc,
        )
        return None, str(exc)


def _datum_frame(get: Callable[[str], Any]) -> int:
    v = get("datum_frame")
    return int(v) if v not in (None, "") else int(_rig(get).get("datum_frame") or 1)


def _png_response(img: np.ndarray):
    """Encode an image as an 8-bit PNG HTTP response (contrast-stretched if needed)."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        mx = float(img.max()) or 1.0
        img = (img.astype(np.float64) / mx * 255.0).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return jsonify({"error": "encode failed"}), 500
    return Response(buf.tobytes(), mimetype="image/png")


def _clicks_from(payload) -> Optional[dict]:
    if not payload:
        return None
    return {
        "origin": np.asarray(payload["origin"], dtype=float),
        "x_axis": np.asarray(payload["x_axis"], dtype=float),
        "y_axis": np.asarray(payload["y_axis"], dtype=float),
    }


def _origin_mm_from(payload) -> Optional[tuple]:
    """World (X, Y) mm of the origin dot, carried inside the clicks payload."""
    if not payload:
        return None
    om = payload.get("origin_mm")
    if om is None:
        return None
    return (float(om[0]), float(om[1]))


_PICK_ROLES = ("origin", "x_axis", "y_axis")


def _stereo_coords(cam1: int, cam2: int, clicks, clicks2) -> Optional[dict]:
    """Stereo sidecar ``coords`` from a request's per-camera world-frame picks.

    Shape (string camera keys, as the stepped sidecar)::

        {"cameras": {"<cam1>": {"origin": [x, y], "x_axis": [x, y], "y_axis": [x, y]},
                     "<cam2>": {...}},
         "origin_mm": [X, Y]}

    ``clicks`` (camera 1) carries the world ``origin_mm`` of the shared frame; any
    ``origin_mm`` in ``clicks2`` is ignored (camera 2 is labelled into camera 1's frame).
    A camera without picks is absent from ``cameras``. Returns None when the request
    carries no picks at all. Raises ``ValueError`` when camera 1's picks lack
    ``origin_mm``.
    """
    if not clicks and not clicks2:
        return None
    cams: dict = {}
    for cam, payload in ((cam1, clicks), (cam2, clicks2)):
        if payload:
            cams[str(int(cam))] = {
                role: [float(v) for v in payload[role]] for role in _PICK_ROLES
            }
    origin_mm = None
    if clicks:
        om = clicks.get("origin_mm")
        if om is None:
            raise ValueError(
                f"clicks for camera {cam1} must carry origin_mm (the world X, Y mm of the "
                "origin dot)"
            )
        origin_mm = [float(om[0]), float(om[1])]
    return {"cameras": cams, "origin_mm": origin_mm}


def _stereo_cameras_block(coords) -> Optional[dict]:
    """The ``cameras`` map of a stereo ``coords`` blob, or None when no picks are stored.

    A non-empty stereo blob without ``cameras`` is the pre-2026-10-19 flat shape (camera 1's
    picks only): raise, never read it as camera 1's frame.
    """
    if not coords:
        return None
    if not isinstance(coords, dict) or "cameras" not in coords:
        raise ValueError(
            "stale sidecar: the stereo world-frame picks in inputs.mat lack 'cameras' "
            "(written before per-camera picks) -- delete it and re-pick both cameras"
        )
    return coords["cameras"]


def _stereo_clicks(coords, camera: int) -> Optional[dict]:
    """One camera's picks from a stereo ``coords`` blob, as ``_clicks_from`` returns them.

    None when the blob is empty or holds no picks for ``camera``.
    """
    cams = _stereo_cameras_block(coords)
    if cams is None:
        return None
    return _clicks_from(cams.get(str(int(camera))))


def _stereo_origin_mm(coords) -> Optional[tuple]:
    """The shared frame's world ``origin_mm`` of a stereo ``coords`` blob (None if unset)."""
    if _stereo_cameras_block(coords) is None:
        return None
    om = coords.get("origin_mm")
    return None if om is None else (float(om[0]), float(om[1]))


def _require_camera_pair(get: Callable[[str], Any]) -> Tuple[int, int]:
    """Stereo camera pair: request ``camera_pair`` (list or "a,b"), else the settings rig's.

    Raises ``ValueError`` when neither names one: a literal default pair would silently
    detect, solve or restore cameras the user never chose. Only a real sidecar counts
    (:func:`_sidecar_rig`), never the defaults template's ``[1, 2]``.
    """
    pair = get("camera_pair") or _sidecar_rig(get).get("camera_pair")
    if not pair:
        raise ValueError(
            "camera_pair is required for stereo: none in the request and no settings "
            "sidecar for this source names rig.camera_pair"
        )
    if isinstance(pair, str):
        pair = [int(x) for x in pair.split(",")]
    if len(pair) != 2:
        raise ValueError(f"camera_pair must name two cameras, got {pair!r}")
    return int(pair[0]), int(pair[1])


def _require_meta(meta: dict, key: str, what: str):
    """``meta[key]`` from a record's ``board_meta``, or ``ValueError`` naming the key.

    For response fields the record always stores: a missing key means the record is stale,
    and a guessed stand-in would report something the solve never did.
    """
    if not isinstance(meta, dict) or meta.get(key) is None:
        raise ValueError(
            f"{what}: board_meta has no {key!r} (written by an older build) -- "
            "regenerate the model"
        )
    return meta[key]


def _rms_units(model_type: str) -> str:
    """Unit of a model type's reported RMS: planar polynomial fits in mm, the rest in px."""
    return "mm" if model_type == "polynomial" else "px"


def _figure_rank(name: str) -> int:
    """Display priority for a proof figure: the dewarped board first (clearest at a glance),
    then detection, then the fit/reprojection residual, then the supporting geometry."""
    n = name.lower()
    if "dewarp" in n:
        return 0
    if n.startswith("detection"):
        return 1
    if n.startswith("reprojection") or n.startswith("polynomial_fit"):
        return 2
    if n.startswith("world_frame"):
        return 3
    if n.startswith("boards_3d"):
        return 4
    return 5


def _list_figures(fig_dir) -> List[str]:
    if not fig_dir or not Path(fig_dir).is_dir():
        return []
    # Skip macOS AppleDouble companions (._*) that appear when writing to non-HFS drives.
    names = [p.name for p in Path(fig_dir).glob("*.png") if not p.name.startswith("._")]
    # Rank by figure kind (dewarp first), alphabetical within a kind. Filenames are unchanged —
    # the /calibration/figure?name= serve endpoint still resolves each by basename.
    return sorted(names, key=lambda n: (_figure_rank(n), n))


def _job_status_response(job_id: str):
    """Shared poll response for every calibration job route: timed status or 404."""
    data = job_manager.get_job_with_timing(job_id)
    if data is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(data)


def _world_frame_payload(wf) -> dict:
    """Saved world frame for the GUI to restore the picked origin/+X/+Y + origin mm."""

    def lst(a):
        return None if a is None else [float(x) for x in np.asarray(a).reshape(-1)]

    return {
        "mode": str(wf.mode),
        "origin": lst(wf.origin_px),
        "x_axis": lst(wf.x_axis_px),
        "y_axis": lst(wf.y_axis_px),
        "origin_mm": lst(wf.origin_mm) or [0.0, 0.0],
    }


def _geometry_payload(board_meta) -> Optional[dict]:
    """The stamped board geometry (``board_meta['geometry']``) coerced to JSON-native scalars —
    Flask jsonify rejects numpy types, and the .mat round-trip can return size-1 numpy scalars.
    None for records saved before geometry stamping, so the GUI falls back to its panel defaults.
    """
    g = board_meta.get("geometry") if isinstance(board_meta, dict) else None
    if not isinstance(g, dict):
        return None
    return {
        k: (v if isinstance(v, str) else (v.item() if hasattr(v, "item") else v))
        for k, v in g.items()
    }


def _inputs_world_frame(model_dir, camera: Optional[int] = None) -> Optional[dict]:
    """World-frame picks recovered from the model dir's ``inputs.mat`` (the clicks survive a
    model delete), shaped like ``_world_frame_payload`` so the GUI restores origin/+X/+Y and a
    deleted model can be re-solved without re-clicking. None when no usable clicks are stored.

    ``camera=None`` reads a mono sidecar's flat coords. ``camera=N`` reads camera N's picks
    from a stereo sidecar (``coords["cameras"]["N"]``) with the pair's shared ``origin_mm``;
    a stereo sidecar in the old flat shape raises ``ValueError`` (stale, re-pick). An
    unreadable sidecar raises ``InputsCorrupt`` (surfaced by the app's JSON error handler).
    """
    side = try_load_inputs(model_dir)
    c = side.coords if side else None
    if camera is not None:
        cams = _stereo_cameras_block(c)
        picks = (cams or {}).get(str(int(camera)))
        if not picks:
            return None
        c = {**picks, "origin_mm": c.get("origin_mm")}
    if not isinstance(c, dict) or not c.get("origin"):
        return None

    def lst(v):
        return [float(x) for x in v] if v else None

    return {
        "mode": "clicks",
        "origin": lst(c.get("origin")),
        "x_axis": lst(c.get("x_axis")),
        "y_axis": lst(c.get("y_axis")),
        "origin_mm": lst(c.get("origin_mm")) or [0.0, 0.0],
    }


def _intrinsics(cm) -> dict:
    """Pinhole intrinsics of a CameraModel, shaped for the GUI results card."""
    return {
        "fx": float(cm.K[0, 0]),
        "fy": float(cm.K[1, 1]),
        "cx": float(cm.K[0, 2]),
        "cy": float(cm.K[1, 2]),
        "camera_matrix": cm.K.tolist(),
        "dist_coeffs": cm.dist.tolist(),
        "rms": float(cm.rms),
        "image_width": int(cm.image_size[0]),
        "image_height": int(cm.image_size[1]),
    }


def _polynomial_summary(pm: PolynomialModel) -> dict:
    """Polynomial coefficients of a PolynomialModel, shaped for the GUI results card."""
    return {
        "model_type": "polynomial",
        "coeffs_x": [float(c) for c in pm.coeffs_x],
        "coeffs_y": [float(c) for c in pm.coeffs_y],
        "x0": float(pm.x0),
        "sx": float(pm.sx),
        "y0": float(pm.y0),
        "sy": float(pm.sy),
        "rms_x_mm": float(pm.rms_x_mm),
        "rms_y_mm": float(pm.rms_y_mm),
        "image_width": int(pm.image_size[0]),
        "image_height": int(pm.image_size[1]),
    }


def _polynomial3d_summary(pm: Polynomial3DModel) -> dict:
    """Single-view 3D polynomial summary (world->image cubic), shaped for the card."""
    return {
        "model_type": "polynomial3d",
        "rms": float(pm.rms_px),
        "plane_rms": [float(v) for v in pm.plane_rms_px],
        "coeffs_u": [float(c) for c in pm.coeffs_u],
        "coeffs_v": [float(c) for c in pm.coeffs_v],
        "x0": float(pm.x0),
        "sx": float(pm.sx),
        "y0": float(pm.y0),
        "sy": float(pm.sy),
        "z0": float(pm.z0),
        "sz": float(pm.sz),
        "image_width": int(pm.image_size[0]),
        "image_height": int(pm.image_size[1]),
    }


def _scale_factor_summary(
    sf: ScaleFactorModel, dt: Optional[float], frame_idx=None, wf=None
) -> dict:
    """Scale-factor params shaped for the GUI results card + restore-on-load.

    ``dt`` is the record's stamped dt, or None when the record carries none (a
    stale model) — reported as null, never replaced by a made-up default.

    ``frame_idx`` (the 1-based frame the origin was picked on), when known, lets the GUI
    restore the origin/axis overlay on that same frame rather than always frame 1.

    ``wf`` (the record's WorldFrame) carries the PICKED origin pixel + its world
    ``origin_mm`` — the model's own origin_px is the world-zero pixel, which differs
    from the picked point once origin_mm is non-zero (the offset is baked in by
    shifting it). The GUI restores what the user picked/typed, so prefer wf.
    """
    px_per_mm = (1.0 / sf.mm_per_pixel) if sf.mm_per_pixel else float("nan")
    picked = (
        wf.origin_px if (wf is not None and wf.origin_px is not None) else sf.origin_px
    )
    origin_mm = (
        [float(wf.origin_mm[0]), float(wf.origin_mm[1])]
        if (wf is not None and wf.origin_mm is not None)
        else [0.0, 0.0]
    )
    out = {
        "model_type": "scale_factor",
        "origin_px": [float(picked[0]), float(picked[1])],
        "origin_mm": origin_mm,
        "mm_per_pixel": float(sf.mm_per_pixel),
        "px_per_mm": float(px_per_mm),
        "dt": float(dt) if dt is not None else None,
        "col_sign": int(sf.col_sign),
        "row_sign": int(sf.row_sign),
        "swap_axes": int(sf.swap_axes),
        "x_dir": "right" if sf.col_sign >= 0 else "left",
        "y_dir": "up" if sf.row_sign < 0 else "down",
    }
    if frame_idx is not None:
        out["frame_idx"] = int(frame_idx)
    return out


def _figures_dir(get: Callable[[str], Any]) -> Path:
    """Resolve the figures dir for a model from request locators (mono or stereo)."""
    board, active_stereo = _request_board(get)
    source = _source_path(_source_idx(get))
    if str(get("joint")) in ("1", "true", "True"):
        return rec.joint_model_dir_for_source(source, board).parent / "figures"
    stereo_arg = get("stereo")
    stereo = (
        str(stereo_arg) in ("1", "true", "True")
        if stereo_arg not in (None, "")
        else bool(active_stereo)
    )
    if stereo:
        cam1, cam2 = _require_camera_pair(get)
        return (
            rec.stereo_model_dir_for_source(source, cam1, cam2, board).parent
            / "figures"
        )
    camera = int(get("camera") or _rig(get).get("camera") or 1)
    return rec.mono_model_dir_for_source(source, camera, board).parent / "figures"


def _meta_float(meta: dict, key: str) -> Optional[float]:
    try:
        return float(meta[key]) if meta and key in meta else None
    except (TypeError, ValueError):
        return None


def _meta_int(meta: dict, key: str) -> Optional[int]:
    v = _meta_float(meta, key)
    return int(v) if v is not None else None


# ---------------------------------------------------------------------------
# Image source validation
# ---------------------------------------------------------------------------


@calibration_bp.route("/calibration/validate", methods=["POST"])
def validate():
    """Validate the calibration image source (all formats): found-count, preview, suggestion."""
    data = request.get_json() or {}
    camera = int(data.get("camera") or _rig(data.get).get("camera") or 1)
    source_idx = _source_idx(data.get)
    try:
        result = validate_calibration_images(
            camera,
            get_config(),
            source_idx,
            image_format=data.get("image_format"),
            num_images=data.get("frame_total"),
            image_type=data.get("image_type"),
        )
    except FileNotFoundError as exc:
        # The only FileNotFoundError reaching here is the missing settings
        # sidecar (validate_images_generic returns read failures in its result
        # dict, and get_calibration_source raises ValueError/IndexError). That
        # is the expected first-visit state of every pre-sidecar source, not a
        # fault — the message is already actionable, so no traceback.
        logger.warning("calibration validate: %s", exc)
        return jsonify({"valid": False, "error": str(exc)}), 200
    except Exception as exc:
        logger.exception("calibration validate failed")
        return jsonify({"valid": False, "error": str(exc)}), 200
    return jsonify(result)


# ---------------------------------------------------------------------------
# Detection + frame navigation
# ---------------------------------------------------------------------------


@calibration_bp.route("/calibration/detect_datum", methods=["POST"])
def detect_datum():
    """Detect the datum view for a camera; cache it and return dot pixels for clicking."""
    data = request.get_json() or {}
    try:
        cfg, board, params, detector = _resolve_board(
            data.get, data.get("board_params")
        )
    except ValueError as exc:
        # Missing board geometry is a routine incomplete-form condition, not a fault —
        # the message is already actionable, so one WARNING line, no traceback.
        logger.warning("calibration detect_datum: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 200
    camera = int(data.get("camera") or _rig(data.get).get("camera") or 1)
    source_idx = _source_idx(data.get)
    frame = _datum_frame(data.get)
    try:
        img = _load_one(
            camera,
            frame,
            source_idx,
            data.get("image_format"),
            data.get("image_type"),
            normalize_uint8=False,
        )
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    det = detector.detect(img)
    if not det.success:
        error, diagnostics = _board_not_detected(detector, img, det)
        return (
            jsonify(
                {
                    "success": False,
                    "error": error,
                    "diagnostics": diagnostics,
                }
            ),
            200,
        )

    with _datum_lock:
        _datum_cache[camera] = ((camera, int(source_idx), str(board), int(frame)), det)
    h, w = np.asarray(img).shape[:2]
    return jsonify(
        {
            "success": True,
            "camera": camera,
            "board": board,
            "frame": int(frame),
            "n_points": det.n,
            "width": int(w),
            "height": int(h),
            "image_points": det.image_points.tolist(),
            "grid_indices": det.grid_indices.tolist(),
        }
    )


@calibration_bp.route("/calibration/datum_image", methods=["GET"])
def datum_image():
    """Return the datum-view image (PNG) for a camera, for the click overlay."""
    camera = int(
        request.args.get("camera") or _rig(request.args.get).get("camera") or 1
    )
    source_idx = _source_idx(request.args.get)
    frame = _datum_frame(request.args.get)
    try:
        img = _load_one(
            camera,
            frame,
            source_idx,
            request.args.get("image_format"),
            request.args.get("image_type"),
        )
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return jsonify({"error": str(exc)}), 404
    return _png_response(img)


@calibration_bp.route("/calibration/frame_image", methods=["GET"])
def frame_image():
    """Serve any calibration frame (1-based image index) as a PNG."""
    camera = int(
        request.args.get("camera") or _rig(request.args.get).get("camera") or 1
    )
    source_idx = _source_idx(request.args.get)
    frame = int(request.args.get("frame", 1))
    try:
        img = _load_one(
            camera,
            frame,
            source_idx,
            request.args.get("image_format"),
            request.args.get("image_type"),
        )
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return jsonify({"error": str(exc)}), 404
    return _png_response(img)


@calibration_bp.route("/calibration/frame", methods=["GET"])
def frame_json():
    """Serve a calibration frame as JSON: ``{image, stats, width, height, frame_count}``.

    Matches the v1 ``/calibration/get_frame`` shape so the original
    ``CalibrationImageViewer`` (contrast sliders + colormap + prefetch) works
    unchanged on the calibration backend. All formats via the app-wide reader;
    ``stats.vmin_pct``/``vmax_pct`` drive the auto-contrast window.
    """
    _cfg()
    camera = camera_number(request.args.get("camera", default=1, type=int))
    idx = request.args.get("idx", default=1, type=int)
    source_idx = _source_idx(request.args.get)
    output_format = (
        request.args.get("format", default="jpeg", type=str) or "jpeg"
    ).lower()
    quality = request.args.get("quality", default=85, type=int)
    frame_count, frame_count_error = _frame_count_or_error(camera, source_idx)
    if frame_count is not None and idx > frame_count:
        return (
            jsonify(
                {
                    "error": f"Frame index {idx} exceeds available frames ({frame_count})",
                    "frame_count": frame_count,
                    "frame_count_error": None,
                    "requested_idx": idx,
                }
            ),
            400,
        )
    try:
        # Display path: keep native bit depth so get_display_contrast_stats and
        # numpy_to_base64 see the real dynamic range (sqrt + percentile window),
        # matching the PIV viewer. Detection routes load native depth too.
        img = _load_one(
            camera,
            idx,
            source_idx,
            request.args.get("image_format"),
            request.args.get("image_type"),
            normalize_uint8=False,
        )
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return (
            jsonify(
                {
                    "error": str(exc),
                    "frame_count": frame_count,
                    "frame_count_error": frame_count_error,
                }
            ),
            404,
        )

    arr = np.asarray(img)
    arr_f = arr.astype(np.float32)
    stats = {
        "min": float(arr_f.min()),
        "max": float(arr_f.max()),
        "mean": float(arr_f.mean()),
        "dtype": str(arr.dtype),
    }
    contrast = get_display_contrast_stats(arr)
    stats["vmin_pct"] = contrast["vmin_pct"]
    stats["vmax_pct"] = contrast["vmax_pct"]
    return jsonify(
        {
            "image": numpy_to_base64(arr, format=output_format, jpeg_quality=quality),
            "mime_type": f"image/{output_format}",
            "width": int(arr.shape[1]),
            "height": int(arr.shape[0]),
            "stats": stats,
            "frame_count": frame_count,
            "frame_count_error": frame_count_error,
            "current_idx": idx,
        }
    )


@calibration_bp.route("/calibration/detect_frame", methods=["POST"])
def detect_frame():
    """Detect one arbitrary frame (browsing/overlay) and report the frame count.

    Does NOT replace the datum cache used by ``snap_fiducial`` — world-frame fiducials
    are picked on the datum frame; this route is for inspection.
    """
    data = request.get_json() or {}
    try:
        cfg, board, params, detector = _resolve_board(
            data.get, data.get("board_params")
        )
    except ValueError as exc:
        # See detect_datum: incomplete board geometry is routine, not a fault.
        logger.warning("calibration detect_frame: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 200
    camera = int(data.get("camera") or _rig(data.get).get("camera") or 1)
    source_idx = _source_idx(data.get)
    frame = int(data.get("frame", 1))
    frame_count, frame_count_error = _frame_count_or_error(camera, source_idx)
    try:
        img = _load_one(
            camera,
            frame,
            source_idx,
            data.get("image_format"),
            data.get("image_type"),
            normalize_uint8=False,
        )
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(exc),
                    "frame_count": frame_count,
                    "frame_count_error": frame_count_error,
                }
            ),
            200,
        )
    det = detector.detect(img)
    h, w = np.asarray(img).shape[:2]
    if not det.success:
        error, diagnostics = _board_not_detected(detector, img, det)
        return (
            jsonify(
                {
                    "success": False,
                    "error": error,
                    "frame": int(frame),
                    "frame_count": frame_count,
                    "frame_count_error": frame_count_error,
                    "width": int(w),
                    "height": int(h),
                    "diagnostics": diagnostics,
                }
            ),
            200,
        )
    return jsonify(
        {
            "success": True,
            "camera": camera,
            "board": board,
            "frame": int(frame),
            "frame_count": frame_count,
            "frame_count_error": frame_count_error,
            "n_points": det.n,
            "width": int(w),
            "height": int(h),
            "image_points": det.image_points.tolist(),
            "grid_indices": det.grid_indices.tolist(),
        }
    )


@calibration_bp.route("/calibration/detect_views", methods=["POST"])
def detect_views():
    """Detect every calibration view for one camera (mono) or a camera pair (stereo) and
    write the result through to the ``inputs.mat`` sidecar that ``generate_model`` reads.

    Backs the "Detect Dots" button. The request shape mirrors ``generate_model``:
    ``camera`` for mono, ``stereo`` + ``camera_pair`` for stereo, so the detections land in
    the same sidecar the solve will look in, under the same ``det_key``. The response carries
    the successful views per camera for the overlay; the sidecar receives the full per-view
    list with failed detections kept in place (a board legitimately absent from a view is
    dropped by the solve, not an error). Stored world-frame clicks are left untouched.

    A view that fails to LOAD leaves the set incomplete: the overlay still shows what
    detected, but the sidecar keeps its previous contents -- a partial set must never
    overwrite a complete one -- and ``persisted`` is false with the reason. ``detect_frame``
    (single-frame browsing) stays overlay-only for the same reason. Like ``detect_frame``
    this does NOT touch the datum cache used by ``snap_fiducial``.
    """
    data = request.get_json() or {}
    try:
        # The detector instance is deliberately unused: detection runs through
        # _detect_parallel, which builds one per task because cv2 detectors are
        # not documented thread-safe. _resolve_board is still what validates the
        # board geometry and yields the board name + params the pool needs.
        cfg, board, params, _ = _resolve_board(data.get, data.get("board_params"))
    except ValueError as exc:
        # See detect_datum: incomplete board geometry is routine, not a fault.
        logger.warning("calibration detect_views: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 200
    source_idx = _source_idx(data.get)
    image_format = data.get("image_format")
    image_type = data.get("image_type")
    stereo = bool(data.get("stereo", False))
    if stereo:
        try:
            cameras = list(_require_camera_pair(data.get))
        except ValueError as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
    else:
        cameras = [int(data.get("camera") or _rig(data.get).get("camera") or 1)]
    try:
        frame_total, frame_total_source = (
            _pair_frame_total(data.get, cameras, source_idx)
            if stereo
            else _frame_total(data.get, cameras[0], source_idx)
        )
    except ValueError as exc:
        logger.warning("calibration detect_views: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 200

    by_camera: dict[str, dict] = {}
    detections: dict[int, list] = {}
    image_size_by_cam: dict[int, tuple] = {}
    unloadable: list[str] = []
    for camera in cameras:
        frames: dict[str, dict] = {}
        imgs: list = []
        loaded: list[int] = []
        width = height = 0
        for frame in range(1, frame_total + 1):
            try:
                img = _load_one(
                    camera,
                    frame,
                    source_idx,
                    image_format,
                    image_type,
                    normalize_uint8=False,
                )
            except (FileNotFoundError, ValueError, IndexError) as exc:
                unloadable.append(f"cam{camera} frame {frame}: {exc}")
                continue
            imgs.append(img)
            loaded.append(frame)
            h, w = np.asarray(img).shape[:2]
            width, height = int(w), int(h)

        # Detection is the expensive half and each view is independent, so the
        # whole set goes through the shared pool — the same call generate_model
        # already makes (and the reason detections are built there per task:
        # cv2 detector objects are not documented thread-safe). Results come back
        # in image order, so ``loaded`` still lines up with ``dets``. A 40-view
        # Detect Dots was ~30 s sequential on gF_Ramp.
        dets: list = _detect_parallel(board, params, imgs)
        for frame, det in zip(loaded, dets):
            if det.success:
                frames[str(frame)] = {
                    "image_points": det.image_points.tolist(),
                    "grid_indices": det.grid_indices.tolist(),
                    "n_points": det.n,
                }
        by_camera[str(camera)] = {
            "frames": frames,
            "n_detected": len(frames),
            "width": width,
            "height": height,
        }
        detections[camera] = dets
        image_size_by_cam[camera] = (width, height)

    persisted = False
    persist_skipped: Optional[str] = None
    if unloadable:
        persist_skipped = "incomplete set, not persisted: " + "; ".join(unloadable)
    else:
        source = _source_path(source_idx)
        model_dir = (
            rec.stereo_model_dir_for_source(source, cameras[0], cameras[1], board)
            if stereo
            else rec.mono_model_dir_for_source(source, cameras[0], board)
        )
        det_key = joint_det_key(
            board,
            frame_total,
            image_format,
            infer_image_type(image_format),
            cameras,
            params,
            zero_based=_zero_based(source_idx),
        )
        try:
            save_inputs(
                model_dir,
                path_type="stereo" if stereo else "mono",
                board_type=board,
                detections=detections,
                image_size_by_cam=image_size_by_cam,
                det_key=det_key,
                board_params=rec.geometry_meta(board, params),
            )
            persisted = True
        except OSError as exc:
            # A read-only source: the overlay is still valid, the next Generate re-detects.
            logger.warning("calibration detect_views: sidecar not written: %s", exc)
            persist_skipped = f"sidecar not written: {exc}"
    return jsonify(
        {
            "success": True,
            "board": board,
            "cameras": cameras,
            "frame_total": frame_total,
            "frame_total_source": frame_total_source,
            "by_camera": by_camera,
            "persisted": persisted,
            "persist_skipped": persist_skipped,
        }
    )


@calibration_bp.route("/calibration/restore_detections", methods=["POST"])
def restore_detections():
    """Return the detections already stored in ``inputs.mat`` — detecting nothing.

    The read-only twin of ``detect_views``: same request body, same ``by_camera``
    response shape, but no image is opened and no detector is built. This backs the
    overlay a tab paints when it opens, so re-opening a calibrated source stops
    re-detecting points that are already on disk two directories away.

    POST rather than GET on purpose. The freshness gate is a ``joint_det_key`` that
    folds in ``repr(params)`` — a nested board-geometry dict resolved by
    ``_resolve_board``. Taking the same JSON body ``detect_views`` takes means the
    key is reproduced by construction, so a mismatch can only mean the stored
    detections are genuinely stale, never a query-string encoding artefact. It also
    keeps the response uncacheable by the browser, which matters for a read of a
    file that Re-detect rewrites.

    Always HTTP 200, like ``stepped/restore_sequence``: this fires on every tab
    open, and a 404 on first use would be noise, not information.

    - ``exists: false`` — nothing usable on disk: first use, unreadable sidecar, or
      a camera the stored set does not cover.
    - ``fresh: false`` — stored, but under a different ``det_key``. **No points are
      returned.** ``generate_model`` gates its own cache on the same key
      (:``cache_hit`` below), so drawing those dots would misrepresent what Generate
      is about to solve. ``stored`` reports the saved view count and geometry so the
      GUI can say what changed rather than just "stale".
    - ``fresh: true`` — ``by_camera`` exactly as ``detect_views`` returns it.

    KNOWN LIMITATION, shared with ``generate_model`` and not introduced here: the
    ``det_key`` carries no image mtime or hash, so re-shot images at the same path
    with unchanged settings still read as fresh. Re-detect is the remedy.
    """
    data = request.get_json() or {}
    try:
        cfg, board, params, detector = _resolve_board(
            data.get, data.get("board_params")
        )
    except ValueError as exc:
        # Incomplete board geometry (no dot spacing entered yet) is the routine
        # first-visit state. Without geometry there is no det_key to compare, so
        # there is nothing to restore — not an error.
        logger.warning("calibration restore_detections: %s", exc)
        return jsonify({"exists": False}), 200

    source_idx = _source_idx(data.get)
    image_format = data.get("image_format")
    stereo = bool(data.get("stereo", False))
    if stereo:
        try:
            cameras = list(_require_camera_pair(data.get))
        except ValueError as exc:
            # Same contract as the other "nothing to restore" answers: 200, with the reason.
            logger.warning("calibration restore_detections: %s", exc)
            return jsonify({"exists": False, "error": str(exc)}), 200
    else:
        cameras = [int(data.get("camera") or _rig(data.get).get("camera") or 1)]
    try:
        frame_total, frame_total_source = (
            _pair_frame_total(data.get, cameras, source_idx)
            if stereo
            else _frame_total(data.get, cameras[0], source_idx)
        )
    except ValueError as exc:
        # Without a view count there is no det_key to compare: nothing to restore,
        # but the reason is returned so the GUI can say why.
        logger.warning("calibration restore_detections: %s", exc)
        return jsonify({"exists": False, "error": str(exc)}), 200

    try:
        source = _source_path(source_idx)
    except (ValueError, IndexError) as exc:
        logger.warning("calibration restore_detections: %s", exc)
        return jsonify({"exists": False}), 200

    model_dir = (
        rec.stereo_model_dir_for_source(source, cameras[0], cameras[1], board)
        if stereo
        else rec.mono_model_dir_for_source(source, cameras[0], board)
    )
    side = try_load_inputs(model_dir)
    if side is None or not side.detections:
        return jsonify({"exists": False}), 200

    # Every requested camera must be present. A half-restored stereo pair is
    # worse than none: the two cameras are detected and persisted as ONE set
    # under one det_key, and the solve expects both.
    if any(cam not in side.detections for cam in cameras):
        return jsonify({"exists": False}), 200

    # Identical to detect_views' construction (see there) so a sidecar written by
    # the CLI, by Detect Dots, or by a solve is matched by all of them.
    det_key = joint_det_key(
        board,
        frame_total,
        image_format,
        infer_image_type(image_format),
        cameras,
        params,
        zero_based=_zero_based(source_idx),
    )
    if side.det_key != det_key:
        stored_views = len(side.detections[cameras[0]])
        return (
            jsonify(
                {
                    "exists": True,
                    "fresh": False,
                    "frame_total": frame_total,
                    "frame_total_source": frame_total_source,
                    "by_camera": None,
                    "stale_reason": (
                        "saved detections were made with different settings"
                    ),
                    "stored": {
                        "det_key": side.det_key,
                        "n_views": stored_views,
                        "board_params": side.board_params,
                    },
                }
            ),
            200,
        )

    by_camera: dict[str, dict] = {}
    for camera in cameras:
        dets = side.detections[camera]
        frames: dict[str, dict] = {}
        for view, det in enumerate(dets):
            # A failed view is stored in place as DetectionResult(success=False),
            # never as None, so the list stays dense and position i is frame i+1.
            # That mapping holds because detect_views only persists a COMPLETE
            # set: an unloadable view skips the write entirely, so no position
            # can shift. Omit failures rather than emitting an empty point list —
            # both render as "no detection", but only omission is honest about
            # there being nothing there.
            if not det.success:
                continue
            frames[str(view + 1)] = {
                "image_points": det.image_points.tolist(),
                "grid_indices": det.grid_indices.tolist(),
                "n_points": det.n,
            }
        width, height = side.image_size_by_cam.get(camera, (0, 0))
        by_camera[str(camera)] = {
            "frames": frames,
            "n_detected": len(frames),
            "width": int(width),
            "height": int(height),
        }

    return (
        jsonify(
            {
                "exists": True,
                "fresh": True,
                "board": board,
                "cameras": cameras,
                "det_key": det_key,
                "frame_total": frame_total,
                "frame_total_source": frame_total_source,
                "n_views": len(side.detections[cameras[0]]),
                "by_camera": by_camera,
                "stale_reason": None,
            }
        ),
        200,
    )


@calibration_bp.route("/calibration/snap_fiducial", methods=["POST"])
def snap_fiducial():
    """Snap a click to the nearest detected dot; return its pixel + grid index.

    ``click_x``/``click_y`` are required. The camera, source, board and datum frame resolve
    exactly as in ``detect_datum`` (request, else settings rig / calibration pointer), and
    the cached datum detection must have been made for that same key -- a detection of
    another source, board or datum frame is refused, never snapped against.
    """
    data = request.get_json() or {}
    missing = [k for k in ("click_x", "click_y") if data.get(k) is None]
    if missing:
        return jsonify({"error": f"{' and '.join(missing)} required"}), 400
    cx, cy = float(data["click_x"]), float(data["click_y"])
    camera = int(data.get("camera") or _rig(data.get).get("camera") or 1)
    key = (
        camera,
        int(_source_idx(data.get)),
        str(_request_board(data.get)[0]),
        int(_datum_frame(data.get)),
    )
    with _datum_lock:
        cached = _datum_cache.get(camera)
    if cached is None or cached[0] != key:
        return (
            jsonify(
                {
                    "error": (
                        f"no datum detection for camera {camera} on source {key[1]}, "
                        f"board {key[2]}, datum frame {key[3]} -- detect the datum first"
                    )
                }
            ),
            400,
        )
    det = cached[1]
    idx = WF._snap(det.image_points, (cx, cy))
    px = det.image_points[idx]
    gi = det.grid_indices[idx]
    return jsonify(
        {
            "snapped_x": float(px[0]),
            "snapped_y": float(px[1]),
            "grid_col": int(gi[0]),
            "grid_row": int(gi[1]),
        }
    )


# ---------------------------------------------------------------------------
# Model generation
# ---------------------------------------------------------------------------


def _fit_knobs(source_idx: int, board: str, data: dict) -> dict:
    """Fit knobs for the mono/stereo Calibrator, read exactly as the CLI reads them.

    ``calibration_cli`` takes ``distortion_model``, ``fix_aspect_ratio`` and
    ``use_release_object`` from the sidecar's ``fit`` block and ``fix_k2`` from the
    board's ``methods`` block. Until 2026-08-26 this route hard-coded the distortion model
    and left the other two at the dataclass defaults, so a sidecar set through the CLI
    calibrated differently on GUI regenerate. A ``fix_k2`` in the request still wins
    (the stereo hooks send it); it falls back to the sidecar, never to a literal.
    """
    knobs = cs.fit_knobs(_settings_idx(source_idx), board)
    fix_k2 = data.get("fix_k2")
    if fix_k2 is not None:
        knobs["fix_k2"] = bool(fix_k2)
    knobs["distortion_model"] = DistortionModel(knobs["distortion_model"])
    return knobs


def _persist_generate_inputs(
    model_dir: Path, request_coords: Any, used_coords: Any, fields: dict
) -> None:
    """Write a generate's detections + picks into the sidecar under its lock.

    The solve read the sidecar before it ran. Writing ``used_coords`` back after it would
    overwrite any pick committed meanwhile (another tab, the CLI) with that stale read. So
    the coords are only written when the REQUEST carried picks; otherwise the stored ones
    are left untouched, and if they no longer match what the solve used the generate is
    refused before the model is saved (the model would carry picks the sidecar no longer
    holds).
    """

    def _mutate(prev) -> dict:
        out = dict(fields)
        if request_coords is not None:
            out["coords"] = request_coords
        elif (prev.coords if prev else None) != used_coords:
            raise RuntimeError(
                f"the world-frame picks in {model_dir / 'inputs.mat'} changed while this "
                "model was being solved (saved from another tab or the CLI) -- nothing was "
                "saved; re-run Generate"
            )
        return out

    update_inputs(model_dir, _mutate)


def _mono_model_type(data: dict, board: str) -> str:
    """Mono model type: the request's ``model_type``, else the settings sidecar's
    ``methods.<board>.model_type`` (the CLI rule). An explicit ``null`` is an error, not
    "use the default"."""
    if "model_type" in data:
        if data["model_type"] is None:
            raise ValueError(
                "model_type is null -- send a model type, or omit the key to use the "
                f"settings methods.{board}.model_type"
            )
        return str(data["model_type"])
    mt = (_methods(data.get).get(board) or {}).get("model_type")
    if not mt:
        raise ValueError(
            f"no model_type in the request and none in the settings methods.{board}"
        )
    return str(mt)


def _generate_mono_camera(
    data: dict,
    camera: int,
    clicks_request: Optional[dict],
    *,
    source: Path,
    source_idx: int,
    board: str,
    params: Any,
    detector: Any,
    spacing: float,
    model_type: str,
    make_figs: bool,
    datum_frame: int,
    datum_index: int,
    gen_dt: float,
    image_format: Optional[str],
    image_type: Optional[str],
) -> dict:
    """Solve, persist and summarise ONE camera's mono model; returns the response dict.

    ``clicks_request`` is this camera's world-frame picks from the request (None: the
    picks stored in the sidecar, if any). Raises on any failure; the caller reports it.
    """
    model_dir = rec.mono_model_dir_for_source(source, camera, board)
    # Detection sidecar (parity with stereo/joint): reuse stored detections + clicks when
    # the request params still match (det_key), so a model can be regenerated without
    # re-detecting or re-clicking — and, with figures off, without the images on disk.
    # force_redetect bypasses the cached DETECTIONS only; the stored clicks are kept.
    force_redetect = bool(data.get("force_redetect", False))
    side = try_load_inputs(model_dir)
    cached = (side.detections or {}).get(camera) if side else None
    # View count from the request/config/auto-count. When the auto-count fails and the
    # sidecar holds a cached set, the cached count is used so a re-solve still runs after
    # the images are gone: the sidecar stores every view (failed ones in place), so its
    # length IS the view count it was detected with, and the det_key gate below still
    # rejects it unless every other detection parameter matches. Logged, not silent.
    try:
        frame_total, frame_total_source = _frame_total(data.get, camera, source_idx)
    except ValueError as exc:
        if not cached:
            raise
        frame_total = len(cached)
        frame_total_source = FRAME_TOTAL_FROM_SIDECAR
        logger.warning(
            "generate_model: %s; using the %d views cached in the sidecar",
            exc,
            frame_total,
        )
    if not (0 <= datum_index < frame_total):
        return {
            "success": False,
            "error": f"datum frame {datum_frame} is outside 1..{frame_total}",
        }
    det_key = joint_det_key(
        board,
        frame_total,
        image_format,
        infer_image_type(image_format),
        [camera],
        params,
        zero_based=_zero_based(source_idx),
    )
    cached_size = None
    cache_hit = (
        not force_redetect
        and side is not None
        and side.det_key == det_key
        and bool(cached)
    )
    if cache_hit:
        cached_size = side.image_size_by_cam.get(camera)
    used_coords = side.coords if side else None
    # World-frame picks: the live request, else the stored coords (re-solve w/o re-click).
    clicks_payload = clicks_request or used_coords
    # Load images only to detect or to draw figures; a cached re-solve with figures off
    # touches no image files. (A cached hit lacking a stored size also needs an image.)
    # Whenever images are needed a missing one raises naming its file -- a re-solve after
    # the images were removed must be run with figures off.
    need_images = make_figs or not cache_hit or cached_size is None
    imgs = (
        _load_views(camera, frame_total, source_idx, image_format, image_type)
        if need_images
        else []
    )
    fig_dir = (model_dir.parent / "figures") if make_figs else None
    calr = Calibrator(
        detector=detector,
        board_type=board,
        model_type=model_type,
        **_fit_knobs(source_idx, board, data),
    )
    dets = list(cached) if cache_hit else _detect_parallel(board, params, imgs)
    record = calr.run_mono(
        imgs,
        camera=camera,
        clicks=_clicks_from(clicks_payload),
        origin_mm=_origin_mm_from(clicks_payload),
        datum_index=datum_index,
        spacing_mm=spacing,
        image_size=cached_size,
        figure_dir=fig_dir,
        detections=dets,
    )
    record.board_meta["dt"] = gen_dt
    if image_format:
        record.board_meta["image_format"] = str(image_format)
    isz = record.camera_model.image_size
    _persist_generate_inputs(
        model_dir,
        clicks_request,
        used_coords,
        {
            "path_type": "mono",
            "board_type": board,
            "detections": {camera: list(dets)},
            "image_size_by_cam": {camera: (int(isz[0]), int(isz[1]))},
            "det_key": det_key,
            "board_params": rec.geometry_meta(board, params),
        },
    )
    path = rec.save_mono(record, model_dir)
    cm = record.camera_model
    resp = {
        "success": True,
        "stereo": False,
        "model_path": str(path),
        "camera": camera,
        "num_images_used": len(record.per_view_rms),
        "per_view_rms": list(record.per_view_rms),
        "detections_cached": bool(cache_hit),
        "frame_total": frame_total,
        "frame_total_source": frame_total_source,
        "figures": _list_figures(fig_dir),
    }
    if isinstance(cm, PolynomialModel):
        resp.update(_polynomial_summary(cm))
    elif isinstance(cm, Polynomial3DModel):
        resp.update(_polynomial3d_summary(cm))
    else:
        resp.update(
            {
                "model_type": "pinhole",
                "rms": cm.rms,
                "fx": float(cm.K[0, 0]),
                "fy": float(cm.K[1, 1]),
                "cx": float(cm.K[0, 2]),
                "cy": float(cm.K[1, 2]),
                "camera_matrix": cm.K.tolist(),
                "dist_coeffs": cm.dist.tolist(),
                "image_width": int(cm.image_size[0]),
                "image_height": int(cm.image_size[1]),
            }
        )
    resp["rms_units"] = _rms_units(resp["model_type"])
    return resp


def _stereo_dotboard_missing_picks(cam1: int, cam2: int, missing: List[int]) -> str:
    """Refusal text for a stereo dotboard generate without picks on every camera."""
    which = " and ".join(str(c) for c in missing)
    noun = "camera" if len(missing) == 1 else "cameras"
    return (
        f"stereo dotboard needs world-frame picks on {noun} {which}: dotboard grid indices "
        f"are re-zeroed per view, so camera {cam2}'s origin/+X/+Y must mark the same "
        f"physical dots as camera {cam1}'s"
    )


@calibration_bp.route("/calibration/generate_model", methods=["POST"])
def generate_model():
    """Run the full calibration (mono or stereo), save it + proof figures into the source.

    Mono: one ``camera`` (else the rig camera); the GUI solves several cameras with one
    request each.

    Stereo: ``clicks`` (camera 1, carries ``origin_mm``) and ``clicks2`` (camera 2). A
    dotboard pair needs both (grid indices are re-zeroed per view) and is refused with a
    400 naming the camera without picks; ChArUco needs at most camera 1's. Without picks in
    the request the sidecar's per-camera picks are used.
    """
    data = request.get_json() or {}
    try:
        cfg, board, params, detector = _resolve_board(
            data.get, data.get("board_params")
        )
    except ValueError as exc:
        # See detect_datum: incomplete board geometry is routine, not a fault.
        logger.warning("calibration generate_model: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 200
    source_idx = _source_idx(data.get)
    image_format, image_type = data.get("image_format"), data.get("image_type")
    stereo = bool(data.get("stereo", False))
    model_type = None
    if not stereo:  # stereo is pinhole only; model_type selects the mono fit
        try:
            model_type = _mono_model_type(data, board)
        except ValueError as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
    make_figs = not bool(data.get("no_figures", False))
    spacing = c2._spacing_mm(board, params)
    datum_frame = _datum_frame(data.get)
    datum_index = datum_frame - 1  # position within the loaded views (frames 1..N)

    try:
        source = _source_path(source_idx)
        # Resolve dt up front: a missing dt must fail before the expensive
        # detection pass, while the user is still at the form.
        gen_dt = _generate_dt(data.get, source)
        if stereo:
            cam1, cam2 = _require_camera_pair(data.get)
            frame_total, frame_total_source = _pair_frame_total(
                data.get, [cam1, cam2], source_idx
            )
            if not (0 <= datum_index < frame_total):
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": f"datum frame {datum_frame} is outside 1..{frame_total}",
                        }
                    ),
                    200,
                )
            model_dir = rec.stereo_model_dir_for_source(source, cam1, cam2, board)
            # The sidecar is always read: force_redetect bypasses its DETECTIONS only. The
            # picked world frames stored beside them must survive a Re-detect without a
            # re-pick.
            force_redetect = bool(data.get("force_redetect", False))
            side = try_load_inputs(model_dir)
            used_coords = side.coords if side else None
            # World-frame picks per camera: the live request, else the sidecar's (so a
            # deleted-model re-solve rebuilds the shared frame without re-picking).
            request_coords = _stereo_coords(
                cam1, cam2, data.get("clicks"), data.get("clicks2")
            )
            coords = request_coords if request_coords is not None else used_coords
            clicks1 = _stereo_clicks(coords, cam1)
            clicks2 = _stereo_clicks(coords, cam2)
            if board == "dotboard":
                missing = [
                    c for c, k in ((cam1, clicks1), (cam2, clicks2)) if k is None
                ]
                if missing:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": _stereo_dotboard_missing_picks(
                                    cam1, cam2, missing
                                ),
                            }
                        ),
                        400,
                    )
            imgs1 = _load_views(cam1, frame_total, source_idx, image_format, image_type)
            imgs2 = _load_views(cam2, frame_total, source_idx, image_format, image_type)
            fig_dir = (model_dir.parent / "figures") if make_figs else None
            # Detection sidecar/cache (parity with the joint path): reuse stored detections
            # when the request params still match (det_key), else detect fresh. Persisted
            # below so a re-run skips detection and the detections survive a model delete.
            det_key = joint_det_key(
                board,
                frame_total,
                image_format,
                infer_image_type(image_format),
                [cam1, cam2],
                params,
                zero_based=_zero_based(source_idx),
            )
            cache_hit = (
                not force_redetect
                and side is not None
                and side.det_key == det_key
                and bool(side.detections)
                and cam1 in side.detections
                and cam2 in side.detections
            )
            if cache_hit:
                det1, det2 = side.detections[cam1], side.detections[cam2]
            else:
                det1 = _detect_parallel(board, params, imgs1)
                det2 = _detect_parallel(board, params, imgs2)
            sc = StereoCalibrator(
                detector=detector,
                board_type=board,
                **_fit_knobs(source_idx, board, data),
            )
            record = sc.run_stereo(
                imgs1,
                imgs2,
                cam1=cam1,
                cam2=cam2,
                clicks=clicks1,
                clicks2=clicks2,
                origin_mm=_stereo_origin_mm(coords),
                datum_index=datum_index,
                spacing_mm=spacing,
                figure_dir=fig_dir,
                det1=det1,
                det2=det2,
            )
            method = _require_meta(record.board_meta, "stereo_method", "stereo solve")
            n_pairs = int(
                _require_meta(record.board_meta, "n_stereo_views", "stereo solve")
            )
            record.board_meta["dt"] = gen_dt
            if image_format:
                record.board_meta["image_format"] = str(image_format)
            _persist_generate_inputs(
                model_dir,
                request_coords,
                used_coords,
                {
                    "path_type": "stereo",
                    "board_type": board,
                    "detections": {cam1: list(det1), cam2: list(det2)},
                    "image_size_by_cam": {
                        cam1: (
                            int(record.model1.image_size[0]),
                            int(record.model1.image_size[1]),
                        ),
                        cam2: (
                            int(record.model2.image_size[0]),
                            int(record.model2.image_size[1]),
                        ),
                    },
                    "det_key": det_key,
                    "board_params": rec.geometry_meta(board, params),
                },
            )
            path = rec.save_stereo(record, model_dir)
            ang = float(
                np.degrees(
                    np.arccos(np.clip((np.trace(record.R_stereo) - 1) / 2, -1, 1))
                )
            )
            return jsonify(
                {
                    "success": True,
                    "stereo": True,
                    "model_path": str(path),
                    "rms_cam1": record.model1.rms,
                    "rms_cam2": record.model2.rms,
                    "rms_units": "px",
                    "per_view_rms1": list(record.per_view_rms1),
                    "per_view_rms2": list(record.per_view_rms2),
                    "intrinsics1": _intrinsics(record.model1),
                    "intrinsics2": _intrinsics(record.model2),
                    "num_pairs_used": n_pairs,
                    "stereo_angle_deg": ang,
                    "baseline_mm": float(np.linalg.norm(record.T_stereo)),
                    "stereo_rms_px": _finite_or_none(
                        record.board_meta.get("stereo_rms_px")
                    ),
                    "method": method,
                    "detections_cached": bool(cache_hit),
                    "frame_total": frame_total,
                    "frame_total_source": frame_total_source,
                    "figures": _list_figures(fig_dir),
                }
            )

        camera = int(data.get("camera") or _rig(data.get).get("camera") or 1)
        return jsonify(
            _generate_mono_camera(
                data,
                camera,
                data.get("clicks"),
                source=source,
                source_idx=source_idx,
                board=board,
                params=params,
                detector=detector,
                spacing=spacing,
                model_type=model_type,
                make_figs=make_figs,
                datum_frame=datum_frame,
                datum_index=datum_index,
                gen_dt=gen_dt,
                image_format=image_format,
                image_type=image_type,
            )
        )
    except (
        Exception
    ) as exc:  # surface failures to the GUI (full traceback to the server log)
        logger.exception("generate_model failed")
        return jsonify({"success": False, "error": str(exc)}), 200


@calibration_bp.route("/calibration/scale_factor/generate", methods=["POST"])
def scale_factor_generate():
    """Build + save a scale-factor mono model from picked origin/axes + px/mm + dt.

    No detection, no fit — a direct uniform pixel->mm map. One frame is loaded only to
    draw the proof figure; the model itself needs no image. An explicit
    ``use_image: false`` (the GUI's "Use calibration images" toggle off) skips the
    frame entirely and no figure is drawn.
    """
    data = request.get_json() or {}
    source_idx = _source_idx(data.get)
    camera = int(data.get("camera") or _rig(data.get).get("camera") or 1)
    make_figs = not bool(data.get("no_figures", False))
    try:
        px_per_mm = float(data["px_per_mm"])
        origin_px = data["origin_px"]
        if origin_px is None or len(origin_px) != 2:
            raise ValueError("origin_px must be [x, y] pixels — click the origin first")
        x_dir = str(data.get("x_dir", "right"))
        y_dir = str(data.get("y_dir", "up"))
        swap = bool(data.get("swap_axes", False))
        frame_idx = int(data.get("frame_idx", 1))
        origin_mm_raw = data.get("origin_mm")
        if origin_mm_raw is not None and len(origin_mm_raw) != 2:
            raise ValueError("origin_mm must be [X, Y] millimetres")
        origin_mm = (
            (float(origin_mm_raw[0]), float(origin_mm_raw[1]))
            if origin_mm_raw is not None
            else (0.0, 0.0)
        )
        source = _source_path(source_idx)
        dt = _generate_dt(data.get, source)
        model_dir = rec.mono_model_dir_for_source(source, camera, "scale_factor")
        use_image = bool(data.get("use_image", True))
        image = (
            _load_one(
                camera,
                frame_idx,
                source_idx,
                data.get("image_format"),
                data.get("image_type"),
            )
            if use_image
            else None
        )
        record = build_scale_factor_record(
            camera=camera,
            origin_px=origin_px,
            px_per_mm=px_per_mm,
            dt=dt,
            x_dir=x_dir,
            y_dir=y_dir,
            swap_axes=swap,
            frame_idx=frame_idx,
            origin_mm=origin_mm,
        )
        path = rec.save_mono(record, model_dir)
        fig_dir = model_dir.parent / "figures"
        if make_figs and image is not None:
            from pivtools_gui.calibration import figures as c2figs

            sf = record.camera_model
            # Draw at the PICKED origin (world_frame) — the model's own origin_px is the
            # world-zero pixel, off the picked point (possibly off-image) when origin_mm != 0.
            c2figs.write_scale_factor_figure(
                fig_dir,
                image=image,
                origin_px=record.world_frame.origin_px,
                col_sign=sf.col_sign,
                row_sign=sf.row_sign,
                swap_axes=bool(sf.swap_axes),
                mm_per_pixel=sf.mm_per_pixel,
                dt=dt,
                origin_mm=origin_mm,
            )
        resp = {
            "success": True,
            "stereo": False,
            "model_path": str(path),
            "camera": camera,
        }
        resp.update(
            _scale_factor_summary(
                record.camera_model, dt, frame_idx=frame_idx, wf=record.world_frame
            )
        )
        resp["figures"] = (
            _list_figures(fig_dir) if (make_figs and image is not None) else []
        )
        return jsonify(resp)
    except Exception as exc:
        logger.exception("scale_factor_generate failed")
        return jsonify({"success": False, "error": str(exc)}), 200


def _optical_axis_angle_deg(model1: CameraModel, model2: CameraModel) -> float:
    """Angle (deg) between two pinhole cameras' optical axes in the world frame.

    ``R.T @ [0, 0, 1]`` is a camera's optical axis in world coordinates; the same formula
    ``stepped_calibrate`` stamps as ``board_meta["relative_angle_deg"]``.
    """
    z = np.array([0.0, 0.0, 1.0])
    a1 = np.asarray(model1.R).T @ z
    a2 = np.asarray(model2.R).T @ z
    return float(np.degrees(np.arccos(np.clip(float(a1 @ a2), -1.0, 1.0))))


@calibration_bp.route("/calibration/model", methods=["GET"])
def load_model():
    """Return a summary of a saved model (mono or stereo) for the summary panel."""
    board, active_stereo = _request_board(request.args.get)
    try:
        source = _source_path(_source_idx(request.args.get))
    except (ValueError, IndexError):
        return jsonify({"exists": False}), 200
    stereo_arg = request.args.get("stereo")
    stereo = (
        stereo_arg in ("1", "true", "True")
        if stereo_arg not in (None, "")
        else bool(active_stereo)
    )
    if stereo:
        try:
            cam1, cam2 = _required_camera_pair(request.args.get)
        except ValueError as exc:
            return jsonify({"exists": False, "error": str(exc)}), 200
        mdir = rec.stereo_model_dir_for_source(source, cam1, cam2, board)
        try:
            mpath = rec.resolve_stereo_path(
                mdir, _model_type_arg(request.args.get, board, stereo=True)
            )
            r = rec.load_stereo(mpath)
            # rms_cam1 / intrinsics1 are model1's: label them with the record's cam1.
            cam1, cam2 = rec.record_pair(r, (cam1, cam2))
        except FileNotFoundError:
            # No model: both cameras' picks come back from the sidecar so the GUI can
            # restore them and re-solve without re-picking.
            try:
                return (
                    jsonify(
                        {
                            "exists": False,
                            "world_frame": _inputs_world_frame(mdir, camera=cam1),
                            "world_frame2": _inputs_world_frame(mdir, camera=cam2),
                        }
                    ),
                    200,
                )
            except ValueError as exc:  # a stale (flat) stereo sidecar
                return jsonify({"exists": False, "error": str(exc)}), 200
        except ValueError as exc:  # several types and none requested, or other cameras
            return jsonify({"exists": False, "error": str(exc)}), 200
        try:
            # Camera 2's frame is an input only (the record stores camera 1's): it comes
            # from the sidecar picks. Pair counts and the stereo method are what the
            # solve stored, never a stand-in.
            world_frame2 = _inputs_world_frame(mdir, camera=cam2)
            num_pairs_used = int(
                _require_meta(r.board_meta, "n_stereo_views", str(mpath))
            )
            stereo_method = _require_meta(r.board_meta, "stereo_method", str(mpath))
        except ValueError as exc:
            return jsonify({"exists": False, "error": str(exc)}), 200
        common = {
            "exists": True,
            "stereo": True,
            "cam1": cam1,
            "cam2": cam2,
            "model_path": str(mpath),
            "per_view_rms1": list(r.per_view_rms1),
            "per_view_rms2": list(r.per_view_rms2),
            "rms_units": "px",
            "num_pairs_used": num_pairs_used,
            "world_frame_mode": r.world_frame.mode,
            "world_frame": _world_frame_payload(r.world_frame),
            "world_frame2": world_frame2,
            "image_width": int(r.model1.image_size[0]),
            "image_height": int(r.model1.image_size[1]),
            "spacing_mm": _meta_float(r.board_meta, "spacing_mm"),
            "n_views": _meta_int(r.board_meta, "n_views"),
            "geometry": _geometry_payload(r.board_meta),
        }
        if isinstance(r.model1, Polynomial3DModel):
            # A polynomial pair has no extrinsic pose -> no baseline/angle (DaVis poly).
            common.update(
                {
                    "model_type": "polynomial3d",
                    "rms_cam1": float(r.model1.rms_px),
                    "rms_cam2": float(r.model2.rms_px),
                    "plane_rms_cam1": [float(v) for v in r.model1.plane_rms_px],
                    "plane_rms_cam2": [float(v) for v in r.model2.plane_rms_px],
                    "stereo_config": r.board_meta.get("stereo_config"),
                    "stereo_angle_deg": None,
                    "baseline_mm": None,
                }
            )
        else:
            common.update(
                {
                    "model_type": "pinhole",
                    "rms_cam1": r.model1.rms,
                    "rms_cam2": r.model2.rms,
                    "intrinsics1": _intrinsics(r.model1),
                    "intrinsics2": _intrinsics(r.model2),
                    "distortion_model": r.model1.distortion_model.value,
                    # The PIV-meaningful angle is the optical-axis angle, derived from the
                    # stored poses (the stepped path also stamps it as relative_angle_deg
                    # with the same formula; the flat stereoCalibrate path does not). Never
                    # the R_stereo rotation angle: it conflates the axis change with camera
                    # roll (e.g. ~178deg in transmission).
                    "stereo_angle_deg": _optical_axis_angle_deg(r.model1, r.model2),
                    "baseline_mm": float(np.linalg.norm(r.T_stereo)),
                    "stereo_rms_px": _finite_or_none(r.board_meta.get("stereo_rms_px")),
                    "method": stereo_method,
                    "stereo_config": r.board_meta.get("stereo_config"),
                }
            )
        return jsonify(common)
    camera = int(request.args.get("camera", 1))
    mdir = rec.mono_model_dir_for_source(source, camera, board)
    try:
        mpath = rec.resolve_mono_path(mdir, _model_type_arg(request.args.get, board))
        r = rec.load_mono(mpath)
    except FileNotFoundError:
        return jsonify({"exists": False, "world_frame": _inputs_world_frame(mdir)}), 200
    except ValueError as exc:  # several types saved, none requested
        return jsonify({"exists": False, "error": str(exc)}), 200
    cm = r.camera_model
    summary = {
        "exists": True,
        "stereo": False,
        "camera": camera,
        "model_path": str(mpath),
        "num_images_used": len(r.per_view_rms),
        "per_view_rms": list(r.per_view_rms),
        "world_frame_mode": r.world_frame.mode,
        "world_frame": _world_frame_payload(r.world_frame),
        "spacing_mm": _meta_float(r.board_meta, "spacing_mm"),
        "n_views": _meta_int(r.board_meta, "n_views"),
        "geometry": _geometry_payload(r.board_meta),
    }
    if isinstance(cm, ScaleFactorModel):
        summary.update(
            _scale_factor_summary(
                cm,
                _meta_float(r.board_meta, "dt"),
                frame_idx=_meta_int(r.board_meta, "frame_idx"),
                wf=r.world_frame,
            )
        )
    elif isinstance(cm, Polynomial3DModel):
        summary.update(_polynomial3d_summary(cm))
    elif isinstance(cm, PolynomialModel):
        summary.update(_polynomial_summary(cm))
    else:
        summary.update(
            {
                "model_type": "pinhole",
                "rms": cm.rms,
                "fx": float(cm.K[0, 0]),
                "fy": float(cm.K[1, 1]),
                "cx": float(cm.K[0, 2]),
                "cy": float(cm.K[1, 2]),
                "camera_matrix": cm.K.tolist(),
                "dist_coeffs": cm.dist.tolist(),
                "distortion_model": cm.distortion_model.value,
                "image_width": int(cm.image_size[0]),
                "image_height": int(cm.image_size[1]),
            }
        )
    if not isinstance(cm, ScaleFactorModel):  # a scale factor has no fit residual
        summary["rms_units"] = _rms_units(summary["model_type"])
    return jsonify(summary)


# ---------------------------------------------------------------------------
# Figure serving
# ---------------------------------------------------------------------------


@calibration_bp.route("/calibration/figures", methods=["GET"])
def list_figures():
    """List the proof figures written beside a model."""
    try:
        fig_dir = _figures_dir(request.args.get)
    except (ValueError, IndexError):
        return jsonify({"exists": False, "figures": []}), 200
    figs = _list_figures(fig_dir)
    return jsonify({"exists": bool(figs), "dir": str(fig_dir), "figures": figs})


@calibration_bp.route("/calibration/figure", methods=["GET"])
def get_figure():
    """Serve one proof-figure PNG (basename only — no path traversal)."""
    name = request.args.get("name", "")
    if not name or Path(name).name != name:
        return jsonify({"error": "invalid figure name"}), 400
    try:
        fig_dir = Path(_figures_dir(request.args.get)).resolve()
    except (ValueError, IndexError):
        return jsonify({"error": "source not configured"}), 404
    path = (fig_dir / name).resolve()
    if path.parent != fig_dir or not path.is_file():
        return jsonify({"error": "figure not found"}), 404
    return Response(path.read_bytes(), mimetype="image/png")


# ---------------------------------------------------------------------------
# Global coordinates — planar N-camera stitching
# ---------------------------------------------------------------------------


class _GlobalChainError(Exception):
    """User-facing failure while resolving the global datum chain."""


def _global_chain(data):
    """Shared setup for /global/compute and /global/save.

    Returns ``(dirs, records, shifts, datum_physical)`` — the per-camera model dirs +
    loaded records, the computed per-camera (shift_x, shift_y) mm, and the datum
    physical point. Raises ``_GlobalChainError`` (with a user-facing message) on a
    missing datum, a missing model, or a broken chain.
    """
    board = _request_board(data.get)[0]
    datum_camera = int(data.get("datum_camera", 1))
    datum_pixel = data.get("datum_pixel")
    datum_physical = data.get("datum_physical", [0.0, 0.0])
    overlap_pairs = data.get("overlap_pairs") or []
    z = float(data.get("z_world", 0.0))
    tx = float(data.get("tilt_x", 0.0))
    ty = float(data.get("tilt_y", 0.0))
    if not datum_pixel:
        raise _GlobalChainError(
            "datum_pixel not set — click a point on the datum camera"
        )

    cams = {datum_camera}
    for p in overlap_pairs:
        cams.add(int(p["camera_a"]))
        cams.add(int(p["camera_b"]))
    try:
        source = _source_path(_source_idx(data.get))
        dirs = {cam: rec.mono_model_dir_for_source(source, cam, board) for cam in cams}
        mtype = _model_type_arg(data.get, board)
        records = {cam: rec.load_mono(d, model_type=mtype) for cam, d in dirs.items()}
    except (FileNotFoundError, ValueError, IndexError) as exc:
        raise _GlobalChainError(
            f"missing mono model — calibrate each camera first ({exc})"
        )
    try:
        shifts = gc2.compute_camera_shifts(
            records, datum_camera, datum_pixel, datum_physical, overlap_pairs, z, tx, ty
        )
    except (ValueError, KeyError) as exc:
        raise _GlobalChainError(str(exc))
    return dirs, records, shifts, datum_physical


def _shifts_payload(shifts, datum_physical, **extra):
    # Stringify camera keys (Flask jsonify mixes int/str keys badly).
    return {
        "camera_shifts": {
            str(c): [float(s[0]), float(s[1])] for c, s in shifts.items()
        },
        "datum_physical": [float(datum_physical[0]), float(datum_physical[1])],
        **extra,
    }


@calibration_bp.route("/calibration/global/compute", methods=["POST"])
def global_compute():
    """Per-camera world shifts from a datum + overlap pairs — preview only, persists nothing."""
    try:
        _dirs, _records, shifts, datum_physical = _global_chain(
            request.get_json() or {}
        )
    except _GlobalChainError as exc:
        return jsonify({"error": str(exc)}), 200
    return jsonify(_shifts_payload(shifts, datum_physical))


@calibration_bp.route("/calibration/global/save", methods=["POST"])
def global_save():
    """Compute the datum-chain shifts and BAKE them into each camera's model.

    Same body as ``/global/compute``, but persistent: each reachable camera's
    ``world_offset_mm`` is written into its model record so the apply step reads it
    and emits coordinates in the shared rig frame. Regenerating a camera's model
    clears its offset (fresh world frame), so re-save the global frame after any
    recalibration. Cameras not reached by the chain are left untouched.
    """
    try:
        dirs, records, shifts, datum_physical = _global_chain(request.get_json() or {})
    except _GlobalChainError as exc:
        return jsonify({"error": str(exc)}), 200

    for cam, (sx, sy) in shifts.items():
        r = records[cam]
        r.world_frame.world_offset_mm = np.array(
            [float(sx), float(sy)], dtype=np.float64
        )
        rec.save_mono(r, dirs[cam])

    return jsonify(
        _shifts_payload(
            shifts,
            datum_physical,
            success=True,
            cameras_saved=sorted(int(c) for c in shifts),
        )
    )


# ---------------------------------------------------------------------------
# Apply the model to PIV output (background job)
# ---------------------------------------------------------------------------


class _ApplyRequestError(ValueError):
    """An apply request that is malformed on its own terms (HTTP 400, nothing planned)."""


# Request keys of an explicit (ad-hoc, single-unit) apply override, per mode. All or none.
_EXPLICIT_KEYS_MONO = ("uncalibrated_dir", "calibrated_dir")
_EXPLICIT_KEYS_STEREO = (
    "uncalibrated_dir_cam1",
    "uncalibrated_dir_cam2",
    "calibrated_dir",
)


def _required_camera_pair(get: Callable[[str], Any]) -> List[int]:
    """Stereo camera pair: request ``camera_pair`` (list or "a,b"), else the settings
    sidecar's ``rig.camera_pair``; neither raises — never the template's ``[1, 2]``
    (:func:`_sidecar_rig`)."""
    pair = get("camera_pair") or _sidecar_rig(get).get("camera_pair")
    if not pair:
        raise ValueError(
            "camera_pair is required for stereo: none in the request and no settings "
            "sidecar for this source names rig.camera_pair"
        )
    if isinstance(pair, str):
        pair = pair.split(",")
    return [int(pair[0]), int(pair[1])]


def _apply_units(data, full_cfg, source, board, stereo, type_name):
    """Resolve apply units for a Flask request — thin wrapper over ``runio.plan_apply_units``.

    Translates the request body into the shared planner's args: an explicit
    ``uncalibrated_dir``/``calibrated_dir`` (mono) or ``uncalibrated_dir_cam1/2`` +
    ``calibrated_dir`` (stereo) forces a single ad-hoc unit; otherwise units are derived
    from config across ``active_paths`` x cameras. The planner does the config derivation +
    per-unit model load (so the CLI's ``--all-paths`` apply uses the identical logic).

    Raises ``_ApplyRequestError`` (a 400) for a partial explicit override — it must never
    fall through to applying every dataset — for a non-list ``active_paths`` (a scalar
    ``0`` would read as "all"), and for a stereo request with no camera pair.
    """
    get = data.get
    keys = _EXPLICIT_KEYS_STEREO if stereo else _EXPLICIT_KEYS_MONO
    given = [k for k in keys if get(k)]
    explicit = None
    if given:
        missing = [k for k in keys if k not in given]
        if missing:
            raise _ApplyRequestError(
                f"explicit apply override is incomplete: got {given}, missing "
                f"{', '.join(repr(k) for k in missing)} (give all of {list(keys)} or none)"
            )
        if stereo:
            explicit = {
                "uncal1": data["uncalibrated_dir_cam1"],
                "uncal2": data["uncalibrated_dir_cam2"],
                "out": data["calibrated_dir"],
            }
        else:
            explicit = {
                "uncal": data["uncalibrated_dir"],
                "out": data["calibrated_dir"],
            }
    active_paths = get("active_paths")
    if active_paths is not None and not isinstance(active_paths, list):
        raise _ApplyRequestError(
            f"active_paths must be a list of base-path indices, got {active_paths!r}"
        )
    camera_pair = None
    if stereo:
        try:
            camera_pair = _required_camera_pair(get)
        except ValueError as exc:
            raise _ApplyRequestError(str(exc)) from exc
    return c2runio.plan_apply_units(
        full_cfg,
        source,
        board,
        stereo,
        type_name,
        active_paths=active_paths,
        camera_pair=camera_pair,
        camera=get("camera"),
        explicit=explicit,
        model_type=_model_type_arg(get, board, stereo=stereo),
    )


@calibration_bp.route("/calibration/set_datum", methods=["POST"])
def calibration_set_datum():
    """Set a new datum (origin) and/or apply offsets to ALL runs in a type's coordinates.

    Serves the vector viewer's Coordinate System panel. The datum ``x``/``y`` (a clicked
    physical position) is subtracted from every coordinate, then ``x_offset``/``y_offset``
    is added — offsets are additive per request, not absolute. Every run in the type's
    ``coordinates.mat`` is rewritten; stereo ``z`` is preserved untouched. Statistics
    coordinates (``statistics/.../mean_stats/coordinates.mat``) are NOT modified — the
    viewer hides coordinate editing for mean-stats variables.

    JSON body: ``base_path`` (optional, wins over idx) or ``base_path_idx``; ``camera``;
    ``run`` (logging only); ``type_name`` (default ``instantaneous``); ``x``/``y``
    (optional datum, both required together); ``x_offset``/``y_offset`` (default 0);
    ``merged``; ``use_stereo`` + ``camera_pair``.
    """
    data = request.get_json() or {}
    full_cfg = get_config()

    base_path = data.get("base_path")
    if not base_path:
        base_idx = int(data.get("base_path_idx", 0))
        try:
            base_path = full_cfg.base_paths[base_idx]
        except IndexError:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"base_path_idx {base_idx} out of range",
                    }
                ),
                400,
            )

    camera = camera_number(data.get("camera", 1))
    run = int(data.get("run", 1))  # logging only: the update spans all runs
    type_name = data.get("type_name", "instantaneous")
    x0 = data.get("x")
    y0 = data.get("y")
    x_offset = float(data.get("x_offset") or 0)
    y_offset = float(data.get("y_offset") or 0)
    use_merged = bool(data.get("merged"))
    use_stereo = bool(data.get("use_stereo"))
    camera_pair = data.get("camera_pair")
    stereo_camera_pair = None
    if use_stereo and isinstance(camera_pair, (list, tuple)) and len(camera_pair) >= 2:
        stereo_camera_pair = (int(camera_pair[0]), int(camera_pair[1]))

    try:
        paths = get_data_paths(
            base_dir=Path(base_path),
            num_frame_pairs=full_cfg.num_frame_pairs,
            cam=camera,
            type_name=type_name,
            use_merged=use_merged,
            use_stereo=use_stereo,
            stereo_camera_pair=stereo_camera_pair,
        )
        coords_path = Path(paths["data_dir"]) / "coordinates.mat"
        if not coords_path.exists():
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Coordinates file not found: {coords_path}",
                    }
                ),
                404,
            )

        mat = scipy.io.loadmat(
            str(coords_path), struct_as_record=False, squeeze_me=True
        )
        if "coordinates" not in mat:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Variable 'coordinates' not found in {coords_path}",
                    }
                ),
                400,
            )
        coordinates = mat["coordinates"]
        num_runs = get_num_coordinate_runs(coordinates)
        is_multi = isinstance(coordinates, np.ndarray) and coordinates.dtype == object
        has_z = hasattr(coordinates[0] if is_multi else coordinates, "z")

        fields = [("x", object), ("y", object)] + ([("z", object)] if has_z else [])
        coords_struct = np.empty((num_runs,), dtype=fields)
        for i in range(num_runs):
            cx, cy = extract_coordinates(coordinates, i + 1)
            if x0 is not None and y0 is not None:
                cx = cx - float(x0)
                cy = cy - float(y0)
            coords_struct["x"][i] = cx + x_offset
            coords_struct["y"][i] = cy + y_offset
            if has_z:
                el = coordinates[i] if is_multi else coordinates
                coords_struct["z"][i] = np.asarray(el.z)

        scipy.io.savemat(
            str(coords_path), {"coordinates": coords_struct}, do_compression=True
        )
        logger.info(
            f"[set_datum] {type_name} Cam{camera} (from run {run}): datum=({x0}, {y0}) "
            f"offset=({x_offset}, {y_offset}) -> {num_runs} runs in {coords_path}"
        )
        return jsonify(
            {
                "success": True,
                "type_name": type_name,
                "num_runs_updated": num_runs,
                "coords_path": str(coords_path),
                "x0": x0,
                "y0": y0,
                "x_offset": x_offset,
                "y_offset": y_offset,
            }
        )
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("[set_datum] failed")
        return jsonify({"success": False, "error": "Internal server error"}), 500


def _dirs_overlap(a: Path, b: Path) -> bool:
    """True when two output directories are the same or one contains the other."""
    return a == b or a in b.parents or b in a.parents


def _overlapping_apply_job(out_dirs: List[str]) -> Optional[tuple[str, str]]:
    """First active apply job writing into any of ``out_dirs``: ``(job_id, its dir)``.

    Paths are compared resolved, so a symlinked or relative spelling of the same
    directory is still caught. Call with ``_apply_lock`` held.
    """
    mine = [Path(d).resolve() for d in out_dirs]
    for job_id, job in job_manager.list_jobs(job_type="calibration_apply").items():
        if job.get("status") not in _APPLY_ACTIVE_STATUSES:
            continue
        for other in job["out_dirs"]:
            other_path = Path(other).resolve()
            if any(_dirs_overlap(m, other_path) for m in mine):
                return job_id, other
    return None


def _apply_cancelled_message(frames_written: int, out_dirs: List[str]) -> str:
    """Cancel outcome for the job status: what stopped and what stays on disk."""
    return (
        f"cancelled by user after {frames_written} frame(s) written; frames already "
        f"written remain on disk in {'; '.join(out_dirs)}"
    )


@calibration_bp.route("/calibration/apply", methods=["POST"])
def apply_model():
    """Start applying a saved model to PIV output (mono apply / stereo 3C) as a job.

    Directories are derived from config: the model from the calibration ``source``, the
    PIV data from each base path in ``active_paths`` x ``camera_numbers`` (mono) or the
    stereo pair. One job spans all units; progress reports completed units.
    """
    data = request.get_json() or {}
    full_cfg = get_config()
    board, active_stereo = _request_board(data.get)
    stereo = (
        bool(data["stereo"]) if data.get("stereo") is not None else bool(active_stereo)
    )
    # Light-sheet plane: request > 0.0 for mono (the board plane, a geometric
    # identity); stereo additionally defaults from the record's self-cal below.
    z = float(data.get("z_world", 0.0))
    tx = float(data.get("tilt_x", 0.0))
    ty = float(data.get("tilt_y", 0.0))
    # Stereo cam2 resample kernel (validated up front, before the worker starts).
    interpolator = data.get("interpolator")
    if interpolator not in ("linear", "cubic", "lanczos", None, ""):
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"interpolator must be linear|cubic|lanczos, got {interpolator!r}",
                }
            ),
            200,
        )
    # Derive the per-frame vector glob from the PIV vector_format (e.g. "%05d.mat" ->
    # "*.mat"); the old hardcoded "B*.mat" matched nothing under the default naming, so
    # apply wrote coordinates.mat but no calibrated vectors.
    vector_glob = data.get("vector_glob") or vector_glob_from_format(
        full_cfg.vector_format
    )
    type_name = data.get("type_name") or "instantaneous"

    try:
        source = _source_path(_source_idx(data.get))
        if stereo and not interpolator:
            # Settings-sidecar knob (load_settings layers its default under the stored
            # value); a missing sidecar raises rather than assuming a kernel. The
            # sidecar value gets the same membership check as a request value — a
            # typo'd rig.interpolator must fail here, not inside the worker. Mono apply
            # never resamples, so it does not read the knob.
            try:
                interpolator = cs.load_settings(source)["rig"]["interpolator"]
            except FileNotFoundError as exc:
                return jsonify({"success": False, "error": str(exc)}), 200
            if interpolator not in ("linear", "cubic", "lanczos"):
                raise ValueError(
                    f"rig.interpolator in the settings sidecar must be "
                    f"linear|cubic|lanczos, got {interpolator!r}"
                )
        units = _apply_units(data, full_cfg, source, board, stereo, type_name)
    except _ApplyRequestError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except (FileNotFoundError, ValueError, IndexError) as exc:
        # The planner's messages already name the camera / model / dataset; only a
        # missing file gets the "generate it first" hint, and only once.
        msg = str(exc)
        if isinstance(exc, FileNotFoundError) and "generate it first" not in msg:
            msg = f"{msg} — generate it first"
        return jsonify({"success": False, "error": msg}), 200
    except KeyError as exc:
        return (
            jsonify({"success": False, "error": f"missing directory field {exc}"}),
            200,
        )

    # Stereo: default the laser-sheet (z_world, tilt) from the saved self-calibration
    # block unless the request explicitly overrides it. So once self-cal is stored, 3C
    # reconstruction sits on the true sheet automatically.
    if stereo and units:
        # StereoRecord.sc_* mirror the CLI apply's fallback — one implementation
        # of "self-cal leg, 0.0 when absent" on the record, not two dict reads.
        rec0 = units[0]["record"]
        if "z_world" not in data:
            z = rec0.sc_z_offset
        if "tilt_x" not in data:
            tx = rec0.sc_tilt_x
        if "tilt_y" not in data:
            ty = rec0.sc_tilt_y

    if not units:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "no datasets selected — configure base paths first",
                }
            ),
            200,
        )

    # dt is resolved PER UNIT: request override > model-stamped (every generated
    # record carries it in board_meta). Resolving per unit (not once from units[0])
    # means a multi-camera rig whose cameras carry different stamped dt calibrates
    # each camera with ITS OWN dt instead of camera 1's. Velocity has no safe
    # default and no config source, so an unresolved dt fails loudly here, before
    # the job thread starts.
    explicit_dt = data.get("dt")
    try:
        for u in units:
            model_dt = (getattr(u["record"], "board_meta", None) or {}).get("dt")
            u["dt"] = c2runio.resolve_dt(explicit_dt, model_dt)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 200
    # resolve_dt prefers the request, so a request dt is used for every unit.
    dt_source = "request" if explicit_dt is not None else "model"

    # Built here (not in the worker) so it captures the source/board this request
    # resolved, independent of any config change made while the job runs.
    applied_pointer = c2runio.build_applied_pointer(source, board, stereo)

    n_units = len(units)
    out_dirs = [str(u["out"]) for u in units]
    # Which cameras this run covers (mono applies every configured camera), so the GUI
    # can label it "Apply to all N cameras".
    cameras = sorted({int(c) for u in units for c in u["cameras"]})
    with _apply_lock:
        clash = _overlapping_apply_job(out_dirs)
        if clash is not None:
            running_id, running_dir = clash
            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            f"apply job {running_id} is already writing to "
                            f"{running_dir} -- wait for it to finish or cancel it"
                        ),
                    }
                ),
                200,
            )
        job_id = job_manager.create_job(
            "calibration_apply",
            stereo=stereo,
            out_dir=out_dirs[0],
            out_dirs=out_dirs,
            # Per unit, aligned with out_dirs: the dt each unit is calibrated with
            # and where it came from ("request" or "model"). Lists rather than a
            # label-keyed dict because unit labels are not guaranteed unique.
            dt_used=[float(u["dt"]) for u in units],
            dt_source=[dt_source] * n_units,
            cameras=cameras,
            n_cameras=len(cameras),
            # Non-fatal problems (a provenance stamp that failed after the vectors were
            # written), one line each, for the GUI to show beside the result.
            warnings=[],
        )
        cancel_event = threading.Event()
        _apply_cancel_events[job_id] = cancel_event

    def _run():
        total_written = 0
        total_skipped = 0
        warnings: List[str] = []
        try:
            job_manager.update_job(
                job_id, status="running", progress=0, processed=0, total=n_units
            )
            for i, u in enumerate(units):
                # Between units: a cancel stops before the next dataset starts, leaving
                # the previous unit's frame counts in the status.
                if cancel_event.is_set():
                    job_manager.cancel_job(
                        job_id,
                        _apply_cancelled_message(total_written, out_dirs),
                        n_frames=total_written,
                    )
                    return
                # Which dataset is running, published before the first frame ticks so the
                # label is never a unit behind. Frame counts reset to 0 ("not yet reported")
                # rather than carrying the previous unit's totals into this one.
                label = u["label"]
                job_manager.update_job(
                    job_id, unit_label=label, frame_done=0, frame_total=0
                )

                def cb(done, total, _i=i, _label=label):
                    pct = int(((_i + done / max(total, 1)) / n_units) * 100)
                    job_manager.update_job(
                        job_id,
                        progress=pct,
                        processed=_i,
                        total=n_units,
                        frame_done=done,
                        frame_total=total,
                        unit_label=_label,
                    )

                n_skipped = 0
                if u["stereo"]:
                    written, n_skipped = c2runio.reconstruct_stereo_run(
                        u["record"],
                        u["uncal1"],
                        u["uncal2"],
                        u["out"],
                        u["dt"],
                        None,
                        z,
                        tx,
                        ty,
                        progress_cb=cb,
                        vector_glob=vector_glob,
                        interpolator=interpolator,
                        cancel_event=cancel_event,
                    )
                else:
                    written = c2runio.calibrate_mono_run(
                        u["record"],
                        u["uncal"],
                        u["out"],
                        u["dt"],
                        z,
                        tx,
                        ty,
                        progress_cb=cb,
                        vector_glob=vector_glob,
                        cancel_event=cancel_event,
                    )
                if not written:
                    # The runs raise on an empty input dir, but a stereo run whose every
                    # frame lacked the pass writes nothing: never report that as success.
                    raise RuntimeError(
                        f"{label}: no frames were written — nothing was calibrated"
                    )
                total_written += len(written)
                total_skipped += n_skipped
                if n_skipped > 0:
                    # The run reconstructs the final populated pass of coordinates.mat; a
                    # frame pair missing it in either camera is not written. Reported, not silent.
                    warnings.append(
                        f"{label}: {n_skipped} frame(s) skipped — the reconstructed pass "
                        "is absent in one or both cameras"
                    )
                    job_manager.update_job(job_id, warnings=list(warnings))
                # Record what actually calibrated these vectors in the run's archived
                # config. The snapshot was written when PIV started, so its calibration
                # block still names whatever dataset was configured before this one.
                stamp_warning = c2runio.stamp_unit(u, applied_pointer)
                if stamp_warning is not None:
                    warnings.append(stamp_warning)
                    job_manager.update_job(job_id, warnings=list(warnings))
                job_manager.update_job(job_id, processed=i + 1, total=n_units)
            job_manager.complete_job(
                job_id,
                n_frames=total_written,
                n_frames_skipped=total_skipped,
                out_dir="; ".join(out_dirs),
            )
        except c2runio.ApplyCancelled as exc:
            # Mid-unit cancel: the partial unit is not stamped (its run config would
            # claim a calibration it only half received). Its exact frame counts come
            # from the exception, which includes frames that were in flight.
            frames_written = total_written + len(exc.written)
            job_manager.cancel_job(
                job_id,
                _apply_cancelled_message(frames_written, out_dirs),
                n_frames=frames_written,
                frame_done=exc.done,
                frame_total=exc.total,
            )
        except c2runio.ApplyFailed as exc:
            # A frame raised: the queued frames were cancelled; what was written stays.
            logger.exception("apply job %s failed", job_id)
            job_manager.fail_job(job_id, f"{exc} (output dirs: {'; '.join(out_dirs)})")
            job_manager.update_job(
                job_id,
                n_frames=total_written + len(exc.written),
                frame_done=exc.done,
                frame_total=exc.total,
            )
        except (
            Exception
        ) as exc:  # surface in the job status (full traceback to the server log)
            logger.exception("apply job %s failed", job_id)
            job_manager.fail_job(job_id, str(exc))
        finally:
            with _apply_lock:
                _apply_cancel_events.pop(job_id, None)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify(
        {
            "success": True,
            "job_id": job_id,
            "cameras": cameras,
            "n_cameras": len(cameras),
        }
    )


@calibration_bp.route("/calibration/apply/status/<job_id>", methods=["GET"])
def apply_status(job_id):
    """Poll an apply job's status (same shape as the v1 vector-job status)."""
    return _job_status_response(job_id)


@calibration_bp.route("/calibration/apply/cancel/<job_id>", methods=["POST"])
def apply_cancel(job_id):
    """Request a cooperative cancel of a running apply job.

    The worker stops between units, and within a unit as frames complete; frames
    not yet started are dropped, frames already written stay on disk. The job then
    reports ``status: cancelled`` with its frame counts. 404 for an unknown job id;
    ``success: false`` (HTTP 200) when the job exists but has already finished.
    """
    with _apply_lock:
        cancel_event = _apply_cancel_events.get(job_id)
    if cancel_event is None:
        job = job_manager.get_job(job_id)
        if job is None or job.get("job_type") != "calibration_apply":
            return jsonify({"success": False, "error": "job not found"}), 404
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"apply job {job_id} is not running (status: {job['status']})",
                }
            ),
            200,
        )
    cancel_event.set()
    return jsonify({"success": True, "job_id": job_id, "status": "cancelling"})


# ---------------------------------------------------------------------------
# Joint multi-camera calibration (continuous global grid + shared released board)
#
# The DaVis-matching solve: one shared board every camera observes, tied by a global dot
# index. ChArUco gets the grid free from corner ids (zero clicks); dotboard needs a datum
# world frame + cross-camera link clicks (the GlobalGridSpec). resolve_grid drives the live
# overlay; generate runs the solve as a job; model loads the saved record. All three share the
# run_joint_from_spec driver with the CLI (calibration_cli.detect_joint_command), so the GUI
# and headless paths cannot drift.
# ---------------------------------------------------------------------------


def _joint_int_list(value) -> List[int]:
    """Coerce a cameras value (list, or "1,2,3" string) to a list of ints."""
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [int(x) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def _joint_setup(get: Callable[[str], Any]):
    """Resolve the shared joint inputs from a request.

    Returns (cfg, gg, board, source, source_idx, cameras, n_views, image_format, image_type,
    params, spacing, datum_camera, datum_view). Raises ValueError for the caller to surface.
    """
    cfg = _cfg()
    board = _request_board(get)[0]
    if board not in ("dotboard", "charuco"):
        raise ValueError(f"joint: board must be dotboard|charuco, got {board!r}")
    source_idx = _source_idx(get)
    source = _source_path(source_idx)
    settings = _settings_idx(source_idx)
    image_settings = settings.get("image") or {}
    # The clicked coords (datum + anchors + camera_extends + cameras) live in the sidecar
    # inputs.mat, not config. Same dict shape as the old global_grid block, so the gg.get(...)
    # reads below and _joint_spec / _joint_origin_mm are unchanged. A request override (the live
    # wizard) still wins in _joint_spec.
    side = try_load_inputs(rec.joint_model_dir_for_source(source, board))
    gg = dict(side.coords) if (side and side.coords) else {}
    cameras = (
        _joint_int_list(get("cameras"))
        or _joint_int_list(gg.get("cameras"))
        or _joint_int_list(get_config().camera_numbers)
    )
    if not cameras:
        raise ValueError("joint: no cameras — set the rig cameras in the joint wizard")

    def _int_or(name, fallback):
        v = get(name)
        return int(v) if v not in (None, "") else int(fallback)

    datum_camera = _int_or("datum_camera", gg.get("datum_camera", cameras[0]))
    datum_view = _int_or("datum_view", gg.get("datum_view", 0))
    # Same contract as the CLI (_n_views_cli): request > sidecar image.n_views > loud error.
    # No literal fallback -- a silent 10-view solve is a different calibration.
    n_views_raw = next(
        (
            v
            for v in (get("n_views"), get("frame_total"), image_settings.get("n_views"))
            if v not in (None, "")
        ),
        None,
    )
    if n_views_raw is None:
        raise ValueError(
            "joint: n_views is required -- send it in the request or set image.n_views in "
            "the calibration settings sidecar"
        )
    n_views = int(n_views_raw)
    if n_views < 1:
        raise ValueError("joint: n_views must be >= 1")
    image_format = get("image_format") or image_settings.get("image_format")
    image_type = get("image_type") or image_settings.get("image_type")
    # Request board_params override the sidecar (same contract as _resolve_board): the GUI's
    # live input wins over a debounce-lagged settings save, so a Detect click right after
    # typing the geometry doesn't read a stale sidecar.
    params = c2._board_params(settings.get("methods") or {}, board, get("board_params"))
    spacing = c2._spacing_mm(board, params)
    return (
        cfg,
        gg,
        board,
        source,
        source_idx,
        cameras,
        n_views,
        image_format,
        image_type,
        params,
        spacing,
        datum_camera,
        datum_view,
    )


def _joint_detect(
    board,
    params,
    cameras,
    n_views,
    source_idx,
    image_format,
    image_type,
    spacing,
    *,
    refresh=False,
    cached_only=False,
    progress_cb=None,
):
    """Detect every view of every camera, cached by (source, board, n_views, format, params,
    0/1 indexing).

    Returns (detections_by_cam, image_size_by_cam, persist_skipped). ``persist_skipped`` is
    None when the detections are in the sidecar (served from it, or written to it) and the
    reason when a read-only source (OSError) kept them in memory only; any other write
    failure raises. A view that fails detection is kept in place
    as an unsuccessful DetectionResult, NOT raised on: a single bad frame must not blank the whole
    overlay or abort the solve. The resolvers skip failed views (reporting them per-view) and the
    solve simply uses the views that detected. The cache makes the live resolve loop and the
    generate job cheap; pass refresh=True after the images change on disk.

    ``cached_only=True`` (the GUI's automatic viewer probes) reads the caches and returns
    ``(None, None, None)`` on a miss instead of detecting — only an explicit button push may start
    a real detection. Never combined with ``refresh``.
    """
    # The camera set is part of the key: a hit only detected the cameras it was asked for, so
    # reusing it for a different set would hand the solver a stale subset (caught by run_joint's
    # expected_cameras guard, but with a confusing message blaming the grid, not the cache).
    # The source enters as its resolved PATH, not its index: an in-place edit of the path at
    # the same index is a different rig and must not be served the old directory's detections.
    zero_based = _zero_based(source_idx)
    key = (
        str(_source_path(source_idx)),
        str(board),
        int(n_views),
        str(image_format),
        str(image_type),
        tuple(sorted(int(c) for c in cameras)),
        repr(params),
        zero_based,
    )
    det_key = joint_det_key(
        board,
        n_views,
        image_format,
        infer_image_type(image_format),
        cameras,
        params,
        zero_based=zero_based,
    )
    if not refresh:
        with _joint_detect_lock:
            hit = _joint_detect_cache.get(key)
        if hit is not None:
            return hit["detections"], hit["image_size_by_cam"], hit["persist_skipped"]
        # Fall back to the persistent sidecar (survives restarts + a model delete); promote it
        # into memory on a hit. det_key guards against serving detections from a different
        # n_views / format / board param.
        idir = _joint_inputs_dir(source_idx, board)
        rec_in = try_load_inputs(idir) if idir is not None else None
        # Reuse only when the sidecar loaded cleanly AND its detections were made with the same
        # params (det_key); absent / corrupt / stale-key all fall through to a fresh detect.
        if rec_in is not None and rec_in.detections and rec_in.det_key == det_key:
            payload = {
                "detections": rec_in.detections,
                "image_size_by_cam": rec_in.image_size_by_cam,
                "persist_skipped": None,
            }
            with _joint_detect_lock:
                _joint_detect_cache[key] = payload
            return payload["detections"], payload["image_size_by_cam"], None

    if cached_only:
        return None, None, None

    total = len(cameras) * int(n_views)
    done = 0
    detections: dict = {}
    image_size_by_cam: dict = {}
    blank_hints: List[str] = []  # ChArUco failure hints for cameras with no detection
    for cam in cameras:
        imgs = _load_views(cam, n_views, source_idx, image_format, image_type)
        if not imgs:
            raise ValueError(f"joint: no images loaded for cam{cam}")

        def _on_done():
            nonlocal done
            done += 1
            if progress_cb is not None:
                progress_cb(done, total)

        detections[cam] = _detect_parallel(
            board, params, imgs, spacing_mm=spacing, on_done=_on_done
        )
        h, w = imgs[0].shape[:2]
        image_size_by_cam[cam] = (int(w), int(h))
        if not any(d.success for d in detections[cam]):
            # One diagnosis per blank camera (its first view), not per view.
            _err, diag = _board_not_detected(
                c2._build_detector(board, params), imgs[0], detections[cam][0]
            )
            if "hint" in diag:
                blank_hints.append(f"cam{cam} view 1: {diag['hint']}")

    # A camera with NO successful view cannot be calibrated at all — that is the one fatal case
    # (almost always a wrong path/format, not a single bad target). Individual failed views are
    # fine: they are dropped downstream and reported per-view.
    blank = [c for c in cameras if not any(d.success for d in detections[c])]
    if blank:
        raise ValueError(
            f"joint: camera(s) {blank} detected no calibration target in any image — check the "
            f"image path, format and board parameters"
            + "".join(f"; {hint}" for hint in blank_hints)
        )

    # Persist into the sidecar for the next session. Merges, so an in-progress coords/click
    # commit is preserved. Same rule as detect_views / the stepped detect: a read-only source
    # (OSError) keeps the in-memory cache and says why; anything else (a corrupt sidecar, a
    # detection that cannot be serialised) raises.
    persist_skipped: Optional[str] = None
    idir = _joint_inputs_dir(source_idx, board)
    if idir is None:
        persist_skipped = (
            "sidecar not written: the calibration source is not configured"
        )
    else:
        try:
            save_inputs(
                idir,
                path_type="joint",
                board_type=board,
                detections=detections,
                image_size_by_cam=image_size_by_cam,
                det_key=det_key,
                board_params=rec.geometry_meta(board, params),
            )
        except OSError as exc:
            logger.warning("joint detect: sidecar not written: %s", exc)
            persist_skipped = f"sidecar not written: {exc}"
    payload = {
        "detections": detections,
        "image_size_by_cam": image_size_by_cam,
        "persist_skipped": persist_skipped,
    }
    with _joint_detect_lock:
        _joint_detect_cache[key] = payload
    return detections, image_size_by_cam, persist_skipped


def _joint_spec(get, gg, board):
    """GlobalGridSpec from the request's global_grid block (preferred) or config; None for
    ChArUco. Raises ValueError on an incomplete dotboard spec (caller decides how to surface).
    """
    if board == "charuco":
        return None
    block = get("global_grid") or gg
    try:
        return c2._global_grid_spec_from_cfg(block)
    except (
        SystemExit
    ) as exc:  # the CLI builder signals validation errors via SystemExit
        raise ValueError(str(exc))


def _joint_origin_mm(board, gg) -> tuple:
    if board == "dotboard":
        om = (gg.get("datum_clicks", {}) or {}).get("origin_mm", [0.0, 0.0])
        return (float(om[0]), float(om[1]))
    return (0.0, 0.0)


@calibration_bp.route("/calibration/joint/inputs", methods=["GET"])
def joint_inputs():
    """The saved joint clicked-coords block (datum + anchors + camera_extends), or null.

    The Set-Global-Coordinates wizard restores its state from this — it replaces reading the
    old ``config.calibration.global_grid`` block.
    """
    get = request.args.get
    board = _request_board(get)[0]
    try:
        source = _source_path(_source_idx(get))
    except (ValueError, IndexError):
        return jsonify({"coords": None}), 200
    side = try_load_inputs(rec.joint_model_dir_for_source(source, board))
    return jsonify({"coords": (side.coords if side else None)}), 200


@calibration_bp.route("/calibration/joint/inputs/save", methods=["POST"])
def joint_inputs_save():
    """Persist the joint clicked-coords block into the sidecar ``inputs.mat``.

    Merges, so the detections already stored by ``_joint_detect`` are untouched. Replaces the
    old ``config.calibration.global_grid`` write.
    """
    data = request.get_json() or {}
    get = data.get
    board = _request_board(get)[0]
    coords = get("global_grid")
    if coords is None:
        return jsonify({"success": False, "error": "missing 'global_grid' block"}), 200
    try:
        source = _source_path(_source_idx(get))
        save_inputs(
            rec.joint_model_dir_for_source(source, board),
            path_type="joint",
            board_type=board,
            coords=coords,
        )
    except (ValueError, IndexError, OSError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 200
    return jsonify({"success": True}), 200


@calibration_bp.route("/calibration/joint/resolve_grid", methods=["POST"])
def joint_resolve_grid():
    """Detect all views + resolve the global grid for the live overlay.

    One entry per (camera, view) with its detected dot pixels and, when resolvable, the global
    (gx, gy) index of each dot. A dotboard spec still incomplete (no datum / missing links) is
    NOT an error here — those views return points but no indices plus a reason, so the GUI shows
    the dots and guides the next click.
    """
    data = request.get_json() or {}
    get = data.get
    try:
        (
            cfg,
            gg,
            board,
            source,
            source_idx,
            cameras,
            n_views,
            image_format,
            image_type,
            params,
            spacing,
            datum_camera,
            datum_view,
        ) = _joint_setup(get)
        detections, _, persist_skipped = _joint_detect(
            board,
            params,
            cameras,
            n_views,
            source_idx,
            image_format,
            image_type,
            spacing,
            refresh=bool(get("refresh")),
            cached_only=bool(get("cached_only")),
        )
    except (
        FileNotFoundError,
        ValueError,
        TypeError,
    ) as exc:  # FileNotFoundError: a missing view image. TypeError: a malformed payload.
        return jsonify({"success": False, "error": str(exc)}), 200

    if detections is None:
        # cached_only probe missed both caches: nothing was ever detected (or the cached
        # detections were made with different params). Not an error — the GUI stays quiet
        # and waits for an explicit Detect push.
        return jsonify({"success": False, "reason": "no_cached_detections"}), 200

    resolved = {}
    unresolved = []
    errors = []
    try:
        spec = _joint_spec(get, gg, board)
        # Partial (non-raising) resolve: a half-built dotboard spec resolves the datum + linked
        # views and reports a per-view reason for the rest, so the overlay fills in as the user
        # clicks instead of blanking on the first unresolved view. ChArUco resolves every view.
        resolved, unresolved = resolve_global_grid_partial(
            detections, spec, spacing_mm=spacing
        )
    except ValueError as exc:
        errors.append(
            str(exc)
        )  # whole-spec construction failure — overlay still shows the dots

    reason_by_view = {(int(c), int(v)): r for c, v, r in unresolved}

    views = []
    for cam in cameras:
        for v, d in enumerate(detections[cam]):
            gi = resolved.get((cam, v))
            # A view that failed detection carries no dots — say so explicitly so the user can skip
            # that frame rather than wonder why the overlay is empty there.
            reason = reason_by_view.get((cam, v))
            if not d.success and reason is None:
                reason = "detection failed — skip this frame or replace the image"
            # The detector's LOCAL (col, row) grid indices let the overlay draw the dot mesh
            # immediately on Preview — before any anchoring assigns a global index. Per-view
            # connectivity is identical whichever indexing is used.
            local_idx = (
                None
                if (not d.success or d.grid_indices is None)
                else np.asarray(d.grid_indices, dtype=int).tolist()
            )
            # When a thin-overlap bridge leaves a mirror ambiguity the dots cannot break, offer the
            # competing layouts so the GUI can ask the user which footprint is real (confirm-on-
            # overlay). Returns [] for any view that resolves cleanly or carries no such bridge, so
            # only the genuinely ambiguous view sprouts a picker.
            candidates = (
                (
                    []
                    if gi is not None
                    else first_view_orientation_candidates(
                        detections, spec, cam, v, spacing_mm=spacing
                    )
                )
                if not errors
                else []
            )
            views.append(
                {
                    "camera": cam,
                    "view": v,
                    "n": int(d.n),
                    "points": np.asarray(d.image_points, dtype=float).tolist(),
                    "grid_indices": local_idx,
                    "global_index": (
                        None if gi is None else np.asarray(gi, dtype=int).tolist()
                    ),
                    "resolved": gi is not None,
                    "reason": reason,
                    "candidates": candidates,
                }
            )
    return jsonify(
        {
            "success": True,
            "board": board,
            "cameras": cameras,
            "n_views": n_views,
            "spacing_mm": float(spacing),
            "datum_camera": datum_camera,
            "datum_view": datum_view,
            "views": views,
            "n_resolved": sum(1 for vw in views if vw["resolved"]),
            "n_views_total": len(views),
            "errors": errors,
            "persist_skipped": persist_skipped,
        }
    )


def _pose_diversity_payload(pd: Optional[PoseDiversity]) -> Optional[dict]:
    """JSON-safe view of the pose-diversity diagnostic, or ``None`` when absent.

    ``None`` means the record predates the diagnostic (or the solve is polynomial, which has
    no poses) -- the frontend shows nothing rather than a fabricated all-clear. NaN metrics
    (azimuth/anisotropy below two tilted views) go through ``_finite_or_none``.
    """
    if pd is None:
        return None
    return {
        "degenerate": bool(pd.degenerate),
        "cameras": [int(c) for c in pd.cameras],
        "flag_low_tilt": {str(c): bool(pd.flag_low_tilt[c]) for c in pd.cameras},
        "flag_single_azimuth": {
            str(c): bool(pd.flag_single_azimuth[c]) for c in pd.cameras
        },
        "flag_constant_standoff": {
            str(c): bool(pd.flag_constant_standoff[c]) for c in pd.cameras
        },
        "tilt_deg_median": {
            str(c): _finite_or_none(pd.tilt_deg_median[c]) for c in pd.cameras
        },
        "tilt_anisotropy": {
            str(c): _finite_or_none(pd.tilt_anisotropy[c]) for c in pd.cameras
        },
        "standoff_range_fraction": {
            str(c): _finite_or_none(pd.standoff_range_fraction[c]) for c in pd.cameras
        },
        "flagged_views": [[int(c), int(v)] for c, v in pd.flagged_views],
        "view_rms_px": {
            f"{c}-{v}": _finite_or_none(r) for (c, v), r in pd.view_rms_px.items()
        },
        "board_planarity_rms_mm": _finite_or_none(pd.board_planarity_rms_mm),
        "report": format_pose_diversity(pd),
    }


def _finite_or_none(x) -> Optional[float]:
    """Float for JSON, or None when non-finite.

    Flask's default ``jsonify`` emits a bare ``NaN``/``Infinity`` token — invalid JSON that
    ``JSON.parse`` rejects, which silently stalls the frontend job poll. The polynomial joint
    solve legitimately has no reprojection-px RMS or cross-camera agreement (both NaN), so those
    fields must serialise as ``null``, not ``NaN``.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


@calibration_bp.route("/calibration/joint/generate", methods=["POST"])
def joint_generate():
    """Run the joint solve (pinhole) or per-camera polynomial as a background job."""
    data = request.get_json() or {}
    get = data.get
    try:
        (
            cfg,
            gg,
            board,
            source,
            source_idx,
            cameras,
            n_views,
            image_format,
            image_type,
            params,
            spacing,
            datum_camera,
            datum_view,
        ) = _joint_setup(get)
        model_type = get("model_type") or _methods(get).get(board, {}).get(
            "model_type", "pinhole"
        )
        if model_type not in ("pinhole", "polynomial"):
            raise ValueError(
                f"joint: model_type must be pinhole|polynomial, got {model_type!r}"
            )
        board_release = get("board_release") or gg.get("board_release", "full3d")
        spec = _joint_spec(get, gg, board)
        origin_mm = _joint_origin_mm(board, gg)
        gen_dt = _generate_dt(get, source)
        if board == "charuco":
            # The dotboard wizard persists its datum through the click flow; ChArUco has no
            # click flow, so persist the datum this request solves with. The CLI reads ONLY
            # the sidecar, and without this it re-anchored on view 0 -- a different
            # calibration from the same inputs (review 2026-08-26).
            idir = _joint_inputs_dir(source_idx, board)
            if idir is not None:
                save_inputs(
                    idir,
                    path_type="joint",
                    board_type=board,
                    coords={
                        **gg,
                        "cameras": [int(c) for c in cameras],
                        "datum_camera": int(datum_camera),
                        "datum_view": int(datum_view),
                        "board_release": board_release,
                    },
                )
    except (
        FileNotFoundError,
        ValueError,
        TypeError,
    ) as exc:  # FileNotFoundError: a missing view image. TypeError: a malformed payload.
        return jsonify({"success": False, "error": str(exc)}), 200

    job_id = job_manager.create_job(
        "calibration_joint", board=board, model_type=model_type
    )

    def _run():
        try:
            job_manager.update_job(job_id, status="running", progress=0)

            def detect_cb(done, total):
                job_manager.update_job(
                    job_id,
                    progress=int(done / max(total, 1) * 80),
                    processed=done,
                    total=total,
                )

            detections, image_size_by_cam, persist_skipped = _joint_detect(
                board,
                params,
                cameras,
                n_views,
                source_idx,
                image_format,
                image_type,
                spacing,
                progress_cb=detect_cb,
            )
            job_manager.update_job(job_id, progress=85)

            # Proof figures land beside the joint record. The driver needs raw images for the
            # detection overlays + dewarp; re-load them lazily one camera at a time (the detect
            # step above discarded them) so we never hold the whole rig in memory at once.
            figure_dir = (
                rec.joint_model_dir_for_source(source, board).parent / "figures"
            )
            _img_cache: dict = {}

            def image_loader(cam, view):
                if _img_cache.get("cam") != cam:
                    _img_cache["cam"] = cam
                    try:
                        _img_cache["imgs"] = _load_views(
                            cam, n_views, source_idx, image_format, image_type
                        )
                    except Exception:
                        # Visible, not silent (CLAUDE.md): the calibration is unaffected, but the
                        # image-based figures (detection overlays, dewarp) are skipped for this
                        # camera — say so once rather than returning blank frames quietly.
                        logger.warning(
                            "joint figures: could not load images for cam%s — its "
                            "detection/dewarp figures are skipped",
                            cam,
                        )
                        _img_cache["imgs"] = []
                imgs = _img_cache.get("imgs") or []
                return imgs[view] if 0 <= view < len(imgs) else None

            res = run_joint_from_spec(
                detections,
                image_size_by_cam,
                source=source,
                board=board,
                model_type=model_type,
                spacing_mm=spacing,
                dt=gen_dt,
                datum_camera=datum_camera,
                datum_view=datum_view,
                board_release=board_release,
                origin_mm=origin_mm,
                spec=spec,
                cameras=cameras,
                distortion_model=_MODEL,
                fix_aspect_ratio=True,
                n_views=n_views,
                figure_dir=figure_dir,
                image_loader=image_loader,
                board_params=params,
            )
            job_manager.complete_job(
                job_id,
                out_dir="; ".join(str(p) for p in res.paths),
                model_type=res.model_type,
                cameras=[int(c) for c in res.cameras],
                per_camera_rms={
                    str(c): _finite_or_none(r) for c, r in res.per_camera_rms.items()
                },
                rms_units=res.rms_units,
                rms_px=_finite_or_none(res.rms_px),
                converged=bool(res.converged),
                cross_camera_board_agreement_mm=_finite_or_none(
                    res.cross_camera_board_agreement_mm
                ),
                n_board_dots=int(res.n_board_dots),
                paths=[str(p) for p in res.paths],
                pose_diversity=_pose_diversity_payload(res.pose_diversity),
                persist_skipped=persist_skipped,
                excluded_views=[[int(c), int(f)] for c, f in res.excluded_views],
            )
        except (
            Exception
        ) as exc:  # surface in the job status (full traceback to the server log)
            logger.exception("joint generate job %s failed", job_id)
            job_manager.fail_job(job_id, str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"success": True, "job_id": job_id})


@calibration_bp.route("/calibration/joint/generate/status/<job_id>", methods=["GET"])
def joint_generate_status(job_id):
    """Poll a joint-generate job's status (same shape as the apply job)."""
    return _job_status_response(job_id)


@calibration_bp.route("/calibration/joint/model", methods=["GET"])
def joint_model():
    """Load a saved joint model for display: the unified pinhole JointRecord, or per-camera
    polynomial records (which have no shared board to unify)."""
    get = request.args.get
    board = _request_board(get)[0]
    source = _source_path(_source_idx(get))
    model_type = _model_type_arg(get, board)
    side = try_load_inputs(rec.joint_model_dir_for_source(source, board))
    gg = dict(side.coords) if (side and side.coords) else {}

    if model_type == "polynomial":
        cameras = (
            _joint_int_list(get("cameras"))
            or _joint_int_list(gg.get("cameras"))
            or _joint_int_list(get_config().camera_numbers)
        )
        out = {}
        geom = None
        excluded: list = []
        for cam in cameras:
            mdir = rec.mono_model_dir_for_source(source, cam, board)
            try:
                mono = rec.load_mono(mdir, "polynomial")
            except (FileNotFoundError, ValueError):
                continue
            try:
                excluded += rec.parse_excluded_views(
                    _require_meta(mono.board_meta, "excluded_views", str(mdir))
                )
            except ValueError as exc:
                return jsonify({"exists": False, "error": str(exc)}), 200
            if geom is None:
                geom = _geometry_payload(mono.board_meta)
            # Full coefficient summary (coeffs_x/y, normalisation, mm RMS) so the results card can
            # show the actual fitted polynomial, mirroring the pinhole per-camera parameters.
            out[str(cam)] = _polynomial_summary(mono.camera_model)
        if not out:
            return jsonify({"exists": False})
        return jsonify(
            {
                "exists": True,
                "model_type": "polynomial",
                "board": board,
                "cameras": [int(c) for c in out],
                "per_camera": out,
                "geometry": geom,
                "excluded_views": [[c, f] for c, f in sorted(excluded)],
            }
        )

    try:
        jp = rec.resolve_joint_path(
            rec.joint_model_dir_for_source(source, board), "pinhole"
        )
        jr = rec.load_joint(jp)
    except (FileNotFoundError, ValueError):
        return jsonify({"exists": False})
    meta = jr.board_meta or {}
    try:
        excluded = rec.parse_excluded_views(
            _require_meta(meta, "excluded_views", str(jp))
        )
    except ValueError as exc:
        return jsonify({"exists": False, "error": str(exc)}), 200

    # Per-camera intrinsics + extrinsics for the results panel. Position is the camera centre in
    # the world/board frame (C = -R^T t); rotation is the world->camera orientation as Tait-Bryan
    # euler angles in degrees (cv2.RQDecomp3x3, the same convention DaVis reports). Focal length is
    # px only — mm would need the sensor pixel size, which the model does not store.
    per_camera = {}
    positions = {}
    for c in jr.cameras:
        m = jr.models[c]
        K = np.asarray(m.K, dtype=np.float64)
        R = np.asarray(m.R, dtype=np.float64)
        t = np.asarray(m.t, dtype=np.float64).reshape(3)
        pos = (-R.T @ t).ravel()
        positions[int(c)] = pos
        euler = cv2.RQDecomp3x3(R)[0]
        per_camera[str(c)] = {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "dist": [float(x) for x in np.asarray(m.dist, dtype=np.float64).ravel()],
            "image_size": [int(m.image_size[0]), int(m.image_size[1])],
            "rms_px": float(jr.per_camera_rms.get(c, float("nan"))),
            "position_mm": [float(x) for x in pos],
            "rotation_deg": [
                float(x) for x in np.asarray(euler, dtype=np.float64).ravel()
            ],
        }

    cams_sorted = sorted(int(c) for c in jr.cameras)
    baselines_mm = {
        f"{a}-{b}": float(np.linalg.norm(positions[a] - positions[b]))
        for i, a in enumerate(cams_sorted)
        for b in cams_sorted[i + 1 :]
    }

    return jsonify(
        {
            "exists": True,
            "model_type": "pinhole",
            "board": jr.board_type,
            "cameras": [int(c) for c in jr.cameras],
            "per_camera_rms": {str(c): float(r) for c, r in jr.per_camera_rms.items()},
            "rms_px": float(jr.rms_px),
            "spacing_mm": float(jr.spacing_mm),
            "board_release": jr.board_release,
            # Required keys: a joint record without them is stale and must be re-solved.
            # No defaults -- a fabricated 0.0 agreement or converged=False reads as a
            # verdict the solve never gave (review 2026-08-26).
            "converged": bool(meta["converged"]),
            "cross_camera_board_agreement_mm": float(
                meta["cross_camera_board_agreement_mm"]
            ),
            "n_board_dots": int(meta["n_board_dots"]),
            "image_sizes": {
                str(c): [
                    int(jr.models[c].image_size[0]),
                    int(jr.models[c].image_size[1]),
                ]
                for c in jr.cameras
            },
            "per_camera": per_camera,
            "baselines_mm": baselines_mm,
            "geometry": _geometry_payload(meta),
            # None when the record predates the diagnostic -- never a default all-clear.
            "pose_diversity": _pose_diversity_payload(
                pose_diversity_from_meta(jr.pose_diversity)
                if jr.pose_diversity
                else None
            ),
            "excluded_views": [[c, f] for c, f in excluded],
        }
    )


# ---------------------------------------------------------------------------
# Stereo self-calibration (Wieneke disparity minimisation)
#
# A post-model step: it loads the saved stereo model + recorded PIV particle frames
# from a base_path dataset, recovers the laser sheet (z_offset, tilt_x, tilt_y), and
# writes the result INTO the stereo record (so apply picks it up automatically) plus
# the six diagnostic figures into the calibration source folder.
# ---------------------------------------------------------------------------


def _stereo_locators(get: Callable[[str], Any]):
    """(source, cam1, cam2, board, model_dir, self_cal_figdir) from request locators.

    The board resolves like every other calibration route (request ``board``, else
    the config's ``active`` board) and scopes the stereo model dir.
    """
    source = _source_path(_source_idx(get))
    board = _request_board(get)[0]
    cam1, cam2 = _required_camera_pair(get)
    model_dir = rec.stereo_model_dir_for_source(source, cam1, cam2, board)
    figdir = model_dir.parent / "figures" / "self_cal"
    return source, cam1, cam2, board, model_dir, figdir


def _require_positive_int(value: Any, field: str) -> int:
    """``value`` as a positive int, else ``ValueError`` naming ``field``.

    Self-cal counts (images, window size, iterations) have no meaningful zero:
    ``load_particle_pairs`` divides by ``n_images``, and a missing stored field
    means a stale block, not a count of zero.
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        raise ValueError(f"{field} must be a positive integer (got {value!r})")
    return n


def _self_cal_payload(record) -> dict:
    """The saved self_cal block, shaped for the GUI (radians + degrees).

    For a baked record the applied z_offset/tilt are zero (the correction lives in the
    extrinsics); the recovered sheet is surfaced from the fitted_* provenance fields
    (the TOTAL over all bakes, in the original clicked frame) so the GUI still shows
    what self-cal found. Legacy (pre-bake) records fall back to the
    applied fields.
    """
    import math as _m

    sc = record.self_cal or {}
    if not sc:
        return {"has_self_calibration": False}
    z = float(sc.get("fitted_z_offset", sc.get("z_offset", 0.0)))
    tx = float(sc.get("fitted_tilt_x", sc.get("tilt_x", 0.0)))
    ty = float(sc.get("fitted_tilt_y", sc.get("tilt_y", 0.0)))
    # Non-finite floats (a diverged self-cal) would make jsonify emit bare NaN/Infinity —
    # invalid JSON that the frontend poll silently rejects (a hang). Send null instead.
    return {
        "has_self_calibration": True,
        "baked": bool(int(sc.get("baked", 0))),
        "z_offset": _finite_or_none(z),
        "tilt_x": _finite_or_none(tx),
        "tilt_y": _finite_or_none(ty),
        "tilt_x_deg": _finite_or_none(_m.degrees(tx)),
        "tilt_y_deg": _finite_or_none(_m.degrees(ty)),
        "converged": bool(int(sc.get("converged", 0))),
        "final_rms_disparity": _finite_or_none(sc.get("final_rms_disparity", 0.0)),
        "n_iterations": _require_positive_int(
            sc.get("n_iterations"), "self_cal.n_iterations"
        ),
        "n_images": _require_positive_int(sc.get("n_images"), "self_cal.n_images"),
        "window_size": _require_positive_int(
            sc.get("window_size"), "self_cal.window_size"
        ),
        "overlap": _finite_or_none(sc.get("overlap", 0.0)),
        "source": str(sc.get("source", "auto")),
        # How many bakes the total spans; null when the block predates the counter.
        "n_bakes": int(sc["n_bakes"]) if "n_bakes" in sc else None,
    }


@calibration_bp.route("/calibration/self_cal/result", methods=["GET"])
def self_cal_result():
    """Current self-calibration block + saved figure list (restore on model load)."""
    try:
        _, _, _, board, model_dir, figdir = _stereo_locators(request.args.get)
        record = rec.load_stereo(
            model_dir, model_type=_model_type_arg(request.args.get, board, stereo=True)
        )
    except (FileNotFoundError, ValueError, IndexError):
        return jsonify({"exists": False, "has_self_calibration": False}), 200
    try:
        payload = _self_cal_payload(record)
    except ValueError as exc:
        # A stored block with a missing/zero count is stale: say so, never fake a 0.
        return (
            jsonify(
                {
                    "success": False,
                    "exists": True,
                    "has_self_calibration": True,
                    "error": str(exc),
                }
            ),
            200,
        )
    payload["exists"] = True
    payload["figures"] = _list_figures(figdir)
    return jsonify(payload)


@calibration_bp.route("/calibration/self_cal/run", methods=["POST"])
def self_cal_run():
    """Run iterative self-calibration as a background job; write the result into the record."""
    data = request.get_json() or {}
    full_cfg = get_config()
    try:
        _, req1, req2, board, model_dir, figdir = _stereo_locators(data.get)
        record = rec.load_stereo(
            model_dir, model_type=_model_type_arg(data.get, board, stereo=True)
        )
        # Images load in the record's order: c2sc.run correlates imgs1 against model1.
        cam1, cam2 = rec.record_pair(record, (req1, req2))
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"no stereo model — generate it first ({exc})",
                }
            ),
            200,
        )

    # Self-cal nudges the rigid pose, so it needs pinhole extrinsics. A polynomial pair
    # has no R/t to rebake; reject it here, before the expensive correlation loop, rather
    # than failing late inside rebake_record.
    if not (hasattr(record.model1, "R") and hasattr(record.model1, "t")):
        return (
            jsonify(
                {
                    "success": False,
                    "error": (
                        "self-calibration needs a pinhole stereo model (it corrects the "
                        f"extrinsics); this model is {type(record.model1).__name__} with "
                        "no pose to rebake — regenerate as pinhole, or skip self-cal for "
                        "the polynomial fit"
                    ),
                }
            ),
            200,
        )
    # A stale baked block cannot be composed with a new bake; reject it here, before
    # the expensive correlation loop, rather than failing late inside baked_block.
    try:
        c2sc.prev_cumulative(record.self_cal)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 200

    base_idx = int(data.get("base_path_idx", 0))
    try:
        n_images = _require_positive_int(data.get("n_images", 20), "n_images")
        window_size = _require_positive_int(data.get("window_size", 64), "window_size")
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 200
    overlap = float(data.get("overlap", 50.0))
    apply_filters = bool(data.get("apply_filters", True))

    job_id = job_manager.create_job("calibration_self_cal", out_dir=str(figdir))

    def _run():
        import math as _m

        try:
            job_manager.update_job(
                job_id, status="running", progress=10, stage="loading_images"
            )
            imgs1, imgs2 = c2sc.load_particle_pairs(
                full_cfg, base_idx, cam1, cam2, n_images, apply_filters
            )
            job_manager.update_job(job_id, progress=25, stage="self_calibrating")
            result = c2sc.run(
                record,
                imgs1,
                imgs2,
                window_size=window_size,
                overlap=overlap,
                figure_dir=figdir,
            )
            # The previous block holds the map already baked into the poses; read it
            # before the rebake so this run's increment composes onto it.
            prev_block = record.self_cal
            c2sc.rebake_record(record, result.z_offset, result.tilt_x, result.tilt_y)
            record.self_cal = c2sc.baked_block(
                result,
                prev_block,
                n_images=len(imgs1),
                window_size=window_size,
                overlap=overlap,
            )
            rec.save_stereo(record, model_dir)
            # Guard every float against NaN/Inf (a diverged fit) so the jsonify'd status
            # payload stays valid JSON and the poll never hangs — null surfaces instead.
            history = [
                dict(
                    iteration=h.iteration,
                    rms_disparity=_finite_or_none(h.rms_disparity),
                    delta_z=_finite_or_none(h.delta_z),
                    delta_tilt_x=_finite_or_none(h.delta_tilt_x),
                    delta_tilt_y=_finite_or_none(h.delta_tilt_y),
                    cumulative_z=_finite_or_none(h.cumulative_z),
                    cumulative_tilt_x=_finite_or_none(h.cumulative_tilt_x),
                    cumulative_tilt_y=_finite_or_none(h.cumulative_tilt_y),
                )
                for h in result.history
            ]
            job_manager.complete_job(
                job_id,
                converged=bool(result.converged),
                n_iterations=int(result.n_iterations),
                z_offset=_finite_or_none(result.z_offset),
                tilt_x=_finite_or_none(result.tilt_x),
                tilt_y=_finite_or_none(result.tilt_y),
                tilt_x_deg=_finite_or_none(_m.degrees(result.tilt_x)),
                tilt_y_deg=_finite_or_none(_m.degrees(result.tilt_y)),
                final_rms_disparity=_finite_or_none(result.final_rms_disparity),
                history=history,
                figures=_list_figures(figdir),
            )
        except Exception as exc:
            logger.exception("self-cal job %s failed", job_id)
            job_manager.fail_job(job_id, str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"success": True, "job_id": job_id})


@calibration_bp.route("/calibration/self_cal/status/<job_id>", methods=["GET"])
def self_cal_status(job_id):
    """Poll a self-calibration job (convergence history + figure list on completion)."""
    return _job_status_response(job_id)


@calibration_bp.route("/calibration/self_cal/figure", methods=["GET"])
def self_cal_figure():
    """Serve one self-cal diagnostic PNG from the model's figures/self_cal folder."""
    name = request.args.get("name", "")
    if not name or Path(name).name != name:
        return jsonify({"error": "invalid figure name"}), 400
    try:
        _, _, _, _, _, figdir = _stereo_locators(request.args.get)
    except (ValueError, IndexError):
        return jsonify({"error": "source not configured"}), 404
    figdir_r = Path(figdir).resolve()
    path = (figdir_r / name).resolve()
    if path.parent != figdir_r or not path.is_file():
        return jsonify({"error": "figure not found"}), 404
    return Response(path.read_bytes(), mimetype="image/png")

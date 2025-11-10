import base64
import io
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, jsonify, request, send_file
from loguru import logger
from matplotlib.colors import Normalize
from scipy.io import loadmat

from ...config import get_config
from ...paths import get_data_paths
from .pod_decompose import pod_decompose

POD_bp = Blueprint("POD", __name__)


# In-memory POD job state
_pod_state: Dict[str, Any] = {
    "processing": False,
    "progress": 0,  # 0..100 (coarse: 0 at start, 100 at end)
    "message": None,
    "started_at": None,
    "finished_at": None,
    "error": None,
}
_pod_thread: Optional[threading.Thread] = None
_pod_cancel_event = threading.Event()
_state_lock = threading.Lock()


def _set_state(**kwargs):
    with _state_lock:
        _pod_state.update(kwargs)


def _reset_state():
    _set_state(
        processing=False,
        progress=0,
        message=None,
        started_at=None,
        finished_at=None,
        error=None,
    )


def _run_pod_job(base: Path, cam: int):
    """Background job entry to run POD and update state."""
    cfg = get_config(refresh=True)
    try:
        _set_state(
            processing=True,
            progress=0,
            started_at=datetime.utcnow().isoformat(),
            message="POD running",
            error=None,
        )
        logger.info(f"[POD] Starting POD job | base='{base}', cam={cam}")
        # Determine k_modes from config (if present) else default 10
        k_modes = 10
        try:
            for entry in cfg.post_processing or []:
                if entry.get("type") == "POD":
                    k_modes = int(entry.get("settings", {}).get("k_modes", k_modes))
                    break
        except Exception:
            pass

        # Note: Cancellation is best-effort; pod_decompose does not currently poll cancel event.
        pod_decompose(cam_num=int(cam), config=cfg, base=Path(base), k_modes=k_modes)
        _set_state(
            progress=100,
            message="POD completed",
            processing=False,
            finished_at=datetime.utcnow().isoformat(),
        )
        logger.info("[POD] Job completed successfully")
    except Exception as e:
        logger.exception(f"[POD] Job failed: {e}")
        _set_state(
            processing=False,
            error=str(e),
            message="POD failed",
            finished_at=datetime.utcnow().isoformat(),
        )


@POD_bp.route("/start_pod", methods=["POST"])
def start_pod():
    """Start POD in a background thread using current config and provided selection.

    Expects JSON payload with optional fields:
      - basepath_idx: int (index into config.base_paths)
      - base_path: str (absolute path to base directory; takes precedence if provided)
      - camera: int or str (camera number)
    """
    global _pod_thread

    # If a job is already running, do not start another
    if _pod_thread is not None and _pod_thread.is_alive():
        with _state_lock:
            st = {k: _pod_state.get(k) for k in ("processing", "progress", "message")}
        return (
            jsonify({"status": "busy", **st}),
            409,
        )

    data = request.get_json(silent=True) or {}
    cfg = get_config(refresh=True)

    # Resolve base directory
    base_path_str = data.get("base_path")
    base: Path
    if isinstance(base_path_str, str) and base_path_str.strip():
        base = Path(base_path_str).expanduser()
    else:
        idx = int(data.get("basepath_idx", 0))
        try:
            base = cfg.base_paths[idx]
        except Exception:
            base = cfg.base_paths[0]

    # Resolve camera
    cam_raw = data.get("camera")
    try:
        cam = int(cam_raw) if cam_raw is not None else int(cfg.camera_numbers[0])
    except Exception:
        cam = int(cfg.camera_numbers[0])

    # Reset and start job
    _pod_cancel_event.clear()
    _reset_state()
    _set_state(message="POD queued")

    _pod_thread = threading.Thread(target=_run_pod_job, args=(base, cam), daemon=True)
    _pod_thread.start()

    return jsonify({"status": "started", "processing": True, "progress": 0}), 202


@POD_bp.route("/cancel_pod", methods=["POST"])
def cancel_pod():
    """Signal cancellation of a running POD job.

    Note: Current compute path does not cooperatively check for cancellation, so this is best-effort.
    We still mark state as cancelling; the job thread may continue until current computation completes.
    """
    _pod_cancel_event.set()
    with _state_lock:
        is_running = bool(_pod_thread is not None and _pod_thread.is_alive())
    if is_running:
        _set_state(message="Cancellation requested")
        return jsonify({"status": "cancelling", "processing": True}), 202
    _reset_state()
    return jsonify({"status": "idle", "processing": False}), 200


@POD_bp.route("/pod_status", methods=["GET"])
def pod_status():
    """Return current POD job status suitable for frontend polling."""
    with _state_lock:
        st = dict(_pod_state)
        st["processing"] = bool(
            st.get("processing", False)
            or (_pod_thread is not None and _pod_thread.is_alive())
        )
    # Keep progress within [0,100]
    try:
        st["progress"] = int(max(0, min(100, int(st.get("progress", 0)))))
    except Exception:
        st["progress"] = 0
    # Provide numeric 'status' for legacy clients expecting it
    st["status"] = st["progress"]
    return jsonify(st), 200


@POD_bp.route("/pod_energy", methods=["GET"])
def pod_energy():
    """Return modal energy breakdown from saved POD results for a given run.

    Query params:
      - base_path or basepath_idx
      - camera (int)
      - run (int, 1-based; default 1)
      - merged ("1"/"0")
    """
    cfg = get_config(refresh=True)

    # Resolve base directory
    base_path_str = request.args.get("base_path")
    if base_path_str and base_path_str.strip():
        base = Path(base_path_str).expanduser()
    else:
        try:
            idx = int(request.args.get("basepath_idx", 0))
        except Exception:
            idx = 0
        try:
            base = cfg.base_paths[idx]
        except Exception:
            base = cfg.base_paths[0]

    # Resolve camera
    try:
        cam = int(request.args.get("camera", cfg.camera_numbers[0]))
    except Exception:
        cam = int(cfg.camera_numbers[0])

    # Run label
    try:
        run_label = int(request.args.get("run", 1))
    except Exception:
        run_label = 1

    # Merged flag
    merged_flag = request.args.get("merged", "0") in ("1", "true", "True")

    # Find POD settings for endpoint/source_type
    endpoint = ""
    source_type = "instantaneous"
    try:
        for entry in cfg.post_processing or []:
            if entry.get("type") == "POD":
                s = entry.get("settings", {}) or {}
                endpoint = entry.get("endpoint", s.get("endpoint", "")) or ""
                source_type = (
                    entry.get("source_type", s.get("source_type", "instantaneous"))
                    or "instantaneous"
                )
                break
    except Exception:
        pass

    # Stats directory (same logic as in pod_decompose)
    paths = get_data_paths(
        base_dir=base,
        num_images=cfg.num_images,
        cam=cam,
        type_name=source_type,
        endpoint=endpoint,
        use_merged=merged_flag,
    )
    stats_base = paths["stats_dir"]

    run_dir_rand = stats_base / "pod_randomised" / f"run_{run_label:02d}"
    run_dir_exact = stats_base / "POD" / f"run_{run_label:02d}"

    joint_file = "POD_joint.mat"
    sep_file = "POD_separate.mat"

    stats_path = None
    stacked = None
    for base_dir in (run_dir_rand, run_dir_exact):
        if (base_dir / joint_file).exists():
            stats_path = base_dir / joint_file
            stacked = True
            break
        if (base_dir / sep_file).exists():
            stats_path = base_dir / sep_file
            stacked = False
            break

    if stats_path is None:
        return jsonify({"error": f"No POD stats found for run {run_label}"}), 404

    try:
        mat = loadmat(str(stats_path), struct_as_record=False, squeeze_me=True)
        # Minimal meta extraction for JSON serialisation

        def _get_meta(meta_obj, key, default=None):
            try:
                if isinstance(meta_obj, dict):
                    return meta_obj.get(key, default)
                return getattr(meta_obj, key, default)
            except Exception:
                return default

        meta_obj = mat.get("meta", {})
        meta = {
            "run_label": int(_get_meta(meta_obj, "run_label", run_label)),
            "cam": int(_get_meta(meta_obj, "cam", cam)),
            "endpoint": _get_meta(meta_obj, "endpoint", endpoint),
            "source_type": _get_meta(meta_obj, "source_type", source_type),
            "stack_U_y": bool(_get_meta(meta_obj, "stack_U_y", bool(stacked))),
            "normalise": bool(_get_meta(meta_obj, "normalise", False)),
            "algorithm": _get_meta(meta_obj, "algorithm", "exact"),
        }

        if stacked:
            ef = np.asarray(mat.get("energy_fraction", []), dtype=float).ravel()
            ec = np.asarray(mat.get("energy_cumulative", []), dtype=float).ravel()
            k = int(ef.size)
            return (
                jsonify(
                    {
                        "stacked": True,
                        "k": k,
                        "energy_fraction": ef.tolist(),
                        "energy_cumulative": ec.tolist(),
                        "meta": meta,
                    }
                ),
                200,
            )
        else:
            ef_u = np.asarray(mat.get("energy_fraction_ux", []), dtype=float).ravel()
            ec_u = np.asarray(mat.get("energy_cumulative_ux", []), dtype=float).ravel()
            ef_v = np.asarray(mat.get("energy_fraction_uy", []), dtype=float).ravel()
            ec_v = np.asarray(mat.get("energy_cumulative_uy", []), dtype=float).ravel()
            k = int(max(ef_u.size, ef_v.size))
            return (
                jsonify(
                    {
                        "stacked": False,
                        "k": k,
                        "energy_fraction_ux": ef_u.tolist(),
                        "energy_cumulative_ux": ec_u.tolist(),
                        "energy_fraction_uy": ef_v.tolist(),
                        "energy_cumulative_uy": ec_v.tolist(),
                        "meta": meta,
                    }
                ),
                200,
            )
    except Exception as e:
        logger.exception(f"[POD] Failed to read energy from {stats_path}: {e}")
        return jsonify({"error": str(e)}), 500


@POD_bp.route("/pod_energy_modes", methods=["GET"])
def pod_energy_modes():
    """
    Return modal energy breakdown (fraction and cumulative) for all modes for a given run.
    Query params:
      - base_path or basepath_idx
      - camera (int)
      - run (int, 1-based; default 1)
      - merged ("1"/"0")
    """
    cfg = get_config(refresh=True)

    # Resolve base directory
    base_path_str = request.args.get("base_path")
    if base_path_str and base_path_str.strip():
        base = Path(base_path_str).expanduser()
    else:
        try:
            idx = int(request.args.get("basepath_idx", 0))
        except Exception:
            idx = 0
        try:
            base = cfg.base_paths[idx]
        except Exception:
            base = cfg.base_paths[0]

    # Resolve camera
    try:
        cam = int(request.args.get("camera", cfg.camera_numbers[0]))
    except Exception:
        cam = int(cfg.camera_numbers[0])

    # Run label
    try:
        run_label = int(request.args.get("run", 1))
    except Exception:
        run_label = 1

    # Merged flag
    merged_flag = request.args.get("merged", "0") in ("1", "true", "True")

    # Find POD settings for endpoint/source_type
    endpoint = ""
    source_type = "instantaneous"
    try:
        for entry in cfg.post_processing or []:
            if entry.get("type") == "POD":
                s = entry.get("settings", {}) or {}
                endpoint = entry.get("endpoint", s.get("endpoint", "")) or ""
                source_type = (
                    entry.get("source_type", s.get("source_type", "instantaneous"))
                    or "instantaneous"
                )
                break
    except Exception:
        pass

    # Stats directory (same logic as in pod_decompose)
    paths = get_data_paths(
        base_dir=base,
        num_images=cfg.num_images,
        cam=cam,
        type_name=source_type,
        endpoint=endpoint,
        use_merged=merged_flag,
    )
    stats_base = paths["stats_dir"]

    # --- Search for POD energy summary in both randomised and exact directories ---
    run_dir_rand = stats_base / "pod_randomised" / f"run_{run_label:02d}"
    run_dir_exact = stats_base / "POD" / f"run_{run_label:02d}"

    # Check for the energy summary file in both locations
    summary_path = None
    stacked = None
    # Try randomised first, then exact
    for base_dir in (run_dir_rand, run_dir_exact):
        f = base_dir / "POD_energy_summary.mat"
        if f.exists():
            summary_path = f
            # Heuristic: if "energy_fraction" present, it's stacked; else separate
            try:
                mat = loadmat(str(f), struct_as_record=False, squeeze_me=True)
                stacked = "energy_fraction" in mat
                break
            except Exception:
                continue

    # If summary not found, try the joint/separate files
    if summary_path is None:
        for base_dir in (run_dir_rand, run_dir_exact):
            for file_name in ["POD_joint.mat", "POD_separate.mat"]:
                f = base_dir / file_name
                if f.exists():
                    summary_path = f
                    stacked = file_name == "POD_joint.mat"
                    break
            if summary_path is not None:
                break

    if summary_path is None:
        # Diagnostic: show where we looked
        logger.error(
            f"[POD] Could not find energy data in: {run_dir_rand} or {run_dir_exact}"
        )
        return jsonify({"error": f"No POD energy data found for run {run_label}"}), 404

    try:
        # If client only wants headers (HEAD) return quickly to avoid expensive loadmat/json work
        if request.method == "HEAD":
            return ("", 200)

        mat = loadmat(str(summary_path), struct_as_record=False, squeeze_me=True)
        meta_obj = mat.get("meta", {})

        # Minimal meta extraction for JSON serialisation
        def _get_meta(meta_obj, key, default=None):
            try:
                if isinstance(meta_obj, dict):
                    return meta_obj.get(key, default)
                return getattr(meta_obj, key, default)
            except Exception:
                return default

        # Normalize meta fields to native Python types (avoid numpy ndarrays etc.)
        meta = {
            "run_label": int(_get_meta(meta_obj, "run_label", run_label) or run_label),
            "cam": int(_get_meta(meta_obj, "cam", cam) or cam),
            "endpoint": str(_get_meta(meta_obj, "endpoint", endpoint) or endpoint),
            "source_type": str(
                _get_meta(meta_obj, "source_type", source_type) or source_type
            ),
            "stack_U_y": bool(_get_meta(meta_obj, "stack_U_y", bool(stacked))),
            "normalise": bool(_get_meta(meta_obj, "normalise", False)),
            "algorithm": str(_get_meta(meta_obj, "algorithm", "exact") or "exact"),
        }

        # Add more diagnostics to the meta information
        meta["file_path"] = str(summary_path)
        meta["file_name"] = summary_path.name

        if stacked:
            ef = np.asarray(mat.get("energy_fraction", []), dtype=float).ravel()
            ec = np.asarray(mat.get("energy_cumulative", []), dtype=float).ravel()

            # Add useful summary statistics
            total_modes = len(ef)
            threshold_95 = (
                next((i + 1 for i, v in enumerate(ec) if v >= 0.95), total_modes)
                if len(ec) > 0
                else None
            )
            threshold_99 = (
                next((i + 1 for i, v in enumerate(ec) if v >= 0.99), total_modes)
                if len(ec) > 0
                else None
            )

            return (
                jsonify(
                    {
                        "stacked": True,
                        "energy_fraction": ef.tolist(),
                        "energy_cumulative": ec.tolist(),
                        "meta": meta,
                        "summary": {
                            "total_modes": total_modes,
                            "modes_for_95_percent": threshold_95,
                            "modes_for_99_percent": threshold_99,
                            "first_mode_energy": float(ef[0]) if len(ef) > 0 else 0,
                        },
                    }
                ),
                200,
            )
        else:
            ef_u = np.asarray(mat.get("energy_fraction_ux", []), dtype=float).ravel()
            ec_u = np.asarray(mat.get("energy_cumulative_ux", []), dtype=float).ravel()
            ef_v = np.asarray(mat.get("energy_fraction_uy", []), dtype=float).ravel()
            ec_v = np.asarray(mat.get("energy_cumulative_uy", []), dtype=float).ravel()

            # Add useful summary statistics
            total_modes_u = len(ef_u)
            total_modes_v = len(ef_v)
            threshold_95_u = (
                next((i + 1 for i, v in enumerate(ec_u) if v >= 0.95), total_modes_u)
                if len(ec_u) > 0
                else None
            )
            threshold_95_v = (
                next((i + 1 for i, v in enumerate(ec_v) if v >= 0.95), total_modes_v)
                if len(ec_v) > 0
                else None
            )
            threshold_99_u = (
                next((i + 1 for i, v in enumerate(ec_u) if v >= 0.99), total_modes_u)
                if len(ec_u) > 0
                else None
            )
            threshold_99_v = (
                next((i + 1 for i, v in enumerate(ec_v) if v >= 0.99), total_modes_v)
                if len(ec_v) > 0
                else None
            )

            return (
                jsonify(
                    {
                        "stacked": False,
                        "energy_fraction_ux": ef_u.tolist(),
                        "energy_cumulative_ux": ec_u.tolist(),
                        "energy_fraction_uy": ef_v.tolist(),
                        "energy_cumulative_uy": ec_v.tolist(),
                        "meta": meta,
                        "summary_ux": {
                            "total_modes": total_modes_u,
                            "modes_for_95_percent": threshold_95_u,
                            "modes_for_99_percent": threshold_99_u,
                            "first_mode_energy": float(ef_u[0]) if len(ef_u) > 0 else 0,
                        },
                        "summary_uy": {
                            "total_modes": total_modes_v,
                            "modes_for_95_percent": threshold_95_v,
                            "modes_for_99_percent": threshold_99_v,
                            "first_mode_energy": float(ef_v[0]) if len(ef_v) > 0 else 0,
                        },
                    }
                ),
                200,
            )
    except Exception as e:
        logger.exception(f"[POD] Failed to read energy data from {summary_path}: {e}")
        return jsonify({"error": str(e)}), 500


@POD_bp.route("/pod_energy_png", methods=["GET"])
def pod_energy_png():
    """Return the cumulative POD energy PNG file for a given run if present.

    Query params: base_path or basepath_idx, camera, run, merged
    """
    cfg = get_config(refresh=True)

    # Resolve base directory
    base_path_str = request.args.get("base_path")
    if base_path_str and base_path_str.strip():
        base = Path(base_path_str).expanduser()
    else:
        try:
            idx = int(request.args.get("basepath_idx", 0))
        except Exception:
            idx = 0
        try:
            base = cfg.base_paths[idx]
        except Exception:
            base = cfg.base_paths[0]

    # Resolve camera
    try:
        cam = int(request.args.get("camera", cfg.camera_numbers[0]))
    except Exception:
        cam = int(cfg.camera_numbers[0])

    # Run label
    try:
        run_label = int(request.args.get("run", 1))
    except Exception:
        run_label = 1

    merged_flag = request.args.get("merged", "0") in ("1", "true", "True")

    # Find POD endpoint settings
    endpoint = ""
    source_type = "instantaneous"
    try:
        for entry in cfg.post_processing or []:
            if entry.get("type") == "POD":
                s = entry.get("settings", {}) or {}
                endpoint = entry.get("endpoint", s.get("endpoint", "")) or ""
                source_type = (
                    entry.get("source_type", s.get("source_type", "instantaneous"))
                    or "instantaneous"
                )
                break
    except Exception:
        pass

    paths = get_data_paths(
        base_dir=base,
        num_images=cfg.num_images,
        cam=cam,
        type_name=source_type,
        endpoint=endpoint,
        use_merged=merged_flag,
    )
    stats_base = paths["stats_dir"]

    run_dir_rand = stats_base / "pod_randomised" / f"run_{run_label:02d}"
    run_dir_exact = stats_base / "POD" / f"run_{run_label:02d}"

    candidates = []
    # try with config.plot_save_extension if available, else .png
    ext = getattr(cfg, "plot_save_extension", ".png") or ".png"
    candidates.append(run_dir_rand / f"POD_energy_cumulative{ext}")
    candidates.append(run_dir_exact / f"POD_energy_cumulative{ext}")
    candidates.append(run_dir_rand / "POD_energy_cumulative.png")
    candidates.append(run_dir_exact / "POD_energy_cumulative.png")

    for f in candidates:
        if f.exists():
            try:
                return send_file(str(f), mimetype="image/png")
            except Exception as e:
                logger.exception(f"[POD] Failed to send PNG {f}: {e}")
                break

    return jsonify({"error": "POD cumulative PNG not found"}), 404


def _resolve_base_cam_run_merged_from_request(req_args, cfg):
    """Helper used by new endpoints to resolve base, cam, run_label, merged_flag and POD settings."""
    # Resolve base directory
    base_path_str = req_args.get("base_path")
    if base_path_str and base_path_str.strip():
        base = Path(base_path_str).expanduser()
    else:
        try:
            idx = int(req_args.get("basepath_idx", 0))
        except Exception:
            idx = 0
        try:
            base = cfg.base_paths[idx]
        except Exception:
            base = cfg.base_paths[0]

    # Resolve camera
    try:
        cam = int(req_args.get("camera", cfg.camera_numbers[0]))
    except Exception:
        cam = int(cfg.camera_numbers[0])

    # Run label
    try:
        run_label = int(req_args.get("run", 1))
    except Exception:
        run_label = 1

    # Merged flag
    merged_flag = req_args.get("merged", "0") in ("1", "true", "True")

    # Find POD settings for endpoint/source_type (same heuristic used elsewhere)
    endpoint = ""
    source_type = "instantaneous"
    try:
        for entry in cfg.post_processing or []:
            if entry.get("type") == "POD":
                s = entry.get("settings", {}) or {}
                endpoint = entry.get("endpoint", s.get("endpoint", "")) or ""
                source_type = (
                    entry.get("source_type", s.get("source_type", "instantaneous"))
                    or "instantaneous"
                )
                break
    except Exception:
        pass

    return base, cam, run_label, merged_flag, endpoint, source_type


@POD_bp.route("/plot/check_pod_available", methods=["GET"])
def check_pod_available():
    """Check which POD algorithms (exact/randomised) have stats for the given run."""
    cfg = get_config(refresh=True)
    base, cam, run_label, merged_flag, endpoint, source_type = (
        _resolve_base_cam_run_merged_from_request(request.args, cfg)
    )

    paths = get_data_paths(
        base_dir=base,
        num_images=cfg.num_images,
        cam=cam,
        type_name=source_type,
        endpoint=endpoint,
        use_merged=merged_flag,
    )
    stats_base = paths["stats_dir"]

    run_dir_rand = stats_base / "pod_randomised" / f"run_{run_label:02d}"
    run_dir_exact = stats_base / "POD" / f"run_{run_label:02d}"

    joint_file = "POD_joint.mat"
    sep_file = "POD_separate.mat"

    available = {
        "exact": False,
        "randomised": False,
        "exact_joint": False,
        "randomised_joint": False,
    }
    if (run_dir_exact / joint_file).exists():
        available["exact"] = True
        available["exact_joint"] = True
    elif (run_dir_exact / sep_file).exists():
        available["exact"] = True

    if (run_dir_rand / joint_file).exists():
        available["randomised"] = True
        available["randomised_joint"] = True
    elif (run_dir_rand / sep_file).exists():
        available["randomised"] = True

    return jsonify({"available": available, "run": run_label, "cam": int(cam)}), 200


@POD_bp.route("/plot/get_pod_energy", methods=["GET"])
def get_pod_energy():
    """Return POD energy summary for a run/algorithm. Query args: algorithm=exact|randomised (optional)."""
    cfg = get_config(refresh=True)
    base, cam, run_label, merged_flag, endpoint, source_type = (
        _resolve_base_cam_run_merged_from_request(request.args, cfg)
    )

    algorithm = (
        request.args.get("algorithm") or ""
    ).lower()  # "exact" or "randomised" preferred
    alg_folder = None

    paths = get_data_paths(
        base_dir=base,
        num_images=cfg.num_images,
        cam=cam,
        type_name=source_type,
        endpoint=endpoint,
        use_merged=merged_flag,
    )
    stats_base = paths["stats_dir"]
    run_dir_rand = stats_base / "pod_randomised" / f"run_{run_label:02d}"
    run_dir_exact = stats_base / "POD" / f"run_{run_label:02d}"

    # Choose folder based on algorithm requested, else prefer randomised if present
    if algorithm == "randomised":
        alg_folder = run_dir_rand
    elif algorithm == "exact":
        alg_folder = run_dir_exact
    else:
        # auto-select: prefer randomised summary, else exact
        if (
            (run_dir_rand / "POD_energy_summary.mat").exists()
            or (run_dir_rand / "POD_joint.mat").exists()
            or (run_dir_rand / "POD_separate.mat").exists()
        ):
            alg_folder = run_dir_rand
            algorithm = "randomised"
        else:
            alg_folder = run_dir_exact
            algorithm = "exact"

    # Look for POD_energy_summary.mat first, then fallback to joint/separate files
    summary_candidates = [
        alg_folder / "POD_energy_summary.mat",
        alg_folder / "POD_joint.mat",
        alg_folder / "POD_separate.mat",
    ]
    summary_path = next((f for f in summary_candidates if f.exists()), None)
    if summary_path is None:
        return (
            jsonify(
                {
                    "error": f"No POD energy summary found for run {run_label} under algorithm '{algorithm}'"
                }
            ),
            404,
        )

    try:
        mat = loadmat(str(summary_path), struct_as_record=False, squeeze_me=True)
        # Prepare JSON-able dict: arrays -> lists, meta -> dict
        meta_obj = mat.get("meta", {}) or {}

        def _to_py(v):
            if v is None:
                return None
            try:
                return np.asarray(v).tolist()
            except Exception:
                return v

        # Determine stacked vs separate heuristically
        stacked = "energy_fraction" in mat or "POD_joint" in summary_path.name
        out = {"meta": {}}
        # convert meta to native types
        try:
            if isinstance(meta_obj, dict):
                for k, vv in meta_obj.items():
                    try:
                        out["meta"][k] = (
                            vv.item()
                            if hasattr(vv, "item") and np.ndim(vv) == 0
                            else vv
                        )
                    except Exception:
                        out["meta"][k] = vv
            else:
                # struct-like object
                for k in [
                    "run_label",
                    "cam",
                    "endpoint",
                    "source_type",
                    "stack_U_y",
                    "normalise",
                    "algorithm",
                ]:
                    val = getattr(meta_obj, k, None)
                    if val is not None:
                        out["meta"][k] = val
        except Exception:
            out["meta"] = meta_obj if isinstance(meta_obj, dict) else {}

        out["algorithm"] = algorithm
        out["stacked"] = bool(stacked)

        if stacked:
            out["energy_fraction"] = _to_py(mat.get("energy_fraction", []))
            out["energy_cumulative"] = _to_py(mat.get("energy_cumulative", []))
            out["singular_values"] = _to_py(mat.get("singular_values", []))
            out["eigenvalues"] = _to_py(mat.get("eigenvalues", []))
        else:
            out["energy_fraction_ux"] = _to_py(mat.get("energy_fraction_ux", []))
            out["energy_cumulative_ux"] = _to_py(mat.get("energy_cumulative_ux", []))
            out["energy_fraction_uy"] = _to_py(mat.get("energy_fraction_uy", []))
            out["energy_cumulative_uy"] = _to_py(mat.get("energy_cumulative_uy", []))
            out["singular_values_ux"] = _to_py(mat.get("singular_values_ux", []))
            out["singular_values_uy"] = _to_py(mat.get("singular_values_uy", []))
            out["eigenvalues_ux"] = _to_py(mat.get("eigenvalues_ux", []))
            out["eigenvalues_uy"] = _to_py(mat.get("eigenvalues_uy", []))

        return jsonify(out), 200
    except Exception as e:
        logger.exception(f"[POD] Failed to load energy summary {summary_path}: {e}")
        return jsonify({"error": str(e)}), 500


@POD_bp.route("/plot_pod_mode", methods=["GET"])
def plot_pod_mode():
    """
    Return a base64 PNG for requested POD mode.
    Query params:
      base_path/basepath_idx, camera, run, mode (1-based), component (ux/uy), algorithm (exact/randomised), merged,
      cmap (matplotlib name), lower_limit, upper_limit
    """
    cfg = get_config(refresh=True)
    base, cam, run_label, merged_flag, endpoint, source_type = (
        _resolve_base_cam_run_merged_from_request(request.args, cfg)
    )

    # params
    try:
        mode_idx = int(request.args.get("mode", 1))
    except Exception:
        mode_idx = 1
    component = (request.args.get("component") or "ux").lower()
    algorithm = (request.args.get("algorithm") or "").lower()
    cmap = request.args.get("cmap") or "viridis"
    lower = request.args.get("lower_limit")
    upper = request.args.get("upper_limit")
    vmin = float(lower) if lower is not None and str(lower) != "" else None
    vmax = float(upper) if upper is not None and str(upper) != "" else None

    paths = get_data_paths(
        base_dir=base,
        num_images=cfg.num_images,
        cam=cam,
        type_name=source_type,
        endpoint=endpoint,
        use_merged=merged_flag,
    )
    stats_base = paths["stats_dir"]

    # select algorithm folder
    if algorithm == "randomised":
        alg_folder = stats_base / "pod_randomised" / f"run_{run_label:02d}"
    elif algorithm == "exact":
        alg_folder = stats_base / "POD" / f"run_{run_label:02d}"
    else:
        # prefer randomised if present
        candidate_rand = stats_base / "pod_randomised" / f"run_{run_label:02d}"
        candidate_exact = stats_base / "POD" / f"run_{run_label:02d}"
        if (candidate_rand / f"{component}_mode_{mode_idx:02d}.mat").exists() or (
            candidate_rand / "POD_joint.mat"
        ).exists():
            alg_folder = candidate_rand
            algorithm = "randomised"
        else:
            alg_folder = candidate_exact
            algorithm = "exact"

    mode_file = alg_folder / f"{component}_mode_{mode_idx:02d}.mat"
    logger.debug(f"[POD] Plotting mode from file: {mode_file}")
    if not mode_file.exists():
        return jsonify({"error": f"Mode file not found: {mode_file}"}), 404

    try:
        mat = loadmat(str(mode_file), struct_as_record=False, squeeze_me=True)
        mode_arr = np.asarray(mat.get("mode"))
        mask = mat.get("mask", None)
        if mask is not None:
            mask = np.asarray(mask).astype(bool)
        else:
            # try meta mask
            mask = np.zeros_like(mode_arr, dtype=bool)

        # Create masked array where mask True indicates masked/invalid
        masked = np.ma.array(mode_arr, mask=mask)

        # Plot
        fig = plt.figure(figsize=(6, 4), dpi=150)
        ax = fig.add_subplot(111)
        cmap_obj = mpl.cm.get_cmap(cmap)
        if vmin is None or vmax is None:
            im = ax.imshow(masked, cmap=cmap_obj)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
            im = ax.imshow(masked, cmap=cmap_obj, norm=norm)
        ax.set_axis_off()
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")

        meta = {
            "run": int(run_label),
            "cam": int(cam),
            "component": component,
            "mode": int(mode_idx),
            "algorithm": algorithm,
            "file": str(mode_file),
        }
        return jsonify({"image_base64": b64, "meta": meta}), 200
    except Exception as e:
        logger.exception(f"[POD] Failed to render mode {mode_file}: {e}")
        return jsonify({"error": str(e)}), 500


@POD_bp.route("/plot/list_pod_modes", methods=["GET"])
def list_pod_modes():
    """Return how many ux/uy mode files exist for the given run/algorithm."""
    cfg = get_config(refresh=True)
    base, cam, run_label, merged_flag, endpoint, source_type = (
        _resolve_base_cam_run_merged_from_request(request.args, cfg)
    )
    algorithm = (request.args.get("algorithm") or "").lower()

    paths = get_data_paths(
        base_dir=base,
        num_images=cfg.num_images,
        cam=cam,
        type_name=source_type,
        endpoint=endpoint,
        use_merged=merged_flag,
    )
    stats_base = paths["stats_dir"]

    if algorithm == "randomised":
        alg_folder = stats_base / "pod_randomised" / f"run_{run_label:02d}"
    elif algorithm == "exact":
        alg_folder = stats_base / "POD" / f"run_{run_label:02d}"
    else:
        # aggregate both if algorithm unspecified; prefer randomised folder if exists
        candidate_rand = stats_base / "pod_randomised" / f"run_{run_label:02d}"
        candidate_exact = stats_base / "POD" / f"run_{run_label:02d}"
        alg_folder = candidate_rand if candidate_rand.exists() else candidate_exact

    ux_files = list(alg_folder.glob("ux_mode_*.mat")) if alg_folder.exists() else []
    uy_files = list(alg_folder.glob("uy_mode_*.mat")) if alg_folder.exists() else []

    return (
        jsonify(
            {
                "run": run_label,
                "cam": int(cam),
                "algorithm": algorithm or None,
                "ux_count": len(ux_files),
                "uy_count": len(uy_files),
                "files_exist": alg_folder.exists(),
                "folder": str(alg_folder),
            }
        ),
        200,
    )


@POD_bp.route("/plot/get_pod_mode_data", methods=["GET"])
def get_pod_mode_data():
    """Return raw mode array and mask as lists for the requested mode."""
    cfg = get_config(refresh=True)
    base, cam, run_label, merged_flag, endpoint, source_type = (
        _resolve_base_cam_run_merged_from_request(request.args, cfg)
    )

    try:
        mode_idx = int(request.args.get("mode", 1))
    except Exception:
        mode_idx = 1
    component = (request.args.get("component") or "ux").lower()
    algorithm = (request.args.get("algorithm") or "").lower()

    paths = get_data_paths(
        base_dir=base,
        num_images=cfg.num_images,
        cam=cam,
        type_name=source_type,
        endpoint=endpoint,
        use_merged=merged_flag,
    )
    stats_base = paths["stats_dir"]

    if algorithm == "randomised":
        alg_folder = stats_base / "pod_randomised" / f"run_{run_label:02d}"
    elif algorithm == "exact":
        alg_folder = stats_base / "POD" / f"run_{run_label:02d}"
    else:
        candidate_rand = stats_base / "pod_randomised" / f"run_{run_label:02d}"
        candidate_exact = stats_base / "POD" / f"run_{run_label:02d}"
        alg_folder = candidate_rand if candidate_rand.exists() else candidate_exact

    mode_file = alg_folder / f"{component}_mode_{mode_idx:02d}.mat"
    if not mode_file.exists():
        return jsonify({"error": f"Mode file not found: {mode_file}"}), 404

    try:
        mat = loadmat(str(mode_file), struct_as_record=False, squeeze_me=True)
        mode_arr = np.asarray(mat.get("mode"))
        mask = mat.get("mask", None)
        if mask is not None:
            mask = np.asarray(mask).astype(bool)
        else:
            mask = np.zeros_like(mode_arr, dtype=bool)

        return (
            jsonify(
                {
                    "run": run_label,
                    "cam": int(cam),
                    "algorithm": algorithm or None,
                    "component": component,
                    "mode": mode_idx,
                    "mode_array": mode_arr.tolist(),
                    "mask": mask.tolist(),
                    "file": str(mode_file),
                }
            ),
            200,
        )
    except Exception as e:
        logger.exception(f"[POD] Failed to load mode data {mode_file}: {e}")
        return jsonify({"error": str(e)}), 500

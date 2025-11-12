import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from flask import Blueprint, jsonify, request, send_file
from loguru import logger
from scipy.io import loadmat

from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths
from ..pod_decompose import pod_decompose

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

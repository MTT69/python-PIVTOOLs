import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request, send_file
from loguru import logger
import numpy as np
from scipy.io import loadmat

from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths
from pivtools_gui.calibration.services.job_manager import job_manager
from pivtools_gui.video_maker.video_maker import (
    VideoMaker,
    find_all_valid_runs_from_file,
)

video_maker_bp = Blueprint("video_maker", __name__)

# Constants
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
MAX_DEPTH = 5  # For deep search


def check_video_data_availability(
    base_path: Path,
    camera: int,
    num_frame_pairs: int,
    vector_format: str,
) -> Dict[str, Any]:
    """
    Check what data sources are available for video creation.
    Returns availability info for calibrated, uncalibrated, merged, and inst_stats data.
    """
    available = {
        "calibrated": {"exists": False, "frame_count": 0, "path": None},
        "uncalibrated": {"exists": False, "frame_count": 0, "path": None},
        "merged": {"exists": False, "frame_count": 0, "path": None},
        "inst_stats": {"exists": False, "frame_count": 0, "path": None},
    }

    # Check calibrated instantaneous
    try:
        cal_paths = get_data_paths(
            base_dir=base_path,
            num_frame_pairs=num_frame_pairs,
            cam=camera,
            type_name="instantaneous",
            use_uncalibrated=False,
            use_merged=False,
        )
        cal_data_dir = Path(cal_paths["data_dir"])
        if cal_data_dir.exists():
            frame_count = len(list(cal_data_dir.glob("[0-9]*.mat")))
            if frame_count > 0:
                available["calibrated"]["exists"] = True
                available["calibrated"]["frame_count"] = frame_count
                available["calibrated"]["path"] = str(cal_data_dir)
    except Exception as e:
        logger.debug(f"Error checking calibrated data: {e}")

    # Check uncalibrated instantaneous
    try:
        uncal_paths = get_data_paths(
            base_dir=base_path,
            num_frame_pairs=num_frame_pairs,
            cam=camera,
            type_name="instantaneous",
            use_uncalibrated=True,
            use_merged=False,
        )
        uncal_data_dir = Path(uncal_paths["data_dir"])
        if uncal_data_dir.exists():
            frame_count = len(list(uncal_data_dir.glob("[0-9]*.mat")))
            if frame_count > 0:
                available["uncalibrated"]["exists"] = True
                available["uncalibrated"]["frame_count"] = frame_count
                available["uncalibrated"]["path"] = str(uncal_data_dir)
    except Exception as e:
        logger.debug(f"Error checking uncalibrated data: {e}")

    # Check merged instantaneous
    try:
        merged_paths = get_data_paths(
            base_dir=base_path,
            num_frame_pairs=num_frame_pairs,
            cam=camera,
            type_name="instantaneous",
            use_uncalibrated=False,
            use_merged=True,
        )
        merged_data_dir = Path(merged_paths["data_dir"])
        if merged_data_dir.exists():
            frame_count = len(list(merged_data_dir.glob("[0-9]*.mat")))
            if frame_count > 0:
                available["merged"]["exists"] = True
                available["merged"]["frame_count"] = frame_count
                available["merged"]["path"] = str(merged_data_dir)
    except Exception as e:
        logger.debug(f"Error checking merged data: {e}")

    # Check instantaneous statistics
    try:
        stats_dir = base_path / "statistics" / str(num_frame_pairs) / f"Cam{camera}" / "instantaneous" / "instantaneous_stats"
        if stats_dir.exists():
            frame_count = len(list(stats_dir.glob("[0-9]*.mat")))
            if frame_count > 0:
                available["inst_stats"]["exists"] = True
                available["inst_stats"]["frame_count"] = frame_count
                available["inst_stats"]["path"] = str(stats_dir)
    except Exception as e:
        logger.debug(f"Error checking inst_stats data: {e}")

    return available


# Thread-local cancel events for job cancellation
_cancel_events: Dict[str, threading.Event] = {}
_cancel_events_lock = threading.Lock()


@video_maker_bp.route("/list_videos", methods=["GET"])
def list_videos():
    """Optimized video listing with glob and caching."""
    try:
        base_path_str = request.args.get("base_path")
        cfg = get_config(refresh=True)

        base = Path(base_path_str).expanduser() if base_path_str else cfg.base_paths[0]

        logger.info(f"[VIDEO] Listing videos under base path: {base}")

        videos: List[str] = []

        videos_dir = base / "videos"
        if videos_dir.exists():
            for ext in VIDEO_EXTENSIONS:
                videos.extend([str(f) for f in videos_dir.glob(f"**/*{ext}")])

        cam_dirs = [d for d in base.glob("**/Cam*") if d.is_dir()]
        for cam_dir in cam_dirs:
            for video_subdir in ["videos", "merged/videos"]:
                video_dir = cam_dir / video_subdir
                if video_dir.exists():
                    for ext in VIDEO_EXTENSIONS:
                        videos.extend([str(f) for f in video_dir.glob(f"*{ext}")])

        if not videos:

            def find_videos(directory: Path, current_depth: int = 0) -> List[str]:
                if current_depth > MAX_DEPTH:
                    return []
                found = []
                try:
                    for item in directory.iterdir():
                        if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
                            found.append(str(item))
                        elif item.is_dir():
                            found.extend(find_videos(item, current_depth + 1))
                except (PermissionError, OSError):
                    pass
                return found

            videos = find_videos(base)

        videos.sort(
            key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True
        )

        logger.info(f"[VIDEO] Found {len(videos)} videos")
        return jsonify({"videos": videos})
    except Exception as e:
        logger.exception(f"[VIDEO] Failed to list videos: {e}")
        return jsonify({"error": str(e), "videos": []}), 500


@video_maker_bp.route("/check_data_sources", methods=["GET"])
def check_data_sources():
    """
    Check what data sources are available for video creation.

    Query params:
    - base_path: Base directory path
    - camera: Camera number (1-based)

    Returns availability info for calibrated, uncalibrated, and merged data.
    """
    try:
        base_path_str = request.args.get("base_path")
        camera_raw = request.args.get("camera", "1")

        cfg = get_config(refresh=True)

        if not base_path_str:
            # Fall back to first configured base path
            if cfg.base_paths:
                base_path_str = cfg.base_paths[0]
            else:
                return jsonify({
                    "success": False,
                    "error": "No base_path provided and no default configured"
                }), 400

        try:
            camera = int(camera_raw)
            if camera < 1:
                raise ValueError("Camera must be positive")
        except ValueError:
            return jsonify({"success": False, "error": "Invalid camera number"}), 400

        base_path = Path(base_path_str).expanduser()
        if not base_path.exists():
            return jsonify({
                "success": False,
                "error": f"Base path does not exist: {base_path}"
            }), 404

        available = check_video_data_availability(
            base_path=base_path,
            camera=camera,
            num_frame_pairs=cfg.num_frame_pairs,
            vector_format=cfg.vector_format,
        )

        # Determine default data source
        default_source = None
        if available["merged"]["exists"]:
            default_source = "merged"
        elif available["calibrated"]["exists"]:
            default_source = "calibrated"
        elif available["uncalibrated"]["exists"]:
            default_source = "uncalibrated"

        has_any_data = any(v["exists"] for v in available.values())

        return jsonify({
            "success": True,
            "available": available,
            "default_source": default_source,
            "has_any_data": has_any_data,
            "base_path": str(base_path),
            "camera": camera,
        })

    except Exception as e:
        logger.exception(f"[VIDEO] Failed to check data sources: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@video_maker_bp.route("/available_variables", methods=["GET"])
def available_variables():
    """
    Check what variables are available for video creation.
    Returns base PIV variables plus any computed instantaneous stats.

    Query params:
    - base_path: Base directory path
    - camera: Camera number (1-based)
    - data_source: Data source type (calibrated, uncalibrated, merged)

    Returns:
    - variables: list of {name, label, group} for dropdown population
    - has_stereo: whether uz (stereo) data is available
    - has_inst_stats: whether instantaneous statistics have been computed
    """
    try:
        base_path_str = request.args.get("base_path")
        camera_raw = request.args.get("camera", "1")
        data_source = request.args.get("data_source", "calibrated")

        cfg = get_config(refresh=True)

        if not base_path_str:
            if cfg.base_paths:
                base_path_str = cfg.base_paths[0]
            else:
                return jsonify({"success": False, "error": "No base_path provided"}), 400

        try:
            camera = int(camera_raw)
            if camera < 1:
                raise ValueError("Camera must be positive")
        except ValueError:
            return jsonify({"success": False, "error": "Invalid camera number"}), 400

        base_path = Path(base_path_str).expanduser()

        # Build base variables (always available from PIV data)
        variables = [
            {"name": "ux", "label": "Velocity (x)", "group": "piv"},
            {"name": "uy", "label": "Velocity (y)", "group": "piv"},
            {"name": "mag", "label": "Velocity Magnitude", "group": "piv"},
        ]

        # Determine flags based on data_source
        use_uncalibrated = data_source == "uncalibrated"
        use_merged = data_source == "merged"

        # Get data paths
        paths = get_data_paths(
            base_dir=base_path,
            num_frame_pairs=cfg.num_frame_pairs,
            cam=camera,
            type_name="instantaneous",
            use_uncalibrated=use_uncalibrated,
            use_merged=use_merged,
        )

        data_dir = Path(paths["data_dir"])
        has_stereo = False

        # Check for stereo (uz) in first PIV file
        if data_dir.exists():
            mat_files = sorted(data_dir.glob("[0-9]*.mat"))
            mat_files = [f for f in mat_files if "coordinate" not in f.name.lower()][:1]
            if mat_files:
                try:
                    mat = loadmat(str(mat_files[0]), struct_as_record=False, squeeze_me=True)
                    piv_result = mat.get("piv_result")
                    if piv_result is not None:
                        if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
                            pr = piv_result[0]
                        else:
                            pr = piv_result
                        has_stereo = hasattr(pr, "uz") and pr.uz is not None and np.asarray(pr.uz).size > 0
                except Exception as e:
                    logger.debug(f"Error checking stereo: {e}")

        if has_stereo:
            variables.append({"name": "uz", "label": "Velocity (z)", "group": "piv"})

        # Check for instantaneous statistics
        # Stats path: {base_dir}/statistics/{num_frame_pairs}/{camera}/instantaneous/instantaneous_stats/
        stats_base = base_path / "statistics" / str(cfg.num_frame_pairs) / f"Cam{camera}" / "instantaneous"
        inst_stats_dir = stats_base / "instantaneous_stats"
        has_inst_stats = inst_stats_dir.exists() and any(inst_stats_dir.glob("*.mat"))

        if has_inst_stats:
            # Check what fields are available in first stats file
            inst_files = sorted(inst_stats_dir.glob("*.mat"))[:1]
            if inst_files:
                try:
                    mat = loadmat(str(inst_files[0]), struct_as_record=False, squeeze_me=True)
                    piv_result = mat.get("piv_result")
                    if piv_result is not None:
                        if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
                            pr = piv_result[0]
                        else:
                            pr = piv_result

                        # Check for each potential stat field
                        stat_fields = [
                            ("u_prime", "u' (fluctuation x)", "stats"),
                            ("v_prime", "v' (fluctuation y)", "stats"),
                            ("w_prime", "w' (fluctuation z)", "stats"),  # stereo only
                            ("vorticity", "Vorticity", "stats"),
                            ("divergence", "Divergence", "stats"),
                            ("gamma1", "Gamma1 (vortex)", "stats"),
                            ("gamma2", "Gamma2 (vortex)", "stats"),
                        ]

                        for field_name, label, group in stat_fields:
                            if hasattr(pr, field_name):
                                arr = np.asarray(getattr(pr, field_name))
                                if arr.size > 0 and not np.all(np.isnan(arr)):
                                    variables.append({
                                        "name": field_name,
                                        "label": label,
                                        "group": group
                                    })
                except Exception as e:
                    logger.debug(f"Error checking stats fields: {e}")

        return jsonify({
            "success": True,
            "variables": variables,
            "has_stereo": has_stereo,
            "has_inst_stats": has_inst_stats,
        })

    except Exception as e:
        logger.exception(f"[VIDEO] Failed to get available variables: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@video_maker_bp.route("/video/config", methods=["POST"])
def video_config():
    """
    Update video configuration in config.yaml.

    This endpoint updates the video block in config.yaml before video creation,
    following the calibration pattern where config is the source of truth.

    Request JSON (all optional):
        base_path_idx: int - Index into base_paths list
        camera: int - Camera number (1-based)
        data_source: str - 'calibrated', 'uncalibrated', 'merged', 'inst_stats'
        variable: str - Variable name (ux, uy, mag, etc.)
        run: int - Run number (1-based)
        piv_type: str - 'instantaneous' or 'ensemble'
        cmap: str - Colormap name ('default' for auto)
        lower: str/float - Lower color limit ('' for auto)
        upper: str/float - Upper color limit ('' for auto)
        fps: int - Frame rate
        crf: int - Quality factor
        resolution: str - '1080p' or '4k'

    Returns:
        JSON with status, updated config values
    """
    data = request.get_json() or {}
    cfg = get_config()

    # Ensure video block exists
    if "video" not in cfg.data:
        cfg.data["video"] = {}

    # Field mapping: json_key -> (config_key, type_converter)
    field_mapping = {
        "base_path_idx": ("base_path_idx", int),
        "camera": ("camera", int),
        "data_source": ("data_source", str),
        "variable": ("variable", str),
        "run": ("run", int),
        "piv_type": ("piv_type", str),
        "cmap": ("cmap", str),
        "lower": ("lower", str),
        "upper": ("upper", str),
        "fps": ("fps", int),
        "crf": ("crf", int),
        "resolution": ("resolution", str),
    }

    updated = {}
    for json_key, (config_key, type_fn) in field_mapping.items():
        if json_key in data:
            val = data[json_key]
            # Handle special cases for lower/upper (keep as string for auto detection)
            if json_key in ("lower", "upper"):
                cfg.data["video"][config_key] = str(val) if val not in (None, "") else ""
            else:
                try:
                    cfg.data["video"][config_key] = type_fn(val) if val is not None else None
                except (ValueError, TypeError):
                    cfg.data["video"][config_key] = val
            updated[config_key] = cfg.data["video"][config_key]

    # Save config to disk
    cfg.save()
    logger.info(f"[VIDEO] Updated config: {updated}")

    return jsonify({
        "status": "ok",
        "updated": updated,
        "video_config": cfg.data.get("video", {}),
    })


@video_maker_bp.route("/start_video", methods=["POST"])
def start_video():
    """
    Start video job using config.yaml as primary source.

    The frontend should call /video/config first to update settings in config.yaml.
    This endpoint reads parameters from config, with request params as optional overrides.

    Request JSON (all optional - defaults from config.yaml):
    - base_path_idx: int - Index into base_paths list (primary method)
    - source_path_idx: int - Alias for base_path_idx (backward compat)
    - base_path: str - Direct path override (backward compat)
    - test_mode: bool - Create test video with limited frames
    - test_frames: int - Number of frames for test mode (default: 50)
    - out_name: str - Custom output filename

    All other parameters (camera, run, var, data_source, fps, cmap, limits, resolution)
    are read from config.yaml's video block.
    """
    data = request.get_json(silent=True) or {}
    cfg = get_config(refresh=True)

    # Get base path from request or config
    # Priority: base_path (direct) > base_path_idx > source_path_idx > config.video_base_path_idx
    base_path_str = data.get("base_path")
    if base_path_str:
        base = Path(base_path_str).expanduser()
    else:
        base_path_idx = data.get("base_path_idx", data.get("source_path_idx", cfg.video_base_path_idx))
        try:
            base_path_idx = int(base_path_idx)
        except (ValueError, TypeError):
            base_path_idx = 0
        if base_path_idx >= len(cfg.base_paths):
            return jsonify({"error": f"Invalid base_path_idx {base_path_idx}. Only {len(cfg.base_paths)} paths configured."}), 400
        base = cfg.base_paths[base_path_idx]

    if not base.exists():
        return jsonify({"error": f"Base path does not exist: {base}"}), 400

    # Read parameters from config (with request overrides for backward compat)
    cam = data.get("camera", cfg.video_camera)
    try:
        cam = int(cam)
        if cam < 1:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid camera number"}), 400

    run = data.get("run", cfg.video_run)
    try:
        run = int(run)
        if run < 1:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid run number"}), 400

    data_source = data.get("data_source", cfg.video_data_source)
    valid_sources = ("calibrated", "uncalibrated", "merged", "inst_stats")
    if data_source not in valid_sources:
        return jsonify({"error": f"Invalid data_source. Must be one of: {', '.join(valid_sources)}"}), 400

    var = data.get("var", cfg.video_variable)

    # Test mode params (not persisted to config)
    test_mode = data.get("test_mode", False)
    if not isinstance(test_mode, bool):
        return jsonify({"error": "test_mode must be boolean"}), 400
    test_frames = int(data.get("test_frames", 50))
    if test_frames < 1:
        return jsonify({"error": "test_frames must be positive"}), 400

    # Check if data is available for the selected source
    available = check_video_data_availability(
        base_path=base,
        camera=cam,
        num_frame_pairs=cfg.num_frame_pairs,
        vector_format=cfg.vector_format,
    )

    # For stats variables, auto-switch to inst_stats source
    STATS_VARIABLES = {"u_prime", "v_prime", "w_prime", "vorticity", "divergence", "gamma1", "gamma2"}
    if var in STATS_VARIABLES and data_source != "inst_stats":
        data_source = "inst_stats"
        logger.info(f"[VIDEO] Auto-switching to inst_stats for variable '{var}'")

    if not available[data_source]["exists"]:
        available_sources = [k for k, v in available.items() if v["exists"]]
        if not available_sources:
            return jsonify({
                "error": f"No PIV data found for camera {cam}. Please run PIV processing first.",
                "available_sources": []
            }), 404
        else:
            return jsonify({
                "error": f"No {data_source} data found for camera {cam}. Available sources: {', '.join(available_sources)}",
                "available_sources": available_sources,
                "selected_source": data_source
            }), 404

    # Validate variable
    VALID_VARIABLES = {
        "ux", "uy", "uz", "mag",  # PIV variables
        "u_prime", "v_prime", "w_prime",  # Fluctuation stats
        "vorticity", "divergence", "gamma1", "gamma2",  # Derived stats
    }
    if var not in VALID_VARIABLES:
        return jsonify({"error": f"Invalid var '{var}'. Valid options: {', '.join(sorted(VALID_VARIABLES))}"}), 400

    # Get video parameters from config (with request overrides for backward compat)
    fps = data.get("fps", cfg.video_fps)
    try:
        fps = int(fps)
        if fps < 1 or fps > 120:
            return jsonify({"error": "FPS must be between 1 and 120"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid FPS value"}), 400

    crf = cfg.video_crf

    # Resolution: request override or config
    resolution_str = data.get("resolution", cfg.video_resolution_str)
    if resolution_str == "4k":
        resolution = (2160, 3840)
    else:
        resolution = cfg.video_resolution

    # Color limits: request override or config
    lower = data.get("lower") if "lower" in data else cfg.video_lower_limit
    upper = data.get("upper") if "upper" in data else cfg.video_upper_limit
    try:
        lower_limit = float(lower) if lower and str(lower).strip() else None
        upper_limit = float(upper) if upper and str(upper).strip() else None
    except (ValueError, TypeError):
        lower_limit = None
        upper_limit = None

    # Colormap: request override or config
    cmap = data.get("cmap") if "cmap" in data else cfg.video_cmap
    if cmap == "default":
        cmap = None

    out_name = data.get("out_name")

    # Create job via job_manager
    job_id = job_manager.create_job(
        "video",
        camera=cam,
        variable=var,
        data_source=data_source,
        run=run,
        current_frame=0,
        total_frames=0,
    )

    # Create cancel event for this job
    cancel_event = threading.Event()
    with _cancel_events_lock:
        _cancel_events[job_id] = cancel_event

    def run_video():
        try:
            job_manager.update_job(job_id, status="running")

            # Create VideoMaker instance
            maker = VideoMaker(
                base_dir=base,
                camera=cam,
                config=cfg,
            )

            def progress_cb(current, total, msg=""):
                progress = int(current / max(total, 1) * 100)
                job_manager.update_job(
                    job_id,
                    progress=progress,
                    current_frame=current,
                    total_frames=total,
                    message=f"Processing frame {current}/{total}" + (f" - {msg}" if msg else ""),
                )

            # Run video generation using process_video
            result = maker.process_video(
                variable=var,
                run=run,
                data_source=data_source,
                fps=fps,
                crf=crf,
                resolution=resolution,
                cmap=cmap,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                test_mode=test_mode,
                test_frames=test_frames,
                out_name=out_name,
                progress_callback=progress_cb,
                cancel_event=cancel_event,
            )

            if cancel_event.is_set():
                job_manager.fail_job(job_id, "Cancelled by user")
            elif result.get("success"):
                job_manager.complete_job(
                    job_id,
                    out_path=result.get("out_path"),
                    vmin=result.get("vmin"),
                    vmax=result.get("vmax"),
                    actual_min=result.get("actual_min"),
                    actual_max=result.get("actual_max"),
                    effective_run=result.get("effective_run"),
                    frames=result.get("frames"),
                    elapsed_sec=result.get("elapsed_sec"),
                    data_source=result.get("data_source"),
                    computed_limits={
                        "lower": result.get("vmin"),
                        "upper": result.get("vmax"),
                        "actual_min": result.get("actual_min"),
                        "actual_max": result.get("actual_max"),
                        "percentile_based": lower_limit is None or upper_limit is None,
                    },
                )
                logger.info(f"[VIDEO] Job {job_id} completed: {result.get('out_path')}")
            else:
                job_manager.fail_job(job_id, result.get("error", "Unknown error"))

        except Exception as e:
            logger.exception(f"[VIDEO] Job {job_id} failed: {e}")
            job_manager.fail_job(job_id, str(e))
        finally:
            # Clean up cancel event
            with _cancel_events_lock:
                _cancel_events.pop(job_id, None)

    thread = threading.Thread(target=run_video, daemon=True)
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "starting",
        "data_source": data_source,
        "frame_count": available[data_source]["frame_count"],
    }), 202


@video_maker_bp.route("/cancel_video", methods=["POST"])
def cancel_video():
    """
    Cancel a video job.

    Request JSON (optional):
        job_id: str - Specific job ID to cancel. If not provided, cancels all running video jobs.
    """
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")

    if job_id:
        # Cancel specific job
        with _cancel_events_lock:
            cancel_event = _cancel_events.get(job_id)
        if cancel_event:
            cancel_event.set()
            job_manager.update_job(job_id, message="Cancellation requested")
            return jsonify({"status": "cancelling", "job_id": job_id}), 202
        else:
            return jsonify({"error": "Job not found or already completed", "job_id": job_id}), 404
    else:
        # Cancel all running video jobs
        cancelled = []
        with _cancel_events_lock:
            for jid, event in list(_cancel_events.items()):
                event.set()
                job_manager.update_job(jid, message="Cancellation requested")
                cancelled.append(jid)
        if cancelled:
            return jsonify({"status": "cancelling", "cancelled_jobs": cancelled}), 202
        return jsonify({"status": "idle", "message": "No running video jobs"}), 200


@video_maker_bp.route("/video/job/<job_id>", methods=["GET"])
def video_job_status(job_id: str):
    """
    Get video job status by ID (matches calibration pattern).

    Returns:
        JSON with status, progress, current_frame, total_frames,
        elapsed_time, estimated_remaining, error (if failed), etc.
    """
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job_data)


@video_maker_bp.route("/video_status", methods=["GET"])
def video_status():
    """
    Get status of video jobs.

    Query params:
        job_id: str (optional) - Specific job ID to query

    Returns status from job_manager.
    """
    job_id = request.args.get("job_id")

    if job_id:
        # Get specific job status
        job_data = job_manager.get_job_with_timing(job_id)
        if job_data is None:
            return jsonify({"error": "Job not found", "processing": False}), 404
        # Add processing flag for compatibility
        job_data["processing"] = job_data.get("status") == "running"
        return jsonify(job_data), 200
    else:
        # Get all video jobs (for backward compatibility)
        video_jobs = job_manager.list_jobs(job_type="video")
        if not video_jobs:
            return jsonify({"processing": False, "message": "No video jobs"}), 200

        # Get most recent job
        most_recent = max(video_jobs.items(), key=lambda x: x[1].get("start_time", 0))
        job_id, job_data = most_recent
        job_data = job_manager.add_timing_info(job_data)
        job_data["processing"] = job_data.get("status") == "running"
        job_data["job_id"] = job_id
        return jsonify(job_data), 200


@video_maker_bp.route("/download", methods=["GET"])
def download_video():
    """Stream video file with range support."""
    try:
        abs_path = Path(request.args.get("path", "")).resolve()
        if not abs_path.is_file() or abs_path.suffix.lower() not in VIDEO_EXTENSIONS:
            return jsonify({"error": "Invalid file"}), 400
        user_home = Path.home()
        cwd = Path.cwd()
        
        # Get configured base paths for data access
        cfg = get_config(refresh=True)
        config_base_paths = [Path(bp).resolve() for bp in cfg.base_paths if Path(bp).exists()]
        
        allowed_roots = [
            user_home,
            cwd,
            Path("/tmp"),
            Path("/var/tmp"),
            Path("/Users"),
            Path("/home"),
        ]
        
        # Add configured base paths to allowed roots
        allowed_roots.extend(config_base_paths)
        
        if os.name == "nt":
            allowed_roots.extend([Path("C:\\Users"), Path("C:\\temp"), Path("C:\\tmp")])
        path_allowed = any(
            allowed_root in abs_path.parents or abs_path == allowed_root
            for allowed_root in allowed_roots
        )
        if not path_allowed:
            logger.warning(f"Attempted download of disallowed path: {abs_path}")
            logger.debug(f"Allowed roots: {allowed_roots}")
            logger.debug(f"File parents: {list(abs_path.parents)}")
            return jsonify({"error": "File not allowed"}), 403
        response = send_file(
            str(abs_path), mimetype="video/mp4", conditional=True, as_attachment=True
        )
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Range")
        return response
    except Exception as e:
        logger.error(f"Error serving video file: {e}")
        return jsonify({"error": f"Error serving file: {str(e)}"}), 500


@video_maker_bp.route("/check_runs", methods=["GET"])
def check_runs():
    """
    Check available runs in the video data for a given camera and data source.
    Returns list of valid runs and the recommended (highest) run.

    Query params:
    - base_path: Base directory path
    - camera: Camera number (1-based)
    - data_source: Data source type (calibrated, uncalibrated, merged, inst_stats)
    - var: Variable to check (ux, uy, mag, u_prime, vorticity, etc.) - defaults to ux
    """
    try:
        base_path_str = request.args.get("base_path")
        camera_raw = request.args.get("camera", "1")
        data_source = request.args.get("data_source", "calibrated")
        var = request.args.get("var", "ux")

        cfg = get_config(refresh=True)

        if not base_path_str:
            if cfg.base_paths:
                base_path_str = cfg.base_paths[0]
            else:
                return jsonify({
                    "success": False,
                    "error": "No base_path provided and no default configured"
                }), 400

        try:
            camera = int(camera_raw)
            if camera < 1:
                raise ValueError("Camera must be positive")
        except ValueError:
            return jsonify({"success": False, "error": "Invalid camera number"}), 400

        base_path = Path(base_path_str).expanduser()
        if not base_path.exists():
            return jsonify({
                "success": False,
                "error": f"Base path does not exist: {base_path}"
            }), 404

        # For stats variables, auto-switch to inst_stats
        STATS_VARIABLES = {"u_prime", "v_prime", "w_prime", "vorticity", "divergence", "gamma1", "gamma2"}
        if var in STATS_VARIABLES:
            data_source = "inst_stats"

        # Get data directory based on data_source
        if data_source == "inst_stats":
            data_dir = base_path / "statistics" / str(cfg.num_frame_pairs) / f"Cam{camera}" / "instantaneous" / "instantaneous_stats"
        else:
            # Determine flags based on data_source
            use_uncalibrated = data_source == "uncalibrated"
            use_merged = data_source == "merged"

            # Get data paths
            paths = get_data_paths(
                base_dir=base_path,
                num_frame_pairs=cfg.num_frame_pairs,
                cam=camera,
                type_name="instantaneous",
                use_uncalibrated=use_uncalibrated,
                use_merged=use_merged,
            )
            data_dir = Path(paths["data_dir"])

        if not data_dir.exists():
            return jsonify({
                "success": False,
                "error": f"Data directory does not exist: {data_dir}",
                "runs": [],
                "highest_run": 1
            }), 404

        # Find first mat file to check runs
        mat_files = sorted(data_dir.glob("[0-9]*.mat"))
        mat_files = [f for f in mat_files if "coordinate" not in f.name.lower()]

        if not mat_files:
            return jsonify({
                "success": False,
                "error": "No .mat files found",
                "runs": [],
                "highest_run": 1
            }), 404

        # Load first file and check valid runs
        first_file = mat_files[0]
        try:
            valid_runs_0based = find_all_valid_runs_from_file(str(first_file), var)
            valid_runs = [r + 1 for r in valid_runs_0based]  # Convert to 1-based
            highest_run = max(valid_runs) if valid_runs else 1
        except Exception as e:
            logger.error(f"Error reading runs from {first_file}: {e}")
            return jsonify({
                "success": False,
                "error": str(e),
                "runs": [1],
                "highest_run": 1
            }), 500

        return jsonify({
            "success": True,
            "runs": valid_runs,
            "highest_run": highest_run,
            "total_runs": len(valid_runs),
            "data_source": data_source,
            "camera": camera,
        })

    except Exception as e:
        logger.exception(f"[VIDEO] Failed to check runs: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

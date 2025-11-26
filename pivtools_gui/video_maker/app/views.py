import os
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request, send_file
from loguru import logger
import numpy as np
from scipy.io import loadmat

from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths
from ..video_maker import PlotSettings, make_video_from_scalar, find_all_valid_runs_from_file, find_highest_valid_run_from_file

video_maker_bp = Blueprint("video_maker", __name__ )

# Constants
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
MAX_DEPTH = 5  # For deep search


def check_ffmpeg_installed() -> Dict[str, Any]:
    """Check if ffmpeg is installed and return version info."""
    result = {
        "installed": False,
        "version": None,
        "path": None,
        "error": None,
    }

    # Try to find ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        result["path"] = ffmpeg_path
        try:
            proc = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0:
                result["installed"] = True
                # Extract version from first line
                first_line = proc.stdout.split("\n")[0]
                result["version"] = first_line
        except subprocess.TimeoutExpired:
            result["error"] = "ffmpeg check timed out"
        except Exception as e:
            result["error"] = str(e)
    else:
        result["error"] = "ffmpeg not found in PATH"

    return result


def check_video_data_availability(
    base_path: Path,
    camera: int,
    num_frame_pairs: int,
    vector_format: str,
) -> Dict[str, Any]:
    """
    Check what data sources are available for video creation.
    Returns availability info for calibrated, uncalibrated, and merged data.
    """
    available = {
        "calibrated": {"exists": False, "frame_count": 0, "path": None},
        "uncalibrated": {"exists": False, "frame_count": 0, "path": None},
        "merged": {"exists": False, "frame_count": 0, "path": None},
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
            frame_count = 0
            for frame in range(1, num_frame_pairs + 1):
                mat_file = cal_data_dir / (vector_format % frame)
                if mat_file.exists():
                    frame_count += 1
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
            frame_count = 0
            for frame in range(1, num_frame_pairs + 1):
                mat_file = uncal_data_dir / (vector_format % frame)
                if mat_file.exists():
                    frame_count += 1
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
            frame_count = 0
            for frame in range(1, num_frame_pairs + 1):
                mat_file = merged_data_dir / (vector_format % frame)
                if mat_file.exists():
                    frame_count += 1
            if frame_count > 0:
                available["merged"]["exists"] = True
                available["merged"]["frame_count"] = frame_count
                available["merged"]["path"] = str(merged_data_dir)
    except Exception as e:
        logger.debug(f"Error checking merged data: {e}")

    return available

# In-memory video job state with thread-safety
_video_state: Dict[str, Any] = {
    "processing": False,
    "progress": 0,
    "message": None,
    "started_at": None,
    "finished_at": None,
    "error": None,
    "meta": None,
    "out_path": None,
    "current_frame": 0,
    "total_frames": 0,
}
_video_thread: Optional[threading.Thread] = None
_video_cancel_event = threading.Event()
_video_state_lock = threading.RLock()  # Reentrant lock for safety


def _video_set_state(**kwargs):
    with _video_state_lock:
        _video_state.update(kwargs)


def _video_reset_state():
    with _video_state_lock:
        _video_state.update(
            {
                "processing": False,
                "progress": 0,
                "message": None,
                "started_at": None,
                "finished_at": None,
                "error": None,
                "meta": None,
                "out_path": None,
                "current_frame": 0,
                "total_frames": 0,
            }
        )


def progress_callback(current_frame: int, total_frames: int, message: str = ""):
    """Thread-safe progress update."""
    _video_set_state(
        progress=int((current_frame / max(total_frames, 1)) * 100),
        current_frame=current_frame,
        total_frames=total_frames,
        message=f"Processing frame {current_frame}/{total_frames}"
        + (f" - {message}" if message else ""),
    )


def _run_video_job(
    base: Path,
    cam: int,
    num_images: int,  # Number of images/files in the folder
    run: int,  # Run number (1-based) for run_index
    source_type: str,
    endpoint: str,
    merged_flag: bool,
    use_uncalibrated: bool,
    var: str,
    pattern: str,
    ps: PlotSettings,
    test_mode: bool = False,
    test_frames: int = 50,
):
    """Optimized job with better error handling."""
    try:
        _video_set_state(
            processing=True,
            progress=0,
            started_at=datetime.utcnow().isoformat(),
            message="Initializing video creation",
            error=None,
            meta=None,
            current_frame=0,
        )

        logger.info(
            f"[VIDEO] Starting video job | base='{base}', cam={cam}, num_images={num_images}, run={run}, var={var}, test_mode={test_mode}, merged={merged_flag}, uncalibrated={use_uncalibrated}"
        )

        cfg = get_config()
        paths = get_data_paths(
            base, cfg.num_frame_pairs, cam, source_type, endpoint,
            use_merged=merged_flag,
            use_uncalibrated=use_uncalibrated,
        )

        data_dir = Path(paths.get("data_dir"))
        video_dir = Path(paths.get("video_dir"))

        video_dir.mkdir(parents=True, exist_ok=True)

        if not Path(ps.out_path).is_absolute():
            ps.out_path = str(video_dir / ps.out_path)

        ps.progress_callback = progress_callback
        ps.test_mode = test_mode
        ps.test_frames = test_frames if test_mode else None

        _video_set_state(message="Starting video generation...")

        meta = make_video_from_scalar(
            data_dir,
            var=var,
            pattern=pattern,
            settings=ps,
            cancel_event=_video_cancel_event,
            run_index=run - 1,  # Convert run (1-based) to run_index (0-based)
        )

        if _video_cancel_event.is_set():
            _video_set_state(
                processing=False,
                progress=0,
                message="Video creation was cancelled",
                finished_at=datetime.utcnow().isoformat(),
                error="Cancelled by user",
            )
            return

        _video_set_state(
            progress=100,
            message="Video completed successfully",
            processing=False,
            finished_at=datetime.utcnow().isoformat(),
            meta=meta,
            out_path=ps.out_path,
            computed_limits={
                "lower": meta.get("vmin"),
                "upper": meta.get("vmax"),
                "actual_min": meta.get("actual_min"),
                "actual_max": meta.get("actual_max"),
                "percentile_based": ps.lower_limit is None or ps.upper_limit is None,
            },
            effective_run=meta.get("effective_run"),  # Return the run that was actually used
        )
        logger.info(f"[VIDEO] Job completed successfully. Output: {ps.out_path}")

    except Exception as e:
        logger.exception(f"[VIDEO] Job failed: {e}")
        _video_set_state(
            processing=False,
            error=str(e),
            message=f"Video creation failed: {str(e)}",
            finished_at=datetime.utcnow().isoformat(),
        )


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


@video_maker_bp.route("/check_ffmpeg", methods=["GET"])
def check_ffmpeg():
    """Check if ffmpeg is installed and available."""
    try:
        result = check_ffmpeg_installed()
        return jsonify({"success": True, **result})
    except Exception as e:
        logger.exception(f"[VIDEO] Failed to check ffmpeg: {e}")
        return jsonify({"success": False, "installed": False, "error": str(e)}), 500


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


@video_maker_bp.route("/start_video", methods=["POST"])
def start_video():
    """
    Start video job with validation.

    Expected JSON parameters:
    - base_path: str - Base directory path for data
    - camera: int - Camera number (1-based)
    - run: int - Run number (1-based)
    - var: str - Variable to visualize ("ux", "uy", "mag")
    - data_source: str - Data source type ("calibrated", "uncalibrated", "merged")
    - fps: int (optional) - Video frame rate (1-120, default: 30)
    - test_mode: bool (optional) - Create test video with limited frames
    - test_frames: int (optional) - Number of frames for test mode (default: 50)
    - lower/upper: float (optional) - Custom color scale limits
    - cmap: str (optional) - Matplotlib colormap name
    - resolution: str (optional) - Video resolution ("4k" or default)
    - out_name: str (optional) - Custom output filename
    """
    global _video_thread

    data = request.get_json(silent=True) or {}
    cfg = get_config(refresh=True)

    # Check ffmpeg first
    ffmpeg_check = check_ffmpeg_installed()
    if not ffmpeg_check["installed"]:
        return jsonify({
            "error": "ffmpeg is not installed. Please install ffmpeg to create videos.",
            "ffmpeg_error": ffmpeg_check.get("error")
        }), 400

    # Validate inputs
    base_path_str = data.get("base_path")
    if not base_path_str:
        return jsonify({"error": "base_path is required"}), 400
    base = Path(base_path_str).expanduser()
    if not base.exists():
        return jsonify({"error": "Invalid base_path"}), 400

    cam_raw = data.get("camera")
    if cam_raw is None:
        return jsonify({"error": "camera is required"}), 400
    try:
        cam = int(cam_raw)
        if cam < 1:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Invalid camera number"}), 400

    test_mode = data.get("test_mode", False)
    if not isinstance(test_mode, bool):
        return jsonify({"error": "test_mode must be boolean"}), 400
    test_frames = int(data.get("test_frames", 50))
    if test_frames < 1:
        return jsonify({"error": "test_frames must be positive"}), 400

    # Parse run as the run number (1-based)
    run_raw = data.get("run")
    if run_raw is None:
        return jsonify({"error": "run is required"}), 400
    try:
        run = int(run_raw)
        if run < 1:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Invalid run number"}), 400

    num_images = int(data.get("num_images", 1))  # Keep for other uses, e.g., if needed elsewhere
    if num_images < 1:
        return jsonify({"error": "num_images must be positive"}), 400

    # Parse data source (new parameter)
    data_source = data.get("data_source", "calibrated")
    if data_source not in ("calibrated", "uncalibrated", "merged"):
        return jsonify({"error": "Invalid data_source. Must be 'calibrated', 'uncalibrated', or 'merged'"}), 400

    # Set flags based on data_source
    use_uncalibrated = data_source == "uncalibrated"
    merged_flag = data_source == "merged"

    # Legacy support: if merged is explicitly set and data_source not provided, use merged flag
    if "merged" in data and "data_source" not in data:
        merged_flag = str(data.get("merged", "0")) in ("1", "true", "True")
        use_uncalibrated = False

    endpoint = data.get("endpoint", "") or ""
    source_type = data.get("type", "instantaneous") or "instantaneous"
    if source_type not in ["instantaneous", "ensemble"]:  # Add allowed types
        return jsonify({"error": "Invalid source_type"}), 400

    # Check if data is available for the selected source
    available = check_video_data_availability(
        base_path=base,
        camera=cam,
        num_frame_pairs=cfg.num_frame_pairs,
        vector_format=cfg.vector_format,
    )

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

    var = data.get("var", None) or data.get("var", "uy")
    if var not in ("ux", "uy", "mag"):
        return jsonify({"error": "Invalid var"}), 400
    pattern = data.get("pattern", "[0-9]*.mat")

    ps = PlotSettings()

    # Parse FPS with validation (frames per second for video output)
    fps = data.get("fps", 30)  # Default to 30 FPS if not provided
    try:
        fps = int(fps)
        if fps < 1 or fps > 120:  # Reasonable range: 1-120 FPS
            return jsonify({"error": "FPS must be between 1 and 120"}), 400
        ps.fps = fps
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid FPS value"}), 400

    ps.crf = 15  # Lower CRF for higher quality (15 is near-lossless)
    ps.upscale = (1080, 1920) if data.get("resolution") != "4k" else (2160, 3840)
    ps.out_path = data.get(
        "out_name",
        f"run{run}_Cam{cam}_{var}{'_test' if test_mode else ''}.mp4",  # Use run for filename
    )

    try:
        lower = data.get("lower")
        upper = data.get("upper")
        ps.lower_limit = float(lower) if lower and str(lower).strip() else None
        ps.upper_limit = float(upper) if upper and str(upper).strip() else None
    except ValueError:
        return jsonify({"error": "Invalid lower/upper limits"}), 400

    cmap = data.get("cmap")
    if cmap and cmap != "default":
        ps.cmap = cmap

    with _video_state_lock:
        running = _video_thread is not None and _video_thread.is_alive()
    if running:
        with _video_state_lock:
            st = {k: _video_state.get(k) for k in ("processing", "progress", "message")}
        return jsonify({"status": "busy", **st}), 409

    _video_cancel_event.clear()
    _video_reset_state()
    _video_set_state(message="Video queued")

    _video_thread = threading.Thread(
        target=_run_video_job,
        args=(
            base,
            cam,
            num_images,  # Pass num_images for folder selection
            run,  # Pass run for run_index
            source_type,
            endpoint,
            merged_flag,
            use_uncalibrated,
            var,
            pattern,
            ps,
            test_mode,
            test_frames,
        ),
        daemon=True,
    )
    _video_thread.start()

    return jsonify({
        "status": "started",
        "processing": True,
        "progress": 0,
        "data_source": data_source,
        "frame_count": available[data_source]["frame_count"],
    }), 202


@video_maker_bp.route("/cancel_video", methods=["POST"])
def cancel_video():
    """Cancel video job safely."""
    _video_cancel_event.set()
    with _video_state_lock:
        is_running = bool(_video_thread is not None and _video_thread.is_alive())
    if is_running:
        _video_set_state(message="Cancellation requested")
        return jsonify({"status": "cancelling", "processing": True}), 202
    _video_reset_state()
    return jsonify({"status": "idle", "processing": False}), 200


@video_maker_bp.route("/video_status", methods=["GET"])
def video_status():
    """Return thread-safe status."""
    with _video_state_lock:
        st = dict(_video_state)
        st["processing"] = bool(
            st.get("processing", False)
            or (_video_thread is not None and _video_thread.is_alive())
        )
    st["progress"] = int(max(0, min(100, int(st.get("progress", 0)))))
    if st.get("out_path"):
        st["out_path"] = st["out_path"]
    elif st.get("meta") and isinstance(st["meta"], dict) and "out_path" in st["meta"]:
        st["out_path"] = st["meta"]["out_path"]
    if st.get("computed_limits"):
        st["computed_limits"] = st["computed_limits"]
    return jsonify(st), 200


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
    - data_source: Data source type (calibrated, uncalibrated, merged)
    - var: Variable to check (ux, uy, mag) - defaults to ux
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

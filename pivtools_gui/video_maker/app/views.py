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

# Excluded coordinate variables (not plottable as fields)
EXCLUDED_VARS = {"x", "y"}

# Label formatting for special variables (matching VectorViewer)
VARIABLE_LABELS = {
    # PIV base variables
    "ux": "ux",
    "uy": "uy",
    "uz": "uz",
    "mag": "Velocity Magnitude",
    "b_mask": "Mask",
    # Instantaneous stats - legacy fluctuations
    "u_prime": "u'",
    "v_prime": "v'",
    "w_prime": "w'",
    # Instantaneous stats - stress tensor components
    "uu_inst": "u'u'",
    "vv_inst": "v'v'",
    "ww_inst": "w'w'",
    "uv_inst": "u'v'",
    "uw_inst": "u'w'",
    "vw_inst": "v'w'",
    # Other computed stats
    "gamma1": "γ₁",
    "gamma2": "γ₂",
    "vorticity": "ω (Vorticity)",
    "divergence": "∇·u (Divergence)",
}


def _extract_plottable_vars(mat_path: Path) -> list:
    """
    Extract plottable 2D variable names from a .mat file.
    Copied from plotting_views.py for consistency.
    """
    if not mat_path.exists():
        return []

    try:
        data_mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
        if "piv_result" not in data_mat:
            return []

        piv_result = data_mat["piv_result"]
        pr = None

        # Handle multiple runs (array of structs)
        if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
            for el in piv_result:
                try:
                    for candidate in ("ux", "uy", "b_mask", "uu", "u_prime", "uu_inst"):
                        val = getattr(el, candidate, None)
                        if val is not None and np.asarray(val).size > 0:
                            pr = el
                            break
                    if pr:
                        break
                except Exception:
                    continue
            if not pr and piv_result.size > 0:
                pr = piv_result.flat[0]
        else:
            pr = piv_result

        if pr is None:
            return []

        # Get all field names
        all_vars = []
        dt = getattr(pr, "dtype", None)
        if dt and getattr(dt, "names", None):
            all_vars = list(dt.names)
        else:
            try:
                if hasattr(pr, "dtype") and getattr(pr.dtype, "names", None):
                    all_vars = list(pr.dtype.names)
                elif hasattr(pr, "dtype") and getattr(pr.dtype, "fields", None):
                    f = pr.dtype.fields
                    if isinstance(f, dict):
                        all_vars = list(f.keys())
            except Exception:
                pass
            if not all_vars:
                try:
                    attrs = [
                        n for n in dir(pr)
                        if not n.startswith("_") and not callable(getattr(pr, n, None))
                    ]
                    all_vars = attrs
                except Exception:
                    all_vars = []

        # Filter to plottable 2D arrays
        plottable_vars = []

        for var_name in all_vars:
            if var_name in EXCLUDED_VARS:
                continue
            try:
                val = getattr(pr, var_name, None)
                if val is None:
                    continue
                arr = np.asarray(val)
                if arr.ndim == 2 and arr.size > 0:
                    plottable_vars.append(var_name)
            except Exception:
                continue

        return plottable_vars
    except Exception:
        return []


def _get_variable_label(var_name: str) -> str:
    """Get display label for a variable, with unicode symbols for special vars."""
    return VARIABLE_LABELS.get(var_name, var_name)


@video_maker_bp.route("/constraints", methods=["GET"])
def get_video_constraints():
    """
    Return constraints for video creation.

    Used by frontend to disable options that are not valid.
    Ensemble data cannot be used for video creation (no temporal sequence).

    Returns:
        JSON with allowed_source_endpoints, ensemble_blocked, ensemble_reason
    """
    cfg = get_config()

    return jsonify({
        "allowed_source_endpoints": cfg.get_allowed_endpoints("video"),
        "ensemble_blocked": True,
        "ensemble_reason": (
            "Ensemble data has no temporal sequence. Videos can only be "
            "created from instantaneous or merged data."
        ),
    })


def check_video_data_availability(
    base_path: Path,
    camera: int,
    num_frame_pairs: int,
    vector_format: str,
    is_stereo: bool = False,
    stereo_camera_pair: Optional[tuple] = None,
) -> Dict[str, Any]:
    """
    Check what data sources are available for video creation.
    Returns availability info for calibrated, uncalibrated, merged, stereo, and inst_stats data.
    """
    available = {
        "calibrated": {"exists": False, "frame_count": 0, "path": None},
        "uncalibrated": {"exists": False, "frame_count": 0, "path": None},
        "merged": {"exists": False, "frame_count": 0, "path": None},
        "stereo": {"exists": False, "frame_count": 0, "path": None, "camera_pair": None},
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

    # Check merged instantaneous (only if not stereo setup)
    if not is_stereo:
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

    # Check stereo instantaneous using FILE-BASED detection
    # (matches VectorViewer behavior - check if stereo_calibrated folder has data)
    try:
        stereo_base_dir = base_path / "stereo_calibrated" / str(num_frame_pairs)
        if stereo_base_dir.exists():
            # Find Cam{n}_Cam{m} folders
            cam_folders = [d for d in stereo_base_dir.iterdir() if d.is_dir() and d.name.startswith("Cam")]
            for cam_folder in cam_folders:
                # Parse camera pair from folder name (e.g., "Cam1_Cam2")
                parts = cam_folder.name.replace("Cam", "").split("_")
                if len(parts) == 2:
                    try:
                        detected_pair = (int(parts[0]), int(parts[1]))
                        # Check for instantaneous data in this folder
                        inst_dir = cam_folder / "instantaneous"
                        if inst_dir.exists():
                            frame_count = len(list(inst_dir.glob("[0-9]*.mat")))
                            if frame_count > 0:
                                available["stereo"]["exists"] = True
                                available["stereo"]["frame_count"] = frame_count
                                available["stereo"]["path"] = str(inst_dir)
                                available["stereo"]["camera_pair"] = list(detected_pair)
                                logger.debug(f"Found stereo data: {inst_dir} with {frame_count} frames")
                                break  # Use first valid stereo folder found
                    except ValueError:
                        continue
    except Exception as e:
        logger.debug(f"Error checking stereo data: {e}")

    # Check instantaneous statistics (camera-specific)
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

    # Check stereo instantaneous statistics (file-based detection)
    if available["stereo"]["exists"] and available["stereo"]["camera_pair"]:
        try:
            cam_pair = available["stereo"]["camera_pair"]
            stereo_stats_dir = base_path / "statistics" / str(num_frame_pairs) / f"Cam{cam_pair[0]}_Cam{cam_pair[1]}" / "instantaneous" / "instantaneous_stats"
            if stereo_stats_dir.exists():
                frame_count = len(list(stereo_stats_dir.glob("[0-9]*.mat")))
                if frame_count > 0:
                    # Override inst_stats with stereo stats if stereo data is primary
                    available["inst_stats"]["exists"] = True
                    available["inst_stats"]["frame_count"] = frame_count
                    available["inst_stats"]["path"] = str(stereo_stats_dir)
                    available["inst_stats"]["is_stereo"] = True
                    logger.debug(f"Found stereo inst_stats: {stereo_stats_dir}")
        except Exception as e:
            logger.debug(f"Error checking stereo inst_stats: {e}")

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

        # Note: check_video_data_availability now uses FILE-BASED stereo detection
        # (no longer relies on cfg.is_stereo_setup for stereo data availability)
        available = check_video_data_availability(
            base_path=base_path,
            camera=camera,
            num_frame_pairs=cfg.num_frame_pairs,
            vector_format=cfg.vector_format,
            is_stereo=cfg.is_stereo_setup,  # Passed for merged check exclusion
            stereo_camera_pair=cfg.stereo_pairs[0] if cfg.stereo_pairs else None,
        )

        # Use file-based detection result for is_stereo flag
        has_stereo_data = available["stereo"]["exists"]

        # Determine default data source (prioritize stereo if detected)
        default_source = None
        if has_stereo_data:
            default_source = "stereo"
        elif available["merged"]["exists"]:
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
            "is_stereo": has_stereo_data,  # File-based detection
            "has_stereo_data": has_stereo_data,  # Explicit flag
        })

    except Exception as e:
        logger.exception(f"[VIDEO] Failed to check data sources: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@video_maker_bp.route("/available_variables", methods=["GET"])
def available_variables():
    """
    Check what variables are available for video creation.
    Returns variables grouped by source, matching VectorViewer's structure.

    Query params:
    - base_path: Base directory path
    - camera: Camera number (1-based)
    - data_source: Data source type (calibrated, uncalibrated, merged)

    Returns:
    - grouped_variables: {instantaneous: [...], instantaneous_stats: [...]}
    - variables: flat list of {name, label, group} for backward compatibility
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

        # Determine flags based on data_source
        use_uncalibrated = data_source == "uncalibrated"
        use_merged = data_source == "merged"
        use_stereo = data_source == "stereo"

        # For stereo, detect camera pair from folder structure
        stereo_camera_pair = None
        if use_stereo:
            stereo_base_dir = base_path / "stereo_calibrated" / str(cfg.num_frame_pairs)
            if stereo_base_dir.exists():
                for cam_folder in stereo_base_dir.iterdir():
                    if cam_folder.is_dir() and cam_folder.name.startswith("Cam"):
                        parts = cam_folder.name.replace("Cam", "").split("_")
                        if len(parts) == 2:
                            try:
                                stereo_camera_pair = (int(parts[0]), int(parts[1]))
                                break
                            except ValueError:
                                continue

        # Get data paths
        paths = get_data_paths(
            base_dir=base_path,
            num_frame_pairs=cfg.num_frame_pairs,
            cam=stereo_camera_pair[0] if stereo_camera_pair else camera,
            type_name="instantaneous",
            use_uncalibrated=use_uncalibrated,
            use_merged=use_merged,
            use_stereo=use_stereo,
            stereo_camera_pair=stereo_camera_pair,
        )

        data_dir = Path(paths["data_dir"])
        has_stereo = use_stereo  # If data_source is stereo, it's stereo data

        # Initialize grouped variables (matching VectorViewer structure)
        grouped_variables = {
            "instantaneous": [],
            "instantaneous_stats": [],
        }

        # Extract instantaneous variables from first PIV frame file
        if data_dir.exists():
            mat_files = sorted(data_dir.glob("[0-9]*.mat"))
            mat_files = [f for f in mat_files if "coordinate" not in f.name.lower()][:1]
            if mat_files:
                inst_vars = _extract_plottable_vars(mat_files[0])
                grouped_variables["instantaneous"] = inst_vars
                has_stereo = "uz" in inst_vars

        # Check for instantaneous statistics (computed per-frame stats)
        # Stats path depends on data source:
        # - Stereo: {base_dir}/statistics/{num_frame_pairs}/Cam{n}_Cam{m}/instantaneous/instantaneous_stats/
        # - Regular: {base_dir}/statistics/{num_frame_pairs}/Cam{camera}/instantaneous/instantaneous_stats/
        if use_stereo and stereo_camera_pair:
            cam_folder = f"Cam{stereo_camera_pair[0]}_Cam{stereo_camera_pair[1]}"
        else:
            cam_folder = f"Cam{camera}"
        stats_base = base_path / "statistics" / str(cfg.num_frame_pairs) / cam_folder / "instantaneous"
        inst_stats_dir = stats_base / "instantaneous_stats"
        has_inst_stats = inst_stats_dir.exists() and any(inst_stats_dir.glob("*.mat"))

        if has_inst_stats:
            inst_stats_files = sorted(inst_stats_dir.glob("*.mat"))[:1]
            if inst_stats_files:
                stats_vars = _extract_plottable_vars(inst_stats_files[0])
                # Filter out base instantaneous vars to avoid duplicates
                base_vars = set(grouped_variables["instantaneous"])
                # Keep stats-specific vars or known computed stats
                known_stats = {
                    "u_prime", "v_prime", "w_prime",
                    "uu_inst", "vv_inst", "ww_inst", "uv_inst", "uw_inst", "vw_inst",
                    "gamma1", "gamma2", "vorticity", "divergence",
                }
                grouped_variables["instantaneous_stats"] = [
                    v for v in stats_vars
                    if v not in base_vars or v in known_stats
                ]

        # Build flat variables list for backward compatibility
        variables = []

        # Add instantaneous variables
        for var_name in grouped_variables["instantaneous"]:
            variables.append({
                "name": var_name,
                "label": _get_variable_label(var_name),
                "group": "piv",
            })

        # Always include mag (velocity magnitude) even if not in file
        if "mag" not in grouped_variables["instantaneous"]:
            variables.append({
                "name": "mag",
                "label": _get_variable_label("mag"),
                "group": "piv",
            })

        # Add instantaneous stats variables
        for var_name in grouped_variables["instantaneous_stats"]:
            variables.append({
                "name": var_name,
                "label": _get_variable_label(var_name),
                "group": "stats",
            })

        return jsonify({
            "success": True,
            "variables": variables,
            "grouped_variables": grouped_variables,
            "has_stereo": has_stereo,
            "has_inst_stats": has_inst_stats,
        })

    except Exception as e:
        logger.exception(f"[VIDEO] Failed to get available variables: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@video_maker_bp.route("/config", methods=["POST"])
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

    # Check ensemble constraint - ensemble data has no temporal sequence
    piv_type = cfg.video_piv_type
    if piv_type == "ensemble":
        return jsonify({
            "error": (
                "Cannot create video from ensemble data. Ensemble averaging "
                "produces a single mean field with no temporal sequence."
            ),
            "constraint_violation": "ensemble_blocked",
        }), 400

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

    is_stereo = cfg.is_stereo_setup
    stereo_camera_pair = cfg.stereo_pairs[0] if cfg.stereo_pairs else None

    data_source = data.get("data_source", cfg.video_data_source)
    valid_sources = ("calibrated", "uncalibrated", "merged", "stereo", "inst_stats")
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
        is_stereo=is_stereo,
        stereo_camera_pair=stereo_camera_pair,
    )

    # For stats variables, auto-switch to inst_stats source
    STATS_VARIABLES = {
        "u_prime", "v_prime", "w_prime",  # Legacy fluctuations
        "uu_inst", "vv_inst", "ww_inst", "uv_inst", "uw_inst", "vw_inst",  # Stress tensor components
        "vorticity", "divergence", "gamma1", "gamma2",  # Derived stats
    }
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

    # Validate variable - allow any plottable variable
    # Instead of a hardcoded list, we allow any variable that was detected as plottable
    # This includes dynamically computed statistics
    VALID_VARIABLES = {
        "ux", "uy", "uz", "mag", "b_mask",  # PIV variables
        "u_prime", "v_prime", "w_prime",  # Legacy fluctuations
        "uu_inst", "vv_inst", "ww_inst", "uv_inst", "uw_inst", "vw_inst",  # Stress tensor components
        "vorticity", "divergence", "gamma1", "gamma2",  # Derived stats
    }
    # Note: We now accept any variable as the backend will try to load it dynamically
    # This allows new computed statistics to work without code changes
    if var not in VALID_VARIABLES:
        logger.info(f"[VIDEO] Variable '{var}' not in known list, will attempt to load dynamically")

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


@video_maker_bp.route("/start_batch", methods=["POST"])
def start_video_batch():
    """
    Start batch video creation.

    Simplified API:
        base_path_idx: int - Single path index (default: 0)
        process_merged: bool - If true: merged only; if false: all cameras
        variable: str (ux, uy, mag, etc.)
        run: int (1-based run number)
        data_source: str (calibrated, uncalibrated - only used when process_merged=false)
        fps: int
        cmap: str
        lower: float/str (color limit)
        upper: float/str (color limit)
        resolution: str (1080p, 4k)
        test_mode: bool
        test_frames: int

    Returns:
        JSON with parent_job_id, sub_jobs list, status
    """
    data = request.get_json() or {}
    logger.info(f"Received batch video request: {data}")

    cfg = get_config(refresh=True)
    base_paths = cfg.base_paths
    is_stereo = cfg.is_stereo_setup
    stereo_camera_pair = cfg.stereo_pairs[0] if cfg.stereo_pairs else None

    base_path_idx = int(data.get("base_path_idx", 0))
    process_merged = bool(data.get("process_merged", False))

    # Validate path index
    if base_path_idx < 0 or base_path_idx >= len(base_paths):
        return jsonify({"error": f"Invalid base_path_idx: {base_path_idx}"}), 400

    base_dir = base_paths[base_path_idx]

    # Get video parameters from request or config
    var = data.get("variable", cfg.video_variable)
    run = data.get("run", cfg.video_run)
    data_source = data.get("data_source", cfg.video_data_source)
    fps = data.get("fps", cfg.video_fps)
    crf = cfg.video_crf
    cmap = data.get("cmap", cfg.video_cmap)
    if cmap == "default":
        cmap = None

    resolution_str = data.get("resolution", cfg.video_resolution_str)
    if resolution_str == "4k":
        resolution = (2160, 3840)
    else:
        resolution = cfg.video_resolution

    lower = data.get("lower", cfg.video_lower_limit)
    upper = data.get("upper", cfg.video_upper_limit)
    try:
        lower_limit = float(lower) if lower and str(lower).strip() else None
        upper_limit = float(upper) if upper and str(upper).strip() else None
    except (ValueError, TypeError):
        lower_limit = None
        upper_limit = None

    test_mode = bool(data.get("test_mode", False))
    test_frames = int(data.get("test_frames", 50))

    # For stats variables, auto-switch to inst_stats source
    STATS_VARIABLES = {
        "u_prime", "v_prime", "w_prime",  # Legacy fluctuations
        "uu_inst", "vv_inst", "ww_inst", "uv_inst", "uw_inst", "vw_inst",  # Stress tensor components
        "vorticity", "divergence", "gamma1", "gamma2",  # Derived stats
    }
    if var in STATS_VARIABLES and data_source != "inst_stats":
        data_source = "inst_stats"
        logger.info(f"[VIDEO] Auto-switching to inst_stats for variable '{var}'")

    try:
        # Build targets based on process_merged flag
        targets = []
        if process_merged:
            # Process merged data only
            targets.append({
                "camera": None,
                "is_merged": True,
                "label": "Merged",
            })
        else:
            # Process all cameras from config.camera_numbers
            for cam in cfg.camera_numbers:
                targets.append({
                    "camera": cam,
                    "is_merged": False,
                    "label": f"Cam{cam}",
                })

        if not targets:
            return jsonify({"error": "No targets to process"}), 400

        # Create parent job to track all sub-jobs
        parent_job_id = job_manager.create_job(
            "video_parent",
            total_targets=len(targets),
        )
        sub_jobs = []

        # Launch a job for each target
        for target in targets:
            use_merged = target["is_merged"]
            cam_num = target["camera"] if target["camera"] else 1

            # Check if data is available for this target
            available = check_video_data_availability(
                base_path=base_dir,
                camera=cam_num,
                num_frame_pairs=cfg.num_frame_pairs,
                vector_format=cfg.vector_format,
                is_stereo=is_stereo,
                stereo_camera_pair=stereo_camera_pair,
            )

            # Determine effective data source for merged targets
            effective_source = "merged" if use_merged else data_source

            if not available.get(effective_source, {}).get("exists", False):
                logger.warning(f"Data not found for {target['label']}, skipping")
                continue

            # Create sub-job
            job_id = job_manager.create_job(
                "video",
                camera=target["label"],
                path_idx=base_path_idx,
                parent_job_id=parent_job_id,
                variable=var,
                data_source=effective_source,
                run=run,
            )
            sub_jobs.append({
                "job_id": job_id,
                "type": "merged" if use_merged else f"camera_{cam_num}",
                "path_idx": base_path_idx,
                "label": target["label"],
            })

            # Create cancel event for this job
            cancel_event = threading.Event()
            with _cancel_events_lock:
                _cancel_events[job_id] = cancel_event

            # Launch thread
            thread = threading.Thread(
                target=_run_video_job,
                args=(
                    job_id,
                    base_dir,
                    cam_num,
                    cfg,
                    var,
                    run,
                    effective_source,
                    fps,
                    crf,
                    resolution,
                    cmap,
                    lower_limit,
                    upper_limit,
                    test_mode,
                    test_frames,
                    cancel_event,
                ),
            )
            thread.daemon = True
            thread.start()

        # Update parent job with sub_jobs list
        job_manager.update_job(parent_job_id, sub_jobs=sub_jobs, status="running")

        return jsonify({
            "parent_job_id": parent_job_id,
            "sub_jobs": sub_jobs,
            "total_targets": len(targets),
            "processed_targets": len(sub_jobs),
            "status": "starting",
            "message": f"Video creation started for {len(sub_jobs)} target(s)",
        })

    except Exception as e:
        logger.error(f"Error starting batch video creation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _run_video_job(
    job_id: str,
    base_dir: Path,
    camera: int,
    cfg,
    var: str,
    run: int,
    data_source: str,
    fps: int,
    crf: int,
    resolution: tuple,
    cmap: Optional[str],
    lower_limit: Optional[float],
    upper_limit: Optional[float],
    test_mode: bool,
    test_frames: int,
    cancel_event: threading.Event,
):
    """Run video creation in a background thread."""
    try:
        cam_folder = "Merged" if data_source == "merged" else f"Cam{camera}"
        logger.info(f"[VIDEO] Starting job {job_id} for {cam_folder}")

        job_manager.update_job(job_id, status="running")

        # Create VideoMaker instance
        maker = VideoMaker(
            base_dir=base_dir,
            camera=camera,
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

        # Run video generation
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
                frames=result.get("frames"),
                elapsed_sec=result.get("elapsed_sec"),
            )
            logger.info(f"[VIDEO] Job {job_id} completed for {cam_folder}")
        else:
            job_manager.fail_job(job_id, result.get("error", "Unknown error"))
            logger.error(f"[VIDEO] Job {job_id} failed: {result.get('error')}")

    except Exception as e:
        logger.error(f"[VIDEO] Job {job_id} error: {e}", exc_info=True)
        job_manager.fail_job(job_id, str(e))
    finally:
        # Clean up cancel event
        with _cancel_events_lock:
            _cancel_events.pop(job_id, None)


@video_maker_bp.route("/batch_status/<job_id>", methods=["GET"])
def get_video_batch_status(job_id):
    """Get batch video job status with aggregated sub-job info."""
    job_data = job_manager.get_job(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404

    # If parent job, aggregate sub-job status
    if "sub_jobs" in job_data:
        sub_job_statuses = []
        all_completed = True
        any_failed = False
        total_progress = 0

        for sub_job in job_data["sub_jobs"]:
            sub_id = sub_job["job_id"]
            sub_status = job_manager.get_job(sub_id)
            if sub_status:
                sub_status["type"] = sub_job["type"]
                sub_status["label"] = sub_job.get("label", "")
                sub_job_statuses.append(sub_status)

                if sub_status["status"] != "completed":
                    all_completed = False
                if sub_status["status"] == "failed":
                    any_failed = True

                total_progress += sub_status.get("progress", 0)

        job_data["sub_job_statuses"] = sub_job_statuses
        job_data["overall_progress"] = total_progress / max(1, len(sub_job_statuses))

        if any_failed:
            job_data["status"] = "failed"
        elif all_completed:
            job_data["status"] = "completed"
        else:
            job_data["status"] = "running"

    # Add timing info
    job_data = job_manager.add_timing_info(job_data)

    return jsonify(job_data)


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


@video_maker_bp.route("/job/<job_id>", methods=["GET"])
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
        STATS_VARIABLES = {
            "u_prime", "v_prime", "w_prime",  # Legacy fluctuations
            "uu_inst", "vv_inst", "ww_inst", "uv_inst", "uw_inst", "vw_inst",  # Stress tensor components
            "vorticity", "divergence", "gamma1", "gamma2",  # Derived stats
        }
        if var in STATS_VARIABLES:
            data_source = "inst_stats"

        # Get data directory based on data_source
        if data_source == "inst_stats":
            data_dir = base_path / "statistics" / str(cfg.num_frame_pairs) / f"Cam{camera}" / "instantaneous" / "instantaneous_stats"
        elif data_source == "stereo":
            # Handle stereo data - find Cam{n}_Cam{m} folder
            stereo_base_dir = base_path / "stereo_calibrated" / str(cfg.num_frame_pairs)
            data_dir = None
            if stereo_base_dir.exists():
                cam_folders = [d for d in stereo_base_dir.iterdir() if d.is_dir() and d.name.startswith("Cam")]
                for cam_folder in cam_folders:
                    inst_dir = cam_folder / "instantaneous"
                    if inst_dir.exists() and list(inst_dir.glob("[0-9]*.mat")):
                        data_dir = inst_dir
                        break
            if data_dir is None:
                return jsonify({
                    "success": False,
                    "error": "No stereo data found",
                    "runs": [],
                    "highest_run": 1
                }), 404
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

"""
Plotting views for PIV data visualization.

Contains endpoints for:
- Vector field plotting (plot_vector, plot_stats, plot_ensemble)
- Data inspection (check_vars, check_limits, check_runs, check_available_data)
- Interactive queries (get_coordinate_at_point, get_vector_at_position, get_stats_value_at_position)
- Uncalibrated data (get_uncalibrated_image)
"""

import random
from pathlib import Path
from typing import Dict

import numpy as np
from flask import Blueprint, jsonify, request
from loguru import logger
from scipy.io import loadmat

from pivtools_core.config import get_config
from pivtools_core.coordinate_utils import extract_coordinates
from pivtools_core.paths import get_data_paths
from pivtools_core.vector_loading import find_non_empty_run, get_plottable_vars

from ..plot_maker import make_scalar_settings
from ...utils import camera_number
from .shared_utils import (
    build_response_meta,
    create_and_return_plot,
    extract_var_and_mask,
    load_and_plot_data,
    load_piv_result,
    parse_plot_params,
    validate_and_get_paths,
)


vector_plot_bp = Blueprint("vector_plot", __name__)


# =============================================================================
# Plotting Endpoints
# =============================================================================


@vector_plot_bp.route("/plot_vector", methods=["GET"])
def plot_vector():
    """Plot vector data from various sources.

    Query params:
        var_source: "inst" (default), "inst_stat", or "mean"
            - inst: frame .mat file
            - inst_stat: instantaneous_stats/NNNNN.mat
            - mean: mean_stats/mean_stats.mat
    """
    try:
        logger.info("plot_vector: received request")
        params = parse_plot_params(request)
        paths = validate_and_get_paths(params)

        data_dir = Path(paths["data_dir"])
        stats_dir = Path(paths["stats_dir"])
        vector_fmt = get_config().vector_format

        # Determine data source
        var_source = request.args.get("var_source", default="inst", type=str)

        if var_source == "mean":
            # Mean statistics - no frame needed
            data_path = stats_dir / "mean_stats" / "mean_stats.mat"
            coords_path = stats_dir / "mean_stats" / "coordinates.mat"
            if not coords_path.exists():
                coords_path = data_dir / "coordinates.mat" if (data_dir / "coordinates.mat").exists() else None
        elif var_source == "inst_stat":
            # Per-frame calculated statistics
            inst_stats_dir = stats_dir / "instantaneous_stats"
            data_path = inst_stats_dir / (vector_fmt % params["frame"])
            coords_path = (
                data_dir / "coordinates.mat"
                if (data_dir / "coordinates.mat").exists()
                else None
            )
        else:
            # Default: instantaneous frame data
            data_path = data_dir / (vector_fmt % params["frame"])
            coords_path = (
                data_dir / "coordinates.mat"
                if (data_dir / "coordinates.mat").exists()
                else None
            )

        b64_img, W, H, extra, effective_run = load_and_plot_data(
            mat_path=data_path,
            coords_path=coords_path,
            var=params["var"],
            run=params["run"],
            save_basepath=Path("plot_vector_tmp"),
            lower_limit=params["lower_limit"],
            upper_limit=params["upper_limit"],
            cmap=params["cmap"],
            raw=params["raw"],
            xlim=params["xlim"],
            ylim=params["ylim"],
            custom_title=params["custom_title"],
        )
        meta = build_response_meta(effective_run, params["var"], W, H, extra)
        return jsonify({"success": True, "image": b64_img, "meta": meta})
    except ValueError as e:
        logger.warning(f"plot_vector: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except FileNotFoundError as e:
        logger.warning(f"plot_vector: file not found: {e}")
        return jsonify({"success": False, "error": f"File not found"}), 404
    except Exception:
        logger.exception("plot_vector: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@vector_plot_bp.route("/plot_stats", methods=["GET"])
def plot_stats():
    """Plot statistics data after running instantaneous_statistics."""
    try:
        params = parse_plot_params(request)
        paths = validate_and_get_paths(params)
        mean_stats_dir = Path(paths["stats_dir"]) / "mean_stats"
        out_file = mean_stats_dir / "mean_stats.mat"
        coords_file = Path(paths["data_dir"]) / "coordinates.mat"
        b64_img, W, H, extra, _ = load_and_plot_data(
            mat_path=out_file,
            coords_path=coords_file if coords_file.exists() else None,
            var=params["var"],
            run=params["run"],
            save_basepath=Path("plot_stats_tmp"),
            lower_limit=params["lower_limit"],
            upper_limit=params["upper_limit"],
            cmap=params["cmap"],
            raw=params["raw"],
            xlim=params["xlim"],
            ylim=params["ylim"],
            custom_title=params["custom_title"],
        )
        meta = build_response_meta(params["run"], params["var"], W, H, extra)
        return jsonify({"success": True, "image": b64_img, "meta": meta})
    except ValueError as e:
        logger.warning(f"plot_stats: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("plot_stats: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@vector_plot_bp.route("/plot_ensemble", methods=["GET"])
def plot_ensemble():
    """Plot ensemble PIV data (single result file with mean fields)."""
    try:
        logger.info("plot_ensemble: received request")
        params = parse_plot_params(request)
        params["type_name"] = "ensemble"  # Override type

        # Strip variable prefix (inst:, mean:, inst_stat:) if present
        # Frontend sends prefixed names like "inst:ux" but ensemble structs use raw names like "ux"
        raw_var = params["var"]
        if ':' in raw_var:
            _, raw_var = raw_var.split(':', 1)

        paths = validate_and_get_paths(params)
        data_dir = Path(paths["data_dir"])

        ensemble_file = data_dir / "ensemble_result.mat"
        coords_path = data_dir / "coordinates.mat" if (data_dir / "coordinates.mat").exists() else None

        if not ensemble_file.exists():
            return jsonify({"success": False, "error": f"Ensemble result not found: {ensemble_file}"}), 404

        mat = loadmat(str(ensemble_file), struct_as_record=False, squeeze_me=True)
        if "ensemble_result" not in mat:
            return jsonify({"success": False, "error": "Variable 'ensemble_result' not found in mat"}), 400

        ensemble_result = mat["ensemble_result"]

        # Use centralized helper from vector_loading (same as instantaneous data)
        pr, effective_run = find_non_empty_run(
            ensemble_result, raw_var, run=params["run"], require_2d=False, reject_all_nan=True
        )

        if pr is None:
            return jsonify({"success": False, "error": f"No valid data found for variable '{raw_var}'"}), 404

        var_arr = np.asarray(getattr(pr, raw_var))
        try:
            mask_arr = np.asarray(getattr(pr, "b_mask")).astype(bool)
        except Exception:
            mask_arr = np.zeros_like(var_arr, dtype=bool)

        cx = cy = None
        if coords_path and coords_path.exists():
            coords_mat = loadmat(str(coords_path), struct_as_record=False, squeeze_me=True)
            if "coordinates" in coords_mat:
                coords = coords_mat["coordinates"]
                cx, cy = extract_coordinates(coords, effective_run)

        settings = make_scalar_settings(
            get_config(),
            variable=raw_var,
            run_label=effective_run,
            save_basepath=Path("plot_ensemble_tmp"),
            variable_units="m/s",
            length_units="mm",
            coords_x=cx,
            coords_y=cy,
            lower_limit=params["lower_limit"],
            upper_limit=params["upper_limit"],
            cmap=params["cmap"],
            xlim=params["xlim"],
            ylim=params["ylim"],
            custom_title=params["custom_title"],
        )

        b64_img, W, H, extra = create_and_return_plot(var_arr, mask_arr, settings, raw=params["raw"])
        meta = build_response_meta(effective_run, raw_var, W, H, extra)

        return jsonify({"success": True, "image": b64_img, "meta": meta})

    except ValueError as e:
        logger.warning(f"plot_ensemble: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except FileNotFoundError as e:
        logger.warning(f"plot_ensemble: file not found: {e}")
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception:
        logger.exception("plot_ensemble: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


# =============================================================================
# Data Inspection Endpoints
# =============================================================================


@vector_plot_bp.route("/check_vars", methods=["GET"])
@vector_plot_bp.route("/check_stat_vars", methods=["GET"])
def check_vars():
    """Inspect a .mat and return available variable names."""
    try:
        frame = request.args.get("frame", default=None, type=int)
        params = parse_plot_params(request)
        paths = validate_and_get_paths(params)
        data_dir = Path(paths["data_dir"])
        mean_stats_dir = Path(paths["stats_dir"]) / "mean_stats"

        if frame is not None:
            vec_fmt = get_config().vector_format
            mat_path = data_dir / (vec_fmt % frame)
        else:
            mat_path = mean_stats_dir / "mean_stats.mat"

        if not mat_path.exists():
            return jsonify({"success": False, "error": f"File not found: {mat_path}"}), 404

        data_mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
        if "piv_result" not in data_mat:
            return jsonify({"success": False, "error": "Variable 'piv_result' not found"}), 400

        piv_result = data_mat["piv_result"]
        pr = None

        if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
            for el in piv_result:
                try:
                    for candidate in ("ux", "uy", "b_mask", "uu"):
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

        # Get all available field names
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
        EXCLUDED_VARS = {"x", "y"}
        plottable_vars = []
        has_peak_choice = hasattr(pr, "peak_choice")

        for var_name in all_vars:
            if var_name in EXCLUDED_VARS:
                continue
            try:
                val = getattr(pr, var_name, None)
                if val is None:
                    continue
                arr = np.asarray(val)
                if arr.ndim == 2:
                    plottable_vars.append(var_name)
                elif var_name == "peak_mag" and arr.ndim == 3 and has_peak_choice:
                    plottable_vars.append(var_name)
            except Exception:
                continue

        return jsonify({"success": True, "vars": plottable_vars})

    except ValueError as e:
        logger.warning(f"check_vars: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("check_vars: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


def _extract_plottable_vars(mat_path: Path) -> list:
    """Extract plottable 2D variable names from a .mat file."""
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
                    for candidate in ("ux", "uy", "b_mask", "uu", "u_prime"):
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
        EXCLUDED_VARS = {"x", "y"}
        plottable_vars = []

        for var_name in all_vars:
            if var_name in EXCLUDED_VARS:
                continue
            try:
                val = getattr(pr, var_name, None)
                if val is None:
                    continue
                arr = np.asarray(val)
                if arr.ndim == 2:
                    plottable_vars.append(var_name)
            except Exception:
                continue

        return plottable_vars
    except Exception:
        return []


@vector_plot_bp.route("/check_all_vars", methods=["GET"])
def check_all_vars():
    """
    Return all available variables grouped by source.

    Returns:
        {
            "instantaneous": ["ux", "uy", ...],      # From frame .mat files
            "instantaneous_stats": ["u_prime", ...], # From instantaneous_stats folder
            "mean_stats": ["ux", "uu", "tke", ...]   # From mean_stats.mat
            "ensemble": ["ux", "uy", "UU_stress", ...] # From ensemble_result.mat
        }
    """
    try:
        frame = request.args.get("frame", default=1, type=int)
        params = parse_plot_params(request)
        paths = validate_and_get_paths(params)

        data_dir = Path(paths["data_dir"])
        stats_dir = Path(paths["stats_dir"])
        mean_stats_dir = stats_dir / "mean_stats"
        inst_stats_dir = stats_dir / "instantaneous_stats"

        result = {
            "instantaneous": [],
            "instantaneous_stats": [],
            "mean_stats": [],
            "ensemble": [],
        }

        # 1. Check instantaneous frame file
        vec_fmt = get_config().vector_format
        frame_path = data_dir / (vec_fmt % frame)
        result["instantaneous"] = get_plottable_vars(frame_path, var_name="piv_result")

        # 2. Check instantaneous_stats folder (per-frame calculated stats)
        if inst_stats_dir.exists():
            # Check first file in inst_stats folder
            inst_stat_files = sorted(inst_stats_dir.glob("*.mat"))
            if inst_stat_files:
                inst_stat_path = inst_stat_files[0]
                inst_vars = get_plottable_vars(inst_stat_path, var_name="piv_result")
                # Filter to only include calculated stats (not duplicates of base vars)
                base_vars = set(result["instantaneous"])
                result["instantaneous_stats"] = [
                    v for v in inst_vars
                    if v not in base_vars or v in ("u_prime", "v_prime", "w_prime",
                                                    "gamma1", "gamma2", "vorticity", "divergence")
                ]

        # 3. Check mean_stats.mat
        mean_stats_path = mean_stats_dir / "mean_stats.mat"
        if mean_stats_path.exists():
            result["mean_stats"] = get_plottable_vars(mean_stats_path, var_name="piv_result")

        # 4. Check ensemble_result.mat
        try:
            ens_paths = get_data_paths(
                base_dir=params["base_path"],
                num_frame_pairs=get_config().num_frame_pairs,
                cam=params["camera"],
                type_name="ensemble",
                use_uncalibrated=params["use_uncalibrated"],
                use_merged=params["use_merged"],
                use_stereo=params.get("use_stereo", False),
                stereo_camera_pair=params.get("stereo_camera_pair"),
            )
            ensemble_file = Path(ens_paths["data_dir"]) / "ensemble_result.mat"
            result["ensemble"] = get_plottable_vars(ensemble_file, var_name="ensemble_result")
        except Exception as e:
            logger.debug(f"check_all_vars: could not check ensemble vars: {e}")

        return jsonify({"success": True, **result})

    except ValueError as e:
        logger.warning(f"check_all_vars: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("check_all_vars: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@vector_plot_bp.route("/check_limits", methods=["GET"])
def check_limits():
    """Sample .mat files to estimate min/max limits for a variable."""
    try:
        params = parse_plot_params(request)
        paths = validate_and_get_paths(params)
        data_dir = Path(paths["data_dir"])

        all_mats = [
            p for p in sorted(data_dir.glob("*.mat"))
            if not any(x in p.name.lower() for x in ["_coordinates", "_mean"])
        ]
        files_total = len(all_mats)

        if files_total == 0:
            return jsonify({"success": False, "error": f"No .mat files found in {data_dir}"}), 404

        sample_count = min(files_total, 50)
        sampled = random.sample(all_mats, sample_count) if files_total > sample_count else all_mats

        all_values = []
        files_checked = 0

        for mat_path in sampled:
            try:
                mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
                if "piv_result" not in mat:
                    continue
                piv_result = mat["piv_result"]
                vals = []
                if isinstance(piv_result, np.ndarray):
                    for el in piv_result:
                        try:
                            arr = np.asarray(getattr(el, params["var"])).ravel()
                            arr = arr[np.isfinite(arr)]
                            if arr.size > 0:
                                vals.append(arr)
                        except Exception:
                            continue
                else:
                    try:
                        arr = np.asarray(getattr(piv_result, params["var"], None)).ravel()
                        arr = arr[np.isfinite(arr)]
                        if arr.size > 0:
                            vals.append(arr)
                    except Exception:
                        pass
                if vals:
                    files_checked += 1
                    all_values.extend(np.concatenate(vals))
            except Exception:
                continue

        if files_checked == 0 or not all_values:
            return jsonify({
                "success": False,
                "error": f"No valid values found for var '{params['var']}'",
            }), 404

        all_values = np.asarray(all_values)
        p5 = float(np.percentile(all_values, 5))
        p95 = float(np.percentile(all_values, 95))
        min_val = float(np.min(all_values))
        max_val = float(np.max(all_values))

        return jsonify({
            "success": True,
            "min": min_val,
            "max": max_val,
            "p5": p5,
            "p95": p95,
            "files_checked": files_checked,
            "files_sampled": len(sampled),
            "files_total": files_total,
            "sampled_files": [p.name for p in sampled],
        })

    except ValueError as e:
        logger.warning(f"check_limits: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("check_limits: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@vector_plot_bp.route("/check_runs", methods=["GET"])
def check_runs():
    """Inspect a .mat file and return available run numbers."""
    try:
        frame = request.args.get("frame", default=1, type=int)
        params = parse_plot_params(request)
        paths = validate_and_get_paths(params)
        data_dir = Path(paths["data_dir"])

        # Handle ensemble mode separately - uses ensemble_result.mat instead of frame files
        is_ensemble = params.get("type_name", "instantaneous") == "ensemble"

        if is_ensemble:
            ensemble_file = data_dir / "ensemble_result.mat"
            if not ensemble_file.exists():
                return jsonify({"success": False, "error": f"Ensemble file not found: {ensemble_file}"}), 404

            mat = loadmat(str(ensemble_file), struct_as_record=False, squeeze_me=True)
            if "ensemble_result" not in mat:
                return jsonify({"success": False, "error": "Variable 'ensemble_result' not found"}), 400

            ensemble_result = mat["ensemble_result"]
            runs = []
            # Use default var for run check (ux is typically available)
            check_var = "ux"

            if isinstance(ensemble_result, np.ndarray) and ensemble_result.dtype == object:
                for i in range(ensemble_result.size):
                    try:
                        pr_candidate = ensemble_result.flat[i]
                        var_arr = np.asarray(getattr(pr_candidate, check_var, None))
                        if var_arr is not None and var_arr.size > 0 and not np.all(np.isnan(var_arr)):
                            runs.append(i + 1)
                    except Exception:
                        continue
            else:
                try:
                    var_arr = np.asarray(getattr(ensemble_result, check_var, None))
                    if var_arr is not None and var_arr.size > 0 and not np.all(np.isnan(var_arr)):
                        runs = [1]
                except Exception:
                    runs = []

            return jsonify({"success": True, "runs": runs})

        # Standard instantaneous mode - check frame files
        vec_fmt = get_config().vector_format
        mat_path = data_dir / (vec_fmt % frame)

        if not mat_path.exists():
            return jsonify({"success": False, "error": f"File not found: {mat_path}"}), 404

        piv_result = load_piv_result(mat_path)
        runs = []

        if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
            for i in range(piv_result.size):
                try:
                    pr_candidate = piv_result.flat[i]
                    var_arr_candidate = np.asarray(getattr(pr_candidate, params["var"], None))
                    if var_arr_candidate is not None and var_arr_candidate.size > 0 and not np.all(np.isnan(var_arr_candidate)):
                        runs.append(i + 1)
                except Exception:
                    continue
        else:
            try:
                var_arr_candidate = np.asarray(getattr(piv_result, params["var"], None))
                if var_arr_candidate is not None and var_arr_candidate.size > 0 and not np.all(np.isnan(var_arr_candidate)):
                    runs = [1]
            except Exception:
                runs = []

        return jsonify({"success": True, "runs": runs})

    except ValueError as e:
        logger.warning(f"check_runs: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("check_runs: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@vector_plot_bp.route("/check_available_data", methods=["GET"])
def check_available_data():
    """
    Check what data sources are available for a given base path and camera.

    Returns availability flags for uncalibrated/calibrated instantaneous and ensemble,
    merged data, and statistics.
    """
    try:
        base_path = request.args.get("base_path", default=None, type=str)
        base_idx = request.args.get("base_path_idx", default=0, type=int)
        cfg = get_config()

        if not base_path:
            try:
                base_path = cfg.base_paths[base_idx]
            except Exception:
                raise ValueError("Invalid base_path and base_path_idx fallback failed")

        camera = camera_number(request.args.get("camera", default=1))
        base_path = Path(base_path)
        num_frame_pairs = cfg.num_frame_pairs
        vector_fmt = cfg.vector_format

        is_stereo = cfg.is_stereo_setup

        available = {
            "uncalibrated_instantaneous": {"exists": False, "frame_count": 0, "variables": []},
            "calibrated_instantaneous": {"exists": False, "frame_count": 0, "variables": []},
            "uncalibrated_ensemble": {"exists": False, "frame_count": 1, "variables": []},
            "calibrated_ensemble": {"exists": False, "frame_count": 1, "variables": []},
            "merged_instantaneous": {"exists": False, "frame_count": 0, "variables": []},
            "merged_ensemble": {"exists": False, "frame_count": 1, "variables": []},
            "statistics": {"exists": False, "variables": []},
            "merged_statistics": {"exists": False, "variables": []},
            "stereo_instantaneous": {"exists": False, "frame_count": 0, "variables": [], "camera_pair": None},
            "stereo_ensemble": {"exists": False, "frame_count": 1, "variables": [], "camera_pair": None},
            "stereo_statistics": {"exists": False, "variables": [], "camera_pair": None},
        }

        def check_directory_for_frames(data_dir: Path, is_ensemble: bool = False, source_name: str = "") -> dict:
            result = {"exists": False, "frame_count": 0, "variables": []}
            logger.debug(f"check_available_data: checking {source_name} at {data_dir}")

            if not data_dir.exists():
                return result

            if is_ensemble:
                ensemble_file = data_dir / "ensemble_result.mat"
                if ensemble_file.exists():
                    result["exists"] = True
                    result["frame_count"] = 1
                    try:
                        mat = loadmat(str(ensemble_file), struct_as_record=False, squeeze_me=True)
                        if "ensemble_result" in mat:
                            er = mat["ensemble_result"]
                            if isinstance(er, np.ndarray) and er.dtype == object and er.size > 0:
                                pr = er.flat[0]
                            else:
                                pr = er
                            attrs = [n for n in dir(pr) if not n.startswith("_") and not callable(getattr(pr, n, None))]
                            result["variables"] = attrs
                    except Exception as e:
                        logger.debug(f"Error reading ensemble vars: {e}")
            else:
                frame_files = []
                for frame in range(1, num_frame_pairs + 1):
                    mat_file = data_dir / (vector_fmt % frame)
                    if mat_file.exists():
                        frame_files.append(mat_file)

                if frame_files:
                    result["exists"] = True
                    result["frame_count"] = len(frame_files)
                    try:
                        mat = loadmat(str(frame_files[0]), struct_as_record=False, squeeze_me=True)
                        if "piv_result" in mat:
                            pr = mat["piv_result"]
                            if isinstance(pr, np.ndarray) and pr.dtype == object and pr.size > 0:
                                pr = pr.flat[0]
                            attrs = [n for n in dir(pr) if not n.startswith("_") and not callable(getattr(pr, n, None))]
                            result["variables"] = attrs
                    except Exception as e:
                        logger.debug(f"Error reading instantaneous vars: {e}")

            return result

        def check_statistics(stats_dir: Path) -> dict:
            result = {"exists": False, "variables": []}
            mean_stats_file = stats_dir / "mean_stats" / "mean_stats.mat"

            if mean_stats_file.exists():
                result["exists"] = True
                try:
                    mat = loadmat(str(mean_stats_file), struct_as_record=False, squeeze_me=True)
                    if "piv_result" in mat:
                        pr = mat["piv_result"]
                        if isinstance(pr, np.ndarray) and pr.dtype == object and pr.size > 0:
                            pr = pr.flat[0]
                        attrs = [n for n in dir(pr) if not n.startswith("_") and not callable(getattr(pr, n, None))]
                        result["variables"] = attrs
                except Exception as e:
                    logger.debug(f"Error reading statistics vars: {e}")

            return result

        # Check all data sources
        uncal_inst_paths = get_data_paths(
            base_dir=base_path, num_frame_pairs=num_frame_pairs, cam=camera,
            type_name="instantaneous", use_uncalibrated=True
        )
        available["uncalibrated_instantaneous"] = check_directory_for_frames(
            Path(uncal_inst_paths["data_dir"]), is_ensemble=False, source_name="uncalibrated_instantaneous"
        )

        uncal_ens_paths = get_data_paths(
            base_dir=base_path, num_frame_pairs=num_frame_pairs, cam=camera,
            type_name="ensemble", use_uncalibrated=True
        )
        available["uncalibrated_ensemble"] = check_directory_for_frames(
            Path(uncal_ens_paths["data_dir"]), is_ensemble=True, source_name="uncalibrated_ensemble"
        )

        # Always check calibrated and merged paths (non-stereo)
        cal_inst_paths = get_data_paths(
            base_dir=base_path, num_frame_pairs=num_frame_pairs, cam=camera,
            type_name="instantaneous", use_uncalibrated=False
        )
        available["calibrated_instantaneous"] = check_directory_for_frames(
            Path(cal_inst_paths["data_dir"]), is_ensemble=False, source_name="calibrated_instantaneous"
        )

        cal_ens_paths = get_data_paths(
            base_dir=base_path, num_frame_pairs=num_frame_pairs, cam=camera,
            type_name="ensemble", use_uncalibrated=False
        )
        available["calibrated_ensemble"] = check_directory_for_frames(
            Path(cal_ens_paths["data_dir"]), is_ensemble=True, source_name="calibrated_ensemble"
        )

        merged_inst_paths = get_data_paths(
            base_dir=base_path, num_frame_pairs=num_frame_pairs, cam=camera,
            type_name="instantaneous", use_merged=True
        )
        available["merged_instantaneous"] = check_directory_for_frames(
            Path(merged_inst_paths["data_dir"]), is_ensemble=False, source_name="merged_instantaneous"
        )

        merged_ens_paths = get_data_paths(
            base_dir=base_path, num_frame_pairs=num_frame_pairs, cam=camera,
            type_name="ensemble", use_merged=True
        )
        available["merged_ensemble"] = check_directory_for_frames(
            Path(merged_ens_paths["data_dir"]), is_ensemble=True, source_name="merged_ensemble"
        )

        available["statistics"] = check_statistics(Path(cal_inst_paths["stats_dir"]))
        available["merged_statistics"] = check_statistics(Path(merged_inst_paths["stats_dir"]))

        # Always check stereo paths (file-based detection)
        # Get stereo camera pair from config, or derive from camera_numbers
        stereo_pairs = cfg.stereo_pairs
        if stereo_pairs:
            cam_pair = stereo_pairs[0]
        elif len(cfg.camera_numbers) >= 2:
            cam_pair = (cfg.camera_numbers[0], cfg.camera_numbers[1])
        else:
            cam_pair = (1, 2)  # Default fallback

        stereo_inst_paths = get_data_paths(
            base_dir=base_path, num_frame_pairs=num_frame_pairs, cam=cam_pair[0],
            type_name="instantaneous", use_stereo=True, stereo_camera_pair=cam_pair
        )
        stereo_result = check_directory_for_frames(
            Path(stereo_inst_paths["data_dir"]), is_ensemble=False, source_name="stereo_instantaneous"
        )
        stereo_result["camera_pair"] = list(cam_pair)
        available["stereo_instantaneous"] = stereo_result

        stereo_ens_paths = get_data_paths(
            base_dir=base_path, num_frame_pairs=num_frame_pairs, cam=cam_pair[0],
            type_name="ensemble", use_stereo=True, stereo_camera_pair=cam_pair
        )
        stereo_ens_result = check_directory_for_frames(
            Path(stereo_ens_paths["data_dir"]), is_ensemble=True, source_name="stereo_ensemble"
        )
        stereo_ens_result["camera_pair"] = list(cam_pair)
        available["stereo_ensemble"] = stereo_ens_result

        # Stereo statistics
        stereo_stats_result = check_statistics(Path(stereo_inst_paths["stats_dir"]))
        stereo_stats_result["camera_pair"] = list(cam_pair)
        available["stereo_statistics"] = stereo_stats_result

        # File-based stereo detection: has_stereo_data is True if stereo data exists
        has_stereo_data = (
            available["stereo_instantaneous"]["exists"] or
            available["stereo_ensemble"]["exists"]
        )

        return jsonify({
            "success": True,
            "camera": camera,
            "base_path": str(base_path),
            "is_stereo": is_stereo,
            "has_stereo_data": has_stereo_data,
            "available": available,
        })

    except ValueError as e:
        logger.warning(f"check_available_data: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("check_available_data: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


# =============================================================================
# Image Endpoints
# =============================================================================


@vector_plot_bp.route("/get_uncalibrated_image", methods=["GET"])
def get_uncalibrated_image():
    """Return a single uncalibrated PNG by index if present."""
    try:
        params = parse_plot_params(request)
        cfg = get_config()
        idx = request.args.get("index", type=int)

        if idx is None:
            raise ValueError("Index parameter required")

        paths = validate_and_get_paths(params)
        data_dir = Path(paths["data_dir"])
        vector_fmt = cfg.vector_format
        mat_path = data_dir / (vector_fmt % idx)

        piv_result = load_piv_result(mat_path)
        max_run = 1
        if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
            max_run = piv_result.size

        params = dict(params)
        params["run"] = max_run

        b64_img, W, H, extra, effective_run = load_and_plot_data(
            mat_path=mat_path,
            coords_path=None,
            var=params["var"],
            run=params["run"],
            save_basepath=Path("plot_vector_tmp"),
            lower_limit=params["lower_limit"],
            upper_limit=params["upper_limit"],
            cmap=params["cmap"],
            variable_units="px/frame",
            length_units="px",
            raw=params["raw"],
            xlim=params["xlim"],
            ylim=params["ylim"],
            custom_title=params["custom_title"],
        )
        meta = build_response_meta(effective_run, params["var"], W, H, extra)
        return jsonify({"success": True, "image": b64_img, "meta": meta})

    except ValueError as e:
        logger.warning(f"get_uncalibrated_image: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except FileNotFoundError as e:
        logger.info(f"get_uncalibrated_image: file not found: {e}")
        return jsonify({"success": False, "error": "File not yet available"}), 404
    except Exception:
        logger.exception("get_uncalibrated_image: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


# =============================================================================
# Interactive Query Endpoints
# =============================================================================


@vector_plot_bp.route("/get_coordinate_at_point", methods=["GET"])
def get_coordinate_at_point():
    """Get the real-world coordinate at a specific point in the image."""
    try:
        base_path = request.args.get("base_path")
        camera = camera_number(request.args.get("camera", "1"))
        x_percent = float(request.args.get("x_percent", 0))
        y_percent = float(request.args.get("y_percent", 0))
        frame = int(request.args.get("frame", 1))

        if not base_path:
            raise ValueError("Base path is required")

        camera_dir = f"Camera_{camera}"
        vector_path = Path(base_path) / camera_dir / f"vec{int(frame):04d}.npz"

        if not vector_path.exists():
            raise ValueError(f"Vector file not found: {vector_path}")

        vector_data = np.load(vector_path, allow_pickle=True)
        x_coords, y_coords = vector_data["x"], vector_data["y"]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        x_coord = x_min + x_percent * (x_max - x_min)
        y_coord = y_min + y_percent * (y_max - y_min)

        return jsonify({
            "success": True,
            "coordinate": {"x": float(x_coord), "y": float(y_coord)}
        })

    except ValueError as e:
        logger.warning(f"get_coordinate_at_point: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("get_coordinate_at_point: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@vector_plot_bp.route("/get_vector_at_position", methods=["GET"])
def get_vector_at_position():
    """Return physical coordinate and values at a given image position."""
    try:
        params = parse_plot_params(request)
        x_percent = float(request.args.get("x_percent"))
        y_percent = float(request.args.get("y_percent"))

        # Determine data source - must be done BEFORE path resolution
        var_source = request.args.get("var_source", default="inst", type=str)

        # Override type_name for ensemble data so path resolution uses ensemble directory
        if var_source == "ens":
            params["type_name"] = "ensemble"

        paths = validate_and_get_paths(params)
        data_dir = Path(paths["data_dir"])
        stats_dir = Path(paths["stats_dir"])
        vec_fmt = get_config().vector_format

        if var_source == "mean":
            mat_path = stats_dir / "mean_stats" / "mean_stats.mat"
        elif var_source == "inst_stat":
            mat_path = stats_dir / "instantaneous_stats" / (vec_fmt % params["frame"])
        elif var_source == "ens":
            mat_path = data_dir / "ensemble_result.mat"
        else:
            mat_path = data_dir / (vec_fmt % params["frame"])

        if not mat_path.exists():
            raise ValueError(f"Vector mat not found: {mat_path}")

        # Handle ensemble-specific loading
        if var_source == "ens":
            mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
            if "ensemble_result" not in mat:
                raise ValueError("Variable 'ensemble_result' not found in mat file")
            piv_result = mat["ensemble_result"]
            # Strip variable prefix if present (e.g., "inst:ux" -> "ux")
            var_name = params["var"]
            if ':' in var_name:
                _, var_name = var_name.split(':', 1)
        else:
            piv_result = load_piv_result(mat_path)
            var_name = params["var"]

        pr, effective_run = find_non_empty_run(piv_result, var_name, params["run"])
        if pr is None:
            raise ValueError("No non-empty run found")

        var_arr = np.asarray(getattr(pr, var_name))
        if var_arr.ndim < 2:
            var_arr = var_arr.reshape(var_arr.shape[0], -1)
        H, W = var_arr.shape

        # Load coordinates - check stats folder for mean, else use data_dir
        if var_source == "mean":
            coords_file = stats_dir / "mean_stats" / "coordinates.mat"
            if not coords_file.exists():
                coords_file = data_dir / "coordinates.mat"
        else:
            coords_file = data_dir / "coordinates.mat"
        cx_arr = cy_arr = None
        physical_coord_used = False

        try:
            if coords_file.exists():
                coords_mat = loadmat(str(coords_file), struct_as_record=False, squeeze_me=True)
                if "coordinates" in coords_mat:
                    coords_struct = coords_mat["coordinates"]
                    cx, cy = extract_coordinates(coords_struct, effective_run)
                    cx_arr, cy_arr = np.asarray(cx), np.asarray(cy)
                    if cx_arr.shape == var_arr.shape:
                        physical_coord_used = True
        except Exception as e:
            logger.debug(f"Coordinates load failed: {e}")

        xlim = params.get("xlim")
        ylim = params.get("ylim")

        if physical_coord_used and cx_arr is not None and cy_arr is not None:
            full_x_min, full_x_max = float(np.nanmin(cx_arr)), float(np.nanmax(cx_arr))
            full_y_min, full_y_max = float(np.nanmin(cy_arr)), float(np.nanmax(cy_arr))

            # Normalize limits to ensure vis_min < vis_max regardless of axis orientation
            if xlim is not None:
                vis_x_min = min(xlim[0], xlim[1])
                vis_x_max = max(xlim[0], xlim[1])
            else:
                vis_x_min, vis_x_max = full_x_min, full_x_max

            if ylim is not None:
                vis_y_min = min(ylim[0], ylim[1])
                vis_y_max = max(ylim[0], ylim[1])
            else:
                vis_y_min, vis_y_max = full_y_min, full_y_max

            target_x = vis_x_min + x_percent * (vis_x_max - vis_x_min)
            target_y = vis_y_max - y_percent * (vis_y_max - vis_y_min)

            dist = np.sqrt((cx_arr - target_x) ** 2 + (cy_arr - target_y) ** 2)
            min_idx = np.unravel_index(np.nanargmin(dist), dist.shape)
            i, j = int(min_idx[0]), int(min_idx[1])
            coord_x, coord_y = float(cx_arr[i, j]), float(cy_arr[i, j])
        else:
            xp = max(0.0, min(1.0, x_percent))
            yp = max(0.0, min(1.0, y_percent))
            j = int(round(xp * (W - 1)))
            i = int(round((1.0 - yp) * (H - 1)))
            i, j = max(0, min(H - 1, i)), max(0, min(W - 1, j))

            x_coords = np.asarray(getattr(pr, "x", None))
            y_coords = np.asarray(getattr(pr, "y", None))
            if x_coords is not None and x_coords.shape == var_arr.shape:
                x_min, x_max = float(np.nanmin(x_coords)), float(np.nanmax(x_coords))
                coord_x = x_min + xp * (x_max - x_min)
            else:
                coord_x = float(j)
            if y_coords is not None and y_coords.shape == var_arr.shape:
                y_min, y_max = float(np.nanmin(y_coords)), float(np.nanmax(y_coords))
                coord_y = y_max - yp * (y_max - y_min)
            else:
                coord_y = float(i)

        ux_arr = np.asarray(getattr(pr, "ux", None))
        uy_arr = np.asarray(getattr(pr, "uy", None))
        ux_val = float(ux_arr[i, j]) if ux_arr is not None and ux_arr.shape == var_arr.shape else None
        uy_val = float(uy_arr[i, j]) if uy_arr is not None and uy_arr.shape == var_arr.shape else None
        value_val = float(var_arr[i, j])

        return jsonify({
            "success": True,
            "x": coord_x,
            "y": coord_y,
            "ux": ux_val,
            "uy": uy_val,
            "value": value_val,
            "i": i,
            "j": j,
        })

    except ValueError as e:
        logger.warning(f"get_vector_at_position: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("get_vector_at_position: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@vector_plot_bp.route("/get_stats_value_at_position", methods=["GET"])
def get_stats_value_at_position():
    """Return values at a position in mean statistics."""
    try:
        params = parse_plot_params(request)
        x_percent = float(request.args.get("x_percent"))
        y_percent = float(request.args.get("y_percent"))

        paths = validate_and_get_paths(params)
        mean_stats_dir = Path(paths["stats_dir"]) / "mean_stats"
        mat_path = mean_stats_dir / "mean_stats.mat"

        if not mat_path.exists():
            raise ValueError(f"Mean stats not found: {mat_path}")

        piv_result = load_piv_result(mat_path)
        pr, effective_run = find_non_empty_run(piv_result, params["var"], params["run"])
        if pr is None:
            raise ValueError("No non-empty run found")

        var_arr = np.asarray(getattr(pr, params["var"]))
        if var_arr.ndim < 2:
            raise ValueError("Unexpected variable array shape")
        H, W = var_arr.shape

        coords_file = mean_stats_dir / "coordinates.mat"
        cx_arr = cy_arr = None
        physical_coord_used = False

        try:
            if coords_file.exists():
                coords_mat = loadmat(str(coords_file), struct_as_record=False, squeeze_me=True)
                if "coordinates" in coords_mat:
                    coords_struct = coords_mat["coordinates"]
                    cx, cy = extract_coordinates(coords_struct, effective_run)
                    cx_arr, cy_arr = np.asarray(cx), np.asarray(cy)
                    if cx_arr.shape == var_arr.shape:
                        physical_coord_used = True
        except Exception as e:
            logger.debug(f"Coordinates load failed: {e}")

        xlim = params.get("xlim")
        ylim = params.get("ylim")

        if physical_coord_used and cx_arr is not None and cy_arr is not None:
            full_x_min, full_x_max = float(np.nanmin(cx_arr)), float(np.nanmax(cx_arr))
            full_y_min, full_y_max = float(np.nanmin(cy_arr)), float(np.nanmax(cy_arr))

            # Normalize limits to ensure vis_min < vis_max regardless of axis orientation
            if xlim is not None:
                vis_x_min = min(xlim[0], xlim[1])
                vis_x_max = max(xlim[0], xlim[1])
            else:
                vis_x_min, vis_x_max = full_x_min, full_x_max

            if ylim is not None:
                vis_y_min = min(ylim[0], ylim[1])
                vis_y_max = max(ylim[0], ylim[1])
            else:
                vis_y_min, vis_y_max = full_y_min, full_y_max

            target_x = vis_x_min + x_percent * (vis_x_max - vis_x_min)
            target_y = vis_y_max - y_percent * (vis_y_max - vis_y_min)

            dist = np.sqrt((cx_arr - target_x) ** 2 + (cy_arr - target_y) ** 2)
            min_idx = np.unravel_index(np.nanargmin(dist), dist.shape)
            i, j = int(min_idx[0]), int(min_idx[1])
            coord_x, coord_y = float(cx_arr[i, j]), float(cy_arr[i, j])
        else:
            xp = max(0.0, min(1.0, x_percent))
            yp = max(0.0, min(1.0, y_percent))
            j = int(round(xp * (W - 1)))
            i = int(round((1.0 - yp) * (H - 1)))
            i, j = max(0, min(H - 1, i)), max(0, min(W - 1, j))

            x_arr = np.asarray(getattr(pr, "x", None))
            y_arr = np.asarray(getattr(pr, "y", None))
            if x_arr is not None and x_arr.shape == var_arr.shape:
                x_min, x_max = float(np.nanmin(x_arr)), float(np.nanmax(x_arr))
                coord_x = x_min + xp * (x_max - x_min)
            else:
                coord_x = float(j)
            if y_arr is not None and y_arr.shape == var_arr.shape:
                y_min, y_max = float(np.nanmin(y_arr)), float(np.nanmax(y_arr))
                coord_y = y_max - yp * (y_max - y_min)
            else:
                coord_y = float(i)

        ux_arr = np.asarray(getattr(pr, "ux", None))
        uy_arr = np.asarray(getattr(pr, "uy", None))
        ux_val = float(ux_arr[i, j]) if ux_arr is not None and ux_arr.shape == var_arr.shape else None
        uy_val = float(uy_arr[i, j]) if uy_arr is not None and uy_arr.shape == var_arr.shape else None
        val = float(var_arr[i, j])

        return jsonify({
            "success": True,
            "x": coord_x,
            "y": coord_y,
            "ux": ux_val,
            "uy": uy_val,
            "value": val,
            "i": i,
            "j": j,
        })

    except ValueError as e:
        logger.warning(f"get_stats_value_at_position: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("get_stats_value_at_position: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500

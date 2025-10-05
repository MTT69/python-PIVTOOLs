import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple
import random
import matplotlib
from loguru import logger
from scipy.io import loadmat

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, jsonify, request

from config import get_config
from paths import get_data_paths
from plotting.plot_maker import make_scalar_settings, plot_scalar_field
from utils import camera_number

vector_plot_bp = Blueprint("vector_plot", __name__, url_prefix="/plot")


def load_piv_result(mat_path: Path) -> np.ndarray:
    """Helper: load a .mat and return its piv_result or raise ValueError with good message."""
    mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    if "piv_result" not in mat:
        raise ValueError(f"Variable 'piv_result' not found in mat: {mat_path}")
    return mat["piv_result"]


def find_non_empty_run(
    piv_result: np.ndarray, var: str, run: int = 1
) -> Tuple[np.ndarray, int]:
    """Find non-empty run in piv_result for variable var"""
    pr = None
    max_runs = 1

    if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
        max_runs = piv_result.size
        current_run = run
        while current_run <= max_runs:
            pr_candidate = piv_result[current_run - 1]
            try:
                var_arr_candidate = np.asarray(getattr(pr_candidate, var))
                if var_arr_candidate.size > 0 and not np.all(
                    np.isnan(var_arr_candidate)
                ):
                    pr = pr_candidate
                    run = current_run
                    break
            except Exception:
                pass
            current_run += 1
    else:
        if run != 1:
            raise ValueError("coordinates contains a single run; use run=1")
        try:
            var_arr_candidate = np.asarray(getattr(piv_result, var))
            if var_arr_candidate.size > 0 and not np.all(np.isnan(var_arr_candidate)):
                pr = piv_result
                run = 1
            else:
                pr = None
        except Exception:
            pr = None

    return pr, run


def extract_coordinates(coords: np.ndarray, run: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract x, y coordinates for the given run"""
    if isinstance(coords, np.ndarray) and coords.dtype == object:
        max_coords_runs = coords.size
        if run < 1 or run > max_coords_runs:
            raise ValueError(f"run out of range for coordinates (1..{max_coords_runs})")
        c_el = coords[run - 1]
        cx, cy = np.asarray(c_el.x), np.asarray(c_el.y)
    else:
        if run != 1:
            raise ValueError("coordinates contains a single run; use run=1")
        c_el = coords
        cx, cy = np.asarray(c_el.x), np.asarray(c_el.y)
    return cx, cy


def extract_var_and_mask(pr: np.ndarray, var: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract variable and mask arrays from piv_result element"""
    try:
        var_arr = np.asarray(getattr(pr, var))
    except Exception:
        raise ValueError(f"'{var}' not found in piv_result element")

    try:
        mask_arr = np.asarray(getattr(pr, "b_mask")).astype(bool)
    except Exception:
        mask_arr = np.zeros_like(var_arr, dtype=bool)

    return var_arr, mask_arr


def create_and_return_plot(
    var_arr: np.ndarray, mask_arr: np.ndarray, settings, raw: bool = False
) -> Tuple[str, int, int, Dict]:
    """
    raw=True -> marginless image (pixel grid == data grid). Always returns extra meta with
    grid_dims (nx=W, ny=H), raw flag, and a simple axes_bbox covering full PNG for legacy.
    """
    H, W = int(var_arr.shape[0]), int(var_arr.shape[1])
    if raw:
        dpi = 100
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        arr = np.asarray(var_arr).squeeze()
        vmin = (
            getattr(settings, "lower_limit", None)
            if hasattr(settings, "lower_limit")
            else None
        )
        vmax = (
            getattr(settings, "upper_limit", None)
            if hasattr(settings, "upper_limit")
            else None
        )
        cmap = getattr(settings, "cmap", None)
        if cmap in (None, "default"):
            cmap = "viridis"
        ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_axis_off()
        buf = BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="white",
        )
        plt.close(fig)
        buf.seek(0)
        from PIL import Image

        with Image.open(buf) as im:
            png_w, png_h = im.size
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        extra = {
            "grid_dims": {"nx": W, "ny": H},
            "raw": True,
            "axes_bbox": {
                "left": 0,
                "top": 0,
                "width": png_w,
                "height": png_h,
                "png_width": png_w,
                "png_height": png_h,
            },
        }
        return b64, W, H, extra

    fig, ax, im = plot_scalar_field(var_arr, mask_arr, settings)
    if im is None or not hasattr(im, "get_array"):
        arr = np.asarray(var_arr).squeeze()
        vmin = (
            getattr(settings, "lower_limit", None)
            if hasattr(settings, "lower_limit")
            else None
        )
        vmax = (
            getattr(settings, "upper_limit", None)
            if hasattr(settings, "upper_limit")
            else None
        )
        cmap = getattr(settings, "cmap", None) if hasattr(settings, "cmap") else None
        if cmap in (None, "default"):
            cmap = "viridis"
        im = ax.imshow(
            arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"
        )
        if hasattr(settings, "title"):
            ax.set_title(settings.title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.canvas.draw()

    fig_width_inches, fig_height_inches = fig.get_size_inches()
    dpi = fig.dpi
    png_width = int(round(fig_width_inches * dpi))
    png_height = int(round(fig_height_inches * dpi))

    ax_extent = ax.get_window_extent()
    axes_left = int(round(ax_extent.x0))
    axes_bottom = int(round(ax_extent.y0))
    axes_width = int(round(ax_extent.width))
    axes_height = int(round(ax_extent.height))

    axes_top = png_height - (axes_bottom + axes_height)

    def clamp(v: float, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    axes_left = clamp(axes_left, 0, png_width)
    axes_top = clamp(axes_top, 0, png_height)
    axes_width = clamp(axes_width, 0, png_width - axes_left)
    axes_height = clamp(axes_height, 0, png_height - axes_top)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    buf.seek(0)
    b64_img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    axes_bbox = {
        "left": axes_left,
        "top": axes_top,
        "width": axes_width,
        "height": axes_height,
        "png_width": png_width,
        "png_height": png_height,
    }
    H, W = int(var_arr.shape[0]), int(var_arr.shape[1])
    return b64_img, W, H, {"axes_bbox": axes_bbox}


def parse_plot_params(req) -> Dict:
    """
    Minimal, explicit parsing. camera expected as int (or string of int).
    Returns a dict with normalized fields.
    """
    base_path = req.args.get("base_path", default=None, type=str)
    base_idx = req.args.get("base_path_idx", default=0, type=int)
    cfg = get_config()
    if not base_path:
        try:
            base_path = cfg.base_paths[base_idx]
        except Exception:
            raise ValueError("Invalid base_path and base_path_idx fallback failed")
    camera = camera_number(req.args.get("camera", default=1))
    merged_raw = req.args.get("merged", default="0", type=str)
    use_merged = merged_raw in ("1", "true", "True", "TRUE")
    is_uncal_raw = req.args.get("is_uncalibrated", default="0", type=str)
    use_uncalibrated = is_uncal_raw in ("1", "true", "True", "TRUE")
    type_name = req.args.get("type_name", default="instantaneous", type=str)
    frame = req.args.get("frame", default=1, type=int)
    run = req.args.get("run", default=1, type=int)
    endpoint = req.args.get("endpoint", default="", type=str)
    var = req.args.get("var", default="ux", type=str)
    lower_limit = req.args.get("lower_limit", type=float)
    upper_limit = req.args.get("upper_limit", type=float)
    cmap = req.args.get("cmap", default=None, type=str)
    if cmap and (cmap.strip() == "" or cmap.lower() == "default"):
        cmap = None
    raw_mode = req.args.get("raw", default="0", type=str) in (
        "1",
        "true",
        "True",
        "TRUE",
    )
    return {
        "base_path": base_path,
        "camera": camera,
        "frame": frame,
        "run": run,
        "endpoint": endpoint,
        "var": var,
        "use_merged": use_merged,
        "use_uncalibrated": use_uncalibrated,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "cmap": cmap,
        "type_name": type_name,
        "raw": raw_mode,
    }


def validate_and_get_paths(params: Dict) -> Dict[str, Path]:
    """Validate parameters and resolve data paths with error handling."""
    try:
        return get_data_paths(
            base_dir=params["base_path"],
            num_images=get_config().num_images,
            cam=params["camera"],
            type_name=params["type_name"],
            endpoint=params["endpoint"],
            use_merged=params["use_merged"],
            use_uncalibrated=params["use_uncalibrated"],
        )
    except Exception as e:
        logger.error(f"Path resolution failed: {e}")
        raise ValueError(f"Failed to resolve paths: {e}")


def load_and_plot_data(
    mat_path: Path,
    coords_path: Optional[Path],
    var: str,
    run: int,
    save_basepath: Path,
    **plot_kwargs,
) -> Tuple[str, int, int, Dict, int]:
    """
    Load data from mat_path, find non-empty run, extract var/mask, load coords if provided,
    build settings, and return plot data.
    """
    piv_result = load_piv_result(mat_path)
    pr, effective_run = find_non_empty_run(piv_result, var, run)
    if pr is None:
        raise ValueError(f"No non-empty run found for variable {var}")
    var_arr, mask_arr = extract_var_and_mask(pr, var)

    cx = cy = None
    if coords_path and coords_path.exists():
        mat = loadmat(str(coords_path), struct_as_record=False, squeeze_me=True)
        if "coordinates" not in mat:
            raise ValueError("Variable 'coordinates' not found in coords mat")
        coords = mat["coordinates"]
        cx, cy = extract_coordinates(coords, effective_run)

    settings = make_scalar_settings(
        get_config(),
        variable=var,
        run_label=effective_run,
        save_basepath=save_basepath,
        variable_units=plot_kwargs.get("variable_units", "m/s"),
        length_units=plot_kwargs.get("length_units", "mm"),
        coords_x=cx,
        coords_y=cy,
        lower_limit=plot_kwargs.get("lower_limit"),
        upper_limit=plot_kwargs.get("upper_limit"),
        cmap=plot_kwargs.get("cmap"),
    )

    b64_img, W, H, extra = create_and_return_plot(
        var_arr, mask_arr, settings, raw=plot_kwargs.get("raw", False)
    )
    return b64_img, W, H, extra, effective_run


def build_response_meta(
    effective_run: int, var: str, width: int, height: int, extra: Optional[Dict] = None
) -> Dict:
    """Build standardized metadata for plot responses."""
    meta = {"run": effective_run, "var": var, "width": width, "height": height}
    if extra:
        meta.update(extra)
    return meta


@vector_plot_bp.route("/plot_vector", methods=["GET"])
def plot_vector():
    """Plot instantaneous vector data."""
    try:
        logger.info("plot_vector: received request")
        params = parse_plot_params(request)
        logger.debug(f"plot_vector: parsed params: {params}")
        paths = validate_and_get_paths(params)
        data_dir = Path(paths["data_dir"])
        vector_fmt = get_config().vector_format
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
        )
        meta = build_response_meta(effective_run, params["var"], W, H, extra)
        return jsonify({"success": True, "image": b64_img, "meta": meta})
    except ValueError as e:
        logger.warning(f"plot_vector: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except FileNotFoundError as e:
        logger.warning(f"plot_vector: file not found: {e}")
        return jsonify({"success": False, "error": "File not found"}), 404
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
        if not out_file.exists():
            from vector_statistics.instantaneous_statistics import (
                instantaneous_statistics,
            )

            instantaneous_statistics(
                cam_num=params["camera"], config=get_config(), base=params["base_path"]
            )
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
        )
        meta = build_response_meta(params["run"], params["var"], W, H, extra)
        return jsonify({"success": True, "image": b64_img, "meta": meta})
    except ValueError as e:
        logger.warning(f"plot_stats: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("plot_stats: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


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
            return (
                jsonify({"success": False, "error": f"File not found: {mat_path}"}),
                404,
            )
        data_mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
        if "piv_result" not in data_mat:
            return (
                jsonify({"success": False, "error": "Variable 'piv_result' not found"}),
                400,
            )
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
        vars_list = []
        dt = getattr(pr, "dtype", None)
        if dt and getattr(dt, "names", None):
            vars_list = list(dt.names)
        else:
            try:
                if hasattr(pr, "dtype") and getattr(pr.dtype, "names", None):
                    vars_list = list(pr.dtype.names)
                elif hasattr(pr, "dtype") and getattr(pr.dtype, "fields", None):
                    f = pr.dtype.fields
                    if isinstance(f, dict):
                        vars_list = list(f.keys())
            except Exception:
                pass
            if not vars_list:
                try:
                    attrs = [
                        n
                        for n in dir(pr)
                        if not n.startswith("_") and not callable(getattr(pr, n, None))
                    ]
                    vars_list = attrs
                except Exception:
                    vars_list = []
        return jsonify({"success": True, "vars": vars_list})
    except ValueError as e:
        logger.warning(f"check_vars: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("check_vars: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@vector_plot_bp.route("/check_limits", methods=["GET"])
def check_limits():
    """Sample .mat files to estimate min/max limits for a variable."""
    try:
        params = parse_plot_params(request)
        paths = validate_and_get_paths(params)
        data_dir = Path(paths["data_dir"])
        all_mats = [
            p
            for p in sorted(data_dir.glob("*.mat"))
            if not any(x in p.name.lower() for x in ["_coordinates", "_mean"])
        ]
        files_total = len(all_mats)
        if files_total == 0:
            return (
                jsonify(
                    {"success": False, "error": f"No .mat files found in {data_dir}"}
                ),
                404,
            )
        sample_count = min(files_total, 50)
        sampled = (
            random.sample(all_mats, sample_count)
            if files_total > sample_count
            else all_mats
        )
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
                        arr = np.asarray(
                            getattr(piv_result, params["var"], None)
                        ).ravel()
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
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"No valid values found for var '{params['var']}'",
                    }
                ),
                404,
            )
        all_values = np.asarray(all_values)
        p5, p95 = float(np.percentile(all_values, 5)), float(
            np.percentile(all_values, 95)
        )
        min_val, max_val = float(np.min(all_values)), float(np.max(all_values))
        return jsonify(
            {
                "success": True,
                "min": min_val,
                "max": max_val,
                "p5": p5,
                "p95": p95,
                "files_checked": files_checked,
                "files_sampled": len(sampled),
                "files_total": files_total,
                "sampled_files": [p.name for p in sampled],
            }
        )
    except ValueError as e:
        logger.warning(f"check_limits: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("check_limits: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


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
        )
        meta = build_response_meta(effective_run, params["var"], W, H, extra)
        return jsonify({"success": True, "image": b64_img, "meta": meta})
    except ValueError as e:
        logger.warning(f"get_uncalibrated_image: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("get_uncalibrated_image: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


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
        return jsonify(
            {"success": True, "coordinate": {"x": float(x_coord), "y": float(y_coord)}}
        )
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
        paths = validate_and_get_paths(params)
        data_dir = Path(paths["data_dir"])
        vec_fmt = get_config().vector_format
        mat_path = data_dir / (vec_fmt % params["frame"])
        if not mat_path.exists():
            raise ValueError(f"Vector mat not found: {mat_path}")
        piv_result = load_piv_result(mat_path)
        pr, effective_run = find_non_empty_run(piv_result, params["var"], params["run"])
        if pr is None:
            raise ValueError("No non-empty run found")
        var_arr = np.asarray(getattr(pr, params["var"]))
        if var_arr.ndim < 2:
            var_arr = var_arr.reshape(var_arr.shape[0], -1)
        H, W = var_arr.shape
        xp = max(0.0, min(1.0, x_percent))
        yp = max(0.0, min(1.0, y_percent))
        j = int(round(xp * (W - 1)))
        i = int(round(yp * (H - 1)))
        i, j = max(0, min(H - 1, i)), max(0, min(W - 1, j))
        physical_coord_used = False
        coord_x = coord_y = None
        try:
            coords_file = data_dir / "coordinates.mat"
            if coords_file.exists():
                coords_mat = loadmat(
                    str(coords_file), struct_as_record=False, squeeze_me=True
                )
                if "coordinates" in coords_mat:
                    coords_struct = coords_mat["coordinates"]
                    cx, cy = extract_coordinates(coords_struct, effective_run)
                    cx_arr, cy_arr = np.asarray(cx), np.asarray(cy)
                    if cx_arr.shape == var_arr.shape:
                        coord_x, coord_y = float(cx_arr[i, j]), float(cy_arr[i, j])
                        physical_coord_used = True
        except Exception as e:
            logger.debug(f"Coordinates load failed: {e}")
        if not physical_coord_used:
            x_coords = np.asarray(getattr(pr, "x", None))
            y_coords = np.asarray(getattr(pr, "y", None))
            if x_coords is not None and x_coords.shape == var_arr.shape:
                x_min, x_max = float(np.nanmin(x_coords)), float(np.nanmax(x_coords))
                coord_x = x_min + xp * (x_max - x_min)
            else:
                coord_x = float(j)
            if y_coords is not None and y_coords.shape == var_arr.shape:
                y_min, y_max = float(np.nanmin(y_coords)), float(np.nanmax(y_coords))
                coord_y = y_min + yp * (y_max - y_min)
            else:
                coord_y = float(i)
        ux_arr = np.asarray(getattr(pr, "ux", None))
        uy_arr = np.asarray(getattr(pr, "uy", None))
        ux_val = (
            float(ux_arr[i, j])
            if ux_arr is not None and ux_arr.shape == var_arr.shape
            else None
        )
        uy_val = (
            float(uy_arr[i, j])
            if uy_arr is not None and uy_arr.shape == var_arr.shape
            else None
        )
        value_val = float(var_arr[i, j]) if var_arr.shape == var_arr.shape else None
        result = {
            "x": coord_x,
            "y": coord_y,
            "ux": ux_val,
            "uy": uy_val,
            "value": value_val,
            "i": i,
            "j": j,
        }
        return jsonify({"success": True, **result})
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
        xp = max(0.0, min(1.0, x_percent))
        yp = max(0.0, min(1.0, y_percent))
        j = int(round(xp * (W - 1)))
        i = int(round(yp * (H - 1)))
        i, j = max(0, min(H - 1, i)), max(0, min(W - 1, j))
        physical_coord_used = False
        coord_x = coord_y = None
        try:
            coords_file = mean_stats_dir / "coordinates.mat"
            if coords_file.exists():
                coords_mat = loadmat(
                    str(coords_file), struct_as_record=False, squeeze_me=True
                )
                if "coordinates" in coords_mat:
                    coords_struct = coords_mat["coordinates"]
                    cx, cy = extract_coordinates(coords_struct, effective_run)
                    cx_arr, cy_arr = np.asarray(cx), np.asarray(cy)
                    if cx_arr.shape == var_arr.shape:
                        coord_x, coord_y = float(cx_arr[i, j]), float(cy_arr[i, j])
                        physical_coord_used = True
        except Exception as e:
            logger.debug(f"Coordinates load failed: {e}")
        if not physical_coord_used:
            x_arr = np.asarray(getattr(pr, "x", None))
            y_arr = np.asarray(getattr(pr, "y", None))
            if x_arr is not None and x_arr.shape == var_arr.shape:
                x_min, x_max = float(np.nanmin(x_arr)), float(np.nanmax(x_arr))
                coord_x = x_min + xp * (x_max - x_min)
            else:
                coord_x = float(j)
            if y_arr is not None and y_arr.shape == var_arr.shape:
                y_min, y_max = float(np.nanmin(y_arr)), float(np.nanmax(y_arr))
                coord_y = y_min + yp * (y_max - y_min)
            else:
                coord_y = float(i)
        ux_arr = np.asarray(getattr(pr, "ux", None))
        uy_arr = np.asarray(getattr(pr, "uy", None))
        ux_val = (
            float(ux_arr[i, j])
            if ux_arr is not None and ux_arr.shape == var_arr.shape
            else None
        )
        uy_val = (
            float(uy_arr[i, j])
            if uy_arr is not None and uy_arr.shape == var_arr.shape
            else None
        )
        val = float(var_arr[i, j]) if var_arr.shape == var_arr.shape else None
        result = {
            "x": coord_x,
            "y": coord_y,
            "ux": ux_val,
            "uy": uy_val,
            "value": val,
            "i": i,
            "j": j,
        }
        return jsonify({"success": True, **result})
    except ValueError as e:
        logger.warning(f"get_stats_value_at_position: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("get_stats_value_at_position: unexpected error")
        return jsonify({"success": False, "error": "Internal server error"}), 500

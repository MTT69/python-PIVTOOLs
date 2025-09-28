import base64
import random
import time
from io import BytesIO
from pathlib import Path

import matplotlib
from loguru import logger

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, jsonify, request
from scipy.io import loadmat

from common.utils import camera_number
from config import get_config
from paths import get_data_paths
from plotting.plot_maker import make_scalar_settings, plot_scalar_field
from vector_statistics.instantaneous_statistics import instantaneous_statistics

vector_plot_bp = Blueprint("vector_plot", __name__, url_prefix="/plot")


def _loadmat_safe(path: Path, max_wait=2.0, poll_interval=0.1, require_stable=True):
    """
    Robust .mat loading to avoid race with file creation.
    Waits until file exists (and optionally size is stable across two polls) or timeout.
    """
    deadline = time.time() + max_wait
    last_size = -1
    stable = False
    while time.time() < deadline:
        if path.exists():
            size = path.stat().st_size
            if not require_stable:
                try:
                    return loadmat(str(path), struct_as_record=False, squeeze_me=True)
                except Exception:
                    pass
            else:
                if size == last_size and size > 0:
                    stable = True
                else:
                    stable = False
                if stable:
                    try:
                        return loadmat(
                            str(path), struct_as_record=False, squeeze_me=True
                        )
                    except Exception:
                        # Possibly still being written; continue until timeout
                        pass
                last_size = size
        time.sleep(poll_interval)
    raise FileNotFoundError(f"Timed out waiting for stable .mat file: {path}")


def load_piv_result(mat_path: Path):
    """Helper: load a .mat and return its piv_result or raise ValueError with good message."""
    try:
        mat = _loadmat_safe(mat_path)
    except FileNotFoundError as e:
        raise ValueError(str(e))
    if "piv_result" not in mat:
        raise ValueError(f"Variable 'piv_result' not found in mat: {mat_path}")
    return mat["piv_result"]


def find_non_empty_run(piv_result, var, run=1):
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
        # Single run; only valid run is 1
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


def extract_coordinates(coords, run):
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


def extract_var_and_mask(pr, var):
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


def safe_get_data_paths(*, base, cam_num, params):
    """
    Support both possible signatures of get_data_paths (cam= or cam_folder=) for compatibility.
    """
    from inspect import signature

    sig = signature(get_data_paths)
    kw = dict(
        base_dir=base,
        num_images=get_config().num_images,
        type_name=params["type_name"],
        endpoint=params["endpoint"],
        use_merged=params["use_merged"],
    )
    # Forward new flag if present
    if "use_uncalibrated" in params:
        kw["use_uncalibrated"] = params["use_uncalibrated"]
    if "cam" in sig.parameters:
        kw["cam"] = cam_num
    else:
        # fallback name
        kw["cam_folder"] = f"Cam{cam_num}" if not params["use_merged"] else "Merged"
    return get_data_paths(**kw)


def create_and_return_plot(var_arr, mask_arr, settings, raw=False):
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

    # fallback to original (non-raw) path
    fig, ax, im = plot_scalar_field(var_arr, mask_arr, settings)
    # Minimal safeguard: ensure we have a valid mappable (first call edge case)
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

    fig.canvas.draw()  # ensure layout

    # --- NEW SIMPLIFIED AXES BBOX LOGIC (replaces old complex/tight bbox code) ---
    # Get figure pixel size
    fig_width_inches, fig_height_inches = fig.get_size_inches()
    dpi = fig.dpi
    png_width = int(round(fig_width_inches * dpi))
    png_height = int(round(fig_height_inches * dpi))

    # Axes extent in display (pixel) coordinates with origin bottom-left
    ax_extent = ax.get_window_extent()
    axes_left = int(round(ax_extent.x0))
    axes_bottom = int(round(ax_extent.y0))
    axes_width = int(round(ax_extent.width))
    axes_height = int(round(ax_extent.height))

    # Convert to top-left origin for frontend
    axes_top = png_height - (axes_bottom + axes_height)

    # Sanity clamp
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    axes_left = clamp(axes_left, 0, png_width)
    axes_top = clamp(axes_top, 0, png_height)
    axes_width = clamp(axes_width, 0, png_width - axes_left)
    axes_height = clamp(axes_height, 0, png_height - axes_top)

    # Save WITHOUT tight bbox so coordinates remain valid
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


def parse_request_params(req, allow_baseidx=True):
    """
    Minimal, explicit parsing. camera expected as int (or string of int).
    Returns a dict with normalized fields.
    """
    # base_path: either explicit or by index into config.base_paths
    base_path = req.args.get("base_path", default=None, type=str)
    base_idx = req.args.get("base_path_idx", default=0, type=int)
    cfg = get_config()
    # Honor explicit base_path if provided; fallback to indexed list only if absent.
    if not base_path:
        try:
            base_path = cfg.base_paths[base_idx]
        except Exception:
            raise ValueError("Invalid base_path and base_path_idx fallback failed")
    # camera: prefer integer; if missing default to 1
    camera = req.args.get("camera", default=1, type=int)
    # merged flag
    merged_raw = req.args.get("merged", default="0", type=str)  # what is this?
    use_merged = merged_raw in ("1", "true", "True", "TRUE")
    # new frontend flag for uncalibrated data
    is_uncal_raw = req.args.get("is_uncalibrated", default="0", type=str)
    use_uncalibrated = is_uncal_raw in ("1", "true", "True", "TRUE")
    # other params
    type_name = req.args.get("type_name", default="instantaneous", type=str)
    frame = req.args.get("frame", default=1, type=int)
    run = req.args.get("run", default=1, type=int)
    endpoint = req.args.get("endpoint", default="", type=str)
    var = req.args.get("var", default="ux", type=str)
    lower_limit = req.args.get("lower_limit", type=float)
    upper_limit = req.args.get("upper_limit", type=float)
    cmap = req.args.get("cmap", default=None, type=str)
    if cmap is not None and (cmap.strip() == "" or cmap.lower() == "default"):
        cmap = None
    raw_mode = req.args.get("raw", default="0", type=str) in (
        "1",
        "true",
        "True",
        "TRUE",
    )
    return dict(
        base_path=base_path,
        camera=camera_number(camera),
        frame=frame,
        run=run,
        endpoint=endpoint,
        var=var,
        use_merged=use_merged,
        use_uncalibrated=use_uncalibrated,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        cmap=cmap,
        type_name=type_name,
        raw=raw_mode,
    )


def plot_from_mat(
    mat_path: Path,
    coords_path: Path,
    var: str,
    run: int,
    save_basepath: Path,
    lower_limit=None,
    upper_limit=None,
    cmap=None,
    raw=False,
    variable_units="m/s",
    length_units="mm",
):
    """
    Load mat_path (expects piv_result), find non-empty run, extract var and mask,
    load coords if provided, build settings and return (b64_img, width, height, effective_run).
    Raises ValueError on missing items.
    """
    piv_result = load_piv_result(mat_path)

    pr, effective_run = find_non_empty_run(piv_result, var, run)
    if pr is None:
        raise ValueError(f"No non-empty run found for variable {var}")

    var_arr, mask_arr = extract_var_and_mask(pr, var)

    # coordinates optional
    cx = cy = None
    if coords_path is not None:
        try:
            coords_mat = _loadmat_safe(coords_path, max_wait=1.0)
        except FileNotFoundError as e:
            raise ValueError(f"coordinates file issue: {e}")
        if "coordinates" not in coords_mat:
            raise ValueError("Variable 'coordinates' not found in coords mat")
        coords = coords_mat["coordinates"]
        cx, cy = extract_coordinates(coords, effective_run)

    settings = make_scalar_settings(
        get_config(),
        variable=var,
        run_label=effective_run,
        save_basepath=save_basepath,
        variable_units=variable_units,
        length_units=length_units,
        coords_x=cx,
        coords_y=cy,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        cmap=cmap,
    )

    b64_img, W, H, extra = create_and_return_plot(var_arr, mask_arr, settings, raw=raw)
    return b64_img, W, H, extra, effective_run


@vector_plot_bp.route("/plot_vector", methods=["GET"])
def plot_vector():
    try:
        logger.debug("plot_vector: received request with args: %s", dict(request.args))
        params = parse_request_params(request)
        logger.debug("plot_vector: parsed params: %s", params)
        get_config()
        base = params["base_path"]
        cam_num = params["camera"]
        try:
            paths = safe_get_data_paths(base=base, cam_num=cam_num, params=params)
        except Exception as e:
            logger.exception("plot_vector: get_data_paths failed")
            return jsonify({"error": f"paths resolution failed: {e}"}), 400
        data_dir = Path(paths["data_dir"])
        vector_fmt = get_config().vector_format
        data_path = data_dir / (vector_fmt % params["frame"])
        coords_path = data_dir / "coordinates.mat"
        try:
            b64_img, W, H, extra, effective_run = plot_from_mat(
                mat_path=data_path,
                coords_path=coords_path if coords_path.exists() else None,
                var=params["var"],
                run=params["run"],
                save_basepath=Path("plot_vector_tmp"),
                lower_limit=params["lower_limit"],
                upper_limit=params["upper_limit"],
                cmap=params["cmap"],
                raw=params["raw"],
            )
        except FileNotFoundError as e:
            logger.warning(f"plot_vector: .mat file not found: {e}")
            return jsonify({"error": f"file not found: {e}"}), 404
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.exception("plot_vector: Exception during plot_from_mat")
            return jsonify({"error": str(e)}), 500
        meta = {"run": effective_run, "var": params["var"], "width": W, "height": H}
        if isinstance(extra, dict):
            meta.update(extra)
        return jsonify({"image": b64_img, "meta": meta})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("plot_vector: Exception")
        return jsonify({"error": str(e)}), 500


@vector_plot_bp.route("/plot_stats", methods=["GET"])
def plot_stats():
    """
    Run instantaneous_statistics if needed, then load the generated mean .mat and coordinates
    and return PNG (base64) of requested scalar.
    """
    params = parse_request_params(request)
    cam_num = params["camera"]
    base = params["base_path"]
    try:
        paths = safe_get_data_paths(base=base, cam_num=cam_num, params=params)
        mean_stats_dir = Path(paths["stats_dir"]) / "mean_stats"
        out_file = mean_stats_dir / "mean_stats.mat"
        coords_file = Path(paths["data_dir"]) / "coordinates.mat"
    except Exception as e:
        return jsonify({"error": f"failed to resolve mean/coords paths: {e}"}), 400

    try:
        instantaneous_statistics(cam_num=cam_num, config=get_config(), base=base)
    except Exception as e:
        return jsonify({"error": f"instantaneous_statistics failed: {e}"}), 500

    try:
        b64_img, W, H, extra, _ = plot_from_mat(
            mat_path=out_file,
            coords_path=coords_file,
            var=params["var"],
            run=params["run"],
            save_basepath=Path("plot_stats_tmp"),
            lower_limit=params["lower_limit"],
            upper_limit=params["upper_limit"],
            cmap=params["cmap"],
            raw=params["raw"],
        )
    except FileNotFoundError as e:
        logger.warning(f"plot_stats: .mat file not found: {e}")
        return jsonify({"error": f"file not found: {e}"}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"plotting failed: {e}"}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"plotting failed: {e}"}), 500


@vector_plot_bp.route("/check_vars", methods=["GET"])
@vector_plot_bp.route("/check_stat_vars", methods=["GET"])  # legacy alias
def check_vars():
    """Inspect a .mat and return available variable names."""
    frame = request.args.get("frame", default=None, type=int)
    params = parse_request_params(request)
    logger.debug("check_vars: request params: %s", dict(request.args))
    # cfg = get_config()
    # Resolve paths using safe_get_data_paths so is_uncalibrated/use_merged are honored
    try:
        paths = safe_get_data_paths(
            base=params["base_path"], cam_num=params["camera"], params=params
        )
        print("Resolved paths:", paths)
    except Exception as e:
        logger.error(f"check_vars: get_data_paths failed, could not resolve path: {e}")
        return jsonify({"error": f"paths resolution failed: {e}"}), 400

    logger.debug("check_vars: resolved paths: %s", paths)
    data_dir = Path(paths["data_dir"])
    mean_stats_dir = Path(paths["stats_dir"]) / "mean_stats"
    # If frame specified -> inspect instantaneous vector .mat
    if frame is not None:
        vec_fmt = get_config().vector_format
        mat_path = Path(data_dir) / (vec_fmt % frame)
    else:
        # Use the aggregated mean_stats.mat in the mean_stats dir
        mat_path = mean_stats_dir / "mean_stats.mat"
    logger.debug("check_vars: mat_path=%s", mat_path)
    # Load and inspect mat_path (use robust loader to avoid races)
    try:
        data_mat = _loadmat_safe(mat_path)
    except FileNotFoundError:
        logger.warning(f"check_vars: .mat file not found: {mat_path}")
        return jsonify({"error": "File not found", "file": str(mat_path)}), 404
    except Exception as e:
        logger.exception("check_vars: failed to load mat %s", mat_path)
        return jsonify({"error": f"failed to load mat: {e}"}), 500

    if "piv_result" not in data_mat:
        return jsonify({"error": "Variable 'piv_result' not found in mat"}), 400
    piv_result = data_mat["piv_result"]

    # Find element to inspect
    pr = None
    try:
        if isinstance(piv_result, np.ndarray):
            # pick first non-empty run if possible
            for el in piv_result:
                try:
                    # Try common field candidates
                    for candidate in ("ux", "uy", "b_mask", "uu"):
                        val = getattr(el, candidate, None)
                        if val is not None and getattr(np.asarray(val), "size", 0) > 0:
                            pr = el
                            break
                    if pr is not None:
                        break
                except Exception:
                    continue
            if pr is None and piv_result.size > 0:
                pr = piv_result.flat[0]
        else:
            pr = piv_result
    except Exception:
        pr = piv_result

    # Get available variables with diagnostics and fallbacks
    vars_list = []
    dt = getattr(pr, "dtype", None)

    # Diagnostics: log types and a short preview of attributes
    try:
        logger.debug(
            "check_vars: piv_result type=%s size=%s",
            type(piv_result),
            getattr(piv_result, "size", None),
        )
        logger.debug("check_vars: pr type=%s", type(pr))
        try:
            pr_repr = repr(pr)
            logger.debug("check_vars: pr repr (truncated)=%s", pr_repr[:1000])
        except Exception:
            pass
        try:
            attrs = [n for n in dir(pr) if not n.startswith("_")]
            logger.debug("check_vars: pr dir preview=%s", attrs[:50])
        except Exception:
            pass
    except Exception:
        logger.exception("check_vars: error while gathering diagnostics for pr")

    # Primary: structured dtype names (numpy structured array)
    if dt is not None and getattr(dt, "names", None):
        vars_list = list(dt.names)
    else:
        # Try numpy structured fields on pr.dtype
        try:
            if hasattr(pr, "dtype") and getattr(pr.dtype, "names", None):
                vars_list = list(pr.dtype.names)
            elif hasattr(pr, "dtype") and getattr(pr.dtype, "fields", None):
                f = pr.dtype.fields
                if isinstance(f, dict):
                    vars_list = list(f.keys())
        except Exception:
            logger.debug("check_vars: exception while inspecting pr.dtype for fields")

        # Try mat_struct-like objects (scipy.io.loadmat sometimes returns simple objects with attributes)
        if not vars_list:
            try:
                if (
                    hasattr(pr, "__dict__")
                    and isinstance(pr.__dict__, dict)
                    and pr.__dict__
                ):
                    vars_list = [k for k in pr.__dict__.keys() if not k.startswith("_")]
            except Exception:
                logger.debug(
                    "check_vars: exception while extracting __dict__ keys from pr"
                )

        # Final fallback: any non-callable attributes from dir(pr)
        if not vars_list:
            try:
                fallback_attrs = [
                    n
                    for n in dir(pr)
                    if not n.startswith("_") and not callable(getattr(pr, n, None))
                ]
                vars_list = fallback_attrs
            except Exception:
                vars_list = []

    logger.debug("check_vars: data_dir=%s", data_dir)
    logger.debug("check_vars: vars_list=%s", vars_list)
    return jsonify({"vars": vars_list})


@vector_plot_bp.route("/check_limits", methods=["GET"])
def check_limits():
    params = parse_request_params(request)
    cfg = get_config()
    base = params["base_path"]
    cam_num = params["camera"]
    try:
        paths = get_data_paths(
            base_dir=base,
            num_images=cfg.num_images,
            cam=cam_num,
            type_name=params["type_name"],
            endpoint=params["endpoint"],
            use_merged=params["use_merged"],
        )
        data_dir = Path(paths["data_dir"])
    except Exception as e:
        return jsonify({"error": f"failed to resolve data_dir: {e}"}), 400

    all_mats = [
        p
        for p in sorted(data_dir.glob("*.mat"))
        if not (
            p.name.lower().endswith("_coordinates.mat")
            or p.name.lower().endswith("_mean.mat")
            or p.name == "coordinates.mat"
        )
    ]
    files_total = len(all_mats)
    if files_total == 0:
        return jsonify({"error": f"No .mat files found in {data_dir}"}), 404

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
                        arr = np.asarray(getattr(el, params["var"]))
                        arr = np.asarray(arr, dtype=float).ravel()
                        arr = arr[np.isfinite(arr)]
                        if arr.size > 0:
                            vals.append(arr)
                    except Exception:
                        continue
            else:
                try:
                    arr = np.asarray(getattr(piv_result, params["var"], None))
                    if arr is not None and arr.size > 0:
                        arr = np.asarray(arr, dtype=float).ravel()
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

    if files_checked == 0 or len(all_values) == 0:
        return (
            jsonify(
                {
                    "error": f"No valid values found for var '{params['var']}' in sampled files"
                }
            ),
            404,
        )

    all_values = np.asarray(all_values)
    p5 = float(np.percentile(all_values, 5))
    p95 = float(np.percentile(all_values, 95))

    # Also provide min/max for frontend compatibility (top-level keys expected by frontend)
    min_val = float(np.min(all_values))
    max_val = float(np.max(all_values))

    return jsonify(
        {
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


@vector_plot_bp.route("/get_uncalibrated_image", methods=["GET"])
def get_uncalibrated_image():
    """Return a single uncalibrated PNG by index if present."""
    # keep compatibility with previous param names
    params = parse_request_params(request)
    cfg = get_config()
    basepath_idx = request.args.get("basepath_idx", default=0, type=int)
    idx = request.args.get("index", type=int)

    # support legacy basepath_idx if user provided it; fallback to parsed base_path
    try:
        base = params["base_path"]
        # if explicit legacy index provided, prefer it
        if request.args.get("basepath_idx") is not None:
            base = cfg.base_paths[basepath_idx]
    except Exception as e:
        logger.debug("get_uncalibrated_image: failed to resolve base: %s", e)
        return jsonify({"error": f"Failed to resolve base: {e}"}), 400

    cam_num = params["camera"]

    try:
        paths = get_data_paths(
            base_dir=base,
            num_images=cfg.num_images,
            cam=cam_num,
            type_name=params["type_name"],
            use_uncalibrated=True,
        )
        data_dir = Path(paths["data_dir"])
    except Exception as e:
        logger.exception("get_uncalibrated_image: Failed to resolve paths")
        return jsonify({"error": f"Failed to resolve paths: {e}"}), 400

    vector_fmt = cfg.vector_format
    name = vector_fmt % idx
    mat_path = data_dir / name
    try:
        b64_img, W, H, extra, effective_run = plot_from_mat(
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
        )
        meta = {"run": effective_run, "var": params["var"], "width": W, "height": H}
        if isinstance(extra, dict):
            meta.update(extra)
        return jsonify({"image": b64_img, "meta": meta})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("get_uncalibrated_image: plotting error")
        return jsonify({"error": str(e)}), 500


@vector_plot_bp.route("/get_coordinate_at_point", methods=["GET"])
def get_coordinate_at_point():
    """Get the real-world coordinate at a specific point in the image."""
    try:
        # Parse parameters
        base_path = request.args.get("base_path")
        camera = request.args.get("camera", "1")
        x_percent = float(request.args.get("x_percent", 0))
        y_percent = float(request.args.get("y_percent", 0))
        frame = request.args.get("frame", "1")

        # Get configuration
        get_config()

        # Validate parameters
        if not base_path:
            return jsonify({"error": "Base path is required"}), 400

        # Load vector data from file
        try:
            # Construct vector file path
            camera_dir = f"Camera_{camera}"
            vector_path = Path(base_path) / camera_dir / f"vec{int(frame):04d}.npz"

            if not vector_path.exists():
                return jsonify({"error": f"Vector file not found: {vector_path}"}), 404

            # Load vector data
            vector_data = np.load(vector_path, allow_pickle=True)
            x_coords = vector_data["x"]
            y_coords = vector_data["y"]

            # Calculate corner coordinates
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)

            # Interpolate the coordinate based on image percentage
            # Previously inverted y; now keep direct mapping (higher percent -> higher physical y)
            x_coord = x_min + x_percent * (x_max - x_min)
            y_coord = y_min + y_percent * (y_max - y_min)
            return jsonify({"coordinate": {"x": float(x_coord), "y": float(y_coord)}})

        except Exception as e:
            logger.exception(f"Error loading vector data: {e}")
            return jsonify({"error": f"Error loading vector data: {str(e)}"}), 500

    except Exception as e:
        logger.exception(f"Error in get_coordinate_at_point: {e}")
        return jsonify({"error": f"Error: {str(e)}"}), 500


@vector_plot_bp.route("/get_vector_at_position", methods=["GET"])
def get_vector_at_position():
    """
    Given x_percent/y_percent in image (0..1), return physical coordinate & values.
    y_percent now maps directly to array index (no vertical inversion).
    """
    try:
        logger.info("get_vector_at_position: request args = %s", dict(request.args))
        params = parse_request_params(request)
        cfg = get_config()
        base = params["base_path"]
        cam_num = params["camera"]
        # frame and percent coordinates
        frame = request.args.get("frame", default=1, type=int)
        try:
            x_percent = float(request.args.get("x_percent", type=float))
            y_percent = float(request.args.get("y_percent", type=float))
            logger.info(
                "get_vector_at_position: x_percent=%f, y_percent=%f",
                x_percent,
                y_percent,
            )
        except Exception as e:
            logger.error("get_vector_at_position: percent parse error: %s", e)
            return jsonify({"error": "x_percent and y_percent required (0..1)"}), 400

        # resolve paths to vector file
        paths = get_data_paths(
            base_dir=base,
            num_images=cfg.num_images,
            cam=cam_num,
            type_name=params["type_name"],
            endpoint=params["endpoint"],
            use_merged=params["use_merged"],
        )
        data_dir = Path(paths["data_dir"])
        vec_fmt = cfg.vector_format
        mat_path = data_dir / (vec_fmt % frame)
        if not mat_path.exists():
            return jsonify({"error": f"vector mat not found: {mat_path}"}), 404

        # load piv_result and pick element/run
        try:
            piv_result = load_piv_result(mat_path)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        pr, effective_run = find_non_empty_run(piv_result, params["var"], params["run"])
        if pr is None:
            return jsonify({"error": "no non-empty run found"}), 404

        # try to extract arrays
        def safe_get_array(obj, name):
            try:
                a = np.asarray(getattr(obj, name))
                return a
            except Exception:
                return None

        # try variable array (for shape)
        var_arr = safe_get_array(pr, params["var"])
        # if var_arr missing, try ux/uy to infer shape
        if var_arr is None:
            var_arr = safe_get_array(pr, "ux") or safe_get_array(pr, "uy")
        if var_arr is None:
            return (
                jsonify(
                    {"error": "no data array found for var or ux/uy in piv_result"}
                ),
                404,
            )

        # ensure 2D shape
        if var_arr.ndim < 2:
            # attempt to reshape or fail
            try:
                var_arr = np.asarray(var_arr).reshape(var_arr.shape[0], -1)
            except Exception:
                return jsonify({"error": "unexpected variable array shape"}), 500

        H, W = int(var_arr.shape[0]), int(var_arr.shape[1])
        # clamp percents to [0,1]
        xp = max(0.0, min(1.0, float(x_percent)))
        yp = max(0.0, min(1.0, float(y_percent)))

        # Map percent -> array indices (direct mapping; 0=top, 1=bottom if display draws that way)
        j = int(round(xp * (W - 1)))
        i = int(round(yp * (H - 1)))  # CHANGED: removed (1.0 - yp) inversion
        i = max(0, min(H - 1, i))
        j = max(0, min(W - 1, j))

        logger.info(
            "get_vector_at_position: Array indices i=%d, j=%d for array shape %s",
            i,
            j,
            var_arr.shape,
        )

        # --- NEW: attempt to load physical coordinates from coordinates.mat ---
        physical_coord_used = False
        try:
            coords_file = data_dir / "coordinates.mat"
            if coords_file.exists():
                coords_mat = _loadmat_safe(coords_file, max_wait=0.5)
                if "coordinates" in coords_mat:
                    coords_struct = coords_mat["coordinates"]
                    cx, cy = extract_coordinates(coords_struct, effective_run)
                    cx_arr = np.asarray(cx)
                    cy_arr = np.asarray(cy)
                    if cx_arr.shape == var_arr.shape and cy_arr.shape == var_arr.shape:
                        coord_x = float(cx_arr[i, j])
                        coord_y = float(cy_arr[i, j])
                        physical_coord_used = True
        except Exception as e:
            logger.debug(
                "get_vector_at_position: coordinates.mat load/parse failed: %s", e
            )

        if not physical_coord_used:
            # existing logic using x/y in piv_result or fallback
            x_coords = safe_get_array(pr, "x")
            y_coords = safe_get_array(pr, "y")
            if (
                x_coords is not None
                and y_coords is not None
                and x_coords.shape == var_arr.shape
                and y_coords.shape == var_arr.shape
            ):
                coord_x = float(x_coords[i, j])
                coord_y = float(y_coords[i, j])
            else:
                try:
                    if x_coords is not None:
                        x_min, x_max = float(np.nanmin(x_coords)), float(
                            np.nanmax(x_coords)
                        )
                        coord_x = float(x_min + xp * (x_max - x_min))
                    else:
                        coord_x = float(j)
                    if y_coords is not None:
                        y_min, y_max = float(np.nanmin(y_coords)), float(
                            np.nanmax(y_coords)
                        )
                        # CHANGED: direct mapping instead of inverted
                        coord_y = float(y_min + yp * (y_max - y_min))
                    else:
                        coord_y = float(i)
                except Exception:
                    coord_x = float(j)
                    coord_y = float(i)

        # try to fetch ux/uy
        ux_arr = safe_get_array(pr, "ux")
        uy_arr = safe_get_array(pr, "uy")
        ux_val = (
            float(ux_arr[i, j])
            if (ux_arr is not None and ux_arr.shape == var_arr.shape)
            else None
        )
        uy_val = (
            float(uy_arr[i, j])
            if (uy_arr is not None and uy_arr.shape == var_arr.shape)
            else None
        )

        # Also provide the requested variable value (could be same as ux)
        value_val = None
        try:
            v = getattr(pr, params["var"], None)
            if v is not None:
                v_arr = np.asarray(v)
                if v_arr.shape == var_arr.shape:
                    value_val = float(v_arr[i, j])
                else:
                    # flatten fallback
                    v_flat = np.asarray(v).ravel()
                    if v_flat.size > 0:
                        value_val = float(v_flat[0])
        except Exception:
            value_val = None

        # Improved logging to debug the issue
        logger.debug(
            f"get_vector_at_position: var_arr shape={var_arr.shape}, indices i={i}, j={j}"
        )

        # Add more logging before returning
        result = {
            "x": coord_x,
            "y": coord_y,
            "ux": ux_val,
            "uy": uy_val,
            "value": value_val,
            "i": int(i),
            "j": int(j),
        }
        logger.info("get_vector_at_position: returning data=%s", result)
        return jsonify(result)
    except Exception as e:
        logger.exception("get_vector_at_position error")
        return jsonify({"error": str(e)}), 500


# @vector_plot_bp.route("/get_vector_at_index", methods=["GET"])
# def get_vector_at_index():
#     """
#     Direct i,j lookup (zero-based) in current frame.
#     Query: base_path, camera, frame, var, run, merged, i, j
#     Returns: x,y,ux,uy,value,i,j (coordinates in same units as stored)
#     """
#     try:
#         params = parse_request_params(request)
#         cfg = get_config()
#         base = params["base_path"]
#         cam = params["camera"]
#         frame = request.args.get("frame", default=1, type=int)
#         i = request.args.get("i", type=int)
#         j = request.args.get("j", type=int)
#         if i is None or j is None:
#             return jsonify({"error": "i and j required"}), 400
#         paths = get_data_paths(
#             base_dir=base,
#             num_images=cfg.num_images,
#             cam=cam,
#             type_name=params["type_name"],
#             endpoint=params["endpoint"],
#             use_merged=params["use_merged"],
#         )
#         data_dir = Path(paths["data_dir"])
#         vec_fmt = cfg.vector_format
#         mat_path = data_dir / (vec_fmt % frame)
#         if not mat_path.exists():
#             return jsonify({"error": f"vector mat not found: {mat_path}"}), 404
#         piv_result = load_piv_result(mat_path)
#         pr, effective_run = find_non_empty_run(piv_result, params["var"], params["run"])
#         if pr is None:
#             return jsonify({"error": "no non-empty run found"}), 404

#         def safe(name):
#             try:
#                 return np.asarray(getattr(pr, name))
#             except Exception:
#                 return None

#         v_arr = safe(params["var"])
#         if v_arr is None:
#             # fallback to ux shape
#             v_arr = safe("ux") or safe("uy")
#         if v_arr is None:
#             return jsonify({"error": "no data array available"}), 500
#         if v_arr.ndim < 2:
#             return jsonify({"error": "data array not 2D"}), 500
#         H, W = v_arr.shape[0], v_arr.shape[1]
#         if i < 0 or j < 0 or i >= H or j >= W:
#             return jsonify({"error": "i,j out of bounds"}), 400
#         x_arr = safe("x")
#         y_arr = safe("y")
#         if (
#             x_arr is not None
#             and y_arr is not None
#             and x_arr.shape == v_arr.shape
#             and y_arr.shape == v_arr.shape
#         ):
#             coord_x = float(x_arr[i, j])
#             coord_y = float(y_arr[i, j])
#         else:
#             coord_x = float(j)
#             coord_y = float(i)
#         ux_arr = safe("ux")
#         uy_arr = safe("uy")
#         ux_val = (
#             float(ux_arr[i, j])
#             if ux_arr is not None and ux_arr.shape == v_arr.shape
#             else None
#         )
#         uy_val = (
#             float(uy_arr[i, j])
#             if uy_arr is not None and uy_arr.shape == v_arr.shape
#             else None
#         )
#         val = None
#         targ = safe(params["var"])
#         if targ is not None and targ.shape == v_arr.shape:
#             val = float(targ[i, j])
#         return jsonify(
#             {
#                 "x": coord_x,
#                 "y": coord_y,
#                 "ux": ux_val,
#                 "uy": uy_val,
#                 "value": val,
#                 "i": int(i),
#                 "j": int(j),
#             }
#         )
#     except Exception as e:
#         logger.exception("get_vector_at_index error")
#         return jsonify({"error": str(e)}), 500


@vector_plot_bp.route("/get_stats_value_at_position", methods=["GET"])
def get_stats_value_at_position():
    """
    Like get_vector_at_position but operates on mean_stats/mean_stats.mat so the frontend
    can query values when displaying the mean statistics (meanMode).
    Query params: base_path, camera, var, run, merged, x_percent, y_percent
    """
    try:
        logger.info("get_stats_value_at_position: args=%s", dict(request.args))
        params = parse_request_params(request)
        x_percent = request.args.get("x_percent", type=float)
        y_percent = request.args.get("y_percent", type=float)
        if x_percent is None or y_percent is None:
            return jsonify({"error": "x_percent and y_percent required"}), 400

        get_config()
        base = params["base_path"]
        cam_num = params["camera"]
        # resolve mean stats paths
        try:
            paths = safe_get_data_paths(base=base, cam_num=cam_num, params=params)
            mean_stats_dir = Path(paths["stats_dir"]) / "mean_stats"
            mat_path = mean_stats_dir / "mean_stats.mat"
        except Exception as e:
            return jsonify({"error": f"failed to resolve mean_stats path: {e}"}), 400
        if not mat_path.exists():
            return jsonify({"error": f"mean stats not found: {mat_path}"}), 404

        try:
            piv_result = load_piv_result(mat_path)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        pr, effective_run = find_non_empty_run(piv_result, params["var"], params["run"])
        if pr is None:
            return jsonify({"error": "no non-empty run found"}), 404

        def safe(name):
            try:
                return np.asarray(getattr(pr, name))
            except Exception:
                return None

        var_arr = safe(params.get("var"))
        if var_arr is None:
            var_arr = safe("ux")
        if var_arr is None:
            var_arr = safe("uy")

        if var_arr is None or var_arr.ndim < 2:
            return jsonify({"error": "no 2D data array available"}), 500
        H, W = var_arr.shape[0], var_arr.shape[1]

        # clamp
        xp = max(0.0, min(1.0, float(x_percent)))
        yp = max(0.0, min(1.0, float(y_percent)))
        j = int(round(xp * (W - 1)))
        i = int(round(yp * (H - 1)))  # CHANGED: removed inversion
        i = max(0, min(H - 1, i))
        j = max(0, min(W - 1, j))

        # --- NEW: attempt to load physical coordinates from mean_stats/coordinates.mat ---
        physical_coord_used = False
        try:
            coords_file = mean_stats_dir / "coordinates.mat"
            if coords_file.exists():
                coords_mat = _loadmat_safe(coords_file, max_wait=0.5)
                if "coordinates" in coords_mat:
                    coords_struct = coords_mat["coordinates"]
                    cx, cy = extract_coordinates(coords_struct, effective_run)
                    cx_arr = np.asarray(cx)
                    cy_arr = np.asarray(cy)
                    if cx_arr.shape == var_arr.shape and cy_arr.shape == var_arr.shape:
                        coord_x = float(cx_arr[i, j])
                        coord_y = float(cy_arr[i, j])
                        physical_coord_used = True
        except Exception as e:
            logger.debug(
                "get_stats_value_at_position: coordinates.mat load/parse failed: %s", e
            )

        if not physical_coord_used:
            x_arr = safe("x")
            y_arr = safe("y")
            if (
                x_arr is not None
                and y_arr is not None
                and x_arr.shape == var_arr.shape
                and y_arr.shape == var_arr.shape
            ):
                coord_x = float(x_arr[i, j])
                coord_y = float(y_arr[i, j])
            else:
                coord_x = float(j)
                coord_y = float(i)

        ux_arr = safe("ux")
        uy_arr = safe("uy")
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
        targ = safe(params["var"])
        val = (
            float(targ[i, j])
            if targ is not None and targ.shape == var_arr.shape
            else None
        )

        return jsonify(
            {
                "x": coord_x,
                "y": coord_y,
                "ux": ux_val,
                "uy": uy_val,
                "value": val,
                "i": int(i),
                "j": int(j),
            }
        )
    except Exception as e:
        logger.exception("get_stats_value_at_position error")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.exception("get_stats_value_at_position error")
        return jsonify({"error": str(e)}), 500

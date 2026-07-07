"""calibration.runio — read/write the production PIV layout for apply.

Reads the uncalibrated PIV output (``coordinates.mat`` + per-frame ``B*.mat``)
written by ``pivtools_cli.piv.save_results`` and writes calibrated output in the
same struct layout, so downstream tooling is unchanged. Crosses the MATLAB/pixel
boundary via ``frames`` (the saved coords are 1-based image-down).

Calibrated coordinates are world mm; calibrated velocities are m/s. No implicit
Y-flips — the sign is carried by the camera model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.io

from pivtools_core.paths import get_data_paths

from . import frames
from . import record as rec
from .apply import calibrate_coordinates, calibrate_displacements, calibrate_stress_tensor
from .camera_model import CameraModel
from .record import MonoRecord, StereoRecord

# Ensemble PIV stores its result under this name + struct (vs per-frame "piv_result").
ENSEMBLE_FILE = "ensemble_result.mat"


def resolve_dt(explicit_dt, model_dt, config_dt) -> float:
    """Resolve the time-between-frames with NO silent default: explicit > model > config.

    Velocity scales linearly with dt, so a wrong dt silently corrupts every vector by a
    constant factor. There is therefore no safe default — if none of the three sources
    supplies dt, this raises instead of falling back to 1.0. ``model_dt`` is the value
    stamped into the model (scale-factor records carry it in ``board_meta``); pass None
    for models that do not stamp it (boards, stereo).
    """
    for cand in (explicit_dt, model_dt, config_dt):
        if cand is not None:
            return float(cand)
    raise ValueError(
        "dt could not be resolved — not given explicitly, not stamped in the model, "
        "and not in config; velocity has no safe default, so set dt")


def plan_apply_units(full_cfg, source, board, stereo, type_name,
                     active_paths=None, camera_pair=None, camera=None, explicit=None,
                     model_type=None):
    """Resolve every (base_path x camera/pair) apply unit, deriving I/O dirs from config.

    Shared by the Flask apply route and the CLI's ``--all-paths`` apply, so both drive the
    same dataset-planning logic. Pure (no Flask / job-manager / threading): it reads config,
    loads the model record per unit, and returns a list of unit dicts:

        mono:   {"stereo": False, "record": MonoRecord,   "uncal": Path,  "out": Path, "label": str}
        stereo: {"stereo": True,  "record": StereoRecord, "uncal1": Path, "uncal2": Path,
                 "out": Path, "label": str}

    ``active_paths`` are indices into ``full_cfg.base_paths`` (None/empty -> all configured).
    ``camera_pair`` is ``(cam1, cam2)`` for stereo (defaults ``[1, 2]``). ``camera`` is the mono
    camera for an explicit single-unit run (defaults ``full_cfg.camera_numbers[0]``). ``explicit``
    forces ONE ad-hoc unit, bypassing config derivation — ``{"uncal", "out"}`` (mono) or
    ``{"uncal1", "uncal2", "out"}`` (stereo). ``model_type`` picks the record when several
    types coexist in the model dir (None -> the single one present, ambiguity raises).
    A missing model fails loudly here (``load_*`` raises), before any job thread starts.
    """
    base_paths = full_cfg.base_paths
    nfp = full_cfg.num_frame_pairs
    base_indices = [int(i) for i in active_paths] if active_paths else list(range(len(base_paths)))

    units = []
    if stereo:
        pair = camera_pair or [1, 2]
        cam1, cam2 = int(pair[0]), int(pair[1])
        record = rec.load_stereo(rec.stereo_model_dir_for_source(source, cam1, cam2),
                                 model_type=model_type)
        if explicit:
            return [dict(stereo=True, record=record,
                         uncal1=Path(explicit["uncal1"]), uncal2=Path(explicit["uncal2"]),
                         out=Path(explicit["out"]), label="manual")]
        for bi in base_indices:
            base = base_paths[bi]
            uncal1 = get_data_paths(base, nfp, cam1, type_name, use_uncalibrated=True)["data_dir"]
            uncal2 = get_data_paths(base, nfp, cam2, type_name, use_uncalibrated=True)["data_dir"]
            out = get_data_paths(base, nfp, cam1, type_name, use_stereo=True,
                                 stereo_camera_pair=(cam1, cam2))["data_dir"]
            units.append(dict(stereo=True, record=record, uncal1=uncal1, uncal2=uncal2,
                              out=out, label=f"{base.name}/Cam{cam1}_Cam{cam2}"))
    else:
        if explicit:
            cam = int(camera) if camera is not None else int(full_cfg.camera_numbers[0])
            record = rec.mono_record_for_camera(rec.mono_model_dir_for_source(source, cam, board),
                                                cam, model_type=model_type)
            return [dict(stereo=False, record=record, uncal=Path(explicit["uncal"]),
                         out=Path(explicit["out"]), label="manual")]
        for bi in base_indices:
            base = base_paths[bi]
            for cam in full_cfg.camera_numbers:
                record = rec.mono_record_for_camera(
                    rec.mono_model_dir_for_source(source, cam, board), cam, model_type=model_type)
                uncal = get_data_paths(base, nfp, cam, type_name, use_uncalibrated=True)["data_dir"]
                out = get_data_paths(base, nfp, cam, type_name)["data_dir"]
                units.append(dict(stereo=False, record=record, uncal=uncal, out=out,
                                  label=f"{base.name}/Cam{cam}"))
    return units


# .mat files in a PIV output dir that are NOT per-frame vector files. A vector glob
# derived from "%05d.mat" is "*.mat", which would otherwise sweep these in.
_NON_VECTOR_MATS = {"coordinates.mat", "mask.mat"}


def _vector_files(uncal_dir: Path, vector_glob: str) -> List[Path]:
    """Per-frame vector files in ``uncal_dir`` matching ``vector_glob``, sorted.

    Excludes non-vector ``.mat`` files (coordinates/mask) and hidden / macOS
    AppleDouble sidecars (``._00001.mat`` on external drives) that a bare ``*.mat``
    glob would otherwise sweep in and choke ``read_vectors``.
    """
    return sorted(
        b for b in Path(uncal_dir).glob(vector_glob)
        if b.name not in _NON_VECTOR_MATS and not b.name.startswith(".")
    )


def _as2d(a) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype == object:
        arr = arr.item()
    return np.asarray(arr, dtype=np.float64)


def read_coordinates(coords_path: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Return {pass_idx: (x_px, y_px)} image-down pixel grids (MATLAB 1-based -> 0-based)."""
    mat = scipy.io.loadmat(str(coords_path), squeeze_me=True)
    coords = mat["coordinates"]
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    n = coords.shape[0] if coords.ndim else 1
    xfield = coords["x"]
    yfield = coords["y"]
    for i in range(n):
        x = _as2d(xfield[i] if n > 1 else xfield)
        y = _as2d(yfield[i] if n > 1 else yfield)
        if x.size == 0:
            continue
        out[i] = (frames.matlab_to_pixel(x), frames.matlab_to_pixel(y))
    return out


def _velocity_fields_from_struct(
    struct, path: Path, struct_name: str
) -> Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """{pass_idx: (ux_px, uy_px, b_mask)} from a per-pass struct array.

    Shared by the per-frame (``piv_result``) and ensemble (``ensemble_result``)
    readers — both store ux/uy/b_mask per pass, empty for unsaved passes.
    """
    out: Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
    n = struct.shape[0] if struct.ndim else 1
    uxf, uyf = struct["ux"], struct["uy"]
    # b_mask is required — every PIV writer emits it (save_results.py). A missing field
    # means a stale pre-b_mask file: fail loudly so it gets re-run, never fake a default
    # (forward-looking, no silent fallback — see CLAUDE.md).
    if "b_mask" not in struct.dtype.names:
        raise ValueError(
            f"{path.name}: {struct_name} has no b_mask field — stale pre-b_mask PIV "
            "output; re-run the PIV pass to regenerate it"
        )
    bmf = struct["b_mask"]
    for i in range(n):
        ux = _as2d(uxf[i] if n > 1 else uxf)
        uy = _as2d(uyf[i] if n > 1 else uyf)
        if ux.size == 0:
            continue
        bb = bmf[i] if n > 1 else bmf
        bb = np.asarray(bb.item() if np.asarray(bb).dtype == object else bb)
        b = bb if bb.size else None
        out[i] = (ux, uy, b)
    return out


def read_vectors(bmat_path: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Return {pass_idx: (ux_px, uy_px, b_mask)} from a per-frame B*.mat."""
    mat = scipy.io.loadmat(str(bmat_path), squeeze_me=True)
    return _velocity_fields_from_struct(mat["piv_result"], bmat_path, "piv_result")


def read_ensemble_vectors(ens_path: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Return {pass_idx: (ux_px, uy_px, b_mask)} from an ensemble_result.mat.

    ``ux/uy`` are the ensemble MEAN displacement per pass (predictor included,
    px/frame, image-down) — directly usable by the stereo reconstruction, which
    is linear in the displacements. Stress fields are ignored here.
    """
    mat = scipy.io.loadmat(str(ens_path), squeeze_me=True)
    return _velocity_fields_from_struct(mat["ensemble_result"], ens_path, "ensemble_result")


def _stress_suffixes(names) -> List[str]:
    """Suffixes for which a full UU/VV/UV triple exists (e.g. '_stress', '_correction').

    Every such triple is a Reynolds-stress-like 2x2 tensor in pixels^2/frame^2, so each
    is calibrated by the same Jacobian transform to stay mutually consistent in m^2/s^2.
    """
    names = set(names or ())
    out = []
    for nm in sorted(names):
        if nm.startswith("UU"):
            suf = nm[2:]
            if ("VV" + suf) in names and ("UV" + suf) in names:
                out.append(suf)
    return out


def calibrate_ensemble_file(
    in_path: Path, out_path: Path, model: CameraModel,
    coords_px: Dict[int, Tuple[np.ndarray, np.ndarray]], dt: float,
    z_world: float = 0.0, tilt_x: float = 0.0, tilt_y: float = 0.0,
) -> None:
    """Calibrate an ``ensemble_result.mat``: mean velocity + every Reynolds-stress triple.

    ``ux/uy`` (mean displacement px/frame) -> m/s via the model; each UU/VV/UV triple
    (px^2/frame^2) -> m^2/s^2 via the tensor transform ``J R J^T / (dt^2 1e6)``. All
    other fields (counts, window centres, masks) pass through untouched. The world
    offset is NOT applied here — it is a position translation; velocity and stress are
    offset-invariant (coordinates carry it, written separately).
    """
    # squeeze_me is deliberately omitted: the struct stays a mutable structured array so
    # `er[field][i] = ...` writes through the reshape(-1) view and persists on savemat.
    mat = scipy.io.loadmat(str(in_path))
    er = mat["ensemble_result"].reshape(-1)   # view; (n_passes,)
    names = er.dtype.names
    suffixes = _stress_suffixes(names)
    has_vel = ("ux" in names) and ("uy" in names)
    for i in range(er.shape[0]):
        if i not in coords_px:
            continue
        xpx, ypx = coords_px[i]
        coords = np.stack([xpx, ypx], axis=-1)
        if has_vel:
            ux = _as2d(er["ux"][i]); uy = _as2d(er["uy"][i])
            if ux.size:
                # A shape mismatch means the ensemble grid does not correspond to
                # coordinates.mat — calibrating it would emit plausible-but-wrong m/s.
                # Fail loudly rather than write the raw px field through uncalibrated.
                if ux.shape != xpx.shape:
                    raise ValueError(
                        f"ensemble pass {i}: ux shape {ux.shape} != coords {xpx.shape}; "
                        f"the ensemble grid does not match coordinates.mat — cannot calibrate")
                u, v = calibrate_displacements(
                    model, coords, np.stack([ux, uy], axis=-1), dt, z_world, tilt_x, tilt_y)
                er["ux"][i] = u
                er["uy"][i] = v
        for suf in suffixes:
            uu = _as2d(er["UU" + suf][i])
            if not uu.size:
                continue
            if uu.shape != xpx.shape:
                raise ValueError(
                    f"ensemble pass {i}: UU{suf} shape {uu.shape} != coords {xpx.shape}; "
                    f"the stress grid does not match coordinates.mat — cannot calibrate")
            vv = _as2d(er["VV" + suf][i])
            uv = _as2d(er["UV" + suf][i])
            cu, cv, cuv = calibrate_stress_tensor(
                model, coords, uu, vv, uv, dt, z_world, tilt_x, tilt_y)
            er["UU" + suf][i] = cu
            er["VV" + suf][i] = cv
            er["UV" + suf][i] = cuv
    scipy.io.savemat(str(out_path), {"ensemble_result": er}, oned_as="row", do_compression=True)


def _coords_struct(per_pass: Dict[int, Tuple[np.ndarray, np.ndarray]], n_passes: int) -> np.ndarray:
    st = np.empty((n_passes,), dtype=[("x", object), ("y", object)])
    empty = np.array([], dtype=np.float64)
    for i in range(n_passes):
        if i in per_pass:
            st["x"][i], st["y"][i] = per_pass[i]
        else:
            st["x"][i] = empty
            st["y"][i] = empty
    return st


def _vectors_struct(per_pass: Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]], n_passes: int) -> np.ndarray:
    st = np.empty((n_passes,), dtype=[("ux", object), ("uy", object), ("b_mask", object)])
    empty = np.array([], dtype=np.float64)
    for i in range(n_passes):
        if i in per_pass:
            ux, uy, b = per_pass[i]
            st["ux"][i], st["uy"][i] = ux, uy
            st["b_mask"][i] = b if b is not None else empty
        else:
            st["ux"][i] = st["uy"][i] = st["b_mask"][i] = empty
    return st


def calibrate_mono_run(
    record: MonoRecord,
    uncal_dir: Path,
    out_dir: Path,
    dt: float,
    z_world: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    vector_glob: str = "B*.mat",
) -> List[str]:
    """Apply a mono model to a whole run: coordinates.mat + every vector .mat.

    Reads ``<uncal_dir>/coordinates.mat`` and ``<uncal_dir>/<vector_glob>``; writes
    calibrated world-mm coordinates and m/s velocities to ``<out_dir>``. ``vector_glob``
    matches the per-frame vector files (default ``B*.mat``; PIV's ``vector_fmt`` may
    differ, e.g. ``[0-9]*.mat``). ``progress_cb(done, total)`` runs after each frame.
    """
    uncal_dir = Path(uncal_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = record.camera_model
    # Per-camera placement into the shared multi-camera rig frame (None == this camera
    # is its own frame). Added to coordinates only; velocities are offset-invariant.
    offset_mm = getattr(record.world_frame, "world_offset_mm", None)

    coords_px = read_coordinates(uncal_dir / "coordinates.mat")
    n_passes = (max(coords_px) + 1) if coords_px else 1

    # Calibrated coordinates (world mm), per pass.
    cal_coords: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for i, (xpx, ypx) in coords_px.items():
        stacked = np.stack([xpx, ypx], axis=-1)
        world = calibrate_coordinates(model, stacked, z_world, tilt_x, tilt_y, offset_mm=offset_mm)
        cal_coords[i] = (world[..., 0], world[..., 1])
    scipy.io.savemat(
        str(out_dir / "coordinates.mat"),
        {"coordinates": _coords_struct(cal_coords, n_passes)},
        oned_as="row", do_compression=True,
    )

    written: List[str] = []
    bmats = _vector_files(uncal_dir, vector_glob)
    total = len(bmats)
    for k, bmat in enumerate(bmats):
        out_b = out_dir / bmat.name
        if bmat.name == ENSEMBLE_FILE:
            # Ensemble: mean velocity + Reynolds-stress tensors (different struct/fields).
            calibrate_ensemble_file(bmat, out_b, model, coords_px, dt, z_world, tilt_x, tilt_y)
            written.append(str(out_b))
            if progress_cb:
                progress_cb(k + 1, total)
            continue
        vecs = read_vectors(bmat)
        cal_vecs: Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
        for i, (ux, uy, b) in vecs.items():
            if i not in coords_px:
                continue
            xpx, ypx = coords_px[i]
            coords_stack = np.stack([xpx, ypx], axis=-1)
            disp_stack = np.stack([ux, uy], axis=-1)
            u, v = calibrate_displacements(model, coords_stack, disp_stack, dt, z_world, tilt_x, tilt_y)
            cal_vecs[i] = (u, v, b)
        scipy.io.savemat(
            str(out_b),
            {"piv_result": _vectors_struct(cal_vecs, n_passes)},
            oned_as="row", do_compression=True,
        )
        written.append(str(out_b))
        if progress_cb:
            progress_cb(k + 1, total)
    return written


def reconstruct_stereo_ensemble(
    record: StereoRecord,
    ens1_path: Path,
    ens2_path: Path,
    coords1: Dict[int, Tuple[np.ndarray, np.ndarray]],
    coords2: Dict[int, Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    dt: float,
    z_world: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    interpolator: str = "lanczos",
    pass_index: Optional[int] = None,
) -> List[str]:
    """Reconstruct both cameras' ensemble MEAN fields to stereo (U,V,W) m/s.

    Mean flow only: the 4x3 Jacobian solve is linear in the displacements, so
    the ensemble mean displacement reconstructs to the exact mean 3C velocity.
    Reynolds stresses do NOT reconstruct linearly and are deliberately absent
    from the output struct (``ux/uy/uz/b_mask`` only) — reconstructing them
    needs the correlation-of-correlations method, out of scope here.

    Every pass populated in BOTH cameras (and both coordinates.mat) is
    reconstructed on its own regular world grid — window size/overlap change
    the pixel grid, hence the world spacing, per pass (mirrors mono
    ``calibrate_ensemble_file`` doing all passes). ``pass_index`` restricts to
    one pass. Writes per-pass world ``coordinates.mat`` + ``ensemble_result.mat``.
    """
    from .stereo_model import reconstruct_3c_field, regular_world_grid

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v1 = read_ensemble_vectors(Path(ens1_path))
    v2 = read_ensemble_vectors(Path(ens2_path))
    passes = sorted(set(v1) & set(v2) & set(coords1) & set(coords2))
    if not passes:
        raise RuntimeError(
            "no common populated pass between the two cameras' ensemble "
            "results — re-run ensemble PIV for both cameras with the same runs")
    # Struct length spans all common passes so pass indices stay aligned with
    # the source files even when pass_index restricts the reconstruction.
    n_passes = max(passes) + 1
    if pass_index is not None:
        if pass_index not in passes:
            raise ValueError(
                f"pass_index {pass_index} is not populated in both cameras' "
                f"ensemble results (common populated passes: {passes})")
        passes = [pass_index]
    empty = np.array([], dtype=np.float64)
    st = np.empty(
        (n_passes,),
        dtype=[("ux", object), ("uy", object), ("uz", object), ("b_mask", object)],
    )
    for i in range(n_passes):
        st["ux"][i] = st["uy"][i] = st["uz"][i] = st["b_mask"][i] = empty
    world_coords: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for p in passes:
        x1, y1 = coords1[p]
        x2, y2 = coords2[p]
        ux1, uy1, b1 = v1[p]
        ux2, uy2, b2 = v2[p]
        # A shape mismatch means the ensemble grid does not correspond to
        # coordinates.mat — reconstructing would emit plausible-but-wrong m/s.
        for cam, (ux, xpx) in enumerate(((ux1, x1), (ux2, x2)), start=1):
            if ux.shape != xpx.shape:
                raise ValueError(
                    f"ensemble pass {p} (cam{cam}): ux shape {ux.shape} != coords "
                    f"{xpx.shape}; the ensemble grid does not match coordinates.mat "
                    f"— cannot reconstruct")
        c1 = np.stack([x1, y1], axis=-1)
        c2 = np.stack([x2, y2], axis=-1)
        gX, gY, gZ, _spacing = regular_world_grid(
            record.model1, record.model2, c1, c2, z_world, tilt_x, tilt_y,
        )
        U, V, Wc, bmask = reconstruct_3c_field(
            record.model1, record.model2, (gX, gY, gZ),
            c1, ux1, uy1, c2, ux2, uy2,
            dt, interpolator=interpolator,
            bmask1=b1, bmask2=b2,
        )
        st["ux"][p], st["uy"][p], st["uz"][p] = U, V, Wc
        st["b_mask"][p] = bmask
        world_coords[p] = (gX, gY)

    scipy.io.savemat(str(out_dir / "coordinates.mat"),
                     {"coordinates": _coords_struct(world_coords, n_passes)},
                     oned_as="row", do_compression=True)
    out_b = out_dir / ENSEMBLE_FILE
    scipy.io.savemat(str(out_b), {"ensemble_result": st},
                     oned_as="row", do_compression=True)
    return [str(out_b)]


def reconstruct_stereo_run(
    record: StereoRecord,
    uncal_dir1: Path,
    uncal_dir2: Path,
    out_dir: Path,
    dt: float,
    pass_index: Optional[int] = None,
    z_world: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    vector_glob: str = "B*.mat",
    interpolator: str = "lanczos",
) -> List[str]:
    """3C reconstruction over a run: cam1 + cam2 uncalibrated -> stereo (U,V,W) m/s.

    Writes world coordinates (x,y mm) and 3C velocities (ux,uy,uz m/s) on a REGULAR
    world-mm grid spanning the two cameras' overlap (spacing = median world-space
    vector pitch — see ``regular_world_grid``) into ``out_dir``. Uses the final
    populated pass unless ``pass_index`` set.

    An ``ensemble_result.mat`` in both uncalibrated dirs routes to
    ``reconstruct_stereo_ensemble`` (mean flow, every populated pass) instead
    of the per-frame flow.
    """
    from .stereo_model import reconstruct_3c_field, regular_world_grid

    uncal_dir1 = Path(uncal_dir1)
    uncal_dir2 = Path(uncal_dir2)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords1 = read_coordinates(uncal_dir1 / "coordinates.mat")
    coords2 = read_coordinates(uncal_dir2 / "coordinates.mat")
    if not coords1 or not coords2:
        raise RuntimeError("missing coordinates.mat for stereo reconstruction")

    bmats1 = {b.name: b for b in _vector_files(uncal_dir1, vector_glob)}
    bmats2 = {b.name: b for b in _vector_files(uncal_dir2, vector_glob)}

    # Ensemble branch: ensemble_result.mat is a run-level mean, not a frame.
    # Detected directly on disk (not via vector_glob — a per-frame glob like
    # "B*.mat" would never match it and the branch would be silently skipped).
    ens1 = uncal_dir1 / ENSEMBLE_FILE
    ens2 = uncal_dir2 / ENSEMBLE_FILE
    have_ens1, have_ens2 = ens1.exists(), ens2.exists()
    if have_ens1 != have_ens2:
        raise RuntimeError(
            "ensemble_result.mat is present for only one camera — stereo "
            "reconstruction needs both cameras; re-run ensemble PIV for both")
    if have_ens1:
        frames1 = [b for b in bmats1 if b != ENSEMBLE_FILE]
        frames2 = [b for b in bmats2 if b != ENSEMBLE_FILE]
        if frames1 or frames2:
            raise RuntimeError(
                "uncalibrated dir mixes ensemble_result.mat with per-frame "
                "vector files — their grids would conflict in coordinates.mat; "
                "separate them (one type_name per output dir) and re-apply")
        written = reconstruct_stereo_ensemble(
            record, ens1, ens2,
            coords1, coords2, out_dir, dt,
            z_world, tilt_x, tilt_y, interpolator, pass_index,
        )
        if progress_cb:
            progress_cb(1, 1)
        return written

    p = pass_index if pass_index is not None else max(coords1)
    x1, y1 = coords1[p]
    x2, y2 = coords2[p]
    c1 = np.stack([x1, y1], axis=-1)
    c2 = np.stack([x2, y2], axis=-1)

    # The output grid is frame-invariant, and building it is the expensive part
    # (back-projections + Jacobians) — do it once, before any frame I/O, so an empty
    # overlap or degenerate spacing fails loudly up front.
    gX, gY, gZ, _spacing = regular_world_grid(
        record.model1, record.model2, c1, c2, z_world, tilt_x, tilt_y,
    )

    written: List[str] = []
    coords_written = False
    common = sorted(set(bmats1) & set(bmats2))
    total = len(common)
    for k, name in enumerate(common):
        v1 = read_vectors(bmats1[name])
        v2 = read_vectors(bmats2[name])
        if p not in v1 or p not in v2:
            continue
        ux1, uy1, b1 = v1[p]
        ux2, uy2, b2 = v2[p]
        U, V, Wc, bmask = reconstruct_3c_field(
            record.model1, record.model2, (gX, gY, gZ),
            c1, ux1, uy1, c2, ux2, uy2,
            dt, interpolator=interpolator,
            bmask1=b1, bmask2=b2,
        )
        n_passes = p + 1
        # The regular world grid is frame-invariant — write coordinates once, on the
        # first processed frame, not once per frame (no frames -> no output).
        if not coords_written:
            cs = _coords_struct({p: (gX, gY)}, n_passes)
            scipy.io.savemat(str(out_dir / "coordinates.mat"),
                             {"coordinates": cs}, oned_as="row", do_compression=True)
            coords_written = True
        st = np.empty(
            (n_passes,),
            dtype=[("ux", object), ("uy", object), ("uz", object), ("b_mask", object)],
        )
        empty = np.array([], dtype=np.float64)
        for i in range(n_passes):
            if i == p:
                st["ux"][i], st["uy"][i], st["uz"][i] = U, V, Wc
                st["b_mask"][i] = bmask
            else:
                st["ux"][i] = st["uy"][i] = st["uz"][i] = st["b_mask"][i] = empty
        out_b = out_dir / name
        scipy.io.savemat(str(out_b), {"piv_result": st}, oned_as="row", do_compression=True)
        written.append(str(out_b))
        if progress_cb:
            progress_cb(k + 1, total)
    return written

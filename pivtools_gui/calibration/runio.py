"""calibration.runio — read/write the production PIV layout for apply.

Reads the uncalibrated PIV output (``coordinates.mat`` + per-frame ``B*.mat``)
written by ``pivtools_cli.piv.save_results`` and writes calibrated output in the
same struct layout, so downstream tooling is unchanged. Crosses the MATLAB/pixel
boundary via ``frames`` (the saved coords are 1-based image-down).

Calibrated coordinates are world mm; calibrated velocities are m/s. No implicit
Y-flips — the sign is carried by the camera model.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.io
import yaml

from pivtools_core.config import CALIBRATION_ACTIVE_METHODS
from pivtools_core.paths import get_data_paths
from pivtools_gui.utils.worker_pool import (
    get_max_workers,
    get_worker_payload,
    payload_worker_initializer,
)

from . import frames
from . import record as rec
from .apply import (
    calibrate_coordinates,
    calibrate_displacements,
    calibrate_stress_tensor,
)
from .camera_model import CameraModel
from .record import MonoRecord, StereoRecord

logger = logging.getLogger(__name__)

# Ensemble PIV stores its result under this name + struct (vs per-frame "piv_result").
ENSEMBLE_FILE = "ensemble_result.mat"


class ApplyCancelled(Exception):
    """An apply run stopped by its cancel event before every frame was written.

    ``written`` lists the output files that WERE written (they stay on disk), ``done``
    counts the frames that finished, ``total`` the frames the run had planned.
    """

    def __init__(self, written: List[str], done: int, total: int) -> None:
        super().__init__(f"apply cancelled after {done}/{total} frames")
        self.written = written
        self.done = done
        self.total = total


class ApplyFailed(RuntimeError):
    """A frame of an apply run raised; the frames not yet started were cancelled.

    ``written`` lists the output files that WERE written (they stay on disk, including
    frames that were already running when the failure landed), ``done`` counts the frames
    that finished, ``total`` the frames the run had planned. The failing frame's exception
    is chained as ``__cause__``.
    """

    def __init__(
        self, error: BaseException, written: List[str], done: int, total: int
    ) -> None:
        super().__init__(
            f"apply failed after {done}/{total} frames; {len(written)} frame(s) written "
            f"remain on disk: {error}"
        )
        self.written = written
        self.done = done
        self.total = total


# Frames named in a stereo frame-set mismatch error (the rest are counted only).
_MISMATCH_NAMES_SHOWN = 3


def require_positive_dt(dt, where: str) -> float:
    """Return ``dt`` as a float, raising ValueError unless it is finite and > 0.

    Velocity scales as 1/dt, so a zero dt writes inf/NaN into every vector, a negative
    dt silently flips the velocity sign, and a non-finite dt poisons the whole field.
    ``where`` names the source of the value (for the error message).
    """
    try:
        value = float(dt)
    except (TypeError, ValueError):
        raise ValueError(f"dt from {where} must be a number, got {dt!r}") from None
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"dt from {where} must be finite and > 0 (velocity scales as 1/dt), "
            f"got {dt!r}"
        )
    return value


def resolve_dt(explicit_dt, model_dt) -> float:
    """Resolve the time-between-frames with NO silent default: explicit > model.

    Velocity scales linearly with dt, so a wrong dt silently corrupts every vector by a
    constant factor. There is therefore no safe default — and no config source: every
    record generated since dt stamping carries it in ``board_meta``, so a record
    without one is stale. If neither source supplies dt, this raises; a supplied dt
    that is not finite and > 0 raises too (see ``require_positive_dt``).
    """
    if explicit_dt is not None:
        return require_positive_dt(explicit_dt, "the explicit request/--dt")
    if model_dt is not None:
        return require_positive_dt(model_dt, "the model-stamped board_meta")
    raise ValueError(
        "dt could not be resolved — not given explicitly and not stamped in the "
        "model; re-generate the model (generation stamps dt) or pass dt"
    )


def plan_apply_units(
    full_cfg,
    source,
    board,
    stereo,
    type_name,
    active_paths=None,
    camera_pair=None,
    camera=None,
    explicit=None,
    model_type=None,
):
    """Resolve every (base_path x camera/pair) apply unit, deriving I/O dirs from config.

    Shared by the Flask apply route and the CLI's ``--all-paths`` apply, so both drive the
    same dataset-planning logic. Pure (no Flask / job-manager / threading): it reads config,
    loads the model record per unit, and returns a list of unit dicts:

        mono:   {"stereo": False, "record": MonoRecord,   "uncal": Path,  "out": Path,
                 "label": str, "base": Optional[Path]}
        stereo: {"stereo": True,  "record": StereoRecord, "uncal1": Path, "uncal2": Path,
                 "out": Path, "label": str, "base": Optional[Path]}

    ``base`` is the base path the unit was derived from, so apply can stamp the run's
    config snapshot with the calibration actually used. It is ``None`` for ``explicit``
    units, which have no base path and therefore no snapshot to stamp.

    ``active_paths`` is a list of indices into ``full_cfg.base_paths``; None or an empty list
    means every configured base path, and a scalar raises (``0`` would otherwise read as
    "all"). ``camera_pair`` is ``(cam1, cam2)``, required for stereo. ``camera`` is the mono
    camera for an explicit single-unit run (defaults ``full_cfg.camera_numbers[0]``). ``explicit``
    forces ONE ad-hoc unit, bypassing config derivation — ``{"uncal", "out"}`` (mono) or
    ``{"uncal1", "uncal2", "out"}`` (stereo). ``model_type`` picks the record when several
    types coexist in the model dir (None -> the single one present, ambiguity raises).
    A missing model fails loudly here (``load_*`` raises), before any job thread starts;
    for a config-derived mono plan the ``FileNotFoundError`` names the unmodelled camera
    (every configured camera is applied, so each needs a model).

    Every unit also carries ``cameras``: ``(cam,)`` for mono, ``(cam1, cam2)`` in record
    order for stereo, so a caller can say which cameras the run covers.

    Stereo ``uncal1`` / ``uncal2`` follow the RECORD's camera order (``rec.record_pair``),
    not ``camera_pair``'s: the pair dir is shared by ``(1, 2)`` and ``(2, 1)``, and
    ``reconstruct_stereo_run`` pairs ``uncal1`` with ``record.model1``, which is
    ``record.cam1``'s model. Explicit ``uncal1`` / ``uncal2`` must therefore be the
    record's cam1 / cam2 vectors; they are used as given and the unit label names that
    order. The derived output folder stays ``Cam{a}_Cam{b}`` in ``camera_pair`` order,
    where the downstream readers look for it; the 3C field it holds is in the shared
    world frame, so the pair's order does not change its content.
    """
    base_paths = full_cfg.base_paths
    nfp = full_cfg.num_frame_pairs
    if active_paths is not None and not isinstance(active_paths, (list, tuple)):
        raise ValueError(
            f"active_paths must be a list of base-path indices, got {active_paths!r}"
        )
    base_indices = (
        [int(i) for i in active_paths] if active_paths else list(range(len(base_paths)))
    )
    # A negative index would wrap to the last dataset and calibrate the wrong run.
    bad = [i for i in base_indices if not 0 <= i < len(base_paths)]
    if bad:
        raise ValueError(
            f"active_paths {bad} out of range: {len(base_paths)} base path(s) configured "
            f"(valid indices 0..{len(base_paths) - 1})"
        )

    units = []
    if stereo:
        if not camera_pair:
            raise ValueError("camera_pair is required for a stereo apply")
        pair = camera_pair
        out_pair = (int(pair[0]), int(pair[1]))
        record = rec.load_stereo(
            rec.stereo_model_dir_for_source(source, out_pair[0], out_pair[1], board),
            model_type=model_type,
        )
        # model1/model2 belong to record.cam1/cam2: read each camera's vectors in that order.
        cam1, cam2 = rec.record_pair(record, out_pair)
        if explicit:
            return [
                dict(
                    stereo=True,
                    record=record,
                    uncal1=Path(explicit["uncal1"]),
                    uncal2=Path(explicit["uncal2"]),
                    out=Path(explicit["out"]),
                    label=f"manual/Cam{cam1}_Cam{cam2}",
                    base=None,
                    cameras=(cam1, cam2),
                )
            ]
        for bi in base_indices:
            base = base_paths[bi]
            uncal1 = get_data_paths(base, nfp, cam1, type_name, use_uncalibrated=True)[
                "data_dir"
            ]
            uncal2 = get_data_paths(base, nfp, cam2, type_name, use_uncalibrated=True)[
                "data_dir"
            ]
            out = get_data_paths(
                base,
                nfp,
                out_pair[0],
                type_name,
                use_stereo=True,
                stereo_camera_pair=out_pair,
            )["data_dir"]
            units.append(
                dict(
                    stereo=True,
                    record=record,
                    uncal1=uncal1,
                    uncal2=uncal2,
                    out=out,
                    label=f"{base.name}/Cam{out_pair[0]}_Cam{out_pair[1]}",
                    base=base,
                    cameras=(cam1, cam2),
                )
            )
    else:
        if explicit:
            cam = int(camera) if camera is not None else int(full_cfg.camera_numbers[0])
            record = rec.mono_record_for_camera(
                rec.mono_model_dir_for_source(source, cam, board),
                cam,
                model_type=model_type,
            )
            return [
                dict(
                    stereo=False,
                    record=record,
                    uncal=Path(explicit["uncal"]),
                    out=Path(explicit["out"]),
                    label="manual",
                    base=None,
                    cameras=(cam,),
                )
            ]
        for bi in base_indices:
            base = base_paths[bi]
            for cam in full_cfg.camera_numbers:
                try:
                    record = rec.mono_record_for_camera(
                        rec.mono_model_dir_for_source(source, cam, board),
                        cam,
                        model_type=model_type,
                    )
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"camera {cam} has no {board} model — generate it first ({exc})"
                    ) from exc
                uncal = get_data_paths(
                    base, nfp, cam, type_name, use_uncalibrated=True
                )["data_dir"]
                out = get_data_paths(base, nfp, cam, type_name)["data_dir"]
                units.append(
                    dict(
                        stereo=False,
                        record=record,
                        uncal=uncal,
                        out=out,
                        label=f"{base.name}/Cam{cam}",
                        base=base,
                        cameras=(int(cam),),
                    )
                )
    return units


# Config.save_timestamped_copy names snapshots "config_<YYYY-MM-DD_HH-MM-SS>.yaml".
# Every field is fixed-width and zero-padded, so a lexical sort IS chronological and
# max() picks the newest without parsing the stamp.
_CONFIG_SNAPSHOT_GLOB = "config_*.yaml"

# Config.save() writes with these; the snapshot rewrite matches so a stamped file stays
# diffable against a freshly saved config.
_YAML_DUMP_KWARGS = dict(default_flow_style=False, sort_keys=False)


def active_method_name(board: str, stereo: bool) -> str:
    """Config's ``calibration.active`` name for a board applied mono or stereo.

    The apply paths carry the bare board ("charuco") plus a separate ``stereo`` flag,
    but config stores the fused name ("stereo_charuco"). Callers may pass either form —
    an already-prefixed board is returned unchanged rather than double-prefixed.
    """
    if not stereo or board.startswith("stereo_"):
        return board
    return f"stereo_{board}"


def parse_active_method(active: Optional[str]) -> Tuple[str, bool]:
    """Split a ``calibration.active`` id into ``(board, stereo)``.

    The inverse of ``active_method_name``: ``"stereo_charuco"`` -> ``("charuco", True)``,
    ``"scale_factor"`` -> ``("scale_factor", False)``. Calibration routes and CLI commands
    use it when the request / command line names no board, so an active id is never used
    verbatim as a board name.

    Raises
    ------
    ValueError
        ``active`` is missing/blank, or not one of ``CALIBRATION_ACTIVE_METHODS``.
    """
    allowed = ", ".join(CALIBRATION_ACTIVE_METHODS)
    if active is None or not str(active).strip():
        raise ValueError(
            f"calibration.active is required in config.yaml (one of {allowed}) when "
            "no board is given. Set it with the calibration tab's 'Set as Active' "
            "button, or pass the board explicitly."
        )
    if active not in CALIBRATION_ACTIVE_METHODS:
        raise ValueError(
            f"calibration.active {active!r} is not a calibration method (one of {allowed})"
        )
    if active.startswith("stereo_"):
        return active[len("stereo_") :], True
    return active, False


def build_applied_pointer(source, board: str, stereo: bool) -> Dict[str, object]:
    """The ``calibration`` pointer block describing what an apply actually used.

    Deliberately a single-entry ``calibration_sources``: the snapshot records one
    concrete apply, so listing the other configured sources would reintroduce the
    ambiguity this stamp exists to remove.
    """
    return {
        "calibration_sources": [str(source)],
        "source": str(source),
        "source_idx": 0,
        "active": active_method_name(board, stereo),
    }


def stamp_applied_calibration(base: Path, pointer: Dict[str, object]) -> Optional[Path]:
    """Rewrite the calibration pointer in the newest config snapshot under ``base``.

    The run's snapshot is archived when PIV *starts*, before any calibration has been
    chosen, so its calibration block is whatever the previous dataset left behind. Apply
    is the first moment the answer is known — this stamps it back so the archived config
    records the calibration that actually produced the vectors.

    Returns the stamped path, or ``None`` if the base path holds no snapshot (logged at
    WARNING — a base path with PIV output but no snapshot means something is off, so it
    is surfaced rather than silently skipped).

    Does NOT go through ``Config.save()``: that writes the live ``config.yaml``, which
    must keep pointing wherever the user currently has the GUI set.
    """
    base = Path(base)
    snapshots = sorted(base.glob(_CONFIG_SNAPSHOT_GLOB))
    if not snapshots:
        logger.warning(
            "no %s found in %s — calibration provenance not stamped for this run",
            _CONFIG_SNAPSHOT_GLOB,
            base,
        )
        return None

    snapshot = snapshots[-1]
    with open(snapshot, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["calibration"] = dict(pointer)

    # Atomic write with the retry from Config.save(): base paths sit on OneDrive and
    # external drives, where os.replace can transiently hit a lock held by a sync client.
    tmp_path = str(snapshot) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, **_YAML_DUMP_KWARGS)
    for attempt in range(5):
        try:
            os.replace(tmp_path, snapshot)
            break
        except PermissionError:
            if attempt < 4:
                time.sleep(0.1 * (attempt + 1))
            else:
                raise

    logger.info(
        "stamped applied calibration into %s (source=%s, active=%s)",
        snapshot.name,
        pointer.get("source"),
        pointer.get("active"),
    )
    return snapshot


def stamp_unit(unit: Dict, pointer: Dict[str, object]) -> Optional[str]:
    """Stamp one apply unit's base path; return a warning string if the stamp failed.

    The vectors are already on disk by the time this runs, so a provenance-write failure
    must not fail the apply. It is logged with its traceback and returned as a one-line
    warning naming the unit, so the caller can show it (the apply job's ``warnings``)
    rather than only the server log knowing. Returns None when the stamp succeeded or
    the unit has no base path to stamp.
    """
    base = unit.get("base")
    if base is None:
        logger.info(
            "apply unit %r has no base path (explicit dirs) — no config snapshot to stamp",
            unit.get("label"),
        )
        return None
    try:
        stamp_applied_calibration(base, pointer)
    except Exception as exc:  # provenance is secondary to the vectors already written
        logger.exception("failed to stamp applied calibration into %s", base)
        return (
            f"{unit.get('label')}: vectors written, but the calibration used was not "
            f"recorded in the run's config snapshot under {base} ({exc})"
        )
    return None


# .mat files in a PIV output dir that are NOT per-frame vector files. A vector glob
# derived from "%05d.mat" is "*.mat", which would otherwise sweep these in.
_NON_VECTOR_MATS = {"coordinates.mat", "mask.mat"}

# Diagnostic sidecars ensemble PIV writes alongside the result when store_planes /
# save_diagnostics is on: correlation planes ("planes_pass_N.mat"), first-pair warped
# images ("warped_pass_N.mat") and LM fit diagnostics ("fit_diagnostics_pass_N.mat"),
# one per pass. They carry no piv_result/ensemble_result struct, so a bare "*.mat" glob
# must skip them by prefix or read_vectors would KeyError on 'piv_result' (they are
# named per pass, so a fixed name set won't catch them).
_NON_VECTOR_MAT_PREFIXES = ("planes_pass_", "warped_pass_", "fit_diagnostics_pass_")


def _vector_files(uncal_dir: Path, vector_glob: str) -> List[Path]:
    """Per-frame vector files in ``uncal_dir`` matching ``vector_glob``, sorted.

    Excludes non-vector ``.mat`` files (coordinates/mask), ensemble diagnostic sidecars
    (``planes_pass_N.mat`` / ``warped_pass_N.mat`` from store_planes/save_diagnostics),
    and hidden / macOS AppleDouble sidecars (``._00001.mat`` on external drives) that a
    bare ``*.mat`` glob would otherwise sweep in and choke ``read_vectors``.
    """
    return sorted(
        b
        for b in Path(uncal_dir).glob(vector_glob)
        if b.name not in _NON_VECTOR_MATS
        and not b.name.startswith(_NON_VECTOR_MAT_PREFIXES)
        and not b.name.startswith(".")
    )


def _require_vector_files(uncal_dir: Path, vector_glob: str) -> List[Path]:
    """``_vector_files``, raising when the directory holds none.

    A run with no vector files would write only ``coordinates.mat`` and report success
    with zero frames — usually a wrong ``vector_glob`` or the wrong directory.
    """
    files = _vector_files(uncal_dir, vector_glob)
    if not files:
        raise FileNotFoundError(
            f"no vector files matching {vector_glob!r} in {uncal_dir} — check the PIV "
            "vector_format and that this is the uncalibrated output directory"
        )
    return files


def _frame_set_mismatch(
    only1: List[str], only2: List[str], dir1: Path, dir2: Path
) -> ValueError:
    """The error for two stereo cameras whose per-frame vector files differ."""

    def describe(names: List[str], where: Path) -> str:
        shown = ", ".join(names[:_MISMATCH_NAMES_SHOWN])
        more = "" if len(names) <= _MISMATCH_NAMES_SHOWN else ", ..."
        return f"{len(names)} frame(s) only in {where} ({shown}{more})"

    parts = [describe(n, d) for n, d in ((only1, dir1), (only2, dir2)) if n]
    return ValueError(
        "stereo cameras have different frame sets: "
        + "; ".join(parts)
        + ". Every frame must exist for both cameras — re-run PIV for the missing frames "
        "or remove the extras"
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


def read_vectors(
    bmat_path: Path,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Return {pass_idx: (ux_px, uy_px, b_mask)} from a per-frame B*.mat."""
    mat = scipy.io.loadmat(str(bmat_path), squeeze_me=True)
    pr = mat["piv_result"]
    out: Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
    n = pr.shape[0] if pr.ndim else 1
    uxf, uyf = pr["ux"], pr["uy"]
    # b_mask is required — every PIV writer emits it (save_results.py). A missing field
    # means a stale pre-b_mask file: fail loudly so it gets re-run, never fake a default
    # (forward-looking, no silent fallback — see CLAUDE.md).
    if "b_mask" not in pr.dtype.names:
        raise ValueError(
            f"{bmat_path.name}: piv_result has no b_mask field — stale pre-b_mask PIV "
            "output; re-run the PIV pass to regenerate it"
        )
    bmf = pr["b_mask"]
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
    in_path: Path,
    out_path: Path,
    model: CameraModel,
    coords_px: Dict[int, Tuple[np.ndarray, np.ndarray]],
    dt: float,
    z_world: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
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
    er = mat["ensemble_result"].reshape(-1)  # view; (n_passes,)
    names = er.dtype.names
    suffixes = _stress_suffixes(names)
    has_vel = ("ux" in names) and ("uy" in names)
    for i in range(er.shape[0]):
        if i not in coords_px:
            # An empty pass (pass 0 in real runs) has no coordinates and nothing to
            # calibrate. A filled one without coordinates would be written through as raw
            # px under a calibrated name, so it raises.
            filled = [
                nm
                for nm in (["ux"] if has_vel else [])
                + [p + suf for suf in suffixes for p in ("UU", "VV", "UV")]
                if _as2d(er[nm][i]).size
            ]
            if filled:
                raise ValueError(
                    f"{in_path.name}: ensemble pass {i} has data ({', '.join(filled)}) "
                    f"but coordinates.mat has no pass {i} — cannot calibrate it"
                )
            continue
        xpx, ypx = coords_px[i]
        coords = np.stack([xpx, ypx], axis=-1)
        if has_vel:
            ux = _as2d(er["ux"][i])
            uy = _as2d(er["uy"][i])
            if ux.size:
                # A shape mismatch means the ensemble grid does not correspond to
                # coordinates.mat — calibrating it would emit plausible-but-wrong m/s.
                # Fail loudly rather than write the raw px field through uncalibrated.
                if ux.shape != xpx.shape:
                    raise ValueError(
                        f"ensemble pass {i}: ux shape {ux.shape} != coords {xpx.shape}; "
                        f"the ensemble grid does not match coordinates.mat — cannot calibrate"
                    )
                u, v = calibrate_displacements(
                    model,
                    coords,
                    np.stack([ux, uy], axis=-1),
                    dt,
                    z_world,
                    tilt_x,
                    tilt_y,
                )
                er["ux"][i] = u
                er["uy"][i] = v
        for suf in suffixes:
            uu = _as2d(er["UU" + suf][i])
            if not uu.size:
                continue
            if uu.shape != xpx.shape:
                raise ValueError(
                    f"ensemble pass {i}: UU{suf} shape {uu.shape} != coords {xpx.shape}; "
                    f"the stress grid does not match coordinates.mat — cannot calibrate"
                )
            vv = _as2d(er["VV" + suf][i])
            uv = _as2d(er["UV" + suf][i])
            cu, cv, cuv = calibrate_stress_tensor(
                model, coords, uu, vv, uv, dt, z_world, tilt_x, tilt_y
            )
            er["UU" + suf][i] = cu
            er["VV" + suf][i] = cv
            er["UV" + suf][i] = cuv
    scipy.io.savemat(
        str(out_path), {"ensemble_result": er}, oned_as="row", do_compression=True
    )


def _coords_struct(
    per_pass: Dict[int, Tuple[np.ndarray, np.ndarray]], n_passes: int
) -> np.ndarray:
    st = np.empty((n_passes,), dtype=[("x", object), ("y", object)])
    empty = np.array([], dtype=np.float64)
    for i in range(n_passes):
        if i in per_pass:
            st["x"][i], st["y"][i] = per_pass[i]
        else:
            st["x"][i] = empty
            st["y"][i] = empty
    return st


def _vectors_struct(
    per_pass: Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    n_passes: int,
) -> np.ndarray:
    st = np.empty(
        (n_passes,), dtype=[("ux", object), ("uy", object), ("b_mask", object)]
    )
    empty = np.array([], dtype=np.float64)
    for i in range(n_passes):
        if i in per_pass:
            ux, uy, b = per_pass[i]
            st["ux"][i], st["uy"][i] = ux, uy
            st["b_mask"][i] = b if b is not None else empty
        else:
            st["ux"][i] = st["uy"][i] = st["b_mask"][i] = empty
    return st


# --- Parallel per-frame apply -------------------------------------------------
# Every vector file in a run is independent (read .mat -> transform -> write .mat),
# so apply/reconstruction fan out over a process pool sized by the
# ``processing.post_processing_workers`` knob (same pattern as transforms/merging).
# The shared per-job payload (model, coords, world grid) is installed once per
# worker via worker_pool.payload_worker_initializer — pre-pickled to bytes so the
# spawn bootstrap does not import numpy before thread pinning — instead of being
# pickled into every task.


def _mono_frame_task(bmat_path: str, out_path: str, ctx: Optional[dict] = None) -> str:
    """Calibrate one vector .mat file. ``ctx`` defaults to the worker's payload."""
    c = ctx if ctx is not None else get_worker_payload()
    bmat = Path(bmat_path)
    out_b = Path(out_path)
    if bmat.name == ENSEMBLE_FILE:
        # Ensemble: mean velocity + Reynolds-stress tensors (different struct/fields).
        calibrate_ensemble_file(
            bmat,
            out_b,
            c["model"],
            c["coords_px"],
            c["dt"],
            c["z_world"],
            c["tilt_x"],
            c["tilt_y"],
        )
        return str(out_b)
    vecs = read_vectors(bmat)
    cal_vecs: Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
    for i, (ux, uy, b) in vecs.items():
        # read_vectors drops empty passes, so every pass here has vectors.
        if i not in c["coords_px"]:
            raise ValueError(
                f"{bmat.name}: pass {i} has vectors but coordinates.mat has no pass {i} "
                "— cannot calibrate it"
            )
        xpx, ypx = c["coords_px"][i]
        coords_stack = np.stack([xpx, ypx], axis=-1)
        disp_stack = np.stack([ux, uy], axis=-1)
        u, v = calibrate_displacements(
            c["model"],
            coords_stack,
            disp_stack,
            c["dt"],
            c["z_world"],
            c["tilt_x"],
            c["tilt_y"],
        )
        cal_vecs[i] = (u, v, b)
    scipy.io.savemat(
        str(out_b),
        {"piv_result": _vectors_struct(cal_vecs, c["n_passes"])},
        oned_as="row",
        do_compression=True,
    )
    return str(out_b)


def _stereo_frame_task(
    bmat1_path: str, bmat2_path: str, out_path: str, ctx: Optional[dict] = None
) -> Optional[str]:
    """3C-reconstruct one frame pair; returns None when the pass is absent (skipped)."""
    from .stereo_model import reconstruct_3c_field

    c = ctx if ctx is not None else get_worker_payload()
    p = c["pass_index"]
    v1 = read_vectors(Path(bmat1_path))
    v2 = read_vectors(Path(bmat2_path))
    if p not in v1 or p not in v2:
        return None
    ux1, uy1, b1 = v1[p]
    ux2, uy2, b2 = v2[p]
    U, V, Wc, bmask = reconstruct_3c_field(
        c["model1"],
        c["model2"],
        c["grid"],
        c["c1"],
        ux1,
        uy1,
        c["c2"],
        ux2,
        uy2,
        c["dt"],
        interpolator=c["interpolator"],
        bmask1=b1,
        bmask2=b2,
    )
    n_passes = p + 1
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
    scipy.io.savemat(
        str(out_path), {"piv_result": st}, oned_as="row", do_compression=True
    )
    return str(out_path)


def _run_frame_tasks(
    task_fn: Callable[..., Optional[str]],
    tasks: List[tuple],
    ctx: dict,
    progress_cb: Optional[Callable[[int, int], None]],
    cancel_event: Optional[threading.Event] = None,
) -> List[str]:
    """Fan per-frame apply tasks out over a process pool sized by the worker knob.

    A single-worker run (tiny run, or knob set to 1) executes inline instead: no pool
    spin-up cost, and the parent keeps ``ctx`` explicit rather than mutating module
    state (which two concurrent apply jobs would race on). Skipped frames (task
    returns None) still advance progress. Output paths return in name order.

    ``cancel_event`` is checked before each inline frame and as each pooled frame
    completes. Once set, frames not yet started are cancelled, frames already running
    in the pool are waited for (so nothing is written after this returns), and
    ``ApplyCancelled`` is raised carrying the frames that did complete.

    A frame that raises stops the run the same way (queued frames cancelled, running
    ones waited for) and ``ApplyFailed`` is raised carrying the frames written, with the
    frame's exception chained.
    """
    total = len(tasks)
    written: List[str] = []
    if total == 0:
        return written
    done = 0
    max_workers = get_max_workers(total)
    if max_workers == 1:
        for t in tasks:
            if cancel_event is not None and cancel_event.is_set():
                raise ApplyCancelled(sorted(written), done, total)
            try:
                out = task_fn(*t, ctx=ctx)
            except Exception as exc:
                raise ApplyFailed(exc, sorted(written), done, total) from exc
            if out is not None:
                written.append(out)
            done += 1
            if progress_cb:
                progress_cb(done, total)
        return sorted(written)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=payload_worker_initializer,
        initargs=(pickle.dumps(ctx),),
    ) as executor:
        futures = [executor.submit(task_fn, *t) for t in tasks]
        collected: set[Future] = set()

        def _drain_running() -> None:
            """After shutdown(cancel_futures): count frames that were already running.

            Their exceptions are not re-raised: the run is already stopping for the
            first failure, and these frames' outcomes are only bookkeeping.
            """
            nonlocal done
            for f in futures:
                if f in collected or f.cancelled():
                    continue
                if f.exception() is None and f.result() is not None:
                    written.append(f.result())
                done += 1

        for future in as_completed(futures):
            collected.add(future)
            try:
                out = future.result()
            except Exception as exc:
                executor.shutdown(wait=True, cancel_futures=True)
                _drain_running()
                raise ApplyFailed(exc, sorted(written), done, total) from exc
            if out is not None:
                written.append(out)
            done += 1
            if progress_cb:
                progress_cb(done, total)
            if cancel_event is not None and cancel_event.is_set():
                executor.shutdown(wait=True, cancel_futures=True)
                # Frames already running when the cancel landed have now finished and
                # written their output; count them. A failure among them still raises.
                for f in futures:
                    if f in collected or f.cancelled():
                        continue
                    out = f.result()
                    if out is not None:
                        written.append(out)
                    done += 1
                if progress_cb:
                    progress_cb(done, total)
                raise ApplyCancelled(sorted(written), done, total)
    return sorted(written)


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
    cancel_event: Optional[threading.Event] = None,
) -> List[str]:
    """Apply a mono model to a whole run: coordinates.mat + every vector .mat.

    Reads ``<uncal_dir>/coordinates.mat`` and ``<uncal_dir>/<vector_glob>``; writes
    calibrated world-mm coordinates and m/s velocities to ``<out_dir>``. ``vector_glob``
    matches the per-frame vector files (default ``B*.mat``; PIV's ``vector_fmt`` may
    differ, e.g. ``[0-9]*.mat``). ``progress_cb(done, total)`` runs after each frame.
    A set ``cancel_event`` stops the run between frames (see ``_run_frame_tasks``).
    """
    uncal_dir = Path(uncal_dir)
    out_dir = Path(out_dir)
    # Listed before anything is written, so an empty run leaves no orphan coordinates.mat.
    bmats = _require_vector_files(uncal_dir, vector_glob)
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
        world = calibrate_coordinates(
            model, stacked, z_world, tilt_x, tilt_y, offset_mm=offset_mm
        )
        cal_coords[i] = (world[..., 0], world[..., 1])
    scipy.io.savemat(
        str(out_dir / "coordinates.mat"),
        {"coordinates": _coords_struct(cal_coords, n_passes)},
        oned_as="row",
        do_compression=True,
    )

    ctx = {
        "model": model,
        "coords_px": coords_px,
        "n_passes": n_passes,
        "dt": dt,
        "z_world": z_world,
        "tilt_x": tilt_x,
        "tilt_y": tilt_y,
    }
    tasks = [(str(b), str(out_dir / b.name)) for b in bmats]
    return _run_frame_tasks(_mono_frame_task, tasks, ctx, progress_cb, cancel_event)


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
    cancel_event: Optional[threading.Event] = None,
) -> Tuple[List[str], int]:
    """3C reconstruction over a run: cam1 + cam2 uncalibrated -> stereo (U,V,W) m/s.

    Writes world coordinates (x,y mm) and 3C velocities (ux,uy,uz m/s) on a REGULAR
    world-mm grid spanning the two cameras' overlap (spacing = median world-space
    vector pitch — see ``regular_world_grid``) into ``out_dir``. Uses the final
    populated pass unless ``pass_index`` set. A set ``cancel_event`` stops the run
    between frames (see ``_run_frame_tasks``); the frames already written keep their
    ``coordinates.mat``.

    Returns ``(written, n_skipped)``: the output paths, and the number of frame pairs
    skipped because the requested pass is absent from one or both cameras' vector files
    (those frames are not written). Cancel/failure raise as described above.
    """
    from .stereo_model import regular_world_grid

    uncal_dir1 = Path(uncal_dir1)
    uncal_dir2 = Path(uncal_dir2)
    out_dir = Path(out_dir)

    # Both cameras must hold the same frames: intersecting them would silently drop
    # frames from the 3C output. Checked before any model work or write.
    bmats1 = {b.name: b for b in _require_vector_files(uncal_dir1, vector_glob)}
    bmats2 = {b.name: b for b in _require_vector_files(uncal_dir2, vector_glob)}
    if set(bmats1) != set(bmats2):
        raise _frame_set_mismatch(
            sorted(set(bmats1) - set(bmats2)),
            sorted(set(bmats2) - set(bmats1)),
            uncal_dir1,
            uncal_dir2,
        )
    common = sorted(bmats1)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords1 = read_coordinates(uncal_dir1 / "coordinates.mat")
    coords2 = read_coordinates(uncal_dir2 / "coordinates.mat")
    if not coords1 or not coords2:
        raise RuntimeError("missing coordinates.mat for stereo reconstruction")
    p = pass_index if pass_index is not None else max(coords1)
    x1, y1 = coords1[p]
    x2, y2 = coords2[p]
    c1 = np.stack([x1, y1], axis=-1)
    c2 = np.stack([x2, y2], axis=-1)

    # The output grid is frame-invariant, and building it is the expensive part
    # (back-projections + Jacobians) — do it once, before any frame I/O, so an empty
    # overlap or degenerate spacing fails loudly up front.
    gX, gY, gZ, _spacing = regular_world_grid(
        record.model1,
        record.model2,
        c1,
        c2,
        z_world,
        tilt_x,
        tilt_y,
    )

    ctx = {
        "model1": record.model1,
        "model2": record.model2,
        "grid": (gX, gY, gZ),
        "c1": c1,
        "c2": c2,
        "pass_index": p,
        "dt": dt,
        "interpolator": interpolator,
    }
    tasks = [
        (str(bmats1[name]), str(bmats2[name]), str(out_dir / name)) for name in common
    ]

    def _write_coords() -> None:
        cs = _coords_struct({p: (gX, gY)}, p + 1)
        scipy.io.savemat(
            str(out_dir / "coordinates.mat"),
            {"coordinates": cs},
            oned_as="row",
            do_compression=True,
        )

    # The regular world grid is frame-invariant — write coordinates once, and only if
    # at least one frame actually reconstructed (no frames -> no output). A cancelled
    # run still gets them for the frames it left on disk, so that partial output is
    # readable rather than orphaned vectors with no grid.
    try:
        written = _run_frame_tasks(
            _stereo_frame_task, tasks, ctx, progress_cb, cancel_event
        )
    except ApplyCancelled as exc:
        if exc.written:
            _write_coords()
        raise
    if written:
        _write_coords()
    return written, len(tasks) - len(written)

from pathlib import Path
from typing import Optional

import dask.array as da
import dask.array.linalg as da_linalg
import matplotlib.pyplot as plt
import numpy as np
from dask.diagnostics.progress import ProgressBar
from scipy.io import loadmat, savemat  # add

from pivtools_core.config import Config
from pivtools_core.paths import get_data_paths
from ...plotting.plot_maker import make_scalar_settings, plot_scalar_field
from pivtools_core.vector_loading import (
    load_coords_from_directory,
    load_vectors_from_directory,
)


def _compute_pod(X: da.Array, k: int, normalise: bool):
    """
    Exact POD via snapshots method.
    X: dask array of shape (L, N) [features x time], float
    Returns (evals_desc, s_desc, phi_k [L,k], V_k [N,k], mu [L], std [L or None])
    """
    # Center (and optionally normalise) along time axis
    mu = X.mean(axis=1, keepdims=True)
    Xc = X - mu
    if normalise:
        std = X.std(axis=1, ddof=1, keepdims=True)
        eps = 1e-12
        Xc = Xc / (std + eps)
    else:
        std = None

    # Exact method of snapshots: C = Xc^T Xc  (N x N)
    # Rechunk for optimal performance (tune chunk size as needed)
    Xc = Xc.rechunk({0: -1, 1: "auto"})
    L_dim = int(Xc.shape[0])

    # Optionally, use Dask's ProgressBar for feedback

    with ProgressBar():
        C = da.dot(Xc.T, Xc)
        C_np = C.compute()
    # Numerical symmetrisation
    C_np = 0.5 * (C_np + C_np.T)

    # Exact eigendecomposition of C
    evals, V = np.linalg.eigh(C_np)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    V = V[:, order]
    svals = np.sqrt(np.clip(evals, 0.0, None))

    k_eff = min(k, V.shape[1])
    V_k = V[:, :k_eff]
    s_k = svals[:k_eff]

    # Spatial modes: phi_k = Xc @ v_k / s_k
    phi_cols = []
    for i in range(k_eff):
        vk = V_k[:, i]
        phi_i = da.dot(Xc, vk) / (s_k[i] + 1e-12)  # (L,)
        phi_cols.append(phi_i.compute())
    if phi_cols:
        Phi = np.stack(phi_cols, axis=1)
    else:
        Phi = np.zeros((L_dim, 0), dtype=float)

    mu_np = da.compute(mu)[0].ravel()
    if normalise:
        std_np = da.compute(std)[0].ravel()
    else:
        std_np = None
    return evals, svals, Phi, V_k, mu_np, std_np


def _compute_pod_randomized(
    X: da.Array,
    k: int,
    normalise: bool,
    oversampling: int = 10,
    power_iter: int = 1,
    random_state: Optional[int] = 0,
):
    """
    Randomized POD (Halko) using Dask-parallel matmuls.
    X: dask array of shape (L, N) [features x time]
    Returns (evals_desc, s_desc, phi_k [L,k], V_k [N,k], mu [L], std [L or None])
    """
    # Center and optionally normalise along time axis
    mu = X.mean(axis=1, keepdims=True)
    Xc = X - mu
    if normalise:
        std = X.std(axis=1, ddof=1, keepdims=True)
        eps = 1e-12
        Xc = Xc / (std + eps)
    else:
        std = None

    Xc = Xc.rechunk({0: -1, 1: "auto"})
    L, N = int(Xc.shape[0]), int(Xc.shape[1])

    k_target = min(k, N)
    r = min(k_target + max(0, oversampling), N)

    # Random test matrix (N x r)
    rng = np.random.default_rng(seed=random_state)
    Omega = rng.standard_normal(size=(N, r))

    # Y = Xc @ Omega  -> (L x r)
    Y = da.dot(Xc, Omega)

    # Power iterations
    for _ in range(max(0, power_iter)):
        Z = da.dot(Xc.T, Y)  # (N x r)
        Y = da.dot(Xc, Z)  # (L x r)

    # Orthonormal basis Q via QR (some stubs return 2 or 3 items)
    qr_out = da_linalg.qr(Y)
    if isinstance(qr_out, tuple):
        Q = qr_out[0]
    else:
        Q = qr_out

    # Small matrix B = Q^T Xc  -> (r x N)
    with ProgressBar():
        B = da.dot(Q.T, Xc).compute()  # numpy

    # SVD of B
    Uhat, S, Vt = np.linalg.svd(B, full_matrices=False)

    k_eff = min(k_target, Uhat.shape[1])
    Uhat_k = Uhat[:, :k_eff]
    S_k = S[:k_eff]
    V_k = Vt[:k_eff, :].T  # (N x k)

    # Spatial modes Phi = Q @ Uhat_k
    phi_cols = []
    for i in range(k_eff):
        qi = da.dot(Q, Uhat_k[:, i])
        with ProgressBar():
            phi_cols.append(qi.compute())
    Phi = np.stack(phi_cols, axis=1) if phi_cols else np.zeros((L, 0), dtype=float)

    evals = (S_k**2).copy()
    svals = S_k.copy()

    mu_np = da.compute(mu)[0].ravel()
    if normalise:
        std_np = da.compute(std)[0].ravel()
    else:
        std_np = None
    return evals, svals, Phi, V_k, mu_np, std_np


def _map_modes_to_grid(phi: np.ndarray, valid_flat: np.ndarray, hw: tuple[int, int]):
    """
    phi: (L, k) with L=P or 2P; valid_flat: (H*W,) boolean
    Returns:
      - if L==P: (modes_grid [k,H,W],)
      - if L==2P: (modes_ux [k,H,W], modes_uy [k,H,W])
    """
    H, W = hw
    P = valid_flat.sum()
    L, k = phi.shape
    if L == P:  # single component
        modes = np.zeros((k, H, W), dtype=phi.dtype)
        for i in range(k):
            g = np.zeros(H * W, dtype=phi.dtype)
            g[valid_flat] = phi[:, i]
            modes[i] = g.reshape(H, W)
        return (modes,)
    elif L == 2 * P:
        modes_ux = np.zeros((k, H, W), dtype=phi.dtype)
        modes_uy = np.zeros((k, H, W), dtype=phi.dtype)
        for i in range(k):
            g_u = np.zeros(H * W, dtype=phi.dtype)
            g_v = np.zeros(H * W, dtype=phi.dtype)
            g_u[valid_flat] = phi[:P, i]
            g_v[valid_flat] = phi[P:, i]
            modes_ux[i] = g_u.reshape(H, W)
            modes_uy[i] = g_v.reshape(H, W)
        return modes_ux, modes_uy
    else:
        raise ValueError("phi length inconsistent with valid mask size")


def pod_decompose(cam_num: int, config: Config, base: Path, k_modes: int = 10):
    """
    Run POD per selected pass (run) for a given camera.
    Uses randomized algorithm when settings.randomised is True.
    Saves under 'POD' (exact) or 'pod_randomised' (randomized).
    """
    if config.post_processing is None:
        return
    for entry in config.post_processing:
        if entry.get("type") != "POD":
            continue

        settings = entry.get("settings", {}) or {}
        # Accept both snake-case and camelCase keys from YAML/UI
        stack_u_y: bool = bool(
            settings.get(
                "stack_U_y", settings.get("stack_u_y", settings.get("stackUy", False))
            )
        )
        normalise: bool = bool(settings.get("normalise", False))
        use_randomised: bool = bool(settings.get("randomised", False))
        oversampling: int = int(settings.get("oversampling", 10))
        power_iter: int = int(settings.get("power_iter", 1))
        random_state: Optional[int] = settings.get("random_state", 0)

        # Allow endpoint/source selection as either top-level entry fields or inside settings
        endpoint: str = entry.get("endpoint", settings.get("endpoint", ""))
        use_merged: bool = bool(
            entry.get("use_merged", settings.get("use_merged", False))
        )
        source_type: str = entry.get(
            "source_type", settings.get("source_type", "instantaneous")
        )

        # Only first camera performs merged aggregation
        if use_merged and cam_num != config.camera_numbers[0]:
            continue

        # Use new get_data_paths signature
        paths = get_data_paths(
            base_dir=base,
            num_images=config.num_images,
            cam=cam_num,
            type_name=source_type,
            endpoint=endpoint,
            use_merged=use_merged,
        )
        data_dir = paths["data_dir"]
        stats_dir = paths["stats_dir"] / ("pod_randomised" if use_randomised else "POD")
        if not data_dir.exists():
            print(f"[POD] Data dir missing: {data_dir}")
            continue
        stats_dir.mkdir(parents=True, exist_ok=True)
        # Determine which runs to process (1-based labels)
        selected_runs_1based = (
            list(config.instantaneous_runs) if config.instantaneous_runs else []
        )
        # Load vector dataset lazily restricted to selected runs (if provided)
        arr = load_vectors_from_directory(
            data_dir,
            config,
            runs=selected_runs_1based if selected_runs_1based else None,
        )  # (N,R_sel,3,H,W)

        # Coordinates for plotting in the same order
        x_list, y_list = load_coords_from_directory(
            data_dir, runs=selected_runs_1based if selected_runs_1based else None
        )
        if not selected_runs_1based:
            R = int(arr.shape[1])
            selected_runs_1based = list(range(1, R + 1))

        print(
            f"[POD {'RAND' if use_randomised else 'EXACT'}] source={source_type}, cam={cam_num}, endpoint='{endpoint}', runs={selected_runs_1based}, stack_U_y={stack_u_y}, normalise={normalise}"
        )

        N = arr.shape[0]  # number of time samples loaded
        H = arr.shape[3]
        W = arr.shape[4]

        for lbl in selected_runs_1based:
            # Local index inside reduced R dimension (order matches selected_runs_1based)
            local_idx = selected_runs_1based.index(lbl)

            # Build mask (True means masked in plotting)
            b_mask = np.asarray(arr[0, local_idx, 2].compute()).astype(bool)
            valid_flat = (~b_mask).ravel()
            if valid_flat.sum() == 0:
                print(f"[POD] No valid points for run {lbl}; skipping")
                continue

            # Flattened time-stacks for ux/uy
            U = da.reshape(arr[:, local_idx, 0], (N, -1))  # (N, H*W)
            V = da.reshape(arr[:, local_idx, 1], (N, -1))  # (N, H*W)

            if stack_u_y:
                # Build X = [U_valid ; V_valid]^T -> (L, N)
                Usel = U[:, valid_flat]
                Vsel = V[:, valid_flat]
                X = da.concatenate([Usel, Vsel], axis=1).T.astype(np.float64)
                if use_randomised:
                    evals, svals, Phi, V_k, mu, std = _compute_pod_randomized(
                        X,
                        k=k_modes,
                        normalise=normalise,
                        oversampling=oversampling,
                        power_iter=power_iter,
                        random_state=random_state,
                    )
                else:
                    evals, svals, Phi, V_k, mu, std = _compute_pod(
                        X, k=k_modes, normalise=normalise
                    )
                modes_tuple = _map_modes_to_grid(Phi, valid_flat, (int(H), int(W)))
                if isinstance(modes_tuple, tuple) and len(modes_tuple) == 2:
                    modes_ux, modes_uy = modes_tuple  # type: ignore[misc]
                else:
                    # Fallback for static analysis; runtime should always return 2 when stacked
                    modes_ux = modes_tuple[0]
                    modes_uy = np.zeros_like(modes_ux)
                # Save MAT
                out_dir = stats_dir / f"run_{lbl:02d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / "POD_joint.mat"

                # Build meta without None values
                meta = {
                    "run_label": int(lbl),
                    "cam": int(cam_num),
                    "endpoint": endpoint,
                    "source_type": source_type,
                    "stack_U_y": True,
                    "normalise": bool(normalise),
                    "algorithm": "randomized" if use_randomised else "exact",
                }
                if use_randomised:
                    meta.update(
                        {
                            "oversampling": int(oversampling),
                            "power_iter": int(power_iter),
                        }
                    )

                # Energy breakdown (exact and randomized both have svals)
                s2 = np.asarray(svals) ** 2
                total = float(np.sum(s2)) if s2.size else 0.0
                energy_fraction = (s2 / total) if total > 0 else np.zeros_like(s2)
                energy_cumulative = (
                    np.cumsum(energy_fraction)
                    if energy_fraction.size
                    else energy_fraction
                )

                # Save summary .mat for all modes (for frontend energy plot)
                summary_file = out_dir / "POD_energy_summary.mat"
                savemat(
                    summary_file,
                    {
                        "eigenvalues": evals,
                        "singular_values": svals,
                        "energy_fraction": energy_fraction,
                        "energy_cumulative": energy_cumulative,
                        "meta": meta,
                    },
                )

                # Save a cumulative energy plot (PNG) so users can inspect energy after POD
                try:
                    fig, ax = plt.subplots(figsize=(6.0, 3.0))
                    modes = np.arange(1, energy_cumulative.size + 1)
                    ax.plot(modes, energy_cumulative, marker="o", lw=1.5)
                    ax.set_xlabel("Mode")
                    ax.set_ylabel("Cumulative Energy")
                    ax.set_title(f"POD cumulative energy - run {lbl}")
                    ax.set_ylim(0.0, 1.0)
                    ax.grid(True, linestyle="--", alpha=0.4)
                    out_png = (
                        out_dir / f"POD_energy_cumulative{config.plot_save_extension}"
                    )
                    fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
                    plt.close(fig)
                except Exception as e:
                    print(
                        f"[POD] Warning: failed to save cumulative energy PNG for run {lbl}: {e}"
                    )

                savemat(
                    out_file,
                    {
                        "eigenvalues": evals,
                        "singular_values": svals,
                        "energy_fraction": energy_fraction,
                        "energy_cumulative": energy_cumulative,
                        "modes_ux": modes_ux,  # [k,H,W]
                        "modes_uy": modes_uy,  # [k,H,W]
                        "mask": b_mask.astype(np.uint8),
                        "meta": meta,
                    },
                )
                # Plot and save each mode as PNG and .mat
                cx = x_list[local_idx] if local_idx < len(x_list) else None
                cy = y_list[local_idx] if local_idx < len(y_list) else None
                for k in range(min(k_modes, modes_ux.shape[0])):
                    # Save per-mode .mat for interactive viewers (already present)
                    savemat(
                        out_dir / f"ux_mode_{k + 1:02d}.mat",
                        {
                            "mode": modes_ux[k],
                            "k": int(k + 1),
                            "component": "ux",
                            "mask": b_mask.astype(np.uint8),
                            "meta": meta,
                        },
                    )
                    # Save PNG
                    save_base_ux = out_dir / f"ux_mode_{k + 1:02d}"
                    s_ux = make_scalar_settings(
                        config,
                        variable="POD ux",
                        run_label=lbl,
                        save_basepath=save_base_ux,
                        variable_units="",
                        coords_x=cx,
                        coords_y=cy,
                    )
                    fig, _, _ = plot_scalar_field(modes_ux[k], b_mask, s_ux)
                    fig.savefig(
                        f"{save_base_ux}{config.plot_save_extension}",
                        dpi=600,
                        bbox_inches="tight",
                    )
                    plt.close(fig)
                    # Save uy as .mat and PNG
                    savemat(
                        out_dir / f"uy_mode_{k + 1:02d}.mat",
                        {
                            "mode": modes_uy[k],
                            "k": int(k + 1),
                            "component": "uy",
                            "mask": b_mask.astype(np.uint8),
                            "meta": meta,
                        },
                    )
                    save_base_uy = out_dir / f"uy_mode_{k + 1:02d}"
                    s_uy = make_scalar_settings(
                        config,
                        variable="POD uy",
                        run_label=lbl,
                        save_basepath=save_base_uy,
                        variable_units="",
                        coords_x=cx,
                        coords_y=cy,
                    )
                    fig, _, _ = plot_scalar_field(modes_uy[k], b_mask, s_uy)
                    fig.savefig(
                        f"{save_base_uy}{config.plot_save_extension}",
                        dpi=600,
                        bbox_inches="tight",
                    )
                    plt.close(fig)
            else:
                # Separate UX
                Usel = U[:, valid_flat].T.astype(np.float64)  # (L, N)
                if use_randomised:
                    evals_u, svals_u, Phi_u, Vku, mu_u, std_u = _compute_pod_randomized(
                        Usel,
                        k=k_modes,
                        normalise=normalise,
                        oversampling=oversampling,
                        power_iter=power_iter,
                        random_state=random_state,
                    )
                else:
                    evals_u, svals_u, Phi_u, Vku, mu_u, std_u = _compute_pod(
                        Usel, k=k_modes, normalise=normalise
                    )
                mapped_u = _map_modes_to_grid(Phi_u, valid_flat, (int(H), int(W)))
                modes_u = mapped_u[0]

                # Separate UY
                Vsel = V[:, valid_flat].T.astype(np.float64)  # (L, N)
                if use_randomised:
                    evals_v, svals_v, Phi_v, Vkv, mu_v, std_v = _compute_pod_randomized(
                        Vsel,
                        k=k_modes,
                        normalise=normalise,
                        oversampling=oversampling,
                        power_iter=power_iter,
                        random_state=random_state,
                    )
                else:
                    evals_v, svals_v, Phi_v, Vkv, mu_v, std_v = _compute_pod(
                        Vsel, k=k_modes, normalise=normalise
                    )
                mapped_v = _map_modes_to_grid(Phi_v, valid_flat, (int(H), int(W)))
                modes_v = mapped_v[0]

                out_dir = stats_dir / f"run_{lbl:02d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / "POD_separate.mat"

                # Build meta without None values
                meta = {
                    "run_label": int(lbl),
                    "cam": int(cam_num),
                    "endpoint": endpoint,
                    "source_type": source_type,
                    "stack_U_y": False,
                    "normalise": bool(normalise),
                    "algorithm": "randomized" if use_randomised else "exact",
                }
                if use_randomised:
                    meta.update(
                        {
                            "oversampling": int(oversampling),
                            "power_iter": int(power_iter),
                        }
                    )

                # Energy breakdown per component
                s2u = np.asarray(svals_u) ** 2
                s2v = np.asarray(svals_v) ** 2
                totu = float(np.sum(s2u)) if s2u.size else 0.0
                totv = float(np.sum(s2v)) if s2v.size else 0.0
                energy_fraction_ux = (s2u / totu) if totu > 0 else np.zeros_like(s2u)
                energy_fraction_uy = (s2v / totv) if totv > 0 else np.zeros_like(s2v)
                energy_cumulative_ux = (
                    np.cumsum(energy_fraction_ux)
                    if energy_fraction_ux.size
                    else energy_fraction_ux
                )
                energy_cumulative_uy = (
                    np.cumsum(energy_fraction_uy)
                    if energy_fraction_uy.size
                    else energy_fraction_uy
                )

                # Save summary .mat for all modes (for frontend energy plot)
                summary_file = out_dir / "POD_energy_summary.mat"
                savemat(
                    summary_file,
                    {
                        "eigenvalues_ux": evals_u,
                        "singular_values_ux": svals_u,
                        "energy_fraction_ux": energy_fraction_ux,
                        "energy_cumulative_ux": energy_cumulative_ux,
                        "eigenvalues_uy": evals_v,
                        "singular_values_uy": svals_v,
                        "energy_fraction_uy": energy_fraction_uy,
                        "energy_cumulative_uy": energy_cumulative_uy,
                        "meta": meta,
                    },
                )

                # Save a cumulative energy plot (PNG) showing both ux and uy cumulative energy
                try:
                    fig, ax = plt.subplots(figsize=(6.0, 3.0))
                    modes_u = np.arange(1, energy_cumulative_ux.size + 1)
                    modes_v = np.arange(1, energy_cumulative_uy.size + 1)
                    if energy_cumulative_ux.size > 0:
                        ax.plot(
                            modes_u,
                            energy_cumulative_ux,
                            marker="o",
                            lw=1.2,
                            label="ux",
                        )
                    if energy_cumulative_uy.size > 0:
                        ax.plot(
                            modes_v,
                            energy_cumulative_uy,
                            marker="s",
                            lw=1.2,
                            label="uy",
                        )
                    ax.set_xlabel("Mode")
                    ax.set_ylabel("Cumulative Energy")
                    ax.set_title(f"POD cumulative energy - run {lbl}")
                    ax.set_ylim(0.0, 1.0)
                    ax.grid(True, linestyle="--", alpha=0.4)
                    ax.legend()
                    out_png = (
                        out_dir / f"POD_energy_cumulative{config.plot_save_extension}"
                    )
                    fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
                    plt.close(fig)
                except Exception as e:
                    print(
                        f"[POD] Warning: failed to save cumulative energy PNG for run {lbl}: {e}"
                    )

                savemat(
                    out_file,
                    {
                        "eigenvalues_ux": evals_u,
                        "singular_values_ux": svals_u,
                        "energy_fraction_ux": energy_fraction_ux,
                        "energy_cumulative_ux": energy_cumulative_ux,
                        "eigenvalues_uy": evals_v,
                        "singular_values_uy": svals_v,
                        "energy_fraction_uy": energy_fraction_uy,
                        "energy_cumulative_uy": energy_cumulative_uy,
                        "modes_ux": modes_u,  # [k,H,W]
                        "modes_uy": modes_v,  # [k,H,W]
                        "mask": b_mask.astype(np.uint8),
                        "meta": meta,
                    },
                )
                # Plot and save each mode as PNG and .mat
                cx = x_list[local_idx] if local_idx < len(x_list) else None
                cy = y_list[local_idx] if local_idx < len(y_list) else None
                for k in range(min(k_modes, modes_u.shape[0])):
                    # Save per-mode .mat for interactive viewers (already present)
                    savemat(
                        out_dir / f"ux_mode_{k + 1:02d}.mat",
                        {
                            "mode": modes_u[k],
                            "k": int(k + 1),
                            "component": "ux",
                            "mask": b_mask.astype(np.uint8),
                            "meta": meta,
                        },
                    )
                    save_base_ux = out_dir / f"ux_mode_{k + 1:02d}"
                    s_ux = make_scalar_settings(
                        config,
                        variable="POD ux",
                        run_label=lbl,
                        save_basepath=save_base_ux,
                        variable_units="",
                        coords_x=cx,
                        coords_y=cy,
                    )
                    fig, _, _ = plot_scalar_field(modes_u[k], b_mask, s_ux)
                    fig.savefig(
                        f"{save_base_ux}{config.plot_save_extension}",
                        dpi=600,
                        bbox_inches="tight",
                    )
                    plt.close(fig)
                    savemat(
                        out_dir / f"uy_mode_{k + 1:02d}.mat",
                        {
                            "mode": modes_v[k],
                            "k": int(k + 1),
                            "component": "uy",
                            "mask": b_mask.astype(np.uint8),
                            "meta": meta,
                        },
                    )
                    save_base_uy = out_dir / f"uy_mode_{k + 1:02d}"
                    s_uy = make_scalar_settings(
                        config,
                        variable="POD uy",
                        run_label=lbl,
                        save_basepath=save_base_uy,
                        variable_units="",
                        coords_x=cx,
                        coords_y=cy,
                    )
                    fig, _, _ = plot_scalar_field(modes_v[k], b_mask, s_uy)
                    fig.savefig(
                        f"{save_base_uy}{config.plot_save_extension}",
                        dpi=600,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

        print(
            f"[POD {'RAND' if use_randomised else 'EXACT'}] Completed POD for cam={cam_num}, endpoint='{endpoint}', saved -> {stats_dir}"
        )


def pod_rebuild(cam_num: int, config: Config, base: Path):
    """
    Rebuild calibrated vector fields for a given camera and selected runs using a prescribed energy fraction.
    - Reads POD stats (prefers 'pod_randomised', else 'POD') and their meta for stack/normalise flags.
    - Projects snapshots onto leading modes to reach 'energy' and reconstructs ux, uy.
    - Saves reconstructed .mat files under data endpoint 'POD_rebuild' with same filenames, only modifying the requested run.
    """
    if not config.post_processing:
        return

    # Find rebuild spec(s)
    rebuild_entries = [
        e for e in config.post_processing if e.get("type") == "POD_rebuild"
    ]
    if not rebuild_entries:
        return

    for entry in rebuild_entries:
        settings = entry.get("settings", {}) or {}
        # energy: accept 0..1 or 0..100
        energy = float(settings.get("energy", 0.8))
        if energy > 1.0:
            energy = energy / 100.0
        energy = float(np.clip(energy, 0.0, 1.0))

        endpoint = "POD_rebuild"  # output endpoint as requested
        use_merged: bool = bool(entry.get("use_merged", False))
        source_type: str = entry.get("source_type", "instantaneous")

        # Only first camera performs merged aggregation
        if use_merged and cam_num != config.camera_numbers[0]:
            continue

        # Input data (original calibrated) and stats base
        in_paths = get_data_paths(
            base_dir=base,
            num_images=config.num_images,
            cam=cam_num,
            type_name=source_type,
            endpoint="",
            use_merged=use_merged,
        )
        data_in_dir = in_paths["data_dir"]
        stats_base = in_paths["stats_dir"]

        # Output data dir (endpoint POD_rebuild)
        out_paths = get_data_paths(
            base_dir=base,
            num_images=config.num_images,
            cam=cam_num,
            type_name=source_type,
            endpoint=endpoint,
            use_merged=use_merged,
        )
        data_out_dir = out_paths["data_dir"]
        data_out_dir.mkdir(parents=True, exist_ok=True)

        # Copy coordinates.mat into the endpoint if available
        coords_src = data_in_dir / "coordinates.mat"
        coords_dst = data_out_dir / "coordinates.mat"
        if coords_src.exists() and not coords_dst.exists():
            try:
                import shutil

                shutil.copy2(coords_src, coords_dst)
            except Exception as e:
                print(f"[POD REBUILD] Warning: failed to copy coordinates.mat -> {e}")

        # Determine which runs to process (1-based labels)
        selected_runs_1based = (
            list(config.instantaneous_runs) if config.instantaneous_runs else []
        )

        # Load vectors lazily (for the runs of interest)
        arr = load_vectors_from_directory(
            data_in_dir,
            config,
            runs=selected_runs_1based if selected_runs_1based else None,
        )  # (N,R_sel,3,H,W)
        # keep existing chunking from loader

        if not selected_runs_1based:
            R = int(arr.shape[1])
            selected_runs_1based = list(range(1, R + 1))

        print(
            f"[POD REBUILD] source={source_type}, cam={cam_num}, runs={selected_runs_1based}, energy={energy:.3f}"
        )

        N = int(arr.shape[0])
        H = int(arr.shape[3])
        W = int(arr.shape[4])

        for lbl in selected_runs_1based:
            local_idx = selected_runs_1based.index(lbl)

            # Locate POD stats for this run: prefer randomized, else exact
            run_dir_rand = stats_base / "pod_randomised" / f"run_{lbl:02d}"
            run_dir_exact = stats_base / "POD" / f"run_{lbl:02d}"
            joint_file = "POD_joint.mat"
            sep_file = "POD_separate.mat"

            stats_path = None
            joint = False
            for base_dir in (run_dir_rand, run_dir_exact):
                if (base_dir / joint_file).exists():
                    stats_path = base_dir / joint_file
                    joint = True
                    break
                if (base_dir / sep_file).exists():
                    stats_path = base_dir / sep_file
                    joint = False
                    break
            if stats_path is None:
                print(
                    f"[POD REBUILD] No POD stats found for run {lbl} under {stats_base}"
                )
                continue

            pod_mat = loadmat(str(stats_path), struct_as_record=False, squeeze_me=True)

            # Extract meta
            def _get_meta_val(meta_obj, key, default=None):
                try:
                    if isinstance(meta_obj, dict):
                        return meta_obj.get(key, default)
                    return getattr(meta_obj, key, default)
                except Exception:
                    return default

            meta = pod_mat.get("meta", {})
            stack_u_y = bool(
                _get_meta_val(meta, "stack_U_y", settings.get("stack_u_y", False))
            )
            normalise = bool(
                _get_meta_val(meta, "normalise", settings.get("normalise", False))
            )

            # Validate consistency between file loaded (joint/separate) and meta.stack_U_y
            if joint and not stack_u_y:
                print(
                    f"[POD REBUILD] Warning: loaded joint POD file but meta.stack_U_y=False for run {lbl}"
                )
            if (not joint) and stack_u_y:
                print(
                    f"[POD REBUILD] Warning: loaded separate POD file but meta.stack_U_y=True for run {lbl}"
                )

            # Modes and singular values
            if joint:
                svals = np.asarray(pod_mat.get("singular_values", []))
                modes_ux = np.asarray(pod_mat["modes_ux"])  # [k,H,W]
                modes_uy = np.asarray(pod_mat["modes_uy"])  # [k,H,W]
                # choose k by energy
                if svals.size == 0:
                    print(f"[POD REBUILD] No singular values in {stats_path.name}")
                    continue
                en_cum = np.cumsum(svals**2) / np.sum(svals**2)
                k_use = (
                    int(np.searchsorted(en_cum, energy) + 1)
                    if "k_use" in locals()
                    else int(min(modes_ux.shape[0], modes_uy.shape[0]))
                )
                k_use = min(k_use, modes_ux.shape[0], modes_uy.shape[0])
            else:
                svals_u = np.asarray(pod_mat.get("singular_values_ux", []))
                svals_v = np.asarray(pod_mat.get("singular_values_uy", []))
                modes_ux = np.asarray(pod_mat["modes_ux"])  # [k,H,W]
                modes_uy = np.asarray(pod_mat["modes_uy"])  # [k,H,W]
                if svals_u.size == 0 or svals_v.size == 0:
                    print(f"[POD REBUILD] No singular values in {stats_path.name}")
                    continue
                en_cum_u = np.cumsum(svals_u**2) / np.sum(svals_u**2)
                en_cum_v = np.cumsum(svals_v**2) / np.sum(svals_v**2)
                k_u = int(np.searchsorted(en_cum_u, energy) + 1)
                k_v = int(np.searchsorted(en_cum_v, energy) + 1)
                k_use_u = min(k_u, modes_ux.shape[0])
                k_use_v = min(k_v, modes_uy.shape[0])

            # Mask and valid points
            b_mask = np.asarray(arr[0, local_idx, 2].compute()).astype(bool)
            valid_flat = (~b_mask).ravel()
            P = int(valid_flat.sum())
            if P == 0:
                print(f"[POD REBUILD] No valid points for run {lbl}; skipping")
                continue

            # Flattened time stacks
            U = arr[:, local_idx, 0].reshape((N, -1))  # (N, H*W)
            V = arr[:, local_idx, 1].reshape((N, -1))  # (N, H*W)
            Usel = U[:, valid_flat]  # (N, P)
            Vsel = V[:, valid_flat]  # (N, P)

            # Compute mu and std across time consistent with POD preprocessing
            # If POD used mean subtraction, compute; else zeros

            mu_u = np.asarray(Usel.mean(axis=0).compute())
            mu_v = np.asarray(Vsel.mean(axis=0).compute())

            if normalise:
                # ddof=1 to match POD
                std_u = np.asarray(Usel.std(axis=0, ddof=1).compute())
                std_v = np.asarray(Vsel.std(axis=0, ddof=1).compute())
                std_u = std_u + 1e-12
                std_v = std_v + 1e-12
            else:
                std_u = np.ones(P, dtype=np.float64)
                std_v = np.ones(P, dtype=np.float64)

            # Build Phi matrices (valid points only)
            if joint:
                # k modes shared
                k_use = int(k_use)
                Phi_u = (
                    modes_ux[:k_use].reshape((k_use, H * W))[:, valid_flat].T
                )  # (P, k)
                Phi_v = (
                    modes_uy[:k_use].reshape((k_use, H * W))[:, valid_flat].T
                )  # (P, k)
                # Stack features: [u_valid; v_valid]
                Phi = np.vstack([Phi_u, Phi_v])  # (2P, k)
                # Build Xc normalized for all times
                X_u_c = (Usel - mu_u) / std_u  # (N,P)
                X_v_c = (Vsel - mu_v) / std_v  # (N,P)
                Xc = da.concatenate([X_u_c, X_v_c], axis=1)  # (N, 2P)
                # Coeffs A = Xc @ Phi
                A = da.dot(Xc, Phi)  # (N, k)
                # Recon in feature space
                Xc_hat = da.dot(A, Phi.T)  # (N, 2P)
                # Split and de-normalise
                Xc_hat_u = Xc_hat[:, :P]
                Xc_hat_v = Xc_hat[:, P:]
                Urec_valid = Xc_hat_u * std_u + mu_u  # (N,P)
                Vrec_valid = Xc_hat_v * std_v + mu_v  # (N,P)
            else:
                # Separate UX
                Phi_u = (
                    modes_ux[:k_use_u].reshape((k_use_u, H * W))[:, valid_flat].T
                )  # (P, ku)
                Xu_c = (Usel - mu_u) / std_u  # (N,P)
                Au = da.dot(Xu_c, Phi_u)  # (N, ku)
                Xu_c_hat = da.dot(Au, Phi_u.T)  # (N,P)
                Urec_valid = Xu_c_hat * std_u + mu_u
                # Separate UY
                Phi_v = (
                    modes_uy[:k_use_v].reshape((k_use_v, H * W))[:, valid_flat].T
                )  # (P, kv)
                Xv_c = (Vsel - mu_v) / std_v  # (N,P)
                Av = da.dot(Xv_c, Phi_v)  # (N, kv)
                Xv_c_hat = da.dot(Av, Phi_v.T)  # (N,P)
                Vrec_valid = Xv_c_hat * std_v + mu_v

            # Prepare saving reconstructed frames to endpoint directory
            fmt = config.vector_format  # e.g., "%05d.mat"
            data_out_dir.mkdir(parents=True, exist_ok=True)

            # Utility to write piv_result using original as template
            import scipy.io as sio

            for t in range(N):
                # Prepare full grids with original for masked points
                u_full = np.asarray(U[t].compute()).reshape(H * W)
                v_full = np.asarray(V[t].compute()).reshape(H * W)
                # Replace only valid entries with reconstructed
                u_flat = u_full.copy()
                v_flat = v_full.copy()
                u_flat[valid_flat] = np.asarray(Urec_valid[t].compute())
                v_flat[valid_flat] = np.asarray(Vrec_valid[t].compute())
                u_grid = u_flat.reshape(H, W)
                v_grid = v_flat.reshape(H, W)

                # Read original .mat to preserve structure (especially multi-run)
                in_file = data_in_dir / (fmt % (t + 1))
                if not in_file.exists():
                    continue
                mat_in = sio.loadmat(
                    str(in_file), struct_as_record=False, squeeze_me=True
                )
                piv_result_in = mat_in["piv_result"]

                # Build MATLAB struct array for piv_result
                def _to_struct_array(piv, run_zero_based, u_new, v_new, bmask):
                    dtype = np.dtype([("ux", "O"), ("uy", "O"), ("b_mask", "O")])
                    if isinstance(piv, np.ndarray) and piv.dtype == object:
                        R = piv.size
                        out = np.empty((R,), dtype=dtype)
                        for rr in range(R):
                            pr = piv[rr]
                            out[rr]["ux"] = (
                                u_new if rr == run_zero_based else np.asarray(pr.ux)
                            )
                            out[rr]["uy"] = (
                                v_new if rr == run_zero_based else np.asarray(pr.uy)
                            )
                            out[rr]["b_mask"] = np.asarray(pr.b_mask)
                        return out
                    else:
                        out = np.empty((1,), dtype=dtype)
                        out[0]["ux"] = u_new
                        out[0]["uy"] = v_new
                        out[0]["b_mask"] = np.asarray(piv.b_mask)
                        return out

                piv_struct = _to_struct_array(
                    piv_result_in,
                    selected_runs_1based.index(lbl),
                    u_grid,
                    v_grid,
                    b_mask,
                )

                out_file = data_out_dir / (fmt % (t + 1))
                sio.savemat(
                    str(out_file), {"piv_result": piv_struct}, do_compression=True
                )

            print(f"[POD REBUILD] Run {lbl} -> saved to {data_out_dir}")

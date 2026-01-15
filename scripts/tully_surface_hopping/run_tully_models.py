#!/usr/bin/env python
"""
Batch runner for the canonical Tully 1D models (models 1–3) with simple
transmission/reflection statistics and a probability plot versus initial
velocity.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .TullyModels import TullyDynamics, TullyFSSH, TullyModel, TullyMolecule


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def get_model(name: str) -> TullyModel:
    name = name.lower()
    if name in ("1", "single", "sac", "single_avoided_crossing", "model1"):
        return TullyModel.single_crossing()
    if name in ("2", "double", "dac", "double_avoided_crossing", "model2"):
        return TullyModel.double_crossing()
    if name in ("3", "extended", "reflection", "model3", "extended_coupling"):
        return TullyModel.extended_coupling()
    raise ValueError(f"Unknown Tully model '{name}'")


def _model_key(name: str) -> str:
    name = name.lower()
    if name in ("1", "single", "sac", "single_avoided_crossing", "model1"):
        return "1"
    if name in ("2", "double", "dac", "double_avoided_crossing", "model2"):
        return "2"
    if name in ("3", "extended", "reflection", "model3", "extended_coupling"):
        return "3"
    return "1"


def _classify_exit(x_final: float, x0: float) -> Optional[bool]:
    """
    Returns True if transmitted (right of -x0), False if reflected (left of x0).
    Returns None if still within interaction window (between x0 and -x0).
    """
    if x_final >= -x0:
        return True
    if x_final <= x0:
        return False
    return None


def _run_single_worker(
    seed: int,
    model_name: str,
    method: str,
    v0: float,
    steps: int,
    timestep: float,
    mass: float,
    x0: float,
    elec_substeps: int,
    collect_density: bool,
) -> tuple[bool, torch.Tensor, Optional[np.ndarray]]:
    """Worker-safe single trajectory run."""
    torch.manual_seed(seed)
    model = get_model(model_name)
    dyn_cls = TullyDynamics if method == "ehrenfest" else TullyFSSH
    dyn = dyn_cls(model, timestep=timestep, electronic_substeps=elec_substeps)
    mol = TullyMolecule(x0=x0, v0=v0, mass=mass, dtype=torch.double)
    dyn._setup_states(mol)
    dyn._init_coeffs(mol)
    if collect_density and hasattr(dyn, "_reset_density_history"):
        dyn._reset_density_history()
    mol.dm = torch.zeros(1, 1, 1, device=mol.coordinates.device)
    with torch.no_grad():
        dyn.run(mol, steps=steps, reuse_P=True, remove_com=None)
    x_final = float(mol.coordinates[0, 0, 0])
    is_trans = _classify_exit(x_final, x0)
    active_state = None
    if hasattr(mol, "active_state"):
        act = mol.active_state
        if torch.is_tensor(act):
            active_state = int(act.view(-1)[0].item())
        else:
            active_state = int(act)
    if active_state is None and hasattr(dyn, "populations"):
        active_state = int(torch.argmax(dyn.populations[0]).item())
    # Convert to zero-based for counting (molecule.active_state is 1-based when set by dynamics)
    if active_state is not None:
        if active_state == 0:
            raise RuntimeError(
                "Encountered ground-state label in Tully dynamics; expected excited-state indices only."
            )
        active_state = active_state - 1
    rho_hist = None
    if collect_density and getattr(dyn, "rho_history", None):
        rho_hist = torch.stack(dyn.rho_history, dim=0).squeeze(1).numpy()
    return is_trans, active_state, rho_hist


import multiprocessing as mp


def run_ensemble(
    model: TullyModel,
    method: str,
    velocities: Sequence[float],
    *,
    ntraj: int = 1000,
    steps: int = 300,
    timestep: float = 0.25,
    mass: float = 1.0971598,  # 2000 a.u. in a.m.u.
    x0: float = -5.0,
    elec_substeps: int = 15,
    workers: int = 1,
    collect_density: bool = False,
) -> List[Dict]:
    model_name = model if isinstance(model, str) else getattr(model, "name", None)
    if workers > 1 and not isinstance(model, str):
        if model_name is None or model_name == "custom":
            raise ValueError("Parallel runs require a named built-in Tully model (1,2,3).")

        def _one(seed, v0):
            torch.manual_seed(seed)
            local_model = get_model(model_name) if isinstance(model_name, str) else model
            dyn_cls = TullyDynamics if method == "ehrenfest" else TullyFSSH
            dyn = dyn_cls(local_model, timestep=timestep, electronic_substeps=elec_substeps)
            mol = TullyMolecule(x0=x0, v0=v0, mass=mass, dtype=torch.double)
            dyn._setup_states(mol)
            dyn._init_coeffs(mol)
            if collect_density and hasattr(dyn, "_reset_density_history"):
                dyn._reset_density_history()
            mol.dm = torch.zeros(1, 1, 1, device=mol.coordinates.device)
            with torch.no_grad():
                dyn.run(mol, steps=steps, reuse_P=True, remove_com=None)
                x_final = float(mol.coordinates[0, 0, 0])
                is_trans = _classify_exit(x_final, x0)
                active_state = None
                if hasattr(mol, "active_state"):
                    act = mol.active_state
                    if torch.is_tensor(act):
                        active_state = int(act.view(-1)[0].item())
                    else:
                        active_state = int(act)
                if active_state is None and hasattr(dyn, "populations"):
                    active_state = int(torch.argmax(dyn.populations[0]).item())
                if active_state is not None:
                    if active_state == 0:
                        raise RuntimeError(
                            "Encountered ground-state label in Tully dynamics; expected excited-state indices only."
                        )
                    active_state = active_state - 1
                rho_hist = None
                if collect_density and getattr(dyn, "rho_history", None):
                    rho_hist = torch.stack(dyn.rho_history, dim=0).squeeze(1).numpy()
                return is_trans, active_state, rho_hist

    stats = []
    for v0 in velocities:
        trans_lower = refl_lower = 0.0
        trans_upper = refl_upper = 0.0
        excluded = 0
        rho_sum = None
        seeds = range(ntraj)
        if workers > 1:
            worker = partial(
                _run_single_worker,
                model_name=model_name,
                method=method,
                v0=v0,
                steps=steps,
                timestep=timestep,
                mass=mass,
                x0=x0,
                elec_substeps=elec_substeps,
                collect_density=collect_density,
            )
            try:
                ctx = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker) as ex:
                    for is_trans, active_state, rho_hist in ex.map(worker, seeds):
                        if is_trans is None or active_state is None:
                            excluded += 1
                            continue
                        if is_trans:
                            if active_state == 0:
                                trans_lower += 1.0
                            else:
                                trans_upper += 1.0
                        else:
                            if active_state == 0:
                                refl_lower += 1.0
                            else:
                                refl_upper += 1.0
                        if collect_density and rho_hist is not None:
                            rho_sum = rho_hist if rho_sum is None else rho_sum + rho_hist
            except PermissionError:
                print("Parallel execution not permitted; falling back to serial.")
                for seed in seeds:
                    is_trans, active_state, rho_hist = _one(seed, v0)
                    if is_trans is None or active_state is None:
                        excluded += 1
                        continue
                    if is_trans:
                        if active_state == 0:
                            trans_lower += 1.0
                        else:
                            trans_upper += 1.0
                    else:
                        if active_state == 0:
                            refl_lower += 1.0
                        else:
                            refl_upper += 1.0
                    if collect_density and rho_hist is not None:
                        rho_sum = rho_hist if rho_sum is None else rho_sum + rho_hist
        else:
            for seed in seeds:
                is_trans, active_state, rho_hist = _one(seed, v0)
                if is_trans is None or active_state is None:
                    excluded += 1
                    continue
                if is_trans:
                    if active_state == 0:
                        trans_lower += 1.0
                    else:
                        trans_upper += 1.0
                else:
                    if active_state == 0:
                        refl_lower += 1.0
                    else:
                        refl_upper += 1.0
                if collect_density and rho_hist is not None:
                    rho_sum = rho_hist if rho_sum is None else rho_sum + rho_hist
        rho_avg = None
        used = ntraj - excluded
        if used <= 0:
            raise RuntimeError("All trajectories were excluded (did not exit interaction region).")
        if excluded / float(ntraj) > 0.10:
            raise RuntimeError(f"More than 10% of trajectories excluded ({excluded}/{ntraj}).")
        if collect_density and rho_sum is not None:
            rho_avg = rho_sum / float(used)
        stats.append(
            {
                "v0": v0,
                "trans_lower": trans_lower / used,
                "refl_lower": refl_lower / used,
                "trans_upper": trans_upper / used,
                "refl_upper": refl_upper / used,
                "rho": rho_avg,
            }
        )
    return stats


ang_per_fs_to_au = 0.04571028904
amu_to_au = 1822.888486209


def plot_probs(stats: List[Dict], outfile: str, *, model_key: str, mass_amu: float):
    v0 = np.array([s["v0"] for s in stats]) * ang_per_fs_to_au
    mass_au = mass_amu * amu_to_au
    if model_key == "2":
        energy = 0.5 * mass_au * v0 * v0
        x = np.log(np.maximum(energy, 1e-18))
        xlabel = "ln(E) (a.u.)"
    else:
        x = mass_au * v0
        xlabel = "Momentum k (a.u.)"
    trans_lower = np.array([s["trans_lower"] for s in stats])
    refl_lower = np.array([s["refl_lower"] for s in stats])
    trans_upper = np.array([s["trans_upper"] for s in stats])

    plt.figure(figsize=(6, 4))
    plt.plot(x, trans_lower, "o-", label="Trans (lower)")
    plt.plot(x, refl_lower, "s-", label="Refl (lower)")
    plt.plot(x, trans_upper, "^-", label="Trans (upper)")
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print(f"Saved plot to {outfile}")


def plot_density_matrix(stats: List[Dict], timestep: float, outfile: str):
    base, ext = outfile.rsplit(".", 1) if "." in outfile else (outfile, "png")
    for s in stats:
        rho = s.get("rho")
        if rho is None:
            continue
        t = np.arange(rho.shape[0]) * timestep
        plt.figure(figsize=(6, 4))
        plt.plot(t, rho[:, 0], label=r"$\rho_{00}$")
        plt.plot(t, rho[:, 1], label=r"$\rho_{11}$")
        plt.plot(t, rho[:, 2], label=r"$|\rho_{01}|$")
        plt.xlabel("Time (fs)")
        plt.ylabel("Average density matrix element")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.tight_layout()
        tag = f"_v{s['v0']:.3f}"
        fname = f"{base}{tag}.{ext}"
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved density plot to {fname}")


def plot_adiabatic_energies(xmin: float, xmax: float, npoints: int, outfile: str):
    x = torch.linspace(xmin, xmax, npoints)
    models = [("1", "Model 1"), ("2", "Model 2"), ("3", "Model 3")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    nac_scale = {"1": 50.0, "2": 12.0, "3": 1.0}
    for ax, (name, title) in zip(axes, models):
        model = get_model(name)
        E, _, nac = model.pot(x)
        ax.plot(x, E[:, 0], label="E1")
        ax.plot(x, E[:, 1], label="E2")
        scale = nac_scale.get(name, 1.0)
        ax.plot(x, nac / scale, "k--", alpha=0.6, label=f"NAC/{scale:g}")
        ax.set_title(title)
        ax.set_xlabel("x (Ang)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Energy (eV)")
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="best")
    fig.tight_layout()
    # fig.savefig(outfile, dpi=200)
    # print(f"Saved plot to {outfile}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run Tully model ensembles and plot probabilities.")
    parser.add_argument(
        "--plot-energies",
        action="store_true",
        help="Plot adiabatic energies for all three Tully models and exit.",
    )
    parser.add_argument("--x-min", type=float, default=-5.0, help="Minimum x for energy plot")
    parser.add_argument("--x-max", type=float, default=5.0, help="Maximum x for energy plot")
    parser.add_argument("--x-points", type=int, default=400, help="Number of x points for energy plot")
    parser.add_argument("--energy-outfile", default="tully_energies.png", help="Energy plot filename")
    parser.add_argument("--model", default="1", help="Tully model: 1,2,3 or name")
    parser.add_argument("--method", default="fssh", choices=["fssh", "ehrenfest"])
    parser.add_argument("--velocities", nargs="+", type=float, default=None)
    parser.add_argument("--ntraj", type=int, default=1000, help="Trajectories per velocity")
    parser.add_argument("--steps", type=int, default=800, help="Steps per trajectory")
    parser.add_argument("--timestep", type=float, default=0.25, help="Time step (fs)")
    parser.add_argument(
        "--elec-substeps", type=int, default=15, help="Electronic RK4 substeps per nuclear step"
    )
    parser.add_argument("--mass", type=float, default=1.0971598, help="Mass in amu")
    parser.add_argument("--x0", type=float, default=-5.0, help="Initial position")
    parser.add_argument("--outfile", default="tully_probs.png", help="Output plot filename")
    parser.add_argument("--plot-density", action="store_true", help="Plot averaged density matrix elements.")
    parser.add_argument("--density-outfile", default="tully_density.png", help="Density matrix plot filename")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for trajectories")
    args = parser.parse_args()

    if args.plot_energies:
        plot_adiabatic_energies(args.x_min, args.x_max, args.x_points, args.energy_outfile)
        return

    model_key = _model_key(args.model)
    model = get_model(args.model)
    if args.velocities is None:
        if model_key == "1":
            args.velocities = [
                0.000000,
                0.062340,
                0.086983,
                0.089577,
                0.091721,
                0.094755,
                0.096916,
                0.098213,
                0.097625,
                0.098909,
                0.100171,
                0.101458,
                0.103606,
                0.106182,
                0.108759,
                0.112204,
                0.114355,
                0.116081,
                0.120809,
                0.125105,
                0.128989,
                0.133298,
                0.141458,
                0.157772,
                0.175405,
                0.213634,
                0.247218,
                0.276081,
                0.302391,
                0.325647,
                0.348513,
            ]
        elif model_key == "2":
            args.velocities = [
                0.108682557,
                0.114511738,
                0.12269802,
                0.130841687,
                0.137556403,
                0.146028095,
                0.162508927,
                0.172346449,
                0.18147174,
                0.193390034,
                0.212080866,
                0.229939171,
                0.251431901,
                0.277310134,
                0.31024587,
                0.346933934,
                0.398937048,
                0.460713493,
                0.552159122,
                0.701306223,
                0.950894222,
            ]
        else:
            args.velocities = np.arange(0.03, 0.33 + 1e-9, 0.01).tolist()
    stats = run_ensemble(
        model,
        args.method,
        args.velocities,
        ntraj=args.ntraj,
        steps=args.steps,
        timestep=args.timestep,
        mass=args.mass,
        x0=args.x0,
        elec_substeps=args.elec_substeps,
        workers=max(1, args.workers),
        collect_density=args.plot_density,
    )
    print("Velocity  Trans(L)  Refl(L)  Trans(U)")
    for s in stats:
        print(f"{s['v0']:7.3f}  {s['trans_lower']:7.3f}  {s['refl_lower']:7.3f}  {s['trans_upper']:7.3f}")
    plot_probs(stats, args.outfile, model_key=model_key, mass_amu=args.mass)
    if args.plot_density:
        plot_density_matrix(stats, args.timestep, args.density_outfile)


if __name__ == "__main__":
    main()

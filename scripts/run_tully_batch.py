#!/usr/bin/env python
"""
Batch runner for the canonical 1D Tully models using a single batched simulation
per velocity value. Keeps the plotting and density-matrix features of the
original script but leverages the batched nonadiabatic integrator instead of
launching many separate trajectories or processes.
"""

import argparse
import os
from typing import Dict, List, Sequence
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import torch

from seqm.TullyModels import TullyDynamics, TullyFSSH, TullyModel


def _set_single_thread():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _init_worker():
    _set_single_thread()


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


class BatchedTullyMolecule:
    """Minimal batched molecule holder for analytic Tully models."""

    def __init__(self, x0: float, v0: Sequence[float], mass: float, dtype=torch.double, device=None):
        device = device or torch.device("cpu")
        v0_tensor = torch.as_tensor(v0, dtype=dtype, device=device).view(-1)
        nmol = v0_tensor.shape[0]
        self.coordinates = torch.zeros((nmol, 1, 3), dtype=dtype, device=device)
        self.coordinates[:, 0, 0] = x0
        self.velocities = torch.zeros_like(self.coordinates)
        self.velocities[:, 0, 0] = v0_tensor
        self.acc = torch.zeros_like(self.coordinates)
        self.mass = torch.full((nmol, 1, 1), mass, dtype=dtype, device=device)
        self.mass_inverse = 1.0 / self.mass
        self.dm = torch.zeros(nmol, 1, 1, device=device, dtype=dtype)
        self.cis_amplitudes = None
        self.cis_energies = None
        self.transition_density_matrices = None
        self.force = torch.zeros_like(self.coordinates)
        self.all_forces = torch.zeros((nmol, 3, 1, 3), dtype=dtype, device=device)
        self.nac = None
        self.nac_dot = None
        self.Etot = torch.zeros(nmol, device=device, dtype=dtype)
        from seqm.seqm_functions.constants import Constants

        self.const = Constants().to(device)
        self.species = torch.ones((nmol, 1), device=device, dtype=torch.int64)
        self.num_atoms = torch.ones((nmol,), device=device, dtype=dtype)
        self.tot_charge = torch.zeros(nmol, device=device, dtype=dtype)
        self.mult = torch.ones(nmol, device=device, dtype=dtype)
        self.seqm_parameters = {"excited_states": {"n_states": 2}}
        # active_state is per-molecule (start on ground)
        self.active_state = torch.zeros(nmol, device=device, dtype=torch.long)


def _run_velocity(
    model: TullyModel,
    method: str,
    v0: float,
    ntraj: int,
    steps: int,
    timestep: float,
    mass: float,
    x0: float,
    elec_substeps: int,
    collect_density: bool,
    seed: int,
) -> Dict:
    torch.manual_seed(seed)
    dyn_cls = TullyDynamics if method == "ehrenfest" else TullyFSSH
    dyn = dyn_cls(model, timestep=timestep, electronic_substeps=elec_substeps)
    mol = BatchedTullyMolecule(x0=x0, v0=[v0] * ntraj, mass=mass, dtype=torch.double)
    dyn._setup_states(mol)
    dyn._init_coeffs(mol)
    if collect_density and hasattr(dyn, "_reset_density_history"):
        dyn._reset_density_history()
    with torch.no_grad():
        dyn.run(mol, steps=steps, reuse_P=True, remove_com=None)

    final_x = mol.coordinates[:, 0, 0]
    # Active state per trajectory; fallback to argmax(pop) if not set (Ehrenfest)
    if hasattr(mol, "active_state") and torch.is_tensor(mol.active_state):
        active_state = mol.active_state.view(-1).long()
    else:
        pop = dyn.populations.detach()
        active_state = torch.argmax(pop, dim=1)
    if torch.any(active_state == 0):
        raise RuntimeError("Encountered ground-state label in Tully dynamics; expected excited-state indices only.")
    active_state = active_state - 1  # convert to zero-based

    in_window = (final_x > x0) & (final_x < -x0)
    is_trans = final_x >= -x0
    include_mask = ~in_window
    lower = active_state == 0
    upper = active_state == 1
    trans_lower = torch.sum(is_trans & lower & include_mask).item()
    trans_upper = torch.sum(is_trans & upper & include_mask).item()
    refl_lower = torch.sum((~is_trans) & lower & include_mask).item()
    refl_upper = torch.sum((~is_trans) & upper & include_mask).item()
    excluded = int(torch.sum(in_window).item())
    included = ntraj - excluded
    if included <= 0:
        raise RuntimeError("All trajectories were excluded (did not exit interaction region).")
    if excluded / float(included + excluded) > 0.10:
        raise RuntimeError(f"More than 10% of trajectories excluded ({excluded}/{included + excluded}).")

    rho_avg = None
    if collect_density and getattr(dyn, "rho_history", None):
        rho_stack = torch.stack(dyn.rho_history, dim=0)  # (nsteps, nmol, 3)
        if include_mask.any():
            rho_sel = rho_stack[:, include_mask, :]
            rho_avg = rho_sel.mean(dim=1).cpu().numpy()

    return {
        "v0": v0,
        "trans_lower": trans_lower / included,
        "refl_lower": refl_lower / included,
        "trans_upper": trans_upper / included,
        "refl_upper": refl_upper / included,
        "rho": rho_avg,
        "included": included,
        "excluded": excluded,
    }


def _run_velocity_worker(args):
    (
        model_name,
        method,
        v0,
        ntraj,
        steps,
        timestep,
        mass,
        x0,
        elec_substeps,
        collect_density,
        seed,
    ) = args
    model = get_model(model_name)
    return _run_velocity(
        model,
        method,
        v0,
        ntraj=ntraj,
        steps=steps,
        timestep=timestep,
        mass=mass,
        x0=x0,
        elec_substeps=elec_substeps,
        collect_density=collect_density,
        seed=seed,
    )


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
    collect_density: bool = False,
    seed: int = 0,
    workers: int = 1,
) -> List[Dict]:
    model_name = model if isinstance(model, str) else getattr(model, "name", None)
    if workers > 1 and (not isinstance(model, str)) and (model_name is None or model_name == "custom"):
        raise ValueError("Parallel execution requires a named built-in Tully model (1,2,3).")
    model_key = model_name if isinstance(model, str) else model_name

    stats: List[Dict] = []
    v_seed_list = [(seed + i, v) for i, v in enumerate(velocities)]

    if workers > 1:
        args_list = [
            (
                model_key,
                method,
                v,
                ntraj,
                steps,
                timestep,
                mass,
                x0,
                elec_substeps,
                collect_density,
                v_seed,
            )
            for v_seed, v in v_seed_list
        ]
        try:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker) as ex:
                for res in ex.map(_run_velocity_worker, args_list):
                    stats.append(res)
        except PermissionError:
            print("Parallel execution not permitted; falling back to serial.")
            workers = 1

    if workers == 1:
        for v_seed, v in v_seed_list:
            torch.manual_seed(v_seed)
            local_model = get_model(model_key) if isinstance(model_key, str) else model
            stats.append(
                _run_velocity(
                    local_model,
                    method,
                    v,
                    ntraj=ntraj,
                    steps=steps,
                    timestep=timestep,
                    mass=mass,
                    x0=x0,
                    elec_substeps=elec_substeps,
                    collect_density=collect_density,
                    seed=v_seed,
                )
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
    models = [
        ("1", "Model 1"),
        ("2", "Model 2"),
        ("3", "Model 3"),
    ]
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
    plt.savefig(outfile, dpi=200)
    print(f"Saved plot to {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Run batched Tully model ensembles and plot probabilities.")
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
    parser.add_argument("--ntraj", type=int, default=1000, help="Trajectories per velocity (batched)")
    parser.add_argument("--steps", type=int, default=800, help="Steps per trajectory")
    parser.add_argument("--timestep", type=float, default=0.25, help="Time step (fs)")
    parser.add_argument("--elec-substeps", type=int, default=15, help="Electronic RK4 substeps per nuclear step")
    parser.add_argument("--mass", type=float, default=1.0971598, help="Mass in amu")
    parser.add_argument("--x0", type=float, default=-5.0, help="Initial position")
    parser.add_argument("--outfile", default="tully_probs.png", help="Output plot filename")
    parser.add_argument("--plot-density", action="store_true", help="Plot averaged density matrix elements.")
    parser.add_argument("--density-outfile", default="tully_density.png", help="Density matrix plot filename")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for hops")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (one velocity per worker)")
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
        collect_density=args.plot_density,
        seed=args.seed,
        workers=max(1, args.workers),
    )
    print("Velocity  Trans(L)  Refl(L)  Trans(U)")
    for s in stats:
        print(f"{s['v0']:7.3f}  {s['trans_lower']:7.3f}  {s['refl_lower']:7.3f}  {s['trans_upper']:7.3f}")
    plot_probs(stats, args.outfile, model_key=model_key, mass_amu=args.mass)
    if args.plot_density:
        plot_density_matrix(stats, args.timestep, args.density_outfile)


if __name__ == "__main__":
    main()

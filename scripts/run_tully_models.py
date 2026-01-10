#!/usr/bin/env python
"""
Batch runner for the canonical Tully 1D models (models 1–3) with simple
transmission/reflection statistics and a probability plot versus initial
velocity.
"""

import argparse
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from seqm.TullyModels import (
    TullyDynamics,
    TullyFSSH,
    TullyMolecule,
    TullyModel,
)
from seqm.MolecularDynamics import CONSTANTS


def get_model(name: str) -> TullyModel:
    name = name.lower()
    if name in ("1", "single", "sac", "single_avoided_crossing", "model1"):
        return TullyModel.single_crossing()
    if name in ("2", "double", "dac", "double_avoided_crossing", "model2"):
        return TullyModel.double_crossing()
    if name in ("3", "extended", "reflection", "model3"):
        return TullyModel.extended_coupling()
    raise ValueError(f"Unknown Tully model '{name}'")


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
) -> tuple[bool, bool]:
    """Worker-safe single trajectory run."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = get_model(model_name)
    dyn_cls = TullyDynamics if method == "ehrenfest" else TullyFSSH
    dyn = dyn_cls(model, timestep=timestep, electronic_substeps=elec_substeps)
    mol = TullyMolecule(x0=x0, v0=v0, mass=mass, dtype=torch.double)
    dyn._setup_states(mol)
    dyn._init_coeffs(mol)
    mol.dm = torch.zeros(1, 1, 1, device=mol.coordinates.device)
    with torch.no_grad():
        dyn.run(mol, steps=steps, reuse_P=True, remove_com=None)
    x_final = float(mol.coordinates[0, 0, 0])
    return (x_final > 0.0), (dyn._active_state == 1)


def run_ensemble(
    model: TullyModel,
    method: str,
    velocities: Sequence[float],
    *,
    ntraj: int = 1000,
    steps: int = 300,
    timestep: float = 0.25,
    mass: float = 1.0971598, # 2000 a.u. in a.m.u.
    x0: float = -5.0,
    elec_substeps: int = 15,
    workers: int = 1,
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
        mol.dm = torch.zeros(1, 1, 1, device=mol.coordinates.device)
        with torch.no_grad():
            dyn.run(mol, steps=steps, reuse_P=True, remove_com=None)
        x_final = float(mol.coordinates[0, 0, 0])
        return (x_final > 0.0), (dyn._active_state == 1)

    stats = []
    for v0 in velocities:
        trans = refl = 0
        state1 = 0
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
            )
            try:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    for is_trans, is_state1 in ex.map(worker, seeds):
                        trans += 1 if is_trans else 0
                        refl += 0 if is_trans else 1
                        state1 += 1 if is_state1 else 0
            except PermissionError:
                print("Parallel execution not permitted; falling back to serial.")
                for seed in seeds:
                    is_trans, is_state1 = _one(seed, v0)
                    trans += 1 if is_trans else 0
                    refl += 0 if is_trans else 1
                    state1 += 1 if is_state1 else 0
        else:
            for seed in seeds:
                is_trans, is_state1 = _one(seed, v0)
                trans += 1 if is_trans else 0
                refl += 0 if is_trans else 1
                state1 += 1 if is_state1 else 0
        stats.append(
            {
                "v0": v0,
                "trans": trans / ntraj,
                "reflect": refl / ntraj,
                "state1": state1 / ntraj,
            }
        )
    return stats


ang_per_fs_to_au = 0.04571028904
def plot_probs(stats: List[Dict], outfile: str):
    v0 = np.array([s["v0"] for s in stats])*ang_per_fs_to_au
    trans = np.array([s["trans"] for s in stats])
    refl = np.array([s["reflect"] for s in stats])
    state1 = np.array([s["state1"] for s in stats])

    plt.figure(figsize=(6, 4))
    plt.plot(v0, trans, "o-", label="Transmission")
    plt.plot(v0, refl, "s-", label="Reflection")
    plt.plot(v0, state1, "^-", label="Final on state 1")
    plt.xlabel("Initial velocity v0 (a.u.)")
    plt.ylabel("Probability")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print(f"Saved plot to {outfile}")


def plot_adiabatic_energies(xmin: float, xmax: float, npoints: int, outfile: str):
    x = torch.linspace(xmin, xmax, npoints)
    models = [
        ("1", "Model 1"),
        ("2", "Model 2"),
        ("3", "Model 3"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    nac_scale = {"1": 1.0, "2": 1.0, "3": 1.0}
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
    parser.add_argument("--velocities", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--ntraj", type=int, default=1000, help="Trajectories per velocity")
    parser.add_argument("--steps", type=int, default=300, help="Steps per trajectory")
    parser.add_argument("--timestep", type=float, default=0.25, help="Time step (fs)")
    parser.add_argument("--elec-substeps", type=int, default=15, help="Electronic RK4 substeps per nuclear step")
    parser.add_argument("--mass", type=float, default=1.0971598, help="Mass in amu")
    parser.add_argument("--x0", type=float, default=-5.0, help="Initial position")
    parser.add_argument("--outfile", default="tully_probs.png", help="Output plot filename")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for trajectories")
    args = parser.parse_args()

    if args.plot_energies:
        plot_adiabatic_energies(args.x_min, args.x_max, args.x_points, args.energy_outfile)
        return

    model = get_model(args.model)
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
    )
    print("Velocity  Trans   Reflect   Final state1")
    for s in stats:
        print(f"{s['v0']:7.3f}  {s['trans']:6.3f}  {s['reflect']:6.3f}  {s['state1']:6.3f}")
    plot_probs(stats, args.outfile)


if __name__ == "__main__":
    main()

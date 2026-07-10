#!/usr/bin/env python3
"""Run FSSH with HiphopNN energies, forces, and NAC vectors."""

import argparse
from pathlib import Path

import numpy as np
import torch
from hippynn_fssh import HANDOFF, HiphopSurfaceHopping, molecule, pairs, states


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=HANDOFF / "model")
    p.add_argument("--xyz-file", required=True)
    p.add_argument(
        "--states", default="0,1,2,3", help="S0 may be included as energy reference; S1+ propagate."
    )
    p.add_argument("--initial-state", type=int, default=3)
    p.add_argument("--nacr-pairs", default=None, help="All pairs among the propagated states.")
    p.add_argument("--nsteps", type=int, default=1000)
    p.add_argument("--timestep", type=float, default=0.25)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--velocities-file", help="N x 3 text velocities in Angstrom/fs; overrides --temperature.")
    p.add_argument("--friction", type=float)
    p.add_argument("--ntraj", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--output-prefix", default="hippynn_fssh")
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--trajectory-interval", type=int, default=10)
    return p


def main():
    a = parser().parse_args()
    label = states(a.states)
    need = pairs(label)
    if a.nacr_pairs and set(a.nacr_pairs.split(",")) != set(need):
        raise ValueError(f"--nacr-pairs must be {','.join(need)}")
    dev = torch.device(
        "cuda"
        if a.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if a.device == "auto"
        else a.device
    )
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output = {
        "molid": list(range(a.ntraj)),
        "prefix": str(out / a.output_prefix),
        "print every": a.log_interval,
        "xyz": a.trajectory_interval,
        "checkpoint every": 0,
        "h5": {
            "data": a.log_interval,
            "coordinates": a.trajectory_interval,
            "velocities": a.trajectory_interval,
            "forces": a.trajectory_interval,
            "nonadiabatic": a.log_interval,
        },
    }
    torch.manual_seed(a.seed)
    mol = molecule(a.xyz_file, a.ntraj, dev)
    if a.velocities_file:
        v = np.loadtxt(a.velocities_file)
        if v.shape != tuple(mol.coordinates.shape[1:]):
            raise ValueError(f"--velocities-file must have shape {tuple(mol.coordinates.shape[1:])}")
        mol.velocities = (
            torch.as_tensor(v, dtype=mol.coordinates.dtype, device=dev).expand_as(mol.coordinates).clone()
        )
    dyn = HiphopSurfaceHopping(
        a.model_dir, label, a.initial_state, a.timestep, a.temperature, output, str(dev), a.friction
    ).to(dev)
    print(f"FSSH states={tuple(x for x in label if x)}, pairs={need}; GDV uses ML forces (unvalidated).")
    dyn.run(mol, a.nsteps, reuse_P=False, remove_com=None, seed=a.seed)
    dyn._print_hop_log()


if __name__ == "__main__":
    main()

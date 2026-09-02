#!/usr/bin/env python3
"""
Parameterized FSSH run with PYSEQM.

One Slurm task should call this script with one frame/method/seed/etc.
"""

import argparse
import json
from pathlib import Path

import torch

from seqm.api import (
    Constants,
    Molecule,
    SurfaceHoppingDynamics,
    XLESurfaceHoppingDynamics,
    read_xyz_trajectory,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--coord-file", required=True)
    p.add_argument("--velocity-file")
    p.add_argument("--frame-index", type=int, required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--dynamics", choices=("cis", "xlesmd"), default="cis")

    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--initial-state", type=int, default=2)

    p.add_argument("--timestep", type=float, default=0.025)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--temp", type=float, default=300.0)

    p.add_argument("--n-states", type=int, default=3)
    p.add_argument("--scf-eps", type=float, default=1.0e-9)
    p.add_argument("--cis-tolerance", type=float, default=1.0e-7)

    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--outdir", required=True)
    p.add_argument("--tdc-method", required=True)
    p.add_argument("--h5-stride", type=int, default=2)
    p.add_argument("--h5-vector-stride", type=int, default=0)
    p.add_argument("--save-xlesmd-diagnostics", action="store_true")
    p.add_argument("--xl-k", type=int, default=6)
    p.add_argument("--xl-max-rank", type=int, default=13)
    p.add_argument("--xl-err-threshold", type=float, default=1.0e-8)
    p.add_argument("--xl-jacobian-regularization", type=float, default=2.0e-3)
    p.add_argument(
        "--xl-force-mode", choices=("autodiff", "experimental_analytic"), default="experimental_analytic"
    )

    return p.parse_args()


def main():
    args = parse_args()

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)

    device = torch.device("cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.h5_stride < 0 or args.h5_vector_stride < 0:
        raise ValueError("HDF5 output strides must be nonnegative.")
    if args.save_xlesmd_diagnostics and args.dynamics != "xlesmd":
        raise ValueError("--save-xlesmd-diagnostics requires --dynamics xlesmd.")

    species, coordinates, vel = read_xyz_trajectory(
        args.coord_file, vel_file=args.velocity_file, return_vel=True, indices=[args.frame_index]
    )
    coordinates = torch.as_tensor(coordinates, device=device)
    species = torch.as_tensor(species, device=device, dtype=torch.long)
    vel = torch.as_tensor(vel, device=device) * 1e-3  # convert from A/ps to A/fs

    nmol = coordinates.shape[0]

    const = Constants().to(device)

    seqm_parameters = {
        "method": args.method,
        "scf_eps": args.scf_eps,
        "scf_converger": [1],
        "excited_states": {"n_states": args.n_states, "method": "cis", "tolerance": args.cis_tolerance},
        "analytical_gradient": [True],
        "nonadiabatic": {"tdc_method": args.tdc_method, "decohere_on_hop": True, "detect_crossings": False},
        "torch_compile": False,
    }

    print(
        f"JOB {args.dynamics}-FSSH/{args.method} from frame {args.frame_index} "
        f"with seed {args.seed} and tdc method {args.tdc_method}"
    )
    output = {
        "molid": list(range(nmol)),
        "prefix": str(outdir / f"vik_{args.dynamics}_fssh.step_{args.timestep:.3f}"),
        "print every": 1000,
        "xyz": 0,
        "h5": {
            "data": args.h5_stride,
            "velocities": args.h5_vector_stride,
            "coordinates": args.h5_vector_stride,
            "forces": args.h5_vector_stride,
            "nonadiabatic": args.h5_stride,
        },
        "checkpoint every": 0,
    }

    torch.manual_seed(args.seed)

    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)

    with torch.no_grad():
        molecule.velocities = vel.clone()

    dynamics_kwargs = {
        "seqm_parameters": seqm_parameters,
        "timestep": args.timestep,
        "Temp": args.temp,
        "initial_state": args.initial_state,
        "output": output,
    }
    if args.dynamics == "cis":
        dyn = SurfaceHoppingDynamics(**dynamics_kwargs).to(device)
    else:
        dyn = XLESurfaceHoppingDynamics(
            **dynamics_kwargs,
            xl_bomd_params={
                "k": args.xl_k,
                "constraint_mode": "ordered_linearized",
                "max_rank": args.xl_max_rank,
                "err_threshold": args.xl_err_threshold,
                "jacobian_regularization": args.xl_jacobian_regularization,
                "krylov_preconditioner": "rank1",
                "force_mode": args.xl_force_mode,
            },
        ).to(device)

    dyn.run(molecule, args.steps, reuse_P=True, remove_com=("angular", 1))

    if args.save_xlesmd_diagnostics:
        payload = {
            "settings": {
                "xl_k": args.xl_k,
                "xl_max_rank": args.xl_max_rank,
                "xl_err_threshold": args.xl_err_threshold,
                "xl_jacobian_regularization": args.xl_jacobian_regularization,
            },
            "orthogonality": dyn.xlesmd_orthogonality_log,
            "energy_order_events": dyn.xlesmd_energy_order_events,
        }
        with (outdir / "xlesmd_diagnostics.json").open("w") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()

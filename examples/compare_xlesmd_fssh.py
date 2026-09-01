"""Compare a short ordered XL-ESMD FSSH trajectory with ordinary CIS-FSSH."""

import argparse
import json
import sys
from pathlib import Path

import torch

from seqm.api import Constants, Molecule, SurfaceHoppingDynamics, XLESurfaceHoppingDynamics, read_xyz


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xyz", type=Path)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--initial-state", type=int, default=1)
    parser.add_argument("--timestep", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=1700)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--tdc-method", choices=("overlap", "hamiltonian_fd", "nac_dot_v"), default="overlap")
    parser.add_argument("--force-mode", choices=("autodiff", "experimental_analytic"), default="autodiff")
    parser.add_argument(
        "--h5-prefix",
        type=Path,
        help=(
            "Write one HDF5 trajectory per method using this prefix.  Files are named "
            "<prefix>.cis.0.h5 and <prefix>.xlesmd.0.h5."
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def parameters(args):
    return {
        "method": "AM1",
        "scf_eps": 1.0e-10,
        "scf_converger": [2],
        "excited_states": {"n_states": args.states, "method": "cis", "tolerance": 1.0e-9},
        "active_state": args.initial_state,
        "analytical_gradient": [True],
        "nonadiabatic": {
            "tdc_method": args.tdc_method,
            "detect_crossings": False,
            "trajectory_termination": {"enabled": False},
        },
    }


def output_config(prefix=None):
    if prefix is None:
        return {"molid": [], "print every": 0, "xyz": 0, "h5": {"data": 0, "nonadiabatic": 0}}
    return {
        "molid": [0],
        "prefix": str(prefix),
        "print every": 0,
        "xyz": 0,
        "h5": {
            "data": 1,
            "coordinates": 1,
            "velocities": 1,
            "forces": 1,
            "nonadiabatic": 1,
        },
    }


def build_molecule(device, species, coordinates, params):
    molecule = Molecule(Constants().to(device), params, coordinates.clone(), species.clone()).to(device)
    molecule.verbose = False
    return molecule


def main():
    args = parse_args()
    if args.steps < 1 or args.states < 2:
        raise ValueError("--steps must be positive and --states must be at least two.")
    torch.set_default_dtype(torch.float64)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(args.device)
    species, coordinates = read_xyz([str(args.xyz.resolve())])
    species = torch.as_tensor(species, dtype=torch.long, device=device)
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64, device=device)

    cis_params = parameters(args)
    xl_params = parameters(args)
    cis_molecule = build_molecule(device, species, coordinates, cis_params)
    xl_molecule = build_molecule(device, species, coordinates, xl_params)
    cis = SurfaceHoppingDynamics(
        seqm_parameters=cis_params,
        timestep=args.timestep,
        Temp=args.temperature,
        initial_state=args.initial_state,
        output=output_config(None if args.h5_prefix is None else args.h5_prefix.with_name(args.h5_prefix.name + ".cis")),
    ).to(device)
    xl = XLESurfaceHoppingDynamics(
        seqm_parameters=xl_params,
        xl_bomd_params={
            "k": 6,
            "constraint_mode": "ordered_linearized",
            "max_rank": 3,
            "err_threshold": 1.0e-8,
            "krylov_preconditioner": "rank1",
            "force_mode": args.force_mode,
        },
        timestep=args.timestep,
        Temp=args.temperature,
        initial_state=args.initial_state,
        output=output_config(
            None if args.h5_prefix is None else args.h5_prefix.with_name(args.h5_prefix.name + ".xlesmd")
        ),
    ).to(device)

    torch.manual_seed(args.seed)
    cis.initialize(cis_molecule, remove_com=None, learned_parameters={}, steps=args.steps)
    torch.manual_seed(args.seed)
    xl.initialize(xl_molecule, remove_com=None, learned_parameters={}, steps=args.steps)
    coordinate_errors = []
    energy_errors = []
    population_errors = []
    for step in range(args.steps):
        torch.manual_seed(args.seed + 1 + step)
        cis._do_integrator_step(step, cis_molecule, {})
        torch.manual_seed(args.seed + 1 + step)
        xl._do_integrator_step(step, xl_molecule, {})
        if args.h5_prefix is not None:
            with torch.no_grad():
                for dynamics, molecule in ((cis, cis_molecule), (xl, xl_molecule)):
                    kinetic_energy = dynamics._kinetic_energy(molecule)
                    temperature = dynamics._calc_temperature(kinetic_energy)
                    potential_energy = dynamics._thermo_potential(molecule)
                    dynamics._h5_writer.append_data(
                        step + 1,
                        molecule,
                        temperature,
                        kinetic_energy,
                        potential_energy,
                        molecule.e_gap,
                    )
                    dynamics._h5_writer.append_vectors(step + 1, molecule)
        coordinate_errors.append(torch.max(torch.abs(xl_molecule.coordinates - cis_molecule.coordinates)))
        energy_errors.append(torch.max(torch.abs(xl._cache_old["energies"] - cis._cache_old["energies"])))
        population_errors.append(torch.max(torch.abs(xl.populations - cis.populations)))
        if args.progress_every > 0 and ((step + 1) % args.progress_every == 0 or step + 1 == args.steps):
            print(
                f"step {step + 1}/{args.steps}: "
                f"dR={coordinate_errors[-1].item():.3e} A, "
                f"dE={energy_errors[-1].item():.3e} eV, "
                f"dP={population_errors[-1].item():.3e}, "
                f"XL Gram={xl.xlesmd_orthogonality_log[-1]['xi_max_gram_error']:.3e}, "
                f"active CIS/XL={int(cis._active_states[0]) + 1}/{int(xl._active_states[0]) + 1}",
                file=sys.stderr,
                flush=True,
            )

    result = {
        "xyz": str(args.xyz.resolve()),
        "steps": args.steps,
        "duration_fs": args.steps * args.timestep,
        "temperature_K": args.temperature,
        "initial_state": args.initial_state,
        "tdc_method": args.tdc_method,
        "active_states_match": bool(torch.equal(xl._active_states, cis._active_states)),
        "maximum_coordinate_difference_angstrom": float(torch.stack(coordinate_errors).max().item()),
        "maximum_excitation_energy_difference_ev": float(torch.stack(energy_errors).max().item()),
        "maximum_population_difference": float(torch.stack(population_errors).max().item()),
        "maximum_xi_gram_error": max(d["xi_max_gram_error"] for d in xl.xlesmd_orthogonality_log),
        "maximum_xi_norm_error": max(d["xi_max_norm_error"] for d in xl.xlesmd_orthogonality_log),
        "maximum_xi_offdiagonal_overlap": max(
            d["xi_max_offdiag_overlap"] for d in xl.xlesmd_orthogonality_log
        ),
        "ordinary_hops": len(cis.hop_log),
        "xlesmd_hops": len(xl.hop_log),
        "ordinary_hop_log": [event.__dict__ for event in cis.hop_log],
        "xlesmd_hop_log": [event.__dict__ for event in xl.hop_log],
    }
    if args.h5_prefix is not None:
        for dynamics in (cis, xl):
            dynamics._h5_writer.flush()
            dynamics._h5_writer.close()
        result["cis_h5"] = str(args.h5_prefix.with_name(args.h5_prefix.name + ".cis.0.h5"))
        result["xlesmd_h5"] = str(args.h5_prefix.with_name(args.h5_prefix.name + ".xlesmd.0.h5"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

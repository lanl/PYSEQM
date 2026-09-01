"""Run the validated three-state XL-ESMD cyclopropene example.

The default is a 500-step NVE run.  ``--stop-at`` creates a restart file
whose final target remains ``--steps``; resume it with ``--resume``.
"""

import argparse
from pathlib import Path

import torch

from seqm.api import Constants, Molecule, read_xyz
from seqm.MolecularDynamics import XL_ESMD, HDF5Writer, Molecular_Dynamics_Basic, XYZWriter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XYZ = ROOT / "xyz_outputs" / "cyclopropene.xyz"
DEFAULT_PREFIX = ROOT / "xlesmd_runs" / "cyclopropene_3state_ordered"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", type=Path, default=DEFAULT_XYZ)
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--stop-at", type=int, help="Stop cleanly at this step and save a restart targeting --steps."
    )
    parser.add_argument("--resume", action="store_true", help="Resume PREFIX.restart.pt.")
    parser.add_argument(
        "--constraint-mode",
        choices=("ordered_linearized", "2a", "2b"),
        default="ordered_linearized",
        help="Ordered mode propagates three labelled adiabatic roots; 2a/2b propagate only their subspace.",
    )
    parser.add_argument(
        "--rank", type=int, default=4, help="Krylov rank (4 is validated with the rank-1 preconditioner)."
    )
    parser.add_argument(
        "--krylov-preconditioner",
        choices=("none", "diagonal", "rank1"),
        default="rank1",
        help="Right preconditioner for labelled-root GMRES.",
    )
    parser.add_argument(
        "--coupled-krylov-preconditioner",
        choices=("auto", "none", "lambda"),
        default="auto",
        help="Right preconditioner for raw block GMRES; auto selects lambda for 2a and none otherwise.",
    )
    parser.add_argument(
        "--force-mode",
        choices=("autodiff", "experimental_analytic"),
        default="autodiff",
        help=(
            "autodiff is the verified off-shell shadow force; experimental_analytic "
            "uses the retained 1a/1b/ordered relaxed-density gradient."
        ),
    )
    parser.add_argument("--timestep", type=float, default=0.1, help="Timestep in fs.")
    parser.add_argument("--temperature", type=float, default=300.0, help="Initial temperature in K.")
    parser.add_argument(
        "--damp",
        type=float,
        help="Optional Langevin damping time in fs; omit for the validated NVE trajectory.",
    )
    parser.add_argument("--seed", type=int, default=197)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps <= 0 or args.rank <= 0:
        raise ValueError("--steps and --rank must be positive.")
    if args.stop_at is not None and not 0 < args.stop_at < args.steps:
        raise ValueError("--stop-at must lie strictly between 0 and --steps.")
    if args.force_mode == "experimental_analytic" and args.constraint_mode in {"2a", "2b"}:
        raise ValueError("experimental_analytic is available only for 1a, 1b, and ordered XL-ESMD modes.")

    torch.set_default_dtype(torch.float64)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(args.device)
    prefix = args.prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    restart = Path(f"{prefix}.restart.pt")

    if args.resume:
        if not restart.exists():
            raise FileNotFoundError(f"No restart file found: {restart}")
        Molecular_Dynamics_Basic.run_from_checkpoint(str(restart), device=device)
        return

    species, coordinates = read_xyz([str(args.xyz.resolve())])
    species = torch.as_tensor(species, dtype=torch.int64, device=device)
    coordinates = torch.as_tensor(coordinates, dtype=torch.get_default_dtype(), device=device)
    parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-10,
        "scf_converger": [2],
        "scf_backward": 2,
        "excited_states": {"n_states": 3, "method": "cis", "tolerance": 1.0e-9},
        "active_state": 1,
        "analytical_gradient": [args.force_mode == "experimental_analytic"],
    }
    coupled_preconditioner = args.coupled_krylov_preconditioner
    if coupled_preconditioner == "auto":
        coupled_preconditioner = "lambda" if args.constraint_mode == "2a" else "none"
    xl_parameters = {
        "k": 6,
        "constraint_mode": args.constraint_mode,
        "max_rank": args.rank,
        "krylov_preconditioner": args.krylov_preconditioner,
        "coupled_krylov_preconditioner": coupled_preconditioner,
        "err_threshold": 1.0e-8,
        "coupled_tolerance": 1.0e-11,
        "coupled_max_iter": 300,
        # Raw coupled modes use a polar/Stiefel retraction; ordered mode uses
        # its label-preserving Gram--Schmidt retraction.
        "coupled_retract_auxiliary": True,
        "transport_mo_auxiliary": True,
        "scf_backward": 2,
        "force_mode": args.force_mode,
    }
    output = {
        "molid": [0],
        "prefix": str(prefix),
        "print every": 50,
        "xyz": 10,
        "h5": {"data": 1, "coordinates": 1, "velocities": 1},
        "checkpoint every": 250,
    }
    torch.manual_seed(args.seed)
    molecule = Molecule(Constants().to(device), parameters, coordinates, species).to(device)
    md = XL_ESMD(
        damp=args.damp,
        xl_bomd_params=xl_parameters,
        Temp=args.temperature,
        seqm_parameters=parameters,
        timestep=args.timestep,
        output=output,
    ).to(device)

    if args.stop_at is not None:
        # Initialize the writers with the final 500-step allocation, then
        # stop only after the requested checkpoint step.  This makes the HDF5
        # output and restart metadata mutually consistent on resume.
        original_step = md._do_integrator_step

        def stop_at_checkpoint(step, *step_args, **step_kwargs):
            result = original_step(step, *step_args, **step_kwargs)
            if step + 1 >= args.stop_at:
                md._terminate_run = True
            return result

        md._do_integrator_step = stop_at_checkpoint

    md.run(molecule, args.steps, reuse_P=True, remove_com=None, seed=args.seed)
    if args.stop_at is not None:
        # The controlled early exit occurs just after the force evaluation;
        # append that checkpoint geometry explicitly so a resumed HDF5/XYZ
        # trajectory has no missing frame at the join.
        with torch.no_grad():
            kinetic = md._kinetic_energy(molecule)
            temperature = md._calc_temperature(kinetic)
            potential = md._thermo_potential(molecule)
            if md._do_h5:
                writer = HDF5Writer(md.output_config, md.seqm_parameters, md.timestep)
                writer.open(molecule, str(prefix), args.steps, 3, resume=True, step_offset=args.stop_at)
                writer.append_data(args.stop_at, molecule, temperature, kinetic, potential, molecule.e_gap)
                writer.append_vectors(args.stop_at, molecule)
                writer.close()
            if md._do_xyz and args.stop_at % md.output_config.xyz_every == 0:
                writer = XYZWriter(md.output_config, step_offset=args.stop_at)
                writer.open()
                writer.write(args.stop_at - 1, molecule, kinetic, potential)
                writer.close()
        md.save_checkpoint(
            molecule, args.steps, reuse_P=True, remove_com=None, step_done=args.stop_at, path=str(restart)
        )
        print(f"Restart saved to {restart}")

    print("Final XL-ESMD diagnostics:")
    for name, value in molecule.xlesmd_diagnostics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()

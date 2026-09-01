#!/usr/bin/env python3
"""Run an MNDO-type SP CIS calculation and export leading transition-density cubes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct execution from a source checkout without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seqm.io.xyz import read_xyz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xyz", type=Path, help="Input XYZ geometry.")
    parser.add_argument("--method", choices=("MNDO", "AM1", "PM3"), default="AM1", help="NDDO SP method.")
    parser.add_argument("--out-dir", type=Path, default=Path("cis_transition_density_cubes"))
    parser.add_argument("--n-states", type=int, default=10, help="Number of CIS roots to calculate.")
    parser.add_argument(
        "--cube-states", type=int, default=3, help="Number of leading roots to export as cubes."
    )
    parser.add_argument("--spacing", type=float, default=0.20, help="Cube spacing in Angstrom.")
    parser.add_argument("--padding", type=float, default=3.0, help="Cube padding in Angstrom.")
    parser.add_argument("--chunk-size", type=int, default=20_000, help="Grid points evaluated per chunk.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cube_states > args.n_states:
        raise SystemExit("--cube-states must not exceed --n-states")

    import torch

    from seqm.ElectronicStructure import Electronic_Structure
    from seqm.io.cube import write_cis_transition_density_cubes
    from seqm.Molecule import Molecule
    from seqm.seqm_functions.constants import Constants

    torch.set_default_dtype(torch.float64)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    species_np, coordinates_np = read_xyz([str(args.xyz)], sort=True)
    species = torch.as_tensor(species_np, dtype=torch.int64, device=device)
    coordinates = torch.as_tensor(coordinates_np, dtype=torch.float64, device=device)
    parameters = {
        "method": args.method,
        "scf_eps": 1.0e-8,
        "scf_converger": [1],
        "excited_states": {
            "n_states": args.n_states,
            "cis_tol": 1.0e-6,
            "method": "cis",
            "compute_transition_properties": True,
        },
        "active_state": 1,
    }
    molecule = Molecule(Constants().to(device), parameters, coordinates, species).to(device)
    molecule.verbose = False
    Electronic_Structure(parameters).to(device)(molecule, do_force=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cube_files = write_cis_transition_density_cubes(
        molecule,
        args.out_dir,
        n_states=args.cube_states,
        spacing_angstrom=args.spacing,
        padding_angstrom=args.padding,
        chunk_size=args.chunk_size,
    )
    table = args.out_dir / "cis_energies.csv"
    with table.open("w") as handle:
        handle.write("state,excitation_energy_eV,oscillator_strength\n")
        for state, (energy, strength) in enumerate(
            zip(molecule.cis_energies[0].detach().cpu(), molecule.oscillator_strength[0].detach().cpu()),
            start=1,
        ):
            handle.write(f"{state},{float(energy):.10f},{float(strength):.10f}\n")
    print(f"Wrote {table}")
    for cube_file in cube_files:
        print(f"Wrote {cube_file}")


if __name__ == "__main__":
    main()

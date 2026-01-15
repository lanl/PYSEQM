"""
Example: run nonadiabatic dynamics (FSSH) for a batch of identical molecules.

Usage:
  python examples/vik_batch_nonadiabatic.py --xyz examples/benzene.xyz --ntraj 8
"""

import argparse

import torch

from seqm.api import Constants, Molecule, SurfaceHoppingDynamics, read_xyz


def build_molecule(xyz_path: str, ntraj: int, device):
    # Load a single geometry, then repeat it ntraj times to form a batch
    species, coords = read_xyz([xyz_path])
    species = torch.as_tensor(species, dtype=torch.int64, device=device)
    coords = torch.as_tensor(coords, dtype=torch.get_default_dtype(), device=device)
    species = species.repeat(ntraj, 1)
    coords = coords.repeat(ntraj, 1, 1)

    const = Constants().to(device)
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-8,
        "scf_converger": [1],
        "excited_states": {"n_states": 5, "method": "cis", "tolerance": 1e-6, "make_best_guess": True},
        "active_state": 3,
        "analytical_gradient": [True],
        "nonadiabatic": {"compute_nac": True},
    }
    mol = Molecule(const, seqm_parameters, coords, species).to(device)
    return mol


def main():
    parser = argparse.ArgumentParser(description="Batched nonadiabatic (FSSH) dynamics example.")
    parser.add_argument("--xyz", required=True, help="Path to a single-molecule XYZ file (will be batched).")
    parser.add_argument("--ntraj", type=int, default=4, help="Number of trajectories (batch size).")
    parser.add_argument("--steps", type=int, default=200, help="MD steps.")
    parser.add_argument("--timestep", type=float, default=0.25, help="Timestep (fs).")
    parser.add_argument("--elec-substeps", type=int, default=10, help="Electronic substeps per nuclear step.")
    parser.add_argument("--temp", type=float, default=300.0, help="Inital temperature (K).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for velocities/hops.")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    mol = build_molecule(args.xyz, args.ntraj, device)

    output = {
        "molid": list(range(args.ntraj)),
        "prefix": "./examples/Outputs/batch_fssh",
        "print every": 1,
        "checkpoint every": 50,
        "xyz": 0,
        "h5": {"data": 10, "coordinates": 10, "velocities": 10, "forces": 10},
    }

    dyn = SurfaceHoppingDynamics(
        mol.seqm_parameters,
        timestep=args.timestep,
        Temp=args.temp,
        electronic_substeps=args.elec_substeps,
        output=output,
    ).to(device)

    # Initialize velocities and run
    dyn.run(mol, steps=args.steps, reuse_P=True, remove_com=None)


if __name__ == "__main__":
    main()

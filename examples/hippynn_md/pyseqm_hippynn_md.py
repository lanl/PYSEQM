#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hippynn.experiment.serialization import load_checkpoint_from_cwd
from hippynn.graphs import Predictor
from hippynn.graphs.nodes.inputs import PositionsNode
from hippynn.graphs.nodes.physics.derivatives import GradientNode
from hippynn.tools import active_directory

from seqm.api import read_xyz
from seqm.MolecularDynamics import Molecular_Dynamics_Basic, Molecular_Dynamics_Langevin
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants


def patch_pickle_for_hippynn():
    main = sys.modules.get("__main__")
    if main is None:
        return

    def _fallback(name):
        stub = type(
            name,
            (),
            {
                "__init__": lambda self, *a, **kw: None,
                "__setstate__": lambda self, state: (
                    self.__dict__.update(state) if isinstance(state, dict) else None
                ),
                "__getattr__": lambda self, attr: None,
            },
        )
        setattr(main, name, stub)
        return stub

    main.__getattr__ = _fallback


def write_final_xyz(path, molecule):
    species = molecule.species.detach().cpu()
    coordinates = molecule.coordinates.detach().cpu()
    natoms = int(torch.sum(species[0] > 0))
    lines = [str(natoms), "final geometry from pyseqm_hippynn_md.py"]
    for z, xyz in zip(species[0, :natoms], coordinates[0, :natoms]):
        symbol = molecule.const.label[int(z.item())]
        lines.append(f"{symbol:<2} {float(xyz[0]): .10f} {float(xyz[1]): .10f} {float(xyz[2]): .10f}")
    Path(path).write_text("\n".join(lines) + "\n")


def coerce(value):
    v = value.strip()
    if v.lower() in ("true", "yes", "on"):
        return True
    if v.lower() in ("false", "no", "off"):
        return False
    try:
        if "." not in v and "e" not in v.lower():
            return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def parse_mdip(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    config = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line[0] in ("#", "!"):
            continue
        if line.upper().startswith(("%BLOCK", "%END")):
            continue
        for ch in ("#", "!"):
            idx = line.find(ch)
            if idx > 0:
                line = line[:idx].strip()
        if not line:
            continue

        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts

        key = key.strip().lower()
        config[key] = coerce(value)
    return config


def build_parser():
    parser = argparse.ArgumentParser(description="Run pyseqm MD using a direct HIPPYNN energy/force driver.")
    parser.add_argument("--infile", "-i", default=None, help="Path to .mdip/.input file.")
    parser.add_argument("--xyz-file", default=None, help="Path to the input XYZ geometry.")
    parser.add_argument("--model-dir", default=None, help="Path to the HIPPYNN checkpoint directory.")
    parser.add_argument(
        "--state-index", type=int, default=None, help="Electronic state index (0=S0, 1=S1, ...)."
    )
    parser.add_argument("--energy-offset", type=float, default=None, help="Constant energy shift in eV.")
    parser.add_argument("--ensemble", choices=["nve", "nvt"], default=None, help="MD ensemble.")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature in K.")
    parser.add_argument("--timestep", type=float, default=None, help="Timestep in fs.")
    parser.add_argument("--nsteps", type=int, default=None, help="Production steps.")
    parser.add_argument("--friction", type=float, default=None, help="Langevin friction in 1/fs.")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--output-prefix", default=None, help="Output file prefix.")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory.")
    parser.add_argument("--trajectory-interval", type=int, default=10, help="Trajectory write interval.")
    parser.add_argument("--log-interval", type=int, default=10, help="Thermo print/HDF5 interval.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    return parser


class HIPPYNN_ESDriver(torch.nn.Module):
    def __init__(self, model_dir, state_index=0, energy_offset=0.0, device="auto"):
        super().__init__()
        patch_pickle_for_hippynn()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.energy_offset = float(energy_offset)

        with active_directory(model_dir):
            bundle = load_checkpoint_from_cwd(map_location="cpu")

        model = bundle["training_modules"].model
        model.eval()

        energy_node = None
        target_name = f"sE{state_index}"
        for node in model.nodes_to_compute:
            if getattr(node, "db_name", None) == target_name:
                energy_node = node
                break
        if energy_node is None:
            candidates = [n for n in model.nodes_to_compute if "mol_energy" in str(type(n)).lower()]
            if not candidates:
                raise ValueError(f"Could not find an energy node for state {state_index}.")
            energy_node = candidates[0]

        positions_node = next(n for n in model.input_nodes if isinstance(n, PositionsNode))
        force_node = GradientNode(f"forces_s{state_index}", (energy_node, positions_node), sign=-1)

        predictor = Predictor.from_graph(
            model,
            additional_outputs=[energy_node, force_node],
            model_device=self.device,
            return_device=self.device,
            requires_grad=False,
        )

        self.graph = predictor.graph
        self.input_db_names = [node.db_name for node in self.graph.input_nodes]
        self.energy_key = getattr(energy_node, "db_name", energy_node.name)
        self.force_key = force_node.name
        self.graph.to(device=self.device, dtype=torch.float64)

    def forward(self, molecule, *args, **kwargs):
        inputs = []
        for db_name in self.input_db_names:
            if db_name == "Z":
                inputs.append(molecule.species.to(self.device))
            elif db_name == "R":
                inputs.append(molecule.coordinates.to(self.device, dtype=torch.float64))
            else:
                raise ValueError(f"Unsupported HIPNN graph input '{db_name}'.")

        out_values = self.graph(*inputs)
        out = {}
        for node, value in zip(self.graph.nodes_to_compute, out_values):
            out[node.name] = value
            if getattr(node, "db_name", None) is not None:
                out[node.db_name] = value

        energy = out[self.energy_key].reshape(-1) + self.energy_offset
        force = out[self.force_key]

        dtype = molecule.coordinates.dtype
        device = molecule.coordinates.device
        nmol = molecule.coordinates.shape[0]

        molecule.Etot = energy.detach().to(device=device, dtype=dtype)
        molecule.force = force.detach().to(device=device, dtype=dtype)
        molecule.Hf = molecule.Etot.clone()
        molecule.Eelec = molecule.Etot.clone()
        molecule.Enuc = torch.zeros_like(molecule.Etot)
        molecule.Eiso = torch.zeros_like(molecule.Etot)
        molecule.e_gap = torch.full((nmol,), float("nan"), dtype=dtype, device=device)
        molecule.dipole = torch.zeros((nmol, 3), dtype=dtype, device=device)
        molecule.dm = None
        molecule.cis_amplitudes = None
        return molecule.force, molecule.Etot


class HIPNN_NVE(Molecular_Dynamics_Basic):
    def __init__(self, model_dir, state_index=0, energy_offset=0.0, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.esdriver = HIPPYNN_ESDriver(model_dir, state_index, energy_offset, device)


class HIPNN_NVT(Molecular_Dynamics_Langevin):
    def __init__(self, model_dir, state_index=0, energy_offset=0.0, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.esdriver = HIPPYNN_ESDriver(model_dir, state_index, energy_offset, device)


def make_output_config(cfg, prefix_path):
    return {
        "molid": [0],
        "prefix": str(prefix_path),
        "print every": int(cfg["log_interval"]),
        "checkpoint every": 0,
        "xyz": int(cfg["trajectory_interval"]),
        "h5": {
            "data": int(cfg["log_interval"]),
            "coordinates": int(cfg["trajectory_interval"]),
            "velocities": int(cfg["trajectory_interval"]),
            "forces": int(cfg["trajectory_interval"]),
        },
    }


def build_molecule(xyz_file, device):
    species_np, coordinates_np = read_xyz([str(Path(xyz_file).expanduser().resolve())])
    species = torch.tensor(species_np, dtype=torch.int64, device=device)
    coordinates = torch.tensor(coordinates_np, dtype=torch.float64, device=device)

    # Dummy SEQM settings. HIPPYNN supplies the energies and forces; these only
    # satisfy Molecule/MD construction and should never matter physically.
    seqm_parameters = {"method": "AM1", "scf_eps": 123.456, "scf_converger": [-1]}
    const = Constants().to(device)
    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    return molecule, seqm_parameters


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--infile", "-i", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = build_parser()
    if pre_args.infile:
        parser.set_defaults(**parse_mdip(pre_args.infile))
    args = parser.parse_args()
    cfg = vars(args)

    required = [
        "model_dir",
        "xyz_file",
        "state_index",
        "energy_offset",
        "ensemble",
        "temperature",
        "timestep",
        "nsteps",
        "random_seed",
        "output_prefix",
    ]
    missing = [key for key in required if cfg.get(key) is None]
    if cfg.get("ensemble") == "nvt" and cfg.get("friction") is None:
        missing.append("friction")
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")

    if cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])

    output_dir = Path(cfg["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = output_dir / cfg["output_prefix"]

    molecule, seqm_parameters = build_molecule(cfg["xyz_file"], device)

    print(f"Model dir: {Path(cfg['model_dir']).expanduser()}")
    print(f"XYZ file:  {Path(cfg['xyz_file']).expanduser()}")
    print(f"Device:    {device}")
    print(f"Ensemble:  {cfg['ensemble']}")
    print(f"State:     S{cfg['state_index']}")
    print(f"Offset:    {cfg['energy_offset']:.12f} eV")

    step_offset = 0
    seed = int(cfg["random_seed"])
    total_steps = int(cfg["nsteps"])

    output = make_output_config(cfg, prefix_path)

    if cfg["ensemble"] == "nve":
        md = HIPNN_NVE(
            model_dir=cfg["model_dir"],
            state_index=int(cfg["state_index"]),
            energy_offset=float(cfg["energy_offset"]),
            device=str(device),
            seqm_parameters=seqm_parameters,
            timestep=float(cfg["timestep"]),
            Temp=float(cfg["temperature"]),
            step_offset=step_offset,
            output=output,
        ).to(device)
    else:
        damp = 1.0 / float(cfg["friction"])
        md = HIPNN_NVT(
            model_dir=cfg["model_dir"],
            state_index=int(cfg["state_index"]),
            energy_offset=float(cfg["energy_offset"]),
            device=str(device),
            damp=damp,
            seqm_parameters=seqm_parameters,
            timestep=float(cfg["timestep"]),
            Temp=float(cfg["temperature"]),
            step_offset=step_offset,
            output=output,
        ).to(device)

    md.run(molecule, steps=total_steps, reuse_P=False, remove_com=None, seed=seed)

    final_xyz = output_dir / f"{cfg['output_prefix']}_final.xyz"
    write_final_xyz(final_xyz, molecule)
    print(f"Final geometry written to {final_xyz}")


if __name__ == "__main__":
    main()

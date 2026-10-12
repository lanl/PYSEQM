import os
import tempfile

import torch

from ..ElectronicStructure import Electronic_Structure as esdriver


def write_trajectory(path, step, molecule, comment=None):
    species, coordinates = molecule.species[0], molecule.coordinates[0]
    comment = comment or f"E = {molecule.Etot[0]:.8f} eV"
    with open(path, "a") as out:
        out.write(f"{species.shape[0]}\nStep {step}, {comment}\n")
        for atom, xyz in zip(species, coordinates):
            out.write(f"{molecule.const.label[atom.item()]} {xyz[0]:15.5f} {xyz[1]:15.5f} {xyz[2]:15.5f}\n")


def build_geomol(molecule):
    from geometric.molecule import Elements
    from geometric.molecule import Molecule as GeomMolecule

    geomol = GeomMolecule()
    geomol.elem = [Elements[z] for z in molecule.species.cpu().numpy()[0]]
    geomol.xyzs = [molecule.coordinates.detach().cpu().numpy()[0]]
    return geomol


def sync_coordinates(molecule, coords):
    from geometric.nifty import bohr2ang

    xyz = torch.as_tensor(
        coords.reshape(-1, 3), device=molecule.coordinates.device, dtype=molecule.coordinates.dtype
    ).mul(bohr2ang)
    with torch.no_grad():
        molecule.coordinates = torch.nn.Parameter(xyz[None])
    return xyz


def run_geometric(
    molecule,
    parameters,
    evaluate,
    traj_file,
    output_file,
    run_kwargs,
    *,
    integrate_gradient=False,
    meci=False,
):
    if molecule.nmol != 1:
        raise ValueError("geomeTRIC optimization supports exactly one molecule.")
    try:
        from geometric.engine import Engine
        from geometric.nifty import bohr2ang, ev2au
        from geometric.optimize import run_optimizer
    except ImportError as error:
        raise ImportError("Install the geomeTRIC integration with `pip install '.[geomopt]'`.") from error

    traj_file = os.path.abspath(traj_file)
    output_file = os.path.abspath(output_file)

    class PYSEQMEngine(Engine):
        def __init__(self):
            super().__init__(build_geomol(molecule))
            self.driver = esdriver(parameters).to(molecule.coordinates.device)
            self.step = 0
            self.last_xyz = self.last_gradient = self.integrated_energy = self.accepted_state = None

        def sync(self, coords):
            return sync_coordinates(molecule, coords)

        def save_guess_files(self, dirname):
            if integrate_gradient and self.integrated_energy is not None:
                self.accepted_state = tuple(
                    value.clone() for value in (self.last_xyz, self.last_gradient, self.integrated_energy)
                )

        def load_guess_files(self, dirname):
            if integrate_gradient and self.accepted_state is not None:
                self.last_xyz, self.last_gradient, self.integrated_energy = (
                    value.clone() for value in self.accepted_state
                )

        def calc(self, coords, dirname, *args, **kwargs):
            xyz = self.sync(coords)
            result = super().calc(coords, dirname, *args, **kwargs)
            if integrate_gradient:
                self.last_xyz = xyz
                self.last_gradient = torch.as_tensor(
                    result["gradient"].reshape(-1, 3), device=xyz.device, dtype=xyz.dtype
                ).div(ev2au * bohr2ang)
                self.integrated_energy = torch.as_tensor(
                    result["energy"] / ev2au, device=xyz.device, dtype=xyz.dtype
                )
            return result

        def calc_new(self, coords, dirname=None):
            energy, gradient, comment = evaluate(self.driver)
            if integrate_gradient:
                xyz = molecule.coordinates[0]
                if self.integrated_energy is None:
                    self.integrated_energy = energy.detach()
                else:
                    displacement = xyz - self.last_xyz
                    self.integrated_energy += 0.5 * torch.sum((self.last_gradient + gradient) * displacement)
                energy = self.integrated_energy
                self.last_xyz, self.last_gradient = xyz.detach(), gradient.detach()
                if self.accepted_state is None:
                    self.save_guess_files(dirname)
            self.step += 1
            write_trajectory(traj_file, self.step, molecule, comment)
            return {
                "energy": float(energy) * ev2au,
                "gradient": (gradient.detach().cpu().numpy() * ev2au * bohr2ang).reshape(-1),
            }

    kwargs = dict(run_kwargs)
    if meci:
        if "meci" in kwargs:
            raise ValueError("meci is managed by geomeTRIC_mecp; use its method argument instead.")
        kwargs["meci"] = ["engine"]
    kwargs.setdefault("qccnv", True)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_optimizer(
            customengine=PYSEQMEngine(), input=os.path.join(tmpdir, "geometric.in"), **kwargs
        )
    with open(output_file, "w") as out:
        out.write("\n".join(result.write_xyz(selection=[-1])))
    return result

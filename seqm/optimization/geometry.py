import os
import tempfile
import uuid

import torch

from ..ElectronicStructure import Electronic_Structure as esdriver


def _write_trajectory(path, step, molecule):
    """Write a multi‐frame .xyz"""
    with open(path, "a") as out:
        out.write(f"{molecule.species.shape[1]}\nStep {step}, E = {molecule.Etot[0]:.8f} eV\n")
        for atom in range(molecule.coordinates.shape[1]):
            out.write(
                f"{molecule.const.label[molecule.species[0, atom].item()]} {molecule.coordinates[0, atom, 0]:15.5f} {molecule.coordinates[0, atom, 1]:15.5f} {molecule.coordinates[0, atom, 2]:15.5f}\n"
            )


# ─── User API ────────────────────────────────────────────────────────────────
def geomeTRIC_optimization(molecule, traj_file: str = "geom_opt_traj.xyz", **run_kwargs):
    """
    Optimize geometry using geomeTRIC

    Args:
      molecule - PYSEQM Molecule object
      traj_file  – optional output .xyz for all steps
      **run_kwargs – passed directly to geomeTRIC’s run_optimizer
                     e.g. maxsteps=100, check=1, etc.

    """
    try:
        from geometric.engine import Engine
        from geometric.molecule import Elements
        from geometric.molecule import Molecule as GeomMolecule
        from geometric.nifty import bohr2ang, ev2au
        from geometric.optimize import run_optimizer
    except ImportError as e:
        raise ImportError(
            "To use geometry-optimization with geomeTRIC, install the optional "
            "dependency via `pip install '.[geomopt]'`."
        ) from e

    class pyseqm_engine(Engine):
        """
        geomeTRIC Engine that calls PYSEQM’s Electronic_Structure
        """

        def __init__(self, molecule, traj_file, learned_parameters=dict()):
            self.molecule = molecule

            geomol = GeomMolecule()
            geomol.elem = [Elements[z] for z in self.molecule.species.cpu().numpy()[0]]
            geomol.xyzs = [self.molecule.coordinates.clone().detach().cpu().numpy()[0]]
            super().__init__(geomol)

            self.learned_parameters = learned_parameters
            self.esdriver = esdriver(self.molecule.seqm_parameters).to(self.molecule.coordinates.device)
            self.step = 0
            self.traj_file = traj_file

        def calc_new(self, coords, dirname=None):
            # 1) Bohr → Å tensor
            xyz_ang = torch.tensor(
                coords.reshape(-1, 3) * bohr2ang,
                device=self.molecule.coordinates.device,
                dtype=self.molecule.coordinates.dtype,
            ).unsqueeze(0)

            with torch.no_grad():
                self.molecule.coordinates = torch.nn.Parameter(xyz_ang)

            # 2) run SCF
            self.esdriver(
                self.molecule, P0=self.molecule.dm, dm_prop="SCF", learned_parameters=self.learned_parameters
            )

            # 3) Energy & gradient → atomic units
            E_h = self.molecule.Etot.item() * ev2au
            grad_eV_A = -self.molecule.force.detach().cpu().numpy()[0]  # (N,3)
            grad_ha = (grad_eV_A * ev2au * bohr2ang).reshape(-1)
            self.step += 1

            # 4) write trajectory
            _write_trajectory(self.traj_file, self.step, self.molecule)

            return {"energy": E_h, "gradient": grad_ha}

    if molecule.nmol != 1:
        raise ValueError(
            "Geometry optimization with geomeTRIC does not work in batch mode, i.e., with inputs of more than one molecule. Please input only one molecule"
        )

    # 3) Run geomeTRIC
    engine = pyseqm_engine(molecule, traj_file)
    # Making a temp directory and a temp file because geomeTRIC uses this for logging but I dont want these files saved
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpf = os.path.join(tmpdir, str(uuid.uuid4()))

        result = run_optimizer(
            customengine=engine, input=tmpf, qccnv=True, **run_kwargs
        )  # use Q-Chem like convergence

    with open("optimized.xyz", "w") as f:
        f.write(
            "\n".join(result.write_xyz(selection=[-1]))
        )  # Write the geometry at the last step of optimization as the optimized geometry

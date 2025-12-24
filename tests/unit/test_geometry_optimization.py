import os
from pathlib import Path

import pytest
import torch

from seqm.seqm_functions.constants import Constants
from seqm.Molecule import Molecule
from seqm.seqm_functions.read_xyz import read_xyz
from tests.reference_data import assert_allclose, load_or_update_reference, reference_path


def _read_xyz_coords(path, device):
    species, coordinates = read_xyz([os.fspath(path)])
    species = torch.as_tensor(species, dtype=torch.int64, device=device)
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64, device=device)
    return species, coordinates


def test_geometry_optimization_matches_reference(device, repo_root, monkeypatch, tmp_path):
    if os.environ.get("PYSEQM_RUN_GEOMOPT") != "1":
        pytest.skip("Set PYSEQM_RUN_GEOMOPT=1 to run geomeTRIC optimization test.")
    # if os.environ.get("KMP_DUPLICATE_LIB_OK") != "TRUE":
    #     pytest.skip("Set KMP_DUPLICATE_LIB_OK=TRUE to run geomeTRIC optimization test.")
    pytest.importorskip("geometric")
    from seqm.geometryOptimization import geomeTRIC_optimization

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    methane_path = repo_root / "tests" / "data" / "methane.xyz"
    species, coordinates = _read_xyz_coords(methane_path, device)

    const = Constants().to(device)
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-8,
        "scf_converger": [0, 0.1],
        "analytical_gradient": [True],
    }

    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)

    traj_file = tmp_path / "geom_opt_traj.xyz"
    monkeypatch.chdir(tmp_path)
    geomeTRIC_optimization(molecule, traj_file=os.fspath(traj_file))

    optimized_path = Path("optimized.xyz")
    assert optimized_path.exists()

    _, optimized_coords = _read_xyz_coords(optimized_path, device)
    data = {
        "optimized_coords": optimized_coords.detach().cpu().tolist(),
    }
    ref_path = reference_path("geom_opt_methane_am1")
    ref = load_or_update_reference(ref_path, data)

    assert_allclose(data["optimized_coords"], ref["optimized_coords"], rtol=1e-5, atol=1e-5)

    optimized_path.unlink(missing_ok=True)
    traj_file.unlink(missing_ok=True)

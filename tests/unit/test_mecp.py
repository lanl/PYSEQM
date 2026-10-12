import os

import pytest
import torch

from seqm.api import geomeTRIC_mecp
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants
from seqm.seqm_functions.read_xyz import read_xyz


@pytest.mark.parametrize("mecp_method", ["gradient_projection", "composed_step", "penalty"])
def test_mecp_optimization_closes_gap(device, repo_root, monkeypatch, tmp_path, mecp_method):
    if os.environ.get("PYSEQM_RUN_GEOMOPT") != "1":
        pytest.skip("Set PYSEQM_RUN_GEOMOPT=1 to run geomeTRIC optimization tests.")
    pytest.importorskip("geometric")
    torch.set_default_dtype(torch.float64)
    species, coordinates = read_xyz([os.fspath(repo_root / "tests" / "data" / "methanal.1.xyz")])
    parameters = {
        "method": "AM1",
        "scf_eps": 1e-8,
        "scf_converger": [1],
        "excited_states": {"method": "cis", "n_states": 2, "tolerance": 1e-7},
    }
    molecule = Molecule(
        Constants().to(device),
        parameters,
        torch.as_tensor(coordinates, device=device),
        torch.as_tensor(species, dtype=torch.int64, device=device),
    )
    molecule.verbose = False
    monkeypatch.chdir(tmp_path)

    geomeTRIC_mecp(molecule, method=mecp_method, maxiter=80, verbose=0)

    assert molecule.mecp_energy_gap < 0.02
    assert (molecule.nac is None) == (mecp_method == "penalty")
    assert (tmp_path / "mecp.xyz").exists()
    assert "gap =" in (tmp_path / "mecp_opt_traj.xyz").read_text()

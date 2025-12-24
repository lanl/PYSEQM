import pytest
import torch

from seqm.seqm_functions.constants import Constants
from seqm.Molecule import Molecule
from seqm.ElectronicStructure import Electronic_Structure
from tests.reference_data import assert_allclose, load_or_update_reference, reference_path


@pytest.mark.parametrize("method", ["MNDO", "AM1", "PM3", "PM6", "PM6_SP"])
def test_single_point_runs_for_all_methods(method, device, methane_molecule_data):
    species, coordinates = methane_molecule_data
    const = Constants().to(device)

    seqm_parameters = {
        "method": method,
        "scf_eps": 1.0e-6,
        "scf_converger": [1],
    }

    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)

    esdriver(molecule)

    assert torch.isfinite(molecule.Etot).all()
    assert torch.isfinite(molecule.Eelec).all()
    assert torch.isfinite(molecule.Enuc).all()
    assert molecule.force is not None
    assert molecule.force.shape == coordinates.shape

    data = {
        "Etot": float(molecule.Etot.item()),
        "Eelec": float(molecule.Eelec.item()),
        "Enuc": float(molecule.Enuc.item()),
        "force": molecule.force.detach().cpu().tolist(),
    }
    ref_path = reference_path(f"smoke_single_point_{method}")
    ref = load_or_update_reference(ref_path, data)

    assert_allclose(data["Etot"], ref["Etot"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["Eelec"], ref["Eelec"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["Enuc"], ref["Enuc"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)

import pytest
import torch

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

from ..reference_data import assert_allclose, load_or_update_reference, reference_path


def _ch3_data(device):
    species = torch.as_tensor([[6, 1, 1, 1]], dtype=torch.int64, device=device)
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-0.5, 0.9, 0.0], [-0.5, -0.9, 0.0]]],
        dtype=torch.float64,
        device=device,
    )
    return species, coordinates


def test_rhf_rejects_odd_electron_counts(device):
    species, coordinates = _ch3_data(device)
    const = Constants().to(device)

    seqm_parameters = {"method": "AM1", "scf_eps": 1.0e-6, "scf_converger": [1]}

    with pytest.raises(ValueError) as excinfo:
        Molecule(const, seqm_parameters, coordinates, species).to(device)

    message = str(excinfo.value)
    ref_path = reference_path("rhf_odd_electron_error")
    ref = load_or_update_reference(ref_path, {"message": message})
    assert message == ref["message"]


def test_uhf_allows_odd_electron_counts(device):
    species, coordinates = _ch3_data(device)
    const = Constants().to(device)

    seqm_parameters = {"method": "AM1", "scf_eps": 1.0e-6, "scf_converger": [1], "UHF": True}

    charges = torch.tensor([0], dtype=torch.int64, device=device)
    mult = torch.tensor([2], dtype=torch.int64, device=device)

    molecule = Molecule(const, seqm_parameters, coordinates, species, charges=charges, mult=mult).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecule)

    data = {"Etot": float(molecule.Etot.item()), "dm_shape": list(molecule.dm.shape)}
    ref_path = reference_path("uhf_ch3_am1")
    ref = load_or_update_reference(ref_path, data)

    assert molecule.dm is not None
    assert molecule.dm.dim() == 4

    assert_allclose(data["Etot"], ref["Etot"], rtol=1e-5, atol=1e-5)
    assert data["dm_shape"] == ref["dm_shape"]

import pytest
import torch

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

from ..reference_data import (
    AM1_AND_OM2_METHODS,
    assert_allclose,
    load_or_update_reference,
    reference_path_for_method,
)


@pytest.mark.parametrize("method", AM1_AND_OM2_METHODS)
def test_batch_single_point(device, batch_molecule_data, method):
    species, coordinates = batch_molecule_data
    const = Constants().to(device)

    seqm_parameters = {"method": method, "scf_eps": 1.0e-6, "scf_converger": [1]}

    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)

    esdriver(molecule)

    assert torch.isfinite(molecule.Etot).all()
    assert molecule.force is not None
    assert molecule.force.shape == coordinates.shape

    data = {"Etot": molecule.Etot.detach().cpu().tolist(), "force": molecule.force.detach().cpu().tolist()}
    ref_path = reference_path_for_method("batch_single_point", method, am1_name="batch_single_point_am1")
    ref = load_or_update_reference(ref_path, data)

    assert_allclose(data["Etot"], ref["Etot"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["force"], ref["force"], rtol=1e-4, atol=1e-4)

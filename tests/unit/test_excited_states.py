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
@pytest.mark.parametrize("excited_method", ["cis", "rpa"])
def test_excited_states_single_molecule(device, methane_molecule_data, excited_method, method):
    species, coordinates = methane_molecule_data
    const = Constants().to(device)

    n_states = 4
    seqm_parameters = {
        "method": method,
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "excited_states": {"n_states": n_states, "method": excited_method},
        "active_state": 1,
    }

    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecule)

    assert molecule.cis_energies is not None
    assert molecule.cis_energies.shape[1] == n_states
    assert torch.isfinite(molecule.cis_energies).all()
    assert molecule.oscillator_strength is not None
    assert torch.isfinite(molecule.oscillator_strength).all()

    data = {
        "excited_method": excited_method,
        "n_states": n_states,
        "cis_energies": molecule.cis_energies.detach().cpu().tolist(),
        "oscillator_strength": molecule.oscillator_strength.detach().cpu().tolist(),
        "force": molecule.force.detach().cpu().tolist(),
    }

    ref_path = reference_path_for_method(
        f"excited_{excited_method}_methane", method, am1_name=f"excited_{excited_method}_am1_methane"
    )
    ref = load_or_update_reference(ref_path, data)

    assert_allclose(data["cis_energies"], ref["cis_energies"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["oscillator_strength"], ref["oscillator_strength"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", AM1_AND_OM2_METHODS)
def test_cis_batch_same_molecule_different_coords(device, methanal_batch_data, method):
    species, coordinates = methanal_batch_data
    const = Constants().to(device)

    n_states = 4
    seqm_parameters = {
        "method": method,
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "excited_states": {"n_states": n_states, "method": "cis"},
        "active_state": 1,
    }

    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecule)

    assert molecule.cis_energies is not None
    assert molecule.cis_energies.shape[1] == n_states
    assert torch.isfinite(molecule.cis_energies).all()
    assert molecule.oscillator_strength is not None
    assert torch.isfinite(molecule.oscillator_strength).all()

    data = {
        "cis_energies": molecule.cis_energies.detach().cpu().tolist(),
        "oscillator_strength": molecule.oscillator_strength.detach().cpu().tolist(),
        "force": molecule.force.detach().cpu().tolist(),
    }
    ref_path = reference_path_for_method(
        "cis_batch_same_molecule_coords", method, am1_name="cis_batch_same_molecule_coords"
    )
    ref = load_or_update_reference(ref_path, data)

    assert_allclose(data["cis_energies"], ref["cis_energies"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["oscillator_strength"], ref["oscillator_strength"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", AM1_AND_OM2_METHODS)
def test_cis_batch_different_molecules(device, excited_mixed_batch_data, method):
    species, coordinates = excited_mixed_batch_data
    const = Constants().to(device)

    n_states = 4
    seqm_parameters = {
        "method": method,
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "excited_states": {"n_states": n_states, "method": "cis"},
        "active_state": 1,
        "scf_backward": 1,
    }

    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecule)

    assert molecule.cis_energies is not None
    assert molecule.cis_energies.shape[1] == n_states
    assert torch.isfinite(molecule.cis_energies).all()
    assert molecule.oscillator_strength is not None
    assert torch.isfinite(molecule.oscillator_strength).all()

    data = {
        "cis_energies": molecule.cis_energies.detach().cpu().tolist(),
        "oscillator_strength": molecule.oscillator_strength.detach().cpu().tolist(),
        "force": molecule.force.detach().cpu().tolist(),
    }
    ref_path = reference_path_for_method(
        "cis_batch_different_molecules", method, am1_name="cis_batch_different_molecules"
    )
    ref = load_or_update_reference(ref_path, data)

    assert_allclose(data["cis_energies"], ref["cis_energies"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["oscillator_strength"], ref["oscillator_strength"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", AM1_AND_OM2_METHODS)
def test_rpa_batch_same_molecule_different_coords(device, methanal_batch_data, method):
    species, coordinates = methanal_batch_data
    const = Constants().to(device)

    n_states = 4
    seqm_parameters = {
        "method": method,
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "excited_states": {"n_states": n_states, "method": "rpa"},
        "active_state": 1,
    }

    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecule)

    assert molecule.cis_energies is not None
    assert molecule.cis_energies.shape[1] == n_states
    assert torch.isfinite(molecule.cis_energies).all()
    assert molecule.oscillator_strength is not None
    assert torch.isfinite(molecule.oscillator_strength).all()

    data = {
        "cis_energies": molecule.cis_energies.detach().cpu().tolist(),
        "oscillator_strength": molecule.oscillator_strength.detach().cpu().tolist(),
        "force": molecule.force.detach().cpu().tolist(),
    }
    ref_path = reference_path_for_method(
        "rpa_batch_same_molecule_coords", method, am1_name="rpa_batch_same_molecule_coords"
    )
    ref = load_or_update_reference(ref_path, data)

    assert_allclose(data["cis_energies"], ref["cis_energies"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["oscillator_strength"], ref["oscillator_strength"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)

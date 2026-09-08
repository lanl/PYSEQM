import pytest
import torch

from seqm.dynamics.xlbomd import ForceXL
from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants
from tests import reference_data

from ..reference_data import OMX_METHODS, assert_allclose, load_or_update_reference, reference_path_for_method

_FORCE_MODES = [
    ("autodiff", {}),
    ("analytical", {"analytical_gradient": [True]}),
    ("semi_numerical", {"analytical_gradient": [True, "numerical"]}),
]

_EXCITED_FORCE_MODES = [("autodiff", {"scf_backward": 1}), ("analytical", {"analytical_gradient": [True]})]


def _load_reference_for_mode(path, data, is_reference_mode):
    if reference_data.UPDATE_REFERENCES and not is_reference_mode:
        ref = reference_data.load_reference(path)
        if ref is not None:
            return ref
    return load_or_update_reference(path, data)


def _run_ground_force(device, species, coordinates, method, mode_overrides):
    const = Constants().to(device)
    seqm_parameters = {"method": method, "scf_eps": 1.0e-7, "scf_converger": [1]}
    seqm_parameters.update(mode_overrides)

    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecule)

    return molecule.force.detach().cpu().tolist()


def _run_excited_force(device, species, coordinates, method, mode_overrides):
    const = Constants().to(device)
    seqm_parameters = {
        "method": method,
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "excited_states": {"n_states": 4, "method": "cis"},
        "active_state": 1,
    }
    seqm_parameters.update(mode_overrides)

    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecule)

    return molecule.force.detach().cpu().tolist()


def test_xl_analytical_force_matches_autodiff(device, methane_molecule_data):
    species, coordinates = methane_molecule_data
    base = {"method": "AM1", "scf_eps": 1.0e-9, "scf_converger": [1], "torch_compile": False}

    molecule = Molecule(Constants().to(device), base, coordinates.clone(), species).to(device)
    Electronic_Structure(base).to(device)(molecule)
    P = molecule.dm.detach().clone()
    delta = 0.01 * torch.sin(torch.arange(P.numel(), dtype=P.dtype, device=device)).reshape_as(P)
    P += 0.5 * (delta + delta.transpose(1, 2))

    forces = []
    for analytical in (False, True):
        params = dict(base, analytical_gradient=[analytical])
        molecule = Molecule(Constants().to(device), params, coordinates.clone(), species).to(device)
        forces.append(ForceXL(params).to(device)(molecule, P.clone())[0])

    assert_allclose(forces[1], forces[0], rtol=1.0e-7, atol=2.0e-8)


@pytest.mark.parametrize("method", OMX_METHODS)
@pytest.mark.parametrize("mode_name, mode_overrides", _FORCE_MODES)
def test_ground_force_methods_single_molecule(
    device, methane_molecule_data, mode_name, mode_overrides, method
):
    species, coordinates = methane_molecule_data
    force = _run_ground_force(device, species, coordinates, method, mode_overrides)

    data = {"force": force}
    ref_path = reference_path_for_method("ground_force_methane", method, am1_name="ground_force_methane")
    ref = _load_reference_for_mode(ref_path, data, mode_name == "autodiff")

    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", OMX_METHODS)
@pytest.mark.parametrize("mode_name, mode_overrides", _FORCE_MODES)
def test_ground_force_methods_batch_same_species(
    device, methanal_batch_data, mode_name, mode_overrides, method
):
    species, coordinates = methanal_batch_data
    force = _run_ground_force(device, species, coordinates, method, mode_overrides)

    data = {"force": force}
    ref_path = reference_path_for_method(
        "ground_force_batch_methanal", method, am1_name="ground_force_batch_methanal"
    )
    ref = _load_reference_for_mode(ref_path, data, mode_name == "autodiff")

    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", OMX_METHODS)
@pytest.mark.parametrize("mode_name, mode_overrides", _FORCE_MODES)
def test_ground_force_methods_batch_mixed(device, batch_molecule_data, mode_name, mode_overrides, method):
    species, coordinates = batch_molecule_data
    force = _run_ground_force(device, species, coordinates, method, mode_overrides)

    data = {"force": force}
    ref_path = reference_path_for_method(
        "ground_force_batch_mixed", method, am1_name="ground_force_batch_mixed"
    )
    ref = _load_reference_for_mode(ref_path, data, mode_name == "autodiff")

    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", OMX_METHODS)
@pytest.mark.parametrize("mode_name, mode_overrides", _EXCITED_FORCE_MODES)
def test_excited_force_methods_single_molecule(
    device, methane_molecule_data, mode_name, mode_overrides, method
):
    species, coordinates = methane_molecule_data
    force = _run_excited_force(device, species, coordinates, method, mode_overrides)

    data = {"force": force}
    ref_path = reference_path_for_method("excited_force_methane", method, am1_name="excited_force_methane")
    ref = _load_reference_for_mode(ref_path, data, mode_name == "autodiff")

    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", OMX_METHODS)
@pytest.mark.parametrize("mode_name, mode_overrides", _EXCITED_FORCE_MODES)
def test_excited_force_methods_batch_same_species(
    device, methanal_batch_data, mode_name, mode_overrides, method
):
    species, coordinates = methanal_batch_data
    force = _run_excited_force(device, species, coordinates, method, mode_overrides)

    data = {"force": force}
    ref_path = reference_path_for_method(
        "excited_force_batch_methanal", method, am1_name="excited_force_batch_methanal"
    )
    ref = _load_reference_for_mode(ref_path, data, mode_name == "autodiff")

    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)

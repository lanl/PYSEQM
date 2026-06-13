import pytest

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants
from tests import reference_data

from ..reference_data import assert_allclose, load_or_update_reference, reference_path

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


def _run_ground_force(device, species, coordinates, mode_overrides):
    const = Constants().to(device)
    seqm_parameters = {"method": "AM1", "scf_eps": 1.0e-7, "scf_converger": [1]}
    seqm_parameters.update(mode_overrides)

    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecule)

    return molecule.force.detach().cpu().tolist()


def _run_excited_force(device, species, coordinates, mode_overrides):
    const = Constants().to(device)
    seqm_parameters = {
        "method": "AM1",
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


def _run_omx(device, species, coordinates, mode_overrides, method):
    const = Constants().to(device)
    seqm_parameters = {"method": method, "scf_eps": 1.0e-7, "scf_converger": [1]}
    seqm_parameters.update(mode_overrides)

    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecule)
    return molecule


def _run_omx_ground_force(device, species, coordinates, method):
    mol = _run_omx(device, species, coordinates, {"analytical_gradient": [True, "numerical"]}, method)
    return mol.force.detach().cpu().tolist()


def _run_omx_ground_force_reference(device, species, coordinates, method):
    mol = _run_omx(device, species, coordinates, {"analytical_gradient": [False]}, method)
    return mol.force.detach().cpu().tolist()


def _assert_omx_ground_force(device, species, coordinates, method, ref_suffix):
    force = _run_omx_ground_force(device, species, coordinates, method)

    data = {"force": force}
    ref_path = reference_path(f"ground_force_{method.lower()}_{ref_suffix}")
    ref = _load_reference_for_mode(
        ref_path, {"force": _run_omx_ground_force_reference(device, species, coordinates, method)}, True
    )

    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


def _run_omx_excited_force(device, species, coordinates, method, mode_overrides):
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
    return molecule


def _run_omx_excited_force_reference(device, species, coordinates, method):
    mol = _run_omx_excited_force(device, species, coordinates, method, {"scf_backward": 1})
    return mol.force.detach().cpu().tolist()


def _assert_omx_excited_force(device, species, coordinates, method, ref_suffix):
    force = (
        _run_omx_excited_force(device, species, coordinates, method, {"analytical_gradient": [True]})
        .force.detach()
        .cpu()
        .tolist()
    )

    data = {"force": force}
    ref_path = reference_path(f"excited_force_{method.lower()}_{ref_suffix}")
    ref = _load_reference_for_mode(
        ref_path, {"force": _run_omx_excited_force_reference(device, species, coordinates, method)}, True
    )

    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode_name, mode_overrides", _FORCE_MODES)
def test_ground_force_methods_single_molecule(device, methane_molecule_data, mode_name, mode_overrides):
    species, coordinates = methane_molecule_data
    force = _run_ground_force(device, species, coordinates, mode_overrides)

    data = {"force": force}
    ref_path = reference_path("ground_force_methane")
    ref = _load_reference_for_mode(ref_path, data, mode_name == "autodiff")

    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode_name, mode_overrides", _FORCE_MODES)
def test_ground_force_methods_batch_same_species(device, methanal_batch_data, mode_name, mode_overrides):
    species, coordinates = methanal_batch_data
    force = _run_ground_force(device, species, coordinates, mode_overrides)

    data = {"force": force}
    ref_path = reference_path("ground_force_batch_methanal")
    ref = _load_reference_for_mode(ref_path, data, mode_name == "autodiff")

    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode_name, mode_overrides", _FORCE_MODES)
def test_ground_force_methods_batch_mixed(device, batch_molecule_data, mode_name, mode_overrides):
    species, coordinates = batch_molecule_data
    force = _run_ground_force(device, species, coordinates, mode_overrides)

    data = {"force": force}
    ref_path = reference_path("ground_force_batch_mixed")
    ref = _load_reference_for_mode(ref_path, data, mode_name == "autodiff")

    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode_name, mode_overrides", _EXCITED_FORCE_MODES)
def test_excited_force_methods_single_molecule(device, methane_molecule_data, mode_name, mode_overrides):
    species, coordinates = methane_molecule_data
    force = _run_excited_force(device, species, coordinates, mode_overrides)

    data = {"force": force}
    ref_path = reference_path("excited_force_methane")
    ref = _load_reference_for_mode(ref_path, data, mode_name == "autodiff")

    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode_name, mode_overrides", _EXCITED_FORCE_MODES)
def test_excited_force_methods_batch_same_species(device, methanal_batch_data, mode_name, mode_overrides):
    species, coordinates = methanal_batch_data
    force = _run_excited_force(device, species, coordinates, mode_overrides)

    data = {"force": force}
    ref_path = reference_path("excited_force_batch_methanal")
    ref = _load_reference_for_mode(ref_path, data, mode_name == "autodiff")

    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", ["OM1", "OM2", "OM3"])
def test_omx_ground_force_single_molecule(device, methane_molecule_data, method):
    species, coordinates = methane_molecule_data
    _assert_omx_ground_force(device, species, coordinates, method, "methane")


@pytest.mark.parametrize("method", ["OM1", "OM2", "OM3"])
def test_omx_ground_force_batch_same_species(device, methanal_batch_data, method):
    species, coordinates = methanal_batch_data
    _assert_omx_ground_force(device, species, coordinates, method, "batch_methanal")


@pytest.mark.parametrize("method", ["OM1", "OM2", "OM3"])
def test_omx_ground_force_batch_mixed(device, batch_molecule_data, method):
    species, coordinates = batch_molecule_data
    _assert_omx_ground_force(device, species, coordinates, method, "batch_mixed")


@pytest.mark.parametrize("method", ["OM1", "OM2", "OM3"])
def test_omx_excited_force_single_molecule(device, methane_molecule_data, method):
    species, coordinates = methane_molecule_data
    _assert_omx_excited_force(device, species, coordinates, method, "methane")


@pytest.mark.parametrize("method", ["OM1", "OM2", "OM3"])
def test_omx_excited_force_batch_same_species(device, methanal_batch_data, method):
    species, coordinates = methanal_batch_data
    _assert_omx_excited_force(device, species, coordinates, method, "batch_methanal")

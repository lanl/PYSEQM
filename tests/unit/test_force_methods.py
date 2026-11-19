import pytest
import torch

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

    assert_allclose(data["force"], ref["force"], rtol=3e-5, atol=3e-5)


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


@pytest.mark.parametrize("mode", [[True], [True, "numerical"]], ids=["analytical", "semi_numerical"])
def test_pm6_sp_gradient_matches_autodiff_for_fluorine(device, mode):
    """
    PM6 gives F a negative hpp = (g_pp - g_p2)/2, clamped in the additive terms. The integrals and their
    derivatives must use the same clamped value (once they did not: 0.26 eV/A error). Only F triggers this and
    no other test molecule contains it. The geometry is generic: a bond along an axis is singular for autodiff.
    """
    species = torch.tensor([[9, 6, 1, 1, 1]], device=device)
    coordinates = torch.tensor(
        [[[0.88712, 0.34408, -0.99952], [0.0, 0.0, 0.0], [-0.20906, 0.87765, 0.61362],
          [0.43897, -0.77877, 0.61856], [-0.92396, -0.35876, -0.44652]]],
        dtype=torch.float64, device=device,
    )  # fmt: skip

    def force(overrides):
        seqm_parameters = {"method": "PM6_SP", "scf_eps": 1.0e-9, "scf_converger": [1], **overrides}
        molecule = Molecule(Constants().to(device), seqm_parameters, coordinates.clone(), species).to(device)
        Electronic_Structure(seqm_parameters).to(device)(molecule)
        return molecule.force.detach()

    assert_allclose(force({"analytical_gradient": mode}), force({}), rtol=0.0, atol=1e-6)

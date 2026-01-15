import torch

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

from ..reference_data import assert_allclose


def _rotation_matrix_z(theta):
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack(
        [
            torch.stack([c, -s, torch.zeros_like(c)]),
            torch.stack([s, c, torch.zeros_like(c)]),
            torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)]),
        ]
    )


def _rotate(coords, R):
    return torch.einsum("...i,ij->...j", coords, R)


def test_rotation_invariance_ground_state(device, methane_molecule_data):
    species, coordinates = methane_molecule_data
    const = Constants().to(device)

    seqm_parameters = {"method": "AM1", "scf_eps": 1.0e-7, "scf_converger": [1]}

    molecule = Molecule(const, seqm_parameters, coordinates.clone(), species).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecule)

    E0 = molecule.Etot.detach().cpu()
    F0 = molecule.force.detach().cpu()

    theta = torch.tensor(0.7, dtype=coordinates.dtype)
    R = _rotation_matrix_z(theta).to(coordinates.device)
    coords_rot = _rotate(coordinates, R)

    molecule_rot = Molecule(const, seqm_parameters, coords_rot, species).to(device)
    esdriver(molecule_rot)

    E1 = molecule_rot.Etot.detach().cpu()
    F1 = molecule_rot.force.detach().cpu()
    F0_rot = _rotate(F0, R.cpu())

    assert_allclose(E0, E1, rtol=1e-5, atol=1e-5)
    assert_allclose(F0_rot, F1, rtol=1e-4, atol=1e-4)


def test_rotation_invariance_excited_state(device, methane_molecule_data):
    species, coordinates = methane_molecule_data
    const = Constants().to(device)

    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "excited_states": {"n_states": 4, "method": "cis"},
        "active_state": 1,
    }

    molecule = Molecule(const, seqm_parameters, coordinates.clone(), species).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecule)

    E0 = molecule.Etot.detach().cpu()
    F0 = molecule.force.detach().cpu()
    cis0 = molecule.cis_energies.detach().cpu()
    osc0 = molecule.oscillator_strength.detach().cpu()

    theta = torch.tensor(-0.9, dtype=coordinates.dtype)
    R = _rotation_matrix_z(theta).to(coordinates.device)
    coords_rot = _rotate(coordinates, R)

    molecule_rot = Molecule(const, seqm_parameters, coords_rot, species).to(device)
    esdriver(molecule_rot)

    E1 = molecule_rot.Etot.detach().cpu()
    F1 = molecule_rot.force.detach().cpu()
    cis1 = molecule_rot.cis_energies.detach().cpu()
    osc1 = molecule_rot.oscillator_strength.detach().cpu()
    F0_rot = _rotate(F0, R.cpu())

    assert_allclose(E0, E1, rtol=1e-5, atol=1e-5)
    assert_allclose(cis0, cis1, rtol=1e-5, atol=1e-5)
    assert_allclose(osc0, osc1, rtol=1e-5, atol=1e-5)
    assert_allclose(F0_rot, F1, rtol=1e-4, atol=1e-4)

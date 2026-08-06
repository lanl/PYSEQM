import pytest
import torch

from seqm.basics import Energy, Hamiltonian
from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants
from seqm.seqm_functions.rcis_batch import matrix_vector_product_batched, rcis_batch
from seqm.seqm_functions.rcis_new import rcis_any_batch
from seqm.seqm_functions.rcis_solver import (
    _protected_root_bases,
    make_rcis_response_builder,
    validate_rcis_stability,
)

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


def test_om2_transition_dipoles_match_mndo_ethene_twist(device):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    species = torch.tensor([[6, 6, 1, 1, 1, 1]], dtype=torch.int64, device=device)
    try:
        coordinates = torch.tensor(
            [
                [
                    [0.00000, 0.66819, 0.00000],
                    [0.00000, -0.66819, 0.00000],
                    [0.66414, 1.23830, -0.64135],
                    [0.66414, -1.23830, 0.64135],
                    [-0.66414, 1.23830, 0.64135],
                    [-0.66414, -1.23830, -0.64135],
                ]
            ],
            dtype=torch.float64,
            device=device,
        )
        const = Constants().to(device)
        seqm_parameters = {
            "method": "OM2",
            "scf_eps": 1.0e-8,
            "scf_converger": [2],
            "excited_states": {"n_states": 4, "method": "cis"},
            "active_state": 1,
        }

        molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
        esdriver = Electronic_Structure(seqm_parameters).to(device)
        esdriver(molecule)

        expected_transition_dipole = torch.tensor(
            [
                [
                    [0.0, -0.5779362869470863, 0.0],
                    [-0.6165675141371499, 0.0, 0.0],
                    [0.0, 0.0, 0.7826884559272248],
                    [-0.4087342822437641, 0.0, 0.0],
                ]
            ],
            dtype=torch.float64,
            device=device,
        )
        expected_oscillator_strength = torch.tensor(
            [[0.008034, 0.038884, 0.074313, 0.022889]], dtype=torch.float64, device=device
        )

        assert molecule.transition_dipole is not None
        assert molecule.oscillator_strength is not None
        assert_allclose(
            molecule.transition_dipole.detach().cpu().tolist(),
            expected_transition_dipole.detach().cpu().tolist(),
            rtol=1e-5,
            atol=1e-5,
        )
        assert_allclose(
            molecule.oscillator_strength.detach().cpu().tolist(),
            expected_oscillator_strength.detach().cpu().tolist(),
            rtol=1e-4,
            atol=1e-5,
        )
    finally:
        torch.set_default_dtype(prev_dtype)


def _rcis_scf_state(device, species, coordinates, n_states):
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-8,
        "scf_converger": [1],
        "scf_backward": 1,
        "excited_states": {"n_states": n_states, "method": "cis", "tolerance": 1.0e-8},
    }
    molecule = Molecule(Constants().to(device), seqm_parameters, coordinates, species).to(device)
    molecule.verbose = False
    hamiltonian = Hamiltonian(seqm_parameters).to(device)
    _, e_mo, _, _, w, _, _, _, _, _, _, molecular_orbitals = hamiltonian(molecule, "AM1")
    molecule.molecular_orbitals = molecular_orbitals
    return molecule, e_mo, w


def _clone_rcis_inputs(molecule, e_mo, w, requires_grad):
    def clone(tensor):
        copy = tensor.detach().clone()
        return copy.requires_grad_(True) if requires_grad else copy

    return (
        clone(molecule.molecular_orbitals),
        clone(e_mo),
        clone(w),
        clone(molecule.parameters["g_ss"]),
        clone(molecule.parameters["g_pp"]),
        clone(molecule.parameters["g_sp"]),
        clone(molecule.parameters["g_p2"]),
        clone(molecule.parameters["h_sp"]),
    )


def _dense_cis_matrix(apply, nmol, nov, dtype, device):
    eye = torch.eye(nov, dtype=dtype, device=device)
    vectors = eye.unsqueeze(0).expand(nmol, -1, -1)
    return apply(vectors).transpose(1, 2)


def _canonicalize(amplitudes):
    pivot = amplitudes.abs().argmax(dim=-1, keepdim=True)
    return amplitudes * torch.where(amplitudes.gather(-1, pivot) < 0, -1.0, 1.0)


def _run_rcis_with_inputs(solver, molecule, inputs, nroots, root_tol):
    original_molecular_orbitals = molecule.molecular_orbitals
    parameter_keys = ("g_ss", "g_pp", "g_sp", "g_p2", "h_sp")
    original_parameters = {key: molecule.parameters[key] for key in parameter_keys}
    molecule.molecular_orbitals = inputs[0]
    for key, value in zip(parameter_keys, inputs[3:]):
        molecule.parameters[key] = value
    try:
        energies, amplitudes, _, _ = solver(molecule, inputs[2], inputs[1], nroots, root_tol)
        return energies, amplitudes
    finally:
        molecule.molecular_orbitals = original_molecular_orbitals
        molecule.parameters.update(original_parameters)


def _run_uniform_with_inputs(molecule, inputs, nroots, root_tol):
    return _run_rcis_with_inputs(rcis_batch, molecule, inputs, nroots, root_tol)


def _run_mixed_with_inputs(molecule, inputs, nroots, root_tol):
    return _run_rcis_with_inputs(rcis_any_batch, molecule, inputs, nroots, root_tol)


def test_rcis_rejects_negative_root(device):
    energies = torch.tensor([[-1.0, 2.0]], dtype=torch.float64, device=device)
    with pytest.raises(RuntimeError, match="RCIS stability failure"):
        validate_rcis_stability(energies, torch.tensor([1], dtype=torch.long, device=device))


def test_protected_degenerate_basis_is_orthonormalized(device):
    amplitudes = torch.tensor([[[1.0, 0.0, 0.0], [0.1, 1.0, 0.0]]], dtype=torch.float64, device=device)
    evals = torch.tensor([[1.0, 1.0 + 1.0e-8]], dtype=torch.float64, device=device)
    protected, counts = _protected_root_bases(
        amplitudes,
        evals,
        torch.tensor([2], dtype=torch.long, device=device),
        torch.tensor([0, 1], dtype=torch.long, device=device),
        torch.ones(1, 3, dtype=torch.bool, device=device),
        root_tol=1.0e-8,
    )

    for root in range(2):
        basis = protected[0, root, : counts[0, root]]
        torch.testing.assert_close(basis @ basis.T, torch.eye(2, dtype=basis.dtype, device=device))


def test_rcis_adjoint_matches_dense_eigenproblem(device, methane_molecule_data):
    species, coordinates = methane_molecule_data
    molecule, e_mo, w = _rcis_scf_state(device, species, coordinates, n_states=3)
    nocc = int(molecule.nocc[0].item())
    nov = nocc * (int(molecule.norb[0].item()) - nocc)

    def loss_and_grads(use_dense):
        inputs = _clone_rcis_inputs(molecule, e_mo, w, requires_grad=True)
        if use_dense:
            apply = make_rcis_response_builder(molecule, matrix_vector_product_batched)(*inputs)
            dense_matrix = _dense_cis_matrix(apply, int(molecule.nmol), nov, inputs[1].dtype, device)
            energies, vectors = torch.linalg.eigh(dense_matrix)
            amplitudes = _canonicalize(vectors[:, :, :3].transpose(1, 2))
            energies = energies[:, :3]
        else:
            energies, amplitudes = _run_uniform_with_inputs(molecule, inputs, nroots=3, root_tol=1.0e-8)
            amplitudes = _canonicalize(amplitudes)

        weights = torch.arange(amplitudes.numel(), device=device, dtype=amplitudes.dtype).view_as(amplitudes)
        loss = energies.sum() + 0.05 * torch.sum(amplitudes * weights / amplitudes.numel())
        grads = torch.autograd.grad(loss, (inputs[1], inputs[2], inputs[3]), allow_unused=False)
        return loss.detach(), tuple(grad.detach() for grad in grads)

    adjoint_loss, adjoint_grads = loss_and_grads(use_dense=False)
    dense_loss, dense_grads = loss_and_grads(use_dense=True)

    assert torch.allclose(adjoint_loss, dense_loss, rtol=1e-7, atol=1e-8)
    for adjoint_grad, dense_grad in zip(adjoint_grads, dense_grads):
        assert torch.allclose(adjoint_grad, dense_grad, rtol=3e-4, atol=3e-5)


def test_rcis_adjoint_mixed_batch_is_finite(device, excited_mixed_batch_data):
    species, coordinates = excited_mixed_batch_data
    molecule, e_mo, w = _rcis_scf_state(device, species, coordinates, n_states=4)
    with torch.no_grad():
        native_energies, native_amplitudes, _, _ = rcis_any_batch(
            molecule, w.detach(), e_mo.detach(), 4, 1.0e-7
        )

    inputs = _clone_rcis_inputs(molecule, e_mo, w, requires_grad=True)

    energies, amplitudes = _run_mixed_with_inputs(molecule, inputs, nroots=4, root_tol=1.0e-7)
    torch.testing.assert_close(energies, native_energies)
    torch.testing.assert_close(amplitudes, native_amplitudes)

    loss = energies[2, 2] + 0.1 * (amplitudes[0, 0, 0] + amplitudes[2, 2, 0])
    gradients = torch.autograd.grad(loss, (inputs[2], inputs[4]), allow_unused=False)

    assert torch.isfinite(energies).all()
    assert torch.isfinite(amplitudes).all()
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_energy_cis_backward_flag_builds_coordinate_gradient(device, methane_molecule_data):
    species, coordinates = methane_molecule_data
    parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-8,
        "scf_converger": [1],
        "scf_backward": 1,
        "cis_backward": True,
        "excited_states": {"n_states": 3, "method": "cis", "tolerance": 1.0e-8},
        "active_state": 1,
    }

    def evaluate(geometry):
        molecule = Molecule(Constants().to(device), parameters, geometry, species).to(device)
        _, total_energy, *_ = Energy(parameters).to(device)(molecule, all_terms=True)
        return total_energy.sum(), molecule.coordinates

    total_energy, graph_coordinates = evaluate(coordinates)
    gradient = torch.autograd.grad(total_energy, graph_coordinates)[0]

    displacement = torch.zeros_like(coordinates)
    displacement[0, 0, 0] = 1.0e-4
    with torch.no_grad():
        energy_plus, _ = evaluate(coordinates + displacement)
        energy_minus, _ = evaluate(coordinates - displacement)
    finite_difference = (energy_plus - energy_minus) / (2.0e-4)

    assert total_energy.requires_grad
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient[0, 0, 0], finite_difference, rtol=1.0e-3, atol=1.0e-4)

import pytest
import torch

from seqm.basics import Hamiltonian
from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants
from seqm.seqm_functions.dipole import calc_dipole_matrix
from seqm.seqm_functions.rcis_batch import _make_uniform_spec, calc_transition_dipoles
from seqm.seqm_functions.rcis_new import _make_any_batch_spec
from seqm.seqm_functions.rcis_solver import build_dense_cis_matrix, canonicalize_eigenvectors, solve_rcis

from ..reference_data import assert_allclose, load_or_update_reference, reference_path


@pytest.mark.parametrize("excited_method", ["cis", "rpa"])
def test_excited_states_single_molecule(device, methane_molecule_data, excited_method):
    species, coordinates = methane_molecule_data
    const = Constants().to(device)

    n_states = 4
    seqm_parameters = {
        "method": "AM1",
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

    ref_path = reference_path(f"excited_{excited_method}_am1_methane")
    ref = load_or_update_reference(ref_path, data)

    assert_allclose(data["cis_energies"], ref["cis_energies"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["oscillator_strength"], ref["oscillator_strength"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


def test_cis_batch_same_molecule_different_coords(device, methanal_batch_data):
    species, coordinates = methanal_batch_data
    const = Constants().to(device)

    n_states = 4
    seqm_parameters = {
        "method": "AM1",
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
    ref_path = reference_path("cis_batch_same_molecule_coords")
    ref = load_or_update_reference(ref_path, data)

    assert_allclose(data["cis_energies"], ref["cis_energies"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["oscillator_strength"], ref["oscillator_strength"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


def test_cis_batch_different_molecules(device, excited_mixed_batch_data):
    species, coordinates = excited_mixed_batch_data
    const = Constants().to(device)

    n_states = 4
    seqm_parameters = {
        "method": "AM1",
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
    ref_path = reference_path("cis_batch_different_molecules")
    ref = load_or_update_reference(ref_path, data)

    assert_allclose(data["cis_energies"], ref["cis_energies"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["oscillator_strength"], ref["oscillator_strength"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


def test_rpa_batch_same_molecule_different_coords(device, methanal_batch_data):
    species, coordinates = methanal_batch_data
    const = Constants().to(device)

    n_states = 4
    seqm_parameters = {
        "method": "AM1",
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
    ref_path = reference_path("rpa_batch_same_molecule_coords")
    ref = load_or_update_reference(ref_path, data)

    assert_allclose(data["cis_energies"], ref["cis_energies"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["oscillator_strength"], ref["oscillator_strength"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)


def _scf_state(device, species, coordinates, n_states=4):
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-8,
        "scf_converger": [1],
        "scf_backward": 1,
        "excited_states": {"n_states": n_states, "method": "cis", "tolerance": 1.0e-8},
    }
    const = Constants().to(device)
    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    molecule.verbose = False
    hamiltonian = Hamiltonian(seqm_parameters).to(device)
    _, e_mo, _, _, w, _, _, _, _, _, _, molecular_orbitals = hamiltonian(molecule, seqm_parameters["method"])
    molecule.molecular_orbitals = molecular_orbitals
    return molecule, e_mo, w


def _clone_rcis_inputs(molecule, e_mo, w, requires_grad=True):
    def clone(tensor):
        out = tensor.detach().clone()
        if requires_grad:
            out.requires_grad_(True)
        return out

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


def _cis_transition_dipole_from_coords(device, species, coordinates, n_states, spec_updates):
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-8,
        "scf_converger": [1],
        "scf_backward": 1,
        "excited_states": {"n_states": n_states, "method": "cis", "tolerance": 1.0e-8},
    }
    const = Constants().to(device)
    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    molecule.verbose = False
    hamiltonian = Hamiltonian(seqm_parameters).to(device)
    _, e_mo, _, _, w, _, _, _, _, _, _, molecular_orbitals = hamiltonian(molecule, seqm_parameters["method"])
    molecule.molecular_orbitals = molecular_orbitals

    spec = _make_uniform_spec(molecule)
    spec["canonicalize"] = True
    spec.update(spec_updates)

    excitation_energies, amplitudes, _ = solve_rcis(
        spec,
        molecular_orbitals,
        e_mo,
        w,
        molecule.parameters["g_ss"],
        molecule.parameters["g_pp"],
        molecule.parameters["g_sp"],
        molecule.parameters["g_p2"],
        molecule.parameters["h_sp"],
        nroots=n_states,
        root_tol=1.0e-8,
    )
    dipole_mat = calc_dipole_matrix(molecule)
    transition_dipole, _ = calc_transition_dipoles(
        molecule,
        amplitudes,
        excitation_energies,
        n_states,
        dipole_mat,
        rpa=False,
        orbital_window=None,
        save_tdm=False,
    )
    return transition_dipole, molecule.coordinates


def _finite_difference_transition_dipole_gradient(
    device, species, coordinates, n_states, spec_updates, state_idx, eps
):
    jac = torch.zeros(3, *coordinates.shape, dtype=coordinates.dtype, device=coordinates.device)
    flat_jac = jac.view(3, -1)
    for idx in range(coordinates.numel()):
        disp = torch.zeros_like(coordinates)
        disp.view(-1)[idx] = eps
        plus = _cis_transition_dipole_from_coords(
            device,
            species,
            (coordinates + disp).detach().clone(),
            n_states=n_states,
            spec_updates=spec_updates,
        )[0][0, state_idx]
        minus = _cis_transition_dipole_from_coords(
            device,
            species,
            (coordinates - disp).detach().clone(),
            n_states=n_states,
            spec_updates=spec_updates,
        )[0][0, state_idx]
        flat_jac[:, idx] = (plus - minus) / (2.0 * eps)
    return jac


def test_rcis_davidson_matches_dense_same_molecule(device, methane_molecule_data):
    species, coordinates = methane_molecule_data
    molecule, e_mo, w = _scf_state(device, species, coordinates, n_states=4)
    spec = _make_uniform_spec(molecule)

    inputs = _clone_rcis_inputs(molecule, e_mo, w, requires_grad=False)
    excitation_energies, amplitudes, _ = solve_rcis(spec, *inputs, nroots=4, root_tol=1.0e-8)
    amplitudes = canonicalize_eigenvectors(amplitudes)

    problem = spec["build_problem"](*inputs)
    dense_A = build_dense_cis_matrix(problem)
    dense_evals, dense_evecs = torch.linalg.eigh(dense_A)
    dense_amp = canonicalize_eigenvectors(dense_evecs[:, :, :4].transpose(1, 2))

    assert torch.allclose(excitation_energies, dense_evals[:, :4], rtol=1e-7, atol=1e-8)
    assert torch.allclose(amplitudes, dense_amp, rtol=1e-6, atol=1e-7)

    flipped_guess = -dense_amp.detach()
    excitation_energies_2, amplitudes_2, _ = solve_rcis(
        spec,
        *inputs,
        nroots=4,
        root_tol=1.0e-8,
        best_guess_from_prev=False,
        init_amplitude_guess=flipped_guess,
    )
    amplitudes_2 = canonicalize_eigenvectors(amplitudes_2)
    assert torch.allclose(excitation_energies_2, excitation_energies, rtol=1e-8, atol=1e-9)
    assert torch.allclose(amplitudes_2, amplitudes, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "adjoint_spec_updates",
    [{}, {"adjoint_solver": "davidson", "dense_adjoint_threshold": 0, "adjoint_max_subspace": 12}],
)
def test_rcis_custom_backward_matches_dense_uniform(device, methane_molecule_data, adjoint_spec_updates):
    species, coordinates = methane_molecule_data
    molecule, e_mo, w = _scf_state(device, species, coordinates, n_states=3)
    spec = _make_uniform_spec(molecule)
    spec.update(adjoint_spec_updates)

    def rcis_loss(use_dense):
        inputs = _clone_rcis_inputs(molecule, e_mo, w, requires_grad=True)
        if use_dense:
            problem = spec["build_problem"](*inputs)
            dense_A = build_dense_cis_matrix(problem)
            evals, evecs = torch.linalg.eigh(dense_A)
            amps = canonicalize_eigenvectors(evecs[:, :, :3].transpose(1, 2))
            energies = evals[:, :3]
        else:
            energies, amps, _ = solve_rcis(spec, *inputs, nroots=3, root_tol=1.0e-8)

        weight = torch.arange(amps.numel(), dtype=amps.dtype, device=amps.device).view_as(amps) / amps.numel()
        loss = energies.sum() + 0.05 * torch.sum(weight * amps)
        grads = torch.autograd.grad(loss, (inputs[1], inputs[2], inputs[3]), allow_unused=False)
        return loss.detach(), tuple(g.detach() for g in grads)

    loss_rcis, grads_rcis = rcis_loss(use_dense=False)
    loss_dense, grads_dense = rcis_loss(use_dense=True)

    assert torch.allclose(loss_rcis, loss_dense, rtol=1e-7, atol=1e-8)
    for grad_rcis, grad_dense in zip(grads_rcis, grads_dense):
        assert torch.allclose(grad_rcis, grad_dense, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize(
    "adjoint_spec_updates",
    [
        {"adjoint_solver": "cg_squared", "dense_adjoint_threshold": 0},
        {"adjoint_solver": "davidson", "dense_adjoint_threshold": 0, "adjoint_max_subspace": 12},
    ],
)
def test_rcis_custom_backward_matches_dense_uniform_batch(device, methanal_batch_data, adjoint_spec_updates):
    species, coordinates = methanal_batch_data
    molecule, e_mo, w = _scf_state(device, species, coordinates, n_states=3)
    spec = _make_uniform_spec(molecule)
    spec.update(adjoint_spec_updates)

    def rcis_loss(use_dense):
        inputs = _clone_rcis_inputs(molecule, e_mo, w, requires_grad=True)
        if use_dense:
            problem = spec["build_problem"](*inputs)
            dense_A = build_dense_cis_matrix(problem)
            evals, evecs = torch.linalg.eigh(dense_A)
            amps = canonicalize_eigenvectors(evecs[:, :, :3].transpose(1, 2))
            energies = evals[:, :3]
        else:
            energies, amps, _ = solve_rcis(spec, *inputs, nroots=3, root_tol=1.0e-8)

        weight = torch.arange(amps.numel(), dtype=amps.dtype, device=amps.device).view_as(amps) / amps.numel()
        loss = energies.sum() + 0.05 * torch.sum(weight * amps)
        grads = torch.autograd.grad(loss, (inputs[1], inputs[2], inputs[3]), allow_unused=False)
        return loss.detach(), tuple(g.detach() for g in grads)

    loss_rcis, grads_rcis = rcis_loss(use_dense=False)
    loss_dense, grads_dense = rcis_loss(use_dense=True)

    assert torch.allclose(loss_rcis, loss_dense, rtol=1e-6, atol=1e-7)
    for grad_rcis, grad_dense in zip(grads_rcis, grads_dense):
        assert torch.allclose(grad_rcis, grad_dense, rtol=3e-4, atol=3e-5)


def test_rcis_custom_backward_near_degenerate_roots_are_finite(device, excited_mixed_batch_data):
    species, coordinates = excited_mixed_batch_data
    molecule, e_mo, w = _scf_state(device, species, coordinates, n_states=4)
    spec = _make_any_batch_spec(molecule)
    inputs = _clone_rcis_inputs(molecule, e_mo, w, requires_grad=True)

    energies, amps, info = solve_rcis(spec, *inputs, nroots=4, root_tol=1.0e-7)
    active_mask = info["active_root_mask"].unsqueeze(-1)
    loss = energies[2, 2] + 0.1 * amps[2, 2, 0] + 0.05 * torch.sum(amps * active_mask)
    grads = torch.autograd.grad(loss, (inputs[2], inputs[4]), allow_unused=False)

    assert torch.isfinite(energies).all()
    assert torch.isfinite(amps).all()
    for grad in grads:
        assert torch.isfinite(grad).all()


@pytest.mark.parametrize(
    "adjoint_spec_updates",
    [
        {"dense_adjoint_threshold": 10_000},
        {"adjoint_solver": "cg_squared", "dense_adjoint_threshold": 0},
        {"adjoint_solver": "davidson", "dense_adjoint_threshold": 0, "adjoint_max_subspace": 12},
    ],
)
def test_rcis_transition_dipole_coordinate_gradient_matches_finite_difference(
    device, methanal_batch_data, adjoint_spec_updates
):
    species, coordinates = methanal_batch_data
    species = species[:1].clone()
    coordinates = coordinates[:1].clone()
    n_states = 3
    eps = 1.0e-4

    with torch.no_grad():
        base_tdm = _cis_transition_dipole_from_coords(
            device, species, coordinates, n_states=n_states, spec_updates={"dense_adjoint_threshold": 10_000}
        )[0]
        state_idx = int(torch.argmax(torch.linalg.vector_norm(base_tdm[0], dim=1)).item())

    transition_dipole, graph_coordinates = _cis_transition_dipole_from_coords(
        device, species, coordinates.clone(), n_states=n_states, spec_updates=adjoint_spec_updates
    )
    autodiff_jac = []
    for cart in range(3):
        grad = torch.autograd.grad(
            transition_dipole[0, state_idx, cart],
            graph_coordinates,
            retain_graph=cart < 2,
            allow_unused=False,
        )[0]
        autodiff_jac.append(grad)
    autodiff_jac = torch.stack(autodiff_jac, dim=0)

    fd_jac = _finite_difference_transition_dipole_gradient(
        device,
        species,
        coordinates,
        n_states=n_states,
        spec_updates=adjoint_spec_updates,
        state_idx=state_idx,
        eps=eps,
    )

    assert torch.allclose(autodiff_jac, fd_jac, rtol=1e-3, atol=5e-5)

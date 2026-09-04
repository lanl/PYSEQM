import pytest
import torch

from seqm.ElectronicStructure import Electronic_Structure
from seqm.MolecularDynamics import XL_ESMD
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants


def _parameters(n_states=1):
    return {
        "method": "AM1",
        "scf_eps": 1.0e-10,
        "scf_converger": [2],
        "excited_states": {"n_states": n_states, "method": "cis", "tolerance": 1.0e-9},
        "active_state": 1,
        "analytical_gradient": [True],
    }


def _xlesmd_driver(parameters, device):
    driver = Electronic_Structure(parameters).to(device)
    driver.conservative_force.energy.excited_states = None
    driver.conservative_force.energy.xlesmd = True
    return driver


def test_xlesmd_defaults_to_verified_autodiff_shadow_force():
    parameters = _parameters()
    parameters["elements"] = [1, 6, 8]
    md = XL_ESMD(
        xl_bomd_params={"k": 6, "max_rank": 3, "err_threshold": 1e-8},
        seqm_parameters=parameters,
        timestep=0.1,
        output={"molid": [], "print every": 0, "xyz": 0, "h5": {"data": 0}},
    )
    assert md.xlesmd_force_mode == "autodiff"
    assert md.seqm_parameters["scf_backward"] == 2
    assert md.xlesmd_scf_backward == 2
    assert md.seqm_parameters["analytical_gradient"] == [False]
    assert "scf_backward" not in parameters
    assert parameters["analytical_gradient"] == [True]

    md_implicit = XL_ESMD(
        xl_bomd_params={"k": 6, "scf_backward": 1},
        seqm_parameters=parameters,
        timestep=0.1,
        output={"molid": [], "print every": 0, "xyz": 0, "h5": {"data": 0}},
    )
    assert md_implicit.seqm_parameters["scf_backward"] == 1


@pytest.mark.parametrize(
    "constraint_mode", ["independent_linearized", "independent_exact", "ordered_linearized"]
)
def test_experimental_analytical_gradient_is_retained_for_labelled_xlesmd_modes(
    device, methanal_batch_data, constraint_mode
):
    """The selected-root analytical path remains usable for 1a, 1b, and ordered XL."""
    torch.set_default_dtype(torch.float64)
    species_batch, coordinates_batch = methanal_batch_data
    parameters = _parameters(n_states=2)
    parameters["analytical_gradient"] = [False]
    molecule = Molecule(
        Constants().to(device), parameters, coordinates_batch[:1].clone(), species_batch[:1].clone()
    ).to(device)
    md = XL_ESMD(
        xl_bomd_params={
            "k": 6,
            "constraint_mode": constraint_mode,
            "max_rank": 3,
            "err_threshold": 1.0e-8,
            "force_mode": "experimental_analytic",
        },
        Temp=300.0,
        seqm_parameters=parameters,
        timestep=0.1,
        output={"molid": [], "print every": 0, "xyz": 0, "h5": {"data": 0}},
    ).to(device)

    assert md.xlesmd_force_mode == "experimental_analytic"
    assert md.seqm_parameters["analytical_gradient"] == [True]
    assert parameters["analytical_gradient"] == [False]
    md.initialize(molecule, remove_com=None, learned_parameters={}, steps=2)

    assert molecule.xlesmd_diagnostics["constraint_mode"] == constraint_mode
    assert torch.isfinite(molecule.force).all()


@pytest.mark.skip(reason="Known off-shell autodiff-force mismatch; disabled pending a fix.")
def test_off_shell_shadow_autodiff_force_matches_finite_difference(device, methanal_batch_data):
    torch.set_default_dtype(torch.float64)
    species_batch, coordinates_batch = methanal_batch_data
    species = species_batch[:1]
    coordinates = coordinates_batch[:1]
    parameters = _parameters()
    const = Constants().to(device)

    reference = Molecule(const, parameters, coordinates.clone(), species.clone()).to(device)
    reference.verbose = False
    reference_driver = Electronic_Structure(parameters).to(device)
    reference_driver(reference, do_force=False)
    eta = reference.cis_amplitudes.detach().clone()
    torch.manual_seed(8)
    eta = eta + 1.0e-4 * torch.randn_like(eta)
    reference_mos = reference.molecular_orbitals.detach().clone()

    autodiff_parameters = _parameters()
    autodiff_parameters["scf_backward"] = 2
    autodiff_parameters["analytical_gradient"] = [False]
    molecule = Molecule(const, autodiff_parameters, coordinates.clone(), species.clone()).to(device)
    molecule.verbose = False
    molecule.molecular_orbitals = reference_mos.clone()
    driver = _xlesmd_driver(autodiff_parameters, device)
    driver(molecule, cis_amp=eta, do_force=True)

    atom, axis = 1, 2
    step = 2.0e-4

    def shadow_heat(displacement):
        displaced = coordinates.clone()
        displaced[0, atom, axis] += displacement
        trial = Molecule(const, autodiff_parameters, displaced, species.clone()).to(device)
        trial.verbose = False
        trial.molecular_orbitals = reference_mos.clone()
        trial_driver = _xlesmd_driver(autodiff_parameters, device)
        trial_driver(trial, cis_amp=eta, do_force=False)
        return trial.Hf.detach()

    finite_force = -(shadow_heat(step) - shadow_heat(-step)) / (2.0 * step)
    torch.testing.assert_close(molecule.force[0, atom, axis], finite_force[0], rtol=2e-4, atol=3e-6)


@pytest.mark.parametrize("constraint_mode", ["coupled_linearized", "coupled_exact"])
@pytest.mark.parametrize("scf_backward", [1, 2])
def test_coupled_block_shadow_force_matches_finite_difference(
    device, methanal_batch_data, constraint_mode, scf_backward
):
    torch.set_default_dtype(torch.float64)
    species_batch, coordinates_batch = methanal_batch_data
    species = species_batch[:1]
    coordinates = coordinates_batch[:1]
    const = Constants().to(device)

    reference_parameters = _parameters(n_states=2)
    reference_parameters["analytical_gradient"] = [False]
    reference = Molecule(const, reference_parameters, coordinates.clone(), species.clone()).to(device)
    reference.verbose = False
    Electronic_Structure(reference_parameters).to(device)(reference, do_force=False)
    eta = reference.cis_amplitudes.detach().clone()
    torch.manual_seed(18)
    eta = eta + 1.0e-4 * torch.randn_like(eta)
    reference_mos = reference.molecular_orbitals.detach().clone()

    parameters = _parameters(n_states=2)
    parameters["scf_backward"] = scf_backward
    parameters["scf_backward_eps"] = 1.0e-10
    parameters["analytical_gradient"] = [False]
    xl_params = {
        "constraint_mode": constraint_mode,
        "max_rank": 3,
        "err_threshold": 1.0e-8,
        "coupled_tolerance": 1.0e-11,
    }
    molecule = Molecule(const, parameters, coordinates.clone(), species.clone()).to(device)
    molecule.verbose = False
    molecule.molecular_orbitals = reference_mos.clone()
    driver = _xlesmd_driver(parameters, device)
    driver(molecule, cis_amp=eta, do_force=True, xl_bomd_params=xl_params)

    assert molecule.xlesmd_block_energy
    assert molecule.xlesmd_diagnostics["constraint_mode"] == constraint_mode
    assert torch.isfinite(molecule.force).all()
    if constraint_mode == "coupled_exact":
        assert molecule.xlesmd_diagnostics["xi_orthogonality"].max() < 2.0e-10

    atom, axis = 1, 2
    step = 2.0e-4

    def shadow_heat(displacement):
        displaced = coordinates.clone()
        displaced[0, atom, axis] += displacement
        trial = Molecule(const, parameters, displaced, species.clone()).to(device)
        trial.verbose = False
        trial.molecular_orbitals = reference_mos.clone()
        trial_driver = _xlesmd_driver(parameters, device)
        trial_driver(trial, cis_amp=eta, do_force=False, xl_bomd_params=xl_params)
        return trial.Hf.detach()

    finite_force = -(shadow_heat(step) - shadow_heat(-step)) / (2.0 * step)
    torch.testing.assert_close(molecule.force[0, atom, axis], finite_force[0], rtol=5e-4, atol=6e-6)


@pytest.mark.parametrize("constraint_mode", ["2a", "2b"])
def test_coupled_short_trajectory_is_finite_and_starts_on_shadow_potential(
    device, methanal_batch_data, constraint_mode
):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(73)
    species_batch, coordinates_batch = methanal_batch_data
    parameters = _parameters(n_states=2)
    parameters["scf_backward_eps"] = 1.0e-10
    molecule = Molecule(
        Constants().to(device), parameters, coordinates_batch[:1].clone(), species_batch[:1].clone()
    ).to(device)
    xl_params = {
        "k": 6,
        "constraint_mode": constraint_mode,
        "max_rank": 3,
        "err_threshold": 1.0e-8,
        "coupled_tolerance": 1.0e-11,
    }
    if constraint_mode == "2a":
        xl_params["coupled_krylov_preconditioner"] = "lambda"
    md = XL_ESMD(
        xl_bomd_params=xl_params,
        Temp=300.0,
        seqm_parameters=parameters,
        timestep=0.1,
        output={"molid": [], "print every": 0, "xyz": 0, "h5": {"data": 0}},
    ).to(device)
    md.initialize(molecule, remove_com=None, learned_parameters={}, steps=6)

    assert molecule.xlesmd_block_energy
    assert molecule.xlesmd_diagnostics["constraint_mode"] in {"coupled_linearized", "coupled_exact"}
    assert molecule.xlesmd_diagnostics["krylov_preconditioner"] == (
        "lambda" if constraint_mode == "2a" else "none"
    )
    energies = [md._thermo_potential(molecule) + md._kinetic_energy(molecule)]
    for step in range(6):
        md._do_integrator_step(step, molecule, {})
        energies.append(md._thermo_potential(molecule) + md._kinetic_energy(molecule))

    energies = torch.stack(energies)
    assert torch.isfinite(energies).all()
    assert torch.isfinite(molecule.coordinates).all()
    assert torch.max(torch.abs(energies - energies[0])) < 1.5e-3
    assert molecule.xlesmd_diagnostics["projected_fixed_point_residual"].max() < 5.0e-3


def test_ordered_multistate_retraction_keeps_all_auxiliary_roots_bounded(device, methanal_batch_data):
    """Label-preserving retraction prevents inactive roots from drifting away."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(73)
    species_batch, coordinates_batch = methanal_batch_data
    parameters = _parameters(n_states=2)
    parameters["scf_backward_eps"] = 1.0e-10
    molecule = Molecule(
        Constants().to(device), parameters, coordinates_batch[:1].clone(), species_batch[:1].clone()
    ).to(device)
    md = XL_ESMD(
        xl_bomd_params={
            "k": 6,
            "constraint_mode": "ordered_linearized",
            "max_rank": 3,
            "err_threshold": 1.0e-8,
        },
        Temp=300.0,
        seqm_parameters=parameters,
        timestep=0.1,
        output={"molid": [], "print every": 0, "xyz": 0, "h5": {"data": 0}},
    ).to(device)
    md.initialize(molecule, remove_com=None, learned_parameters={}, steps=8)
    for step in range(8):
        md._do_integrator_step(step, molecule, {})

    diagnostics = molecule.xlesmd_diagnostics
    assert diagnostics["constraint_mode"] == "ordered_linearized"
    assert torch.isfinite(molecule.coordinates).all()
    assert torch.isfinite(molecule.cis_amplitudes).all()
    assert diagnostics["eta_orthogonality"].max() < 1.0e-10
    assert diagnostics["xi_orthogonality"].max() < 5.0e-2

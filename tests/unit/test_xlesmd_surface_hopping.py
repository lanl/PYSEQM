from types import SimpleNamespace

import pytest
import torch

from seqm.api import Constants, Molecule, SurfaceHoppingDynamics, XLESurfaceHoppingDynamics


def _parameters(tdc_method="overlap"):
    return {
        "method": "AM1",
        "scf_eps": 1.0e-10,
        "scf_converger": [2],
        "excited_states": {"n_states": 2, "method": "cis", "tolerance": 1.0e-9},
        "active_state": 1,
        "analytical_gradient": [True],
        "nonadiabatic": {
            "tdc_method": tdc_method,
            "detect_crossings": False,
            "trajectory_termination": {"enabled": False},
        },
    }


def _output():
    return {"molid": [], "print every": 0, "xyz": 0, "h5": {"data": 0, "nonadiabatic": 0}}


def _xl_parameters():
    return {
        "k": 6,
        "constraint_mode": "ordered_linearized",
        "max_rank": 3,
        "err_threshold": 1.0e-8,
        "krylov_preconditioner": "none",
    }


def _molecule(device, species, coordinates, parameters):
    molecule = Molecule(Constants().to(device), parameters, coordinates[:1].clone(), species[:1].clone()).to(
        device
    )
    molecule.verbose = False
    velocity = torch.linspace(
        -2.0e-3, 2.0e-3, molecule.coordinates.numel(), dtype=molecule.coordinates.dtype, device=device
    ).reshape_as(molecule.coordinates)
    molecule.velocities = velocity
    return molecule


@pytest.mark.parametrize("tdc_method", ["overlap", "hamiltonian_fd", "nac_dot_v"])
def test_xlesmd_surface_hopping_one_step_is_finite(device, methanal_batch_data, tdc_method):
    torch.set_default_dtype(torch.float64)
    species, coordinates = methanal_batch_data
    parameters = _parameters(tdc_method)
    molecule = _molecule(device, species, coordinates, parameters)
    dynamics = XLESurfaceHoppingDynamics(
        seqm_parameters=parameters,
        xl_bomd_params=_xl_parameters(),
        timestep=0.05,
        Temp=0.0,
        initial_state=1,
        output=_output(),
    ).to(device)

    dynamics.initialize(molecule, remove_com=None, learned_parameters={}, steps=1)
    torch.manual_seed(91)
    dynamics._do_integrator_step(0, molecule, {})

    assert torch.isfinite(molecule.coordinates).all()
    assert torch.isfinite(molecule.force).all()
    assert torch.isfinite(dynamics.populations).all()
    assert torch.isfinite(dynamics._cache_old["nac_dot"]).all()
    assert molecule.xlesmd_nac_diagnostics["xi_max_gram_error"] < 0.1


def test_short_xlesmd_fssh_tracks_ordinary_cis_fssh(device, methanal_batch_data):
    torch.set_default_dtype(torch.float64)
    species, coordinates = methanal_batch_data
    cis_parameters = _parameters("overlap")
    xl_parameters = _parameters("overlap")
    cis_molecule = _molecule(device, species, coordinates, cis_parameters)
    xl_molecule = _molecule(device, species, coordinates, xl_parameters)

    cis = SurfaceHoppingDynamics(
        seqm_parameters=cis_parameters, timestep=0.05, Temp=0.0, initial_state=1, output=_output()
    ).to(device)
    xl = XLESurfaceHoppingDynamics(
        seqm_parameters=xl_parameters,
        xl_bomd_params=_xl_parameters(),
        timestep=0.05,
        Temp=0.0,
        initial_state=1,
        output=_output(),
    ).to(device)

    cis.initialize(cis_molecule, remove_com=None, learned_parameters={}, steps=3)
    xl.initialize(xl_molecule, remove_com=None, learned_parameters={}, steps=3)

    # Force the same accepted S1 -> S2 hop on the first step so the comparison
    # also exercises the post-hop exact-CIS versus shadow-force paths.
    cis_attempt_hop = cis._attempt_hop
    xl_attempt_hop = xl._attempt_hop
    cis_rescale = cis._rescale_velocity_along_nac
    xl_rescale = xl._rescale_velocity_along_nac
    cis._attempt_hop = lambda: torch.tensor([1], dtype=torch.long, device=device)
    xl._attempt_hop = lambda: torch.tensor([1], dtype=torch.long, device=device)
    cis._rescale_velocity_along_nac = lambda *args, **kwargs: True
    xl._rescale_velocity_along_nac = lambda *args, **kwargs: True
    for step in range(3):
        torch.manual_seed(700 + step)
        cis._do_integrator_step(step, cis_molecule, {})
        torch.manual_seed(700 + step)
        xl._do_integrator_step(step, xl_molecule, {})
        if step == 0:
            cis._attempt_hop = cis_attempt_hop
            xl._attempt_hop = xl_attempt_hop
            cis._rescale_velocity_along_nac = cis_rescale
            xl._rescale_velocity_along_nac = xl_rescale

    torch.testing.assert_close(xl_molecule.coordinates, cis_molecule.coordinates, rtol=0.0, atol=2.0e-4)
    torch.testing.assert_close(xl.populations, cis.populations, rtol=2.0e-2, atol=2.0e-3)
    torch.testing.assert_close(
        xl._cache_old["energies"], cis._cache_old["energies"], rtol=5.0e-3, atol=5.0e-3
    )
    assert torch.equal(xl._active_states, cis._active_states)
    assert xl._active_states.tolist() == [1]
    assert xl.hop_log[0].accepted and cis.hop_log[0].accepted
    assert max(item["xi_max_gram_error"] for item in xl.xlesmd_orthogonality_log) < 0.1


def test_xlesmd_surface_hopping_accepted_hop_switches_shadow_force(device, methanal_batch_data, monkeypatch):
    torch.set_default_dtype(torch.float64)
    species, coordinates = methanal_batch_data
    parameters = _parameters("overlap")
    molecule = _molecule(device, species, coordinates, parameters)
    dynamics = XLESurfaceHoppingDynamics(
        seqm_parameters=parameters,
        xl_bomd_params=_xl_parameters(),
        timestep=0.05,
        Temp=0.0,
        initial_state=1,
        output=_output(),
    ).to(device)
    dynamics.initialize(molecule, remove_com=None, learned_parameters={}, steps=1)

    monkeypatch.setattr(dynamics, "_attempt_hop", lambda: torch.tensor([1], dtype=torch.long, device=device))
    monkeypatch.setattr(dynamics, "_rescale_velocity_along_nac", lambda *args, **kwargs: True)
    dynamics._do_integrator_step(0, molecule, {})

    assert dynamics._active_states.tolist() == [1]
    assert molecule.active_state.tolist() == [2]
    assert len(dynamics.hop_log) == 1
    assert dynamics.hop_log[0].accepted
    assert torch.isfinite(molecule.force).all()
    assert torch.isfinite(molecule.acc).all()


def test_xlesmd_energy_order_relabels_auxiliary_history_without_remapping_fssh_rank(device):
    torch.set_default_dtype(torch.float64)
    parameters = _parameters()
    parameters["elements"] = [1, 6, 8]
    dynamics = XLESurfaceHoppingDynamics(
        seqm_parameters=parameters,
        xl_bomd_params=_xl_parameters(),
        timestep=0.05,
        Temp=0.0,
        initial_state=2,
        output=_output(),
    ).to(device)
    rows = torch.tensor([[[10.0], [20.0], [30.0]]], device=device)
    history = torch.stack((rows + 100.0, rows + 200.0))
    molecule = SimpleNamespace(
        cis_energies=torch.tensor([[3.0, 1.0, 2.0]], device=device),
        cis_amplitudes=rows.clone(),
        transition_density_matrices=rows.unsqueeze(-1).clone(),
        dxi2dt2=(rows + 10.0).clone(),
        xlesmd_multipliers=torch.arange(9, dtype=torch.float64, device=device).reshape(1, 3, 3),
        active_state=torch.tensor([1], dtype=torch.long, device=device),
    )
    dynamics._xl_ctx = {"es_amp": (rows + 30.0).clone(), "es_amp_t": history.clone()}
    # Root 0 moves to rank 2, outside this two-state FSSH manifold.  The
    # active FSSH index must remain the S1 energy rank, not follow root 0.
    dynamics._active_states = torch.tensor([0], dtype=torch.long, device=device)
    dynamics._nstates = 2

    dynamics._apply_xlesmd_energy_order(molecule, step=7)

    permutation = torch.tensor([[1, 2, 0]], device=device)
    expected_rows = rows.gather(1, permutation.unsqueeze(-1))
    expected_history = history.gather(2, permutation.unsqueeze(0).unsqueeze(-1).expand_as(history))
    torch.testing.assert_close(molecule.cis_energies, torch.tensor([[1.0, 2.0, 3.0]], device=device))
    torch.testing.assert_close(molecule.cis_amplitudes, expected_rows)
    torch.testing.assert_close(dynamics._xl_ctx["es_amp"], expected_rows + 30.0)
    torch.testing.assert_close(dynamics._xl_ctx["es_amp_t"], expected_history)
    assert dynamics._active_states.tolist() == [0]
    assert molecule.active_state.tolist() == [1]
    assert dynamics.xlesmd_energy_order_events == [{"step": 7, "molecule": 0, "permutation": [1, 2, 0]}]

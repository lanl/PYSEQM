from types import SimpleNamespace

import torch

from seqm.MolecularDynamics import CONSTANTS
from seqm.NonadiabaticDynamics import NonadiabaticDynamicsBase, SurfaceHoppingDynamics


class DummyNAD(NonadiabaticDynamicsBase):
    """Lightweight subclass that bypasses heavy init/ES for unit testing."""

    def __init__(self):
        # Skip parent init
        pass


class DummyFSSH(SurfaceHoppingDynamics):
    """Lightweight surface-hopping subclass for unit testing hop logic."""

    def __init__(self):
        # Skip parent init
        pass


def make_dummy(nstates=2, timestep=1.0, substeps=4):
    nad = DummyNAD()
    nad.timestep = timestep
    nad.damp = None
    nad._electronic_substeps = substeps
    nad._nstates = nstates
    nad._amp_phase = torch.zeros((1, nstates, 3), dtype=torch.float64)
    nad._amp_phase[0, 0, 0] = 1.0
    nad._current_potential = None
    nad._hop_integral = None
    nad._apc_window = 2
    nad._detect_crossings_flag = True
    # Minimal state/caches used by internal helpers.
    nad._eye_cache = {}
    nad._arange_cache = {}
    nad._perm_cost_buffers = {}
    nad._trivial_zero_buffers = {}
    nad._trivial_swap_buffers = {}
    nad._hop_buffer = None
    nad._active_states = torch.zeros((1,), dtype=torch.long)
    nad.post_hop_holdoff = torch.zeros((1,), dtype=torch.long)
    nad.prev_state = torch.full((1,), -1, dtype=torch.long)
    return nad


def make_dummy_fssh(nmol=1, nstates=2):
    dyn = DummyFSSH()
    dyn.timestep = 1.0
    dyn.damp = None
    dyn._nstates = nstates
    dyn._amp_phase = torch.zeros((nmol, nstates, 3), dtype=torch.float64)
    dyn._amp_phase[:, 0, 0] = 1.0
    dyn._current_potential = None
    dyn._hop_integral = None
    dyn._apc_window = 2
    dyn._detect_crossings_flag = True
    dyn._eye_cache = {}
    dyn._arange_cache = {}
    dyn._perm_cost_buffers = {}
    dyn._trivial_zero_buffers = {}
    dyn._trivial_swap_buffers = {}
    dyn._hop_buffer = None
    dyn._active_states = torch.zeros((nmol,), dtype=torch.long)
    dyn.post_hop_holdoff = torch.zeros((nmol,), dtype=torch.long)
    dyn.prev_state = torch.full((nmol,), -1, dtype=torch.long)
    dyn._force_mode = "active"
    dyn._decohere_on_hop = True
    dyn._trivial_crossing_mask = None
    dyn.hop_log = []
    return dyn


def make_dummy_molecule(nmol=1, molsize=1, dtype=torch.float64):
    zeros = torch.zeros((nmol, molsize, 3), dtype=dtype)
    return SimpleNamespace(
        coordinates=zeros.clone(),
        velocities=zeros.clone(),
        force=zeros.clone(),
        mass_inverse=torch.ones((nmol, molsize, 1), dtype=dtype),
        Etot=torch.full((nmol,), 1.2, dtype=dtype),
    )


def test_rk4_preserves_norm_no_coupling():
    nad = make_dummy()
    energies = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
    nac = torch.zeros((1, 2, 2), dtype=torch.float64)
    cache = {"energies": energies, "nac_dot": nac, "nac_vec": None}
    nad._propagate_electronic(cache, cache, substeps=4)
    pop = nad.populations
    assert torch.allclose(pop.sum(), torch.tensor(1.0, dtype=pop.dtype), atol=1e-12)
    # With zero NAC, populations should remain on state 0
    assert torch.allclose(pop[0, 0], torch.tensor(1.0, dtype=pop.dtype), atol=1e-6)


def test_hop_integral_accumulates_coupling():
    nad = make_dummy(substeps=10, timestep=1.0)
    energies = torch.zeros((1, 2))
    nac = torch.zeros((1, 2, 2), dtype=torch.float64)
    nac[0, 0, 1] = 0.05
    nac[0, 1, 0] = -0.05
    cache = {"energies": energies, "nac_dot": nac, "nac_vec": None}
    nad._propagate_electronic(cache, cache, substeps=10)
    assert nad._hop_integral is not None
    assert torch.abs(nad._hop_integral[0, 0, 1]) > 0
    assert torch.abs(nad._hop_integral[0, 1, 0]) > 0
    assert torch.allclose(nad._hop_integral[0, 0, 1], -nad._hop_integral[0, 1, 0], atol=1.0e-12)


def test_crossing_detection_triggers():
    nad = make_dummy()
    ref_amp = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    tgt_amp = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])  # swapped
    nac_old = torch.tensor([[[0.0, 0.3], [-0.3, 0.0]]])
    nac_new = torch.tensor([[[0.0, 0.4], [-0.4, 0.0]]])
    cache_old = {"cis_amp": ref_amp, "nac_dot": nac_old.clone()}
    cache_new = {"cis_amp": tgt_amp, "nac_dot": nac_new.clone()}
    crossings = nad._detect_crossings(cache_old, cache_new)
    assert crossings.tolist() == [[1, 0]]
    assert torch.allclose(cache_old["nac_dot"], torch.zeros_like(nac_old))
    assert torch.allclose(cache_new["nac_dot"], torch.zeros_like(nac_new))


def test_surface_hopping_accepted_hop_updates_state_and_recomputes_force(monkeypatch):
    dyn = make_dummy_fssh()
    molecule = make_dummy_molecule()
    molecule.force.fill_(0.1)
    excitation_energies = torch.tensor([[0.2, 0.5]], dtype=torch.float64)
    dyn._amp_phase[0, 0] = torch.tensor([0.8, 0.1, 0.3], dtype=torch.float64)
    dyn._amp_phase[0, 1] = torch.tensor([0.2, -0.2, -0.4], dtype=torch.float64)

    monkeypatch.setattr(dyn, "_attempt_hop", lambda: [1])
    monkeypatch.setattr(
        dyn,
        "_compute_NACR_for_hop",
        lambda molecule, nac_pairs: torch.zeros((1, 2, 2, 1, 3), dtype=torch.float64),
    )
    monkeypatch.setattr(dyn, "_rescale_velocity_along_nac", lambda *args, **kwargs: True)

    recompute_calls = []

    def _fake_recompute_active_force(molecule):
        recompute_calls.append(True)
        molecule.force = torch.tensor([[[0.7, -0.1, 0.2]]], dtype=torch.float64)

    monkeypatch.setattr(dyn, "_recompute_active_force", _fake_recompute_active_force)

    dyn._after_electronic_update(
        molecule,
        excitation_energies=excitation_energies,
        nac_matrix=None,
        nac_dot=torch.zeros((1, 2, 2), dtype=torch.float64),
        step=3,
    )

    assert dyn._active_states.tolist() == [1]
    assert len(dyn.hop_log) == 1
    assert dyn.hop_log[0].accepted is True
    assert dyn.hop_log[0].step == 4
    assert torch.allclose(dyn._amp_phase[0, 0], torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(dyn._amp_phase[0, 1], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    assert recompute_calls == [True]
    expected_acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE
    assert torch.allclose(molecule.acc, expected_acc)
    assert torch.allclose(dyn._current_potential, torch.tensor([1.5], dtype=torch.float64))
    assert torch.allclose(molecule.Etot, torch.tensor([1.5], dtype=torch.float64))


def test_surface_hopping_trivial_crossing_logs_hop_event(monkeypatch):
    dyn = make_dummy_fssh()
    molecule = make_dummy_molecule()
    excitation_energies = torch.tensor([[0.2, 0.5]], dtype=torch.float64)
    dyn._trivial_crossing_mask = torch.tensor([[1, 0]], dtype=torch.long)

    monkeypatch.setattr(dyn, "_attempt_hop", lambda: [None])
    monkeypatch.setattr(dyn, "_recompute_active_force", lambda _molecule: None)

    dyn._after_electronic_update(
        molecule,
        excitation_energies=excitation_energies,
        nac_matrix=None,
        nac_dot=torch.zeros((1, 2, 2), dtype=torch.float64),
        step=5,
    )

    assert dyn._active_states.tolist() == [1]
    assert len(dyn.hop_log) == 1
    assert dyn.hop_log[0].accepted is True
    assert dyn.hop_log[0].reason == "Trivial crossing"
    assert dyn.hop_log[0].from_state == 0
    assert dyn.hop_log[0].to_state == 1
    assert dyn.hop_log[0].step == 6
    assert torch.allclose(dyn._amp_phase[0, 0], torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(dyn._amp_phase[0, 1], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))


def test_surface_hopping_frustrated_hop_keeps_state_and_records_failure(monkeypatch):
    dyn = make_dummy_fssh()
    molecule = make_dummy_molecule()
    excitation_energies = torch.tensor([[0.2, 0.5]], dtype=torch.float64)
    dyn._amp_phase[0, 0] = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    dyn._amp_phase[0, 1] = torch.tensor([0.4, -0.2, -0.2], dtype=torch.float64)

    monkeypatch.setattr(dyn, "_attempt_hop", lambda: [1])
    monkeypatch.setattr(
        dyn,
        "_compute_NACR_for_hop",
        lambda molecule, nac_pairs: torch.zeros((1, 2, 2, 1, 3), dtype=torch.float64),
    )
    monkeypatch.setattr(dyn, "_rescale_velocity_along_nac", lambda *args, **kwargs: False)

    dyn._after_electronic_update(
        molecule,
        excitation_energies=excitation_energies,
        nac_matrix=None,
        nac_dot=torch.zeros((1, 2, 2), dtype=torch.float64),
        step=1,
    )

    assert dyn._active_states.tolist() == [0]
    assert len(dyn.hop_log) == 1
    assert dyn.hop_log[0].accepted is False
    assert dyn.hop_log[0].reason == "Frustrated hop"
    assert torch.allclose(dyn._amp_phase[0, 0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.allclose(dyn._amp_phase[0, 1], torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(dyn._current_potential, torch.tensor([1.2], dtype=torch.float64))
    assert torch.allclose(molecule.Etot, torch.tensor([1.2], dtype=torch.float64))

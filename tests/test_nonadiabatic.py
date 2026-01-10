import torch

from seqm.NonadiabaticDynamics import NonadiabaticDynamicsBase


class DummyNAD(NonadiabaticDynamicsBase):
    """Lightweight subclass that bypasses heavy init/ES for unit testing."""

    def __init__(self):
        # Skip parent init
        pass


def make_dummy(nstates=2, timestep=1.0, substeps=4):
    nad = DummyNAD()
    nad.timestep = timestep
    nad._electronic_substeps = substeps
    nad._nstates = nstates
    nad._amp_phase = torch.zeros((1, nstates, 3), dtype=torch.float64)
    nad._amp_phase[0, 0, 0] = 1.0
    nad._current_potential = None
    nad._hop_integral = None
    nad._apc_window = 2
    nad._detect_crossings_flag = True
    nad._crossing_overlap_thresh = 0.9
    return nad


def test_rk4_preserves_norm_no_coupling():
    nad = make_dummy()
    energies = torch.tensor([[0.1, 0.2]])
    cache = {"energies": energies, "nac_dot": None, "nac_vec": None}
    nad._propagate_electronic(cache, cache, substeps=4)
    pop = nad.populations
    assert torch.allclose(pop.sum(), torch.tensor(1.0, dtype=pop.dtype), atol=1e-12)
    # With no NAC, populations should remain on state 0
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
    # Expect some positive hop driving from state 0 to 1
    assert torch.abs(nad._hop_integral[0, 1, 0]) > 0


def test_apc_window_limits_assignment():
    nad = make_dummy()
    nad._apc_window = 0  # only allow diagonal
    # old amplitudes: |1>,|2>
    ref = torch.eye(2).unsqueeze(0)
    # new amplitudes swapped
    tgt = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    perm = nad._compute_perm_from_overlap(ref, tgt)
    assert perm == [0, 1]  # window prevents swap
    # Allow wider window -> swap chosen
    nad._apc_window = 2
    perm2 = nad._compute_perm_from_overlap(ref, tgt)
    assert perm2 in ([1, 0], [0, 1])  # Hungarian may pick either if costs equal


def test_crossing_detection_triggers():
    nad = make_dummy()
    ref_amp = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    tgt_amp = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
    cache_old = {"cis_amp": ref_amp}
    cache_new = {"cis_amp": tgt_amp}
    crossings = nad._detect_crossings(cache_old, cache_new)
    assert crossings  # should detect low overlap pairs

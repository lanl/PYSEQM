import pytest
import torch

from seqm.basics import Energy
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

from ..reference_data import AM1_AND_OM2_METHODS

_LEARNED = ["g_ss", "g_pp", "g_sp", "g_p2", "h_sp", "U_ss", "beta_s"]


def _parameter_gradients(device, species, coordinates, method, scf_backward):
    seqm_parameters = {
        "method": method,
        "scf_eps": 1.0e-10,
        "scf_converger": [1],
        "scf_backward": scf_backward,
        "cis_backward": True,
        "elements": [0, 1, 6, 7, 8],
        "Hf_flag": False,
        "eig": True,
        "pair_outer_cutoff": 1.0e10,
        "excited_states": {"n_states": 3, "method": "cis", "tolerance": 1.0e-9},
        "active_state": 1,
    }
    const = Constants().to(device)
    stock = Molecule(const, seqm_parameters, coordinates, species).to(device).parameters
    table = {name: stock[name].detach().clone().requires_grad_(True) for name in _LEARNED}
    seqm_parameters["learned"] = _LEARNED

    molecule = Molecule(const, seqm_parameters, coordinates, species, learned_parameters=table).to(device)
    total = Energy(seqm_parameters).to(device)(molecule, learned_parameters=table, all_terms=True)[1].sum()
    return torch.autograd.grad(total, list(table.values()))


@pytest.mark.parametrize("method", AM1_AND_OM2_METHODS)
def test_implicit_scf_backward_matches_direct_backprop_for_parameters(device, methanal_batch_data, method):
    species, coordinates = methanal_batch_data

    implicit = _parameter_gradients(device, species, coordinates, method, 1)
    direct = _parameter_gradients(device, species, coordinates, method, 2)

    for name, grad_implicit, grad_direct in zip(_LEARNED, implicit, direct):
        torch.testing.assert_close(
            grad_implicit, grad_direct, rtol=1.0e-4, atol=1.0e-6, msg=lambda m, name=name: f"{name}: {m}"
        )

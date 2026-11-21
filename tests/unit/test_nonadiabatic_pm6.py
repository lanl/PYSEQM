import pytest
import torch

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

from .test_nonadiabatic_checkpoint_resume import _compare_runs, _run_continuous, _seqm_parameters


@pytest.mark.parametrize("tdc_method", ["hamiltonian_fd", "overlap"])
def test_nonadiabatic_dynamics_runs_with_pm6(methanal_batch_data, device, tmp_path, tdc_method):
    species, coordinates = methanal_batch_data
    runs = {}
    for method in ("PM6", "PM6_SP"):
        parameters = _seqm_parameters(method, excited=True)
        parameters["nonadiabatic"] = {"tdc_method": tdc_method}
        prefix = str(tmp_path / method)
        runs[method] = _run_continuous(
            parameters, device, species.clone(), coordinates.clone(), prefix, [0, 1], 3
        )
    _compare_runs(runs["PM6"], runs["PM6_SP"], 3)


def test_excited_states_with_d_orbitals_are_rejected():
    torch.set_default_dtype(torch.float64)
    parameters = {
        "method": "PM6",
        "scf_eps": 1e-8,
        "scf_converger": [1],
        "excited_states": {"n_states": 3, "method": "cis"},
    }
    phosphine = [[0, 0, 0], [1.2, 0.5, 0.6], [-1.2, 0.5, 0.6], [0, -1.3, 0.6]]
    molecule = Molecule(Constants(), parameters, torch.tensor([phosphine]), torch.tensor([[15, 1, 1, 1]]))
    with pytest.raises(NotImplementedError, match="Excited states are not implemented with d orbitals"):
        Electronic_Structure(parameters)(molecule)

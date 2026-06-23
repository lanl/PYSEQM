from types import SimpleNamespace

import pytest
import torch

from seqm.dynamics.tdc_hamiltonian_fd import compute_tdc_hamiltonian_fd
from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

from ..reference_data import OMX_METHODS


@pytest.mark.parametrize("method", OMX_METHODS)
def test_tdc_hamiltonian_fd_matches_nac_component_methane_batch(device, methane_molecule_data, method):
    species_1, coordinates_1 = methane_molecule_data
    const = Constants().to(device)

    # Build a batch of two methanes; second geometry is slightly displaced.
    species = torch.cat((species_1, species_1.clone()), dim=0)
    coordinates_2 = coordinates_1.clone()
    coordinates_2[0, 0, 0] += 5.0e-1
    coordinates = torch.cat((coordinates_1, coordinates_2), dim=0)

    n_states = 4
    seqm_parameters = {
        "method": method,
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "excited_states": {"n_states": n_states, "method": "cis", "tolerance": 1.0e-6},
        "nonadiabatic": {"compute_nac": True, "states": [1, 2, 3, 4], "include_response_terms": False},
        "active_state": 1,
    }

    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    molecule.verbose = False
    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecule)

    assert molecule.nac is not None
    assert molecule.cis_amplitudes is not None
    assert molecule.cis_energies is not None

    nad = SimpleNamespace(_dtnact=1.0e-5, damp=None, _tdc_energy_tol=1.0e-10, timestep=0.1)
    cache_new = {"cis_amp": molecule.cis_amplitudes, "energies": molecule.cis_energies}

    idx_i, idx_j = torch.triu_indices(n_states, n_states, offset=1, device=device)
    real_atoms = (species[0] > 0).nonzero(as_tuple=False).squeeze(1)

    vel_old = torch.zeros_like(molecule.coordinates)
    acc_old = torch.zeros_like(molecule.coordinates)
    velocity_rows = torch.tensor(
        [[1.0, -0.25, 0.50], [-0.40, 0.20, 0.10], [0.30, -0.10, 0.70]], dtype=vel_old.dtype, device=device
    )
    for atom, row in zip(real_atoms[: velocity_rows.shape[0]].tolist(), velocity_rows):
        vel_old[:, atom, :] = row

    nac_dt = compute_tdc_hamiltonian_fd(
        nad, molecule, cache_new, learned_parameters={}, vel_old=vel_old, acc_old=acc_old
    )

    tdc_upper = nac_dt[:, :n_states, :n_states][:, idx_i, idx_j]
    expected_pairs = []
    for i, j in zip(idx_i.tolist(), idx_j.tolist()):
        vec = molecule.nac.get((i, j))
        if vec is None:
            vec = torch.zeros((species.shape[0], molecule.molsize, 3), dtype=tdc_upper.dtype, device=device)
        expected_pairs.append((vec * vel_old).sum(dim=(1, 2)))
    expected_upper = torch.stack(expected_pairs, dim=1)
    diff = torch.abs(tdc_upper - expected_upper)
    rel = diff / torch.clamp(torch.abs(expected_upper), min=1.0e-12)
    max_abs_diff = float(diff.max().item())
    max_rel_diff = float(rel.max().item())

    assert max_abs_diff < 1.0e-4, f"Max abs diff too large: {max_abs_diff:.6e}"
    assert max_rel_diff < 1.0e-6, f"Max rel diff too large: {max_rel_diff:.6e}"

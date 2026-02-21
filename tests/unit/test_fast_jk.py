import pytest

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

from ..reference_data import assert_allclose


def _run_single_point(device, species, coordinates, params):
    const = Constants().to(device)
    molecule = Molecule(const, params, coordinates, species).to(device)
    esdriver = Electronic_Structure(params).to(device)
    esdriver(molecule)
    return {
        "Etot": molecule.Etot.detach().cpu(),
        "Eelec": molecule.Eelec.detach().cpu(),
        "Enuc": molecule.Enuc.detach().cpu(),
        "force": molecule.force.detach().cpu(),
    }


@pytest.mark.parametrize("method", ["AM1", "PM3", "PM6_SP"])
def test_fast_jk_matches_baseline_single(method, device, methane_molecule_data):
    species, coordinates = methane_molecule_data
    base_params = {"method": method, "scf_eps": 1.0e-6, "scf_converger": [1]}
    fast_params = {"method": method, "scf_eps": 1.0e-6, "scf_converger": [1], "fast_jk": True}

    base = _run_single_point(device, species, coordinates, base_params)
    fast = _run_single_point(device, species, coordinates, fast_params)

    assert_allclose(fast["Etot"], base["Etot"], rtol=1e-9, atol=1e-9)
    assert_allclose(fast["Eelec"], base["Eelec"], rtol=1e-9, atol=1e-9)
    assert_allclose(fast["Enuc"], base["Enuc"], rtol=1e-9, atol=1e-9)
    assert_allclose(fast["force"], base["force"], rtol=1e-9, atol=1e-9)


def test_fast_jk_matches_baseline_batch(device, batch_molecule_data):
    species, coordinates = batch_molecule_data
    base_params = {"method": "AM1", "scf_eps": 1.0e-6, "scf_converger": [1]}
    fast_params = {"method": "AM1", "scf_eps": 1.0e-6, "scf_converger": [1], "fast_jk": True}

    base = _run_single_point(device, species, coordinates, base_params)
    fast = _run_single_point(device, species, coordinates, fast_params)

    assert_allclose(fast["Etot"], base["Etot"], rtol=1e-9, atol=1e-9)
    assert_allclose(fast["Eelec"], base["Eelec"], rtol=1e-9, atol=1e-9)
    assert_allclose(fast["Enuc"], base["Enuc"], rtol=1e-9, atol=1e-9)
    assert_allclose(fast["force"], base["force"], rtol=1e-9, atol=1e-9)


def test_fast_jk_matches_baseline_uhf(device, methane_molecule_data):
    species, coordinates = methane_molecule_data
    base_params = {"method": "AM1", "scf_eps": 1.0e-6, "scf_converger": [1], "UHF": True}
    fast_params = {"method": "AM1", "scf_eps": 1.0e-6, "scf_converger": [1], "UHF": True, "fast_jk": True}

    base = _run_single_point(device, species, coordinates, base_params)
    fast = _run_single_point(device, species, coordinates, fast_params)

    assert_allclose(fast["Etot"], base["Etot"], rtol=1e-9, atol=1e-9)
    assert_allclose(fast["Eelec"], base["Eelec"], rtol=1e-9, atol=1e-9)
    assert_allclose(fast["Enuc"], base["Enuc"], rtol=1e-9, atol=1e-9)
    assert_allclose(fast["force"], base["force"], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("excited_method", ["cis", "rpa"])
def test_fast_jk_matches_baseline_excited_states(device, methane_molecule_data, excited_method):
    species, coordinates = methane_molecule_data
    base_params = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "excited_states": {"n_states": 4, "method": excited_method},
        "active_state": 1,
    }
    fast_params = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "excited_states": {"n_states": 4, "method": excited_method},
        "active_state": 1,
        "fast_jk": True,
    }

    const = Constants().to(device)
    mol_base = Molecule(const, base_params, coordinates, species).to(device)
    mol_fast = Molecule(const, fast_params, coordinates, species).to(device)

    es_base = Electronic_Structure(base_params).to(device)
    es_fast = Electronic_Structure(fast_params).to(device)
    es_base(mol_base)
    es_fast(mol_fast)

    assert_allclose(
        mol_fast.cis_energies.detach().cpu(), mol_base.cis_energies.detach().cpu(), rtol=1e-7, atol=1e-7
    )
    assert_allclose(
        mol_fast.oscillator_strength.detach().cpu(),
        mol_base.oscillator_strength.detach().cpu(),
        rtol=1e-7,
        atol=1e-7,
    )
    assert_allclose(mol_fast.force.detach().cpu(), mol_base.force.detach().cpu(), rtol=1e-7, atol=1e-7)

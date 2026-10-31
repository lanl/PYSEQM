import pytest
import torch

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants
from seqm.seqm_functions.dispersion import dispersion_energy, dispersion_kind, dispersion_pair_grad

# H2O...CH4 complex, batched with a (padded) water. Dispersion-only energy (kcal/mol) and gradient (kcal/mol/A:
# component of the O atom and norm over all atoms) of MNDO2020 = gradient with the correction minus without it.
_SPECIES = [[8, 6, 1, 1, 1, 1, 1, 1], [8, 1, 1, 0, 0, 0, 0, 0]]
_XYZ = [[0, 0, 0], [0.2, 0.3, 3.4], [0, 0, 0.96], [0.93, 0, -0.24], [0.2, 1.38, 3.4]]
_XYZ += [[0.56, -0.06, 4.37], [-0.82, -0.06, 3.23], [0.79, -0.39, 2.75]]
_MNDO = {  # (method, kind): energy, grad_O, grad_norm
    ("OM2", "D2-ELSTNER"): (-0.79499, (0.0084, -0.0081, -0.1339), 1.0045),
    ("OM2", "D2"): (-1.17620, (-0.0675, -0.0052, -0.3234), 0.5642),
    ("OM2", "D3"): (-1.37330, (-0.0042, -0.0092, -0.1511), 0.2662),
    ("OM2", "D3T"): (-1.37278, (-0.0047, -0.0092, -0.1510), 0.2668),
    ("OM3", "D2-ELSTNER"): (-1.38189, (-0.0402, -0.0030, -0.3997), 2.2889),
    ("OM3", "D2"): (-1.09383, (-0.0362, -0.0031, -0.3829), 1.4059),
    ("OM3", "D3"): (-2.12922, (-0.0107, -0.0110, -0.2458), 0.4528),
    ("OM3", "D3T"): (-2.12870, (-0.0112, -0.0110, -0.2457), 0.4537),
}
_KCAL = 23.061


def _molecule(method, **extra):
    torch.set_default_dtype(torch.float64)
    xyz = torch.tensor(_XYZ, dtype=torch.float64)
    coordinates = torch.stack([xyz, torch.cat([xyz[[0, 2, 3]], torch.zeros(5, 3)])])
    species = torch.tensor(_SPECIES)
    if "excited_states" in extra:  # excited states need a uniform batch: drop the padded water
        species, coordinates = species[:1], coordinates[:1]
    params = {
        "method": method,
        "scf_eps": 1.0e-11,
        "scf_converger": [1],
        "elements": [0, 1, 6, 7, 8],
        **extra,
    }
    return params, Molecule(Constants(), params, coordinates, species)


@pytest.mark.parametrize("method, kind", _MNDO)
def test_dispersion_matches_mndo(method, kind):
    energy, grad_O, norm = _MNDO[method, kind]
    _, mol = _molecule(method, dispersion=kind)
    pair_grad = dispersion_pair_grad(mol)
    grad = torch.zeros(len(mol.Z), 3).index_add_(0, mol.idxi, pair_grad).index_add_(0, mol.idxj, -pair_grad)
    assert dispersion_energy(mol, None)[0].item() * _KCAL == pytest.approx(energy, abs=2.0e-5)
    assert (grad[0] * _KCAL).tolist() == pytest.approx(grad_O, abs=2.0e-4)
    assert (grad[:8] * _KCAL).norm().item() == pytest.approx(norm, abs=2.0e-4)


@pytest.mark.parametrize(
    "extra",
    [{}, {"excited_states": {"n_states": 4, "method": "cis", "tolerance": 1.0e-10}, "active_state": 2}],
    ids=["ground", "cis"],
)
def test_analytic_force_matches_autodiff(extra):
    forces = []
    for grad in ({"scf_backward": 1}, {"analytical_gradient": [True]}):
        params, mol = _molecule("OM3", dispersion="D3T", **extra, **grad)
        if "scf_backward" in grad:
            mol.coordinates.requires_grad_()
        Electronic_Structure(params)(mol)
        forces.append(mol.force)
    torch.testing.assert_close(forces[0], forces[1], rtol=0.0, atol=1.0e-6)


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"method": "AM1", "dispersion": "fs1"}, "FS1"),
        ({"method": "OM2", "dispersion": "d3t"}, "D3T"),
        ({"method": "OM3", "dispersion": False}, None),
        ({"method": "AM1", "dispersion": "D3"}, ValueError),
        ({"method": "AM1", "dispersion": True}, ValueError),
        ({"method": "OM2", "dispersion": "D4"}, ValueError),
        ({"method": "OM2", "dispersion": "D3", "pair_outer_cutoff": 5.0}, ValueError),
    ],
)
def test_dispersion_kind(params, expected):
    if expected is ValueError:
        with pytest.raises(ValueError):
            dispersion_kind(params)
    else:
        assert dispersion_kind(params) == expected

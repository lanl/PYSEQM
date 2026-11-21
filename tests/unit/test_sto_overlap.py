"""Tests of the STO overlap integrals in seqm.seqm_functions.diat_overlapD.

The local-frame integrals are checked against direct numerical quadrature
(independent of the prolate-spheroidal algebra used by the implementation),
the molecular-frame rotation against numerically determined harmonic rotation
matrices, the derivatives with gradcheck and the energies against MOPAC.
"""

import math

import numpy as np
import pytest
import torch

from seqm.seqm_functions.diat_overlapD import diatom_overlap_matrixD

DTYPE = torch.float64
FOUR_PI = 4.0 * math.pi

# --- numerical oracle ------------------------------------------------------
# real harmonics in PYSEQM order: s, px, py, pz, dx2-y2, dxz, dz2, dyz, dxy  (functions of x, y, z, r)
HARMONICS = [
    lambda x, y, z, r: np.full_like(r, 1.0 / math.sqrt(FOUR_PI)),
    lambda x, y, z, r: math.sqrt(3.0 / FOUR_PI) * x / r,
    lambda x, y, z, r: math.sqrt(3.0 / FOUR_PI) * y / r,
    lambda x, y, z, r: math.sqrt(3.0 / FOUR_PI) * z / r,
    lambda x, y, z, r: math.sqrt(15.0 / (4 * math.pi) / 4.0) * (x * x - y * y) / r**2,
    lambda x, y, z, r: math.sqrt(15.0 / FOUR_PI) * x * z / r**2,
    lambda x, y, z, r: math.sqrt(5.0 / (4 * math.pi) / 4.0) * (3 * z * z - r * r) / r**2,
    lambda x, y, z, r: math.sqrt(15.0 / FOUR_PI) * y * z / r**2,
    lambda x, y, z, r: math.sqrt(15.0 / FOUR_PI) * x * y / r**2,
]
L_OF = [0, 1, 1, 1, 2, 2, 2, 2, 2]
M_OF = [0, 1, 1, 0, 2, 1, 0, 1, 2]
TYPE_OF = ["", "c", "s", "", "c", "c", "", "s", "s"]  # cos/sin type of the phi dependence (m = 0: none)
# sin-type orbitals give the same integral as their cos-type partner (rotation by 90 degrees about z)
COS_PARTNER = {2: 1, 7: 5, 8: 4}


def _radial(n, zeta, r):
    return (2 * zeta) ** (n + 0.5) / math.sqrt(math.factorial(2 * n)) * r ** (n - 1) * np.exp(-zeta * r)


def _graded_edges(centres, lo, hi):
    """Panel edges between lo and hi, geometrically refined towards every centre."""
    edges = {lo, hi, *centres}
    for c in centres:
        for d in (0.02, 0.06, 0.15, 0.35, 0.7, 1.2, 2.0, 3.2, 5.0, 8.0, 12.0, 18.0, 26.0):
            for sgn in (-1, 1):
                if lo <= c + sgn * d <= hi:
                    edges.add(c + sgn * d)
    return sorted(edges)


def _gauss_legendre(edges, order=20):
    x, w = np.polynomial.legendre.leggauss(order)
    pts, wts = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pts.append(0.5 * (hi - lo) * x + 0.5 * (hi + lo))
        wts.append(0.5 * (hi - lo) * w)
    return np.concatenate(pts), np.concatenate(wts)


def numeric_local_overlap(rb, na, za, ia, nb, zb, ib):
    """<orbital ia on A (origin) | orbital ib on B (+z, distance rb)> by quadrature in cylindrical coordinates.

    The phi integral is analytic: factor 2 pi for m = 0 and pi otherwise, with both orbitals evaluated in
    the xz plane (x = rho).
    """
    rho, w_rho = _gauss_legendre(_graded_edges([0.0], 0.0, 40.0))
    z, w_z = _gauss_legendre(_graded_edges([0.0, rb], -40.0, rb + 40.0))
    rr, zz = np.meshgrid(rho, z, indexing="ij")
    weight = w_rho[:, None] * w_z[None, :] * rr
    ia, ib = COS_PARTNER.get(ia, ia), COS_PARTNER.get(ib, ib)
    ra, rb_ = np.hypot(rr, zz), np.hypot(rr, zz - rb)
    fa = _radial(na, za, ra) * HARMONICS[ia](rr, 0.0 * rr, zz, ra)
    fb = _radial(nb, zb, rb_) * HARMONICS[ib](rr, 0.0 * rr, zz - rb, rb_)
    return float(np.sum(weight * fa * fb) * (2 * math.pi if M_OF[ia] == 0 else math.pi))


def _tables(shells):
    """qn_int, qnD_int indexed by 'atomic number' = 1 + position in shells; shells = [(qn, qnd), ...]"""
    qn = torch.zeros(90, dtype=torch.int64)
    qd = torch.zeros(90, dtype=torch.int64)
    for k, (q, d) in enumerate(shells):
        qn[k + 1], qd[k + 1] = q, d
    return qn, qd


def _overlap(shells, zetas, rb, direction=(0.0, 0.0, 1.0)):
    qn, qd = _tables(shells)
    u = torch.tensor([direction], dtype=DTYPE)
    u = u / u.norm()
    return diatom_overlap_matrixD(
        torch.tensor([1]),
        torch.tensor([2]),
        u,
        torch.tensor([rb], dtype=DTYPE),
        torch.tensor([zetas[0]], dtype=DTYPE),
        torch.tensor([zetas[1]], dtype=DTYPE),
        qn,
        qd,
    )[0].numpy()


def _shell_n(shell, orbital):
    qn, qd = shell
    return qd if L_OF[orbital] == 2 else qn


def _has(shell, orbital):
    qn, qd = shell
    l = L_OF[orbital]
    return qn >= 1 if l == 0 else qn >= 2 if l == 1 else qd >= 1


def _same_block(i, j):
    """Pairs of local orbitals with a non-zero overlap when the bond is along z."""
    if M_OF[i] != M_OF[j]:
        return False
    return TYPE_OF[i] == TYPE_OF[j]


# --- tests -----------------------------------------------------------------
# zeta values chosen differently on the two atoms and for each shell
ZETA_A = [2.1, 1.7, 1.4]
ZETA_B = [1.6, 2.3, 1.9]
# ((qn of s/p shell, qn of d shell) of atom A, same for atom B): 3d-3d, 5s5p, 5d-1s, 4d-5s5p, 6s6p5d with 3d and 6s6p5d
SHELL_PAIRS = [
    ((3, 3), (3, 3)),
    ((5, 0), (5, 0)),
    ((5, 5), (1, 0)),
    ((4, 4), (5, 0)),
    ((6, 5), (3, 3)),
    ((6, 5), (6, 5)),
]


@pytest.mark.parametrize("shell_a, shell_b", SHELL_PAIRS)
def test_local_overlaps_match_quadrature(shell_a, shell_b):
    """Full 9x9 local-frame overlap (bond along +z) against numerical integration."""
    rb = 4.2
    s = _overlap([shell_a, shell_b], (ZETA_A, ZETA_B), rb)
    for i in range(9):
        for j in range(9):
            exists = _has(shell_a, i) and _has(shell_b, j)
            if not exists or not _same_block(i, j):
                assert abs(s[i, j]) < 1e-12, (i, j, s[i, j])
                continue
            za = ZETA_A[0 if L_OF[i] == 0 else L_OF[i]]
            zb = ZETA_B[0 if L_OF[j] == 0 else L_OF[j]]
            ref = numeric_local_overlap(rb, _shell_n(shell_a, i), za, i, _shell_n(shell_b, j), zb, j)
            assert s[i, j] == pytest.approx(ref, abs=2e-6), (shell_a, shell_b, i, j)


def _harmonic_rotation(rg, rng):
    """Matrix D with Y_mu(Rg r) = sum_k D[mu, k] Y_k(r), fitted at random points."""
    pts = rng.normal(size=(80, 3))
    r = np.linalg.norm(pts, axis=1)
    rot = pts @ rg.T
    y_rot = np.stack([h(rot[:, 0], rot[:, 1], rot[:, 2], r) for h in HARMONICS], axis=1)
    y = np.stack([h(pts[:, 0], pts[:, 1], pts[:, 2], r) for h in HARMONICS], axis=1)
    d, *_ = np.linalg.lstsq(y, y_rot, rcond=None)
    return d.T


def _rotation_taking_z_to(u, rng):
    u = u / np.linalg.norm(u)
    a = rng.normal(size=3)
    e1 = a - a.dot(u) * u
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    # columns (e1, e2, u) rotate z -> u; add an extra random spin about u so the frame choice is not special
    t = rng.uniform(0, 2 * math.pi)
    f1, f2 = math.cos(t) * e1 + math.sin(t) * e2, -math.sin(t) * e1 + math.cos(t) * e2
    return np.stack([f1, f2, u], axis=1)


def test_rotation_covariance():
    """S(u) = D S(z) D^T with D determined numerically from the real harmonics (d-d pair)."""
    shells = [(3, 3), (3, 3)]
    rng = np.random.default_rng(7)
    rb = 4.5
    s_z = _overlap(shells, (ZETA_A, ZETA_B), rb)
    for _ in range(6):
        u = rng.normal(size=3)
        d = _harmonic_rotation(_rotation_taking_z_to(u, rng), rng)
        got = _overlap(shells, (ZETA_A, ZETA_B), rb, direction=tuple(u))
        np.testing.assert_allclose(got, d @ s_z @ d.T, atol=1e-10)


def test_gradients_wrt_coordinates_and_exponents():
    """Autograd derivatives agree with finite differences for axis-aligned bonds and beta = 0."""
    qn, qd = _tables([(3, 3), (3, 3)])
    ni, nj = torch.tensor([1, 1, 1]), torch.tensor([2, 2, 2])
    za = torch.tensor([ZETA_A] * 3, dtype=DTYPE)
    zb = torch.tensor([ZETA_A, ZETA_B, ZETA_B], dtype=DTYPE)  # first pair: equal exponents (beta = 0)
    pos = torch.tensor([[0.0, 0.0, 3.5], [2.1, -1.3, 2.9], [3.2, 0.0, 0.0]], dtype=DTYPE)

    def fn(p, ze_a, ze_b):
        r = torch.sqrt((p * p).sum(dim=1))
        return diatom_overlap_matrixD(ni, nj, p / r.unsqueeze(1), r, ze_a, ze_b, qn, qd)

    args = (pos.requires_grad_(True), za.requires_grad_(True), zb.requires_grad_(True))
    assert torch.autograd.gradcheck(fn, args, eps=1e-6, atol=1e-6, rtol=1e-5)


# --- end-to-end PM6 checks -------------------------------------------------
def _pm6_heat_of_formation(species, coordinates):
    from seqm.ElectronicStructure import Electronic_Structure
    from seqm.Molecule import Molecule
    from seqm.seqm_functions.constants import Constants

    torch.set_default_dtype(DTYPE)
    params = {
        "method": "PM6",
        "scf_eps": 1.0e-10,
        "scf_converger": [0, 0.2],
        "sp2": [False, 1.0e-5],
        "pair_outer_cutoff": 1.0e10,
        "eig": True,
        "Hf_flag": True,
        "scf_backward": 0,
        "UHF": False,
    }
    molecule = Molecule(
        Constants(), params, torch.tensor([coordinates], dtype=DTYPE), torch.tensor([species])
    )
    Electronic_Structure(params)(molecule)
    return molecule.Hf.item() * 23.060547830619  # eV -> kcal/mol


def test_pm6_heat_of_formation_matches_mopac_for_iodine():
    """PM6 heat of formation of I2 (5s5p5d) at 2.70 A against MOPAC 2016 (17.64885 kcal/mol)."""
    assert _pm6_heat_of_formation([53, 53], [[0, 0, 0], [0, 0, 2.70]]) == pytest.approx(17.64885, abs=0.15)


def test_pm6_energy_is_independent_of_bond_direction():
    """Cl2 has d orbitals on both atoms; its energy must not depend on the orientation."""
    energies = []
    for direction in ((0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0.3, -0.7, 0.4)):
        u = np.array(direction, dtype=float)
        u = 2.0 * u / np.linalg.norm(u)
        energies.append(_pm6_heat_of_formation([17, 17], [[0.0, 0.0, 0.0], u.tolist()]))
    assert max(energies) - min(energies) < 1e-8

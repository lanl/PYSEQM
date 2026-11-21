"""
Overlap integrals between Slater-type orbitals (s, p, d) on two centers.

The overlap of normalised STOs
    chi = N r^(n-1) exp(-zeta r) Y_lm(theta, phi),   N = (2 zeta)^(n+1/2) / sqrt((2n)!)
is evaluated for *any* principal quantum numbers (n_a, n_b) and l <= 2 with one
general expression, instead of one hand-expanded formula per (n, l) combination.

Local frame (z axis along the bond, atom A at the origin, atom B at +R z):
prolate spheroidal coordinates mu = (r_a + r_b)/R, nu = (r_a - r_b)/R give

    S = N_a N_b (R/2)^(n_a+n_b+1) sum_ij C_ij A_i(alpha) B_j(beta),
    A_i(alpha) = int_1^inf mu^i exp(-alpha mu) dmu,   alpha = R (zeta_a + zeta_b)/2
    B_j(beta)  = int_-1^1  nu^j exp(-beta  nu) dnu,   beta  = R (zeta_a - zeta_b)/2

The integer-valued polynomial coefficients C_ij (times angular normalisation)
are built once with numpy from the solid harmonics, see `_coefficient_table`.
Only sigma (m=0), pi (m=1) and delta (m=2) integrals are non-zero in the local
frame; they are rotated to the molecular frame with explicit real spherical
harmonic rotation matrices (l = 1: the 3x3 rotation, l = 2: its induced 5x5
representation), which is exact for every bond direction.

Orbital order (PYSEQM/MOPAC): 0 s, 1 px, 2 py, 3 pz, 4 dx2-y2, 5 dxz, 6 dz2, 7 dyz, 8 dxy.
"""

import math
from functools import lru_cache

import numpy as np
import torch as th

_NMAX = 6  # largest principal quantum number supported
_NPOLY = 2 * _NMAX + 2  # A_i, B_j are needed for i, j = 0 .. n_a + n_b + 1 <= 13
_BETA_SERIES_MAX = 16.0  # |beta| below this: Taylor series for B_j, above: upward recursion
_BETA_SERIES_TERMS = 100

# (l_a, l_b, m) of the independent local-frame integrals
_SPECS = (
    (0, 0, 0),
    (0, 1, 0),
    (0, 2, 0),
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
    (1, 2, 0),
    (1, 2, 1),
    (2, 0, 0),
    (2, 1, 0),
    (2, 1, 1),
    (2, 2, 0),
    (2, 2, 1),
    (2, 2, 2),
)

# local-frame orbital indices (in the 9-orbital ordering) for every (l, m)
_LOCAL_INDEX = {0: {0: (0,)}, 1: {0: (3,), 1: (1, 2)}, 2: {0: (6,), 1: (5, 7), 2: (4, 8)}}

# normalisation of the real harmonics: Y_lm = _K[l][m] * (polynomial)
_K = {
    0: {0: 1.0 / math.sqrt(4.0 * math.pi)},
    1: {0: math.sqrt(3.0 / (4.0 * math.pi)), 1: math.sqrt(3.0 / (4.0 * math.pi))},
    2: {
        0: math.sqrt(5.0 / (16.0 * math.pi)),
        1: math.sqrt(15.0 / (4.0 * math.pi)),
        2: math.sqrt(15.0 / (16.0 * math.pi)),
    },
}


# ---------------------------------------------------------------------------
# polynomial helpers: p[i, j] is the coefficient of mu^i nu^j
# ---------------------------------------------------------------------------
def _pmul(a, b):
    out = np.zeros((a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1))
    for i, j in zip(*np.nonzero(a)):
        out[i : i + b.shape[0], j : j + b.shape[1]] += a[i, j] * b
    return out


def _padd(a, b, sb=1.0):
    shape = (max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1]))
    out = np.zeros(shape)
    out[: a.shape[0], : a.shape[1]] += a
    out[: b.shape[0], : b.shape[1]] += sb * b
    return out


def _ppow(a, n):
    out = np.ones((1, 1))
    for _ in range(n):
        out = _pmul(out, a)
    return out


def _poly(entries):
    shape = (max(i for i, _ in entries) + 1, max(j for _, j in entries) + 1)
    out = np.zeros(shape)
    for (i, j), c in entries.items():
        out[i, j] = c
    return out


_ONE = _poly({(0, 0): 1.0})
_RA = _poly({(1, 0): 1.0, (0, 1): 1.0})  # r_a / (R/2) = mu + nu
_RB = _poly({(1, 0): 1.0, (0, 1): -1.0})  # r_b / (R/2) = mu - nu
_ZA = _poly({(0, 0): 1.0, (1, 1): 1.0})  # z_a / (R/2) = 1 + mu nu
_ZB = _poly({(0, 0): -1.0, (1, 1): 1.0})  # z_b / (R/2) = mu nu - 1
_RHO2 = _pmul(_poly({(2, 0): 1.0, (0, 0): -1.0}), _poly({(0, 0): 1.0, (0, 2): -1.0}))  # (mu^2-1)(1-nu^2)
_VOL = _poly({(2, 0): 1.0, (0, 2): -1.0})  # mu^2 - nu^2


def _angular_poly(l, m, z, r):
    """Solid harmonic r^l Y_lm / K with the rho^m (cos m phi) factor removed."""
    if l == 0:
        return _ONE
    if l == 1:
        return z if m == 0 else _ONE
    if l == 2:
        if m == 0:  # 3 z^2 - r^2
            return _padd(3.0 * _pmul(z, z), _pmul(r, r), -1.0)
        return z if m == 1 else _ONE
    raise ValueError("only s, p and d orbitals are supported")


@lru_cache(maxsize=None)
def _coefficient_table():
    """C[k, na, nb, i, j] for the 14 local integrals and n = 0 .. 6 (n = 0 means 'no orbital')."""
    table = np.zeros((len(_SPECS), _NMAX + 1, _NMAX + 1, _NPOLY, _NPOLY))
    for k, (la, lb, m) in enumerate(_SPECS):
        const = _K[la][m] * _K[lb][m] * (2.0 * math.pi if m == 0 else math.pi)
        base = _pmul(_pmul(_angular_poly(la, m, _ZA, _RA), _angular_poly(lb, m, _ZB, _RB)), _ppow(_RHO2, m))
        base = _pmul(base, _VOL)
        for na in range(la + 1, _NMAX + 1):
            for nb in range(lb + 1, _NMAX + 1):
                poly = _pmul(_pmul(base, _ppow(_RA, na - 1 - la)), _ppow(_RB, nb - 1 - lb))
                table[k, na, nb, : poly.shape[0], : poly.shape[1]] = const * poly
    return table


@lru_cache(maxsize=None)
def _series_matrix():
    """W[k, j] = 2/(j+k+1) for even j+k, else 0:  B_j(beta) = sum_k (-beta)^k/k! W[k, j]."""
    k = np.arange(_BETA_SERIES_TERMS)[:, None]
    j = np.arange(_NPOLY)[None, :]
    return np.where((j + k) % 2 == 0, 2.0 / (j + k + 1.0), 0.0)


# real d-orbital quadratic forms Y_k = c_k r^T M_k r, order dx2-y2, dxz, dz2, dyz, dxy
_DM = np.zeros((5, 3, 3))
_DM[0] = np.diag([1.0, -1.0, 0.0])
_DM[1][0, 2] = _DM[1][2, 0] = 0.5
_DM[2] = np.diag([-1.0, -1.0, 2.0])
_DM[3][1, 2] = _DM[3][2, 1] = 0.5
_DM[4][0, 1] = _DM[4][1, 0] = 0.5
_DC = np.array(
    [
        math.sqrt(15.0 / (16.0 * math.pi)),
        math.sqrt(15.0 / (4.0 * math.pi)),
        math.sqrt(5.0 / (16.0 * math.pi)),
        math.sqrt(15.0 / (4.0 * math.pi)),
        math.sqrt(15.0 / (4.0 * math.pi)),
    ]
)
_DNORM = np.einsum("kab,kba->k", _DM, _DM)
_SQRT_FACT = np.array([math.sqrt(math.factorial(2 * n)) for n in range(_NMAX + 1)])


# ---------------------------------------------------------------------------
# radial auxiliary integrals
# ---------------------------------------------------------------------------
def _a_integrals(alpha):
    """A_i(alpha) = int_1^inf mu^i exp(-alpha mu) dmu, i = 0 .. _NPOLY-1; alpha > 0."""
    inv = alpha.reciprocal()
    a0 = th.exp(-alpha) * inv
    out = [a0]
    for i in range(1, _NPOLY):
        out.append(a0 + i * inv * out[-1])
    return th.stack(out, dim=-1)


def _b_integrals(beta):
    """B_j(beta) = int_-1^1 nu^j exp(-beta nu) dnu, j = 0 .. _NPOLY-1 (smooth at beta = 0)."""
    dtype, device = beta.dtype, beta.device
    small = beta.abs() < _BETA_SERIES_MAX

    # Taylor series (all terms have one sign, so no cancellation)
    bs = th.where(small, beta, th.zeros_like(beta))
    k = th.arange(1, _BETA_SERIES_TERMS, dtype=dtype, device=device)
    factors = th.cat([th.ones_like(bs).unsqueeze(-1), -bs.unsqueeze(-1) / k], dim=-1)
    powers = th.cumprod(factors, dim=-1)
    w = th.as_tensor(_series_matrix(), dtype=dtype, device=device)
    b_series = powers @ w

    # upward recursion, stable for large |beta|
    bl = th.where(small, th.full_like(beta, _BETA_SERIES_MAX), beta)
    inv = bl.reciprocal()
    ep, em = th.exp(bl), th.exp(-bl)
    out = [(ep - em) * inv]
    for j in range(1, _NPOLY):
        sign = -1.0 if j % 2 else 1.0
        out.append((sign * ep - em) * inv + j * inv * out[-1])
    b_rec = th.stack(out, dim=-1)

    return th.where(small.unsqueeze(-1), b_series, b_rec)


# ---------------------------------------------------------------------------
# rotation to the molecular frame
# ---------------------------------------------------------------------------
def _bond_frame(u):
    """Right-handed orthonormal frame (e1, e2, u) as columns, shape (npairs, 3, 3).

    e1 is built from the coordinate axis least aligned with u (chosen without
    gradient), so the frame is well conditioned for every bond direction.  The
    overlap does not depend on the choice of e1, e2.
    """
    axis = u.detach().abs().argmin(dim=1)
    h = th.nn.functional.one_hot(axis, 3).to(u.dtype)
    e1 = h - (h * u).sum(dim=1, keepdim=True) * u
    e1 = e1 / th.sqrt((e1 * e1).sum(dim=1, keepdim=True))
    e2 = th.cross(u, e1, dim=1)
    return th.stack([e1, e2, u], dim=2)


def _rotation_matrices(frame):
    """Block-diagonal (s, p, d) representation of the frame rotation, shape (npairs, 9, 9)."""
    dtype, device = frame.dtype, frame.device
    npairs = frame.shape[0]
    mats = th.as_tensor(_DM, dtype=dtype, device=device)
    coef = th.as_tensor(_DC, dtype=dtype, device=device)
    norm = th.as_tensor(_DNORM, dtype=dtype, device=device)

    # R^T M_k R expanded in the (orthogonal) basis M_l
    t = th.einsum("pba,kbc,pcd->pkad", frame, mats, frame)
    r2 = th.einsum("pkad,lad->pkl", t, mats) / norm
    r2 = r2 * (coef.unsqueeze(1) / coef.unsqueeze(0))

    rot = th.zeros((npairs, 9, 9), dtype=dtype, device=device)
    rot[:, 0, 0] = 1.0
    rot[:, 1:4, 1:4] = frame
    rot[:, 4:9, 4:9] = r2
    return rot


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def diatom_overlap_matrixD(ni, nj, xij, rij, zeta_a, zeta_b, qn_int, qnD_int, qnS_int=None, qnP_int=None):
    """
    Overlap matrix between the s, p, d valence orbitals of every atom pair.

    ni, nj     : atomic numbers of atom i and j, shape (npairs,)
    xij        : unit vector pointing from i to j, shape (npairs, 3)
    rij        : distance between i and j in bohr, shape (npairs,)
    zeta_a/b   : Slater exponents (s, p, d) of atom i / j, shape (npairs, 3)
    qn_int     : principal quantum number of the s and p shell, indexed by atomic number
    qnD_int    : principal quantum number of the d shell (0: no d shell), indexed by atomic number
    qnS_int, qnP_int : optional principal quantum numbers of the s and p shell (0: no p shell), indexed by
                 atomic number, for elements where they differ from qn_int (the noble gases in MOPAC's npq).
                 Default: s = qn_int, p = qn_int (absent for qn_int < 2).

    Returns S[pair, mu_i, nu_j] with shape (npairs, 9, 9) in the molecular frame.
    Orbitals that an element does not have (p on H, d on s/p elements) give zero rows/columns.
    """
    dtype, device = xij.dtype, xij.device
    npairs = xij.shape[0]
    if npairs == 0:
        return th.zeros((0, 9, 9), dtype=dtype, device=device)

    if qnS_int is None:
        qnS_int = qn_int
    if qnP_int is None:
        qnP_int = th.where(qn_int >= 2, qn_int, th.zeros_like(qn_int))
    sa, pa = qnS_int[ni], qnP_int[ni]
    sb, pb = qnS_int[nj], qnP_int[nj]
    da, db = qnD_int[ni], qnD_int[nj]
    if int(max(sa.max(), sb.max(), pa.max(), pb.max(), da.max(), db.max())) > _NMAX:
        raise ValueError("overlap integrals are implemented for principal quantum numbers up to 6")

    # principal quantum number of the s, p, d shell of each atom (0 = shell absent)
    shell_a = th.stack([sa, pa, da], dim=1)
    shell_b = th.stack([sb, pb, db], dim=1)
    # shells absent from an element get a dummy exponent; their coefficients are zero
    za = th.where(shell_a > 0, zeta_a[:, :3], th.ones_like(zeta_a[:, :3]))
    zb = th.where(shell_b > 0, zeta_b[:, :3], th.ones_like(zeta_b[:, :3]))

    r = rij.unsqueeze(-1).unsqueeze(-1)
    alpha = 0.5 * r * (za.unsqueeze(2) + zb.unsqueeze(1))  # (npairs, 3, 3)
    beta = 0.5 * r * (za.unsqueeze(2) - zb.unsqueeze(1))
    aint = _a_integrals(alpha)  # (npairs, 3, 3, _NPOLY)
    bint = _b_integrals(beta)

    table = th.as_tensor(_coefficient_table(), dtype=dtype, device=device)
    sqrt_fact = th.as_tensor(_SQRT_FACT, dtype=dtype, device=device)
    half_r = 0.5 * rij

    s_loc = th.zeros((npairs, 9, 9), dtype=dtype, device=device)
    for k, (la, lb, m) in enumerate(_SPECS):
        na, nb = shell_a[:, la], shell_b[:, lb]
        coeff = table[k][na, nb]  # (npairs, _NPOLY, _NPOLY)
        radial = th.einsum("pi,pij,pj->p", aint[:, la, lb], coeff, bint[:, la, lb])
        norm_a = (2.0 * za[:, la]) ** (na.to(dtype) + 0.5) / sqrt_fact[na]
        norm_b = (2.0 * zb[:, lb]) ** (nb.to(dtype) + 0.5) / sqrt_fact[nb]
        value = norm_a * norm_b * half_r ** (na + nb + 1).to(dtype) * radial
        for ia, ib in zip(_LOCAL_INDEX[la][m], _LOCAL_INDEX[lb][m]):
            s_loc[:, ia, ib] = value

    rot = _rotation_matrices(_bond_frame(xij))
    return rot @ s_loc @ rot.transpose(1, 2)

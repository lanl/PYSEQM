import os
from functools import lru_cache

import numpy as np
import torch

from .constants import a0, ev_kcalpmol
from .dispersion_am1_fs1 import dEdisp_dr, dispersion_am1_fs1

# OM2-D / OM3-D of MNDO2020: D2 with Elstner or Yang damping, D3(BJ) without (D3) or with (D3T) the three-body term.
# Port of MMDPE.f/MMDPI.f/PARAM.f and the DFT-D3 routines of MMDPG.f. D3 tables cover H-Cl, i.e. all OMx elements.
ZMAX = 17
AUTOANG = 0.52917726  # Bohr in Angstrom, as in MNDO's DFT-D3
HARTREE_KCAL = 627.509541
JMOL_KCAL = 4184.043
_DIR = os.path.join(os.path.dirname(__file__), "../params/dispersion")

_D2 = {  # s6, d
    ("OM2", "D2-ELSTNER"): (0.8, 0.6),
    ("OM2", "D2"): (0.5, 12.0),
    ("OM3", "D2-ELSTNER"): (0.6, 1.4),
    ("OM3", "D2"): (0.6, 18.6),
}
_METHODS = {"FS1": ("AM1",), **{k: ("OM2", "OM3") for k in ("D2-ELSTNER", "D2", "D3", "D3T")}}
_D3 = {"OM2": (0.531, 0.690, 3.446), "OM3": (0.501, 0.613, 3.258)}  # s8, a1, a2 (s6 = 1)


def _csv(name):
    return np.loadtxt(os.path.join(_DIR, name), delimiter=",", skiprows=1, ndmin=2).T


@lru_cache(maxsize=None)
def _d2_tables(method, kind, device, dtype):
    # C6 (J nm^6/mol) and R0 (nm); elements without parameters keep C6 = 0
    C6, R0 = torch.zeros(ZMAX + 1), torch.ones(ZMAX + 1)
    with open(os.path.join(_DIR, "omx_d2.csv")) as f:
        for m, k, z, c6, r0 in (line.split(",") for line in list(f)[1:]):
            if m == method and k == kind:
                C6[int(z)], R0[int(z)] = float(c6), float(r0)
    return C6.to(device, dtype), R0.to(device, dtype)


@lru_cache(maxsize=None)
def _d3_tables(device, dtype):
    z, rcov_, r2r4_ = _csv("d3_elements.csv")
    z = torch.as_tensor(z).long()
    rcov, r2r4 = (torch.zeros(ZMAX + 1, dtype=torch.float64) for _ in range(2))
    rcov[z], r2r4[z] = torch.as_tensor(rcov_), torch.as_tensor(r2r4_)

    zi, zj, r = _csv("d3_r0ab.csv")
    zi, zj = torch.as_tensor(zi).long(), torch.as_tensor(zj).long()
    r0ab = torch.ones(ZMAX + 1, ZMAX + 1, dtype=torch.float64)  # 1 for padding keeps everything finite
    r0ab[zi, zj] = r0ab[zj, zi] = torch.as_tensor(r) / AUTOANG

    zi, zj, ia, ja, c6, cni, cnj = (torch.as_tensor(c) for c in _csv("d3_c6ref.csv"))
    zi, zj, ia, ja = zi.long(), zj.long(), ia.long() - 1, ja.long() - 1
    ref = torch.zeros(
        3, ZMAX + 1, ZMAX + 1, 5, 5, dtype=torch.float64
    )  # C6, CN_i, CN_j of every reference pair
    ref[:, zi, zj, ia, ja] = torch.stack((c6, cni, cnj))
    ref[:, zj, zi, ja, ia] = torch.stack((c6, cnj, cni))
    return tuple(t.to(device, dtype) for t in (rcov, r2r4, r0ab, ref))


def _d3_energy(R, Z, method, three_body):
    """D3(BJ) energy (kcal/mol) from dense distances R (nmol, N, N) in Bohr and species Z (nmol, N)."""
    s8, a1, a2 = _D3[method]
    rcov, r2r4, r0ab, ref = _d3_tables(R.device, R.dtype)
    real = Z > 0
    valid = real[:, :, None] & real[:, None, :] & ~torch.eye(R.shape[-1], dtype=torch.bool, device=R.device)
    R = torch.where(valid, R, torch.ones_like(R))
    Zi, Zj = Z[:, :, None], Z[:, None, :]

    cn = torch.where(valid, torch.sigmoid(16.0 * ((rcov[Zi] + rcov[Zj]) / R - 1.0)), 0.0).sum(-1)

    # Gaussian-weighted interpolation of C6 over the reference coordination numbers
    c6ref, cni, cnj = ref[:, Zi, Zj]
    dist = (cni - cn[:, :, None, None, None]) ** 2 + (cnj - cn[:, None, :, None, None]) ** 2
    exists = c6ref > 0
    dist = torch.where(exists, dist, 1e30)
    w = exists * torch.exp(-4.0 * (dist - dist.amin((-2, -1), keepdim=True).detach()))
    c6 = torch.where(valid, (w * c6ref).sum((-2, -1)) / w.sum((-2, -1)).clamp_min(1e-30), 1.0)

    q = 3.0 * r2r4[Zi] * r2r4[Zj]  # C8/C6
    cut = a1 * q.sqrt() + a2
    pair = c6 * (1.0 / (R**6 + cut**6) + s8 * q / (R**8 + cut**8))
    E = -0.5 * torch.where(valid, pair, 0.0).sum((-2, -1))

    if three_body:  # Axilrod-Teller-Muto term with averaged C6
        r2 = R**2
        dmp = (R / r0ab[Zi, Zj]) ** (1.0 / 3.0)
        sl = lambda m, a, b: m[(slice(None),) + tuple(slice(None) if s in (a, b) else None for s in "ijk")]
        r2ij, r2jk, r2ik = sl(r2, "i", "j"), sl(r2, "j", "k"), sl(r2, "i", "k")
        t1, t2, t3 = r2ij + r2jk - r2ik, r2ij + r2ik - r2jk, r2ik + r2jk - r2ij
        prod = r2ij * r2jk * r2ik
        ang = (0.375 * t1 * t2 * t3 / prod + 1.0) / prod**1.5
        damp = 1.0 / (
            1.0 + 6.0 * ((4.0 / 3.0) / (sl(dmp, "i", "j") * sl(dmp, "j", "k") * sl(dmp, "i", "k"))) ** 16
        )
        c9 = (sl(c6, "i", "j") * sl(c6, "j", "k") * sl(c6, "i", "k")).sqrt()
        triple = sl(valid, "i", "j") & sl(valid, "j", "k") & sl(valid, "i", "k")
        E = E + torch.where(triple, damp * c9 * ang, 0.0).sum((-3, -2, -1)) / 6.0
    return E * HARTREE_KCAL


def dispersion_kind(seqm_parameters):
    """Validated name of the requested correction ("FS1", "D2-ELSTNER", "D2", "D3", "D3T"), or None if switched off.

    seqm_parameters["dispersion"]: False | "FS1" (AM1) | "D2-Elstner", "D2", "D3", "D3T" (OM2, OM3).
    """
    kind, method = seqm_parameters.get("dispersion", False), seqm_parameters["method"]
    if not kind:
        return None
    name = str(kind).upper()
    if method not in _METHODS.get(name, ()):
        options = [n for n, methods in _METHODS.items() if method in methods]
        raise ValueError(f"dispersion={kind!r} is not available for {method}; options: {options or 'none'}")
    if name != "FS1" and seqm_parameters.get("pair_outer_cutoff", 1e10) != 1e10:
        raise ValueError(f"{name} dispersion needs the full pair list: do not set pair_outer_cutoff")
    return name


def omx_dispersion_energy(mol, rij, kind):
    """OM2/OM3 dispersion energy (eV) per molecule as a function of the pair distances rij (Bohr)."""
    if kind.startswith("D3"):
        nmol, N = mol.species.shape
        R = rij.new_zeros(nmol * N * N).index_put((mol.mask,), rij).index_put((mol.mask_l,), rij)
        E = _d3_energy(R.view(nmol, N, N) * a0 / AUTOANG, mol.species, mol.method, kind == "D3T")
    else:
        s6, d = _D2[mol.method, kind]
        C6, R0 = _d2_tables(mol.method, kind, rij.device, rij.dtype)
        if (C6[mol.Z] == 0).any():
            raise ValueError(
                f"{kind} dispersion parameters of {mol.method} are only available for H, C, N, O, F"
            )
        x = 0.1 * a0 * rij / (R0[mol.ni] + R0[mol.nj])  # r / R0 with r in nm
        damp = (1 - torch.exp(-d * x**7)) ** 4 if kind == "D2-ELSTNER" else torch.sigmoid(d * (x - 1.0))
        pair = -s6 * (C6[mol.ni] * C6[mol.nj]).sqrt() / (0.1 * a0 * rij) ** 6 * damp / JMOL_KCAL
        E = pair.new_zeros(mol.nmol).index_add_(0, mol.pair_molid, pair)
    return E / ev_kcalpmol


def dispersion_energy(mol, P):
    """Dispersion correction to the total energy (eV) per molecule; 0 if not requested."""
    kind = dispersion_kind(mol.seqm_parameters)
    if kind == "FS1":
        return dispersion_am1_fs1(mol, P)
    return omx_dispersion_energy(mol, mol.rij, kind) if kind else 0.0


def dispersion_pair_grad(mol):
    """Dispersion contribution to the pair gradient dE/dx_i (the quantity added to core_core_der); 0 if not requested."""
    kind = dispersion_kind(mol.seqm_parameters)
    if kind == "FS1":
        return dEdisp_dr(mol)
    if not kind:
        return 0.0
    with torch.enable_grad():
        rij = mol.rij.detach().requires_grad_()
        (g,) = torch.autograd.grad(omx_dispersion_energy(mol, rij, kind).sum(), rij)
    return -(g / a0).unsqueeze(1) * mol.xij  # eV/Angstrom

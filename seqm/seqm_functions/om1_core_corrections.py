import torch

from .constants import ev
from .omx_basis import gather_om1_basis

_OM1_HEAVY_SCALE_INDICES = {"HH": (0,), "XH": (0, 1, 2, 3), "XX": tuple(range(22))}


def _om1_orbital_count(z):
    return torch.where(z == 1, torch.ones_like(z), torch.full_like(z, 4))


def _boys_f012(x):
    dtype = x.dtype
    device = x.device
    eps = torch.tensor(1.0e-10, dtype=dtype, device=device)
    small = x < 1.0e-8
    safe_x = torch.where(small, eps, x)
    sqrt_x = torch.sqrt(safe_x)
    exp_mx = torch.exp(-safe_x)

    f0 = 0.5 * torch.sqrt(torch.tensor(torch.pi, dtype=dtype, device=device) / safe_x) * torch.erf(sqrt_x)
    f1 = (f0 - exp_mx) / (2.0 * safe_x)
    f2 = (3.0 * f1 - exp_mx) / (2.0 * safe_x)

    if small.any():
        xs = x[small]
        f0_small = 1.0 - xs / 3.0 + xs * xs / 10.0
        f1_small = 1.0 / 3.0 - xs / 5.0 + xs * xs / 14.0
        f2_small = 1.0 / 5.0 - xs / 7.0 + xs * xs / 18.0
        f0 = f0.clone()
        f1 = f1.clone()
        f2 = f2.clone()
        f0[small] = f0_small
        f1[small] = f1_small
        f2[small] = f2_small
    return f0, f1, f2


def om1_gscale(rij, ri, ni, nj, g_ss):
    """
    Apply the OMx GSCALE KLOPMAN-OHNO scaling to the analytic pair integrals.

    Parameters
    ----------
    rij : (npairs,) tensor
        Interatomic distance in bohr.
    ri : (npairs, 22) tensor
        Analytic pair-integral array in the REPGAU packing.
    ni, nj : (npairs,) int tensors
        Atomic numbers for the current pair.
    g_ss : tensor
        Element-indexed OM1 ``g_ss`` parameter table.

    Returns
    -------
    scaled_ri : (npairs, 22) tensor
        GSCALE-adjusted integrals.
    fko : (npairs,) tensor
        Pairwise KLOPMAN-OHNO scaling factor.
    """
    poi = 0.5 * ev / g_ss[ni]
    poj = 0.5 * ev / g_ss[nj]
    semi = ev / torch.sqrt(rij * rij + (poi + poj).pow(2))
    fko = semi / ri[:, 0]

    scaled = ri.clone()
    iorbs = _om1_orbital_count(ni)
    jorbs = _om1_orbital_count(nj)

    hh = (iorbs == 1) & (jorbs == 1)
    xh = (iorbs == 4) & (jorbs == 1)
    xx = (iorbs == 4) & (jorbs == 4)

    if hh.any():
        idx = hh.nonzero(as_tuple=False).squeeze(1)
        scaled[idx[:, None], torch.tensor(_OM1_HEAVY_SCALE_INDICES["HH"], device=ri.device)] *= fko[
            idx
        ].unsqueeze(1)
    if xh.any():
        idx = xh.nonzero(as_tuple=False).squeeze(1)
        scaled[idx[:, None], torch.tensor(_OM1_HEAVY_SCALE_INDICES["XH"], device=ri.device)] *= fko[
            idx
        ].unsqueeze(1)
    if xx.any():
        idx = xx.nonzero(as_tuple=False).squeeze(1)
        scaled[idx] *= fko[idx].unsqueeze(1)

    return scaled, fko


def om1_corgto(atomic_numbers, zeta, rij, basis):
    """
    Basic analytical Gaussian core-electron attraction integrals for one OM1 shell.

    Returns the current H / ``sp`` subset of the Fortran ``VB(4)`` array:
    ``[ss, sp_sigma, pp_sigma, pp_pi]`` in atomic units.
    """
    shell_type, exponents, coeff_s, coeff_p = gather_om1_basis(atomic_numbers, zeta, basis)
    dtype = rij.dtype
    device = rij.device

    a = exponents.unsqueeze(2)
    b = exponents.unsqueeze(1)
    csa = coeff_s.unsqueeze(2)
    csb = coeff_s.unsqueeze(1)
    cpa = coeff_p.unsqueeze(2)
    cpb = coeff_p.unsqueeze(1)

    g = a + b
    rr = rij.view(-1, 1, 1).pow(2)
    x = g * rr
    f0, f1, f2 = _boys_f012(x)

    vb = torch.zeros((atomic_numbers.shape[0], 4), dtype=dtype, device=device)

    ss_mask = shell_type == 0
    if ss_mask.any():
        v00 = (2.0 * torch.pi / g[ss_mask]) * f0[ss_mask]
        vb[ss_mask, 0] = torch.sum(csa[ss_mask] * csb[ss_mask] * v00, dim=(1, 2))

    sp_mask = shell_type == 1
    if sp_mask.any():
        g_sp = g[sp_mask]
        r_sp = rij[sp_mask].view(-1, 1, 1)
        ff1 = 2.0 * torch.pi / g_sp
        ff2 = torch.pi / (g_sp * g_sp)
        f0_sp = f0[sp_mask]
        f1_sp = f1[sp_mask]
        f2_sp = f2[sp_mask]
        v00 = f0_sp * ff1
        v03 = f1_sp * ff1 * r_sp
        v11 = (f0_sp - f1_sp) * ff2
        v33 = v11 + f2_sp * ff1 * r_sp * r_sp

        vb[sp_mask, 0] = torch.sum(csa[sp_mask] * csb[sp_mask] * v00, dim=(1, 2))
        vb[sp_mask, 1] = torch.sum(csa[sp_mask] * cpb[sp_mask] * v03, dim=(1, 2))
        vb[sp_mask, 2] = torch.sum(cpa[sp_mask] * cpb[sp_mask] * v33, dim=(1, 2))
        vb[sp_mask, 3] = torch.sum(cpa[sp_mask] * cpb[sp_mask] * v11, dim=(1, 2))

    return vb


def om1_corgau(ni, nj, rij, zeta_i, zeta_j, tore, basis):
    """
    Analytical OM1 core-electron attraction tensor from CORGAU.

    Returns the current ``4 x 2`` subset of the Fortran ``CORT(10,2)`` tensor.
    """
    dtype = rij.dtype
    device = rij.device
    cort = torch.zeros((ni.shape[0], 4, 2), dtype=dtype, device=device)

    vb_i = om1_corgto(ni, zeta_i, rij, basis)
    vb_j = om1_corgto(nj, zeta_j, rij, basis)

    # Local factors follow CORGAU conventions:
    # column 0: electrons on atom i, core of atom j
    # column 1: electrons on atom j, core of atom i
    # The p-s term in column 1 carries the Fortran sign flip.
    fj = -tore[nj] * ev
    fi = -tore[ni] * ev

    cort[:, 0, 0] = vb_i[:, 0] * fj
    cort[:, 0, 1] = vb_j[:, 0] * fi

    heavy_i = _om1_orbital_count(ni) >= 4
    heavy_j = _om1_orbital_count(nj) >= 4

    cort[heavy_i, 1, 0] = vb_i[heavy_i, 1] * fj[heavy_i]
    cort[heavy_i, 2, 0] = vb_i[heavy_i, 2] * fj[heavy_i]
    cort[heavy_i, 3, 0] = vb_i[heavy_i, 3] * fj[heavy_i]

    cort[heavy_j, 1, 1] = -vb_j[heavy_j, 1] * fi[heavy_j]
    cort[heavy_j, 2, 1] = vb_j[heavy_j, 2] * fi[heavy_j]
    cort[heavy_j, 3, 1] = vb_j[heavy_j, 3] * fi[heavy_j]

    return cort


def om1_cordef(ri, ni, nj, tore):
    """
    Map packed OM1 local two-electron integrals ``RI(22)`` onto the local
    core-electron attraction tensor used by COROM/ROTCOH.

    This mirrors the Fortran CORDEF routine for the current H / ``sp``-heavy
    scope. The returned tensor uses the leading ``4 x 2`` block of the full
    Fortran ``CORE(10,2)`` layout:

    - column 0: electron distribution on atom ``i`` attracted by nucleus ``j``
    - column 1: electron distribution on atom ``j`` attracted by nucleus ``i``
    """
    dtype = ri.dtype
    device = ri.device
    core = torch.zeros((ri.shape[0], 4, 2), dtype=dtype, device=device)

    iorbs = _om1_orbital_count(ni)
    jorbs = _om1_orbital_count(nj)
    toreni = tore[ni]
    torenj = tore[nj]

    core[:, 0, 0] = -ri[:, 0] * torenj
    core[:, 0, 1] = -ri[:, 0] * toreni

    heavy_i = iorbs >= 4
    heavy_j = jorbs >= 4

    core[heavy_i, 1, 0] = -ri[heavy_i, 1] * torenj[heavy_i]
    core[heavy_i, 2, 0] = -ri[heavy_i, 2] * torenj[heavy_i]
    core[heavy_i, 3, 0] = -ri[heavy_i, 3] * torenj[heavy_i]

    core[heavy_j, 1, 1] = -ri[heavy_j, 4] * toreni[heavy_j]
    core[heavy_j, 2, 1] = -ri[heavy_j, 10] * toreni[heavy_j]
    core[heavy_j, 3, 1] = -ri[heavy_j, 11] * toreni[heavy_j]

    return core


def om1_penetration(cort, rept, fko):
    """
    OM1 penetration contribution from COROM:

    ``PEN = FKO * (CORT - REPT)``
    """
    return fko.view(-1, 1, 1) * (cort - rept)


def om1_assemble_core(core_semi, pen, corpp=None, valpp=None):
    """
    Assemble the final OM1 local core-attraction tensor used in Hcore.

    Mirrors the additive update in COROM:

    ``CORE = CORE + PEN + CORPP + VALPP``
    """
    out = core_semi.clone()
    out = out + pen
    if corpp is not None:
        out = out + corpp
    if valpp is not None:
        out = out + valpp
    return out


def om1_valpot(ni, nj, core, s_local, t_local, u_ss, u_pp):
    """
    Evaluate the OMx VALPOT second-order pseudopotential tensors.

    This mirrors the Fortran VALPOT routine before the OM1 FVAL scaling in
    COROM. The supported shell layout is the current OM1 scope:
    H -> ``1s`` and heavy atoms -> ``2s,2p``.

    Parameters
    ----------
    ni, nj : (npairs,) int tensors
        Atomic numbers for the current pair.
    core : (npairs, 4, 2) tensor
        Semiempirical core-electron attractions before COROM corrections.
        The leading dimension corresponds to Fortran indices 1..4.
    s_local, t_local : (npairs, 5) tensor
        Local overlap and resonance terms in BETOM shell order 1..5.
    u_ss, u_pp : tensor
        Element-indexed OM1 atomic one-center parameters.

    Returns
    -------
    valpp : (npairs, 4, 2) tensor
        First-type orthogonalization terms from resonance integrals.
    valpp1 : (npairs, 4, 2) tensor
        Second-type orthogonalization terms involving local Hcore terms.
    """
    dtype = core.dtype
    device = core.device
    valpp = torch.zeros((ni.shape[0], 4, 2), dtype=dtype, device=device)
    valpp1 = torch.zeros_like(valpp)

    iorbs = _om1_orbital_count(ni)
    jorbs = _om1_orbital_count(nj)
    is_hh = (iorbs == 1) & (jorbs == 1)
    is_xh = (iorbs == 4) & (jorbs == 1)
    is_xx = (iorbs == 4) & (jorbs == 4)

    usi = u_ss[ni] + core[:, 0, 0]
    usj = u_ss[nj] + core[:, 0, 1]
    upi = u_pp[ni] + core[:, 2, 0]
    uppi = u_pp[ni] + core[:, 3, 0]
    upj = u_pp[nj] + core[:, 2, 1]
    uppj = u_pp[nj] + core[:, 3, 1]

    s1, s2, s3, s4, s5 = [s_local[:, i] for i in range(5)]
    t1, t2, t3, t4, t5 = [t_local[:, i] for i in range(5)]

    if is_hh.any():
        valpp[is_hh, 0, 0] = -s1[is_hh] * t1[is_hh]
        valpp[is_hh, 0, 1] = valpp[is_hh, 0, 0]

    if is_xh.any():
        valpp[is_xh, 0, 0] = -s1[is_xh] * t1[is_xh]
        valpp[is_xh, 0, 1] = -s1[is_xh] * t1[is_xh] - s3[is_xh] * t3[is_xh]
        valpp[is_xh, 2, 0] = -s3[is_xh] * t3[is_xh]
        valpp[is_xh, 1, 0] = -0.5 * (s1[is_xh] * t3[is_xh] + s3[is_xh] * t1[is_xh])

        valpp1[is_xh, 0, 0] = s1[is_xh].pow(2) * (usi[is_xh] - usj[is_xh])
        valpp1[is_xh, 0, 1] = s1[is_xh].pow(2) * (usj[is_xh] - usi[is_xh]) + s3[is_xh].pow(2) * (
            usj[is_xh] - upi[is_xh]
        )
        valpp1[is_xh, 2, 0] = s3[is_xh].pow(2) * (upi[is_xh] - usj[is_xh])
        valpp1[is_xh, 1, 0] = 0.5 * s1[is_xh] * s3[is_xh] * (usi[is_xh] + upi[is_xh] - 2.0 * usj[is_xh])

    if is_xx.any():
        valpp[is_xx, 0, 0] = -s1[is_xx] * t1[is_xx] - s2[is_xx] * t2[is_xx]
        valpp[is_xx, 0, 1] = -s1[is_xx] * t1[is_xx] - s3[is_xx] * t3[is_xx]
        valpp[is_xx, 2, 0] = -s3[is_xx] * t3[is_xx] - s4[is_xx] * t4[is_xx]
        valpp[is_xx, 2, 1] = -s2[is_xx] * t2[is_xx] - s4[is_xx] * t4[is_xx]
        valpp[is_xx, 3, 0] = -s5[is_xx] * t5[is_xx]
        valpp[is_xx, 3, 1] = valpp[is_xx, 3, 0]
        valpp[is_xx, 1, 0] = -0.5 * (
            s1[is_xx] * t3[is_xx] + s3[is_xx] * t1[is_xx] + s2[is_xx] * t4[is_xx] + s4[is_xx] * t2[is_xx]
        )
        valpp[is_xx, 1, 1] = -0.5 * (
            s1[is_xx] * t2[is_xx] + s2[is_xx] * t1[is_xx] + s3[is_xx] * t4[is_xx] + s4[is_xx] * t3[is_xx]
        )

        valpp1[is_xx, 0, 0] = s1[is_xx].pow(2) * (usi[is_xx] - usj[is_xx]) + s2[is_xx].pow(2) * (
            usi[is_xx] - upj[is_xx]
        )
        valpp1[is_xx, 0, 1] = s1[is_xx].pow(2) * (usj[is_xx] - usi[is_xx]) + s3[is_xx].pow(2) * (
            usj[is_xx] - upi[is_xx]
        )
        valpp1[is_xx, 2, 0] = s3[is_xx].pow(2) * (upi[is_xx] - usj[is_xx]) + s4[is_xx].pow(2) * (
            upi[is_xx] - upj[is_xx]
        )
        valpp1[is_xx, 2, 1] = s2[is_xx].pow(2) * (upj[is_xx] - usi[is_xx]) + s4[is_xx].pow(2) * (
            upj[is_xx] - upi[is_xx]
        )
        valpp1[is_xx, 3, 0] = s5[is_xx].pow(2) * (uppi[is_xx] - uppj[is_xx])
        valpp1[is_xx, 3, 1] = -valpp1[is_xx, 3, 0]
        valpp1[is_xx, 1, 0] = 0.5 * (
            s3[is_xx] * s1[is_xx] * (usi[is_xx] + upi[is_xx] - 2.0 * usj[is_xx])
            + s2[is_xx] * s4[is_xx] * (usi[is_xx] + upi[is_xx] - 2.0 * upj[is_xx])
        )
        valpp1[is_xx, 1, 1] = 0.5 * (
            s1[is_xx] * s2[is_xx] * (usj[is_xx] + upj[is_xx] - 2.0 * usi[is_xx])
            + s3[is_xx] * s4[is_xx] * (usj[is_xx] + upj[is_xx] - 2.0 * upi[is_xx])
        )

    return valpp, valpp1


def om1_apply_valpot_scaling(ni, nj, valpp, valpp1, fval1, fval2):
    """
    Apply the OM1 COROM FVAL scaling to the raw VALPOT tensors.
    """
    scaled = valpp.clone()
    iorbs = _om1_orbital_count(ni)
    jorbs = _om1_orbital_count(nj)
    fsi = fval1[ni]
    fsj = fval1[nj]
    fpi = fval2[ni]
    fpj = fval2[nj]

    scaled[:, 0, 0] = scaled[:, 0, 0] * fsi + valpp1[:, 0, 0] * fpi * 0.25
    scaled[:, 0, 1] = scaled[:, 0, 1] * fsj + valpp1[:, 0, 1] * fpj * 0.25

    heavy_i = iorbs >= 4
    heavy_j = jorbs >= 4
    both_heavy = heavy_i & heavy_j

    scaled[heavy_i, 1, 0] = scaled[heavy_i, 1, 0] * fsi[heavy_i] + valpp1[heavy_i, 1, 0] * fpi[heavy_i] * 0.25
    scaled[heavy_i, 2, 0] = scaled[heavy_i, 2, 0] * fsi[heavy_i] + valpp1[heavy_i, 2, 0] * fpi[heavy_i] * 0.25
    scaled[heavy_j, 1, 1] = scaled[heavy_j, 1, 1] * fsj[heavy_j] + valpp1[heavy_j, 1, 1] * fpj[heavy_j] * 0.25
    scaled[heavy_j, 2, 1] = scaled[heavy_j, 2, 1] * fsj[heavy_j] + valpp1[heavy_j, 2, 1] * fpj[heavy_j] * 0.25
    scaled[both_heavy, 3, 0] = (
        scaled[both_heavy, 3, 0] * fsi[both_heavy] + valpp1[both_heavy, 3, 0] * fpi[both_heavy] * 0.25
    )
    scaled[both_heavy, 3, 1] = (
        scaled[both_heavy, 3, 1] * fsj[both_heavy] + valpp1[both_heavy, 3, 1] * fpj[both_heavy] * 0.25
    )
    return scaled

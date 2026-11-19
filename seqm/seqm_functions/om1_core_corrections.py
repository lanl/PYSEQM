"""
Core-electron attraction corrections of the OMx methods (Fortran CORGTO/CORGAU/CORDEF/COROM/VALPOT).

Everything here is written densely over the pair list. H atoms carry zero p coefficients and zero
p-type overlap/resonance terms, so the p-containing expressions vanish for them on their own and no
per-orbital-class masking (and hence no host synchronisation) is needed.
"""

import torch

from .constants import ev


def _is_heavy(z):
    return z != 1


def om1_gscale(rij, ri, ni, nj, gi, gj):
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
    gi, gj : (npairs,) tensors
        ``g_ss`` of the first and second atom of each pair.

    Returns
    -------
    scaled_ri : (npairs, 22) tensor
        GSCALE-adjusted integrals. H-H pairs scale RI[0], heavy-H pairs RI[0:4], heavy-heavy all 22.
    fko : (npairs,) tensor
        Pairwise KLOPMAN-OHNO scaling factor.
    """
    poi = 0.5 * ev / gi
    poj = 0.5 * ev / gj
    semi = ev / torch.sqrt(rij * rij + (poi + poj).pow(2))
    fko = semi / ri[:, 0]

    heavy_i = _is_heavy(ni)
    heavy_j = _is_heavy(nj)
    n_scaled = torch.where(heavy_i & heavy_j, 22, torch.where(heavy_i, 4, torch.where(heavy_j, 0, 1)))
    scaled_cols = torch.arange(ri.shape[1], device=ri.device) < n_scaled.unsqueeze(1)
    return ri * torch.where(scaled_cols, fko.unsqueeze(1), torch.ones_like(ri)), fko


def _corgto_f012(x, boys_table):
    """F0, F1, F2 of the Boys function: stored table for x <= XLIM, asymptotic form beyond."""
    f0, f1, f2 = boys_table(x, 3)

    xa = x.clamp(min=boys_table.xlim)
    fa0 = 0.5 * torch.sqrt(torch.pi / xa)
    fa1 = 0.5 * fa0 / xa
    fa2 = 1.5 * fa1 / xa

    asymp = x > boys_table.xlim
    return torch.where(asymp, fa0, f0), torch.where(asymp, fa1, f1), torch.where(asymp, fa2, f2)


def om1_corgto(basis_payload, rij, basis_tables):
    """
    Basic analytical Gaussian core-electron attraction integrals for one OM1 shell.

    Returns the ``[ss, sp_sigma, pp_sigma, pp_pi]`` subset of the Fortran ``VB(4)`` array in atomic
    units, shape (npairs, 4). The p columns are exactly zero for H.
    """
    exponents = basis_payload["exponents"]
    coeff_s = basis_payload["coeff_s"]
    coeff_p = basis_payload["coeff_p"]

    g = exponents.unsqueeze(2) + exponents.unsqueeze(1)
    r = rij.view(-1, 1, 1)
    rr = r * r
    f0, f1, f2 = _corgto_f012(g * rr, basis_tables["boys_integrals"])

    ff1 = 2.0 * torch.pi / g
    ff2 = torch.pi / (g * g)
    v00 = f0 * ff1
    v03 = f1 * ff1 * r
    v11 = (f0 - f1) * ff2
    v33 = v11 + f2 * ff1 * rr

    csa, csb = coeff_s.unsqueeze(2), coeff_s.unsqueeze(1)
    cpa, cpb = coeff_p.unsqueeze(2), coeff_p.unsqueeze(1)
    return torch.stack(
        [
            torch.sum(csa * csb * v00, dim=(1, 2)),
            torch.sum(csa * cpb * v03, dim=(1, 2)),
            torch.sum(cpa * cpb * v33, dim=(1, 2)),
            torch.sum(cpa * cpb * v11, dim=(1, 2)),
        ],
        dim=1,
    )


def om1_corgau(ni, nj, rij, tore, basis_tables, basis_i, basis_j):
    """
    Analytical OM1 core-electron attraction tensor from CORGAU.

    Returns the ``4 x 2`` subset of the Fortran ``CORT(10,2)`` tensor:
    column 0: electrons on atom i, core of atom j; column 1: electrons on atom j, core of atom i.
    """
    vb_i = om1_corgto(basis_i, rij, basis_tables)
    vb_j = om1_corgto(basis_j, rij, basis_tables)

    # The p-s term of the second column (electrons on atom j) carries the Fortran sign flip.
    vb_j = torch.cat([vb_j[:, :1], -vb_j[:, 1:2], vb_j[:, 2:]], dim=1)
    col0 = vb_i * (-tore[nj] * ev).unsqueeze(1)
    col1 = vb_j * (-tore[ni] * ev).unsqueeze(1)
    return torch.stack([col0, col1], dim=2)


def om1_cordef(ri, ni, nj, tore):
    """
    Map packed OM1 local two-electron integrals ``RI(22)`` onto the local
    core-electron attraction tensor used by COROM/ROTCOH (Fortran CORDEF).

    The returned (npairs, 4, 2) tensor is the leading block of the Fortran ``CORE(10,2)`` layout:

    - column 0: electron distribution on atom ``i`` attracted by nucleus ``j``
    - column 1: electron distribution on atom ``j`` attracted by nucleus ``i``
    """
    toreni = tore[ni].unsqueeze(1)
    torenj = tore[nj].unsqueeze(1)

    heavy_i = _is_heavy(ni).unsqueeze(1)
    heavy_j = _is_heavy(nj).unsqueeze(1)
    zero = torch.zeros_like(ri[:, :3])

    # ri[:, 1:4] = (s_i p_sigma_i | ss), ... ; ri[:, [4, 10, 11]] are the same on atom j.
    col0 = -ri[:, :4] * torenj
    col0 = torch.cat([col0[:, :1], torch.where(heavy_i, col0[:, 1:], zero)], dim=1)
    col1 = -ri[:, [0, 4, 10, 11]] * toreni
    col1 = torch.cat([col1[:, :1], torch.where(heavy_j, col1[:, 1:], zero)], dim=1)
    return torch.stack([col0, col1], dim=2)


def om1_penetration(cort, rept, fko):
    """
    OM1 penetration contribution from COROM:

    ``PEN = FKO * (CORT - REPT)``
    """
    return fko.view(-1, 1, 1) * (cort - rept)


def om1_valpot(ni, nj, core, s_local, t_local, pi, pj, need_valpp1=True):
    """
    Evaluate the OMx VALPOT second-order pseudopotential tensors.

    This mirrors the Fortran VALPOT routine before the OM1 FVAL scaling in
    COROM. The supported shell layout is H -> ``1s`` and heavy atoms -> ``2s,2p``.

    A single heavy-heavy expression covers H-H and heavy-H pairs as well: the overlap and resonance
    terms that involve a p orbital of H are exactly zero, which removes the corresponding summands.
    The only exception is ``valpp1`` for H-H, which must be zeroed explicitly because its
    ``s1^2 (U_i - U_j)`` term does not involve a p overlap.

    Parameters
    ----------
    ni, nj : (npairs,) int tensors
        Atomic numbers for the current pair.
    core : (npairs, 4, 2) tensor
        Semiempirical core-electron attractions before COROM corrections.
        The leading dimension corresponds to Fortran indices 1..4.
    s_local, t_local : (npairs, 5) tensor
        Local overlap and resonance terms in BETOM shell order 1..5.
    pi, pj : dict of (npairs,) tensors
        Pair-gathered ``U_ss`` and ``U_pp`` of the first and second atom.

    Returns
    -------
    valpp : (npairs, 4, 2) tensor
        First-type orthogonalization terms from resonance integrals.
    valpp1 : (npairs, 4, 2) tensor or None
        Second-type orthogonalization terms involving local Hcore terms.
    """
    s1, s2, s3, s4, s5 = s_local.unbind(1)
    t1, t2, t3, t4, t5 = t_local.unbind(1)

    valpp = _stack_tensor_4x2(
        -s1 * t1 - s2 * t2,
        -s1 * t1 - s3 * t3,
        -0.5 * (s1 * t3 + s3 * t1 + s2 * t4 + s4 * t2),
        -0.5 * (s1 * t2 + s2 * t1 + s3 * t4 + s4 * t3),
        -s3 * t3 - s4 * t4,
        -s2 * t2 - s4 * t4,
        -s5 * t5,
        -s5 * t5,
    )
    if not need_valpp1:
        return valpp, None

    usi = pi["U_ss"] + core[:, 0, 0]
    usj = pj["U_ss"] + core[:, 0, 1]
    upi = pi["U_pp"] + core[:, 2, 0]
    uppi = pi["U_pp"] + core[:, 3, 0]
    upj = pj["U_pp"] + core[:, 2, 1]
    uppj = pj["U_pp"] + core[:, 3, 1]

    v30 = s5.pow(2) * (uppi - uppj)
    valpp1 = _stack_tensor_4x2(
        s1.pow(2) * (usi - usj) + s2.pow(2) * (usi - upj),
        s1.pow(2) * (usj - usi) + s3.pow(2) * (usj - upi),
        0.5 * (s3 * s1 * (usi + upi - 2.0 * usj) + s2 * s4 * (usi + upi - 2.0 * upj)),
        0.5 * (s1 * s2 * (usj + upj - 2.0 * usi) + s3 * s4 * (usj + upj - 2.0 * upi)),
        s3.pow(2) * (upi - usj) + s4.pow(2) * (upi - upj),
        s2.pow(2) * (upj - usi) + s4.pow(2) * (upj - upi),
        v30,
        -v30,
    )
    has_heavy = _is_heavy(ni) | _is_heavy(nj)
    return valpp, valpp1 * has_heavy.to(valpp1.dtype).view(-1, 1, 1)


def _stack_tensor_4x2(v00, v01, v10, v11, v20, v21, v30, v31):
    """Pack the eight (npairs,) entries of a Fortran (4, 2) tensor, listed row by row."""
    return torch.stack(
        [torch.stack([v00, v01], dim=1), torch.stack([v10, v11], dim=1),
         torch.stack([v20, v21], dim=1), torch.stack([v30, v31], dim=1)],
        dim=1,
    )  # fmt: skip


def om1_apply_valpot_scaling(ni, nj, valpp, valpp1, pi, pj):
    """
    Apply the OM1 COROM FVAL scaling to the raw VALPOT tensors:

    ``VALPP * FVAL1 [+ 0.25 * VALPP1 * FVAL2]``

    Row r of column 0/1 is scaled by the factor of atom i/j. Rows 1, 2 need that atom to be heavy;
    row 3 needs both atoms heavy. Entries that are not scaled are exactly zero in ``valpp``/``valpp1``.
    """
    heavy_i = _is_heavy(ni)
    heavy_j = _is_heavy(nj)
    both = heavy_i & heavy_j
    always = torch.ones_like(both)
    allowed = torch.stack(
        [torch.stack([always, always], 1), torch.stack([heavy_i, heavy_j], 1),
         torch.stack([heavy_i, heavy_j], 1), torch.stack([both, both], 1)],
        dim=1,
    )  # fmt: skip

    def factors(fa, fb):
        values = torch.stack([fa, fb], dim=1).unsqueeze(1)
        return torch.where(allowed, values, torch.zeros_like(valpp))

    scaled = valpp * factors(pi["fval1"], pj["fval1"])
    if valpp1 is None:
        return scaled
    return scaled + 0.25 * valpp1 * factors(pi["fval2"], pj["fval2"])

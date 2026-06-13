import math

import torch

_XQQ_CUTOFF = 60.0


def om1_local_overlap_terms(rij, basis_i, basis_j):
    """
    Return the local OM1 Gaussian overlap terms in the Fortran SPOVER/BETOM ordering:
    [ss, s-p_sigma, p_sigma-s, p_sigma-p_sigma, p_pi-p_pi].
    """
    dtype = rij.dtype
    device = rij.device

    shell_i, exp_i, cs_i, cp_i = (
        basis_i["shell_type"],
        basis_i["exponents"],
        basis_i["coeff_s"],
        basis_i["coeff_p"],
    )
    shell_j, exp_j, cs_j, cp_j = (
        basis_j["shell_type"],
        basis_j["exponents"],
        basis_j["coeff_s"],
        basis_j["coeff_p"],
    )

    a = exp_i.unsqueeze(2)
    b = exp_j.unsqueeze(1)
    g = a + b
    inv_g = 1.0 / g
    rab = rij.view(-1, 1, 1)
    rab_sq = rab * rab
    xqq = a * b * rab_sq * inv_g
    active = xqq <= _XQQ_CUTOFF

    pie = math.pi * inv_g
    s00 = pie * torch.sqrt(pie) * torch.exp(-xqq)
    s00 = torch.where(active, s00, torch.zeros_like(s00))

    abz = -rab
    s03 = a * abz * s00 * inv_g
    s30 = -b * abz * s00 * inv_g
    s11 = 0.5 * s00 * inv_g
    s33 = s00 * (0.5 * inv_g - a * b * abz * abz * inv_g * inv_g)

    out = torch.zeros((rij.shape[0], 5), dtype=dtype, device=device)

    ss_weight = cs_i.unsqueeze(2) * cs_j.unsqueeze(1)
    out[:, 0] = torch.sum(ss_weight * s00, dim=(1, 2))

    sp_mask = shell_j == 1
    if sp_mask.any():
        sp_weight = cs_i[sp_mask].unsqueeze(2) * cp_j[sp_mask].unsqueeze(1)
        out[sp_mask, 1] = torch.sum(sp_weight * s03[sp_mask], dim=(1, 2))

    ps_mask = shell_i == 1
    if ps_mask.any():
        ps_weight = cp_i[ps_mask].unsqueeze(2) * cs_j[ps_mask].unsqueeze(1)
        out[ps_mask, 2] = torch.sum(ps_weight * s30[ps_mask], dim=(1, 2))

    pp_mask = (shell_i == 1) & (shell_j == 1)
    if pp_mask.any():
        pp_weight = cp_i[pp_mask].unsqueeze(2) * cp_j[pp_mask].unsqueeze(1)
        out[pp_mask, 3] = torch.sum(pp_weight * s33[pp_mask], dim=(1, 2))
        out[pp_mask, 4] = torch.sum(pp_weight * s11[pp_mask], dim=(1, 2))

    return out


def diatom_overlap_matrix_OM1(xij, rij, direction, basis_i, basis_j, terms=None):
    """
    Build the 4x4 OM1 overlap block for each atom pair in Cartesian AO order [s, px, py, pz].
    """
    if terms is None:
        terms = om1_local_overlap_terms(rij, basis_i, basis_j)
    ss = terms[:, 0]
    sp = terms[:, 1]
    ps = terms[:, 2]
    pp_sigma = terms[:, 3]
    pp_pi = terms[:, 4]

    npairs = xij.shape[0]
    dtype = xij.dtype
    device = xij.device
    di = torch.zeros((npairs, 4, 4), dtype=dtype, device=device)
    di[:, 0, 0] = ss

    di[:, 0, 1:] = sp.unsqueeze(1) * direction
    di[:, 1:, 0] = ps.unsqueeze(1) * direction

    eye3 = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    outer = direction.unsqueeze(2) * direction.unsqueeze(1)
    di[:, 1:, 1:] = pp_pi.view(-1, 1, 1) * eye3 + (pp_sigma - pp_pi).view(-1, 1, 1) * outer
    return di


def om1_local_resonance_terms(ni, nj, rij, parameters):
    """
    Return the local OM1 resonance terms in the Fortran BETOM ordering:
    [ss, s-p_sigma, p_sigma-s, p_sigma-p_sigma, p_pi-p_pi].
    """
    dtype = rij.dtype
    device = rij.device
    n = rij.shape[0]

    beta_s = parameters["beta_s"]
    beta_p = parameters["beta_p"]
    beta_pi = parameters["beta_pi"]
    beta_sh = parameters["beta_sh"]
    beta_ph = parameters["beta_ph"]
    alpha_s = parameters["alpha_s"]
    alpha_p = parameters["alpha_p"]
    alpha_pi = parameters["alpha_pi"]
    alpha_s_h = parameters["alpha_s_h"]
    alpha_p_h = parameters["alpha_p_h"]

    heavy_i = ni > 2
    heavy_j = nj > 2

    bas = beta_s[ni].clone()
    bbs = beta_s[nj].clone()
    aas = alpha_s[ni].clone()
    cbs = alpha_s[nj].clone()

    bap = beta_p[ni]
    bai = beta_pi[ni]
    aap = alpha_p[ni]
    aai = alpha_pi[ni]
    bbp = beta_p[nj]
    bbi = beta_pi[nj]
    abp = alpha_p[nj]
    abi = alpha_pi[nj]

    hx_i = heavy_i & (nj <= 2) & (ni < 86)
    if hx_i.any():
        bas[hx_i] = beta_sh[ni[hx_i]]
        bap = bap.clone()
        aap = aap.clone()
        aas[hx_i] = alpha_s_h[ni[hx_i]]
        bap[hx_i] = beta_ph[ni[hx_i]]
        aap[hx_i] = alpha_p_h[ni[hx_i]]

    sqrt_r = torch.sqrt(rij)
    r2 = rij * rij

    out = torch.zeros((n, 5), dtype=dtype, device=device)

    exp1 = (aas + cbs) * r2
    ok1 = exp1 < 50.0
    out[ok1, 0] = 0.5 * (bas[ok1] + bbs[ok1]) * sqrt_r[ok1] * torch.exp(-exp1[ok1])

    mask_jp = heavy_j
    if mask_jp.any():
        exp2 = (aas[mask_jp] + abp[mask_jp]) * r2[mask_jp]
        ok2 = exp2 < 50.0
        vals = torch.zeros_like(exp2)
        vals[ok2] = -sqrt_r[mask_jp][ok2] * torch.exp(-exp2[ok2])
        out[mask_jp, 1] = 0.5 * (bas[mask_jp] + bbp[mask_jp]) * vals

    mask_ip = heavy_i
    if mask_ip.any():
        exp3 = (aap[mask_ip] + cbs[mask_ip]) * r2[mask_ip]
        ok3 = exp3 < 50.0
        vals = torch.zeros_like(exp3)
        vals[ok3] = sqrt_r[mask_ip][ok3] * torch.exp(-exp3[ok3])
        out[mask_ip, 2] = 0.5 * (bap[mask_ip] + bbs[mask_ip]) * vals

    mask_pp = heavy_i & heavy_j
    if mask_pp.any():
        exp4 = (aap[mask_pp] + abp[mask_pp]) * r2[mask_pp]
        ok4 = exp4 < 50.0
        vals4 = torch.zeros_like(exp4)
        vals4[ok4] = -sqrt_r[mask_pp][ok4] * torch.exp(-exp4[ok4])
        out[mask_pp, 3] = 0.5 * (bap[mask_pp] + bbp[mask_pp]) * vals4

        exp5 = (aai[mask_pp] + abi[mask_pp]) * r2[mask_pp]
        ok5 = exp5 < 50.0
        vals5 = torch.zeros_like(exp5)
        vals5[ok5] = sqrt_r[mask_pp][ok5] * torch.exp(-exp5[ok5])
        out[mask_pp, 4] = 0.5 * (bai[mask_pp] + bbi[mask_pp]) * vals5

    return out


def diatom_resonance_matrix_OM1(xij, terms, direction):
    """
    Build the 4x4 OM1 resonance block for each atom pair in Cartesian AO order [s, px, py, pz].
    """
    # terms = om1_local_resonance_terms(ni, nj, rij, parameters)
    ss = terms[:, 0]
    sp = terms[:, 1]
    ps = terms[:, 2]
    pp_sigma = terms[:, 3]
    pp_pi = terms[:, 4]

    npairs = xij.shape[0]
    dtype = xij.dtype
    device = xij.device
    di = torch.zeros((npairs, 4, 4), dtype=dtype, device=device)
    di[:, 0, 0] = ss

    di[:, 0, 1:] = sp.unsqueeze(1) * direction
    di[:, 1:, 0] = ps.unsqueeze(1) * direction

    eye3 = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    outer = direction.unsqueeze(2) * direction.unsqueeze(1)
    di[:, 1:, 1:] = pp_pi.view(-1, 1, 1) * eye3 + (pp_sigma - pp_pi).view(-1, 1, 1) * outer
    return di

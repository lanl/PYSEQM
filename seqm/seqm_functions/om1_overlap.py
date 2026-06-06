import math

import torch

from .two_elec_two_center_int import rotate_with_quaternion

_FAC_S = (2.0 / math.pi) ** 0.75
_FAC_P = 2.0 * _FAC_S
_XQQ_CUTOFF = 60.0

# GTOMIN(IGTO=0) uses STO-3G for hydrogen and ECPSET 3G for second-row sp atoms.
# These are the unscaled primitive exponents and contraction coefficients before the
# OM1 zeta scaling EXX <- EXX * Z**2.
_OM1_BASIS_RAW = {
    1: {
        "shell_type": 0,
        "exponents": (2.227660584, 0.4057711562, 0.1098175104),
        "cs": (0.1543289673, 0.5353281423, 0.4446345422),
        "cp": (0.0, 0.0, 0.0),
    },
    6: {
        "shell_type": 1,
        "exponents": (2.64486, 0.54215, 0.14466),
        "cs": (-0.19188, 0.61628, 0.54896),
        "cp": (0.20259, 0.55830, 0.45514),
    },
    7: {
        "shell_type": 1,
        "exponents": (3.68849, 0.77534, 0.20498),
        "cs": (-0.19269, 0.61888, 0.54926),
        "cp": (0.22281, 0.56032, 0.43859),
    },
    8: {
        "shell_type": 1,
        "exponents": (4.78499, 0.99860, 0.25687),
        "cs": (-0.19248, 0.66952, 0.50270),
        "cp": (0.24158, 0.55890, 0.43160),
    },
    9: {
        "shell_type": 1,
        "exponents": (6.01783, 1.25315, 0.31760),
        "cs": (-0.18850, 0.69800, 0.47427),
        "cp": (0.25667, 0.56013, 0.42139),
    },
}


def _lookup_basis(atomic_numbers, zeta):
    device = atomic_numbers.device
    dtype = zeta.dtype
    n = atomic_numbers.shape[0]

    shell_type = torch.empty(n, dtype=torch.int64, device=device)
    exponents = torch.empty((n, 3), dtype=dtype, device=device)
    coeff_s = torch.empty((n, 3), dtype=dtype, device=device)
    coeff_p = torch.empty((n, 3), dtype=dtype, device=device)

    unsupported = []
    z_list = atomic_numbers.detach().cpu().tolist()
    for atomic_number in sorted(set(z_list)):
        basis = _OM1_BASIS_RAW.get(int(atomic_number))
        if basis is None:
            unsupported.append(int(atomic_number))
            continue
        mask = atomic_numbers == atomic_number
        shell_type[mask] = basis["shell_type"]
        exponents[mask] = torch.tensor(basis["exponents"], dtype=dtype, device=device)
        coeff_s[mask] = torch.tensor(basis["cs"], dtype=dtype, device=device)
        coeff_p[mask] = torch.tensor(basis["cp"], dtype=dtype, device=device)

    if unsupported:
        raise ValueError(f"OM1 overlap only supports H/C/N/O/F; got atomic numbers {unsupported}")

    scaled_exponents = exponents * zeta.unsqueeze(1) ** 2
    norm_s = _FAC_S * scaled_exponents.pow(0.75)
    norm_p = _FAC_P * scaled_exponents.pow(1.25)
    coeff_s = coeff_s * norm_s
    coeff_p = coeff_p * norm_p
    return shell_type, scaled_exponents, coeff_s, coeff_p


def om1_local_overlap_terms(ni, nj, rij, zeta_i, zeta_j):
    """
    Return the local OM1 Gaussian overlap terms in the Fortran SPOVER/BETOM ordering:
    [ss, s-p_sigma, p_sigma-s, p_sigma-p_sigma, p_pi-p_pi].
    """
    dtype = rij.dtype
    device = rij.device

    shell_i, exp_i, cs_i, cp_i = _lookup_basis(ni, zeta_i)
    shell_j, exp_j, cs_j, cp_j = _lookup_basis(nj, zeta_j)

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


def omx_local_overlap_terms(ni, nj, rij, zeta_i, zeta_j):
    return om1_local_overlap_terms(ni, nj, rij, zeta_i, zeta_j)


def diatom_overlap_matrix_OM1(ni, nj, xij, rij, zeta_a, zeta_b):
    """
    Build the 4x4 OM1 overlap block for each atom pair in Cartesian AO order [s, px, py, pz].
    """
    terms = om1_local_overlap_terms(ni, nj, rij, zeta_a[:, 0], zeta_b[:, 0])
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

    direction = rotate_with_quaternion(xij).transpose(1, 2)[:, :, 0]
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

    hx_j = (ni <= 2) & heavy_j & (nj < 86)
    if hx_j.any():
        bbs[hx_j] = beta_sh[nj[hx_j]]
        bbp = bbp.clone()
        abp = abp.clone()
        cbs[hx_j] = alpha_s_h[nj[hx_j]]
        bbp[hx_j] = beta_ph[nj[hx_j]]
        abp[hx_j] = alpha_p_h[nj[hx_j]]

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


def omx_local_resonance_terms(ni, nj, rij, parameters):
    return om1_local_resonance_terms(ni, nj, rij, parameters)


def omx_betom_terms(ni, nj, rij, parameters, zeta_i, zeta_j):
    """
    Return the local BETOM overlap and resonance terms in the Fortran shell order.
    """
    s_local = omx_local_overlap_terms(ni, nj, rij, zeta_i, zeta_j)
    t_local = omx_local_resonance_terms(ni, nj, rij, parameters)
    return s_local, t_local


def diatom_resonance_matrix_OM1(ni, nj, xij, rij, parameters):
    """
    Build the 4x4 OM1 resonance block for each atom pair in Cartesian AO order [s, px, py, pz].
    """
    terms = om1_local_resonance_terms(ni, nj, rij, parameters)
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

    direction = rotate_with_quaternion(xij).transpose(1, 2)[:, :, 0]
    di[:, 0, 1:] = sp.unsqueeze(1) * direction
    di[:, 1:, 0] = ps.unsqueeze(1) * direction

    eye3 = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    outer = direction.unsqueeze(2) * direction.unsqueeze(1)
    di[:, 1:, 1:] = pp_pi.view(-1, 1, 1) * eye3 + (pp_sigma - pp_pi).view(-1, 1, 1) * outer
    return di

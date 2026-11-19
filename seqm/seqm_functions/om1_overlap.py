import math

import torch

_XQQ_CUTOFF = 60.0


def om1_local_overlap_terms(rij, basis_i, basis_j):
    """
    Return the local OM1 Gaussian overlap terms in the Fortran SPOVER/BETOM ordering:
    [ss, s-p_sigma, p_sigma-s, p_sigma-p_sigma, p_pi-p_pi].

    H atoms carry zero p coefficients, so the p-containing terms vanish for them without masking.
    """
    exp_i, cs_i, cp_i = basis_i["exponents"], basis_i["coeff_s"], basis_i["coeff_p"]
    exp_j, cs_j, cp_j = basis_j["exponents"], basis_j["coeff_s"], basis_j["coeff_p"]

    a = exp_i.unsqueeze(2)
    b = exp_j.unsqueeze(1)
    inv_g = 1.0 / (a + b)
    rab = rij.view(-1, 1, 1)
    xqq = a * b * rab * rab * inv_g

    pie = math.pi * inv_g
    s00 = torch.where(xqq <= _XQQ_CUTOFF, pie * torch.sqrt(pie) * torch.exp(-xqq), torch.zeros_like(xqq))

    abz = -rab
    s03 = a * abz * s00 * inv_g
    s30 = -b * abz * s00 * inv_g
    s11 = 0.5 * s00 * inv_g
    s33 = s00 * (0.5 * inv_g - a * b * abz * abz * inv_g * inv_g)

    def contract(ci, cj, s):
        return torch.sum(ci.unsqueeze(2) * cj.unsqueeze(1) * s, dim=(1, 2))

    return torch.stack(
        [
            contract(cs_i, cs_j, s00),
            contract(cs_i, cp_j, s03),
            contract(cp_i, cs_j, s30),
            contract(cp_i, cp_j, s33),
            contract(cp_i, cp_j, s11),
        ],
        dim=1,
    )


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


_EXP_CUTOFF = 50.0


def _resonance_term(avg_beta, alpha_sum, r2, sqrt_r, sign):
    """sign * avg_beta * sqrt(r) * exp(-alpha_sum * r^2), zero beyond the exponent cutoff."""
    arg = alpha_sum * r2
    return torch.where(arg < _EXP_CUTOFF, sign * avg_beta * sqrt_r * torch.exp(-arg), torch.zeros_like(arg))


def om1_local_resonance_terms(ni, nj, rij, pi, pj):
    """
    Return the local OM1 resonance terms in the Fortran BETOM ordering:
    [ss, s-p_sigma, p_sigma-s, p_sigma-p_sigma, p_pi-p_pi].
    """
    heavy_i = ni > 2
    heavy_j = nj > 2

    # A heavy atom bonded to H uses its separate "H" resonance parameters.
    hx_i = heavy_i & (nj <= 2) & (ni < 86)
    bas = torch.where(hx_i, pi["beta_sh"], pi["beta_s"])
    aas = torch.where(hx_i, pi["alpha_s_h"], pi["alpha_s"])
    bap = torch.where(hx_i, pi["beta_ph"], pi["beta_p"])
    aap = torch.where(hx_i, pi["alpha_p_h"], pi["alpha_p"])

    r2 = rij * rij
    sqrt_r = torch.sqrt(rij)
    zero = torch.zeros_like(rij)

    ss = _resonance_term(0.5 * (bas + pj["beta_s"]), aas + pj["alpha_s"], r2, sqrt_r, 1.0)
    sp = _resonance_term(0.5 * (bas + pj["beta_p"]), aas + pj["alpha_p"], r2, sqrt_r, -1.0)
    ps = _resonance_term(0.5 * (bap + pj["beta_s"]), aap + pj["alpha_s"], r2, sqrt_r, 1.0)
    pp_sigma = _resonance_term(0.5 * (bap + pj["beta_p"]), aap + pj["alpha_p"], r2, sqrt_r, -1.0)
    pp_pi = _resonance_term(
        0.5 * (pi["beta_pi"] + pj["beta_pi"]), pi["alpha_pi"] + pj["alpha_pi"], r2, sqrt_r, 1.0
    )

    return torch.stack(
        [
            ss,
            torch.where(heavy_j, sp, zero),
            torch.where(heavy_i, ps, zero),
            torch.where(heavy_i & heavy_j, pp_sigma, zero),
            torch.where(heavy_i & heavy_j, pp_pi, zero),
        ],
        dim=1,
    )


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

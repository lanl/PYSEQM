import math

import torch

from .om1_core_corrections import (
    om1_apply_valpot_scaling,
    om1_assemble_core,
    om1_cordef,
    om1_corgau,
    om1_gscale,
    om1_penetration,
    om1_valpot,
)
from .om1_ppecp import om1_ppecp_local
from .omx_basis import gather_om1_basis

_XX3498 = 34.9868366552497


def _om1_pair_basis_data_batch(atomic_numbers, zeta, basis):
    if atomic_numbers.numel() == 0:
        device = zeta.device
        dtype = zeta.dtype
        empty_f = torch.zeros((0, 9), dtype=dtype, device=device)
        empty_i = torch.zeros((0,), dtype=torch.int64, device=device)
        return {
            "shell_type": empty_i,
            "gab": empty_f,
            "eab": empty_f,
            "dp00": empty_f,
            "dp10": empty_f,
            "dp11": empty_f,
        }

    shell_type, exponents, coeff_s, coeff_p = gather_om1_basis(atomic_numbers, zeta[atomic_numbers], basis)
    ai = exponents.unsqueeze(2)
    aj = exponents.unsqueeze(1)
    gab = (ai + aj).reshape(exponents.shape[0], -1)
    eab = 1.0 / gab
    pref = _XX3498 * eab
    csij = (coeff_s.unsqueeze(2) * coeff_s.unsqueeze(1)).reshape(exponents.shape[0], -1)
    cpij = (coeff_p.unsqueeze(2) * coeff_p.unsqueeze(1)).reshape(exponents.shape[0], -1)
    cpis = (coeff_p.unsqueeze(2) * coeff_s.unsqueeze(1)).reshape(exponents.shape[0], -1)
    shell_heavy = shell_type == 1
    dp00 = pref * csij
    dp10 = torch.zeros_like(dp00)
    dp11 = torch.zeros_like(dp00)
    if shell_heavy.any():
        idx = shell_heavy.nonzero(as_tuple=False).squeeze(1)
        dp11[idx] = pref[idx] * cpij[idx]
        dp00[idx] = (pref[idx] * csij[idx]) / dp11[idx]
        dp10[idx] = (pref[idx] * cpis[idx]) / dp11[idx]
    return {"shell_type": shell_type, "gab": gab, "eab": eab, "dp00": dp00, "dp10": dp10, "dp11": dp11}


def _om1_pair_q_data_batch(atomic_numbers, zeta, basis):
    if atomic_numbers.numel() == 0:
        device = zeta.device
        dtype = zeta.dtype
        empty_f = torch.zeros((0, 9), dtype=dtype, device=device)
        empty_i = torch.zeros((0,), dtype=torch.int64, device=device)
        return {
            "shell_type": empty_i,
            "gcd": empty_f,
            "ecd": empty_f,
            "dq00": empty_f,
            "dq10": empty_f,
            "dq11": empty_f,
        }

    shell_type, exponents, coeff_s, coeff_p = gather_om1_basis(atomic_numbers, zeta[atomic_numbers], basis)
    ai = exponents.unsqueeze(2)
    aj = exponents.unsqueeze(1)
    gcd = (ai + aj).reshape(exponents.shape[0], -1)
    ecd = 1.0 / gcd
    dq00 = (coeff_s.unsqueeze(2) * coeff_s.unsqueeze(1)).reshape(exponents.shape[0], -1)
    dq10 = (coeff_p.unsqueeze(2) * coeff_s.unsqueeze(1)).reshape(exponents.shape[0], -1)
    dq11 = (coeff_p.unsqueeze(2) * coeff_p.unsqueeze(1)).reshape(exponents.shape[0], -1)
    return {"shell_type": shell_type, "gcd": gcd, "ecd": ecd, "dq00": dq00, "dq10": dq10, "dq11": dq11}


def _boys_series_all_0_to_4(x, n_terms=10):
    """
    Taylor series:
        F_m(x) = sum_k (-x)^k / (k! * (2m + 2k + 1))

    Good near x = 0.
    """

    f = [torch.zeros_like(x) for _ in range(5)]

    xpow = torch.ones_like(x)
    fact = torch.ones_like(x)

    for k in range(n_terms):
        if k > 0:
            xpow = xpow * (-x)
            fact = fact * k

        coeff = xpow / fact
        f[0] = f[0] + coeff / (2 * 0 + 2 * k + 1)
        f[1] = f[1] + coeff / (2 * 1 + 2 * k + 1)
        f[2] = f[2] + coeff / (2 * 2 + 2 * k + 1)
        f[3] = f[3] + coeff / (2 * 3 + 2 * k + 1)
        f[4] = f[4] + coeff / (2 * 4 + 2 * k + 1)

    return tuple(f)


def _boys_erf_upward_0_to_4(x):
    """
    Fast middle-region evaluation using erf + upward recurrence.

    This is accurate enough away from x = 0.
    """
    dtype = x.dtype
    device = x.device

    pi = torch.tensor(math.pi, dtype=dtype, device=device)

    sqrtx = torch.sqrt(x)
    expx = torch.exp(-x)

    f0 = 0.5 * torch.sqrt(pi / x) * torch.erf(sqrtx)
    f1 = (f0 - expx) / (2.0 * x)
    f2 = (3.0 * f1 - expx) / (2.0 * x)
    f3 = (5.0 * f2 - expx) / (2.0 * x)
    f4 = (7.0 * f3 - expx) / (2.0 * x)

    return f0, f1, f2, f3, f4


def _boys_asymp_downward_0_to_4(x):
    """
    Large-x evaluation.

    Compute F4 using leading asymptotic formula, then recurse downward.

        F_m(x) ~ 0.5 * Gamma(m + 1/2) / x^(m + 1/2)

    For F4:
        Gamma(4.5) = 105/16 * sqrt(pi)
    """
    dtype = x.dtype
    device = x.device

    sqrt_pi = torch.tensor(math.sqrt(math.pi), dtype=dtype, device=device)
    expx = torch.exp(-x)

    gamma_4p5 = (105.0 / 16.0) * sqrt_pi

    f4 = 0.5 * gamma_4p5 / x.pow(4.5)

    # Stable downward recurrence:
    # F_m = (2x F_{m+1} + exp(-x)) / (2m + 1)
    f3 = (2.0 * x * f4 + expx) / 7.0
    f2 = (2.0 * x * f3 + expx) / 5.0
    f1 = (2.0 * x * f2 + expx) / 3.0
    f0 = 2.0 * x * f1 + expx

    return f0, f1, f2, f3, f4


def boys_0_to_4(x, *, small_cutoff=1.0e-6, large_cutoff=40.0):
    """
    Robust PyTorch Boys F_0 ... F_4.

    Parameters
    ----------
    x : torch.Tensor
        Nonnegative Boys argument.
    small_cutoff : float or None
        If None, chosen based on dtype.
    large_cutoff : float
        Above this, use asymptotic + downward recurrence.

    Returns
    -------
    f0, f1, f2, f3, f4 : torch.Tensor
    """
    dtype = x.dtype
    default_small = 1.0e-6

    if dtype in (torch.float32, torch.bfloat16):
        # float32 loses cancellation accuracy much earlier
        default_small = 1.0e-3

    if small_cutoff is None:
        small_cutoff = default_small

    # Boys arguments should be >= 0. Clamp tiny negative roundoff.
    x = torch.clamp(x, min=0.0)

    small = x <= small_cutoff
    large = x >= large_cutoff
    middle = ~(small | large)

    f0 = torch.empty_like(x)
    f1 = torch.empty_like(x)
    f2 = torch.empty_like(x)
    f3 = torch.empty_like(x)
    f4 = torch.empty_like(x)

    if small.any():
        fs = _boys_series_all_0_to_4(x[small], n_terms=10)
        f0[small], f1[small], f2[small], f3[small], f4[small] = fs

    if middle.any():
        fm = _boys_erf_upward_0_to_4(x[middle])
        f0[middle], f1[middle], f2[middle], f3[middle], f4[middle] = fm

    if large.any():
        fl = _boys_asymp_downward_0_to_4(x[large])
        f0[large], f1[large], f2[large], f3[large], f4[large] = fl

    return f0, f1, f2, f3, f4


def _sp0000_batch(ni, nj, rab, zeta, basis):
    p = _om1_pair_basis_data_batch(ni, zeta, basis)
    q = _om1_pair_q_data_batch(nj, zeta, basis)
    rab2 = rab.view(-1, 1, 1) * rab.view(-1, 1, 1)
    x = rab2 / (p["eab"].view(-1, 1, 9) + q["ecd"].view(-1, 9, 1))
    sq2 = 1.0 / torch.sqrt(p["gab"].view(-1, 1, 9) + q["gcd"].view(-1, 9, 1))
    f0, _, _, _, _ = boys_0_to_4(x)
    h0000 = torch.sum(f0 * (p["dp00"].view(-1, 1, 9) * sq2), dim=2)
    g0000 = torch.sum(h0000 * (q["dq00"].view(-1, 9) * q["ecd"].view(-1, 9)) * 27.21, dim=1)
    out = torch.zeros((rab.shape[0], 22), dtype=rab.dtype, device=rab.device)
    out[:, 0] = g0000
    return out


def _sp0011_batch(ni, nj, rab, zeta, basis):
    p = _om1_pair_basis_data_batch(ni, zeta, basis)
    q = _om1_pair_q_data_batch(nj, zeta, basis)
    rab2 = rab.view(-1, 1, 1) * rab.view(-1, 1, 1)
    eab = p["eab"].view(-1, 1, 9)
    ecd = q["ecd"].view(-1, 9, 1)
    gcd = q["gcd"].view(-1, 9, 1)
    x = rab2 / (eab + ecd)
    gfac = (1.0 / (eab + ecd)) * ecd
    sq2 = 1.0 / torch.sqrt(p["gab"].view(-1, 1, 9) + gcd)
    y = p["dp00"].view(-1, 1, 9) * sq2
    gy = gfac * y
    ggy = gfac * gy
    f0, f1, f2, _, _ = boys_0_to_4(x)
    h0000 = torch.sum(f0 * y, dim=2)
    h0001 = torch.sum(f1 * gy, dim=2)
    h0033 = torch.sum(f2 * ggy, dim=2)
    h0030 = -h0001 * rab.view(-1, 1)
    h0033 = h0033 * rab.view(-1, 1).pow(2)
    h0022 = 0.5 * q["ecd"].view(-1, 9) * (h0000 - h0001)
    h0033 = h0033 + h0022

    dq_scale = q["ecd"] * 27.21
    g0000 = torch.sum(h0000 * (q["dq00"] * dq_scale), dim=1)
    g0022 = torch.sum(h0022 * (q["dq11"] * dq_scale), dim=1)
    g0033 = torch.sum(h0033 * (q["dq11"] * dq_scale), dim=1)
    g0030 = torch.sum(h0030 * (q["dq10"] * dq_scale), dim=1)

    out = torch.zeros((rab.shape[0], 22), dtype=rab.dtype, device=rab.device)
    out[:, 0] = g0000
    out[:, 4] = g0030
    out[:, 10] = g0033
    out[:, 11] = g0022
    return out


def _sp1111_batch(ni, nj, rab, zeta, basis):
    p = _om1_pair_basis_data_batch(ni, zeta, basis)
    q = _om1_pair_q_data_batch(nj, zeta, basis)
    rab1 = rab.view(-1, 1)
    rab2 = rab1 * rab1
    rab3 = rab2 * rab1
    rab4 = rab2 * rab2
    eab = p["eab"].view(-1, 1, 9)
    eab2 = eab * eab
    ecd = q["ecd"].view(-1, 9, 1)
    x = rab2.view(-1, 1, 1) / (eab + ecd)
    gfac = 1.0 / (eab + ecd)
    sq2 = 1.0 / torch.sqrt(p["gab"].view(-1, 1, 9) + q["gcd"].view(-1, 9, 1))
    y = p["dp11"].view(-1, 1, 9) * sq2
    gy = gfac * y
    ggy = gfac * gy
    gggy = gfac * ggy
    f0, f1, f2, f3, f4 = boys_0_to_4(x)
    ff0 = f0 * y
    ff1 = f1 * gy
    ff2 = f2 * ggy
    ff3 = f3 * gggy
    ff4 = f4 * gggy * gfac

    dp00 = p["dp00"].view(-1, 1, 9)
    edp10 = eab * p["dp10"].view(-1, 1, 9)

    x0 = torch.sum(ff0 * dp00, dim=2)
    x1 = torch.sum(ff1 * dp00, dim=2)
    x2 = torch.sum(ff2 * dp00, dim=2)
    w1 = torch.sum(ff1 * edp10, dim=2)
    w2 = torch.sum(ff2 * edp10, dim=2)
    w3 = torch.sum(ff3 * edp10, dim=2)
    s0 = torch.sum(ff0 * eab, dim=2)
    s1 = torch.sum(ff1 * eab, dim=2)
    s2 = torch.sum(ff2 * eab, dim=2)
    t1 = torch.sum(ff1 * eab2, dim=2)
    t2 = torch.sum(ff2 * eab2, dim=2)
    t3 = torch.sum(ff3 * eab2, dim=2)
    t4 = torch.sum(ff4 * eab2, dim=2)

    hecd = 0.5 * q["ecd"].view(-1, 9)
    ecd2 = q["ecd"].view(-1, 9) * q["ecd"].view(-1, 9)
    hecd2 = 0.5 * ecd2
    h0000 = x0
    h0030 = -q["ecd"].view(-1, 9) * x1 * rab1
    h0022 = hecd * (x0 - q["ecd"].view(-1, 9) * x1)
    h0033 = h0022 + ecd2 * x2 * rab2
    h2020 = hecd * w1
    h2023 = -hecd2 * w2 * rab1
    h3000 = w1 * rab1
    h3030 = h2020 - q["ecd"].view(-1, 9) * w2 * rab2
    h3022 = h2023 + hecd * h3000
    h3033 = h3022 + h2023 + h2023 + ecd2 * w3 * rab3
    h2200 = 0.5 * (s0 - t1)
    h3300 = h2200 + t2 * rab2
    h2320 = hecd * t2 * rab1
    h2230 = hecd * (t2 - s1) * rab1
    h3330 = h2230 + q["ecd"].view(-1, 9) * (t2 * rab1 - t3 * rab3)
    h1212 = 0.25 * ecd2 * t2
    h2323 = hecd2 * (0.5 * t2 - t3 * rab2)
    hxxyy = 0.25 * (q["ecd"].view(-1, 9) * (s0 - t1) - ecd2 * (s1 - t2))
    h2222 = hxxyy + hecd2 * t2
    h1122 = hxxyy
    h3322 = hxxyy + hecd * t2 * rab2 - hecd2 * t3 * rab2
    h2233 = hxxyy + hecd2 * (s2 - t3) * rab2
    h3333 = hxxyy + hecd2 * (t2 + s2 * rab2) + ecd2 * (-3.0 * t3 * rab2 + t4 * rab4) + hecd * t2 * rab2

    dq_scale = q["ecd"] * 27.21
    dq00 = q["dq00"] * dq_scale
    dq10 = q["dq10"] * dq_scale
    dq11 = q["dq11"] * dq_scale

    out = torch.zeros((rab.shape[0], 22), dtype=rab.dtype, device=rab.device)
    out[:, 0] = torch.sum(h0000 * dq00, dim=1)
    out[:, 1] = torch.sum(h3000 * dq00, dim=1)
    out[:, 2] = torch.sum(h3300 * dq00, dim=1)
    out[:, 3] = torch.sum(h2200 * dq00, dim=1)
    out[:, 4] = torch.sum(h0030 * dq10, dim=1)
    out[:, 5] = torch.sum(h3030 * dq10, dim=1)
    out[:, 6] = torch.sum(h2020 * dq10, dim=1)
    out[:, 7] = torch.sum(h3330 * dq10, dim=1)
    out[:, 8] = torch.sum(h2230 * dq10, dim=1)
    out[:, 9] = torch.sum(h2320 * dq10, dim=1)
    out[:, 10] = torch.sum(h0033 * dq11, dim=1)
    out[:, 11] = torch.sum(h0022 * dq11, dim=1)
    out[:, 12] = torch.sum(h3033 * dq11, dim=1)
    out[:, 13] = torch.sum(h3022 * dq11, dim=1)
    out[:, 14] = torch.sum(h2023 * dq11, dim=1)
    out[:, 15] = torch.sum(h3333 * dq11, dim=1)
    out[:, 16] = torch.sum(h2233 * dq11, dim=1)
    out[:, 17] = torch.sum(h3322 * dq11, dim=1)
    out[:, 18] = torch.sum(h2222 * dq11, dim=1)
    out[:, 19] = torch.sum(h2323 * dq11, dim=1)
    out[:, 20] = torch.sum(h1122 * dq11, dim=1)
    out[:, 21] = torch.sum(h1212 * dq11, dim=1)
    return out


def om1_repgau_local_batch(ni, nj, rij, zeta, basis):
    """
    Batched OM1 REPGAU local ``RI(22)`` tensor for all pair shell classes.
    """
    device = rij.device
    dtype = rij.dtype

    shell_type = basis["shell_type"]
    shell_i = shell_type[ni]
    shell_j = shell_type[nj]

    hh = (shell_i == 0) & (shell_j == 0)
    xh = (shell_i == 1) & (shell_j == 0)
    xx = (shell_i == 1) & (shell_j == 1)

    out = torch.zeros((ni.shape[0], 22), dtype=dtype, device=device)
    if hh.any():
        idx = hh.nonzero(as_tuple=False).squeeze(1)
        out[idx] = _sp0000_batch(ni[idx], nj[idx], rij[idx], zeta, basis)
    if xh.any():
        idx = xh.nonzero(as_tuple=False).squeeze(1)
        tmp = _sp0011_batch(nj[idx], ni[idx], rij[idx], zeta, basis)
        out[idx, 0] = tmp[:, 0]
        out[idx, 1] = -tmp[:, 4]
        out[idx, 2] = tmp[:, 10]
        out[idx, 3] = tmp[:, 11]
    if xx.any():
        idx = xx.nonzero(as_tuple=False).squeeze(1)
        out[idx] = _sp1111_batch(ni[idx], nj[idx], rij[idx], zeta, basis)
    return out


def om1_local_pair_integrals_batch(ni, nj, rij, zeta, g_ss, tore, basis):
    """
    Batched OM1 local two-center Gaussian pair data for all pairs.
    """
    ri = om1_repgau_local_batch(ni, nj, rij, zeta, basis)
    scaled_ri, fko = om1_gscale(rij, ri, ni, nj, g_ss)
    rept = om1_cordef(ri, ni, nj, tore)
    core_semi = om1_cordef(scaled_ri, ni, nj, tore)
    return {"ri": ri, "scaled_ri": scaled_ri, "fko": fko, "rept": rept, "core_semi": core_semi}


def _spgto2_local_batch(atomic_numbers, zeta_a, zeta_b, rab, basis):
    """
    Batched version of `_spgto2_local`.
    """
    device = rab.device
    dtype = rab.dtype

    shell_type, exps, cs, cp = gather_om1_basis(atomic_numbers, zeta_a, basis)
    shell_heavy = shell_type > 0

    core_exps = torch.tensor([2.227660584, 0.4057711562, 0.1098175104], dtype=dtype, device=device)
    core_coeffs = torch.tensor([0.1543289673, 0.5353281423, 0.4446345422], dtype=dtype, device=device)

    ec = core_exps.view(1, 3) * zeta_b.view(-1, 1).pow(2)
    cc = core_coeffs.view(1, 3) * (2.0 * ec / math.pi).pow(0.75)

    a = exps
    csa = cs
    cpa = cp
    rab2 = rab.view(-1, 1, 1).pow(2)
    g = a.unsqueeze(2) + ec.unsqueeze(1)
    xqq = a.unsqueeze(2) * ec.unsqueeze(1) * rab2 / g
    s00 = (math.pi / g).pow(1.5) * torch.exp(-xqq)

    s1 = torch.sum(csa.unsqueeze(2) * cc.unsqueeze(1) * s00, dim=(1, 2))
    s2 = torch.sum(
        cpa.unsqueeze(2) * cc.unsqueeze(1) * (ec.unsqueeze(1) * rab.view(-1, 1, 1) * s00 / g), dim=(1, 2)
    )
    s2 = torch.where(shell_heavy, s2, torch.zeros_like(s2))
    return s1, s2


def om2_corpp2_local_batch(ni, nj, rij, om2_tables, basis):
    """
    Batched OM2 CORPP2 for all pairs.
    """
    device = rij.device
    dtype = rij.dtype

    corpp = torch.zeros((ni.shape[0], 4, 2), dtype=dtype, device=device)
    zscor = om2_tables["zscor"]
    fscor = om2_tables["fscor"]
    bscor = om2_tables["bscor"]
    ascor = om2_tables["ascor"]

    idx0 = (nj > 2).nonzero(as_tuple=False).squeeze(1)
    # Column 0: atom i is the shell center, atom j is the ECP center.
    if idx0.numel():
        na = ni[idx0]
        nb = nj[idx0]
        rij0 = rij[idx0]
        s1, s2 = _spgto2_local_batch(na, om2_tables["zeta"][na], zscor[nb], rij0, basis)
        bas = om2_tables["beta_s"][na]
        aas = om2_tables["alpha_s"][na]
        bbs = bscor[nb]
        cbs = ascor[nb]
        r2 = rij0 * rij0
        sqr = torch.sqrt(rij0)
        t1 = 0.5 * (bas + bbs) * sqr * torch.exp(-(aas + cbs) * r2)
        corpp[idx0, 0, 0] = -2.0 * s1 * t1 - s1 * s1 * fscor[nb]
        heavy = na > 2
        if heavy.any():
            nah = na[heavy]
            nbh = nb[heavy]
            rijh = rij0[heavy]
            s1h = s1[heavy]
            s2h = s2[heavy]
            t1h = t1[heavy]
            bap = om2_tables["beta_p"][nah]
            aap = om2_tables["alpha_p"][nah]
            t3 = 0.5 * (bap + bbs[heavy]) * torch.sqrt(rijh) * torch.exp(-(aap + cbs[heavy]) * rijh * rijh)
            idxh = idx0[heavy]
            corpp[idxh, 1, 0] = -(s1h * t3 + t1h * s2h) - s1h * s2h * fscor[nbh]
            corpp[idxh, 2, 0] = -(s2h * t3 + t3 * s2h) - s2h * s2h * fscor[nbh]

    idx1 = (ni > 2).nonzero(as_tuple=False).squeeze(1)
    # Column 1: atom j is the shell center, atom i is the ECP center.
    if idx1.numel():
        na = nj[idx1]
        nb = ni[idx1]
        rij1 = rij[idx1]
        s1, s2 = _spgto2_local_batch(na, om2_tables["zeta"][na], zscor[nb], rij1, basis)
        bas = om2_tables["beta_s"][na]
        aas = om2_tables["alpha_s"][na]
        bbs = bscor[nb]
        cbs = ascor[nb]
        r2 = rij1 * rij1
        sqr = torch.sqrt(rij1)
        t1 = 0.5 * (bas + bbs) * sqr * torch.exp(-(aas + cbs) * r2)
        corpp[idx1, 0, 1] = -2.0 * s1 * t1 - s1 * s1 * fscor[nb]
        heavy = na > 2
        if heavy.any():
            nah = na[heavy]
            nbh = nb[heavy]
            rijh = rij1[heavy]
            s1h = s1[heavy]
            s2h = s2[heavy]
            t1h = t1[heavy]
            bap = om2_tables["beta_p"][nah]
            aap = om2_tables["alpha_p"][nah]
            t3 = 0.5 * (bap + bbs[heavy]) * torch.sqrt(rijh) * torch.exp(-(aap + cbs[heavy]) * rijh * rijh)
            idx = idx1[heavy]
            corpp[idx, 1, 1] = -(-(s1h * t3 + t1h * s2h) - s1h * s2h * fscor[nbh])
            corpp[idx, 2, 1] = -(s2h * t3 + t3 * s2h) - s2h * s2h * fscor[nbh]
    return corpp


def omx_local_pair_corrections(
    method, ni, nj, rij, zeta, g_ss, tore, s_local, t_local, u_ss, u_pp, fval1, fval2, basis, om2_tables=None
):
    """
    Build the full native Torch OM1 local pair bundle up to, but excluding,
    the OM1 ``CORPP`` pseudopotential correction.

    Parameters
    ----------
    ni, nj : (npairs,) tensor
        Atomic numbers of the pairs.
    rij : (npairs,) tensor
        Interatomic distances in bohr.
    s_local, t_local : (npairs, 5) tensor
        OM1 local overlap and resonance terms in BETOM shell order.

    Returns
    -------
    dict
        Contains the native OM1 pair objects:
        ``ri``, ``scaled_ri``, ``fko``, ``rept``, ``core_semi``, ``cort``,
        ``pen``, ``valpp_raw``, ``valpp1``, ``valpp``, and
        ``core_no_corpp``.
    """
    zeta_i = zeta[ni]
    zeta_j = zeta[nj]

    pair = om1_local_pair_integrals_batch(ni, nj, rij, zeta, g_ss, tore, basis)
    cort = om1_corgau(ni, nj, rij, zeta_i, zeta_j, tore, basis)
    pen = om1_penetration(cort, pair["rept"], pair["fko"])
    valpp_raw, valpp1 = om1_valpot(ni, nj, pair["core_semi"], s_local, t_local, u_ss, u_pp)
    if method == "OM3":
        valpp1 = torch.zeros_like(valpp1)
    valpp = om1_apply_valpot_scaling(ni, nj, valpp_raw, valpp1, fval1, fval2)
    if method == "OM1":
        corpp = om1_ppecp_local(ni, nj, rij, zeta_i, zeta_j, basis) * pair["fko"].view(-1, 1, 1)
    elif method in {"OM2", "OM3"}:
        if om2_tables is None:
            raise ValueError(f"{method} pair corrections require om2_tables")
        corpp = om2_corpp2_local_batch(ni, nj, rij, om2_tables, basis)
    else:
        raise ValueError(f"Unsupported method: {method}")
    core = om1_assemble_core(pair["core_semi"], pen, corpp, valpp)
    core_no_corpp = om1_assemble_core(pair["core_semi"], pen, None, valpp)

    return {
        **pair,
        "cort": cort,
        "pen": pen,
        "corpp": corpp,
        "valpp_raw": valpp_raw,
        "valpp1": valpp1,
        "valpp": valpp,
        "core": core,
        "core_no_corpp": core_no_corpp,
    }


def _om1_rotate_core_columns_batch(core_columns, rot):
    bsz = core_columns.shape[0]
    local = torch.zeros((bsz, 4, 4), dtype=core_columns.dtype, device=core_columns.device)
    local[:, 0, 0] = core_columns[:, 0]
    local[:, 0, 1] = core_columns[:, 1]
    local[:, 1, 0] = core_columns[:, 1]
    local[:, 1, 1] = core_columns[:, 2]
    local[:, 2, 2] = core_columns[:, 3]
    local[:, 3, 3] = core_columns[:, 3]
    u = torch.eye(4, dtype=core_columns.dtype, device=core_columns.device).unsqueeze(0).repeat(bsz, 1, 1)
    u[:, 1:, 1:] = rot
    return torch.triu(u @ local @ u.transpose(1, 2))


def _om1_rotate_w_batch(ri, rot):
    dtype = ri.dtype
    device = ri.device
    r0 = rot[:, 0, :]
    r1 = rot[:, 1, :]
    r2 = rot[:, 2, :]
    ri_s = [ri[:, i] for i in range(ri.shape[1])]
    bsz = ri.shape[0]
    w = torch.zeros((bsz, 100), dtype=dtype, device=device)
    combos = tuple(
        (kk, ll, mm, nn, kk - 1, ll - 1, mm - 1, nn - 1)
        for kk in range(4)
        for ll in range(kk + 1)
        for mm in range(4)
        for nn in range(mm + 1)
    )
    for idx, (kk, ll, mm, nn, k, l, m, n) in enumerate(combos):
        if kk == 0:
            if mm == 0:
                w[:, idx] = ri_s[0]
            elif nn == 0:
                w[:, idx] = ri_s[4] * r0[:, m]
            else:
                w[:, idx] = ri_s[10] * (r0[:, m] * r0[:, n]) + ri_s[11] * (
                    r1[:, m] * r1[:, n] + r2[:, m] * r2[:, n]
                )
        elif ll == 0:
            if mm == 0:
                w[:, idx] = ri_s[1] * r0[:, k]
            elif nn == 0:
                w[:, idx] = ri_s[5] * (r0[:, k] * r0[:, m]) + ri_s[6] * (
                    r1[:, k] * r1[:, m] + r2[:, k] * r2[:, m]
                )
            else:
                t0 = r0[:, k] * r0[:, m] * r0[:, n]
                t1 = (r1[:, m] * r1[:, n] + r2[:, m] * r2[:, n]) * r0[:, k]
                mix = r1[:, k] * (r1[:, n] * r0[:, m] + r1[:, m] * r0[:, n]) + r2[:, k] * (
                    r2[:, m] * r0[:, n] + r2[:, n] * r0[:, m]
                )
                w[:, idx] = ri_s[12] * t0 + ri_s[13] * t1 + ri_s[14] * mix
        else:
            if mm == 0:
                t0 = r0[:, k] * r0[:, l]
                t1 = r1[:, k] * r1[:, l] + r2[:, k] * r2[:, l]
                w[:, idx] = ri_s[2] * t0 + ri_s[3] * t1
            elif nn == 0:
                t0 = r0[:, k] * r0[:, l] * r0[:, m]
                t1 = (r1[:, k] * r1[:, l] + r2[:, k] * r2[:, l]) * r0[:, m]
                t2 = r1[:, l] * r1[:, m] + r2[:, l] * r2[:, m]
                w[:, idx] = (
                    ri_s[7] * t0
                    + ri_s[8] * t1
                    + ri_s[9] * (r0[:, k] * t2 + r0[:, l] * (r1[:, k] * r1[:, m] + r2[:, k] * r2[:, m]))
                )
            else:
                t0 = r0[:, k] * r0[:, l] * r0[:, m] * r0[:, n]
                t1 = (r1[:, k] * r1[:, l] + r2[:, k] * r2[:, l]) * r0[:, m] * r0[:, n]
                t2 = (r1[:, m] * r1[:, n] + r2[:, m] * r2[:, n]) * (r0[:, k] * r0[:, l])
                quad = r1[:, k] * r1[:, l] * r1[:, m] * r1[:, n] + r2[:, k] * r2[:, l] * r2[:, m] * r2[:, n]
                mix1 = r0[:, m] * (r1[:, l] * r1[:, n] + r2[:, l] * r2[:, n])
                mix2 = r0[:, n] * (r1[:, l] * r1[:, m] + r2[:, l] * r2[:, m])
                val5 = r0[:, k] * (mix1 + mix2) + r0[:, l] * (
                    r0[:, m] * (r1[:, k] * r1[:, n] + r2[:, k] * r2[:, n])
                    + r0[:, n] * (r1[:, k] * r1[:, m] + r2[:, k] * r2[:, m])
                )
                mix3 = r1[:, k] * r1[:, l] * r2[:, m] * r2[:, n] + r2[:, k] * r2[:, l] * r1[:, m] * r1[:, n]
                cross = (r1[:, k] * r2[:, l] + r2[:, k] * r1[:, l]) * (
                    r1[:, m] * r2[:, n] + r2[:, m] * r1[:, n]
                )
                w[:, idx] = (
                    ri_s[15] * t0
                    + ri_s[16] * t1
                    + ri_s[17] * t2
                    + ri_s[18] * quad
                    + ri_s[19] * val5
                    + ri_s[20] * mix3
                    + ri_s[21] * cross
                )
    return w.view(bsz, 10, 10)


def _omx_pair_hcore_terms_impl(
    method,
    ni,
    nj,
    rij,
    zeta,
    g_ss,
    tore,
    s_local,
    t_local,
    u_ss,
    u_pp,
    fval1,
    fval2,
    rot,
    rot_t,
    basis,
    om2_tables=None,
):
    zeta_i = zeta[ni]
    zeta_j = zeta[nj]

    pair = om1_local_pair_integrals_batch(ni, nj, rij, zeta, g_ss, tore, basis)
    cort = om1_corgau(ni, nj, rij, zeta_i, zeta_j, tore, basis)
    pen = om1_penetration(cort, pair["rept"], pair["fko"])
    valpp_raw, valpp1 = om1_valpot(ni, nj, pair["core_semi"], s_local, t_local, u_ss, u_pp)
    if method == "OM3":
        valpp1 = torch.zeros_like(valpp1)
    valpp = om1_apply_valpot_scaling(ni, nj, valpp_raw, valpp1, fval1, fval2)

    if method == "OM1":
        corpp = om1_ppecp_local(ni, nj, rij, zeta_i, zeta_j, basis) * pair["fko"].view(-1, 1, 1)
    elif method in {"OM2", "OM3"}:
        if om2_tables is None:
            raise ValueError(f"{method} batched pair corrections require om2_tables")
        corpp = om2_corpp2_local_batch(ni, nj, rij, om2_tables, basis)
    else:
        raise ValueError(f"Unsupported method: {method}")

    core = om1_assemble_core(pair["core_semi"], pen, corpp=corpp, valpp=valpp)
    w = _om1_rotate_w_batch(pair["scaled_ri"], rot)
    e1b = _om1_rotate_core_columns_batch(core[:, :, 0], rot_t)
    e2a = _om1_rotate_core_columns_batch(core[:, :, 1], rot_t)

    out = {"w": w, "e1b": e1b, "e2a": e2a, "fko": pair["fko"]}
    if method != "OM1":
        out["core_semi"] = pair["core_semi"]
    return out


def omx_pair_hcore_terms(
    method,
    ni,
    nj,
    rij,
    zeta,
    g_ss,
    tore,
    s_local,
    t_local,
    u_ss,
    u_pp,
    fval1,
    fval2,
    rot,
    rot_t,
    basis,
    om2_tables=None,
):
    """
    Batched OMx pair builder for all pairs.
    """
    return _omx_pair_hcore_terms_impl(
        method,
        ni,
        nj,
        rij,
        zeta,
        g_ss,
        tore,
        s_local,
        t_local,
        u_ss,
        u_pp,
        fval1,
        fval2,
        rot,
        rot_t,
        basis,
        om2_tables=om2_tables,
    )

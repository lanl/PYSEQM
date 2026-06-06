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
from .om1_overlap import _lookup_basis
from .om1_ppecp import om1_ppecp_local
from .two_elec_two_center_int import rotate_with_quaternion

_XX3498 = 34.9868366552497


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
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    dtype = x.dtype

    if dtype in (torch.float32, torch.bfloat16):
        # float32 loses cancellation accuracy much earlier
        default_small = 1.0e-3
    else:
        # float64 is safer
        default_small = 1.0e-6

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


def _shell_pair_data(atomic_number, zeta):
    shell_type, exponents, coeff_s, coeff_p = _lookup_basis(
        torch.tensor([atomic_number], dtype=torch.int64), torch.tensor([zeta], dtype=torch.float64)
    )
    shell_type = int(shell_type.item())
    a = exponents[0]
    cs = coeff_s[0]
    cp = coeff_p[0]
    ai = a.view(-1, 1)
    aj = a.view(1, -1)
    gab = (ai + aj).reshape(-1)
    eab = 1.0 / gab
    pref = _XX3498 * eab
    csij = (cs.view(-1, 1) * cs.view(1, -1)).reshape(-1)
    out = {"shell_type": shell_type, "gab": gab, "eab": eab}
    if shell_type == 0:
        out["dp00"] = pref * csij
    else:
        cpij = (cp.view(-1, 1) * cp.view(1, -1)).reshape(-1)
        cpis = (cp.view(-1, 1) * cs.view(1, -1)).reshape(-1)
        out["dp11"] = pref * cpij
        out["dp00"] = (pref * csij) / out["dp11"]
        out["dp10"] = (pref * cpis) / out["dp11"]
    return out


def _q_pair_data(atomic_number, zeta):
    shell_type, exponents, coeff_s, coeff_p = _lookup_basis(
        torch.tensor([atomic_number], dtype=torch.int64), torch.tensor([zeta], dtype=torch.float64)
    )
    shell_type = int(shell_type.item())
    a = exponents[0]
    cs = coeff_s[0]
    cp = coeff_p[0]
    ai = a.view(-1, 1)
    aj = a.view(1, -1)
    gcd = (ai + aj).reshape(-1)
    ecd = 1.0 / gcd
    if shell_type == 0:
        return {
            "shell_type": shell_type,
            "gcd": gcd,
            "ecd": ecd,
            "dq00": (cs.view(-1, 1) * cs.view(1, -1)).reshape(-1),
        }
    return {
        "shell_type": shell_type,
        "gcd": gcd,
        "ecd": ecd,
        "dq00": (cs.view(-1, 1) * cs.view(1, -1)).reshape(-1),
        "dq10": (cp.view(-1, 1) * cs.view(1, -1)).reshape(-1),
        "dq11": (cp.view(-1, 1) * cp.view(1, -1)).reshape(-1),
    }


def _s_only_pair_data(atomic_number, zeta):
    shell_type, exponents, coeff_s, _coeff_p = _lookup_basis(
        torch.tensor([atomic_number], dtype=torch.int64), torch.tensor([zeta], dtype=torch.float64)
    )
    a = exponents[0]
    cs = coeff_s[0]
    ai = a.view(-1, 1)
    aj = a.view(1, -1)
    gab = (ai + aj).reshape(-1)
    eab = 1.0 / gab
    pref = _XX3498 * eab
    csij = (cs.view(-1, 1) * cs.view(1, -1)).reshape(-1)
    return {"shell_type": 0, "gab": gab, "eab": eab, "dp00": pref * csij}


def _sp0000(p, q, rab):
    rab2 = rab * rab
    x = rab2 / (p["eab"] + q["ecd"].view(-1, 1))
    sq2 = 1.0 / torch.sqrt(p["gab"].view(1, -1) + q["gcd"].view(-1, 1))
    f0, _, _, _, _ = boys_0_to_4(x)
    h0000 = torch.sum(f0 * (p["dp00"].view(1, -1) * sq2), dim=1)
    g0000 = torch.sum(h0000 * (q["dq00"] * q["ecd"]) * 27.21)
    out = torch.zeros(22, dtype=torch.float64)
    out[0] = g0000
    return out


def _sp0011(p, q, rab):
    rab2 = rab * rab
    eab = p["eab"].view(1, -1)
    ecd = q["ecd"].view(-1, 1)
    gcd = q["gcd"].view(-1, 1)
    x = rab2 / (eab + ecd)
    gfac = (1.0 / (eab + ecd)) * ecd
    sq2 = 1.0 / torch.sqrt(p["gab"].view(1, -1) + gcd)
    y = p["dp00"].view(1, -1) * sq2
    gy = gfac * y
    ggy = gfac * gy
    f0, f1, f2, _, _ = boys_0_to_4(x)
    h0000 = torch.sum(f0 * y, dim=1)
    h0001 = torch.sum(f1 * gy, dim=1)
    h0033 = torch.sum(f2 * ggy, dim=1)
    h0030 = -h0001 * rab
    h0033 = h0033 * rab2
    h0022 = 0.5 * q["ecd"] * (h0000 - h0001)
    h0033 = h0033 + h0022

    dq_scale = q["ecd"] * 27.21
    g0000 = torch.sum(h0000 * (q["dq00"] * dq_scale))
    g0022 = torch.sum(h0022 * (q["dq11"] * dq_scale))
    g0033 = torch.sum(h0033 * (q["dq11"] * dq_scale))
    g0030 = torch.sum(h0030 * (q["dq10"] * dq_scale))

    out = torch.zeros(22, dtype=torch.float64)
    out[0] = g0000
    out[4] = g0030
    out[10] = g0033
    out[11] = g0022
    return out


def _sp1111(p, q, rab):
    rab2 = rab * rab
    rab3 = rab * rab2
    rab4 = rab2 * rab2
    eab = p["eab"].view(1, -1)
    eab2 = eab * eab
    ecd = q["ecd"].view(-1, 1)
    x = rab2 / (eab + ecd)
    # gfac = (1.0 / (eab + ecd)) * ecd
    gfac = 1.0 / (eab + ecd)
    sq2 = 1.0 / torch.sqrt(p["gab"].view(1, -1) + q["gcd"].view(-1, 1))
    y = p["dp11"].view(1, -1) * sq2
    gy = gfac * y
    ggy = gfac * gy
    gggy = gfac * ggy
    f0, f1, f2, f3, f4 = boys_0_to_4(x)
    ff0 = f0 * y
    ff1 = f1 * gy
    ff2 = f2 * ggy
    ff3 = f3 * gggy
    ff4 = f4 * gggy * gfac

    dp00 = p["dp00"].view(1, -1)
    edp10 = eab * p["dp10"].view(1, -1)

    x0 = torch.sum(ff0 * dp00, dim=1)
    x1 = torch.sum(ff1 * dp00, dim=1)
    x2 = torch.sum(ff2 * dp00, dim=1)
    w1 = torch.sum(ff1 * edp10, dim=1)
    w2 = torch.sum(ff2 * edp10, dim=1)
    w3 = torch.sum(ff3 * edp10, dim=1)
    s0 = torch.sum(ff0 * eab, dim=1)
    s1 = torch.sum(ff1 * eab, dim=1)
    s2 = torch.sum(ff2 * eab, dim=1)
    t1 = torch.sum(ff1 * eab2, dim=1)
    t2 = torch.sum(ff2 * eab2, dim=1)
    t3 = torch.sum(ff3 * eab2, dim=1)
    t4 = torch.sum(ff4 * eab2, dim=1)

    hecd = 0.5 * q["ecd"]
    ecd2 = q["ecd"] * q["ecd"]
    hecd2 = 0.5 * ecd2
    h0000 = x0
    h0030 = -q["ecd"] * x1 * rab
    h0022 = hecd * (x0 - q["ecd"] * x1)
    h0033 = h0022 + ecd2 * x2 * rab2
    h2020 = hecd * w1
    h2023 = -hecd2 * w2 * rab
    h3000 = w1 * rab
    h3030 = h2020 - q["ecd"] * w2 * rab2
    h3022 = h2023 + hecd * h3000
    h3033 = h3022 + h2023 + h2023 + ecd2 * w3 * rab3
    h2200 = 0.5 * (s0 - t1)
    h3300 = h2200 + t2 * rab2
    h2320 = hecd * t2 * rab
    h2230 = hecd * (t2 - s1) * rab
    h3330 = h2230 + q["ecd"] * (t2 * rab - t3 * rab3)
    h1212 = 0.25 * ecd2 * t2
    h2323 = hecd2 * (0.5 * t2 - t3 * rab2)
    hxxyy = 0.25 * (q["ecd"] * (s0 - t1) - ecd2 * (s1 - t2))
    h2222 = hxxyy + hecd2 * t2
    h1122 = hxxyy
    h3322 = hxxyy + hecd * t2 * rab2 - hecd2 * t3 * rab2
    h2233 = hxxyy + hecd2 * (s2 - t3) * rab2
    h3333 = hxxyy + hecd2 * (t2 + s2 * rab2) + ecd2 * (-3.0 * t3 * rab2 + t4 * rab4) + hecd * t2 * rab2

    dq_scale = q["ecd"] * 27.21
    dq00 = q["dq00"] * dq_scale
    dq10 = q["dq10"] * dq_scale
    dq11 = q["dq11"] * dq_scale

    out = torch.zeros(22, dtype=torch.float64)
    out[0] = torch.sum(h0000 * dq00)
    out[1] = torch.sum(h3000 * dq00)
    out[2] = torch.sum(h3300 * dq00)
    out[3] = torch.sum(h2200 * dq00)
    out[4] = torch.sum(h0030 * dq10)
    out[5] = torch.sum(h3030 * dq10)
    out[6] = torch.sum(h2020 * dq10)
    out[7] = torch.sum(h3330 * dq10)
    out[8] = torch.sum(h2230 * dq10)
    out[9] = torch.sum(h2320 * dq10)
    out[10] = torch.sum(h0033 * dq11)
    out[11] = torch.sum(h0022 * dq11)
    out[12] = torch.sum(h3033 * dq11)
    out[13] = torch.sum(h3022 * dq11)
    out[14] = torch.sum(h2023 * dq11)
    out[15] = torch.sum(h3333 * dq11)
    out[16] = torch.sum(h2233 * dq11)
    out[17] = torch.sum(h3322 * dq11)
    out[18] = torch.sum(h2222 * dq11)
    out[19] = torch.sum(h2323 * dq11)
    out[20] = torch.sum(h1122 * dq11)
    out[21] = torch.sum(h1212 * dq11)
    return out


def om1_repgau_local(ni, nj, rij, zeta_s):
    """
    Native Torch implementation of the OM1 REPGAU local ``RI(22)`` tensor for the
    current H/C/N/O/F ``s/sp`` scope.
    """
    p = _shell_pair_data(int(ni), float(zeta_s[int(ni)]))
    q = _q_pair_data(int(nj), float(zeta_s[int(nj)]))

    if p["shell_type"] == 0 and q["shell_type"] == 0:
        return _sp0000(p, q, float(rij))
    if p["shell_type"] == 0 and q["shell_type"] == 1:
        return _sp0011(p, q, float(rij))
    if p["shell_type"] == 1 and q["shell_type"] == 0:
        tmp = _sp0011(
            _s_only_pair_data(int(nj), float(zeta_s[int(nj)])),
            _q_pair_data(int(ni), float(zeta_s[int(ni)])),
            float(rij),
        )
        out = torch.zeros(22, dtype=torch.float64)
        out[0] = tmp[0]
        out[1] = -tmp[4]
        out[2] = tmp[10]
        out[3] = tmp[11]
        return out
    out = _sp1111(p, q, float(rij))
    return out


def om1_local_pair_integrals(ni, nj, rij, zeta_s, g_ss, tore):
    """
    Build OM1 local two-center Gaussian pair data in native Torch.
    """
    ri = om1_repgau_local(ni, nj, rij, zeta_s).view(1, 22)
    scaled_ri, fko = om1_gscale(
        torch.tensor([float(rij)], dtype=torch.float64),
        ri,
        torch.tensor([int(ni)], dtype=torch.int64),
        torch.tensor([int(nj)], dtype=torch.int64),
        g_ss,
    )
    rept = om1_cordef(ri, torch.tensor([int(ni)]), torch.tensor([int(nj)]), tore)[0]
    core_semi = om1_cordef(scaled_ri, torch.tensor([int(ni)]), torch.tensor([int(nj)]), tore)[0]
    return {"ri": ri[0], "scaled_ri": scaled_ri[0], "fko": fko[0], "rept": rept, "core_semi": core_semi}


def omx_local_pair_integrals(ni, nj, rij, zeta_s, g_ss, tore):
    return om1_local_pair_integrals(ni, nj, rij, zeta_s, g_ss, tore)


def _spgto2_local(atomic_number, zeta_a, zeta_b, rab):
    device = rab.device
    dtype = torch.float64

    z = torch.tensor([atomic_number], dtype=torch.int64, device=device)
    za = torch.as_tensor([zeta_a], dtype=dtype, device=device)

    shell_type, exps, cs, cp = _lookup_basis(z, za)

    shell_type = int(shell_type[0].item())
    exps = exps[0]
    cs = cs[0]
    cp = cp[0]

    core_exps = torch.tensor([2.227660584, 0.4057711562, 0.1098175104], dtype=dtype, device=device)
    core_coeffs = torch.tensor([0.1543289673, 0.5353281423, 0.4446345422], dtype=dtype, device=device)

    ec = core_exps * zeta_b**2
    cc = core_coeffs * (2.0 * ec / math.pi).pow(0.75)

    rab2 = rab * rab
    s1 = torch.zeros((), dtype=dtype, device=device)
    s2 = torch.zeros((), dtype=dtype, device=device)

    for ii in range(3):
        a = exps[ii]
        csa = cs[ii]
        cpa = cp[ii]

        for jj in range(3):
            b = ec[jj]
            csb = cc[jj]

            g = a + b
            xqq = a * b * rab2 / g
            if xqq > 60.0:
                continue

            s00 = (math.pi / g) ** 1.5 * torch.exp(-xqq)
            s1 = s1 + csa * csb * s00

            if shell_type > 0:
                s30 = b * rab * s00 / g
                s2 = s2 + cpa * csb * s30

    return s1, s2


def om2_corpp2_local(ni, nj, rij, om2_tables):
    """
    Literal OM2 CORPP2 for the current H/C/N/O/F first-row scope.
    """
    dtype = torch.float64
    device = rij.device if torch.is_tensor(rij) else torch.device("cpu")
    corpp = torch.zeros((4, 2), dtype=dtype, device=device)

    ni = int(ni)
    nj = int(nj)
    if ni <= 2 and nj <= 2:
        return corpp

    zscor = om2_tables["zscor"]
    fscor = om2_tables["fscor"]
    bscor = om2_tables["bscor"]
    ascor = om2_tables["ascor"]

    for iatom, (na, nb, nshell, col) in enumerate(((ni, nj, ni, 0), (nj, ni, nj, 1))):
        if nb <= 2:
            continue
        zb = zscor[nb]
        fb = fscor[nb]
        s1, s2 = _spgto2_local(na, om2_tables["zeta_s"][na], zb, rij)
        bas = om2_tables["beta_s"][na]
        aas = om2_tables["alpha_s"][na]
        bbs = bscor[nb]
        cbs = ascor[nb]
        r2 = rij * rij
        sqr = torch.sqrt(rij)
        t1 = 0.5 * (bas + bbs) * sqr * torch.exp(-(aas + cbs) * r2)
        corpp[0, col] = -(s1 * t1 + t1 * s1) - s1 * s1 * fb
        if na > 2:
            bap = om2_tables["beta_p"][na]
            aap = om2_tables["alpha_p"][na]
            t3 = 0.5 * (bap + bbs) * sqr * torch.exp(-(aap + cbs) * r2)
            corpp[1, col] = -(s1 * t3 + t1 * s2) - s1 * s2 * fb
            corpp[2, col] = -(s2 * t3 + t3 * s2) - s2 * s2 * fb
            if col == 1:
                corpp[1, col] = -corpp[1, col]
    return corpp


def omx_local_pair_corrections(
    method, ni, nj, rij, zeta_s, g_ss, tore, s_local, t_local, u_ss, u_pp, fval1, fval2, om2_tables=None
):
    """
    Build the full native Torch OM1 local pair bundle up to, but excluding,
    the OM1 ``CORPP`` pseudopotential correction.

    Parameters
    ----------
    ni, nj : int
        Atomic numbers of the pair.
    rij : float
        Interatomic distance in bohr.
    s_local, t_local : (5,) tensor-like
        OM1 local overlap and resonance terms in BETOM shell order.

    Returns
    -------
    dict
        Contains the native OM1 pair objects:
        ``ri``, ``scaled_ri``, ``fko``, ``rept``, ``core_semi``, ``cort``,
        ``pen``, ``valpp_raw``, ``valpp1``, ``valpp``, and
        ``core_no_corpp``.
    """
    pair = omx_local_pair_integrals(ni, nj, rij, zeta_s, g_ss, tore)
    ni_t = torch.tensor([int(ni)], dtype=torch.int64)
    nj_t = torch.tensor([int(nj)], dtype=torch.int64)
    rij_t = torch.tensor([float(rij)], dtype=torch.float64)
    s_local_t = torch.as_tensor(s_local, dtype=torch.float64).view(1, 5)
    t_local_t = torch.as_tensor(t_local, dtype=torch.float64).view(1, 5)

    cort = om1_corgau(ni_t, nj_t, rij_t, zeta_s, tore)[0]
    pen = om1_penetration(cort.unsqueeze(0), pair["rept"].unsqueeze(0), pair["fko"].view(1))[0]
    valpp_raw, valpp1 = om1_valpot(
        ni_t, nj_t, pair["core_semi"].unsqueeze(0), s_local_t, t_local_t, u_ss, u_pp
    )
    if method == "OM3":
        valpp1 = torch.zeros_like(valpp1)
    valpp = om1_apply_valpot_scaling(ni_t, nj_t, valpp_raw, valpp1, fval1, fval2)[0]
    if method == "OM1":
        corpp = om1_ppecp_local(int(ni), int(nj), float(rij), zeta_s) * pair["fko"]
    elif method in {"OM2", "OM3"}:
        if om2_tables is None:
            raise ValueError(f"{method} pair corrections require om2_tables")
        corpp = om2_corpp2_local(int(ni), int(nj), torch.tensor(float(rij), dtype=torch.float64), om2_tables)
    else:
        raise ValueError(f"Unsupported method: {method}")
    core = om1_assemble_core(
        pair["core_semi"].unsqueeze(0), pen.unsqueeze(0), corpp.unsqueeze(0), valpp.unsqueeze(0)
    )[0]
    core_no_corpp = om1_assemble_core(
        pair["core_semi"].unsqueeze(0), pen.unsqueeze(0), None, valpp.unsqueeze(0)
    )[0]

    return {
        **pair,
        "cort": cort,
        "pen": pen,
        "corpp": corpp,
        "valpp_raw": valpp_raw[0],
        "valpp1": valpp1[0],
        "valpp": valpp,
        "core": core,
        "core_no_corpp": core_no_corpp,
    }


def _om1_rotate_core_column(core_column, xij):
    rot = rotate_with_quaternion(xij.view(1, 3))[0]
    local = torch.zeros((4, 4), dtype=core_column.dtype, device=core_column.device)
    local[0, 0] = core_column[0]
    local[0, 1] = core_column[1]
    local[1, 0] = core_column[1]
    local[1, 1] = core_column[2]
    local[2, 2] = core_column[3]
    local[3, 3] = core_column[3]
    u = torch.eye(4, dtype=core_column.dtype, device=core_column.device)
    u[1:, 1:] = rot.transpose(0, 1)
    return (u @ local @ u.transpose(0, 1)).triu()


def _om1_rotate_w(ri, xij):
    dtype = ri.dtype
    device = ri.device
    v = xij.view(1, 3)
    rot = rotate_with_quaternion(v)[0]
    r0 = rot[0]
    r1 = rot[1]
    r2 = rot[2]
    ri_s = ri.unbind(dim=-1)
    w = torch.zeros(100, dtype=dtype, device=device)
    combos = [
        (kk, ll, mm, nn) for kk in range(4) for ll in range(kk + 1) for mm in range(4) for nn in range(mm + 1)
    ]
    idx = 0
    for kk, ll, mm, nn in combos:
        k = kk - 1
        l = ll - 1
        m = mm - 1
        n = nn - 1
        if kk == 0:
            if mm == 0:
                w[idx] = ri_s[0]
            elif nn == 0:
                w[idx] = ri_s[4] * r0[m]
            else:
                w[idx] = ri_s[10] * (r0[m] * r0[n]) + ri_s[11] * (r1[m] * r1[n] + r2[m] * r2[n])
        elif ll == 0:
            if mm == 0:
                w[idx] = ri_s[1] * r0[k]
            elif nn == 0:
                w[idx] = ri_s[5] * (r0[k] * r0[m]) + ri_s[6] * (r1[k] * r1[m] + r2[k] * r2[m])
            else:
                t0 = r0[k] * r0[m] * r0[n]
                t1 = (r1[m] * r1[n] + r2[m] * r2[n]) * r0[k]
                mix = r1[k] * (r1[n] * r0[m] + r1[m] * r0[n]) + r2[k] * (r2[m] * r0[n] + r2[n] * r0[m])
                w[idx] = ri_s[12] * t0 + ri_s[13] * t1 + ri_s[14] * mix
        else:
            if mm == 0:
                t0 = r0[k] * r0[l]
                t1 = r1[k] * r1[l] + r2[k] * r2[l]
                w[idx] = ri_s[2] * t0 + ri_s[3] * t1
            elif nn == 0:
                t0 = r0[k] * r0[l] * r0[m]
                t1 = (r1[k] * r1[l] + r2[k] * r2[l]) * r0[m]
                t2 = r1[l] * r1[m] + r2[l] * r2[m]
                w[idx] = (
                    ri_s[7] * t0
                    + ri_s[8] * t1
                    + ri_s[9] * (r0[k] * t2 + r0[l] * (r1[k] * r1[m] + r2[k] * r2[m]))
                )
            else:
                t0 = r0[k] * r0[l] * r0[m] * r0[n]
                t1 = (r1[k] * r1[l] + r2[k] * r2[l]) * r0[m] * r0[n]
                t2 = (r1[m] * r1[n] + r2[m] * r2[n]) * (r0[k] * r0[l])
                quad = r1[k] * r1[l] * r1[m] * r1[n] + r2[k] * r2[l] * r2[m] * r2[n]
                mix1 = r0[m] * (r1[l] * r1[n] + r2[l] * r2[n])
                mix2 = r0[n] * (r1[l] * r1[m] + r2[l] * r2[m])
                val5 = r0[k] * (mix1 + mix2) + r0[l] * (
                    r0[m] * (r1[k] * r1[n] + r2[k] * r2[n]) + r0[n] * (r1[k] * r1[m] + r2[k] * r2[m])
                )
                mix3 = r1[k] * r1[l] * r2[m] * r2[n] + r2[k] * r2[l] * r1[m] * r1[n]
                cross = (r1[k] * r2[l] + r2[k] * r1[l]) * (r1[m] * r2[n] + r2[m] * r1[n])
                w[idx] = (
                    ri_s[15] * t0
                    + ri_s[16] * t1
                    + ri_s[17] * t2
                    + ri_s[18] * quad
                    + ri_s[19] * val5
                    + ri_s[20] * mix3
                    + ri_s[21] * cross
                )
        idx += 1
    return w.view(10, 10)


def omx_pair_hcore_terms(
    method, ni, nj, xij, rij, zeta_s, g_ss, tore, s_local, t_local, u_ss, u_pp, fval1, fval2, om2_tables=None
):
    pair = omx_local_pair_corrections(
        method, ni, nj, rij, zeta_s, g_ss, tore, s_local, t_local, u_ss, u_pp, fval1, fval2, om2_tables
    )
    return {
        **pair,
        "w": _om1_rotate_w(pair["scaled_ri"], xij),
        "e1b": _om1_rotate_core_column(pair["core"][:, 0], xij),
        "e2a": _om1_rotate_core_column(pair["core"][:, 1], xij),
    }


def om1_local_pair_corrections(ni, nj, rij, zeta_s, g_ss, tore, s_local, t_local, u_ss, u_pp, fval1, fval2):
    return omx_local_pair_corrections(
        "OM1", ni, nj, rij, zeta_s, g_ss, tore, s_local, t_local, u_ss, u_pp, fval1, fval2
    )


def om1_pair_hcore_terms(ni, nj, xij, rij, zeta_s, g_ss, tore, s_local, t_local, u_ss, u_pp, fval1, fval2):
    return omx_pair_hcore_terms(
        "OM1", ni, nj, xij, rij, zeta_s, g_ss, tore, s_local, t_local, u_ss, u_pp, fval1, fval2
    )

import math

import torch

from .constants import ev
from .om1_core_corrections import (
    _fmtgen_fortran,
    om1_apply_valpot_scaling,
    om1_assemble_core,
    om1_cordef,
    om1_corgau,
    om1_gscale,
    om1_penetration,
    om1_valpot,
)
from .om1_ppecp import om1_ppecp_local

_XX3498 = 34.9868366552497


def _om1_pair_basis_data_batch(basis):
    if basis["shell_type"].numel() == 0:
        device = basis["exponents"].device
        dtype = basis["exponents"].dtype
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

    shell_type = basis["shell_type"]
    exponents = basis["exponents"]
    coeff_s = basis["coeff_s"]
    coeff_p = basis["coeff_p"]
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


def _om1_pair_q_data_batch(basis):
    if basis["shell_type"].numel() == 0:
        device = basis["exponents"].device
        dtype = basis["exponents"].dtype
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

    shell_type = basis["shell_type"]
    exponents = basis["exponents"]
    coeff_s = basis["coeff_s"]
    coeff_p = basis["coeff_p"]
    ai = exponents.unsqueeze(2)
    aj = exponents.unsqueeze(1)
    gcd = (ai + aj).reshape(exponents.shape[0], -1)
    ecd = 1.0 / gcd
    dq00 = (coeff_s.unsqueeze(2) * coeff_s.unsqueeze(1)).reshape(exponents.shape[0], -1)
    dq10 = (coeff_p.unsqueeze(2) * coeff_s.unsqueeze(1)).reshape(exponents.shape[0], -1)
    dq11 = (coeff_p.unsqueeze(2) * coeff_p.unsqueeze(1)).reshape(exponents.shape[0], -1)
    return {"shell_type": shell_type, "gcd": gcd, "ecd": ecd, "dq00": dq00, "dq10": dq10, "dq11": dq11}


MAXFMT = 400
PT05 = 0.05
PT184 = 0.184
XMAX = 19.9  # (MAXFMT - 2) * PT05   # 19.9
XLIM = 20.0
PT7853 = 0.785398163397448  # PI / 4, as in GTPREP

# _GAMGEN_CACHE = {}
#


# def _gamgen_table(dtype, device):
#     key = (str(device), dtype)
#     if key in _GAMGEN_CACHE:
#         return _GAMGEN_CACHE[key]
#
#     # Fortran: T=-0.15; then I=1..MAXFMT+4: T=T+0.05
#     i = torch.arange(1, MAXFMT + 5, dtype=dtype, device=device)
#     t = -0.15 + PT05 * i
#     f = _fmtgen_fortran(t, 5)  # [404, 5]
#
#     base = f[2:2 + MAXFMT]
#     nxt = f[3:3 + MAXFMT]
#     prv = f[1:1 + MAXFMT]
#     mm2 = f[0:0 + MAXFMT]
#     pp2 = f[4:4 + MAXFMT]
#
#     a = base
#     b = nxt - base
#     t1 = nxt + prv - 2.0 * base
#     t2 = 6.0 * base - 4.0 * (nxt + prv) + mm2 + pp2
#     c = (t1 - PT184 * t2) / 6.0
#
#     _GAMGEN_CACHE[key] = (a, b, c)
#     return a, b, c
#

# def boys_fortran(t, m_count=5):
#     """
#     Fortran-compatible GAMGEN/FMTGEN path.
#     For SP kernels, still handle X > XLIM outside this function,
#     because Fortran uses SP-specific large-X shortcuts there.
#     """
#     a, b, c = _gamgen_table(t.dtype, t.device)
#     out = t.new_empty(t.shape + (m_count,))
#
#     interp = t < XMAX
#     direct = ~interp
#
#     if bool(interp.any()):
#         x = t[interp]
#         qq = x * 20.0
#         n = torch.floor(qq).to(torch.long).clamp(0, MAXFMT - 2)
#         th = qq - n.to(x.dtype)
#
#         th2 = th * (th - 1.0)
#         th3 = th2 * (th - 2.0)
#         th4 = th2 * (th + 1.0)
#
#         out[interp] = (
#             a[n, :m_count]
#             + th[..., None] * b[n, :m_count]
#             - th3[..., None] * c[n, :m_count]
#             + th4[..., None] * c[n + 1, :m_count]
#         )
#
#     if bool(direct.any()):
#         out[direct] = _fmtgen_fortran(t[direct], m_count)
#
#     return tuple(out[..., m] for m in range(m_count))
#


def boys_from_table(x, boys_table, m_count):
    """
        X < XMAX          table interpolation
        XMAX <= X <= XLIM direct FMTGEN
        X > XLIM          unused here; SP code handles asymptotic separately
    Returns F0...F_{m_count-1}.
    """
    vals = x.new_zeros(x.shape + (m_count,))

    use_table = x < boys_table.xmax
    use_direct = (x >= boys_table.xmax) & (x <= boys_table.xlim)

    if bool(use_table.any()):
        tab_vals = boys_table(x[use_table], m_count)
        vals[use_table] = torch.stack(tab_vals, dim=-1)

    if bool(use_direct.any()):
        vals[use_direct] = _fmtgen_fortran(x[use_direct], m_count)

    return tuple(vals[..., m] for m in range(m_count))


def _sq_terms(p, q):
    gab = p["gab"].view(-1, 1, 9)
    gcd = q["gcd"].view(-1, 9, 1)
    sq1 = torch.sqrt(PT7853 / (gab * gcd))
    sq2 = 1.0 / torch.sqrt(gab + gcd)
    return sq1, sq2


def _select_pair_data(data, idx):
    return {k: v[idx] for k, v in data.items()}


def _sp0000_batch(p, q, rab, boys_table):
    rab1 = rab.view(-1, 1, 1)
    rab2 = rab1 * rab1
    x = rab2 / (p["eab"].view(-1, 1, 9) + q["ecd"].view(-1, 9, 1))

    sq1, sq2 = _sq_terms(p, q)
    dp00 = p["dp00"].view(-1, 1, 9)

    large = x > boys_table.xmax
    # xb = torch.where(large, torch.zeros_like(x), x)
    # f0 = boys_table(xb, 1)[0]
    f0 = boys_from_table(x, boys_table, 1)[0]

    normal = f0 * dp00 * sq2
    asymp = dp00 * sq1 / rab1
    h0000 = torch.sum(torch.where(large, asymp, normal), dim=2)

    dq = q["dq00"] * q["ecd"] * ev
    g0000 = torch.sum(h0000 * dq, dim=1)

    out = torch.zeros((rab.shape[0], 22), dtype=rab.dtype, device=rab.device)
    out[:, 0] = g0000
    return out


def _sp0011_batch(p, q, rab, boys_table):
    rab1 = rab.view(-1, 1)
    rab3 = rab.view(-1, 1, 1)
    rab2 = rab3 * rab3

    eab = p["eab"].view(-1, 1, 9)
    ecd = q["ecd"].view(-1, 9, 1)
    x = rab2 / (eab + ecd)
    g = ecd / (eab + ecd)

    sq1, sq2 = _sq_terms(p, q)
    y = p["dp00"].view(-1, 1, 9) * sq2
    gy = g * y
    ggy = g * gy

    large = x > boys_table.xlim
    f0, f1, f2 = boys_from_table(x, boys_table, 3)

    f0n = f0 * y
    f1n = f1 * gy
    f2n = f2 * ggy

    f0a = p["dp00"].view(-1, 1, 9) * sq1 / rab3
    gtx = ecd / (rab3 * rab3)
    f1a = 0.5 * f0a * gtx
    f2a = 1.5 * f1a * gtx

    ff0 = torch.where(large, f0a, f0n)
    ff1 = torch.where(large, f1a, f1n)
    ff2 = torch.where(large, f2a, f2n)

    h0000 = torch.sum(ff0, dim=2)
    h0001 = torch.sum(ff1, dim=2)
    h0033 = torch.sum(ff2, dim=2)

    ecd2d = q["ecd"].view(-1, 9)
    h0030 = -h0001 * rab1
    h0022 = 0.5 * ecd2d * (h0000 - h0001)
    h0033 = h0033 * rab1.pow(2) + h0022

    dq_scale = q["ecd"] * ev
    dq00 = q["dq00"] * dq_scale
    dq10 = q["dq10"] * dq_scale
    dq11 = q["dq11"] * dq_scale

    out = torch.zeros((rab.shape[0], 22), dtype=rab.dtype, device=rab.device)
    out[:, 0] = torch.sum(h0000 * dq00, dim=1)
    out[:, 4] = torch.sum(h0030 * dq10, dim=1)
    out[:, 10] = torch.sum(h0033 * dq11, dim=1)
    out[:, 11] = torch.sum(h0022 * dq11, dim=1)
    return out


def _sp1111_batch(p, q, rab, boys_table):
    rab1 = rab.view(-1, 1)
    rab3d = rab.view(-1, 1, 1)
    rab2 = rab1 * rab1
    rab3 = rab2 * rab1
    rab4 = rab2 * rab2

    eab = p["eab"].view(-1, 1, 9)
    eab2 = eab * eab
    ecd = q["ecd"].view(-1, 9, 1)

    g = 1.0 / (eab + ecd)
    x = g * rab3d.pow(2)

    sq1, sq2 = _sq_terms(p, q)
    dp11 = p["dp11"].view(-1, 1, 9)
    y = dp11 * sq2
    gy = g * y
    ggy = g * gy
    gggy = g * ggy

    # large = x > XLIM
    # xb = torch.where(large, torch.zeros_like(x), x)
    # f0, f1, f2, f3, f4 = boys_table(xb, 5)
    large = x > boys_table.xlim
    f0, f1, f2, f3, f4 = boys_from_table(x, boys_table, 5)

    f0n = f0 * y
    f1n = f1 * gy
    f2n = f2 * ggy
    f3n = f3 * gggy
    f4n = f4 * gggy * g

    f0a = dp11 * sq1 / rab3d
    gtx = 1.0 / (rab3d * rab3d)
    f1a = 0.5 * f0a * gtx
    f2a = 1.5 * f1a * gtx
    f3a = 2.5 * f2a * gtx
    f4a = 3.5 * f3a * gtx

    ff0 = torch.where(large, f0a, f0n)
    ff1 = torch.where(large, f1a, f1n)
    ff2 = torch.where(large, f2a, f2n)
    ff3 = torch.where(large, f3a, f3n)
    ff4 = torch.where(large, f4a, f4n)

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

    e = q["ecd"].view(-1, 9)
    he = 0.5 * e
    e2 = e * e
    he2 = 0.5 * e2

    h0000 = x0
    h0030 = -e * x1 * rab1
    h0022 = he * (x0 - e * x1)
    h0033 = h0022 + e2 * x2 * rab2
    h2020 = he * w1
    h2023 = -he2 * w2 * rab1
    h3000 = w1 * rab1
    h3030 = h2020 - e * w2 * rab2
    h3022 = h2023 + he * h3000
    h3033 = h3022 + h2023 + h2023 + e2 * w3 * rab3
    h2200 = 0.5 * (s0 - t1)
    h3300 = h2200 + t2 * rab2
    h2320 = he * t2 * rab1
    h2230 = he * (t2 - s1) * rab1
    h3330 = h2230 + e * (t2 * rab1 - t3 * rab3)
    h1212 = 0.25 * e2 * t2
    h2323 = he2 * (0.5 * t2 - t3 * rab2)
    hxxyy = 0.25 * (e * (s0 - t1) - e2 * (s1 - t2))
    h2222 = hxxyy + he2 * t2
    h1122 = hxxyy
    h3322 = hxxyy + he * t2 * rab2 - he2 * t3 * rab2
    h2233 = hxxyy + he2 * (s2 - t3) * rab2
    h3333 = hxxyy + he2 * (t2 + s2 * rab2) + e2 * (-3.0 * t3 * rab2 + t4 * rab4) + he * t2 * rab2

    dq_scale = q["ecd"] * ev
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


def om1_repgau_local_batch(rij, basis_i, basis_j, basis_tables):
    """
    Batched OM1 REPGAU local ``RI(22)`` tensor for all pair shell classes.
    """
    device = rij.device
    dtype = rij.dtype

    shell_i = basis_i["shell_type"]
    shell_j = basis_j["shell_type"]

    hh = (shell_i == 0) & (shell_j == 0)
    xh = (shell_i == 1) & (shell_j == 0)
    xx = (shell_i == 1) & (shell_j == 1)

    out = torch.zeros((rij.shape[0], 22), dtype=dtype, device=device)
    p = _om1_pair_basis_data_batch(basis_i)
    q = _om1_pair_q_data_batch(basis_j)
    boys_table = basis_tables["boys_integrals"]
    if hh.any():
        idx = hh.nonzero(as_tuple=False).squeeze(1)
        out[idx] = _sp0000_batch(_select_pair_data(p, idx), _select_pair_data(q, idx), rij[idx], boys_table)
    if xh.any():
        idx = xh.nonzero(as_tuple=False).squeeze(1)
        basis_i_xh = _select_pair_data(basis_i, idx)
        basis_j_xh = _select_pair_data(basis_j, idx)
        p_hx = _om1_pair_basis_data_batch(basis_j_xh)
        q_hx = _om1_pair_q_data_batch(basis_i_xh)
        tmp = _sp0011_batch(p_hx, q_hx, rij[idx], boys_table)
        out[idx, 0] = tmp[:, 0]
        out[idx, 1] = -tmp[:, 4]
        out[idx, 2] = tmp[:, 10]
        out[idx, 3] = tmp[:, 11]
    if xx.any():
        idx = xx.nonzero(as_tuple=False).squeeze(1)
        out[idx] = _sp1111_batch(_select_pair_data(p, idx), _select_pair_data(q, idx), rij[idx], boys_table)
    return out


def om1_local_pair_integrals_batch(ni, nj, rij, g_ss, tore, basis_i, basis_j, basis_tables):
    """
    Batched OM1 local two-center Gaussian pair data for all pairs.
    """
    ri = om1_repgau_local_batch(rij, basis_i, basis_j, basis_tables)
    scaled_ri, fko = om1_gscale(rij, ri, ni, nj, g_ss)
    rept = om1_cordef(ri, ni, nj, tore)
    # core_semi = om1_cordef(scaled_ri, ni, nj, tore)
    core_semi = fko[:, None, None] * rept
    return {"scaled_ri": scaled_ri, "fko": fko, "rept": rept, "core_semi": core_semi}


def _spgto2_local_batch(basis_payload, zeta_b, rab):
    """
    Batched version of `_spgto2_local`.
    """
    device = rab.device
    dtype = rab.dtype

    shell_type = basis_payload["shell_type"]
    exps = basis_payload["exponents"]
    cs = basis_payload["coeff_s"]
    cp = basis_payload["coeff_p"]
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


def om2_corpp2_local_batch(ni, nj, rij, om2_tables, basis_i, basis_j):
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
        shell_basis = _select_pair_data(basis_i, idx0)
        s1, s2 = _spgto2_local_batch(shell_basis, zscor[nb], rij0)
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
        shell_basis = _select_pair_data(basis_j, idx1)
        s1, s2 = _spgto2_local_batch(shell_basis, zscor[nb], rij1)
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
    method,
    ni,
    nj,
    rij,
    g_ss,
    tore,
    s_local,
    t_local,
    u_ss,
    u_pp,
    fval1,
    fval2,
    basis_tables,
    basis_i,
    basis_j,
    om2_tables=None,
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
    pair = om1_local_pair_integrals_batch(ni, nj, rij, g_ss, tore, basis_i, basis_j, basis_tables)
    cort = om1_corgau(ni, nj, rij, tore, basis_tables, basis_i, basis_j)
    pen = om1_penetration(cort, pair["rept"], pair["fko"])
    valpp_raw, valpp1 = om1_valpot(ni, nj, pair["core_semi"], s_local, t_local, u_ss, u_pp)
    if method == "OM3":
        valpp1 = torch.zeros_like(valpp1)
    valpp = om1_apply_valpot_scaling(ni, nj, valpp_raw, valpp1, fval1, fval2)
    if method == "OM1":
        corpp = om1_ppecp_local(ni, nj, rij, basis_tables, basis_i, basis_j) * pair["fko"].view(-1, 1, 1)
    elif method in {"OM2", "OM3"}:
        if om2_tables is None:
            raise ValueError(f"{method} pair corrections require om2_tables")
        corpp = om2_corpp2_local_batch(ni, nj, rij, om2_tables, basis_i, basis_j)
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
    basis_tables,
    basis_i,
    basis_j,
    om2_tables=None,
):
    pair = om1_local_pair_integrals_batch(ni, nj, rij, g_ss, tore, basis_i, basis_j, basis_tables)
    cort = om1_corgau(ni, nj, rij, tore, basis_tables, basis_i, basis_j)
    pen = om1_penetration(cort, pair["rept"], pair["fko"])
    valpp_raw, valpp1 = om1_valpot(ni, nj, pair["core_semi"], s_local, t_local, u_ss, u_pp)
    if method == "OM3":
        valpp1 = torch.zeros_like(valpp1)
    valpp = om1_apply_valpot_scaling(ni, nj, valpp_raw, valpp1, fval1, fval2)

    if method == "OM1":
        corpp = om1_ppecp_local(ni, nj, rij, basis_tables, basis_i, basis_j) * pair["fko"].view(-1, 1, 1)
    elif method in {"OM2", "OM3"}:
        if om2_tables is None:
            raise ValueError(f"{method} batched pair corrections require om2_tables")
        corpp = om2_corpp2_local_batch(ni, nj, rij, om2_tables, basis_i, basis_j)
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
    basis_tables,
    basis_i,
    basis_j,
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
        basis_tables,
        basis_i,
        basis_j,
        om2_tables=om2_tables,
    )

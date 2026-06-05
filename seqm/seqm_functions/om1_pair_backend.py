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


def _boys_series_coeff(n, x):
    return 1.0 / (2 * n + 1) - x / (2 * n + 3) + 0.5 * x * x / (2 * n + 5)


def _boys_0_to_4(x):
    dtype = x.dtype
    device = x.device
    small = x < 1.0e-8
    xsafe = torch.where(small, torch.ones_like(x), x)
    sqrtx = torch.sqrt(xsafe)
    f0 = 0.5 * torch.sqrt(torch.tensor(math.pi, dtype=dtype, device=device) / xsafe) * torch.erf(sqrtx)
    expx = torch.exp(-xsafe)
    f1 = (f0 - expx) / (2.0 * xsafe)
    f2 = (3.0 * f1 - expx) / (2.0 * xsafe)
    f3 = (5.0 * f2 - expx) / (2.0 * xsafe)
    f4 = (7.0 * f3 - expx) / (2.0 * xsafe)
    if small.any():
        xx = x[small]
        f0 = f0.clone()
        f1 = f1.clone()
        f2 = f2.clone()
        f3 = f3.clone()
        f4 = f4.clone()
        f0[small] = _boys_series_coeff(0, xx)
        f1[small] = _boys_series_coeff(1, xx)
        f2[small] = _boys_series_coeff(2, xx)
        f3[small] = _boys_series_coeff(3, xx)
        f4[small] = _boys_series_coeff(4, xx)
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
    f0, _, _, _, _ = _boys_0_to_4(x)
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
    f0, f1, f2, _, _ = _boys_0_to_4(x)
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
    f0, f1, f2, f3, f4 = _boys_0_to_4(x)
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


def om1_local_pair_corrections(ni, nj, rij, zeta_s, g_ss, tore, s_local, t_local, u_ss, u_pp, fval1, fval2):
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
    pair = om1_local_pair_integrals(ni, nj, rij, zeta_s, g_ss, tore)
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
    valpp = om1_apply_valpot_scaling(ni_t, nj_t, valpp_raw, valpp1, fval1, fval2)[0]
    corpp = om1_ppecp_local(int(ni), int(nj), float(rij), zeta_s) * pair["fko"]
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


def om1_pair_hcore_terms(ni, nj, xij, rij, zeta_s, g_ss, tore, s_local, t_local, u_ss, u_pp, fval1, fval2):
    pair = om1_local_pair_corrections(
        ni, nj, rij, zeta_s, g_ss, tore, s_local, t_local, u_ss, u_pp, fval1, fval2
    )
    return {
        **pair,
        "w": _om1_rotate_w(pair["scaled_ri"], xij),
        "e1b": _om1_rotate_core_column(pair["core"][:, 0], xij),
        "e2a": _om1_rotate_core_column(pair["core"][:, 1], xij),
    }

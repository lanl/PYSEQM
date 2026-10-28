import math

import torch

from .constants import ev
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

_XX3498 = 34.9868366552497

MAXFMT = 400
PT05 = 0.05
PT184 = 0.184
XMAX = 19.9  # (MAXFMT - 2) * PT05   # 19.9
XLIM = 20.0
PT7853 = 0.785398163397448  # PI / 4, as in GTPREP


def _sq_terms(p, q):
    gab = p["gab"].view(-1, 1, 9)
    gcd = q["gcd"].view(-1, 9, 1)
    sq1 = torch.sqrt(PT7853 / (gab * gcd))
    sq2 = 1.0 / torch.sqrt(gab + gcd)
    return sq1, sq2


# ---------------------------------------------------------------------
# Balanced fast-path helpers / caches
# ---------------------------------------------------------------------

_ROT_W_INDEX_CACHE = {}
_CORE_GTO_CONST_CACHE = {}


def _cache_device_key(device):
    return (device.type, -1 if device.index is None else device.index)


def _cpu_idx(mask_cpu):
    return mask_cpu.nonzero(as_tuple=False).squeeze(1)


def _select_pair_data(data, idx):
    return {k: v[idx] for k, v in data.items()}


def _make_repgau_groups(basis_i, basis_j):
    """
    Build pair-class indices once.

    This does one CPU transfer at cache-build time so the hot path avoids
    CUDA synchronizations from things like if mask.any().
    """
    device = basis_i["shell_type"].device

    shell_i = basis_i["shell_type"].detach().cpu()
    shell_j = basis_j["shell_type"].detach().cpu()

    hh = _cpu_idx((shell_i == 0) & (shell_j == 0)).to(device)
    xh = _cpu_idx((shell_i == 1) & (shell_j == 0)).to(device)
    xx = _cpu_idx((shell_i == 1) & (shell_j == 1)).to(device)

    return {"hh": hh, "xh": xh, "xx": xx}


def _make_corpp_groups(ni, nj):
    """
    Build OM2/OM3 CORPP2 groups once.
    """
    device = ni.device

    ni_cpu = ni.detach().cpu()
    nj_cpu = nj.detach().cpu()

    idx0_cpu = _cpu_idx(nj_cpu > 2)
    idx1_cpu = _cpu_idx(ni_cpu > 2)

    if idx0_cpu.numel() != 0:
        idx0h_pos_cpu = _cpu_idx(ni_cpu[idx0_cpu] > 2)
    else:
        idx0h_pos_cpu = torch.empty((0,), dtype=torch.long)

    if idx1_cpu.numel() != 0:
        idx1h_pos_cpu = _cpu_idx(nj_cpu[idx1_cpu] > 2)
    else:
        idx1h_pos_cpu = torch.empty((0,), dtype=torch.long)

    return {
        "idx0": idx0_cpu.to(device),
        "idx0h_pos": idx0h_pos_cpu.to(device),
        "idx1": idx1_cpu.to(device),
        "idx1h_pos": idx1h_pos_cpu.to(device),
    }


def make_omx_fast_cache(basis_i, basis_j, ni=None, nj=None):
    """
    Balanced cache.

    Stores:
      - forward REPGAU p/q data for all pairs,
      - reversed xh data only for xh pairs,
      - optional CORPP grouping.

    Rebuild when the pair list, basis, ni/nj, device, or dtype changes.
    """
    groups = _make_repgau_groups(basis_i, basis_j)
    xh = groups["xh"]

    repgau_cache = {
        "p_ij": _om1_pair_basis_data_batch(basis_i),
        "q_ij": _om1_pair_q_data_batch(basis_j),
        "groups": groups,
    }

    # Only cache reversed orientation for the xh subset.
    if xh.numel() != 0:
        basis_i_xh = _select_pair_data(basis_i, xh)
        basis_j_xh = _select_pair_data(basis_j, xh)

        repgau_cache["p_ji_xh"] = _om1_pair_basis_data_batch(basis_j_xh)
        repgau_cache["q_ji_xh"] = _om1_pair_q_data_batch(basis_i_xh)
    else:
        repgau_cache["p_ji_xh"] = None
        repgau_cache["q_ji_xh"] = None

    cache = {"repgau": repgau_cache}

    if ni is not None and nj is not None:
        cache["corpp"] = _make_corpp_groups(ni, nj)

    return cache


# ---------------------------------------------------------------------
# Pair basis data
# ---------------------------------------------------------------------


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

    base_dp00 = pref * csij
    raw_dp11 = pref * cpij

    heavy = (shell_type == 1).view(-1, 1)
    safe_den = torch.where(heavy, raw_dp11, torch.ones_like(raw_dp11))

    dp00_heavy = base_dp00 / safe_den
    dp10_heavy = (pref * cpis) / safe_den

    dp00 = torch.where(heavy, dp00_heavy, base_dp00)
    dp10 = torch.where(heavy, dp10_heavy, torch.zeros_like(base_dp00))
    dp11 = torch.where(heavy, raw_dp11, torch.zeros_like(base_dp00))

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


# ---------------------------------------------------------------------
# Boys table
# ---------------------------------------------------------------------


def boys_from_table(x, boys_table, m_count):
    """
        X < XMAX          table interpolation
        XMAX <= X <= XLIM direct FMTGEN
        X > XLIM          unused here; SP code handles asymptotic separately
    Returns F0...F_{m_count-1}.
    """
    tab_vals = torch.stack(boys_table(x.clamp(max=boys_table.xmax), m_count), dim=-1)
    direct_vals = _boys_direct_fixed(x.clamp(min=boys_table.xmax, max=boys_table.xlim), m_count)
    use_direct = ((x >= boys_table.xmax) & (x <= boys_table.xlim)).unsqueeze(-1)
    vals = torch.where(use_direct, direct_vals, tab_vals)
    return tuple(vals[..., m] for m in range(m_count))


def _boys_direct_fixed(t, m_count):
    sqrt_t = torch.sqrt(t)
    texp = torch.exp(-t)
    cur = 0.5 * math.sqrt(math.pi) * torch.erf(sqrt_t) / sqrt_t
    vals = [cur]
    for m in range(1, m_count):
        cur = (float(2 * m - 1) * cur - texp) / (2.0 * t)
        vals.append(cur)
    return torch.stack(vals, dim=-1)


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


# ---------------------------------------------------------------------
# Balanced low-memory SP1111
# ---------------------------------------------------------------------


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
    x = g * (rab3d * rab3d)

    sq1, sq2 = _sq_terms(p, q)

    dp11 = p["dp11"].view(-1, 1, 9)
    y = dp11 * sq2

    gy = g * y
    ggy = g * gy
    gggy = g * ggy

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


# ---------------------------------------------------------------------
# REPGAU dispatcher
# ---------------------------------------------------------------------


def om1_repgau_local_batch(rij, basis_i, basis_j, basis_tables, pair_cache=None):
    """
    Batched OM1 REPGAU local RI(22) tensor.

    For speed, pass:
        pair_cache = fast_cache["repgau"]
    """
    device = rij.device
    dtype = rij.dtype

    if pair_cache is None:
        pair_cache = make_omx_fast_cache(basis_i, basis_j)["repgau"]

    p_ij = pair_cache["p_ij"]
    q_ij = pair_cache["q_ij"]
    groups = pair_cache["groups"]

    boys_table = basis_tables["boys_integrals"]
    out = torch.zeros((rij.shape[0], 22), dtype=dtype, device=device)

    idx = groups["hh"]
    if idx.numel() != 0:
        out[idx] = _sp0000_batch(
            _select_pair_data(p_ij, idx), _select_pair_data(q_ij, idx), rij[idx], boys_table
        )

    idx = groups["xh"]
    if idx.numel() != 0:
        tmp = _sp0011_batch(pair_cache["p_ji_xh"], pair_cache["q_ji_xh"], rij[idx], boys_table)

        out[idx, 0] = tmp[:, 0]
        out[idx, 1] = -tmp[:, 4]
        out[idx, 2] = tmp[:, 10]
        out[idx, 3] = tmp[:, 11]

    idx = groups["xx"]
    if idx.numel() != 0:
        out[idx] = _sp1111_batch(
            _select_pair_data(p_ij, idx), _select_pair_data(q_ij, idx), rij[idx], boys_table
        )

    return out


def om1_local_pair_integrals_batch(
    ni, nj, rij, pi, pj, tore, basis_i, basis_j, basis_tables, fast_cache=None
):
    pair_cache = None if fast_cache is None else fast_cache.get("repgau")

    ri = om1_repgau_local_batch(rij, basis_i, basis_j, basis_tables, pair_cache=pair_cache)

    scaled_ri, fko = om1_gscale(rij, ri, ni, nj, pi["g_ss"], pj["g_ss"])
    rept = om1_cordef(ri, ni, nj, tore)
    core_semi = fko[:, None, None] * rept

    return {"scaled_ri": scaled_ri, "fko": fko, "rept": rept, "core_semi": core_semi}


# ---------------------------------------------------------------------
# Fast rotate_w
# ---------------------------------------------------------------------

_PAIR_I_LIST = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
_PAIR_J_LIST = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]

_LOCAL_ROWS_LIST = [
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    5,
    5,
    5,
    6,
    6,
    7,
    7,
    8,
    9,
    9,
    9,
    9,
    9,
]

_LOCAL_COLS_LIST = [
    0,
    1,
    2,
    5,
    9,
    0,
    1,
    2,
    5,
    9,
    0,
    1,
    2,
    5,
    9,
    3,
    4,
    3,
    4,
    0,
    1,
    2,
    5,
    9,
    6,
    7,
    6,
    7,
    8,
    0,
    1,
    2,
    5,
    9,
]

_LOCAL_RIDS_LIST = [
    0,
    4,
    10,
    11,
    11,
    1,
    5,
    12,
    13,
    13,
    2,
    7,
    15,
    17,
    17,
    6,
    14,
    9,
    19,
    3,
    8,
    16,
    18,
    20,
    6,
    14,
    9,
    19,
    21,
    3,
    8,
    16,
    20,
    18,
]


def _rotate_w_indices(device):
    key = _cache_device_key(device)
    cached = _ROT_W_INDEX_CACHE.get(key)

    if cached is not None:
        return cached

    pair_i = torch.tensor(_PAIR_I_LIST, dtype=torch.long, device=device)
    pair_j = torch.tensor(_PAIR_J_LIST, dtype=torch.long, device=device)
    rows = torch.tensor(_LOCAL_ROWS_LIST, dtype=torch.long, device=device)
    cols = torch.tensor(_LOCAL_COLS_LIST, dtype=torch.long, device=device)
    rids = torch.tensor(_LOCAL_RIDS_LIST, dtype=torch.long, device=device)

    cached = (pair_i, pair_j, rows, cols, rids)
    _ROT_W_INDEX_CACHE[key] = cached

    return cached


def _om1_rotate_w_batch(ri, rot):
    bsz = ri.shape[0]
    dtype = ri.dtype
    device = ri.device

    pair_i, pair_j, rows, cols, rids = _rotate_w_indices(device)

    local = ri.new_zeros((bsz, 10, 10))
    local[:, rows, cols] = ri[:, rids]

    # u[global_ao, local_ao].
    # rot rows are local axes in global coordinates, so transpose here.
    u = ri.new_zeros((bsz, 4, 4))
    u[:, 0, 0] = 1.0
    u[:, 1:, 1:] = rot.transpose(1, 2)

    g0 = pair_i[:, None]
    g1 = pair_j[:, None]
    l0 = pair_i[None, :]
    l1 = pair_j[None, :]

    T = u[:, g0, l0] * u[:, g1, l1]

    local_offdiag = (pair_i != pair_j).to(dtype).view(1, 1, 10)
    T = T + local_offdiag * (u[:, g0, l1] * u[:, g1, l0])

    return T @ local @ T.transpose(1, 2)


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


# ---------------------------------------------------------------------
# Core-GTO / CORPP2
# ---------------------------------------------------------------------


def _core_gto_constants(dtype, device):
    key = (str(dtype), _cache_device_key(device))
    cached = _CORE_GTO_CONST_CACHE.get(key)

    if cached is not None:
        return cached

    core_exps = torch.tensor([2.227660584, 0.4057711562, 0.1098175104], dtype=dtype, device=device)
    core_coeffs = torch.tensor([0.1543289673, 0.5353281423, 0.4446345422], dtype=dtype, device=device)

    cached = (core_exps, core_coeffs)
    _CORE_GTO_CONST_CACHE[key] = cached

    return cached


def _spgto2_local_batch(basis_payload, zeta_b, rab):
    """
    Batched version of _spgto2_local.
    """
    device = rab.device
    dtype = rab.dtype

    shell_type = basis_payload["shell_type"]
    exps = basis_payload["exponents"]
    cs = basis_payload["coeff_s"]
    cp = basis_payload["coeff_p"]

    shell_heavy = shell_type > 0

    core_exps, core_coeffs = _core_gto_constants(dtype, device)
    ec = core_exps.view(1, 3) * zeta_b.view(-1, 1).square()
    cc = core_coeffs.view(1, 3) * (2.0 * ec / math.pi).pow(0.75)

    r = rab.view(-1, 1, 1)
    rab2 = r * r

    g = exps.unsqueeze(2) + ec.unsqueeze(1)
    xqq = exps.unsqueeze(2) * ec.unsqueeze(1) * rab2 / g

    s00 = (math.pi / g).pow(1.5) * torch.exp(-xqq)

    s1 = torch.sum(cs.unsqueeze(2) * cc.unsqueeze(1) * s00, dim=(1, 2))

    s2_raw = torch.sum(cp.unsqueeze(2) * cc.unsqueeze(1) * (ec.unsqueeze(1) * r * s00 / g), dim=(1, 2))

    s2 = torch.where(shell_heavy, s2_raw, torch.zeros_like(s2_raw))

    return s1, s2


def _corpp2_column(corpp, col, idx, hpos, rij, shell_basis, pa, pb):
    if idx.numel() == 0:
        return
    r = rij[idx]
    s1, s2 = _spgto2_local_batch(shell_basis, pb["zscor"][idx], r)
    bscor, ascor, fscor = pb["bscor"][idx], pb["ascor"][idx], pb["fscor"][idx]
    t1 = 0.5 * (pa["beta_s"][idx] + bscor) * torch.sqrt(r) * torch.exp(-(pa["alpha_s"][idx] + ascor) * r * r)
    corpp[idx, 0, col] = -2.0 * s1 * t1 - s1 * s1 * fscor
    if hpos.numel() == 0:
        return
    h, s1h, s2h, t1h, rh, fh = idx[hpos], s1[hpos], s2[hpos], t1[hpos], r[hpos], fscor[hpos]
    t3 = (
        0.5
        * (pa["beta_p"][idx][hpos] + bscor[hpos])
        * torch.sqrt(rh)
        * torch.exp(-(pa["alpha_p"][idx][hpos] + ascor[hpos]) * rh * rh)
    )
    c1 = -(s1h * t3 + t1h * s2h) - s1h * s2h * fh
    corpp[h, 1, col] = c1 if col == 0 else -c1
    corpp[h, 2, col] = -(s2h * t3 + t3 * s2h) - s2h * s2h * fh


def om2_corpp2_local_batch(ni, nj, rij, pi, pj, basis_i, basis_j, fast_cache=None):
    corpp = torch.zeros((ni.shape[0], 4, 2), dtype=rij.dtype, device=rij.device)
    groups = (fast_cache or {}).get("corpp") or _make_corpp_groups(ni, nj)
    _corpp2_column(
        corpp, 0, groups["idx0"], groups["idx0h_pos"], rij, _select_pair_data(basis_i, groups["idx0"]), pi, pj
    )
    _corpp2_column(
        corpp, 1, groups["idx1"], groups["idx1h_pos"], rij, _select_pair_data(basis_j, groups["idx1"]), pj, pi
    )
    return corpp


# ---------------------------------------------------------------------
# Public callers
# ---------------------------------------------------------------------


def omx_pair_hcore_terms(
    method,
    ni,
    nj,
    rij,
    pi,
    pj,
    tore,
    s_local,
    t_local,
    rot,
    rot_t,
    basis_tables,
    basis_i,
    basis_j,
    fast_cache=None,
):
    if fast_cache is None:
        fast_cache = make_omx_fast_cache(basis_i, basis_j, ni=ni, nj=nj)

    pair = om1_local_pair_integrals_batch(
        ni, nj, rij, pi, pj, tore, basis_i, basis_j, basis_tables, fast_cache=fast_cache
    )

    cort = om1_corgau(ni, nj, rij, tore, basis_tables, basis_i, basis_j)
    pen = om1_penetration(cort, pair["rept"], pair["fko"])

    need_valpp1 = method != "OM3"

    valpp_raw, valpp1 = om1_valpot(ni, nj, pair["core_semi"], s_local, t_local, pi, pj, need_valpp1)

    if not need_valpp1:
        valpp1 = torch.zeros_like(valpp_raw)

    valpp = om1_apply_valpot_scaling(ni, nj, valpp_raw, None if not need_valpp1 else valpp1, pi, pj)

    if method == "OM1":
        corpp = om1_ppecp_local(ni, nj, rij, basis_tables, basis_i, basis_j) * pair["fko"].view(-1, 1, 1)

    elif method in {"OM2", "OM3"}:
        corpp = om2_corpp2_local_batch(ni, nj, rij, pi, pj, basis_i, basis_j, fast_cache=fast_cache)

    else:
        raise ValueError(f"Unsupported method: {method}")

    core = om1_assemble_core(pair["core_semi"], pen, corpp=corpp, valpp=valpp)

    w = _om1_rotate_w_batch(pair["scaled_ri"], rot)
    e1b = _om1_rotate_core_columns_batch(core[:, :, 0], rot_t)
    e2a = _om1_rotate_core_columns_batch(core[:, :, 1], rot_t)

    out = {"w": w, "e1b": e1b, "e2a": e2a, "fko": pair["fko"]}

    if method == "OM2":
        out["core_semi"] = pair["core_semi"]

    return out

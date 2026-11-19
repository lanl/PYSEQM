"""
Pair (two-center) Hcore terms of OM1/OM2/OM3 in the local diatomic frame, rotated to the molecular frame.

Pipeline per atom pair (Fortran names in parentheses):
    REPGAU analytic two-electron integrals RI(22)   -> om1_repgau_local_batch
    GSCALE Klopman-Ohno scaling                     -> om1_gscale
    CORDEF/CORGAU core attractions, penetration     -> om1_core_corrections
    VALPOT orthogonalisation terms                  -> om1_core_corrections
    PPECP (OM1) / CORPP2 (OM2, OM3) core potentials -> om1_ppecp / om2_corpp2_local_batch
    ROTATE to the molecular frame                   -> _om1_rotate_w_batch / _om1_rotate_core_columns_batch
"""

import math

import torch

from .constants import ev
from .om1_core_corrections import (
    om1_apply_valpot_scaling,
    om1_cordef,
    om1_corgau,
    om1_gscale,
    om1_penetration,
    om1_valpot,
)
from .om1_ppecp import om1_ppecp_local

_XX3498 = 34.9868366552497
PT7853 = 0.785398163397448  # PI / 4, as in GTPREP


# ---------------------------------------------------------------------
# Per-atom Gaussian data
# ---------------------------------------------------------------------


def atom_gaussian_data(basis):
    """
    Primitive-pair data of every atom's 3x3 contracted Gaussian product (flattened to 9).

    The same atom data serves as bra ("p" side: g, e, dp00, dp10, dp11) and as ket ("q" side:
    g, e, dq00, dq10, dq11) of the REPGAU integrals, so it is built once per atom and gathered by
    pair index, not rebuilt per pair.
    """
    exponents = basis["exponents"]
    n = exponents.shape[0]
    cs, cp = basis["coeff_s"], basis["coeff_p"]

    def outer(a, b):
        return (a.unsqueeze(2) * b.unsqueeze(1)).reshape(n, 9)

    g = (exponents.unsqueeze(2) + exponents.unsqueeze(1)).reshape(n, 9)
    e = 1.0 / g
    css, cps, cpp = outer(cs, cs), outer(cp, cs), outer(cp, cp)

    pref = _XX3498 * e
    heavy = (basis["shell_type"] == 1).view(-1, 1)
    raw_dp11 = pref * cpp
    safe_den = torch.where(heavy, raw_dp11, torch.ones_like(raw_dp11))
    zeros = torch.zeros_like(css)

    return {
        "shell_type": basis["shell_type"],
        "g": g,
        "e": e,
        "dp00": torch.where(heavy, pref * css / safe_den, pref * css),
        "dp10": torch.where(heavy, pref * cps / safe_den, zeros),
        "dp11": torch.where(heavy, raw_dp11, zeros),
        "dq00": css,
        "dq10": cps,
        "dq11": cpp,
    }


_BRA_KEYS = ("g", "e", "dp00", "dp10", "dp11")
_KET_KEYS = ("g", "e", "dq00", "dq10", "dq11")


def _gather(atoms, idx, keys):
    return {k: atoms[k][idx] for k in keys}


def _sq_terms(p, q):
    g_bra = p["g"].view(-1, 1, 9)
    g_ket = q["g"].view(-1, 9, 1)
    return torch.sqrt(PT7853 / (g_bra * g_ket)), 1.0 / torch.sqrt(g_bra + g_ket)


def _boys_terms(boys_table, x, g, y, asym0, gtx, m_count, far=False):
    """
    Return [ff_0 .. ff_{m-1}] with ff_k = F_k(x) * y * g^k (table), or the large-x asymptote
    asym0 * prod_{j<k} (j + 1/2) gtx beyond XLIM.

    ``far=True`` asserts that every element is beyond XLIM (``x`` is then not needed and may be None),
    which skips the table lookup altogether.
    """
    if far:
        terms, asym = [], asym0
        for k in range(m_count):
            terms.append(asym)
            asym = asym * ((k + 0.5) * gtx)
        return terms

    f = boys_table(x, m_count)
    large = x > boys_table.xlim
    terms = []
    scale, asym = y, asym0
    for k in range(m_count):
        terms.append(torch.where(large, asym, f[k] * scale))
        scale = scale * g
        asym = asym * ((k + 0.5) * gtx)
    return terms


def _contract_ket(h_list, dq):
    """sum_q h[q] * dq[q] for each integral type; h entries and dq are (npairs, 9)."""
    return torch.einsum("nhq,nq->nh", torch.stack(h_list, dim=1), dq)


# ---------------------------------------------------------------------
# REPGAU: s-s, heavy(sp)-H and sp-sp integral classes
# ---------------------------------------------------------------------


def _sp0000(p, q, rab, boys_table, far=False):
    """H-H: only RI[0]. Switches to the asymptote above XMAX, like the Fortran code."""
    r = rab.view(-1, 1, 1)
    sq1, sq2 = _sq_terms(p, q)
    dp00 = p["dp00"].view(-1, 1, 9)

    if far:
        h0000 = torch.sum(dp00 * sq1 / r, dim=2)
    else:
        x = r * r / (p["e"].view(-1, 1, 9) + q["e"].view(-1, 9, 1))
        f0 = boys_table(x, 1)[0]
        h0000 = torch.sum(torch.where(x > boys_table.xmax, dp00 * sq1 / r, f0 * dp00 * sq2), dim=2)
    return torch.sum(h0000 * (q["dq00"] * q["e"] * ev), dim=1, keepdim=True)


def _sp0011(p, q, rab, boys_table, far=False):
    """
    Heavy-H integrals, evaluated with the H atom as bra and the heavy atom as ket.
    Returns RI[0:4] = [(ss|ss), (sp|ss), (pp|ss) sigma, (pp|ss) pi] of the heavy-H orientation.
    """
    rab1 = rab.view(-1, 1)
    r = rab.view(-1, 1, 1)
    eab = p["e"].view(-1, 1, 9)
    ecd = q["e"].view(-1, 9, 1)

    x = None if far else r * r / (eab + ecd)
    g = ecd / (eab + ecd)
    sq1, sq2 = _sq_terms(p, q)
    dp00 = p["dp00"].view(-1, 1, 9)
    ff0, ff1, ff2 = _boys_terms(boys_table, x, g, dp00 * sq2, dp00 * sq1 / r, ecd / (r * r), 3, far)

    h0000 = ff0.sum(dim=2)
    h0001 = ff1.sum(dim=2)
    h0030 = -h0001 * rab1
    h0022 = 0.5 * q["e"] * (h0000 - h0001)
    h0033 = ff2.sum(dim=2) * rab1.pow(2) + h0022

    dq_scale = q["e"] * ev
    return torch.cat(
        [
            _contract_ket([h0000], q["dq00"] * dq_scale),
            -_contract_ket([h0030], q["dq10"] * dq_scale),
            _contract_ket([h0033, h0022], q["dq11"] * dq_scale),
        ],
        dim=1,
    )


def _sp1111(p, q, rab, boys_table, far=False):
    """Heavy-heavy: all 22 RI integrals."""
    rab1 = rab.view(-1, 1)
    r = rab.view(-1, 1, 1)
    rab2 = rab1 * rab1
    rab3 = rab2 * rab1
    rab4 = rab2 * rab2

    eab = p["e"].view(-1, 1, 9)
    g = 1.0 / (eab + q["e"].view(-1, 9, 1))
    x = None if far else g * (r * r)
    sq1, sq2 = _sq_terms(p, q)
    dp11 = p["dp11"].view(-1, 1, 9)
    ff = _boys_terms(boys_table, x, g, dp11 * sq2, dp11 * sq1 / r, 1.0 / (r * r), 5, far)

    # Reduce over the bra primitives with one batched matmul per Boys order. Columns of the weights:
    # 0: dp00, 1: e * dp10, 2: e, 3: e^2; order k only needs the columns that multiply F_k.
    e_bra = p["e"]
    weights = torch.stack([p["dp00"], e_bra * p["dp10"], e_bra, e_bra * e_bra], dim=2)
    columns = ([0, 2], [0, 1, 2, 3], [0, 1, 2, 3], [1, 3], [3])
    red = [torch.bmm(ff[k], weights[:, :, columns[k]]) for k in range(5)]
    x0, s0 = red[0].unbind(-1)
    x1, w1, s1, t1 = red[1].unbind(-1)
    x2, w2, s2, t2 = red[2].unbind(-1)
    w3, t3 = red[3].unbind(-1)
    t4 = red[4][..., 0]

    e = q["e"]
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

    dq_scale = q["e"] * ev
    # RI packing: [0:4] <- dq00, [4:10] <- dq10, [10:22] <- dq11.
    return torch.cat(
        [
            _contract_ket([h0000, h3000, h3300, h2200], q["dq00"] * dq_scale),
            _contract_ket([h0030, h3030, h2020, h3330, h2230, h2320], q["dq10"] * dq_scale),
            _contract_ket(
                [h0033, h0022, h3033, h3022, h2023, h3333, h2233, h3322, h2222, h2323, h1122, h1212],
                q["dq11"] * dq_scale,
            ),
        ],
        dim=1,
    )


def om1_repgau_local_batch(rij, idxi, idxj, atoms, boys_table):
    """
    Batched OM1 REPGAU local RI(22) tensor, shape (npairs, 22).

    ``atoms`` is ``atom_gaussian_data(...)``; ``idxi``/``idxj`` are the pair atom indices. Pairs are
    grouped by shell class (H-H, heavy-H, heavy-heavy) so each class only pays for the integrals it has,
    and each class is split into near pairs (Boys table) and far pairs. A pair is far when even its
    smallest primitive argument x_min = r^2 / max(e_bra + e_ket) exceeds the asymptotic threshold, so the
    table would never be used for it; that test is exact, not a distance heuristic.
    """
    shell_i = atoms["shell_type"][idxi]
    shell_j = atoms["shell_type"][idxj]
    e_max = atoms["e"].amax(dim=1)

    def x_min_00(r, e_sum):
        return r * r / e_sum

    def x_min_11(r, e_sum):
        return (1.0 / e_sum) * (r * r)

    classes = (
        # mask, kernel, columns, H is bra (swap), far threshold, smallest x of a pair
        ((shell_i == 0) & (shell_j == 0), _sp0000, 1, False, boys_table.xmax, x_min_00),
        ((shell_i == 1) & (shell_j == 0), _sp0011, 4, True, boys_table.xlim, x_min_00),
        ((shell_i == 1) & (shell_j == 1), _sp1111, 22, False, boys_table.xlim, x_min_11),
    )

    out = rij.new_zeros((rij.shape[0], 22))
    for mask, kernel, ncols, swap, threshold, x_min in classes:
        sel = mask.nonzero().squeeze(1)
        if sel.numel() == 0:
            continue
        bra, ket = (idxj[sel], idxi[sel]) if swap else (idxi[sel], idxj[sel])
        is_far = x_min(rij[sel], e_max[bra] + e_max[ket]) > threshold
        for far in (False, True):
            part = (is_far == far).nonzero().squeeze(1)
            if part.numel() == 0:
                continue
            out[sel[part], :ncols] = kernel(
                _gather(atoms, bra[part], _BRA_KEYS),
                _gather(atoms, ket[part], _KET_KEYS),
                rij[sel[part]],
                boys_table,
                far,
            )
    return out


# ---------------------------------------------------------------------
# Rotation to the molecular frame
# ---------------------------------------------------------------------

# The 10 unique AO products (ab) of one atom, as (a, b) with a >= b over (s, px, py, pz), grouped by tensor
# rank: [ss] [px s, py s, pz s] [px px, py px, py py, pz px, pz py, pz pz]. Rotations preserve the rank, so
# the 10x10 rotation of the two-center integrals is block diagonal: T = diag(1, T3, T6).
_PAIR_ORDER = [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)]
_PAIR_INDEX = {pair: k for k, pair in enumerate(_PAIR_ORDER)}
# Row/column order of W as the rest of the code (Fock build, gradients) expects it.
_STANDARD_ORDER = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3)]
_PP_FIRST = [a - 1 for a, _ in _PAIR_ORDER[4:]]
_PP_SECOND = [b - 1 for _, b in _PAIR_ORDER[4:]]
_BLOCKS = (slice(0, 1), slice(1, 4), slice(4, 10))

# Local-frame two-center integral matrix: (row pair, column pair, index into RI(22)), in _PAIR_ORDER.
# Row = product on atom i, column = product on atom j; both sides use the x axis along the bond.
# fmt: off
_LOCAL_ENTRIES = [
    ((0, 0), (0, 0), 0), ((0, 0), (1, 0), 4), ((0, 0), (1, 1), 10), ((0, 0), (2, 2), 11), ((0, 0), (3, 3), 11),
    ((1, 0), (0, 0), 1), ((1, 0), (1, 0), 5), ((1, 0), (1, 1), 12), ((1, 0), (2, 2), 13), ((1, 0), (3, 3), 13),
    ((1, 1), (0, 0), 2), ((1, 1), (1, 0), 7), ((1, 1), (1, 1), 15), ((1, 1), (2, 2), 17), ((1, 1), (3, 3), 17),
    ((2, 0), (2, 0), 6), ((2, 0), (2, 1), 14), ((2, 1), (2, 0), 9), ((2, 1), (2, 1), 19),
    ((2, 2), (0, 0), 3), ((2, 2), (1, 0), 8), ((2, 2), (1, 1), 16), ((2, 2), (2, 2), 18), ((2, 2), (3, 3), 20),
    ((3, 0), (3, 0), 6), ((3, 0), (3, 1), 14), ((3, 1), (3, 0), 9), ((3, 1), (3, 1), 19),
    ((3, 2), (3, 2), 21),
    ((3, 3), (0, 0), 3), ((3, 3), (1, 0), 8), ((3, 3), (1, 1), 16), ((3, 3), (2, 2), 20), ((3, 3), (3, 3), 18),
]
# fmt: on

_ROT_W_INDEX_CACHE = {}


def _rotate_w_indices(device):
    key = (device.type, -1 if device.index is None else device.index)
    cached = _ROT_W_INDEX_CACHE.get(key)
    if cached is None:
        rows = [_PAIR_INDEX[r] for r, _, _ in _LOCAL_ENTRIES]
        cols = [_PAIR_INDEX[c] for _, c, _ in _LOCAL_ENTRIES]
        rids = [i for _, _, i in _LOCAL_ENTRIES]
        first, second = (torch.tensor(v, dtype=torch.long, device=device) for v in (_PP_FIRST, _PP_SECOND))
        rows, cols, rids = (torch.tensor(v, dtype=torch.long, device=device) for v in (rows, cols, rids))
        to_standard = torch.tensor([_PAIR_INDEX[p] for p in _STANDARD_ORDER], dtype=torch.long, device=device)
        cached = _ROT_W_INDEX_CACHE[key] = (first, second, rows, cols, rids, to_standard)
    return cached


def _pp_rotation(rot_t, first, second):
    """6x6 rotation of the p-p products from the 3x3 p rotation (rot_t[g, l]: global p_g in local p_l)."""
    direct = rot_t[:, first[:, None], first[None, :]] * rot_t[:, second[:, None], second[None, :]]
    swapped = rot_t[:, first[:, None], second[None, :]] * rot_t[:, second[:, None], first[None, :]]
    return direct + (first != second).to(rot_t.dtype).view(1, 1, 6) * swapped


def _om1_rotate_w_batch(ri, rot):
    """
    Rotate the local two-center integrals RI(22) to the molecular frame: W = T W_local T^T with
    T = diag(1, T3, T6). The three diagonal blocks are done separately because batched 3x3 and 6x6
    products are far cheaper than the 10x10 ones (by 10-20x per matrix on CPU).
    """
    first, second, rows, cols, rids, to_standard = _rotate_w_indices(ri.device)

    local = ri.new_zeros((ri.shape[0], 10, 10))
    local[:, rows, cols] = ri[:, rids]

    rot_t = rot.transpose(1, 2)
    transforms = (None, rot_t, _pp_rotation(rot_t, first, second))

    blocks = []
    for a, ta in enumerate(transforms):
        row = []
        for b, tb in enumerate(transforms):
            block = local[:, _BLOCKS[a], _BLOCKS[b]]
            if ta is not None:
                block = ta @ block
            if tb is not None:
                block = block @ tb.transpose(1, 2)
            row.append(block)
        blocks.append(torch.cat(row, dim=2))
    w = torch.cat(blocks, dim=1)
    return w[:, to_standard[:, None], to_standard[None, :]]


def _om1_rotate_core_columns_batch(core_columns, rot):
    """Rotate one column of the local core tensor (ss, sp, pp_sigma, pp_pi) to a 4x4 upper triangle."""
    c = core_columns
    local = torch.zeros((c.shape[0], 4, 4), dtype=c.dtype, device=c.device)
    local[:, 0, 0] = c[:, 0]
    local[:, 0, 1] = local[:, 1, 0] = c[:, 1]
    local[:, 1, 1] = c[:, 2]
    local[:, 2, 2] = local[:, 3, 3] = c[:, 3]
    u = torch.eye(4, dtype=c.dtype, device=c.device).repeat(c.shape[0], 1, 1)
    u[:, 1:, 1:] = rot
    return torch.triu(u @ local @ u.transpose(1, 2))


# ---------------------------------------------------------------------
# Core-GTO / CORPP2 (OM2, OM3)
# ---------------------------------------------------------------------

_CORE_GTO_EXPONENTS = (2.227660584, 0.4057711562, 0.1098175104)
_CORE_GTO_COEFFS = (0.1543289673, 0.5353281423, 0.4446345422)


def _spgto2_local_batch(basis_payload, zeta_b, rab):
    """Overlaps (s-core, p-core) of the valence Gaussians with the STO-3G core density of the partner."""
    exps = basis_payload["exponents"]
    cs = basis_payload["coeff_s"]
    cp = basis_payload["coeff_p"]

    # Built from Python scalars so no host-to-device copy happens per call.
    ec = torch.stack([c * zeta_b.square() for c in _CORE_GTO_EXPONENTS], dim=1)
    cc = torch.stack(
        [c * (2.0 * ec[:, k] / math.pi).pow(0.75) for k, c in enumerate(_CORE_GTO_COEFFS)], dim=1
    )

    r = rab.view(-1, 1, 1)
    g = exps.unsqueeze(2) + ec.unsqueeze(1)
    xqq = exps.unsqueeze(2) * ec.unsqueeze(1) * (r * r) / g
    s00 = (math.pi / g).pow(1.5) * torch.exp(-xqq)

    s1 = torch.sum(cs.unsqueeze(2) * cc.unsqueeze(1) * s00, dim=(1, 2))
    s2 = torch.sum(cp.unsqueeze(2) * cc.unsqueeze(1) * (ec.unsqueeze(1) * r * s00 / g), dim=(1, 2))
    return s1, s2


def _corpp2_column(shell_basis, rij, pa, pb, heavy_b, heavy_a):
    """
    CORPP2 correction (rows s, ps, pp) for electrons of atom a feeling the core of atom b.

    Evaluated for every pair and then masked: row s needs a heavy core b, the p rows need both atoms
    heavy. The core exponent parameter of H is zero, which would give an infinite derivative in
    ``_spgto2_local_batch``, so masked-out entries are fed a harmless value.
    """
    zscor = torch.where(heavy_b, pb["zscor"], torch.ones_like(pb["zscor"]))
    s1, s2 = _spgto2_local_batch(shell_basis, zscor, rij)
    fscor = pb["fscor"]

    gauss = torch.sqrt(rij) * 0.5
    t1 = (pa["beta_s"] + pb["bscor"]) * gauss * torch.exp(-(pa["alpha_s"] + pb["ascor"]) * rij * rij)
    t3 = (pa["beta_p"] + pb["bscor"]) * gauss * torch.exp(-(pa["alpha_p"] + pb["ascor"]) * rij * rij)

    zero = torch.zeros_like(rij)
    both = heavy_a & heavy_b
    return (
        torch.where(heavy_b, -2.0 * s1 * t1 - s1 * s1 * fscor, zero),
        torch.where(both, -(s1 * t3 + t1 * s2) - s1 * s2 * fscor, zero),
        torch.where(both, -(s2 * t3 + t3 * s2) - s2 * s2 * fscor, zero),
    )


def om2_corpp2_local_batch(ni, nj, rij, pi, pj, basis_i, basis_j):
    """CORPP2 tensor (npairs, 4, 2): column 0 = electrons on i / core j, column 1 = electrons on j / core i."""
    heavy_i, heavy_j = ni > 2, nj > 2
    a0, a1, a2 = _corpp2_column(basis_i, rij, pi, pj, heavy_j, heavy_i)
    b0, b1, b2 = _corpp2_column(basis_j, rij, pj, pi, heavy_i, heavy_j)
    zero = torch.zeros_like(rij)
    col0 = torch.stack([a0, a1, a2, zero], dim=1)
    col1 = torch.stack([b0, -b1, b2, zero], dim=1)  # the p-s row of the second column flips sign
    return torch.stack([col0, col1], dim=2)


# ---------------------------------------------------------------------
# Public caller
# ---------------------------------------------------------------------


def omx_pair_local_terms(
    method, ni, nj, rij, pi, pj, tore, s_local, t_local, basis_tables, basis_i, basis_j, atoms, idxi, idxj
):
    """
    Two-center Hcore ingredients in the local diatomic frame. They depend on the pair distance only.

    ``basis_i/basis_j`` are the per-pair basis payloads; ``atoms`` (``atom_gaussian_data`` of the
    per-atom payload) with ``idxi/idxj`` provides the REPGAU primitive data.

    Returns ``scaled_ri`` (npairs, 22) two-center integrals, ``core`` (npairs, 4, 2) core attractions
    (column 0: electrons on i, column 1: on j), ``fko`` and the plain semiempirical ``core_semi``.
    """
    if method not in ("OM1", "OM2", "OM3"):
        raise ValueError(f"Unsupported method: {method}")

    ri = om1_repgau_local_batch(rij, idxi, idxj, atoms, basis_tables["boys_integrals"])
    scaled_ri, fko = om1_gscale(rij, ri, ni, nj, pi["g_ss"], pj["g_ss"])
    rept = om1_cordef(ri, ni, nj, tore)
    core_semi = fko[:, None, None] * rept

    cort = om1_corgau(ni, nj, rij, tore, basis_tables, basis_i, basis_j)
    pen = om1_penetration(cort, rept, fko)

    # OM3 has no second-type VALPOT term (its fval2 is zero).
    need_valpp1 = method != "OM3"
    valpp_raw, valpp1 = om1_valpot(ni, nj, core_semi, s_local, t_local, pi, pj, need_valpp1)
    valpp = om1_apply_valpot_scaling(ni, nj, valpp_raw, valpp1, pi, pj)

    if method == "OM1":
        corpp = om1_ppecp_local(ni, nj, rij, basis_tables, basis_i, basis_j) * fko.view(-1, 1, 1)
    else:
        corpp = om2_corpp2_local_batch(ni, nj, rij, pi, pj, basis_i, basis_j)

    return {
        "scaled_ri": scaled_ri,
        "core": core_semi + pen + corpp + valpp,
        "fko": fko,
        "core_semi": core_semi,
    }


def omx_rotate_local_terms(local, rot, rot_t):
    """Rotate the local-frame terms to the molecular frame: ``w`` (npairs, 10, 10), ``e1b``/``e2a`` (npairs, 4, 4)."""
    return {
        "w": _om1_rotate_w_batch(local["scaled_ri"], rot),
        "e1b": _om1_rotate_core_columns_batch(local["core"][:, :, 0], rot_t),
        "e2a": _om1_rotate_core_columns_batch(local["core"][:, :, 1], rot_t),
    }

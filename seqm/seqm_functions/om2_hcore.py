import psutil
import torch

from .om1_overlap import (
    diatom_overlap_matrix_OM1,
    diatom_resonance_matrix_OM1,
    om1_local_overlap_terms,
    om1_local_resonance_terms,
)
from .om1_pair_backend import omx_pair_hcore_terms
from .omx_basis import select_om1_basis_payload
from .two_elec_two_center_int import rotate_with_quaternion

_OMX_HCORE_CONFIG = {
    "OM1": {"use_cor": False, "direct_resonance": True},
    "OM2": {"use_cor": True, "direct_resonance": False},
    "OM3": {"use_cor": False, "direct_resonance": False},
}


def _build_cor_table(molecule, pair_core_semi, tables):
    nmol, molsize = molecule.species.shape
    dtype = molecule.coordinates.dtype
    device = molecule.coordinates.device

    cor = torch.zeros((nmol, molsize * 4, molsize), dtype=dtype, device=device)

    mol = molecule.pair_molid
    atom_i = molecule.maskd[molecule.idxi] % molsize
    atom_j = molecule.maskd[molecule.idxj] % molsize
    row_i = atom_i * 4
    row_j = atom_j * 4

    cor[mol, row_i + 0, atom_j] = tables["U_ss"][molecule.ni] + pair_core_semi[:, 0, 0]
    cor[mol, row_j + 0, atom_i] = tables["U_ss"][molecule.nj] + pair_core_semi[:, 0, 1]

    heavy_i = molecule.ni > 1
    if heavy_i.any():
        temp_i = (
            tables["U_pp"][molecule.ni[heavy_i]]
            + (pair_core_semi[heavy_i, 2, 0] + 2.0 * pair_core_semi[heavy_i, 3, 0]) / 3.0
        )
        cor[mol[heavy_i], row_i[heavy_i] + 1, atom_j[heavy_i]] = temp_i
        cor[mol[heavy_i], row_i[heavy_i] + 2, atom_j[heavy_i]] = temp_i
        cor[mol[heavy_i], row_i[heavy_i] + 3, atom_j[heavy_i]] = temp_i

    heavy_j = molecule.nj > 1
    if heavy_j.any():
        temp_j = (
            tables["U_pp"][molecule.nj[heavy_j]]
            + (pair_core_semi[heavy_j, 2, 1] + 2.0 * pair_core_semi[heavy_j, 3, 1]) / 3.0
        )
        cor[mol[heavy_j], row_j[heavy_j] + 1, atom_i[heavy_j]] = temp_j
        cor[mol[heavy_j], row_j[heavy_j] + 2, atom_i[heavy_j]] = temp_j
        cor[mol[heavy_j], row_j[heavy_j] + 3, atom_i[heavy_j]] = temp_j

    return cor


def build_omx_pair_context(molecule, method, idxi, idxj, ni, nj, xij, rij):
    # if method not in _OMX_HCORE_CONFIG:
    #     raise ValueError(f"Unsupported OMx method: {method}")

    tables = molecule.parameters.get("_omx_tables")
    # if tables is None:
    #     raise RuntimeError("OMx parameter tables have not been cached on the molecule")
    basis_tables = molecule.parameters.get("_omx_basis")
    # if basis_tables is None:
    #     raise RuntimeError("OMx basis tables have not been cached on the molecule")
    basis_data = molecule.parameters.get("_omx_basis_data")
    # if basis_data is None:
    #     raise RuntimeError("OMx gathered basis data have not been cached on the molecule")

    basis_i = select_om1_basis_payload(basis_data, idxi)
    basis_j = select_om1_basis_payload(basis_data, idxj)

    rot = rotate_with_quaternion(xij)
    rot_t = rot.transpose(1, 2)
    direction = rot_t[:, :, 0]

    s_local = om1_local_overlap_terms(rij, basis_i, basis_j)
    t_local = om1_local_resonance_terms(ni, nj, rij, tables)
    pair = omx_pair_hcore_terms(
        method,
        ni,
        nj,
        rij,
        tables["g_ss"],
        molecule.const.tore,
        s_local,
        t_local,
        tables["U_ss"],
        tables["U_pp"],
        tables["fval1"],
        tables["fval2"],
        rot,
        rot_t,
        basis_tables,
        basis_i,
        basis_j,
        om2_tables=tables,
    )

    pair_resonance = diatom_resonance_matrix_OM1(xij, t_local, direction)
    context = {"pair": pair, "pair_resonance": pair_resonance, "pair_overlap": None}
    if method in {"OM2", "OM3"}:
        context["pair_overlap"] = diatom_overlap_matrix_OM1(xij, rij, direction, basis_i, basis_j, s_local)
    return context


def build_omx_hcore(molecule):
    method = molecule.method
    cfg = _OMX_HCORE_CONFIG[method]
    use_cor = cfg["use_cor"]
    direct_resonance = cfg["direct_resonance"]

    orb_dim = 4
    device = molecule.coordinates.device
    dtype = molecule.coordinates.dtype
    nblocks = molecule.nmol * molecule.molsize * molecule.molsize
    H_blocks = torch.zeros((nblocks, orb_dim, orb_dim), dtype=dtype, device=device)

    context = build_omx_pair_context(
        molecule, method, molecule.idxi, molecule.idxj, molecule.ni, molecule.nj, molecule.xij, molecule.rij
    )
    tables = molecule.parameters.get("_omx_tables")
    pair_data = context["pair"]

    for orb, key in enumerate(["U_ss", "U_pp", "U_pp", "U_pp"]):
        H_blocks[molecule.maskd, orb, orb] = molecule.parameters[key]
    H_blocks.index_add_(0, molecule.maskd[molecule.idxi], pair_data["e1b"])
    H_blocks.index_add_(0, molecule.maskd[molecule.idxj], pair_data["e2a"])

    pair_resonance = context["pair_resonance"]
    H_blocks[molecule.mask] = pair_resonance

    if direct_resonance or molecule.molsize < 3:
        return H_blocks, pair_data["w"], pair_data["fko"], None, None, None

    pair_overlap = context["pair_overlap"]
    S_blocks = torch.zeros_like(H_blocks)
    B_blocks = torch.zeros_like(H_blocks)
    S_blocks[molecule.mask] = pair_overlap
    S_blocks[molecule.mask_l] = pair_overlap.transpose(-1, -2)
    B_blocks[molecule.mask] = pair_resonance
    B_blocks[molecule.mask_l] = pair_resonance.transpose(-1, -2)

    _apply_omx_orthogonalization(
        molecule, H_blocks, S_blocks, B_blocks, tables, pair_data.get("core_semi"), use_cor
    )
    return H_blocks, pair_data["w"], pair_data["fko"], None, None, None


def _add_betor_shell_chunked_(
    out_blocks,
    S5,
    B5,
    molecule,
    gval1,
    COR,
    gval2,
    pair_idx,
    use_hi_p,
    use_hj_p,
    k_chunk=32,
    # cuts=None,
):
    molsize = molecule.molsize
    dtype = S5.dtype
    device = S5.device

    pair_mol = molecule.pair_molid
    ai = (molecule.mask // molsize) % molsize
    aj = molecule.mask % molsize

    CORs = COR[:, 0::4, :]
    CORp = COR[:, 1::4, :]
    heavy = molecule.species > 1

    mol = pair_mol[pair_idx]
    # Since ni >= nj, ai is the heavier/equal atom and aj is the lighter/equal atom.
    # The COR BETOR implementation assumes src_i <= src_j in atomic number,
    # so use reversed atom order and transpose before writing to the stored block.
    src_i = aj[pair_idx]
    src_j = ai[pair_idx]

    # if use_hi_p:
    #     heavy_i_f = heavy[mol, src_i].to(dtype)[:, None, None]
    # if use_hj_p:
    #     heavy_j_f = heavy[mol, src_j].to(dtype)[:, None, None]

    P = pair_idx.numel()

    ts1 = torch.zeros((P, 4, 4), dtype=dtype, device=device)
    ts2 = torch.zeros_like(ts1)

    for k0 in range(0, molsize, k_chunk):
        k1 = min(k0 + k_chunk, molsize)
        k_ids = torch.arange(k0, k1, device=device)
        K = k1 - k0

        # valid_k = (
        #     (k_ids[None, :] < nat_per_mol[mol, None])
        #     & (k_ids[None, :] != src_i[:, None])
        #     & (k_ids[None, :] != src_j[:, None])
        # )

        Si = S5[mol[:, None], src_i[:, None], k_ids[None, :]]
        Sj = S5[mol[:, None], src_j[:, None], k_ids[None, :]]
        Bi = B5[mol[:, None], src_i[:, None], k_ids[None, :]]
        Bj = B5[mol[:, None], src_j[:, None], k_ids[None, :]]

        # vf = valid_k.to(dtype)[..., None, None]

        ts1 += (Si @ Bj.transpose(-1, -2)).sum(dim=1)
        ts1 += (Bi @ Sj.transpose(-1, -2)).sum(dim=1)

        hi = torch.zeros((P, K, 4), dtype=dtype, device=device)
        hj = torch.zeros_like(hi)
        hi[..., 0] = CORs[mol[:, None], src_i[:, None], k_ids[None, :]]
        hj[..., 0] = CORs[mol[:, None], src_j[:, None], k_ids[None, :]]
        if use_hi_p:
            hi[..., 1:] = CORp[mol[:, None], src_i[:, None], k_ids[None, :]][..., None]  # * heavy_i_f
        if use_hj_p:
            hj[..., 1:] = CORp[mol[:, None], src_j[:, None], k_ids[None, :]][..., None]  # * heavy_j_f

        heavy_k = heavy[mol[:, None], k_ids[None, :]]
        hk = torch.zeros((P, K, 4), dtype=dtype, device=device)
        hk[..., 0] = (
            CORs[mol[:, None], k_ids[None, :], src_i[:, None]]
            + CORs[mol[:, None], k_ids[None, :], src_j[:, None]]
        )
        if heavy_k.any():
            hk_sel = heavy_k.nonzero(as_tuple=False)
            hk_p = (
                CORp[mol[hk_sel[:, 0]], k_ids[hk_sel[:, 1]], src_i[hk_sel[:, 0]]]
                + CORp[mol[hk_sel[:, 0]], k_ids[hk_sel[:, 1]], src_j[hk_sel[:, 0]]]
            )
            hk[hk_sel[:, 0], hk_sel[:, 1], 1:] = hk_p[:, None]

        base = Si @ Sj.transpose(-1, -2)
        weighted = (Si * hk[..., None, :]) @ Sj.transpose(-1, -2)

        ts2_k = base * (hi[..., :, None] + hj[..., None, :]) - weighted
        ts2 = ts2 + (ts2_k).sum(dim=1)

    ft1 = 0.25 * (gval1[molecule.ni[pair_idx]] + gval1[molecule.nj[pair_idx]])
    hsrc = -ft1[:, None, None] * ts1
    ft2 = 0.0625 * (gval2[molecule.ni[pair_idx]] + gval2[molecule.nj[pair_idx]])
    hsrc = hsrc + ft2[:, None, None] * ts2
    out_blocks[molecule.mask[pair_idx]] += hsrc.transpose(-1, -2)


def _add_betor_from_pairs_chunked_(out_blocks, S5, B5, molecule, gval1, COR, gval2, k_chunk=32):
    molsize = molecule.molsize
    heavy = molecule.species > 1
    pair_mol = molecule.pair_molid
    src_i = molecule.mask % molsize
    src_j = (molecule.mask // molsize) % molsize
    hi = heavy[pair_mol, src_i]
    hj = heavy[pair_mol, src_j]
    shells = ((~hi) & (~hj), (~hi) & hj, hi & hj)
    for mask, use_hi_p, use_hj_p in zip(shells, (False, False, True), (False, True, True)):
        if mask.any():
            sel = mask.nonzero(as_tuple=False).squeeze(1)
            _add_betor_shell_chunked_(
                out_blocks, S5, B5, molecule, gval1, COR, gval2, sel, use_hi_p, use_hj_p, k_chunk=k_chunk
            )


def _apply_omx_orthogonalization(molecule, H_blocks, S_blocks, B_blocks, tables, pair_core_semi, use_cor):
    S5 = S_blocks.reshape(molecule.nmol, molecule.molsize, molecule.molsize, 4, 4)
    B5 = B_blocks.reshape(molecule.nmol, molecule.molsize, molecule.molsize, 4, 4)
    if use_cor:
        COR = _build_cor_table(molecule, pair_core_semi, tables)
        # Save COR to use for gradients
        molecule.om2_COR = COR
        k_chunk = choose_k_chunk(
            molecule.mask.numel(), molecule.molsize, S5.dtype, S5.device, min_chunk=16, mat_equiv=16
        )
        _add_betor_from_pairs_chunked_(
            H_blocks,
            S5,
            B5,
            molecule,
            tables["gval1"],
            COR=COR,
            gval2=tables["gval2"],
            k_chunk=k_chunk,
            # cuts=None,
        )
    else:
        k_chunk = choose_k_chunk(
            molecule.mask.numel(), molecule.molsize, S5.dtype, S5.device, min_chunk=32, mat_equiv=8
        )
        _add_betor3_from_pairs_chunked_(
            H_blocks, S5, B5, molecule, tables["gval1"], k_chunk=k_chunk, cuts=None
        )


def _add_betor3_from_pairs_chunked_(out_blocks, S5, B5, molecule, gval1, k_chunk=64, cuts=1.0e-12):
    molsize = molecule.molsize
    dtype = S5.dtype
    device = S5.device

    Sss = S5[..., 0, 0]  # Cheap scalar view for screening.

    pair_mol = molecule.pair_molid
    ai = (molecule.mask // molsize) % molsize
    aj = molecule.mask % molsize

    mol = pair_mol
    src_i = ai
    src_j = aj

    P = molecule.mask.numel()
    ts1 = torch.zeros((P, 4, 4), dtype=dtype, device=device)

    for k0 in range(0, molsize, k_chunk):
        k1 = min(k0 + k_chunk, molsize)
        k_ids = torch.arange(k0, k1, device=device)

        # valid_k = (
        #     (k_ids[None, :] < nat_per_mol[mol, None])
        #     & (k_ids[None, :] != src_i[:, None])
        #     & (k_ids[None, :] != src_j[:, None])
        # )

        if cuts is not None:
            ss_i = Sss[mol[:, None], src_i[:, None], k_ids[None, :]]
            ss_j = Sss[mol[:, None], src_j[:, None], k_ids[None, :]]
            valid_k = (ss_i * ss_j) >= cuts
            # valid_k = valid_k & ((ss_i * ss_j) >= cuts)

            pk = valid_k.nonzero(as_tuple=False)
            if pk.numel() == 0:
                continue

            p_sel = pk[:, 0]
            k_sel = k_ids[pk[:, 1]]
            mol_sel = mol[p_sel]
            si_atom = src_i[p_sel]
            sj_atom = src_j[p_sel]
            Si = S5[mol_sel, si_atom, k_sel]
            Sj = S5[mol_sel, sj_atom, k_sel]
            Bi = B5[mol_sel, si_atom, k_sel]
            Bj = B5[mol_sel, sj_atom, k_sel]
            ts1.index_add_(0, p_sel, Si @ Bj.transpose(-1, -2) + Bi @ Sj.transpose(-1, -2))
        else:
            Si = S5[mol[:, None], src_i[:, None], k_ids[None, :]]
            Sj = S5[mol[:, None], src_j[:, None], k_ids[None, :]]
            Bi = B5[mol[:, None], src_i[:, None], k_ids[None, :]]
            Bj = B5[mol[:, None], src_j[:, None], k_ids[None, :]]
            ts1 += (Si @ Bj.transpose(-1, -2)).sum(dim=1)
            ts1 += (Bi @ Sj.transpose(-1, -2)).sum(dim=1)

    ft1 = 0.25 * (gval1[molecule.ni] + gval1[molecule.nj])
    hsrc = -ft1[:, None, None] * ts1
    out_blocks[molecule.mask] += hsrc


def choose_k_chunk(P_active, molsize, dtype, device, min_chunk=8, safety=0.35, mat_equiv=48):
    """
    Estimate safe k_chunk for BETOR-like [P_active, K, ...] temporaries.

    mat_equiv = approximate number of live [P,K,4,4] tensors.
    Use ~24 for BETOR3, ~40-64 for full BETOR depending on implementation.
    """
    dev = torch.device(device)
    elem = torch.empty((), dtype=dtype).element_size()

    if dev.type == "cuda":
        with torch.cuda.device(dev):
            free_mem, _ = torch.cuda.mem_get_info()
    else:
        free_mem = psutil.virtual_memory().available

    usable = int(free_mem * safety)

    bytes_per_k = P_active * 16 * elem * mat_equiv

    k = usable // max(bytes_per_k, 1)
    if k < min_chunk:
        torch.cuda.empty_cache()

    k = int(min(molsize, max(min_chunk, k)))
    
    return k

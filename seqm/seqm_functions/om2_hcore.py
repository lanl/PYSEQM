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


def _omx_tables(molecule):
    tables = molecule.parameters.get("_omx_tables")
    if tables is None:
        raise RuntimeError("OMx parameter tables have not been cached on the molecule")
    return tables


def _omx_basis(molecule):
    basis = molecule.parameters.get("_omx_basis")
    if basis is None:
        raise RuntimeError("OMx basis tables have not been cached on the molecule")
    return basis


def _omx_basis_data(molecule):
    basis_data = molecule.parameters.get("_omx_basis_data")
    if basis_data is None:
        raise RuntimeError("OMx gathered basis data have not been cached on the molecule")
    return basis_data


def _omx_resonance_tables(tables):
    return {
        name: tables[name]
        for name in [
            "beta_s",
            "beta_p",
            "beta_pi",
            "beta_sh",
            "beta_ph",
            "alpha_s",
            "alpha_p",
            "alpha_pi",
            "alpha_s_h",
            "alpha_p_h",
        ]
    }


def _fill_omx_one_center_blocks(H_blocks, molecule):
    for orb, key in enumerate(["U_ss", "U_pp", "U_pp", "U_pp"]):
        H_blocks[molecule.maskd, orb, orb] = molecule.parameters[key]


def _prepare_pair_rotation(molecule):
    rot = rotate_with_quaternion(molecule.xij)
    rot_t = rot.transpose(1, 2)
    direction = rot_t[:, :, 0]
    return rot, rot_t, direction


def _prepare_omx_local_pair_terms(molecule, tables, basis_i, basis_j):
    return (
        om1_local_overlap_terms(molecule.rij, basis_i, basis_j),
        om1_local_resonance_terms(molecule.ni, molecule.nj, molecule.rij, tables),
    )


def _prepare_omx_pair_overlap(molecule, rot_direction, basis_i, basis_j):
    return diatom_overlap_matrix_OM1(molecule.xij, molecule.rij, rot_direction, basis_i, basis_j)


def _prepare_omx_pair_resonance(molecule, resonance_tables, rot_direction):
    return diatom_resonance_matrix_OM1(
        molecule.ni, molecule.nj, molecule.xij, molecule.rij, resonance_tables, rot_direction
    )


def _assemble_omx_pair_terms(
    molecule, method, cfg, tables, basis_tables, s_local, t_local, rot, rot_t, basis_i, basis_j
):
    use_cor = cfg["use_cor"]

    pair = omx_pair_hcore_terms(
        method,
        molecule.ni,
        molecule.nj,
        molecule.rij,
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
    return {
        "pair_core_semi": pair["core_semi"] if use_cor else None,
        "w": pair["w"],
        "rho0xi": pair["fko"],
        "e1b": pair["e1b"],
        "e2a": pair["e2a"],
    }


def _build_omx_hcore(molecule, doTETCI=True, method="OM2"):
    if method not in _OMX_HCORE_CONFIG:
        raise ValueError(f"Unsupported OMx method: {method}")
    cfg = _OMX_HCORE_CONFIG[method]
    use_cor = cfg["use_cor"]
    direct_resonance = cfg["direct_resonance"]

    orb_dim = 4
    device = molecule.coordinates.device
    dtype = molecule.coordinates.dtype
    nblocks = molecule.nmol * molecule.molsize * molecule.molsize
    H_blocks = torch.zeros((nblocks, orb_dim, orb_dim), dtype=dtype, device=device)

    tables = _omx_tables(molecule)
    basis_tables = _omx_basis(molecule)
    basis_data = _omx_basis_data(molecule)
    basis_i = select_om1_basis_payload(basis_data, molecule.idxi)
    basis_j = select_om1_basis_payload(basis_data, molecule.idxj)
    _fill_omx_one_center_blocks(H_blocks, molecule)
    resonance_tables = _omx_resonance_tables(tables)

    rot, rot_t, direction = _prepare_pair_rotation(molecule)

    s_local, t_local = _prepare_omx_local_pair_terms(molecule, tables, basis_i, basis_j)

    pair_data = _assemble_omx_pair_terms(
        molecule, method, cfg, tables, basis_tables, s_local, t_local, rot, rot_t, basis_i, basis_j
    )
    H_blocks.index_add_(0, molecule.maskd[molecule.idxi], pair_data["e1b"])
    H_blocks.index_add_(0, molecule.maskd[molecule.idxj], pair_data["e2a"])

    pair_resonance = _prepare_omx_pair_resonance(molecule, resonance_tables, direction)

    if direct_resonance:
        H_blocks[molecule.mask] = pair_resonance
        return H_blocks, pair_data["w"], pair_data["rho0xi"], None, None, None

    pair_overlap = _prepare_omx_pair_overlap(molecule, direction, basis_i, basis_j)
    S_blocks = torch.zeros_like(H_blocks)
    B_blocks = torch.zeros_like(H_blocks)
    S_blocks[molecule.mask] = pair_overlap
    S_blocks[molecule.mask_l] = pair_overlap.transpose(-1, -2)
    B_blocks[molecule.mask] = pair_resonance
    B_blocks[molecule.mask_l] = pair_resonance.transpose(-1, -2)

    H_blocks = _apply_omx_orthogonalization(
        molecule, H_blocks, S_blocks, B_blocks, tables, pair_data["pair_core_semi"], use_cor, doTETCI
    )
    return H_blocks, pair_data["w"], pair_data["rho0xi"], None, None, None


def build_om1_hcore(molecule, doTETCI=True):
    return _build_omx_hcore(molecule, doTETCI=doTETCI, method="OM1")


def build_om2_hcore(molecule, doTETCI=True):
    return _build_omx_hcore(molecule, doTETCI=doTETCI, method="OM2")


def build_om3_hcore(molecule, doTETCI=True):
    return _build_omx_hcore(molecule, doTETCI=doTETCI, method="OM3")


def build_omx_hcore(molecule, doTETCI=True):
    return _build_omx_hcore(molecule, doTETCI=doTETCI, method=molecule.method)


def _add_betor_shell_chunked_(
    out_blocks,
    S_blocks,
    B_blocks,
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
    nmol = molecule.nmol
    molsize = molecule.molsize
    dtype = S_blocks.dtype
    device = S_blocks.device

    S5 = S_blocks.reshape(nmol, molsize, molsize, 4, 4)
    B5 = B_blocks.reshape(nmol, molsize, molsize, 4, 4)

    pair_mol = molecule.pair_molid
    ai = (molecule.mask // molsize) % molsize
    aj = molecule.mask % molsize
    nat_per_mol = (molecule.species > 0).sum(dim=1)

    CORs = COR[:, 0::4, :]
    CORp = COR[:, 1::4, :]
    heavy = molecule.species > 1

    mol = pair_mol[pair_idx]
    src_i = aj[pair_idx]
    src_j = ai[pair_idx]
    if use_hi_p:
        heavy_i_f = heavy[mol, src_i].to(dtype)[:, None, None]
    if use_hj_p:
        heavy_j_f = heavy[mol, src_j].to(dtype)[:, None, None]

    P = pair_idx.numel()

    ts1 = torch.zeros((P, 4, 4), dtype=dtype, device=device)
    ts2 = torch.zeros_like(ts1)

    for k0 in range(0, molsize, k_chunk):
        k1 = min(k0 + k_chunk, molsize)
        k_ids = torch.arange(k0, k1, device=device)
        K = k1 - k0

        valid_k = (
            (k_ids[None, :] < nat_per_mol[mol, None])
            & (k_ids[None, :] != src_i[:, None])
            & (k_ids[None, :] != src_j[:, None])
        )

        Si = S5[mol[:, None], src_i[:, None], k_ids[None, :]]
        Sj = S5[mol[:, None], src_j[:, None], k_ids[None, :]]
        Bi = B5[mol[:, None], src_i[:, None], k_ids[None, :]]
        Bj = B5[mol[:, None], src_j[:, None], k_ids[None, :]]

        vf = valid_k.to(dtype)[..., None, None]

        ts1 = ts1 + ((Si @ Bj.transpose(-1, -2)) * vf).sum(dim=1)
        ts1 = ts1 + ((Bi @ Sj.transpose(-1, -2)) * vf).sum(dim=1)

        hi = torch.zeros((P, K, 4), dtype=dtype, device=device)
        hj = torch.zeros_like(hi)
        hi[..., 0] = CORs[mol[:, None], src_i[:, None], k_ids[None, :]]
        hj[..., 0] = CORs[mol[:, None], src_j[:, None], k_ids[None, :]]
        if use_hi_p:
            hi[..., 1:] = CORp[mol[:, None], src_i[:, None], k_ids[None, :]][..., None] * heavy_i_f
        if use_hj_p:
            hj[..., 1:] = CORp[mol[:, None], src_j[:, None], k_ids[None, :]][..., None] * heavy_j_f

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
        ts2 = ts2 + (ts2_k * vf).sum(dim=1)

    ft1 = 0.25 * (gval1[molecule.ni[pair_idx]] + gval1[molecule.nj[pair_idx]])
    hsrc = -ft1[:, None, None] * ts1
    ft2 = 0.0625 * (gval2[molecule.ni[pair_idx]] + gval2[molecule.nj[pair_idx]])
    hsrc = hsrc + ft2[:, None, None] * ts2
    out_blocks[molecule.mask[pair_idx]] += hsrc.transpose(-1, -2)


def _add_betor_from_pairs_chunked_(out_blocks, S_blocks, B_blocks, molecule, gval1, COR, gval2, k_chunk=32):
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
                out_blocks,
                S_blocks,
                B_blocks,
                molecule,
                gval1,
                COR,
                gval2,
                sel,
                use_hi_p,
                use_hj_p,
                k_chunk=k_chunk,
            )


def _apply_omx_orthogonalization(
    molecule, H_blocks, S_blocks, B_blocks, tables, pair_core_semi, use_cor, doTETCI
):
    out = H_blocks.clone()

    # Preserve upper-block-only output behavior.
    out[molecule.mask] += B_blocks[molecule.mask]

    if not doTETCI:
        return out

    nat_per_mol = (molecule.species > 0).sum(dim=1)
    has_three_or_more_atoms = bool((nat_per_mol > 2).any().item())
    if not has_three_or_more_atoms:
        return out

    if use_cor:
        COR = _build_cor_table(molecule, pair_core_semi, tables)
        _add_betor_from_pairs_chunked_(
            out,
            S_blocks,
            B_blocks,
            molecule,
            tables["gval1"],
            COR=COR,
            gval2=tables["gval2"],
            k_chunk=16,
            # cuts=None,
        )
    else:
        _add_betor3_from_pairs_chunked_(out, S_blocks, B_blocks, molecule, tables["gval1"], k_chunk=32)

    return out


def _add_betor3_from_pairs_chunked_(
    out_blocks, S_blocks, B_blocks, molecule, gval1, k_chunk=64, cuts=1.0e-12
):
    nmol = molecule.nmol
    molsize = molecule.molsize
    dtype = S_blocks.dtype
    device = S_blocks.device

    S5 = S_blocks.reshape(nmol, molsize, molsize, 4, 4)
    B5 = B_blocks.reshape(nmol, molsize, molsize, 4, 4)

    Sss = S5[..., 0, 0]  # Cheap scalar view for screening.

    nat_per_mol = (molecule.species > 0).sum(dim=1)
    pair_mol = molecule.pair_molid
    ai = (molecule.mask // molsize) % molsize
    aj = molecule.mask % molsize

    mol = pair_mol
    src_i = aj
    src_j = ai

    P = molecule.mask.numel()
    ts1 = torch.zeros((P, 4, 4), dtype=dtype, device=device)

    for k0 in range(0, molsize, k_chunk):
        k1 = min(k0 + k_chunk, molsize)
        k_ids = torch.arange(k0, k1, device=device)

        valid_k = (
            (k_ids[None, :] < nat_per_mol[mol, None])
            & (k_ids[None, :] != src_i[:, None])
            & (k_ids[None, :] != src_j[:, None])
        )

        if cuts is not None:
            ss_i = Sss[mol[:, None], src_i[:, None], k_ids[None, :]]
            ss_j = Sss[mol[:, None], src_j[:, None], k_ids[None, :]]
            valid_k = valid_k & ((ss_i * ss_j) >= cuts)

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
            vf = valid_k.to(dtype)[..., None, None]
            ts1 += ((Si @ Bj.transpose(-1, -2)) * vf).sum(dim=1)
            ts1 += ((Bi @ Sj.transpose(-1, -2)) * vf).sum(dim=1)

    ft1 = 0.25 * (gval1[molecule.ni] + gval1[molecule.nj])
    hsrc = -ft1[:, None, None] * ts1
    out_blocks[molecule.mask] += hsrc.transpose(-1, -2)

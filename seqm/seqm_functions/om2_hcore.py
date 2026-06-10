import torch

from .om1_overlap import (
    diatom_overlap_matrix_OM1,
    diatom_resonance_matrix_OM1,
    om1_local_overlap_terms,
    om1_local_resonance_terms,
)
from .om1_pair_backend import omx_pair_hcore_terms
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


def _prepare_omx_local_pair_terms(molecule, tables):
    zeta_i = molecule.parameters["zeta"][molecule.idxi]
    zeta_j = molecule.parameters["zeta"][molecule.idxj]
    basis = _omx_basis(molecule)
    return (
        om1_local_overlap_terms(molecule.ni, molecule.nj, molecule.rij, zeta_i, zeta_j, basis),
        om1_local_resonance_terms(molecule.ni, molecule.nj, molecule.rij, tables),
    )


def _prepare_omx_pair_overlap(molecule, rot_direction):
    basis = _omx_basis(molecule)
    return diatom_overlap_matrix_OM1(
        molecule.ni,
        molecule.nj,
        molecule.xij,
        molecule.rij,
        molecule.parameters["zeta"][molecule.idxi],
        molecule.parameters["zeta"][molecule.idxj],
        rot_direction,
        basis,
    )


def _prepare_omx_pair_resonance(molecule, resonance_tables, rot_direction):
    return diatom_resonance_matrix_OM1(
        molecule.ni, molecule.nj, molecule.xij, molecule.rij, resonance_tables, rot_direction
    )


def _assemble_omx_pair_terms(molecule, method, cfg, tables, s_local, t_local, rot, rot_t):
    use_cor = cfg["use_cor"]
    basis = _omx_basis(molecule)

    pair = omx_pair_hcore_terms(
        method,
        molecule.ni,
        molecule.nj,
        molecule.rij,
        tables["zeta"],
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
        basis,
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
    _fill_omx_one_center_blocks(H_blocks, molecule)
    resonance_tables = _omx_resonance_tables(tables)

    rot, rot_t, direction = _prepare_pair_rotation(molecule)

    s_local, t_local = _prepare_omx_local_pair_terms(molecule, tables)

    pair_data = _assemble_omx_pair_terms(molecule, method, cfg, tables, s_local, t_local, rot, rot_t)
    H_blocks.index_add_(0, molecule.maskd[molecule.idxi], pair_data["e1b"])
    H_blocks.index_add_(0, molecule.maskd[molecule.idxj], pair_data["e2a"])

    pair_resonance = _prepare_omx_pair_resonance(molecule, resonance_tables, direction)

    if direct_resonance:
        H_blocks[molecule.mask] = pair_resonance
        return H_blocks, pair_data["w"], pair_data["rho0xi"], None, None, None

    pair_overlap = _prepare_omx_pair_overlap(molecule, direction)
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


def _add_betor_from_pairs_chunked_(
    out_blocks,
    S_blocks,
    B_blocks,
    molecule,
    gval1,
    COR,
    gval2,
    pair_chunk=2048,
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
    heavy = (molecule.species > 1).to(dtype)

    npairs = molecule.mask.numel()

    for p0 in range(0, npairs, pair_chunk):
        p1 = min(p0 + pair_chunk, npairs)

        mol = pair_mol[p0:p1]
        out_i = ai[p0:p1]
        out_j = aj[p0:p1]

        # Match old orientation:
        # old HO[upper i,j] = Hpair[j,i].T
        src_i = out_j
        src_j = out_i

        P = p1 - p0

        ts1 = torch.zeros((P, 4, 4), dtype=dtype, device=device)
        ts2 = torch.zeros_like(ts1) if COR is not None else None

        for k0 in range(0, molsize, k_chunk):
            k1 = min(k0 + k_chunk, molsize)
            k_ids = torch.arange(k0, k1, device=device)
            K = k1 - k0

            valid_k = (
                (k_ids[None, :] < nat_per_mol[mol, None])
                & (k_ids[None, :] != src_i[:, None])
                & (k_ids[None, :] != src_j[:, None])
            )

            # [P, K, 4, 4]
            Si = S5[mol[:, None], src_i[:, None], k_ids[None, :]]
            Sj = S5[mol[:, None], src_j[:, None], k_ids[None, :]]
            Bi = B5[mol[:, None], src_i[:, None], k_ids[None, :]]
            Bj = B5[mol[:, None], src_j[:, None], k_ids[None, :]]

            # if cuts is not None:
            #     valid_k = valid_k & ((Si[..., 0, 0] * Sj[..., 0, 0]) >= cuts)

            vf = valid_k.to(dtype)[..., None, None]

            ts1 = ts1 + ((Si @ Bj.transpose(-1, -2)) * vf).sum(dim=1)
            ts1 = ts1 + ((Bi @ Sj.transpose(-1, -2)) * vf).sum(dim=1)

            hi_s = CORs[mol[:, None], src_i[:, None], k_ids[None, :]]
            hj_s = CORs[mol[:, None], src_j[:, None], k_ids[None, :]]

            hi_p = CORp[mol[:, None], src_i[:, None], k_ids[None, :]]
            hj_p = CORp[mol[:, None], src_j[:, None], k_ids[None, :]]

            heavy_i = heavy[mol, src_i]
            heavy_j = heavy[mol, src_j]
            heavy_k = heavy[mol[:, None], k_ids[None, :]]

            hi = torch.empty((P, K, 4), dtype=dtype, device=device)
            hj = torch.empty_like(hi)

            hi[..., 0] = hi_s
            hj[..., 0] = hj_s

            hi[..., 1:] = hi_p[..., None] * heavy_i[:, None, None]
            hj[..., 1:] = hj_p[..., None] * heavy_j[:, None, None]

            hk_s = (
                CORs[mol[:, None], k_ids[None, :], src_i[:, None]]
                + CORs[mol[:, None], k_ids[None, :], src_j[:, None]]
            )

            hk_p = (
                CORp[mol[:, None], k_ids[None, :], src_i[:, None]]
                + CORp[mol[:, None], k_ids[None, :], src_j[:, None]]
            )

            hk = torch.empty((P, K, 4), dtype=dtype, device=device)
            hk[..., 0] = hk_s
            hk[..., 1:] = hk_p[..., None] * heavy_k[..., None]

            base = Si @ Sj.transpose(-1, -2)
            weighted = (Si * hk[..., None, :]) @ Sj.transpose(-1, -2)

            ts2_k = base * (hi[..., :, None] + hj[..., None, :]) - weighted
            ts2 = ts2 + (ts2_k * vf).sum(dim=1)

        ft1 = 0.25 * (gval1[molecule.ni[p0:p1]] + gval1[molecule.nj[p0:p1]])

        hsrc = -ft1[:, None, None] * ts1

        ft2 = 0.0625 * (gval2[molecule.ni[p0:p1]] + gval2[molecule.nj[p0:p1]])

        hsrc = hsrc + ft2[:, None, None] * ts2

        # Store old orientation into upper block.
        out_blocks[molecule.mask[p0:p1]] += hsrc.transpose(-1, -2)


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

    pair_chunk = molecule.mask.numel()  # npairs
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
            pair_chunk=pair_chunk,
            k_chunk=16,
            # cuts=None,
        )
    else:
        _add_betor3_from_pairs_chunked_(
            out, S_blocks, B_blocks, molecule, tables["gval1"], pair_chunk=pair_chunk, k_chunk=32
        )

    return out


def _add_betor3_from_pairs_chunked_(
    out_blocks, S_blocks, B_blocks, molecule, gval1, pair_chunk=4096, k_chunk=64, cuts=1.0e-12
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

    npairs = molecule.mask.numel()

    for p0 in range(0, npairs, pair_chunk):
        p1 = min(p0 + pair_chunk, npairs)

        mol = pair_mol[p0:p1]
        out_i = ai[p0:p1]
        out_j = aj[p0:p1]

        # Preserve orientation from old code:
        # HO[upper i,j] = Hpair[j,i].T
        src_i = out_j
        src_j = out_i

        P = p1 - p0
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

                term = Si @ Bj.transpose(-1, -2)
                term = term + Bi @ Sj.transpose(-1, -2)

                ts1.index_add_(0, p_sel, term)

            else:
                Si = S5[mol[:, None], src_i[:, None], k_ids[None, :]]
                Sj = S5[mol[:, None], src_j[:, None], k_ids[None, :]]
                Bi = B5[mol[:, None], src_i[:, None], k_ids[None, :]]
                Bj = B5[mol[:, None], src_j[:, None], k_ids[None, :]]

                vf = valid_k.to(dtype)[..., None, None]

                ts1 += ((Si @ Bj.transpose(-1, -2)) * vf).sum(dim=1)
                ts1 += ((Bi @ Sj.transpose(-1, -2)) * vf).sum(dim=1)

        ft1 = 0.25 * (gval1[molecule.ni[p0:p1]] + gval1[molecule.nj[p0:p1]])

        hsrc = -ft1[:, None, None] * ts1
        out_blocks[molecule.mask[p0:p1]] += hsrc.transpose(-1, -2)

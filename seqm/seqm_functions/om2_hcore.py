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


def _apply_omx_orthogonalization(molecule, H_blocks, S_blocks, B_blocks, tables, pair_core_semi, use_cor):
    S5 = S_blocks.reshape(molecule.nmol, molecule.molsize, molecule.molsize, 4, 4)
    B5 = B_blocks.reshape(molecule.nmol, molecule.molsize, molecule.molsize, 4, 4)
    COR = None
    gval2 = None
    if use_cor:
        gval2 = tables["gval2"]
        COR = _build_cor_table(molecule, pair_core_semi, tables)
        # Save COR to use for gradients
        molecule.om2_COR = COR
    _add_betor_cor_dense_(H_blocks, S5, B5, molecule, tables["gval1"], COR, gval2)


def _shell4_to_flat44(X):
    """
    X: [M, N, N, 4, 4] with indices [mol, atom_i, atom_k, orb_a, orb_c]
    returns [M, 4N, 4N] with rows (atom_i, orb_a), cols (atom_k, orb_c)
    """
    M, N = X.shape[:2]
    return X.permute(0, 1, 3, 2, 4).reshape(M, 4 * N, 4 * N)


def _flat44_to_blocks(T, N):
    """
    T: [M, 4N, 4N]
    returns [M, N, N, 4, 4] with [mol, atom_i, atom_j, orb_a, orb_b]
    """
    M = T.shape[0]
    return T.reshape(M, N, 4, N, 4).permute(0, 1, 3, 2, 4)


def _add_betor_cor_dense_(out_blocks, S5, B5, molecule, gval1, COR, gval2):
    """
    Dense GEMM implementation of full BETOR with COR.

    Equivalent target form:

        T1 = S @ B.T + B @ S.T

        D[i,k,a,c] = S[i,k,a,c] * (H[i,k,a] - C[k,i,c])
        T2 = D @ S.T + S @ D.T

        H += -ft1 * T1 + ft2 * T2

    S5/B5: [nmol, molsize, molsize, 4, 4]
    COR:   same layout expected by existing code:
           COR[:, 0::4, :] -> s correction, [M, N, N]
           COR[:, 1::4, :] -> p correction, [M, N, N]
    """

    M, N = S5.shape[:2]
    dtype = S5.dtype

    mask = molecule.mask
    mol = molecule.pair_molid
    ai = (mask // N) % N
    aj = mask % N

    # ---------------------------------------------------------------------
    # BETOR3-like part:
    # T1[i,j,a,b] = sum_kc S[i,k,a,c] B[j,k,b,c]
    #             + sum_kc B[i,k,a,c] S[j,k,b,c]
    # ---------------------------------------------------------------------
    S2 = _shell4_to_flat44(S5)  # [M, 4N, 4N]
    B2 = _shell4_to_flat44(B5)  # [M, 4N, 4N]

    T1_flat = torch.bmm(S2, B2.transpose(1, 2))
    T1_flat += torch.bmm(B2, S2.transpose(1, 2))

    T1 = _flat44_to_blocks(T1_flat, N)

    ft1 = 0.25 * (gval1[molecule.ni] + gval1[molecule.nj])
    hsrc = -ft1[:, None, None] * T1[mol, ai, aj]

    if COR is None:
        out_blocks.index_add_(0, molecule.mask, hsrc)
        return

    # ---------------------------------------------------------------------
    # COR part:
    #
    # Original pair expression:
    #   sum_kc S[i,k,a,c] S[j,k,b,c]
    #       * (hi[i,k,a] + hj[j,k,b] - hk[k,i,j,c])
    #
    # This can be written as:
    #   D @ S.T + S @ D.T
    # where:
    #   D[i,k,a,c] = S[i,k,a,c] * (Hrow[i,k,a] - Ccol[i,k,c])
    # Hrow is the row-atom correction.
    # Ccol is the k-atom correction transposed into the contraction orbital.
    # ---------------------------------------------------------------------
    CORs = COR[:, 0::4, :]  # [M, N, N]
    CORp = COR[:, 1::4, :]  # [M, N, N]

    heavy = (molecule.species > 1).to(dtype)  # [M, N]

    Hrow = S5.new_zeros((M, N, N, 4))
    Hrow[..., 0] = CORs
    Hrow[..., 1:] = CORp[..., None] * heavy[:, :, None, None]

    Ccol = S5.new_zeros((M, N, N, 4))
    Ccol[..., 0] = CORs.transpose(1, 2)
    Ccol[..., 1:] = CORp.transpose(1, 2)[..., None] * heavy[:, None, :, None]

    # D: [M, N, N, 4, 4]
    # D[m,i,k,a,c] = S[m,i,k,a,c] * (Hrow[m,i,k,a] - Ccol[m,i,k,c])
    D = S5 * (Hrow.unsqueeze(-1) - Ccol.unsqueeze(-2))

    D2 = _shell4_to_flat44(D)

    T2_flat = torch.bmm(D2, S2.transpose(1, 2))
    T2_flat += torch.bmm(S2, D2.transpose(1, 2))

    T2 = _flat44_to_blocks(T2_flat, N)

    ft2 = 0.0625 * (gval2[molecule.ni] + gval2[molecule.nj])
    hsrc = hsrc + ft2[:, None, None] * T2[mol, ai, aj]

    out_blocks.index_add_(0, mask, hsrc)


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
        H_blocks[molecule.maskd, orb, orb] = molecule.parameters[key].to(
            dtype=H_blocks.dtype, device=H_blocks.device
        )
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

import torch

from .om1_overlap import (
    diatom_overlap_matrix_OM1,
    diatom_resonance_matrix_OM1,
    om1_local_overlap_terms,
    om1_local_resonance_terms,
)
from .om1_pair_backend import atom_gaussian_data, omx_pair_local_terms, omx_rotate_local_terms
from .omx_basis import select_om1_basis_payload
from .omx_utils import blocks_to_dense, dense_to_blocks, omx_pair_params
from .two_elec_two_center_int import rotate_with_quaternion

_OMX_HCORE_CONFIG = {
    "OM1": {"use_cor": False, "direct_resonance": True},
    "OM2": {"use_cor": True, "direct_resonance": False},
    "OM3": {"use_cor": False, "direct_resonance": False},
}


def _pair_radial(molecule, idxi, idxj, ni, nj, rij):
    """Distance-only pair ingredients: parameters, basis payloads and the local overlap/resonance terms."""
    pi, pj = omx_pair_params(molecule.parameters, idxi, idxj)
    basis_data = molecule.parameters.get("_omx_basis_data")
    basis_i = select_om1_basis_payload(basis_data, idxi)
    basis_j = select_om1_basis_payload(basis_data, idxj)
    return {
        "pi": pi,
        "pj": pj,
        "basis_i": basis_i,
        "basis_j": basis_j,
        "s_local": om1_local_overlap_terms(rij, basis_i, basis_j),
        "t_local": om1_local_resonance_terms(ni, nj, rij, pi, pj),
    }


def pair_directions(xij):
    """Direction-only ingredients: the rotation to the diatomic frame."""
    rot = rotate_with_quaternion(xij)
    rot_t = rot.transpose(1, 2)
    return {"rot": rot, "rot_t": rot_t, "direction": rot_t[:, :, 0]}


def _pair_geometry(molecule, idxi, idxj, ni, nj, xij, rij):
    """Geometry-dependent pair ingredients shared by the Hcore build, the gradients and the ortho cache."""
    return {**_pair_radial(molecule, idxi, idxj, ni, nj, rij), **pair_directions(xij)}


def _local_terms(molecule, method, idxi, idxj, ni, nj, rij, radial):
    atoms = atom_gaussian_data(molecule.parameters.get("_omx_basis_data"))
    return omx_pair_local_terms(
        method,
        ni,
        nj,
        rij,
        radial["pi"],
        radial["pj"],
        molecule.const.tore,
        radial["s_local"],
        radial["t_local"],
        molecule.parameters.get("_omx_basis"),
        radial["basis_i"],
        radial["basis_j"],
        atoms,
        idxi,
        idxj,
    )


def build_omx_radial_context(molecule, method, idxi, idxj, ni, nj, rij, with_pair=True):
    """
    Everything that depends on the pair distance only: the local overlap/resonance terms (``s_local``,
    ``t_local``) and, with ``with_pair``, the local-frame two-center terms (``local``, see
    ``omx_pair_local_terms``). Rotating these with a pair direction gives ``build_omx_pair_context``.
    """
    radial = _pair_radial(molecule, idxi, idxj, ni, nj, rij)
    local = _local_terms(molecule, method, idxi, idxj, ni, nj, rij, radial) if with_pair else None
    return {"local": local, "s_local": radial["s_local"], "t_local": radial["t_local"]}


def build_omx_pair_context(molecule, method, idxi, idxj, ni, nj, xij, rij, with_pair=True):
    """
    Pair Hcore terms (``pair``), resonance and (OM2/OM3) overlap blocks of every atom pair.
    ``with_pair=False`` skips the two-center integral terms for callers that only need overlap/resonance.
    """
    geo = _pair_geometry(molecule, idxi, idxj, ni, nj, xij, rij)

    pair = None
    if with_pair:
        local = _local_terms(molecule, method, idxi, idxj, ni, nj, rij, geo)
        pair = omx_rotate_local_terms(local, geo["rot"], geo["rot_t"])
        pair["fko"] = local["fko"]
        if method == "OM2":
            pair["core_semi"] = local["core_semi"]

    context = {
        "pair": pair,
        "pair_resonance": diatom_resonance_matrix_OM1(xij, geo["t_local"], geo["direction"]),
        "pair_overlap": None,
        "pi": geo["pi"],
        "pj": geo["pj"],
    }
    if method in {"OM2", "OM3"}:
        context["pair_overlap"] = diatom_overlap_matrix_OM1(
            xij, rij, geo["direction"], geo["basis_i"], geo["basis_j"], geo["s_local"]
        )
    return context


def _symmetric_pair_blocks(molecule, pair_blocks):
    """Scatter per-pair 4x4 blocks into the full [nmol * molsize^2, 4, 4] layout (upper + transposed lower)."""
    blocks = pair_blocks.new_zeros((molecule.nmol * molecule.molsize**2, 4, 4))
    blocks[molecule.mask] = pair_blocks
    blocks[molecule.mask_l] = pair_blocks.transpose(-1, -2)
    return blocks.reshape(molecule.nmol, molecule.molsize, molecule.molsize, 4, 4)


def build_omx_ortho_cache(molecule):
    """Overlap/resonance blocks at the current geometry (the OM2/OM3 orthogonalisation gradient needs them)."""
    geo = _pair_geometry(
        molecule, molecule.idxi, molecule.idxj, molecule.ni, molecule.nj, molecule.xij, molecule.rij
    )
    resonance = diatom_resonance_matrix_OM1(molecule.xij, geo["t_local"], geo["direction"])
    overlap = diatom_overlap_matrix_OM1(
        molecule.xij, molecule.rij, geo["direction"], geo["basis_i"], geo["basis_j"], geo["s_local"]
    )
    return {
        "S5": _symmetric_pair_blocks(molecule, overlap),
        "B5": _symmetric_pair_blocks(molecule, resonance),
        "pi": geo["pi"],
        "pj": geo["pj"],
    }


def _build_cor_table(molecule, pair_core_semi, pi, pj):
    """
    OM2 orthogonalisation reference energies, shape [nmol, molsize, molsize, 4]:
    ``cor[m, a, k, c]`` is the energy of AO ``c`` of atom ``a`` in the field of atom ``k``'s core.
    """
    nmol, molsize = molecule.species.shape
    cor = pair_core_semi.new_zeros((nmol, molsize, molsize, 4))

    mol = molecule.pair_molid
    atom_i = molecule.maskd[molecule.idxi] % molsize
    atom_j = molecule.maskd[molecule.idxj] % molsize

    def orbital_energies(p, core, heavy):
        e_s = p["U_ss"] + core[:, 0]
        e_p = torch.where(heavy, p["U_pp"] + (core[:, 2] + 2.0 * core[:, 3]) / 3.0, torch.zeros_like(e_s))
        return torch.stack([e_s, e_p, e_p, e_p], dim=1)

    cor[mol, atom_i, atom_j] = orbital_energies(pi, pair_core_semi[:, :, 0], molecule.ni > 1)
    cor[mol, atom_j, atom_i] = orbital_energies(pj, pair_core_semi[:, :, 1], molecule.nj > 1)
    return cor


def omx_cor_difference(cor):
    """``F[m, i, k, a, c] = cor[m, i, k, a] - cor[m, k, i, c]`` for the OM2 orthogonalisation."""
    return cor.unsqueeze(-1) - cor.transpose(1, 2).unsqueeze(-2)


def _apply_omx_orthogonalization(molecule, H_blocks, S5, B5, pi, pj, pair_core_semi, use_cor):
    cor = gval2 = None
    if use_cor:
        gval2 = (pi["gval2"], pj["gval2"])
        cor = _build_cor_table(molecule, pair_core_semi, pi, pj)
        # The gradient code reuses COR.
        molecule.om2_COR = cor
    _add_betor_cor_dense_(H_blocks, S5, B5, molecule, (pi["gval1"], pj["gval1"]), cor, gval2)


def _plus_transpose(X):
    return X + X.transpose(1, 2)


def _add_betor_cor_dense_(out_blocks, S5, B5, molecule, gval1, cor, gval2):
    """
    Dense GEMM implementation of full BETOR with COR.

        T1 = S @ B.T + B @ S.T

        D[i,k,a,c] = S[i,k,a,c] * (cor[i,k,a] - cor[k,i,c])
        T2 = D @ S.T + S @ D.T

        H += -ft1 * T1 + ft2 * T2

    S5/B5: [nmol, molsize, molsize, 4, 4]; cor: [nmol, molsize, molsize, 4] or None (OM3).
    """
    N = S5.shape[1]
    mask = molecule.mask
    mol = molecule.pair_molid
    ai = (mask // N) % N
    aj = mask % N

    S2 = blocks_to_dense(S5)
    B2 = blocks_to_dense(B5)

    # T1[i,j,a,b] = sum_kc S[i,k,a,c] B[j,k,b,c] + B[i,k,a,c] S[j,k,b,c]; the second term is the
    # transpose of the first, so one product suffices.
    T1 = dense_to_blocks(_plus_transpose(torch.bmm(S2, B2.transpose(1, 2))), N)
    ft1 = 0.25 * (gval1[0] + gval1[1])
    hsrc = -ft1[:, None, None] * T1[mol, ai, aj]

    if cor is not None:
        D2 = blocks_to_dense(S5 * omx_cor_difference(cor))
        T2 = dense_to_blocks(_plus_transpose(torch.bmm(D2, S2.transpose(1, 2))), N)
        ft2 = 0.0625 * (gval2[0] + gval2[1])
        hsrc = hsrc + ft2[:, None, None] * T2[mol, ai, aj]

    out_blocks.index_add_(0, mask, hsrc)


def build_omx_hcore(molecule):
    method = molecule.method
    cfg = _OMX_HCORE_CONFIG[method]

    H_blocks = molecule.coordinates.new_zeros((molecule.nmol * molecule.molsize**2, 4, 4))
    context = build_omx_pair_context(
        molecule, method, molecule.idxi, molecule.idxj, molecule.ni, molecule.nj, molecule.xij, molecule.rij
    )
    pair_data = context["pair"]

    for orb, key in enumerate(["U_ss", "U_pp", "U_pp", "U_pp"]):
        H_blocks[molecule.maskd, orb, orb] = molecule.parameters[key].to(
            dtype=H_blocks.dtype, device=H_blocks.device
        )
    H_blocks.index_add_(0, molecule.maskd[molecule.idxi], pair_data["e1b"])
    H_blocks.index_add_(0, molecule.maskd[molecule.idxj], pair_data["e2a"])
    H_blocks[molecule.mask] = context["pair_resonance"]

    if not (cfg["direct_resonance"] or molecule.molsize < 3):
        _apply_omx_orthogonalization(
            molecule,
            H_blocks,
            _symmetric_pair_blocks(molecule, context["pair_overlap"]),
            _symmetric_pair_blocks(molecule, context["pair_resonance"]),
            context["pi"],
            context["pj"],
            pair_data.get("core_semi"),
            cfg["use_cor"],
        )
    return H_blocks, pair_data["w"], pair_data["fko"], None, None, None

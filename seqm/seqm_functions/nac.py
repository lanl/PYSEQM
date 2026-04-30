import torch

from seqm.seqm_functions.anal_grad import overlap_der_finiteDiff, w_der, w_derivative_numerical
from seqm.seqm_functions.rcis_batch import unpackone_batch

from .constants import a0


def _state_pair_tensors(state_pairs, device):
    state_pairs = list(state_pairs)
    if len(state_pairs) == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    pair_tensor = torch.as_tensor(state_pairs, dtype=torch.long, device=device)
    if pair_tensor.dim() != 2 or pair_tensor.shape[1] != 2:
        raise ValueError("state_pairs must be an iterable of (state1, state2) pairs.")
    return pair_tensor[:, 0] - 1, pair_tensor[:, 1] - 1


def _build_nac_derivative_operators(mol, P, ri, riXH, dtype, device):
    npairs = mol.rij.shape[0]
    overlap_x = torch.zeros((npairs, 3, 4, 4), dtype=dtype, device=device)
    zeta = torch.cat((mol.parameters["zeta_s"].unsqueeze(1), mol.parameters["zeta_p"].unsqueeze(1)), dim=1)
    Xij = mol.xij * mol.rij.unsqueeze(1) * a0
    overlap_der_finiteDiff(
        overlap_x,
        mol.idxi,
        mol.idxj,
        mol.rij,
        Xij,
        mol.parameters["beta"],
        mol.ni,
        mol.nj,
        zeta,
        mol.const.qn_int,
    )

    w_x = torch.zeros(npairs, 3, 10, 10, dtype=dtype, device=device)
    if riXH is not None and ri is not None:
        e1b_x, e2a_x = w_der(
            mol.const,
            mol.Z,
            mol.const.tore,
            mol.ni,
            mol.nj,
            w_x,
            mol.rij,
            mol.xij,
            Xij,
            mol.idxi,
            mol.idxj,
            mol.parameters["g_ss"],
            mol.parameters["g_pp"],
            mol.parameters["g_p2"],
            mol.parameters["h_sp"],
            mol.parameters["zeta_s"],
            mol.parameters["zeta_p"],
            riXH,
            ri,
        )
    else:
        e1b_x, e2a_x = w_derivative_numerical(mol, Xij, w_x)

    # The following logic to form the coulomb and exchange integrals by contracting the two-electron integrals
    # with the density matrix has been cribbed from fock.py.
    ind = torch.tensor(
        [[0, 1, 3, 6], [1, 2, 4, 7], [3, 4, 5, 8], [6, 7, 8, 9]], dtype=torch.int64, device=device
    )
    overlap_KAB_x = overlap_x
    Pp = P[mol.mask].unsqueeze(1)
    for i in range(4):
        w_x_i = w_x[..., ind[i], :]
        for j in range(4):
            overlap_KAB_x[..., i, j] -= torch.sum(Pp * (w_x_i[..., :, ind[j]]), dim=(2, 3))

    weight = torch.tensor(
        [1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0], dtype=dtype, device=device
    ).reshape((-1, 10))
    indices = (0, 0, 1, 0, 1, 2, 0, 1, 2, 3), (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)
    PA = (P[mol.maskd[mol.idxi]][..., indices[0], indices[1]] * weight).unsqueeze(-1)
    PB = (P[mol.maskd[mol.idxj]][..., indices[0], indices[1]] * weight).unsqueeze(-2)

    suma = torch.sum(PA.unsqueeze(1) * w_x, dim=2)
    sumA = torch.zeros_like(overlap_KAB_x)
    sumA[..., indices[0], indices[1]] = suma
    e2a_x.add_(sumA)

    sumb = torch.sum(PB.unsqueeze(1) * w_x, dim=3)
    sumB = torch.zeros_like(overlap_KAB_x)
    sumB[..., indices[0], indices[1]] = sumb
    e1b_x.add_(sumB)

    scale_emat = torch.tensor(
        [[1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    e1b_x *= scale_emat
    e2a_x *= scale_emat
    return overlap_KAB_x, e1b_x, e2a_x


def _contract_nac_density_batch(mol, B, overlap_KAB_x, e1b_x, e2a_x, nmol, molsize):
    pair_grad = torch.einsum("pbxy,pcxy->pbc", B[mol.mask], overlap_KAB_x)
    pair_grad.add_(
        torch.einsum("pbxy,pcxy->pbc", B[mol.maskd[mol.idxj]], e2a_x)
        + torch.einsum("pbxy,pcxy->pbc", B[mol.maskd[mol.idxi]], e1b_x)
    )

    nac_cis = torch.zeros(nmol * molsize, pair_grad.shape[1], 3, dtype=B.dtype, device=B.device)
    nac_cis.index_add_(0, mol.idxi, pair_grad)
    nac_cis.index_add_(0, mol.idxj, pair_grad, alpha=-1.0)
    return nac_cis.view(nmol, molsize, pair_grad.shape[1], 3).permute(0, 2, 1, 3)


def calc_nac(mol, amp, e_exc, P0, ri, riXH, state_pairs, rpa=False, pair_batch_size=4):
    """
    amp: tensor of CIS amplitudes of shape [nmol, nroots, nov].
    state_pairs: iterable of 1-based (state1, state2) pairs.

    Returns a tensor with shape [nmol, len(state_pairs), molsize, 3], ordered like state_pairs.
    """
    if rpa:
        raise NotImplementedError(
            "Nonadiabatic coupling vecotrs not yet implemented for RPA. Use CIS instead."
        )
    device = amp.device
    dtype = amp.dtype
    norb = int(mol.norb[0].item())
    nocc = int(mol.nocc[0].item())
    nvirt = norb - nocc
    nmol = int(mol.nmol)
    molsize = int(mol.molsize)
    state_i, state_j = _state_pair_tensors(state_pairs, device)
    n_state_pairs = int(state_i.numel())
    if n_state_pairs == 0:
        return torch.empty((nmol, 0, molsize, 3), dtype=dtype, device=device)

    # CIS unrelaxed density:
    # B = \sum_iab C_\mu a * t_ai * t_bi * C_\nu b - \sum_ija C_\mu i * t_ai * t_aj * C_\nu j
    C = mol.molecular_orbitals
    Cocc = C[:, :, :nocc]
    Cvirt = C[:, :, nocc:norb]
    nroots = amp.shape[1]
    amp_ia = amp.view(nmol, nroots, nocc, nvirt)
    P = P0.reshape(nmol, molsize, 4, molsize, 4).transpose(2, 3).reshape(nmol * molsize * molsize, 4, 4)
    overlap_KAB_x, e1b_x, e2a_x = _build_nac_derivative_operators(mol, P, ri, riXH, dtype, device)

    nHeavy = int(mol.nHeavy[0].item())
    nHydro = int(mol.nHydro[0].item())
    size_full = molsize * 4
    pair_batch_size = max(1, int(pair_batch_size))
    nac_cis = torch.empty((nmol, n_state_pairs, molsize, 3), dtype=dtype, device=device)

    for start in range(0, n_state_pairs, pair_batch_size):
        stop = min(start + pair_batch_size, n_state_pairs)
        i_chunk = state_i[start:stop]
        j_chunk = state_j[start:stop]
        nbatch = int(i_chunk.numel())

        amp_i = amp_ia.index_select(1, i_chunk)
        amp_j = amp_ia.index_select(1, j_chunk)
        v_i = torch.einsum("Nma,Nbia->Nbmi", Cvirt, amp_i)
        v_j = torch.einsum("Nma,Nbia->Nbmi", Cvirt, amp_j)
        o_i = torch.einsum("Nmi,Nbia->Nbma", Cocc, amp_i)
        o_j = torch.einsum("Nmi,Nbia->Nbma", Cocc, amp_j)
        Bij_chunk = torch.matmul(v_i, v_j.transpose(-1, -2)) - torch.matmul(o_i, o_j.transpose(-1, -2))
        Bij_chunk = 0.5 * (Bij_chunk + Bij_chunk.transpose(-1, -2))

        B0 = unpackone_batch(
            Bij_chunk.reshape(nmol * nbatch, Bij_chunk.shape[2], Bij_chunk.shape[3]),
            4 * nHeavy,
            nHydro,
            size_full,
        )
        B = (
            B0.reshape(nmol, nbatch, molsize, 4, molsize, 4)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(nmol * molsize * molsize, nbatch, 4, 4)
        )
        nac_cis[:, start:stop] = _contract_nac_density_batch(
            mol, B, overlap_KAB_x, e1b_x, e2a_x, nmol, molsize
        )

    denom = e_exc[:, state_j] - e_exc[:, state_i]
    nac_cis = nac_cis / denom[:, :, None, None]
    return nac_cis

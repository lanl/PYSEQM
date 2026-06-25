import torch

from .fock import UPPER_IDX0_4, UPPER_IDX1_4, WEIGHT_10, K_ind_4, _cached_index, _cached_tensor

# it is better to define mask as the same way defining maskd
# as it will be better to do summation using the representation of P in


def fock_sdc(P, P_sub, M, w_2, block_indices, nmol, molsize, idxi, idxj, parameters, maskd_sub, mask_sub):
    idxi_sub_ovrlp_with_rest = torch.isin(idxi, block_indices)
    idxj_sub_ovrlp_with_rest = torch.isin(idxj, block_indices)
    idxi_in_block = idxi_sub_ovrlp_with_rest
    idxj_in_block = idxj_sub_ovrlp_with_rest
    dtype = w_2.dtype
    device = w_2.device
    weight = _cached_tensor(WEIGHT_10, device, dtype).reshape((-1, 10))
    idx0 = _cached_index(UPPER_IDX0_4, device)
    idx1 = _cached_index(UPPER_IDX1_4, device)

    F = M.clone()

    Pptot = P_sub[..., 1, 1] + P_sub[..., 2, 2] + P_sub[..., 3, 3]

    TMP = torch.zeros_like(M)
    TMP[maskd_sub, 0, 0] = 0.5 * P_sub[maskd_sub, 0, 0] * parameters["g_ss"][block_indices] + Pptot[
        maskd_sub
    ] * (parameters["g_sp"][block_indices] - 0.5 * parameters["h_sp"][block_indices])
    for i in range(1, 4):
        # (p,p)
        TMP[maskd_sub, i, i] = (
            P_sub[maskd_sub, 0, 0]
            * (parameters["g_sp"][block_indices] - 0.5 * parameters["h_sp"][block_indices])
            + 0.5 * P_sub[maskd_sub, i, i] * parameters["g_pp"][block_indices]
            + (Pptot[maskd_sub] - P_sub[maskd_sub, i, i])
            * (1.25 * parameters["g_p2"][block_indices] - 0.25 * parameters["g_pp"][block_indices])
        )
        # (s,p) = (p,s) upper triangle
        TMP[maskd_sub, 0, i] = P_sub[maskd_sub, 0, i] * (
            1.5 * parameters["h_sp"][block_indices] - 0.5 * parameters["g_sp"][block_indices]
        )
    # (p,p*)
    for i, j in [(1, 2), (1, 3), (2, 3)]:
        TMP[maskd_sub, i, j] = P_sub[maskd_sub, i, j] * (
            0.75 * parameters["g_pp"][block_indices] - 1.25 * parameters["g_p2"][block_indices]
        )

    F.add_(TMP)

    ##############################################

    PA_test = (P[idxi[idxj_sub_ovrlp_with_rest]][..., idx0, idx1] * weight).reshape((-1, 10, 1))
    PB_test = (P[idxj[idxi_sub_ovrlp_with_rest]][..., idx0, idx1] * weight).reshape((-1, 1, 10))

    suma_test = torch.sum(PA_test * w_2[idxj_in_block], dim=1)
    sumb_test = torch.sum(PB_test * w_2[idxi_in_block], dim=2)

    sumA_test = torch.zeros(w_2[idxj_in_block].shape[0], 4, 4, dtype=dtype, device=device)
    sumB_test = torch.zeros(w_2[idxi_in_block].shape[0], 4, 4, dtype=dtype, device=device)

    sumA_test[..., idx0, idx1] = suma_test
    sumB_test[..., idx0, idx1] = sumb_test

    indi_of_new_diag_in_old = maskd_sub[
        (idxi[idxi_in_block].unsqueeze(1) == block_indices.unsqueeze(0)).max(dim=1).indices
    ]
    indj_of_new_diag_in_old = maskd_sub[
        (idxj[idxj_in_block].unsqueeze(1) == block_indices.unsqueeze(0)).max(dim=1).indices
    ]

    F.index_add_(0, indi_of_new_diag_in_old, sumB_test)
    F.index_add_(0, indj_of_new_diag_in_old, sumA_test)

    ####################################################

    sub_inds = idxi_sub_ovrlp_with_rest * idxj_sub_ovrlp_with_rest

    exchange_sum = torch.zeros(w_2[sub_inds].shape[0], 4, 4, dtype=dtype, device=device)
    ind = _cached_index(K_ind_4, device)
    # Pp =P[mask], P_{mu \in A, lambda \in B}
    Pp = -0.5 * P_sub[mask_sub]
    for i in range(4):
        for j in range(4):
            # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            exchange_sum[..., i, j] = torch.sum(
                Pp * w_2[sub_inds][..., ind[i], :][..., :, ind[j]], dim=(1, 2)
            )

    F.index_add_(0, mask_sub, exchange_sum)

    F0 = (
        F.reshape(nmol, len(block_indices), len(block_indices), 4, 4)
        .transpose(2, 3)
        .reshape(nmol, 4 * len(block_indices), 4 * len(block_indices))
    )
    F0.add_(F0.triu(1).transpose(1, 2))
    return F0

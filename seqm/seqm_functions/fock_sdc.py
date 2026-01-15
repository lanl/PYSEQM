import torch

# it is better to define mask as the same way defining maskd
# as it will be better to do summation using the representation of P in


def fock_sdc(P, P_sub, M, w_2, block_indices, nmol, molsize, idxi, idxj, parameters, maskd_sub, mask_sub):
    idxi_sub_ovrlp_with_rest = torch.isin(idxi, block_indices)
    idxj_sub_ovrlp_with_rest = torch.isin(idxj, block_indices)
    dtype = w_2.dtype
    device = w_2.device
    weight = torch.tensor(
        [1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0], dtype=dtype, device=device
    ).reshape((-1, 10))

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

    PA_test = (
        P[idxi[idxj_sub_ovrlp_with_rest]][..., (0, 0, 1, 0, 1, 2, 0, 1, 2, 3), (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)]
        * weight
    ).reshape((-1, 10, 1))
    PB_test = (
        P[idxj[idxi_sub_ovrlp_with_rest]][..., (0, 0, 1, 0, 1, 2, 0, 1, 2, 3), (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)]
        * weight
    ).reshape((-1, 1, 10))

    suma_test = torch.sum(PA_test * w_2[torch.isin(idxj, block_indices)], dim=1)
    sumb_test = torch.sum(PB_test * w_2[torch.isin(idxi, block_indices)], dim=2)

    sumA_test = torch.zeros(w_2[torch.isin(idxj, block_indices)].shape[0], 4, 4, dtype=dtype, device=device)
    sumB_test = torch.zeros(w_2[torch.isin(idxi, block_indices)].shape[0], 4, 4, dtype=dtype, device=device)

    sumA_test[..., (0, 0, 1, 0, 1, 2, 0, 1, 2, 3), (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)] = suma_test
    sumB_test[..., (0, 0, 1, 0, 1, 2, 0, 1, 2, 3), (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)] = sumb_test

    indi_of_new_diag_in_old = maskd_sub[
        (idxi[torch.isin(idxi, block_indices)].unsqueeze(1) == block_indices.unsqueeze(0)).max(dim=1).indices
    ]
    indj_of_new_diag_in_old = maskd_sub[
        (idxj[torch.isin(idxj, block_indices)].unsqueeze(1) == block_indices.unsqueeze(0)).max(dim=1).indices
    ]

    F.index_add_(0, indi_of_new_diag_in_old, sumB_test)
    F.index_add_(0, indj_of_new_diag_in_old, sumA_test)

    ####################################################

    sub_inds = idxi_sub_ovrlp_with_rest * idxj_sub_ovrlp_with_rest

    sum = torch.zeros(w_2[sub_inds].shape[0], 4, 4, dtype=dtype, device=device)
    ind = torch.tensor(
        [[0, 1, 3, 6], [1, 2, 4, 7], [3, 4, 5, 8], [6, 7, 8, 9]], dtype=torch.int64, device=device
    )
    # Pp =P[mask], P_{mu \in A, lambda \in B}
    Pp = -0.5 * P_sub[mask_sub]
    for i in range(4):
        for j in range(4):
            # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            sum[..., i, j] = torch.sum(Pp * w_2[sub_inds][..., ind[i], :][..., :, ind[j]], dim=(1, 2))

    F.index_add_(0, mask_sub, sum)

    F0 = (
        F.reshape(nmol, len(block_indices), len(block_indices), 4, 4)
        .transpose(2, 3)
        .reshape(nmol, 4 * len(block_indices), 4 * len(block_indices))
    )
    F0.add_(F0.triu(1).transpose(1, 2))
    return F0

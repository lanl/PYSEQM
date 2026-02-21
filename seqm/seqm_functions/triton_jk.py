import torch

from .triton_sp_common import TRITON_AVAILABLE, get_sp_const_tensors, tl, triton, triton_eligible

_TRITON_AVAILABLE = TRITON_AVAILABLE
_CONST_CACHE = {}
_TRITON_FAILED = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _jk_j_kernel(
        P_ptr, w_ptr, maskd_ptr, idxi_ptr, idxj_ptr, pack_off_ptr, pack_w_ptr, JA_ptr, JB_ptr, npairs
    ):
        pid = tl.program_id(0)
        if pid >= npairs:
            return

        offs10 = tl.arange(0, 10)
        cols10 = tl.arange(0, 10)

        pair_i = tl.load(idxi_ptr + pid).to(tl.int32)
        pair_j = tl.load(idxj_ptr + pid).to(tl.int32)
        center_i = tl.load(maskd_ptr + pair_i).to(tl.int32)
        center_j = tl.load(maskd_ptr + pair_j).to(tl.int32)
        base_i = center_i * 16
        base_j = center_j * 16

        p_off = tl.load(pack_off_ptr + offs10).to(tl.int32)
        p_wt = tl.load(pack_w_ptr + offs10)
        pA = tl.load(P_ptr + base_i + p_off) * p_wt
        pB = tl.load(P_ptr + base_j + p_off) * p_wt

        wA = tl.load(w_ptr + pid * 100 + offs10[:, None] * 10 + cols10[None, :])
        J_A = tl.sum(wA * pA[:, None], axis=0)

        wB = tl.load(w_ptr + pid * 100 + cols10[:, None] * 10 + offs10[None, :])
        J_B = tl.sum(wB * pB[None, :], axis=1)

        tl.store(JA_ptr + pid * 10 + cols10, J_A)
        tl.store(JB_ptr + pid * 10 + cols10, J_B)

    @triton.jit
    def _jk_k_rhf_kernel(P_ptr, w_ptr, mask_ptr, ind_ptr, K_ptr, npairs):
        pid = tl.program_id(0)
        if pid >= npairs:
            return

        offs4 = tl.arange(0, 4)
        center = tl.load(mask_ptr + pid).to(tl.int32)
        base = center * 16
        Pp = -0.5 * tl.load(P_ptr + base + offs4[:, None] * 4 + offs4[None, :])
        ind = tl.load(ind_ptr + offs4[:, None] * 4 + offs4[None, :]).to(tl.int32)

        for i in tl.static_range(4):
            for j in tl.static_range(4):
                acc = tl.zeros((), dtype=Pp.dtype)
                for nu in tl.static_range(4):
                    left = ind[i, nu]
                    right = ind[j, :]
                    wv = tl.load(w_ptr + pid * 100 + left * 10 + right)
                    acc += tl.sum(Pp[nu, :] * wv, axis=0)
                tl.store(K_ptr + pid * 16 + i * 4 + j, acc)

    @triton.jit
    def _jk_k_uhf_kernel(P_ptr, w_ptr, mask_ptr, ind_ptr, K_ptr, npairs, nblocks):
        pid_pair = tl.program_id(0)
        pid_spin = tl.program_id(1)
        if pid_pair >= npairs:
            return

        offs4 = tl.arange(0, 4)
        center = tl.load(mask_ptr + pid_pair).to(tl.int32)
        base = (pid_spin * nblocks + center) * 16
        Pp = -1.0 * tl.load(P_ptr + base + offs4[:, None] * 4 + offs4[None, :])
        ind = tl.load(ind_ptr + offs4[:, None] * 4 + offs4[None, :]).to(tl.int32)

        out_base = (pid_spin * npairs + pid_pair) * 16
        for i in tl.static_range(4):
            for j in tl.static_range(4):
                acc = tl.zeros((), dtype=Pp.dtype)
                for nu in tl.static_range(4):
                    left = ind[i, nu]
                    right = ind[j, :]
                    wv = tl.load(w_ptr + pid_pair * 100 + left * 10 + right)
                    acc += tl.sum(Pp[nu, :] * wv, axis=0)
                tl.store(K_ptr + out_base + i * 4 + j, acc)


def triton_jk_restricted_sp(P, w, maskd, mask, idxi, idxj):
    global _TRITON_FAILED
    if not triton_eligible(P, w, _TRITON_FAILED):
        return None
    try:
        P_t = P.contiguous()
        w_t = w.contiguous()
        maskd_t = maskd.contiguous()
        mask_t = mask.contiguous()
        idxi_t = idxi.contiguous()
        idxj_t = idxj.contiguous()
        npairs = int(w_t.shape[0])
        const = get_sp_const_tensors(_CONST_CACHE, P_t.device, P_t.dtype)

        J_A = torch.empty((npairs, 10), device=P_t.device, dtype=P_t.dtype)
        J_B = torch.empty((npairs, 10), device=P_t.device, dtype=P_t.dtype)
        Ksum = torch.empty((npairs, 4, 4), device=P_t.device, dtype=P_t.dtype)

        grid = (npairs,)
        _jk_j_kernel[grid](
            P_t,
            w_t,
            maskd_t,
            idxi_t,
            idxj_t,
            const["pack_off"],
            const["pack_w"],
            J_A,
            J_B,
            npairs,
            num_warps=1,
            num_stages=1,
        )
        _jk_k_rhf_kernel[grid](P_t, w_t, mask_t, const["ind4_i32"], Ksum, npairs, num_warps=1, num_stages=1)
        return J_A, J_B, Ksum
    except Exception:
        _TRITON_FAILED = True
        return None


def triton_jk_unrestricted_sp(P_tot, P_spin, w, maskd, mask, idxi, idxj):
    global _TRITON_FAILED
    if not triton_eligible(P_tot, w, _TRITON_FAILED):
        return None
    try:
        P_tot_t = P_tot.contiguous()
        P_spin_t = P_spin.contiguous()
        w_t = w.contiguous()
        maskd_t = maskd.contiguous()
        mask_t = mask.contiguous()
        idxi_t = idxi.contiguous()
        idxj_t = idxj.contiguous()
        npairs = int(w_t.shape[0])
        nblocks = int(P_spin_t.shape[1])
        const = get_sp_const_tensors(_CONST_CACHE, P_tot_t.device, P_tot_t.dtype)

        J_A = torch.empty((npairs, 10), device=P_tot_t.device, dtype=P_tot_t.dtype)
        J_B = torch.empty((npairs, 10), device=P_tot_t.device, dtype=P_tot_t.dtype)
        Ksum = torch.empty((2, npairs, 4, 4), device=P_tot_t.device, dtype=P_tot_t.dtype)

        grid_j = (npairs,)
        _jk_j_kernel[grid_j](
            P_tot_t,
            w_t,
            maskd_t,
            idxi_t,
            idxj_t,
            const["pack_off"],
            const["pack_w"],
            J_A,
            J_B,
            npairs,
            num_warps=1,
            num_stages=1,
        )
        grid_k = (npairs, 2)
        _jk_k_uhf_kernel[grid_k](
            P_spin_t, w_t, mask_t, const["ind4_i32"], Ksum, npairs, nblocks, num_warps=1, num_stages=1
        )
        return J_A, J_B, Ksum
    except Exception:
        _TRITON_FAILED = True
        return None

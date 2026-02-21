import torch

from .triton_sp_common import TRITON_AVAILABLE, get_sp_const_tensors, tl, triton, triton_eligible

_TRITON_AVAILABLE = TRITON_AVAILABLE
_CONST_CACHE = {}
_TRITON_FAILED = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _rcis_jk_batch_kernel(
        P_ptr,
        w_ptr,
        maskd_ptr,
        idxi_ptr,
        idxj_ptr,
        mask_ptr,
        pack_off_ptr,
        pack_w_ptr,
        ind_ptr,
        JA_ptr,
        JB_ptr,
        K_ptr,
        npairs,
        nblocks,
        nroots,
    ):
        pid_pair = tl.program_id(0)
        pid_dens = tl.program_id(1)
        if pid_pair >= npairs:
            return

        offs10 = tl.arange(0, 10)
        cols10 = tl.arange(0, 10)
        offs4 = tl.arange(0, 4)

        idx_i = tl.load(idxi_ptr + pid_pair).to(tl.int32)
        idx_j = tl.load(idxj_ptr + pid_pair).to(tl.int32)
        center_i = tl.load(maskd_ptr + idx_i).to(tl.int32)
        center_j = tl.load(maskd_ptr + idx_j).to(tl.int32)

        p_off = tl.load(pack_off_ptr + offs10).to(tl.int32)
        p_wt = tl.load(pack_w_ptr + offs10)

        base_i = (pid_dens * nblocks + center_i) * 16
        base_j = (pid_dens * nblocks + center_j) * 16
        pA = tl.load(P_ptr + base_i + p_off) * p_wt
        pB = tl.load(P_ptr + base_j + p_off) * p_wt

        mol_id = pid_dens // nroots
        w_base = (mol_id * npairs + pid_pair) * 100
        w_mat = tl.load(w_ptr + w_base + offs10[:, None] * 10 + cols10[None, :])

        J_A = tl.sum(w_mat * pA[:, None], axis=0)
        J_B = tl.sum(w_mat * pB[None, :], axis=1)

        out10 = (pid_dens * npairs + pid_pair) * 10
        tl.store(JA_ptr + out10 + cols10, J_A)
        tl.store(JB_ptr + out10 + cols10, J_B)

        pair_block = tl.load(mask_ptr + pid_pair).to(tl.int32)
        pair_base = (pid_dens * nblocks + pair_block) * 16
        Pp = -0.5 * tl.load(P_ptr + pair_base + offs4[:, None] * 4 + offs4[None, :])
        ind = tl.load(ind_ptr + offs4[:, None] * 4 + offs4[None, :]).to(tl.int32)

        out16 = (pid_dens * npairs + pid_pair) * 16
        for i in tl.static_range(4):
            for j in tl.static_range(4):
                acc = tl.zeros((), dtype=Pp.dtype)
                for nu in tl.static_range(4):
                    left = ind[i, nu]
                    right = ind[j, :]
                    wv = tl.load(w_ptr + w_base + left * 10 + right)
                    acc += tl.sum(Pp[nu, :] * wv, axis=0)
                tl.store(K_ptr + out16 + i * 4 + j, acc)

    @triton.jit
    def _rcis_jk_any_kernel(
        P_ptr,
        w_ptr,
        mdi_ptr,
        mdj_ptr,
        mask_ptr,
        pack_off_ptr,
        pack_w_ptr,
        ind_ptr,
        JA_ptr,
        JB_ptr,
        K_ptr,
        npairs,
        ntasks,
    ):
        pid = tl.program_id(0)
        if pid >= ntasks:
            return

        offs10 = tl.arange(0, 10)
        cols10 = tl.arange(0, 10)
        offs4 = tl.arange(0, 4)

        pair = pid % npairs
        block_i = tl.load(mdi_ptr + pid).to(tl.int32)
        block_j = tl.load(mdj_ptr + pid).to(tl.int32)
        pair_block = tl.load(mask_ptr + pid).to(tl.int32)

        p_off = tl.load(pack_off_ptr + offs10).to(tl.int32)
        p_wt = tl.load(pack_w_ptr + offs10)

        base_i = block_i * 16
        base_j = block_j * 16
        pA = tl.load(P_ptr + base_i + p_off) * p_wt
        pB = tl.load(P_ptr + base_j + p_off) * p_wt

        w_base = pair * 100
        w_mat = tl.load(w_ptr + w_base + offs10[:, None] * 10 + cols10[None, :])

        J_A = tl.sum(w_mat * pA[:, None], axis=0)
        J_B = tl.sum(w_mat * pB[None, :], axis=1)

        out10 = pid * 10
        tl.store(JA_ptr + out10 + cols10, J_A)
        tl.store(JB_ptr + out10 + cols10, J_B)

        pair_base = pair_block * 16
        Pp = -0.5 * tl.load(P_ptr + pair_base + offs4[:, None] * 4 + offs4[None, :])
        ind = tl.load(ind_ptr + offs4[:, None] * 4 + offs4[None, :]).to(tl.int32)

        out16 = pid * 16
        for i in tl.static_range(4):
            for j in tl.static_range(4):
                acc = tl.zeros((), dtype=Pp.dtype)
                for nu in tl.static_range(4):
                    left = ind[i, nu]
                    right = ind[j, :]
                    wv = tl.load(w_ptr + w_base + left * 10 + right)
                    acc += tl.sum(Pp[nu, :] * wv, axis=0)
                tl.store(K_ptr + out16 + i * 4 + j, acc)


def triton_makeA_pi_jk_batch(P, w, maskd, mask, idxi, idxj):
    global _TRITON_FAILED
    if not triton_eligible(P, w, _TRITON_FAILED):
        return None

    try:
        nmol, nroots, nblocks = P.shape[:3]
        npairs = int(mask.shape[0])
        ndens = nmol * nroots

        P_t = P.contiguous().view(ndens, nblocks, 4, 4)
        w_t = w.contiguous()
        maskd_t = maskd.contiguous().to(torch.int32)
        mask_t = mask.contiguous().to(torch.int32)
        idxi_t = idxi.contiguous().to(torch.int32)
        idxj_t = idxj.contiguous().to(torch.int32)
        const = get_sp_const_tensors(_CONST_CACHE, P_t.device, P_t.dtype)

        JA = torch.empty((ndens, npairs, 10), device=P_t.device, dtype=P_t.dtype)
        JB = torch.empty((ndens, npairs, 10), device=P_t.device, dtype=P_t.dtype)
        K = torch.empty((ndens, npairs, 4, 4), device=P_t.device, dtype=P_t.dtype)

        grid = (npairs, ndens)
        _rcis_jk_batch_kernel[grid](
            P_t,
            w_t,
            maskd_t,
            idxi_t,
            idxj_t,
            mask_t,
            const["pack_off"],
            const["pack_w"],
            const["ind4_i32"],
            JA,
            JB,
            K,
            npairs,
            nblocks,
            nroots,
            num_warps=1,
            num_stages=1,
        )

        return (
            JA.view(nmol, nroots, npairs, 10),
            JB.view(nmol, nroots, npairs, 10),
            K.view(nmol, nroots, npairs, 4, 4),
        )
    except Exception:
        _TRITON_FAILED = True
        return None


def triton_makeA_pi_jk_any(P, w, md_i, md_j, mask):
    global _TRITON_FAILED
    if not triton_eligible(P, w, _TRITON_FAILED):
        return None

    try:
        nD, npairs = md_i.shape
        ntasks = nD * npairs

        P_t = P.contiguous()
        w_t = w.contiguous()
        mdi_t = md_i.reshape(-1).contiguous().to(torch.int32)
        mdj_t = md_j.reshape(-1).contiguous().to(torch.int32)
        mask_t = mask.reshape(-1).contiguous().to(torch.int32)
        const = get_sp_const_tensors(_CONST_CACHE, P_t.device, P_t.dtype)

        JA = torch.empty((ntasks, 10), device=P_t.device, dtype=P_t.dtype)
        JB = torch.empty((ntasks, 10), device=P_t.device, dtype=P_t.dtype)
        K = torch.empty((ntasks, 4, 4), device=P_t.device, dtype=P_t.dtype)

        grid = (ntasks,)
        _rcis_jk_any_kernel[grid](
            P_t,
            w_t,
            mdi_t,
            mdj_t,
            mask_t,
            const["pack_off"],
            const["pack_w"],
            const["ind4_i32"],
            JA,
            JB,
            K,
            npairs,
            ntasks,
            num_warps=1,
            num_stages=1,
        )

        return JA.view(nD, npairs, 10), JB.view(nD, npairs, 10), K.view(nD, npairs, 4, 4)
    except Exception:
        _TRITON_FAILED = True
        return None

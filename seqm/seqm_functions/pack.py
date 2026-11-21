import torch

# as there are 0 padding in many matrixes
# to save space as well as to make diag & matmul faster
# pack and unpack remove this padding and put the padding back


def packone(x, nho, nHydro, norb):
    x0 = torch.zeros((norb, norb), dtype=x.dtype, device=x.device)
    x0[:nho, :nho] = x[:nho, :nho]
    x0[:nho, nho : (nho + nHydro)] = x[:nho, nho : (nho + 4 * nHydro) : 4]
    x0[nho : (nho + nHydro), nho : (nho + nHydro)] = x[
        nho : (nho + 4 * nHydro) : 4, nho : (nho + 4 * nHydro) : 4
    ]
    x0[nho : (nho + nHydro), :nho] = x[nho : (nho + 4 * nHydro) : 4, :nho]
    return x0


def unpackone(x0, nho, nHydro, size):
    x = torch.zeros((size, size), dtype=x0.dtype, device=x0.device)
    x[:nho, :nho] = x0[:nho, :nho]
    x[:nho, nho : (nho + 4 * nHydro) : 4] = x0[:nho, nho : (nho + nHydro)]
    x[nho : (nho + 4 * nHydro) : 4, nho : (nho + 4 * nHydro) : 4] = x0[
        nho : (nho + nHydro), nho : (nho + nHydro)
    ]
    x[nho : (nho + 4 * nHydro) : 4, :nho] = x0[nho : (nho + nHydro), :nho]
    return x


def _pack_batch_same(x_flat, nho: int, nHy: int):
    """
    x_flat: (B, size, size)
    nH, nHy: same for all molecules
    """
    norb = nho + nHy
    B = x_flat.shape[0]

    x0 = x_flat.new_zeros(B, norb, norb)
    x0[:, :nho, :nho] = x_flat[:, :nho, :nho]
    x0[:, :nho, nho : (nho + nHy)] = x_flat[:, :nho, nho : (nho + 4 * nHy) : 4]
    x0[:, nho : (nho + nHy), nho : (nho + nHy)] = x_flat[
        :, nho : (nho + 4 * nHy) : 4, nho : (nho + 4 * nHy) : 4
    ]
    x0[:, nho : (nho + nHy), :nho] = x_flat[:, nho : (nho + 4 * nHy) : 4, :nho]
    return x0


def _unpack_batch_same(x0_flat, nho: int, nHy: int, size: int):
    """
    x0_flat: (B, norb, norb)
    """
    B = x0_flat.shape[0]

    x = x0_flat.new_zeros(B, size, size)
    x[:, :nho, :nho] = x0_flat[:, :nho, :nho]
    x[:, :nho, nho : (nho + 4 * nHy) : 4] = x0_flat[:, :nho, nho : (nho + nHy)]
    x[:, nho : (nho + 4 * nHy) : 4, nho : (nho + 4 * nHy) : 4] = x0_flat[
        :, nho : (nho + nHy), nho : (nho + nHy)
    ]
    x[:, nho : (nho + 4 * nHy) : 4, :nho] = x0_flat[:, nho : (nho + nHy), :nho]
    return x


def _same_counts(*counts):
    return all(count.numel() == 1 or torch.equal(count, count[:1].expand_as(count)) for count in counts)


def _packed_to_padded_index(nHeavy, nHydro, norb):
    """
    Position in the padded 4-orbitals-per-atom basis of every packed orbital, shape (nmol, norb), and a mask of
    the positions that exist. Packed order: 4 orbitals per heavy atom, then the s orbital of each hydrogen.
    """
    nho = (4 * nHeavy).reshape(-1, 1)
    p = torch.arange(norb, device=nho.device)[None, :]
    idx = torch.where(p < nho, p, nho + (p - nho) * 4)
    valid = p < nho + nHydro.reshape(-1, 1)
    return torch.where(valid, idx, torch.zeros_like(idx)), valid


def pack(x, nHeavy, nHydro):
    nho = 4 * nHeavy
    if x.dim() == 2:
        return packone(x, nho, nHydro, nho + nHydro)

    if x.dim() == 4:
        x = x.flatten(start_dim=0, end_dim=1)

    if isinstance(nho, int) and isinstance(nHydro, int):
        return _pack_batch_same(x, nho, nHydro)

    if _same_counts(nho, nHydro):
        return _pack_batch_same(x, nho[0].item(), nHydro[0].item())

    norb = int((nho + nHydro).max().item())
    idx, valid = _packed_to_padded_index(nHeavy, nHydro, norb)
    b = torch.arange(x.shape[0], device=x.device)[:, None, None]
    return x[b, idx[:, :, None], idx[:, None, :]] * (valid[:, :, None] & valid[:, None, :])


def unpack(x0, nHeavy, nHydro, size):
    nho = 4 * nHeavy
    if x0.dim() == 2:
        return unpackone(x0, nho, nHydro, size)

    if isinstance(nho, int) and isinstance(nHydro, int):
        return _unpack_batch_same(x0, nho, nHydro, size)

    if _same_counts(nho, nHydro):
        return _unpack_batch_same(x0, nho[0].item(), nHydro[0].item(), size)

    idx, valid = _packed_to_padded_index(nHeavy, nHydro, x0.shape[-1])
    b = torch.arange(x0.shape[0], device=x0.device)[:, None, None]
    x = torch.zeros((x0.shape[0], size, size), dtype=x0.dtype, device=x0.device)
    # padding positions all point at index 0 but add zero
    mask = valid[:, :, None] & valid[:, None, :]
    return x.index_put((b, idx[:, :, None], idx[:, None, :]), x0 * mask, accumulate=True)

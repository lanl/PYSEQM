import torch
from torch.func import vmap

#as there are 0 padding in many matrixes
#to save space as well as to make diag & matmul faster
#pack and unpack remove this padding and put the padding back

def packone(x, nho, nHydro, norb):
    x0 = torch.zeros((norb,norb), dtype=x.dtype, device=x.device)
    x0[:nho,:nho]=x[:nho,:nho]
    x0[:nho, nho:(nho+nHydro)] = x[:nho,nho:(nho+4*nHydro):4]
    x0[nho:(nho+nHydro),nho:(nho+nHydro)] = x[nho:(nho+4*nHydro):4,nho:(nho+4*nHydro):4]
    x0[nho:(nho+nHydro), :nho] = x[nho:(nho+4*nHydro):4, :nho]
    return x0

def unpackone(x0, nho, nHydro, size):
    x = torch.zeros((size, size), dtype=x0.dtype, device=x0.device)
    x[:nho,:nho] = x0[:nho,:nho]
    x[:nho,nho:(nho+4*nHydro):4] = x0[:nho, nho:(nho+nHydro)]
    x[nho:(nho+4*nHydro):4,nho:(nho+4*nHydro):4] = x0[nho:(nho+nHydro),nho:(nho+nHydro)]
    x[nho:(nho+4*nHydro):4, :nho] = x0[nho:(nho+nHydro), :nho]
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
    x0[:, :nho, nho:(nho+nHy)] = x_flat[:, :nho, nho:(nho+4*nHy):4]
    x0[:, nho:(nho+nHy), nho:(nho+nHy)] = x_flat[:, nho:(nho+4*nHy):4, nho:(nho+4*nHy):4]
    x0[:, nho:(nho+nHy), :nho] = x_flat[:, nho:(nho+4*nHy):4, :nho]
    return x0


def _unpack_batch_same(x0_flat, nho: int, nHy: int, size: int):
    """
    x0_flat: (B, norb, norb)
    """
    B  = x0_flat.shape[0]

    x = x0_flat.new_zeros(B, size, size)
    x[:, :nho, :nho] = x0_flat[:, :nho, :nho]
    x[:, :nho, nho:(nho+4*nHy):4] = x0_flat[:, :nho, nho:(nho+nHy)]
    x[:, nho:(nho+4*nHy):4, nho:(nho+4*nHy):4] = x0_flat[:, nho:(nho+nHy), nho:(nho+nHy)]
    x[:, nho:(nho+4*nHy):4, :nho] = x0_flat[:, nho:(nho+nHy), :nho]
    return x

def pack(x, nHeavy, nHydro):
    nho = 4*nHeavy
    # single matrix
    if x.dim()==2:
        x0 = packone(x, nHeavy*4, nHydro, nho+nHydro)
    # batch 
    else:
        if x.dim()==4:
            x = x.flatten(start_dim=0, end_dim=1)
        
        same = (nho.unique().numel() == 1) and (nHydro.unique().numel() == 1)
        if same:
            h, hy = nho[0].item(), nHydro[0].item()
            x0 = _pack_batch_same(x, h, hy)
        else:
            norb = int((nho + nHydro).max().item())
            x0 = torch.stack(list(map(lambda a, b, c : packone(a, b, c, norb), x, nho, nHydro)))

    return x0


def unpack(x0, nHeavy, nHydro, size):

    nho = 4*nHeavy
    if x0.dim()==2: # single matrix
        x = unpackone(x0, nho, nHydro, size)
    else: # batch
        same = (nho.unique().numel() == 1) and (nHydro.unique().numel() == 1)
        if same:
            h, hy = nho[0].item(), nHydro[0].item()
            x = _unpack_batch_same(x0, h, hy, size)
        else:
            x = torch.stack(list(map(lambda a, b, c : unpackone(a, b, c, size), x0, nho, nHydro)))
    return x
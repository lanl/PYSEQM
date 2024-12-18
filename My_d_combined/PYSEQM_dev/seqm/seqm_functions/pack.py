import torch
#as there are 0 padding in many matrixes
#to save space as well as to make diag, matmul faster
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

def pack(x, nHeavy, nHydro):
    #print(x.shape)
    nho = 4*nHeavy
    #print('Pack ', x.dim())
    #print(x)
    if x.dim()==2:
        #print('pack 2')
        x0 = packone(x, nHeavy*4, nHydro, nho+nHydro)
    elif x.dim()==4:
        #print('pack 4')
        norb = torch.max(nho+nHydro)
        x = x.flatten(start_dim=0, end_dim=1)
        x0 = torch.stack(list(map(lambda a, b, c : packone(a, b, c, norb), x, nho, nHydro)))
    else:
        norb = torch.max(nho+nHydro)
        x0 = torch.stack(list(map(lambda a, b, c : packone(a, b, c, norb), x, nho, nHydro)))

    return x0


def unpack(x0, nHeavy, nHydro, size):

    if x0.dim()==2:
        x = unpackone(x0, nHeavy*4, nHydro, size)
    else:
        nho = 4*nHeavy
        x = torch.stack(list(map(lambda a, b, c : unpackone(a, b, c, size), x0, nho, nHydro)))
    return x

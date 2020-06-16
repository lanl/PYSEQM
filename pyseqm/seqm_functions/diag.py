import torch
from .pack import *
#this pseudo_diag is not efficient to be implemented in python
#as it applies a series Jacobi transformation on eigenvectors (in place operations)
#have to use python for loop
def pseudo_diag(x,C,E,nheavyatom,nH,nocc):
    #x single Fock matrix
    #here x has padding 0, but is symmetric, i.e. lower and upper trianlge parts are filled
    #C eigenvectors from previous iteration
    #C is from symeig of x0, see sym_eig_trunc, C[:,i] is eigenvectors i
    #E eigenvalues with shape x.shape[0]
    dtype =  x.dtype
    device = x.device
    """
    norbs = nheavyatom*4 + nH
    F=torch.zeros((norbs, norbs),dtype=dtype,device=device)

    nho = nheavyatom*4
    F[:nho,:nho]=x[:nho,:nho]
    F[:nho, nho:(nho+nH)] = x[:nho,nho:(nho+4*nH):4]
    F[nho:(nho+nH),nho:(nho+nH)] = x[nho:(nho+4*nH):4,nho:(nho+4*nH):4]
    F[nho:(nho+nH), :nho] = x[nho:(nho+4*nH):4,:nho]
    """
    F = pack(x, nheavyatom, nH)
    #Fov = C_o^T * F * C_v
    if nocc>norbs/2.0:
        Fov = torch.matmul(C[:,:nocc].T, torch.matmul(F, C[:,nocc:]))
    else:
        Fov = torch.matmul(torch.matmul(C[:,:nocc].T,F), C[:,nocc:])
    D = E[:nocc].unsqueeze(1)-E[nocc:norbs].unsqueeze(0)
    Cr = C.clone()

    tmp = - torch.sqrt(4.0*Fov**2+D**2)
    alp = torch.sqrt(0.5*(1.0+D/tmp))
    bet = - torch.sqrt(1.0-alp**2) * Fov.sign()
    #these two criterias are taken from mopac7 diag.f
    tiny = 0.05*torch.max(torch.abs(Fov))
    #bigeps = 1.4901161193847656e-7
    bigeps = 1.49e-7
    cond = (Fov>=tiny) * (torch.abs(Fov/D)>=bigeps)

    for i in range(Norbs-Nocc): # virtual orbs
        for j in range(Nocc): # occupied orbs
            if cond[j,i]:
                A = Cr[:,j]
                B = Cr[:,i]
                Cr[:,j] = alp[j,i]*A + bet[j,i]*B
                Cr[:,i] = alp[j,i]*B - bet[j,i]*A

    #

    # each column of v is a eigenvectors
    # P_alpha_beta = 2.0 * |sum_i c_{i,alpha}*c_{i,beta}, i \in occupied MO
    # v : v_{alpha,i}
    # 2.0*v_{alpha,i,1}*v_{1,i,beta}
    t = 2.0*torch.matmul(Cr[:,:nocc], Cr[:,:nocc].transpose(0,1))
    """
    P =  torch.zeros_like(x)
    P[:nho,:nho] = t[:nho,:nho]
    P[:nho,nho:(nho+4*nH):4] = t[:nho, nho:(nho+nH)]
    P[nho:(nho+4*nH):4,nho:(nho+4*nH):4] = t[nho:(nho+nH),nho:(nho+nH)]
    #in pulay, the lower triangle part is also needed
    P[nho:(nho+4*nH):4,:nho] = t[nho:(nho+nH), :nho]
    """
    P = unpack(t, nheavyatom, nH, x.shape)

    if eigenvectors:
        return P, v
    else:
        return P

def sym_eig_trunc(x,nheavyatom,nH,nocc, eig_only=False):

    dtype =  x.dtype
    device = x.device
    if x.dim()==2:
        e0,v = torch.symeig(pack(x, nheavyatom, nH),eigenvectors=True,upper=True)
        e = torch.zeros((x.shape[0]),dtype=dtype,device=device)
        e[:(nheavyatom*4+nH)] = e0
    else:#need to add large diagonal values to replace 0 padding
        #Gershgorin circle theorem estimate upper bounds of eigenvalues
        x0 = pack(x, nheavyatom, nH)
        nmol, size, _ = x0.shape

        aii = x0.diagonal(dim1=1,dim2=2)
        ri = torch.sum(torch.abs(x0),dim=2)-torch.abs(aii)
        hN = torch.max(aii+ri,dim=1)[0]
        dE = hN - torch.min(aii-ri,dim=1)[0] #(maximal - minimal) get range

        norb = nheavyatom*4+nH
        pnorb = size - norb
        nn = torch.max(pnorb).item()
        dx = 0.005
        mutipler = torch.arange(1.0+dx, 1.0+nn*dx+dx, dx, dtype=dtype, device=device)[:nn]
        ind = torch.arange(size, dtype=torch.int64, device=device)
        cond = pnorb>0
        for i in range(nmol):
            if cond[i]:
                x0[i,ind[norb[i]:], ind[norb[i]:]] = mutipler[:pnorb[i]]*dE[i]+hN[i]
        try:
            e0,v = torch.symeig(x0,eigenvectors=True,upper=True)
        except:
            if torch.isnan(x0).any():
                print(x0)
            #print(x0.detach().data.numpy())
            e0,v = torch.symeig(x0,eigenvectors=True,upper=True)
        e = torch.zeros((nmol, x.shape[-1]),dtype=dtype,device=device)
        e[...,:size] = e0
        for i in range(nmol):
            if cond[i]:
                e[i,norb[i]:size] = 0.0

    if eig_only:
        return e, v

    # each column of v is a eigenvectors
    # P_alpha_beta = 2.0 * |sum_i c_{i,alpha}*c_{i,beta}, i \in occupied MO
    if x.dim()==2:
        t = 2.0*torch.matmul(v[:,:nocc], v[:,:nocc].transpose(0,1))
    else:
        """
        t = torch.zeros_like(v)
        for i in range(v.shape[0]):
            t[i] = torch.matmul(v[i,:,:nocc[i]], v[i, :,:nocc[i]].transpose(0,1))

        t*=2.0
        """
        t = 2.0*torch.stack(list(map(lambda a,n : torch.matmul(a[:,:n], a[:,:n].transpose(0,1)), v, nocc)))
    P = unpack(t, nheavyatom, nH, x.shape[-1])

    return e,P, v


def sym_eig_trunc1(x,nheavyatom,nH,nocc, eig_only=False):

    dtype =  x.dtype
    device = x.device
    if x.dim()==2:
        e0,v = torch.symeig(pack(x, nheavyatom, nH),eigenvectors=True,upper=True)
        e = torch.zeros((x.shape[0]),dtype=dtype,device=device)
        e[:(nheavyatom*4+nH)] = e0
    else:#need to add large diagonal values to replace 0 padding
        #Gershgorin circle theorem estimate upper bounds of eigenvalues

        e0, v0 = list(zip(*list(map(
                        lambda a,b,c: torch.symeig(pack(a,b,c),eigenvectors=True,upper=True),
                        x,nheavyatom, nH))))
        P0 = list(map(
                     lambda v, nc : 2.0*torch.matmul(v[:,:nc], v[:,:nc].transpose(0,1)),
                     v0,nocc))
        #
        nmol = x.shape[0]
        norb = nheavyatom*4+nH
        e=torch.zeros(x.shape[:2], dtype=dtype, device=device)
        P = torch.zeros_like(x)
        for i in range(nmol):
            e[i,:norb[i]] = e0[i]
            P[i] = unpack(P0[i], nheavyatom[i], nH[i], x.shape[-1])

    if eig_only:
        return e, v0

    # each column of v is a eigenvectors
    # P_alpha_beta = 2.0 * |sum_i c_{i,alpha}*c_{i,beta}, i \in occupied MO

    return e,P, v0

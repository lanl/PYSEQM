import torch
from .pack import *

#this pseudo_diag is not efficient to be implemented in python
#as it applies a series Jacobi transformation on eigenvectors (in place operations)
#have to use python for loop

CHECK_DEGENERACY = False
# flag to control whether consider degeneracy when constructing the density matrix
# if no, then occupied orbitals are from 0, 1, ..., nocc-1
# if yes, then equal contributions are used for degenarated orbitals near fermi level

DEGEN_EIGENSOLVER = True
# flag whether or not to use eigensolver that's differentiable for degenerate
# eigenpairs (keep option to use default pytorch solver for sanity checks)
DEGEN_THRESHOLD = torch.finfo(torch.float).eps ** 0.6
# threshold below which to consider eigenpairs degenerate


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


def construct_P(e, v, nocc):
    # e: eigenvalue, sorted, ascending
    # v: eigenvector
    # nocc: int 
    if e.dtype==torch.float32:
            atol = 1.0e-7
    elif e.dtype==torch.float64:
            atol = 1.0e-14
    cond = (e-e[nocc-1]).abs()<=atol
    if cond[nocc:].any():
        c = torch.nonzero(cond)
        indx1 = c[0].item()
        indx2 = c[-1].item()+1
        nd = indx2-indx1
        coeff = torch.ones(1, indx2, device=e.device,dtype=e.dtype)
        coeff[0,indx1:] = (nocc.type(torch.double)-indx1)/nd
        t = 2.0*torch.matmul(coeff*v[:,:indx2], v[:,:indx2].transpose(0,1))
    else:
        t = 2.0*torch.matmul(v[:,:nocc], v[:,:nocc].transpose(0,1))
    return t


def sym_eig_trunc(x, nheavyatom, nH, nocc, eig_only=False):
    sym_eigh = degen_symeig.apply if DEGEN_EIGENSOLVER else pytorch_symeig
    dtype =  x.dtype
    device = x.device
    if x.dim()==2:

        #print('diag: DOING 2')
        e0, v = sym_eigh(pack(x, nheavyatom, nH))
        e = torch.zeros((x.shape[0]),dtype=dtype,device=device)
        e[:(nheavyatom*4+nH)] = e0
    elif x.dim()==4:
        nheavyatom = nheavyatom.repeat_interleave(2)
        nH = nH.repeat_interleave(2)
        nocc = nocc.flatten()
        #Gershgorin circle theorem estimate upper bounds of eigenvalues  
        x_orig_shape = x.size()
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
            e0, v = sym_eigh(x0)
        except:
            if torch.isnan(x0).any(): print(x0)
            e0, v = sym_eigh(x0)
        e = torch.zeros((nmol, x.shape[-1]),dtype=dtype,device=device)
        e[...,:size] = e0
        for i in range(nmol):
            if cond[i]: e[i,norb[i]:size] = 0.0
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
            e0, v = sym_eigh(x0)
        except:
            if torch.isnan(x0).any(): print(x0)
            e0, v = sym_eigh(x0)
        e = torch.zeros((nmol, x.shape[-1]),dtype=dtype,device=device)
        e[...,:size] = e0
        for i in range(nmol):
            if cond[i]: e[i,norb[i]:size] = 0.0

    if eig_only:
        if x.dim()==4:
            e = e.reshape(x_orig_shape[0:3])
            v = v.reshape(int(v.shape[0]/2),2,v.shape[1],v.shape[2])
        return e, v

    # each column of v is a eigenvectors
    # P_alpha_beta = 2.0 * |sum_i c_{i,alpha}*c_{i,beta}, i \in occupied MO
    if x.dim()==2:
        if CHECK_DEGENERACY:
            t = construct_P(e, v, nocc)
        else:
            t = 2.0*torch.matmul(v[:,:nocc], v[:,:nocc].transpose(0,1))
    else:
        """
        t = torch.zeros_like(v)
        for i in range(v.shape[0]):
            t[i] = torch.matmul(v[i,:,:nocc[i]], v[i, :,:nocc[i]].transpose(0,1))

        t*=2.0
        """
        if CHECK_DEGENERACY:
            t = torch.stack(list(map(lambda a,b,n : construct_P(a, b, n), e, v, nocc)))
        else:
            t = 2.0*torch.stack(list(map(lambda a,n : torch.matmul(a[:,:n], a[:,:n].transpose(0,1)), v, nocc)))
            
    P = unpack(t, nheavyatom, nH, x.shape[-1])

    if x.dim()==4:
        e = e.reshape(x_orig_shape[0:3])
        v = v.reshape(int(v.shape[0]/2),2,v.shape[1],v.shape[2])
        P = P.reshape(x_orig_shape)

    return e, P, v



def sym_eig_trunc1(x, nheavyatom, nH, nocc, eig_only=False):
    sym_eigh = degen_symeig.apply if DEGEN_EIGENSOLVER else pytorch_symeig
    dtype = x.dtype
    device = x.device
    
    if x.dim()==2:
        print('diag: DOING 2')
        e0, v = sym_eigh(pack(x, nheavyatom, nH))
        e = torch.zeros((x.shape[0]), dtype=dtype, device=device)
        e[:(nheavyatom*4+nH)] = e0
    elif x.dim()==4:#need to add large diagonal values to replace 0 padding
        #Gershgorin circle theorem estimate upper bounds of eigenvalues
        nheavyatom = nheavyatom.repeat_interleave(2)
        nH = nH.repeat_interleave(2)
        nocc = nocc.flatten()
        x_orig_shape = x.size()
        x = x.flatten(start_dim=0, end_dim=1)
        
        e0, v0 = list(zip(*list(map(
                        lambda a, b, c: sym_eigh(pack(a, b, c)),
                        x, nheavyatom, nH))))
        if CHECK_DEGENERACY:
            P0 = list(map(
                     lambda e, v, nc : construct_P(e, v, nc),
                     e0, v0, nocc))
        else:
            P0 = list(map(
                     lambda v, nc : 2.0*torch.matmul(v[:,:nc], v[:,:nc].transpose(0,1)),
                     v0, nocc))
        nmol = x.shape[0]
        norb = nheavyatom*4+nH
        e = torch.zeros(x.shape[:2], dtype=dtype, device=device)
        P = torch.zeros_like(x)
        for i in range(nmol):
            e[i,:norb[i]] = e0[i]
            P[i] = unpack(P0[i], nheavyatom[i], nH[i], x.shape[-1])
            
        e = e.reshape(x_orig_shape[0:3])
        v0 = tuple(map(lambda a, b : torch.stack((a, b), dim=0), v0[::2], v0[1::2]))
        P = P.reshape(x_orig_shape)
    else:#need to add large diagonal values to replace 0 padding
        #Gershgorin circle theorem estimate upper bounds of eigenvalues

        e0, v0 = list(zip(*list(map(
                        lambda a, b, c: sym_eigh(pack(a, b, c)),
                        x, nheavyatom, nH))))
        if CHECK_DEGENERACY:
            P0 = list(map(
                     lambda e, v, nc : construct_P(e, v, nc),
                     e0, v0, nocc))
        else:
            P0 = list(map(
                     lambda v, nc : 2.0*torch.matmul(v[:,:nc], v[:,:nc].transpose(0,1)),
                     v0, nocc))
        nmol = x.shape[0]
        norb = nheavyatom*4+nH
        e=torch.zeros(x.shape[:2], dtype=dtype, device=device)
        P = torch.zeros_like(x)
        for i in range(nmol):
            e[i,:norb[i]] = e0[i]
            P[i] = unpack(P0[i], nheavyatom[i], nH[i], x.shape[-1])

    if eig_only: return e, v0
    # each column of v is a eigenvectors
    # P_alpha_beta = 2.0 * |sum_i c_{i,alpha}*c_{i,beta}, i \in occupied MO
    return e, P, v0


def pytorch_symeig(A):
    """
    Effectively only a wrapper around pytorch's standard eigh
    (for testing, sanity checks, and comparison)
    """
    eival, eivec = torch.linalg.eigh(A, UPLO='U')
    return eival, eivec


class degen_symeig(torch.autograd.Function):
    """
    Eigensolver with custom backward for degenerate eigenvectors
    !!! ONLY APPLICABLE IF FINAL RESULT DOESN'T EXPLICITLY DEPEND
                   ON DEGENERATE EIGENVECTORS !!!
    !!! BEWARE: REMOVED SANITY CHECKS FOR PERFORMANCE REASONS !!!
    Based on idea in M.F. Kasim, arXiv:2011.04366 (2020)
    """
    @staticmethod
    def forward(ctx, A):
        eival, eivec = torch.linalg.eigh(A, UPLO='U')
        ctx.save_for_backward(eival, eivec)
        return eival, eivec

    @staticmethod
    def backward(ctx, grad_eival, grad_eivec):
        eival, eivec = ctx.saved_tensors
        eivecT = eivec.transpose(-2, -1).conj()
        if grad_eivec is None: return torch.zeros_like(eivec)
        delta = eival.unsqueeze(-2) - eival.unsqueeze(-1)
        # remove parallel and degenerate part of eigenvector deriv
        idx = torch.abs(delta) <= DEGEN_THRESHOLD
        delta[idx] = torch.inf
        delta_inv = delta.pow(-1)
        dC_proj = delta_inv * torch.matmul(eivecT, grad_eivec)

        # transform eigenvector and eigenvalue parts to A, sum up
        CdCCT = torch.matmul(eivec, torch.matmul(dC_proj, eivecT))
        CdLCT = torch.matmul(eivec, grad_eival.unsqueeze(-1) * eivecT)
        dA = CdLCT + CdCCT
        # symmetrize for increased stability
        dA = (dA + dA.transpose(-2, -1).conj()) * 0.5
        return dA
        


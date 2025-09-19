import torch
from .pack import *
from .packd import *
from .diag import *

# THIS IS AN OLD PROTOTYPE. IDEALLY, SHOULD BE CODED THE SAME WAY AS diag.py



#######################################################################################
#### DEGENERACY CHECK FOR open-shells and PM6 (truncd functions) is not tested!!!! ####
#######################################################################################

CHECK_DEGENERACY = False
# flag to control whether consider degeneracy when constructing the density matrix
# if no, then occupied orbitals are from 0, 1, ..., nocc-1
# if yes, then equal contributions are used for degenarated orbitals near fermi level



def sym_eig_truncd(x,nSuperHeavy,nheavyatom,nH,nocc, eig_only=False):
    '''
    eigenproblem solver for spd basis. Always used in pm6 model, despite the presence of d-functions.
    Called when propagation through SCF is not reguested, i.e. 'scf_backward' : 0.
    
    Input:
    x: Fock matrix
    nSuperHeavy: number of spd atoms
    nheavyatom: number of sp atoms
    nH: number of s atoms (hydrogens)
    Output:
    e: eigenvalues
    P: density matrix
    v: eigenvectors
    '''

    sym_eigh = degen_symeig.apply if DEGEN_EIGENSOLVER else pytorch_symeig

    dtype =  x.dtype
    device = x.device
    
    if x.dim()==2:
        e0, v = sym_eigh(packd(x, nSuperHeavy, nheavyatom, nH))
        e = torch.zeros((x.shape[0]),dtype=dtype,device=device)
        e[:(nheavyatom*4+nH+9*nSuperHeavy)] = e0
        
    elif x.dim()==4:
        nheavyatom = nheavyatom.repeat_interleave(2)
        nSuperHeavy = nSuperHeavy.repeat_interleave(2)
        nH = nH.repeat_interleave(2)
        nocc = nocc.flatten()
        #Gershgorin circle theorem estimate upper bounds of eigenvalues  
        x_orig_shape = x.size()
        
        x0 = packd(x, nSuperHeavy, nheavyatom, nH)
        nmol, size, _ = x0.shape

        aii = x0.diagonal(dim1=1,dim2=2)
        ri = torch.sum(torch.abs(x0),dim=2)-torch.abs(aii)
        hN = torch.max(aii+ri,dim=1)[0]
        dE = hN - torch.min(aii-ri,dim=1)[0] #(maximal - minimal) get range

        norb = nheavyatom*4+nH+nSuperHeavy*9
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
            if torch.isnan(x0).any():
                print(x0)
            print("Diagonalization failed")
            raise
        e = torch.zeros((nmol, x.shape[-1]),dtype=dtype,device=device)
        e[...,:size] = e0
        for i in range(nmol):
            if cond[i]:
                e[i,norb[i]:size] = 0.0
    
    
    
    else:#need to add large diagonal values to replace 0 padding
        #Gershgorin circle theorem estimate upper bounds of eigenvalues
        
        x0 = packd(x, nSuperHeavy, nheavyatom, nH)
        nmol, size, _ = x0.shape

        aii = x0.diagonal(dim1=1,dim2=2)
        ri = torch.sum(torch.abs(x0),dim=2)-torch.abs(aii)
        hN = torch.max(aii+ri,dim=1)[0]
        dE = hN - torch.min(aii-ri,dim=1)[0] #(maximal - minimal) get range
        
        norb = nheavyatom*4+nH+nSuperHeavy*9
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
            if torch.isnan(x0).any(): print('NaN problem\n',x0)
            e0, v = sym_eigh(x0)
            
        e = torch.zeros((nmol, x.shape[-1]),dtype=dtype,device=device)
        e[...,:size] = e0
        for i in range(nmol):
            if cond[i]:
                e[i,norb[i]:size] = 0.0

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
            
    P = unpackd(t, nSuperHeavy,nheavyatom, nH, x.shape[-1])
    
    if x.dim()==4:
        e = e.reshape(x_orig_shape[0:3])
        v = v.reshape(int(v.shape[0]/2),2,v.shape[1],v.shape[2])
        P = P.reshape(x_orig_shape)
        
    return e,P, v

def sym_eig_trunc1d(x,nSuperHeavy,nheavyatom,nH,nocc, eig_only=False):
    '''
    eigenproblem solver for spd basis. Always used in pm6 model, despite the presence of d-functions.
    Called when propagation through SCF is  reguested, i.e. 'scf_backward' : 2.
    
    Input:
    x: Fock matrix
    nSuperHeavy: number of spd atoms
    nheavyatom: number of sp atoms
    nH: number of s atoms (hydrogens)
    Output:
    e: eigenvalues
    P: density matrix
    v: eigenvectors
    '''

    sym_eigh = degen_symeig.apply if DEGEN_EIGENSOLVER else pytorch_symeig
    dtype =  x.dtype
    device = x.device
    if x.dim()==2:
        e0, V = sym_eigh(packd(x, nSuperHeavy, nheavyatom, nH))
        e = torch.zeros((x.shape[0]),dtype=dtype,device=device)
        e[:(nheavyatom*4+nH+nSuperHeavy*9)] = e0
        
    elif x.dim()==4:#need to add large diagonal values to replace 0 padding
        #Gershgorin circle theorem estimate upper bounds of eigenvalues
        nSuperHeavy = nSuperHeavy.repeat_interleave(2)
        nheavyatom = nheavyatom.repeat_interleave(2)
        nH = nH.repeat_interleave(2)
        nocc = nocc.flatten()
        x_orig_shape = x.size()
        x = x.flatten(start_dim=0, end_dim=1)
        
        e0, v0 = list(zip(*list(map(
                        lambda a,b,c,d: sym_eigh(packd(a,b,c,d)),
                        x,nSuperHeavy,nheavyatom, nH))))
        if CHECK_DEGENERACY:
            P0 = list(map(
                     lambda e, v, nc : construct_P(e,v,nc),
                     e0, v0,nocc))
        else:
            P0 = list(map(
                     lambda v, nc : 2.0*torch.matmul(v[:,:nc], v[:,:nc].transpose(0,1)),
                     v0,nocc))

        nmol = x.shape[0]
        norb = nheavyatom*4+nH+nSuperHeavy*9
        e=torch.zeros(x.shape[:2], dtype=dtype, device=device)
        P = torch.zeros_like(x)
        for i in range(nmol):
            e[i,:norb[i]] = e0[i]
            P[i] = unpackd(P0[i], nSuperHeavy[i], nheavyatom[i], nH[i], x.shape[-1])
            
        e = e.reshape(x_orig_shape[0:3])
        v0 = tuple(map(lambda a, b : torch.stack((a,b), dim=0), v0[::2], v0[1::2]))
        V = v0
        P = P.reshape(x_orig_shape)
    
    else:#need to add large diagonal values to replace 0 padding
        #Gershgorin circle theorem estimate upper bounds of eigenvalues

        e0, v0 = list(zip(*list(map(
                        lambda a,b,c,d: sym_eigh(packd(a,b,c,d)),
                        x,nSuperHeavy,nheavyatom, nH))))
        if CHECK_DEGENERACY:
            P0 = list(map(
                     lambda e, v, nc : construct_P(e,v,nc),
                     e0, v0,nocc))
        else:
            P0 = list(map(
                     lambda v, nc : 2.0*torch.matmul(v[:,:nc], v[:,:nc].transpose(0,1)),
                     v0,nocc))
        
        #
        nmol = x.shape[0]
        norb = nheavyatom*4+nH+9*nSuperHeavy
        e=torch.zeros(x.shape[:2], dtype=dtype, device=device)
        P = torch.zeros_like(x)
        for i in range(nmol):
            e[i,:norb[i]] = e0[i]
            P[i] = unpackd(P0[i], nSuperHeavy[i], nheavyatom[i], nH[i], x.shape[-1])

        norb = nheavyatom*4 + nH+9*nSuperHeavy
        nmax = int(torch.max(norb))        # the largest “norb[i]” over all molecules
        V = torch.zeros((nmol, nmax, nmax), dtype=dtype, device=device)

        for i in range(nmol):
            V[i, :norb[i], :norb[i]] = v0[i]

    if eig_only: return e, V
    # each column of v is a eigenvectors
    # P_alpha_beta = 2.0 * |sum_i c_{i,alpha}*c_{i,beta}, i \in occupied MO

    return e,P, V







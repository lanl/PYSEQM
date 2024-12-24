import torch
from seqm.seqm_functions.pack import packone, unpackone


def rcis(mol, w, e_mo, nroots):
    torch.set_printoptions(linewidth=200)
    """TODO: Docstring for rcis.

    :param mol: Molecule Orbital Coefficients
    :param w: 2-electron integrals
    :param nroots: Number of CIS states requested
    :returns: CIS excited state energies and CIS eigenvectors

    """

    device = w.device
    dtype = w.dtype
    # I'm going to build the full A matrix

    # TODO: norb, nvirt should be members of the Molecule class
    # Have to be careful here while batching because I assume norb = norb[0]. So all other molecules should in a batch of molecules
    # should have the same norb
    nHeavy = mol.nHeavy
    nHydro = mol.nHydro
    norb = nHydro + 4 * nHeavy
    norb = norb[0]
    nocc = mol.nocc[0]
    nvirt = norb - nocc
    nov = nocc * nvirt
    nroots = nov

    # TODO: need to batch over many molecules at once. 
    # Right now I assume that we have only 1 molecule I'll use this index to get the data for the 1st molecule
    mol0=0
    # ea_ei contains the list of orbital energy difference between the virtual and occupied orbitals
    ea_ei = e_mo[mol0,nocc:norb].unsqueeze(0)-e_mo[mol0,:nocc].unsqueeze(1)

    # Make the davidson guess vectors 
    _, sortedidx = torch.sort(ea_ei.view(nov), stable=True, descending=False) # stable to preserve order of degenerate

    # TODO: If the last chosen root was degenerate in ea_ei, then expand the subspace to include all the degenerate roots

    maxSubspacesize = nov # TODO: User-defined/fixed
    nStart = nroots # starting subspace size
    V = torch.zeros(maxSubspacesize,nov,device=device,dtype=dtype)
    HV = torch.empty_like(V)
    V[torch.arange(nStart),sortedidx[:nStart]] = 1.0

    max_iter = maxSubspacesize//nroots # TODO: User-defined/fixed
    converged = False
    iter = 0
    vstart = 0
    vend = nroots 

    while iter < max_iter and not converged: # Davidson loop
        
        if iter > 0: # skip first step, as initial V is orthonormal
            V = orthogonalize(V,vstart,vend)

        HV[vstart:vend,:] = matrix_vector_product(mol,V, w, ea_ei, 0, maxSubspacesize)
        H = torch.einsum('nia,ria->nr',V[:vend,:].view(vend,nocc,nvirt),HV[:vend,:].view(vend,nocc,nvirt))

        iter = iter + 1

        # Diagonalize the subspace hamiltonian
        r_eval, r_evec = torch.linalg.eigh(H) # find eigenvalues and eigenvectors
        # r_eval = r_eval.real
        # r_evec = r_evec.real
        r_eval, r_idx = torch.sort(r_eval, descending=False) # sort eigenvalues in ascending order
        r_evec = r_evec[:, r_idx] # sort eigenvectors accordingly
        e_val_n = r_eval[:vend] # keep only the lowest keep_n eigenvalues; full are still stored as e_val
        e_vec_n = r_evec[:, :vend]

        # TODO: better to allocate memory for residual outside the loop instead of making it anew every time in the loop?
        residual = torch.einsum('vr,vo->ro',e_vec_n, HV[:vend,:]) - torch.einsum('vr,vo->ro',e_vec_n,V[:vend,:])*e_val_n.unsqueeze(1)
        resid_norm = torch.norm(residual,dim=1)
        print(f"DEBUGPRINT[21]: rcis.py:181: e={e_val_n}")

def matrix_vector_product(mol, V, w, ea_ei, vstart, vend):
    # C: Molecule Orbital Coefficients
    C = mol.eig_vec
    device = C.device
    dtype = C.dtype

    mask  = mol.mask
    maskd = mol.maskd
    mask_l = mol.mask_l
    idxi  = mol.idxi
    idxj  = mol.idxj
    nmol  = mol.nmol
    molsize = mol.molsize
    nHeavy = mol.nHeavy
    nHydro = mol.nHydro

    # From here I assume there is only one molecule. I'll worry about batching later
    if (nmol != 1):
        raise Exception("Not yet implemented for more than one molecule")

    # TODO: norb, nvirt should be members of the Molecule class
    # Have to be careful here while batching because I assume norb = norb[0]. So all other molecules should in a batch of molecules
    # should have the same norb
    norb = nHydro + 4 * nHeavy
    norb = norb[0]
    nocc = mol.nocc[0]
    nvirt = norb - nocc
    nov = nocc,nvirt

    # TODO: I'm using my knowledge from implementing CIS in Q-Chem where memory is column major and not row major.
    # In Q-Chem it made sense to have the guess vector V laid out as V_ai instead of V_ia. Check which is
    # better for pytorch.
    # I'll try V_ia now
    nnewRoots = vend-vstart
    Via = V[vstart:vend+1,:].view(-1, nocc, nvirt)
    # print(f"DEBUGPRINT[16]: rcis.py:31: V={V}")
    # P_xi = C[:,:,nocc:] @ Vai @ C[:,:,:nocc].transpose(1,2)
    # This below line will fail if the molecues in the batch have different norb, nocc etc
    # CV = torch.einsum('nab,nrbc->nrac', C[:, :, nocc:], Vai)
    # P_xi = torch.einsum('nrac,ncd->nrad', CV, C[:, :, :nocc].transpose(1, 2))
    # del CV
    # VC = torch.einsum('ria,ma->rim',  Via, C[0, :, nocc:])
    # P_xi = torch.einsum('li,rim->rlm', C[0, :, :nocc],VC)
    # del VC
    P_xi = torch.einsum('mi,ria,na->rmn', C[0, :, :nocc],Via, C[0, :, nocc:])
    print(f"C is {C.shape}")
    print(f"Pxi is {P_xi.shape}")
    # print(f"DEBUGPRINT[17]: rcis.py:35: P_xi={P_xi}")
    # P0 = unpack(P_xi, mol.nHeavy, mol.nHydro, mol.molsize*4) #
    P0 = torch.stack([
        unpackone(P_xi[i], 4*nHeavy[i // nnewRoots], nHydro[i // nnewRoots], molsize * 4)
        for i in range(nnewRoots * nmol)
    ]).view(nmol*nnewRoots, molsize * 4, molsize * 4)
    # print(f"DEBUGPRINT[18]: rcis.py:43: P0={P0}")

    # Compute the (ai||jb)X_jb

    print("Debugging: Symmetrizing P0")
    P0_sym = 0.5*(P0 + P0.transpose(1,2))

    P = P0_sym.reshape(nnewRoots,molsize,4,molsize,4)\
              .transpose(2,3).reshape(nnewRoots,molsize*molsize,4,4)
    del P0_sym
    F = torch.zeros_like(P)

    # Two center-two elecron integrals
    weight = torch.tensor([1.0,
                           2.0, 1.0,
                           2.0, 2.0, 1.0,
                           2.0, 2.0, 2.0, 1.0],dtype=dtype, device=device).reshape((-1,10))

    # Pdiag_symmetrized = P[:,maskd]+P[:,maskd].transpose(2,3)
    # weight *= 0.5 # dividing by 2 because I didn't do it while making the symmetrized P matrix
    Pdiag_symmetrized = P[:,maskd]

    PA = (Pdiag_symmetrized[:,idxi][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).unsqueeze(-1)
    PB = (Pdiag_symmetrized[:,idxj][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).unsqueeze(-2)

    #suma \sum_{mu,nu \in A} P_{mu, nu in A} (mu nu, lamda sigma) = suma_{lambda sigma \in B}
    #suma shape (npairs, 10)
    suma = torch.sum(PA*w.unsqueeze(0),dim=2)
    #sumb \sum_{l,s \in B} P_{l, s inB} (mu nu, l s) = sumb_{mu nu \in A}
    #sumb shape (npairs, 10)
    sumb = torch.sum(PB*w.unsqueeze(0),dim=3)
    #reshape back to (npairs 4,4)
    # as will use index add in the following part

    sumA = torch.zeros(PA.shape[0],w.shape[0],4,4,dtype=dtype, device=device)
    sumB = torch.zeros_like(sumA)
    sumA[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = suma
    sumB[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = sumb
    sumA.add_(sumA.triu(1).transpose(2,3))
    sumB.add_(sumB.triu(1).transpose(2,3))
    #F^A_{mu, nu} = Hcore + \sum^A + \sum_{B} \sum_{l, s \in B} P_{l,s \in B} * (mu nu, l s)
    #\sum_B
    F.index_add_(1,maskd[idxi],sumB)
    #\sum_A
    F.index_add_(1,maskd[idxj],sumA)


    # off diagonal block part, check KAB in forck2.f
    # mu, nu in A
    # lambda, sigma in B
    # F_mu_lambda = Hcore - 0.5* \sum_{nu \in A} \sum_{sigma in B} P_{nu, sigma} * (mu nu, lambda, sigma)
    
    sumK = torch.empty(nnewRoots,w.shape[0],4,4,dtype=dtype, device=device)

    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #   0,     1         2       3       4         5       6      7         8        9
    ind = torch.tensor([[0,1,3,6],
                        [1,2,4,7],
                        [3,4,5,8],
                        [6,7,8,9]],dtype=torch.int64, device=device)
    # Pp =P[mask], P_{mu \in A, lambda \in B}
    Pp = -0.5*P[:,mask]
    for i in range(4):
        for j in range(4):
            #\sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            sumK[...,i,j] = torch.sum(Pp*w[...,ind[i],:][...,:,ind[j]].unsqueeze(0),dim=(2,3))
    F.index_add_(1,mask,sumK)
    
    Pptot = P[...,1,1]+P[...,2,2]+P[...,3,3]

    # One center-two electron integrals
    gss = mol.parameters['g_ss'] 
    gsp = mol.parameters['g_sp']
    gpp = mol.parameters['g_pp']
    gp2 = mol.parameters['g_p2']
    hsp = mol.parameters['h_sp']  
    
    F2e1c = torch.zeros(nnewRoots,maskd.shape[0],4,4,device=device,dtype=dtype)

    # F2e1c[...,0,0] = 0.5*P[:,maskd,0,0]*gss.unsqueeze(0) + Pptot[:,maskd]*(gsp-0.5*hsp).unsqueeze(0)
    # for i in range(1,4):
    #     #(p,p)
    #     F2e1c[...,i,i] = P[:,maskd,0,0]*(gsp-0.5*hsp).unsqueeze(0) + 0.5*P[:,maskd,i,i]*gpp.unsqueeze(0) \
    #             + (Pptot[:,maskd] - P[:,maskd,i,i]) * (1.25*gp2-0.25*gpp).unsqueeze(0)
    #     #(s,p) = (p,s) upper triangle
    #     F2e1c[...,0,i] = P[:,maskd,i,0]*(0.5*hsp).unsqueeze(0) + P[:,maskd,0,i]*(hsp - 0.5*gsp).unsqueeze(0)
    #     F2e1c[...,i,0] = P[:,maskd,0,i]*(0.5*hsp).unsqueeze(0) + P[:,maskd,i,0]*(hsp - 0.5*gsp).unsqueeze(0)
    # #(p,p*)
    # for i,j in [(1,2),(1,3),(2,3),(2,1),(3,1),(3,2)]:
    #     F2e1c[...,i,j] = P[:,maskd,i,j]* (0.5*gpp - gp2).unsqueeze(0) + P[:,maskd,j,i]* (0.25*gpp - 0.25*gp2).unsqueeze(0)

    F2e1c[...,0,0] = 0.5*P[:,maskd,0,0]*gss.unsqueeze(0) + Pptot[:,maskd]*(gsp-0.5*hsp).unsqueeze(0)
    for i in range(1,4):
        #(p,p)
        F2e1c[...,i,i] = P[:,maskd,0,0]*(gsp-0.5*hsp).unsqueeze(0) + 0.5*P[:,maskd,i,i]*gpp.unsqueeze(0) \
                + (Pptot[:,maskd] - P[:,maskd,i,i]) * (1.25*gp2-0.25*gpp).unsqueeze(0)
        #(s,p) = (p,s) upper triangle
        F2e1c[...,0,i] = P[:,maskd,0,i]*(1.5*hsp - 0.5*gsp).unsqueeze(0)
    #(p,p*)
    for i,j in [(1,2),(1,3),(2,3)]:
        F2e1c[...,i,j] = P[:,maskd,i,j]* (0.75*gpp - 1.25*gp2).unsqueeze(0)

    # F.add_(F2e1c)
    F2e1c.add_(F2e1c.triu(1).transpose(2,3))
    F[:,maskd] += F2e1c

    F[:,mask_l] += sumK.transpose(2,3)
    
    # Gsymao = F.reshape(nroots,molsize,molsize,4,4) \
    #          .transpose(2,3) \
    #          .reshape(nroots, 4*molsize, 4*molsize)
    #
    # Gsymao = torch.stack([
    #     packone(Gsymao[i], 4*nHeavy[i // nroots], nHydro[i // nroots], norb)
    #     for i in range(nroots * nmol)
    # ])

    P0_antisym = 0.5*(P0 - P0.transpose(1,2))
    P_anti = P0_antisym.reshape(nnewRoots,molsize,4,molsize,4)\
              .transpose(2,3).reshape(nnewRoots,molsize*molsize,4,4)
    del P0_antisym

    Pp = -0.5*P_anti[:,mask]
    for i in range(4):
        for j in range(4):
            #\sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            sumK[...,i,j] = torch.sum(Pp*w[...,ind[i],:][...,:,ind[j]].unsqueeze(0),dim=(2,3))
    F.index_add_(1,mask,sumK)
    F[:,mask_l] -= sumK.transpose(2,3)
    
    F2e1c.zero_()
    for i in range(1,4):
        #(s,p) = (p,s) upper triangle
        F2e1c[...,0,i] = P_anti[:,maskd,0,i]*(0.5*hsp - 0.5*gsp).unsqueeze(0)
    #(p,p*)
    for i,j in [(1,2),(1,3),(2,3)]:
        F2e1c[...,i,j] = P_anti[:,maskd,i,j]* (0.25*gpp - 0.75*gp2).unsqueeze(0)

    F2e1c.add_(F2e1c.triu(1).transpose(2,3),alpha=-1.0)
    F[:,maskd] += F2e1c

    # dummy = torch.zeros_like(F); dummy.index_add_(1,mask,sumK);dummy[:,mask_l]-= sumK.transpose(2,3);dummy[:,maskd] += F2e1c;
    # dummy = dummy.reshape(nroots,molsize,molsize,4,4) \
    #          .transpose(2,3) \
    #          .reshape(nroots, 4*molsize, 4*molsize)[6]*2
    # dummy = packone(dummy,4,2,6)
    
    F0 = F.reshape(nnewRoots,molsize,molsize,4,4) \
             .transpose(2,3) \
             .reshape(nnewRoots, 4*molsize, 4*molsize)
    del F

    F0 = torch.stack([
        packone(F0[i], 4*nHeavy[i // nnewRoots], nHydro[i // nnewRoots], norb)
        for i in range(nnewRoots * nmol)
    ])

    # since I assume that we have only 1 molecule I'll use this index to get the data for the 1st molecule
    mol0 = 0

    # CAao = torch.einsum('ma,rmn->ran', C[mol0, :, nocc:], F0)
    # A = torch.einsum('ran,ni->rai',CAao,C[mol0,:, :nocc ])
    # CAao = torch.einsum('mi,rmn->rin', C[mol0, :, :nocc], F0)
    # A = torch.einsum('rin,na->ria',CAao,C[mol0,:, nocc: ])*2.0
    # TODO: why am I multiplying by 2?
    A = torch.einsum('mi,rmn,na->ria', C[mol0, :, :nocc], F0,C[mol0,:, nocc: ])*2.0

    # Add (ea-ei) to the diagonal of A
    # ea_ei = e_mo[mol0,nocc:norb].unsqueeze(1)-e_mo[mol0,:nocc].unsqueeze(0)
    # A += Vai[mol0]*ea_ei.unsqueeze(0)
    # A = torch.einsum('nai,rai->nr',Vai[mol0],A)
    A += Via*ea_ei.unsqueeze(0)

    return A.view(nnewRoots,-1)
    
def orthogonalize(V,vstart,vend):
    """TODO: Docstring for orthogonalize.

    :arg1: TODO
    :returns: TODO

    """
    return V

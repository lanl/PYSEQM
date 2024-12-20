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
    # Calculate the A matrix

    # TODO: norb, nvirt should be members of the Molecule class
    # Have to be careful here while batching because I assume norb = norb[0]. So all other molecules should in a batch of molecules
    # should have the same norb
    norb = nHydro + 4 * nHeavy
    norb = norb[0]
    print(f"DEBUGPRINT[9]: rcis.py:21: norb={norb}")
    nocc = mol.nocc[0]
    print(f"DEBUGPRINT[13]: rcis.py:23: nocc={nocc}")
    nvirt = norb - nocc
    nov = nocc * nvirt
    print(f"DEBUGPRINT[14]: rcis.py:25: nvirt={nvirt}")
    nroots = nov
    print(f"DEBUGPRINT[15]: rcis.py:27: nroots={nroots}")

    # I'm going to build the full A matrix
    # TODO: I'm using my knowledge from implementing CIS in Q-Chem where memory is column major and not row major.
    # In Q-Chem it made sense to have the guess vector V laid out as V_ai instead of V_ia. Check which is
    # better for pytorch
    V = torch.eye(nroots).unsqueeze(0).repeat(nmol, 1, 1)
    Vai = V.reshape(nmol, nroots, nvirt, nocc)
    # print(f"DEBUGPRINT[16]: rcis.py:31: V={V}")
    # P_xi = C[:,:,nocc:] @ Vai @ C[:,:,:nocc].transpose(1,2)
    # This below line will fail if the molecues in the batch have different norb, nocc etc
    CV = torch.einsum('nab,nrbc->nrac', C[:, :, nocc:], Vai)
    P_xi = torch.einsum('nrac,ncd->nrad', CV, C[:, :, :nocc].transpose(1, 2))
    del CV
    P_xi = P_xi.reshape(nmol * nroots, norb, norb)
    print(f"C is {C.shape}")
    print(f"Pxi is {P_xi.shape}")
    # print(f"DEBUGPRINT[17]: rcis.py:35: P_xi={P_xi}")
    # P0 = unpack(P_xi, mol.nHeavy, mol.nHydro, mol.molsize*4) #
    P0 = torch.stack([
        unpackone(P_xi[i], 4*nHeavy[i // nroots], nHydro[i // nroots], molsize * 4)
        for i in range(nroots * nmol)
    ]).reshape(nmol, nroots, molsize * 4, molsize * 4)
    # print(f"DEBUGPRINT[18]: rcis.py:43: P0={P0}")

    # Compute the (ai||jb)X_jb
    # From here I assume there is only one molecule. I'll worry about batching later
    if (nmol != 1):
        raise Exception("Not yet implemented for more than one molecule")
    P0 = P0[0, ...]

    print("Debugging: Symmetrizing P0")
    P0_sym = 0.5*(P0 + P0.transpose(1,2))

    P = P0_sym.reshape(nroots,molsize,4,molsize,4)\
              .transpose(2,3).reshape(nroots,molsize*molsize,4,4)
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
    # sumA.add_(sumA.triu(1).transpose(2,3))
    # sumB.add_(sumB.triu(1).transpose(2,3))
    #F^A_{mu, nu} = Hcore + \sum^A + \sum_{B} \sum_{l, s \in B} P_{l,s \in B} * (mu nu, l s)
    #\sum_B
    F.index_add_(1,maskd[idxi],sumB)
    #\sum_A
    F.index_add_(1,maskd[idxj],sumA)


    # off diagonal block part, check KAB in forck2.f
    # mu, nu in A
    # lambda, sigma in B
    # F_mu_lambda = Hcore - 0.5* \sum_{nu \in A} \sum_{sigma in B} P_{nu, sigma} * (mu nu, lambda, sigma)
    
    sumK = torch.empty(nroots,w.shape[0],4,4,dtype=dtype, device=device)

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
    
    TMP = torch.zeros(nroots,maskd.shape[0],4,4,device=device,dtype=dtype)

    # TMP[...,0,0] = 0.5*P[:,maskd,0,0]*gss.unsqueeze(0) + Pptot[:,maskd]*(gsp-0.5*hsp).unsqueeze(0)
    # for i in range(1,4):
    #     #(p,p)
    #     TMP[...,i,i] = P[:,maskd,0,0]*(gsp-0.5*hsp).unsqueeze(0) + 0.5*P[:,maskd,i,i]*gpp.unsqueeze(0) \
    #             + (Pptot[:,maskd] - P[:,maskd,i,i]) * (1.25*gp2-0.25*gpp).unsqueeze(0)
    #     #(s,p) = (p,s) upper triangle
    #     TMP[...,0,i] = P[:,maskd,i,0]*(0.5*hsp).unsqueeze(0) + P[:,maskd,0,i]*(hsp - 0.5*gsp).unsqueeze(0)
    #     TMP[...,i,0] = P[:,maskd,0,i]*(0.5*hsp).unsqueeze(0) + P[:,maskd,i,0]*(hsp - 0.5*gsp).unsqueeze(0)
    # #(p,p*)
    # for i,j in [(1,2),(1,3),(2,3),(2,1),(3,1),(3,2)]:
    #     TMP[...,i,j] = P[:,maskd,i,j]* (0.5*gpp - gp2).unsqueeze(0) + P[:,maskd,j,i]* (0.25*gpp - 0.25*gp2).unsqueeze(0)

    TMP[...,0,0] = 0.5*P[:,maskd,0,0]*gss.unsqueeze(0) + Pptot[:,maskd]*(gsp-0.5*hsp).unsqueeze(0)
    for i in range(1,4):
        #(p,p)
        TMP[...,i,i] = P[:,maskd,0,0]*(gsp-0.5*hsp).unsqueeze(0) + 0.5*P[:,maskd,i,i]*gpp.unsqueeze(0) \
                + (Pptot[:,maskd] - P[:,maskd,i,i]) * (1.25*gp2-0.25*gpp).unsqueeze(0)
        #(s,p) = (p,s) upper triangle
        TMP[...,0,i] = P[:,maskd,0,i]*(1.5*hsp - 0.5*gsp).unsqueeze(0)
    #(p,p*)
    for i,j in [(1,2),(1,3),(2,3)]:
        TMP[...,i,j] = P[:,maskd,i,j]* (0.75*gpp - 1.25*gp2).unsqueeze(0)

    # F.add_(TMP)
    TMP.add_(TMP.triu(1).transpose(2,3))
    F[:,maskd] += TMP

    F[:,mask_l] += sumK.transpose(2,3)
    
    P0_antisym = 0.5*(P0 - P0.transpose(1,2))
    P_anti = P0_antisym.reshape(nroots,molsize,4,molsize,4)\
              .transpose(2,3).reshape(nroots,molsize*molsize,4,4)
    del P0_antisym

    Pp = 0.5*P_anti[:,mask]
    for i in range(4):
        for j in range(4):
            #\sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            sumK[...,i,j] = torch.sum(Pp*w[...,ind[i],:][...,:,ind[j]].unsqueeze(0),dim=(2,3))
    F.index_add_(1,mask,sumK)
    F[:,mask_l] -= sumK.transpose(2,3)
    
    TMP.zero_()
    for i in range(1,4):
        #(s,p) = (p,s) upper triangle
        TMP[...,0,i] = P_anti[:,maskd,0,i]*(0.5*hsp - 0.5*gsp).unsqueeze(0)
    #(p,p*)
    for i,j in [(1,2),(1,3),(2,3)]:
        TMP[...,i,j] = P_anti[:,maskd,i,j]* (0.25*gpp - 0.75*gp2).unsqueeze(0)

    TMP.add_(TMP.triu(1).transpose(2,3),alpha=-1.0)
    F[:,maskd] += TMP

    dummy = torch.zeros_like(F); dummy.index_add_(1,mask,sumK);dummy[:,mask_l]-= sumK.transpose(2,3);dummy[:,maskd] += TMP;
    dummy = dummy.reshape(nroots,molsize,molsize,4,4) \
             .transpose(2,3) \
             .reshape(nroots, 4*molsize, 4*molsize)[3]*2
    dummy = packone(dummy,4,2,6)
    
    F0 = F.reshape(nroots,molsize,molsize,4,4) \
             .transpose(2,3) \
             .reshape(nroots, 4*molsize, 4*molsize)
    del F

    F0 = torch.stack([
        packone(F0[i], 4*nHeavy[i // nroots], nHydro[i // nroots], norb)
        for i in range(nroots * nmol)
    ])

    # since I assume that we have only 1 molecule I'll use this index to get the data for the 1st molecule
    mol0 = 0

    CAao = torch.einsum('ma,rmn->ran', C[mol0, :, nocc:], F0)
    A = torch.einsum('ran,ni->rai',CAao,C[mol0,:, :nocc ])

    # Add (ea-ei) to the diagonal of A
    ea_ei = e_mo[mol0,nocc:norb].unsqueeze(1)-e_mo[mol0,:nocc].unsqueeze(0)
    A += Vai[mol0]*ea_ei.unsqueeze(0)

    A = torch.einsum('nai,rai->nr',Vai[mol0],A)
    print(f"Diff in symmetry of A is {torch.sum(A-A.transpose(0,1))}")

    print(f"DEBUGPRINT[20]: rcis.py:177: A={A}")

    # Diagonalize the A matrix
    e, v = torch.linalg.eigh(A) 
    print(f"DEBUGPRINT[21]: rcis.py:181: e={e}")
     

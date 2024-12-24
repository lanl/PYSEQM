import torch
from seqm.seqm_functions.pack import packone, unpackone


def rcis(mol, w, e_mo, nroots):
    torch.set_printoptions(linewidth=200)
    """Calculate the restricted Configuration Interaction Single (RCIS) excitation energies and amplitudes
       using davidson diagonalization

    :param mol: Molecule Orbital Coefficients
    :param w: 2-electron integrals
    :param e_mo: Orbital energies
    :param nroots: Number of CIS states requested
    :returns: 

    """

    device = w.device
    dtype = w.dtype

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
    # nroots = nov

    # TODO: need to batch over many molecules at once. 
    # Right now I assume that we have only 1 molecule I'll use this index to get the data for the 1st molecule
    mol0=0
    # ea_ei contains the list of orbital energy difference between the virtual and occupied orbitals
    ea_ei = e_mo[mol0,nocc:norb].unsqueeze(0)-e_mo[mol0,:nocc].unsqueeze(1)
    approxH = ea_ei.view(nov)

    # Make the davidson guess vectors 
    sorted_ediff, sortedidx = torch.sort(approxH, stable=True, descending=False) # stable to preserve the order of degenerate orbitals

    nroots_expand = nroots
    # If the last chosen root was degenerate in ea_ei, then expand the subspace to include all the degenerate roots
    while nroots_expand < len(sorted_ediff) and (sorted_ediff[nroots_expand] - sorted_ediff[nroots_expand-1]) < 1e-5:
        nroots_expand += 1
    
    if nroots_expand>nroots:
        print(f"More roots will be calculated because of degeneracy in the MOs. NRoots changed from {nroots} to {nroots_expand}")
        nroots = nroots_expand

    maxSubspacesize = getMaxSubspacesize(dtype,device,nov) # TODO: User-defined
    nStart = nroots # starting subspace size
    V = torch.zeros(maxSubspacesize,nov,device=device,dtype=dtype)
    HV = torch.empty_like(V)
    V[torch.arange(nStart),sortedidx[:nStart]] = 1.0

    max_iter = 3*maxSubspacesize//nroots # Heuristic: allow one or two subspace collapse. TODO: User-defined
    root_tol = 1e-6 # TODO: User-defined/fixed
    iter = 0
    vstart = 0
    vend = nroots 

    # TODO: Test if orthogonal or nonorthogonal version is more efficient
    nonorthogonal = True # TODO: User-defined/fixed

    while iter < max_iter: # Davidson loop
        
        HV[vstart:vend,:] = matrix_vector_product(mol,V, w, ea_ei, vstart, vend)
        # Make H by multiplying V.T * HV
        # Option 1: direct multiplication
        # H = torch.einsum('nia,ria->nr',V[:vend,:].view(vend,nocc,nvirt),HV[:vend,:].view(vend,nocc,nvirt))

        # Option 2: avoid redundant multipication by only forming the new blocks of H coming from the guess vectors V[vstart:vend]
        H = torch.empty(vend,vend,device=device,dtype=dtype)
        if vstart == 0:
            H[0:vend,0:vend] = torch.einsum('nia,ria->nr',V[:vend,:].view(vend,nocc,nvirt),HV[:vend,:].view(vend,nocc,nvirt))
        else:
            H[:vstart,:vstart] = Hold
            H[vstart:vend,vstart:vend] = torch.einsum('nia,ria->nr',V[vstart:vend,:].view(vend-vstart,nocc,nvirt),HV[vstart:vend,:].view(vend-vstart,nocc,nvirt)) 
            H[vstart:vend,:vstart] = torch.einsum('nia,ria->nr',V[vstart:vend,:].view(vend-vstart,nocc,nvirt),HV[0:vstart,:].view(vstart,nocc,nvirt)) 
            H[:vstart,vstart:vend] = H[vstart:vend,:vstart].T
        Hold = H[:vend,:vend]

        # TODO: Check if Option 2 is necessary or go with Option 1 (much more simple and readable) by testing on big molecules 

        iter = iter + 1

        # Diagonalize the subspace hamiltonian
        e_val_n, e_vec_n = get_subspace_eig(H,nroots,V,vend,nonorthogonal)

        amplitudes = torch.einsum('vr,vo->ro',e_vec_n,V[:vend,:])
        residual = torch.einsum('vr,vo->ro',e_vec_n, HV[:vend,:]) - amplitudes*e_val_n.unsqueeze(1)

        resid_norm = torch.norm(residual,dim=1)
        # print(f"DEBUGPRINT[23]: rcis.py:95: resid_norm={resid_norm}")

        roots_not_converged = resid_norm > root_tol
        n_not_converged = roots_not_converged.sum()

        print(f"Iteration {iter}: Found {nroots-n_not_converged}/{nroots} states, Total Error: {torch.sum(resid_norm):.4e}")

        if n_not_converged == 0:
            break

        if n_not_converged + vend > maxSubspacesize:
            #  collapse the subspace
            print("Maximum subspace size reached, increase the subspace size. Collapsing subspace")
            V[:nroots,:] = amplitudes
            # HV[:nroots,:] = torch.einsum('vr,vo->ro',e_vec_n, HV[:vend,:]) 
            vstart = 0
            vend = nroots
            continue

        newsubspace = residual[roots_not_converged,:]/(e_val_n[roots_not_converged].unsqueeze(1) - approxH.unsqueeze(0))

        vstart = vend
        if nonorthogonal:
            newsubspace_norm = torch.norm(newsubspace,dim=1)
            nonzero_newsubspace = newsubspace_norm > root_tol
            vend = vstart + nonzero_newsubspace.sum()
            V[vstart:vend] = newsubspace[nonzero_newsubspace]/newsubspace_norm[nonzero_newsubspace].unsqueeze(1)

        else:
            vend = orthogonalize_to_current_subspace(V, newsubspace, vend, root_tol)

        if vstart == vend:
            print('No new vectors to be added to the subspace because the new search directions are zero after orthonormalization')
            raise Exception("Roots have not convered")

        if iter == max_iter:
            print("Maximum iterations reached but roots not converged")

    print("")
    for i, energy in enumerate(e_val_n, start=1):
        print(f"  State {i}: {energy:.6f} eV")


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

    # TODO: I'm using my knowledge from implementing CIS in Q-Chem where memory is column major and not row major.
    # In Q-Chem it made sense to have the guess vector V laid out as V_ai instead of V_ia. Check which is
    # better for pytorch.
    # I'll try V_ia now
    nnewRoots = vend-vstart
    Via = V[vstart:vend,:].view(-1, nocc, nvirt)
    P_xi = torch.einsum('mi,ria,na->rmn', C[0, :, :nocc],Via, C[0, :, nocc:])

    P0 = torch.stack([
        unpackone(P_xi[i], 4*nHeavy[i // nnewRoots], nHydro[i // nnewRoots], molsize * 4)
        for i in range(nnewRoots * nmol)
    ]).view(nmol*nnewRoots, molsize * 4, molsize * 4)

    # Compute the (ai||jb)X_jb

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

    # TODO: why am I multiplying by 2?
    A = torch.einsum('mi,rmn,na->ria', C[mol0, :, :nocc], F0,C[mol0,:, nocc: ])*2.0

    A += Via*ea_ei.unsqueeze(0)

    return A.view(nnewRoots,-1)
    
def orthogonalize_to_current_subspace(V, newsubspace, vend, tol):
    """Orthogonalizes the vectors in newsubspace against the original subspace in V
       with Gram-Schmidt orthogonalization. We cannot use Modified-Gram-Schmidt because 
       we want to keep the original subspace vectors the same

    :V: Original subspace vectors (with pre-allocated memory for new vectors)
    :newsubspace: vectors that have to be orthonormalized
    :vend: original subspace size
    :tol: the tolerance for the norm of new vectors below which the vector will be discarded
    :returns: vend: size of the subspace after adding in the new vectors 

    """
    for i in range(newsubspace.shape[0]):
        vec = newsubspace[i]
        # Instead of batch processing like below, it is more numerically stable to do it one by one 
        # because vec is changed at each step
        # projection = V[:vend] @ vec
        # vec -= projection @ V[:vend]
        for j in range(vend):
            vec -= torch.dot(V[j], vec) * V[j]

        vecnorm = torch.norm(vec)
        
        if vecnorm > tol:
            vec /= vecnorm

            # reorthogonalize because dividing by the norm can make it numerically non-orthogonal
            # projection = V[:vend] @ vec
            # vec -= projection @ V[:vend]
            for j in range(vend):
                vec -= torch.dot(V[j], vec) * V[j]
            vecnorm = torch.norm(vec)
            
            if vecnorm > tol:
                V[vend] = vec / vecnorm
                vend = vend + 1

    return vend

import psutil # to get the memory size

def getMaxSubspacesize(dtype,device,nov):
    """Calculate the maximum size of the subspace dimension 
    based on available memory. The full subspace size is nov 
    """

    device = device.type
    # Get available memory
    if device == 'cpu':
        available_memory = psutil.virtual_memory().available
    elif device == 'cuda':
        available_memory, _ = torch.cuda.mem_get_info(torch.device('cuda'))
    else:
        raise ValueError("Unsupported device. Use 'cpu' or 'cuda'.")

    bytes_per_element = torch.finfo(dtype).bits // 8  # Bytes per element
    num_matrices = 2  # Number of big matrices that will take up a big chunk of memory: V, HV

    # Define a memory fraction to use (e.g., 50% of available memory)
    memory_fraction = 0.5
    usable_memory = available_memory * memory_fraction

    # Calculate maximum n based on memory
    n_calculated = int(usable_memory // (nov * bytes_per_element * num_matrices))

    # Ensure n does not exceed nmax
    return min(n_calculated, nov)

def get_subspace_eig(H,nroots,V,vend,nonorthogonal=False):
    
    if nonorthogonal:
        # Need to solve the generalized eigenvalue problem
        # Method as described in Appendix A of J. Chem. Phys. 144, 174105 (2016) https://doi.org/10.1063/1.4947245

        S = torch.einsum('ro,no->rn',V[:vend],V[:vend])
        # S = 0.5*(S+S.T) # symmetrize for numerical stability
        # Step 1: Calculate D^(-1/2)
        D = torch.diag(S)  # Extract the diagonal elements of S
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D))

        # Step 2: Compute D^(-1/2) S D^(-1/2) this is done to reduce the condition number of S
        S_tilde = torch.einsum('ab,bc,cd->ad',D_inv_sqrt,S,D_inv_sqrt) # D_inv_sqrt @ S @ D_inv_sqrt

        # Step 3: Cholesky decomposition of S_tilde
        L = torch.linalg.cholesky(S_tilde)
        L_inv_D_inv_sqrt = torch.linalg.solve_triangular(L,D_inv_sqrt,upper=False)
        D_inv_sqrt_L_inv_T = L_inv_D_inv_sqrt.T

        # Step 4: Compute the modified A matrix
        A_tilde = torch.einsum('ab,bc,cd->ad',L_inv_D_inv_sqrt,H,D_inv_sqrt_L_inv_T)
        # A_tilde = torch.linalg.inv(L) @ (D_inv_sqrt @ H @ D_inv_sqrt) @ torch.linalg.inv(L).T

        # Step 5: Solve the standard eigenvalue problem A_tilde X = X λ
        r_eval, X = torch.linalg.eigh(A_tilde)

        # Step 6: Transform the eigenvectors back to the original problem
        r_evec = D_inv_sqrt_L_inv_T @ X[:,:nroots]

    else:
        r_eval, r_evec = torch.linalg.eigh(H) # find the eigenvalues and the eigenvectors
        r_evec = r_evec[:,:nroots]

    # No need to sort the eigenvalues because torch.linalg.eigh return eigenvalues in ascending order
    # r_eval, r_idx = torch.sort(r_eval, descending=False) # sort the eigenvalues in ascending order
    # e_val_n = r_eval[:nroots] # keep only the lowest nroots eigenvalues; 
    # sort the eigenvectors accordingly
    # e_vec_n = r_evec[:, r_idx[:nroots]]
    
    r_eval = r_eval[:nroots] # keep only the lowest nroots eigenvalues; 
    return r_eval, r_evec

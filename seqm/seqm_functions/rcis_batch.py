import torch
from seqm.seqm_functions.pack import packone, unpackone
import warnings

def rcis_batch(mol, w, e_mo, nroots):
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

    norb = mol.norb
    nocc = mol.nocc
    nmol = mol.nmol

    if not torch.all(norb == norb[0]) or not torch.all(nocc == nocc[0]):
        raise ValueError(
            'All molecules in the batch should be of the same type with same number of orbitals and electrons')
    norb = norb[0]
    nocc = nocc[0]
    nvirt = norb - nocc
    nov = nocc * nvirt

    if nroots > nov:
        raise Exception(f"Maximum number of roots for this molecule is {nov}. Reduce the requested number of roots")
    nstart = nroots+min(2,nov-nroots)

    # ea_ei contains the list of orbital energy difference between the virtual and occupied orbitals
    ea_ei = e_mo[:,nocc:norb].unsqueeze(1)-e_mo[:,:nocc].unsqueeze(2)
    approxH = ea_ei.view(-1,nov)

    # Make the davidson guess vectors
    sorted_ediff, sortedidx = torch.sort(approxH, stable=True, descending=False) # stable to preserve the order of degenerate orbitals

    nroots_expand = nstart
    # If the last chosen root was degenerate in ea_ei, then expand the subspace to include all the degenerate roots
    while nroots_expand < len(sorted_ediff[0]) and torch.all((sorted_ediff[:,nroots_expand] - sorted_ediff[:,nroots_expand-1]) < 1e-5):
        nroots_expand += 1

    if nroots_expand>nstart:
        # print(f"More roots will be calculated because of degeneracy in the MOs")
        nstart = nroots_expand
        # print(f"More roots will be calculated because of degeneracy in the MOs. NRoots changed from {nroots} to {nroots_expand}")
        # nroots = nroots_expand

    maxSubspacesize = getMaxSubspacesize(dtype,device,nov,nmol=nmol) # TODO: User-defined
    V = torch.zeros(nmol,nov,maxSubspacesize,device=device,dtype=dtype)
    HV = torch.empty_like(V)

    # z = torch.arange(nstart)
    # for i in range(nmol):
    #     V[i,sortedidx[i,:nstart],z] = 1.0
    V[torch.arange(nmol).unsqueeze(1),torch.arange(nstart),sortedidx[:,:nstart]] = 1.0

    max_iter = 3*maxSubspacesize//nroots # Heuristic: allow one or two subspace collapse. TODO: User-defined
    root_tol = 1e-6 # TODO: User-defined/fixed
    vector_tol = root_tol*0.02 # Vectors whose norm is smaller than this will be discarded
    iter = 0
    vstart = torch.zeros(nmol,dtype=torch.int,device=device)
    vend = torch.full((nmol,),nstart,dtype=torch.int,device=device)
    done = torch.zeros(nmol,dtype=torch.bool,device=device)

    # TODO: Test if orthogonal or nonorthogonal version is more efficient
    nonorthogonal = False # TODO: User-defined/fixed

    C = mol.eig_vec
    Cocc = torch.stack([C[b,:,:nocc] for b in range(nmol)],dim=0)
    Cvirt = torch.stack([C[b,:,nocc:] for b in range(nmol)],dim=0)

    e_val_n = torch.empty(nmol,nroots,dtype=dtype,device=device)

    while iter <= max_iter: # Davidson loop

        max_v = torch.max(vend-vstart).item()
        V_batched = torch.zeros(nmol,max_v,nov,dtype=dtype,device=device)
        for i in range(nmol):
            if not done[i]:
                V_batched[i,: (vend[i]-vstart[i]),:] = V[i,vstart[i]:vend[i],:]

        # TODO: Think about how HV is going to be built
        HV_batch = matrix_vector_product_batched(mol,V_batched, w, ea_ei, Cocc, Cvirt)

        for i in range(nmol):
            if not done[i]:
                HV[i, vstart[i]:vend[i]] = HV_batch[i, :(vend[i]-vstart[i])]

        # Make H by multiplying V.T * HV
        # Option 1: direct multiplication
        vend_max = torch.max(vend).item()
        H = torch.einsum('bnia,bria->bnr',V[:,:vend_max].view(nmol,vend_max,nocc,nvirt),HV[:,:vend_max].view(nmol,vend_max,nocc,nvirt))

        # Option 2: avoid redundant multipication by only forming the new blocks of H coming from the guess vectors V[vstart:vend]
        # H = torch.empty(vend,vend,device=device,dtype=dtype)
        # if vstart == 0:
        #     H[0:vend,0:vend] = torch.einsum('nia,ria->nr',V[:vend,:].view(vend,nocc,nvirt),HV[:vend,:].view(vend,nocc,nvirt))
        # else:
        #     H[:vstart,:vstart] = Hold
        #     H[vstart:vend,vstart:vend] = torch.einsum('nia,ria->nr',V[vstart:vend,:].view(vend-vstart,nocc,nvirt),HV[vstart:vend,:].view(vend-vstart,nocc,nvirt))
        #     H[vstart:vend,:vstart] = torch.einsum('nia,ria->nr',V[vstart:vend,:].view(vend-vstart,nocc,nvirt),HV[0:vstart,:].view(vstart,nocc,nvirt))
        #     H[:vstart,vstart:vend] = H[vstart:vend,:vstart].T
        # Hold = H[:vend,:vend]

        # TODO: Check if Option 2 is necessary or go with Option 1 (much more simple and readable)
        # Option 2 builds H block-by-block and avoids redundant multiplications

        iter = iter + 1

        # Diagonalize the subspace hamiltonian
        zero_pad = vend_max - vend
        e_vec_n =  get_subspace_eig_batched(H,nroots,zero_pad,e_val_n,done,nonorthogonal)


        amplitudes = torch.einsum('bvr,bvo->bro',e_vec_n,V[:,:vend_max,:])
        residual = torch.einsum('bvr,bvo->bro',e_vec_n, HV[:,:vend_max,:]) - amplitudes*e_val_n.unsqueeze(2)

        resid_norm = torch.norm(residual,dim=2)
        roots_not_converged = resid_norm > root_tol

        for i in range(nmol):
            if done[i]:
                continue
            
            n_not_converged = roots_not_converged[i].sum()

            if n_not_converged > 0 and n_not_converged + vend[i] > maxSubspacesize:
                #  collapse the subspace
                print(f"Maximum subspace size reached for molecule {i+1}, increase the subspace size. Collapsing subspace")
                V[i,:] = 0.0
                V[i,:nroots,:] = amplitudes[i]
                # HV[:nroots,:] = torch.einsum('vr,vo->ro',e_vec_n, HV[:vend,:])
                vstart[i] = 0
                vend[i] = nroots
                if iter > max_iter:
                    warnings.warn("Maximum iterations reached but roots have not converged")
                continue

            newsubspace = residual[i,roots_not_converged[i],:]/(e_val_n[i,roots_not_converged[i]].unsqueeze(1) - approxH[i].unsqueeze(0))

            vstart[i] = vend[i]
            if nonorthogonal:
                raise NotImplementedError("Non-orthogonal davidson not yet implemented")
                newsubspace_norm = torch.norm(newsubspace,dim=1)
                nonzero_newsubspace = newsubspace_norm > vector_tol
                vend = vstart + nonzero_newsubspace.sum()
                V[vstart:vend] = newsubspace[nonzero_newsubspace]/newsubspace_norm[nonzero_newsubspace].unsqueeze(1)

            else:
                vend[i] = orthogonalize_to_current_subspace(V[i], newsubspace, vend[i], vector_tol)

            roots_left = vend[i] - vstart[i]
            if roots_left==0:
                done[i] = True

            print(f"Iteration {iter:2}: Found {nroots-roots_left}/{nroots} states, Total Error: {torch.sum(resid_norm[i]):.4e}")

        if torch.all(done):
            break
        if iter > max_iter:
            warnings.warn("Maximum iterations reached but roots have not converged")

    print("")
    for j in range(nmol):
        print(f"Molecule {j}\n")
        for i, energy in enumerate(e_val_n[j], start=1):
            print(f"  State {i}: {energy:.15f} eV")

    return e_val_n, amplitudes


def matrix_vector_product_batched(mol, V, w, ea_ei, Cocc, Cvirt):
    # C: Molecule Orbital Coefficients
    nmol, nvec, nov = V.shape

    norb = mol.norb[0]
    nocc = mol.nocc[0]
    nvirt = norb - nocc


    Via = V.view(nmol,nvec, nocc, nvirt)
    P_xi = torch.einsum('bmi,bria,bna->brmn', Cocc,Via, Cvirt)

    F0 = makeA_pi_batched(mol,P_xi,w)

    # TODO: why am I multiplying by 2?
    A = torch.einsum('bmi,brmn,bna->bria', Cocc, F0, Cvirt)*2.0

    A += Via*ea_ei.unsqueeze(1)

    return A.view(nmol, nvec, -1)

def makeA_pi_batched(mol,P_xi,w_,allSymmetric=False):
    """
    Given the amplitudes in the AO basis (i.e. the transition densities)
    calculates the contraction with two-electron integrals
    In other words for an amplitued X_jb, this function calculates \sum_jb (\mu\nu||jb)X_jb
    """
    device = P_xi.device
    dtype = P_xi.dtype

    npairs_per_mol = (mol.molsize*(mol.molsize-1)) // 2
    nmol  = mol.nmol
    mask  = mol.mask[:npairs_per_mol]
    maskd = mol.maskd[:mol.molsize]
    mask_l = mol.mask_l[:npairs_per_mol]
    molsize = mol.molsize
    nHeavy = mol.nHeavy[0]
    nHydro = mol.nHydro[0]
    norb = mol.norb[0]

    nnewRoots = P_xi.shape[1]
    P0 = torch.stack([
        unpackone(P_xi[i,j], 4*nHeavy, nHydro, molsize * 4)
        for i in range(nmol) for j in range(nnewRoots)
    ]).view(nmol,nnewRoots, molsize * 4, molsize * 4)

    w = w_.view(nmol,npairs_per_mol,10,10)
    # Compute the (ai||jb)X_jb
    F = makeA_pi_symm_batch(mol,P0,w)

    if not allSymmetric:

        P0_antisym = 0.5*(P0 - P0.transpose(2,3))
        P_anti = P0_antisym.reshape(nmol,nnewRoots,molsize,4,molsize,4)\
                  .transpose(3,4).reshape(nmol,nnewRoots,molsize*molsize,4,4)
        del P0_antisym

        # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
        #   0,     1         2       3       4         5       6      7         8        9
        ind = torch.tensor([[0,1,3,6],
                            [1,2,4,7],
                            [3,4,5,8],
                            [6,7,8,9]],dtype=torch.int64, device=device)
        sumK = torch.empty(nmol,nnewRoots, w.shape[1], 4, 4, dtype=dtype, device=device)
        Pp = -0.5 * P_anti[:, :, mask]
        for i in range(4):
            for j in range(4):
                #\sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
                sumK[...,i,j] = torch.sum(Pp*w[...,ind[i],:][...,:,ind[j]].unsqueeze(1),dim=(3,4))
        F.index_add_(2,mask,sumK)
        F[:,:,mask_l] -= sumK.transpose(3,4)
        del Pp
        del sumK

        gsp = mol.parameters['g_sp'].view(nmol,-1)
        gpp = mol.parameters['g_pp'].view(nmol,-1)
        gp2 = mol.parameters['g_p2'].view(nmol,-1)
        hsp = mol.parameters['h_sp'].view(nmol,-1)

        F2e1c = torch.zeros(nmol,nnewRoots,maskd.shape[0],4,4,device=device,dtype=dtype)
        for i in range(1,4):
            #(s,p) = (p,s) upper triangle
            F2e1c[...,0,i] = P_anti[...,maskd,0,i]*(0.5*hsp - 0.5*gsp).unsqueeze(1)
        #(p,p*)
        for i,j in [(1,2),(1,3),(2,3)]:
            F2e1c[...,i,j] = P_anti[...,maskd,i,j]* (0.25*gpp - 0.75*gp2).unsqueeze(1)

        F2e1c.add_(F2e1c.triu(1).transpose(3,4),alpha=-1.0)
        F[:,:,maskd] += F2e1c
        del P_anti
        del F2e1c

    F0 = F.reshape(nmol,nnewRoots,molsize,molsize,4,4) \
             .transpose(3,4) \
             .reshape(nmol,nnewRoots, 4*molsize, 4*molsize)
    del F

    F0 = torch.stack([
        packone(F0[i,j], 4*nHeavy, nHydro, norb)
        for i in range(nmol) for j in range(nnewRoots)
    ])

    return F0.view(nmol,nnewRoots,norb,norb)

def makeA_pi_symm_batch(mol,P0,w):

    P0_sym = 0.5*(P0 + P0.transpose(2,3))

    molsize = mol.molsize
    nnewRoots = P0.shape[1]
    dtype = P0.dtype
    device = P0.device

    npairs_per_mol = (mol.molsize*(mol.molsize-1)) // 2
    mask  = mol.mask[:npairs_per_mol]
    maskd = mol.maskd[:mol.molsize]
    mask_l = mol.mask_l[:npairs_per_mol]
    idxi  = mol.idxi[:npairs_per_mol]
    idxj  = mol.idxj[:npairs_per_mol]
    nmol = mol.nmol

    P = P0_sym.reshape(nmol,nnewRoots,molsize,4,molsize,4)\
              .transpose(3,4).reshape(nmol,nnewRoots,molsize*molsize,4,4)
    del P0_sym
    F = torch.zeros_like(P)

    # Two center-two elecron integrals
    weight = torch.tensor([1.0,
                           2.0, 1.0,
                           2.0, 2.0, 1.0,
                           2.0, 2.0, 2.0, 1.0],dtype=dtype, device=device).reshape((-1,10))

    #0.030472556808550877, Pdiag_symmetrized = P[:,maskd]+P[:,maskd].transpose(2,3)
    # weight *= 0.5 # dividing by 2 because I didn't do it while making the symmetrized P matrix
    Pdiag_symmetrized = P[:,:,maskd]

    PA = (Pdiag_symmetrized[:,:,idxi][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).unsqueeze(-1)
    PB = (Pdiag_symmetrized[:,:,idxj][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).unsqueeze(-2)
    del Pdiag_symmetrized

    #suma \sum_{mu,nu \in A} P_{mu, nu in A} (mu nu, lamda sigma) = suma_{lambda sigma \in B}
    #suma shape (npairs, 10)
    suma = torch.sum(PA*w.unsqueeze(1),dim=3)
    #sumb \sum_{l,s \in B} P_{l, s inB} (mu nu, l s) = sumb_{mu nu \in A}
    #sumb shape (npairs, 10)
    sumb = torch.sum(PB*w.unsqueeze(1),dim=4)
    #reshape back to (npairs 4,4)
    # as will use index add in the following part
    sumA = torch.zeros(nmol,nnewRoots,w.shape[1],4,4,dtype=dtype, device=device)
    sumB = torch.zeros_like(sumA)
    sumA[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = suma
    sumB[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = sumb
    sumA.add_(sumA.triu(1).transpose(3,4))
    sumB.add_(sumB.triu(1).transpose(3,4))
    #F^A_{mu, nu} = Hcore + \sum^A + \sum_{B} \sum_{l, s \in B} P_{l,s \in B} * (mu nu, l s)
    #\sum_B
    F.index_add_(2,maskd[idxi],sumB)
    #\sum_A
    F.index_add_(2,maskd[idxj],sumA)

    del PA
    del PB
    del sumA
    # del sumB

    # off diagonal block part, check KAB in forck2.f
    # mu, nu in A
    # lambda, sigma in B
    # F_mu_lambda = Hcore - 0.5* \sum_{nu \in A} \sum_{sigma in B} P_{nu, sigma} * (mu nu, lambda, sigma)

    sumK = sumB

    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #   0,     1         2       3       4         5       6      7         8        9
    ind = torch.tensor([[0,1,3,6],
                        [1,2,4,7],
                        [3,4,5,8],
                        [6,7,8,9]],dtype=torch.int64, device=device)
    # Pp =P[mask], P_{mu \in A, lambda \in B}
    Pp = -0.5*P[:,:,mask]
    for i in range(4):
        for j in range(4):
            #\sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            sumK[...,i,j] = torch.sum(Pp*w[...,ind[i],:][...,:,ind[j]].unsqueeze(1),dim=(3,4))
    F.index_add_(2,mask,sumK)
    F[:,:,mask_l] += sumK.transpose(3,4)
    del Pp

    Pptot = P[...,1,1]+P[...,2,2]+P[...,3,3]

    # One center-two electron integrals
    gss = mol.parameters['g_ss'].view(nmol,-1)
    gsp = mol.parameters['g_sp'].view(nmol,-1)
    gpp = mol.parameters['g_pp'].view(nmol,-1)
    gp2 = mol.parameters['g_p2'].view(nmol,-1)
    hsp = mol.parameters['h_sp'].view(nmol,-1)

    F2e1c = torch.zeros(nmol,nnewRoots,maskd.shape[0],4,4,device=device,dtype=dtype)

    F2e1c[...,0,0] = 0.5*P[...,maskd,0,0]*gss.unsqueeze(1) + Pptot[...,maskd]*(gsp-0.5*hsp).unsqueeze(1)
    for i in range(1,4):
        #(p,p)
        F2e1c[...,i,i] = P[...,maskd,0,0]*(gsp-0.5*hsp).unsqueeze(1) + 0.5*P[...,maskd,i,i]*gpp.unsqueeze(1) \
                + (Pptot[...,maskd] - P[...,maskd,i,i]) * (1.25*gp2-0.25*gpp).unsqueeze(1)
        #(s,p) = (p,s) upper triangle
        F2e1c[...,0,i] = P[...,maskd,0,i]*(1.5*hsp - 0.5*gsp).unsqueeze(1)
    #(p,p*)
    for i,j in [(1,2),(1,3),(2,3)]:
        F2e1c[...,i,j] = P[...,maskd,i,j]* (0.75*gpp - 1.25*gp2).unsqueeze(1)

    # F.add_(F2e1c)
    F2e1c.add_(F2e1c.triu(1).transpose(3,4))
    F[:,:,maskd] += F2e1c

    del Pptot
    del P

    return F


def orthogonalize_to_current_subspace(V, newsubspace, vend, tol):
    """Orthogonalizes the vectors in the newsubspace against the original subspace 
       with Gram-Schmidt orthogonalization. We cannot use Modified-Gram-Schmidt because 
       we want leave the original subspace vectors untouched

    :V: Original subspace vectors (with pre-allocated memory for new vectors)
    :newsubspace: vectors that have to be orthonormalized
    :vend: original subspace size
    :tol: the tolerance for the norm of new vectors below which the vector will be discarded
    :returns: vend: size of the subspace after adding in the new vectors 

    """
    for i in range(newsubspace.shape[0]):
        vec = newsubspace[i]
        # Instead of batch processing like below, it is more numerically stable to do it one by one in a loop
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

def getMaxSubspacesize(dtype,device,nov,nmol=1):
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
    n_calculated = int(usable_memory // (nov * nmol * bytes_per_element * num_matrices))

    # Ensure n does not exceed nmax
    return min(n_calculated, nov)

def get_subspace_eig_batched(H,nroots,zero_pad,e_val_n,done,nonorthogonal):

    if nonorthogonal:
        raise NotImplementedError("Non-orthogonal davidson not yet implemented")
        # Need to solve the generalized eigenvalue problem
        # Method as described in Appendix section 1 of J. Chem. Phys. 144, 174105 (2016) https://doi.org/10.1063/1.4947245

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

        e_vec_n = torch.empty(e_val_n.shape[0],H.shape[1],nroots,device=H.device,dtype=H.dtype)
        for i in range(done.shape[0]):
            if done[i]:
                continue
            e_val_n[i] = r_eval[i,zero_pad[i]:zero_pad[i]+nroots]
            e_vec_n[i] = r_evec[i,:,zero_pad[i]:zero_pad[i]+nroots]

        return e_vec_n


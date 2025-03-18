import torch
from .dipole import calc_dipole_matrix
from .constants import a0
import math
# from seqm.seqm_functions.pack import packone, unpackone

def rcis_batch(mol, w, e_mo, nroots, root_tol, init_amplitude_guess=None):
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

    norb_batch, nocc_batch, nmol = mol.norb, mol.nocc, mol.nmol
    if not torch.all(norb_batch == norb_batch[0]) or not torch.all(nocc_batch == nocc_batch[0]):
        raise ValueError("All molecules in the batch must have the same number of orbitals and electrons")
    norb, nocc = norb_batch[0], nocc_batch[0]
    nvirt = norb - nocc
    nov = nocc * nvirt

    if nroots > nov:
        raise Exception(f"Maximum number of roots for this molecule is {nov}. Reduce the requested number of roots")

    # Precompute energy differences (ea_ei) and form the approximate diagonal of the Hamiltonian (approxH)
    # ea_ei contains the list of orbital energy difference between the virtual and occupied orbitals
    ea_ei = e_mo[:,nocc:norb].unsqueeze(1)-e_mo[:,:nocc].unsqueeze(2)
    approxH = ea_ei.view(-1,nov)
    
    maxSubspacesize = getMaxSubspacesize(dtype,device,nov,nmol=nmol) # TODO: User-defined

    V = torch.zeros(nmol,maxSubspacesize,nov,device=device,dtype=dtype)
    HV = torch.empty_like(V)

    if init_amplitude_guess is None:
        nstart, nroots = make_guess(approxH,nroots,maxSubspacesize,V,nmol,nov)
    else:
        nstart = nroots
        V[:,:nstart,:] = init_amplitude_guess

    max_iter = 100 # TODO: User-defined
    vector_tol = root_tol*0.02 # Vectors whose norm is smaller than this will be discarded
    davidson_iter = 0
    vstart = torch.zeros(nmol,dtype=torch.long,device=device)
    vend = torch.full((nmol,),nstart,dtype=torch.long,device=device)
    done = torch.zeros(nmol,dtype=torch.bool,device=device)

    # TODO: Test if orthogonal or nonorthogonal version is more efficient
    nonorthogonal = False # TODO: User-defined/fixed

    C = mol.eig_vec
    Cocc = C[:,:,:nocc]
    Cvirt = C[:,:,nocc:norb]

    e_val_n = torch.empty(nmol,nroots,dtype=dtype,device=device)
    amplitude_store = torch.empty(nmol,nroots,nov,dtype=dtype,device=device)

    # header = f"{'Iteration':>10} | {'States Found':^15} | {'Total Error':>15}"
    # print("-" * len(header))
    # print(header)
    # print("-" * len(header))

    while davidson_iter <= max_iter: # Davidson loop

        # Determine current subspace dimensions per molecule
        delta = vend - vstart
        max_v = int(delta.max().item())
        rel_idx = torch.arange(max_v, device=device).unsqueeze(0)  # (1, max_v)
        abs_idx = rel_idx + vstart.unsqueeze(1)  # (nmol, max_v)
        mask = rel_idx < delta.unsqueeze(1)  # (nmol, max_v)
        batch_idx = torch.arange(nmol, device=device).unsqueeze(1).expand(-1, max_v)

        # Gather current subspace vectors into V_batched
        V_batched = torch.zeros(nmol, max_v, nov, dtype=dtype, device=device)
        V_batched[mask] = V[batch_idx[mask], abs_idx[mask], :]

        # Compute the matrix-vector product in the current subspace
        HV_batch = matrix_vector_product_batched(mol, V_batched, w, ea_ei, Cocc, Cvirt)
        HV[batch_idx[mask], abs_idx[mask], :] = HV_batch[mask]

        # Make H by multiplying V.T * HV
        vend_max = int(torch.max(vend).item())
        H = torch.einsum('bnia,bria->bnr',V[:,:vend_max].view(nmol,vend_max,nocc,nvirt),HV[:,:vend_max].view(nmol,vend_max,nocc,nvirt))

        davidson_iter = davidson_iter + 1

        # Diagonalize the subspace hamiltonian
        zero_pad = vend_max - vend  # Zero-padding for molecules with smaller subspaces
        e_vec_n = get_subspace_eig_batched(H, nroots, zero_pad, e_val_n, done, nonorthogonal)

        # Compute CIS amplitudes and the residual
        amplitudes = torch.einsum('bvr,bvo->bro',e_vec_n,V[:,:vend_max,:])
        residual = torch.einsum('bvr,bvo->bro',e_vec_n, HV[:,:vend_max,:]) - amplitudes*e_val_n.unsqueeze(2)
        resid_norm = torch.norm(residual,dim=2)
        roots_not_converged = resid_norm > root_tol

        # Mark molecules with all roots converged and store amplitudes
        mol_converged = roots_not_converged.sum(dim=1) == 0
        done_this_loop = (~done) & mol_converged
        done[done_this_loop] = True
        amplitude_store[done_this_loop] = amplitudes[done_this_loop]

        # Collapse the subspace for those molecules whose subspace will exceed maxSubspacesize
        collapse_condition = ((roots_not_converged.sum(dim=1) + vend > maxSubspacesize)) & (maxSubspacesize != nov)
        collapse_mask = (~done) & (~mol_converged) & collapse_condition 
        if collapse_mask.sum() > 0:
            if davidson_iter == 1:
                raise Exception("Insufficient memory to perform even a single iteration of subspace expansion")

            V[collapse_mask] = 0
            V[collapse_mask,:nroots,:] = amplitudes[collapse_mask]
            HV[collapse_mask,:nroots,:] = torch.einsum('bvr,bvo->bro',e_vec_n[collapse_mask], HV[collapse_mask,:vend_max,:]) 
            HV[collapse_mask,nroots:] = 0
            vstart[collapse_mask] = 0
            vend[collapse_mask] = nroots

        # Orthogonalize the residual vectors for molecules
        orthogonalize_mask = (~done) & (~mol_converged) # & (~collapse_condition)
        mols_to_ortho = torch.nonzero(orthogonalize_mask).squeeze(1)
        for i in mols_to_ortho:
            newsubspace = residual[i,roots_not_converged[i],:]/(e_val_n[i,roots_not_converged[i]].unsqueeze(1) - approxH[i].unsqueeze(0))

            vstart[i] = vend[i]
            # The original 'V' vector is passed by reference to the 'orthogonalize_to_current_subspace' function. 
            # This means changes inside the function will directly modify 'V[i]'
            vend[i] = orthogonalize_to_current_subspace(V[i], newsubspace, vend[i], vector_tol)
            if vend[i] - vstart[i] == 0:
                done[i] = True
                amplitude_store[i] = amplitudes[i]

            # if davidson_iter % 5 == 0:
                # print(f"davidson_iteration {davidson_iter:2}: Found {nroots-roots_left}/{nroots} states, Total Error: {torch.sum(resid_norm[i]):.4e}")
                # states_found = f'{nroots-roots_left:3d}/{nroots:3d}'
                # print(f"{davidson_iter:10d} | {states_found:^15} | {total_error[i]:15.4e}")

        if torch.all(done):
            break
        if davidson_iter > max_iter:
            raise Exception("Maximum iterations reached but roots have not converged")

    # print("-" * len(header))
    print("\nCIS excited states:")
    for j in range(nmol):
        if nmol>1: print(f"\nMolecule {j+1}")
        for i, energy in enumerate(e_val_n[j], start=1):
            print(f"State {i:3d}: {energy:.15f} eV")
    print("")

    # Post CIS analysis
    rcis_analysis(mol,e_val_n,amplitude_store,nroots)

    return e_val_n, amplitude_store


def matrix_vector_product_batched(mol, V, w, ea_ei, Cocc, Cvirt, makeB=False):
    # C: Molecule Orbital Coefficients
    nmol, nNewRoots, nov = V.shape

    norb = mol.norb[0]
    nocc = mol.nocc[0]
    nvirt = norb - nocc

    Via = V.view(nmol,nNewRoots, nocc, nvirt)

    # I often run out of memory because I calculate \sum_ia (ia||jb)V_ia in makeA_pi_batched for all the roots in V_ia
    # at once. To avoid this, we first estimate the peak memory usage and then chunk over nNewRoots.
    # TODO: Also chunk over the nmol dimension
    need_to_chunk, chunk_size = getMemUse(V.dtype,V.device,mol,nNewRoots)

    if not need_to_chunk:
        P_xi = torch.einsum('bmi,bria,bna->brmn', Cocc,Via, Cvirt)
        F0 = makeA_pi_batched(mol,P_xi,w)
    else:
        F0 = torch.empty(nmol,nNewRoots,norb,norb,device=V.device,dtype=V.dtype)
        for start in range(0, nNewRoots, chunk_size):
            end = min(start + chunk_size, nNewRoots)
            P_xi = torch.einsum('bmi,bria,bna->brmn', Cocc,Via[:,start:end], Cvirt)
            F0[:,start:end,:] = makeA_pi_batched(mol,P_xi,w)

    # why am I multiplying by 2?
    A = torch.einsum('bmi,brmn,bna->bria', Cocc, F0, Cvirt)*2.0

    A += Via*ea_ei.unsqueeze(1)
    A = A.view(nmol, nNewRoots, -1)

    if makeB:
        B = torch.einsum('bmi,brnm,bna->bria', Cocc, F0, Cvirt)*2.0
        B = B.view(nmol,nNewRoots, -1)
        return  A, B

    return A


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
    # P0 = torch.stack([
    #     unpackone(P_xi[i,j], 4*nHeavy, nHydro, molsize * 4)
    #     for i in range(nmol) for j in range(nnewRoots)
    # ]).view(nmol,nnewRoots, molsize * 4, molsize * 4)
    P0 = unpackone_batch(P_xi.view(nmol*nnewRoots,norb,norb), 4*nHeavy, nHydro, molsize * 4).view(nmol,nnewRoots,4*molsize,4*molsize)
    del P_xi

    w = w_.view(nmol,npairs_per_mol,10,10)
    # Compute the (ai||jb)X_jb
    F = makeA_pi_symm_batch(mol,P0,w)

    if not allSymmetric:

        P0_antisym = 0.5*(P0 - P0.transpose(2,3))
        P_anti = P0_antisym.reshape(nmol,nnewRoots,molsize,4,molsize,4)\
                  .transpose(3,4).reshape(nmol,nnewRoots,molsize*molsize,4,4)
        del P0_antisym, P0

        # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
        #   0,     1         2       3       4         5       6      7         8        9
        ind = torch.tensor([[0,1,3,6],
                            [1,2,4,7],
                            [3,4,5,8],
                            [6,7,8,9]],dtype=torch.int64, device=device)
        sumK = torch.empty(nmol,nnewRoots, w.shape[1], 4, 4, dtype=dtype, device=device)
        Pp = P_anti[:, :, mask]
        for i in range(4):
            for j in range(4):
                #\sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
                sumK[...,i,j] = -0.5 * torch.sum(Pp*w[...,ind[i],:][...,:,ind[j]].unsqueeze(1),dim=(3,4))
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
             .reshape(nmol*nnewRoots, 4*molsize, 4*molsize)
    del F

    # F0 = torch.stack([
    #     packone(F0[i,j], 4*nHeavy, nHydro, norb)
    #     for i in range(nmol) for j in range(nnewRoots)
    # ])
    F0 = packone_batch(F0, 4*nHeavy, nHydro, norb).view(nmol,nnewRoots,norb,norb)

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
    # print_memory_usage("After P_symm, and Fock_symm")

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
    # sumA.add_(sumA.triu(1).transpose(3,4))
    # sumB.add_(sumB.triu(1).transpose(3,4))
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
    Pp = P[:,:,mask]
    for i in range(4):
        for j in range(4):
            #\sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            sumK[...,i,j] = -0.5*torch.sum(Pp*w[...,ind[i],:][...,:,ind[j]].unsqueeze(1),dim=(3,4))
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
    # F2e1c.add_(F2e1c.triu(1).transpose(3,4))
    F2e1c += F[:,:,maskd]
    F2e1c.add_(F2e1c.triu(1).transpose(3,4))
    F[:,:,maskd] = F2e1c
    # F[:,:,maskd] += F2e1c
    # F[:,:,maskd] += F[:,:,maskd].triu(1).transpose(3,4)

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
    # reorthogonalization will dramatically improve the loss of orthogonality from numerical errors.
    # See: https://doi.org/10.1016/j.camwa.2005.08.009
    # Giraud, Luc, Julien Langou, and Miroslav Rozloznik. "The loss of orthogonality in the Gram-Schmidt orthogonalization process." Computers & Mathematics with Applications 50.7 (2005): 1069-1075.
    n = newsubspace.shape[0]
    for i in range(n):
        vec = newsubspace[i]
        vec -= (vec @ V[:vend].T) @ V[:vend] 
        vec -= (vec @ V[:vend].T) @ V[:vend] 
        vecnorm = torch.norm(vec)

        if vecnorm > tol:
            V[vend] = vec / vecnorm
            vend = vend + 1

    return vend

import psutil # to get the memory size

def getMaxSubspacesize(dtype,device,nov,nmol=1,num_big_matrices=2):
    """Calculate the maximum size of the subspace dimension 
    based on available memory. The full subspace size is nov 
    """

    device = device.type
    # Get available memory
    if device == 'cpu':
        available_memory = psutil.virtual_memory().available
    elif device == 'cuda':
        available_memory, _ = torch.cuda.mem_get_info(torch.device('cuda:0'))
    else:
        raise ValueError("Unsupported device. Use 'cpu' or 'cuda'.")

    bytes_per_element = torch.finfo(dtype).bits // 8  # Bytes per element

    # Define a memory fraction to use (e.g., 50% of available memory)
    memory_fraction = 0.3
    usable_memory = available_memory * memory_fraction

    # Calculate maximum n based on memory
    n_calculated = int(usable_memory // (nov * nmol * bytes_per_element * num_big_matrices))

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
        r_eval, r_evec = torch.linalg.eigh(H[~done]) # find the eigenvalues and the eigenvectors

        nmol, subspacesize = H.shape[0], H.shape[1]
        e_vec_n = torch.zeros(nmol,subspacesize,nroots,device=H.device,dtype=H.dtype)

        active_indices = torch.nonzero(~done, as_tuple=False).squeeze(1)
        
        # Update eigenvalues and eigenvectors for each active molecule.
        for j, mol_idx in enumerate(active_indices):
            start_idx = int(zero_pad[mol_idx].item())
            end_idx = start_idx + nroots
            e_val_n[mol_idx] = r_eval[j, start_idx:end_idx]
            e_vec_n[mol_idx] = r_evec[j, :, start_idx:end_idx]
        return e_vec_n

def unpackone_batch(x0, nho, nHydro, size):
    x = torch.zeros((x0.shape[0],size, size), dtype=x0.dtype, device=x0.device)
    x[:,:nho,:nho] = x0[:,:nho,:nho]
    x[:,:nho,nho:(nho+4*nHydro):4] = x0[:,:nho, nho:(nho+nHydro)]
    x[:,nho:(nho+4*nHydro):4,nho:(nho+4*nHydro):4] = x0[:,nho:(nho+nHydro),nho:(nho+nHydro)]
    x[:,nho:(nho+4*nHydro):4, :nho] = x0[:,nho:(nho+nHydro), :nho]
    return x

def packone_batch(x, nho, nHydro, norb):
    x0 = torch.zeros((x.shape[0],norb,norb), dtype=x.dtype, device=x.device)
    x0[:,:nho,:nho]=x[:,:nho,:nho]
    x0[:,:nho, nho:(nho+nHydro)] = x[:,:nho,nho:(nho+4*nHydro):4]
    x0[:,nho:(nho+nHydro),nho:(nho+nHydro)] = x[:,nho:(nho+4*nHydro):4,nho:(nho+4*nHydro):4]
    x0[:,nho:(nho+nHydro), :nho] = x[:,nho:(nho+4*nHydro):4, :nho]
    return x0

def print_memory_usage(step_description, device=0):
    allocated = torch.cuda.memory_allocated(device) / 1024**2  # Convert to MB
    reserved = torch.cuda.memory_reserved(device) / 1024**2    # Convert to MB
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    print(f"[{step_description}]")
    print(f"  Allocated Memory: {allocated:.2f} MB")
    print(f"  Reserved Memory: {reserved:.2f} MB")
    print(f"  Max Allocated Memory: {max_allocated:.2f} MB")
    print(f"  Max Reserved Memory: {max_reserved:.2f} MB\n")

def rcis_analysis(mol,excitation_energies,amplitudes,nroots,rpa=False):
    dipole_mat = calc_dipole_matrix(mol) 
    transition_dipole, oscillator_strength =  calc_transition_dipoles(mol,amplitudes,excitation_energies,nroots,dipole_mat,rpa)
    print_rcis_analysis(excitation_energies,transition_dipole,oscillator_strength)

def calc_transition_dipoles(mol,amplitudes,excitation_energies,nroots,dipole_mat,rpa=False):

    nocc, norb = mol.nocc[0], mol.norb[0]
    nvirt = norb - nocc
    C = mol.eig_vec
    Cocc = C[:,:,:nocc]
    Cvirt = C[:,:,nocc:norb]

    if rpa:
        amp_ia_X = amplitudes[0].view(mol.nmol,nroots,nocc,nvirt)
        amp_ia_Y = amplitudes[1].view(mol.nmol,nroots,nocc,nvirt)
    else:
        amp_ia_X = amplitudes.view(mol.nmol,nroots,nocc,nvirt)

    nHeavy = mol.nHeavy[0]
    nHydro = mol.nHydro[0]
    norb = mol.norb[0]
    dipole_mat_packed = packone_batch(dipole_mat.view(3*mol.nmol,4*mol.molsize,4*mol.molsize), 4*nHeavy, nHydro, norb).view(mol.nmol,3,norb,norb)


    # CIS transition density R = \sum_ia C_\mu i * t_ia * C_\nu a 
    R = torch.einsum('bmi,bria,bna->brmn',Cocc,amp_ia_X,Cvirt)
    if rpa:
        R += torch.einsum('bma,bria,bni->brmn',Cvirt,amp_ia_Y,Cocc)

    # Transition dipole in AU as calculated in NEXMD
    transition_dipole = torch.einsum('brmn,bdmn->brd',R,dipole_mat_packed)*math.sqrt(2.0)/a0
    hartree = 27.2113962 # value used in NEXMD
    oscillator_strength = 2.0/3.0*excitation_energies/hartree*torch.square(transition_dipole).sum(dim=2)
    return transition_dipole, oscillator_strength


def print_rcis_analysis(excitation_energies,transition_dipole,oscillator_strength):

    print(f"Number of excited states: {excitation_energies.shape[1]}\n")
    print("Excitation energies E (eV), Transition dipoles d (au), and Oscillator strengths f (unitless)")
    row_format = "{:<10}   {:>10}   {:>10}   {:>10}      {:<10}"

    # Print header
    print(row_format.format(
        "E",
        "d x",
        "d y",
        "d z",
        "f"
    ))
    print("-" * 65)

    nmol = excitation_energies.shape[0]
    # Loop over molecules and states using enumerate and zip
    for mol_idx, (mol_energy, mol_dipole, mol_strength) in enumerate(zip(excitation_energies, transition_dipole, oscillator_strength), start=1):
        if nmol>1: print(f"Molecule {mol_idx}:")
        for energy_val, dipole_vals, strength_val in zip(mol_energy, mol_dipole, mol_strength):
            # Convert single-value tensors to Python floats
            e = energy_val.item()
            dx, dy, dz = dipole_vals.tolist()
            s = strength_val.item()
            print(row_format.format(
                f"{e:.6f}",
                f"{dx:.6f}",
                f"{dy:.6f}",
                f"{dz:.6f}",
                f"{s:.6f}"
            ))
        print("")

def getMemUse(dtype,device,mol,nroots=1):
    """
    Estimate the peak memory usage while calculating  \sum_ia (ia||jb)V_ia 
    (contracting the roots with the two-e integrals) 
    We approximate the largest intermediate allocations (P, F, PA, PB, suma, sumA etc. in makeA_pi_symm_batch()) 
    via a factor ~ 200 * nmol * nnewRoots * (molsize^2) * bytes_per_element.
    If that estimate exceeds available memory, we return need_to_chunk=True
    and compute a chunk_size to avoid out-of-memory issues.

    Returns:
        need_to_chunk (bool): Whether we must chunk the computation.
        chunk_size (int or None): Suggested chunk size if need_to_chunk=True, else None.
    """

    dev_type = device.type
    if dev_type == 'cpu':
        available_memory = psutil.virtual_memory().available
    elif dev_type == 'cuda':
        available_memory, _ = torch.cuda.mem_get_info(torch.device('cuda:0'))
    else:
        raise ValueError("Unsupported device type. Use 'cpu' or 'cuda'.")

    bytes_per_element = torch.finfo(dtype).bits // 8
    # 200 is an approximate factor from analyzing the shape and count
    # of all major tensors in makeA_pi_symm_batch. The factor seems to perform well while benchmarking
    mem_per_root = 200.0 * mol.nmol * (mol.molsize**2) * bytes_per_element
    total_mem_estimate = mem_per_root * nroots

    # print(f"Available: {available_memory/1073741824} GiB; Max memory used will be {total_mem_estimate/1073741824} GiB, per root: {mem_per_root/1073741824} GiB")
    need_to_chunk = total_mem_estimate > available_memory
    chunk_size = nroots

    if need_to_chunk:
        # Ensure at least 1, up to nroots
        chunk_size = max(1, min(nroots, int(available_memory // mem_per_root)))

    return need_to_chunk, chunk_size

def make_guess(ea_ei,nroots,maxSubspacesize,V,nmol,nov):
    # Make the davidson guess vectors
    sorted_ediff, sortedidx = torch.sort(ea_ei, stable=True, descending=False) # stable to preserve the order of degenerate orbitals

    nroots_expand = nroots
    # If the last chosen root was degenerate in ea_ei, then expand the subspace to include all the degenerate roots
    while nroots_expand < len(sorted_ediff[0]) and torch.all((sorted_ediff[:,nroots_expand] - sorted_ediff[:,nroots_expand-1]) < 1e-5):
        nroots_expand += 1
    if nroots_expand > nroots:
        print(f"Increasing the number of states calculated from {nroots} to {nroots_expand} because of orbital degeneracies")
        nroots = nroots_expand

    extra_subspace = min(7,nov-nroots)
    # if after the extra_subspace i dont have enough space for subspace expansion then i shouldnt use extra_subspace
    extra_subspace = extra_subspace if 2*nroots+extra_subspace<maxSubspacesize else max(0,maxSubspacesize-2*nroots)
    nstart = nroots+extra_subspace
    V[torch.arange(nmol).unsqueeze(1),torch.arange(nstart),sortedidx[:,:nstart]] = 1.0

    return nstart, nroots


import torch
from .rcis_batch import getMaxSubspacesize, make_guess, orthogonalize_to_current_subspace, matrix_vector_product_batched, rcis_analysis

def rpa(mol, w, e_mo, nroots, root_tol, init_amplitude_guess=None):
    torch.set_printoptions(linewidth=200)
    """Calculate the RPA (random phase approximation) excitation energies and amplitudes
       for TDHF (time-dependent Hartree-Fock)
       using davidson diagonalization

    :param mol: Molecule Orbital Coefficients
    :param w: 2-electron integrals
    :param e_mo: Molecular Orbital energies
    :param nroots: Number of RPA states requested
    :returns: 
        e_val_n: RPA excitation energies
        amplitude_store: tensor containing the [X;Y] RPA amplitudes.
                         amplitude_store has shape (2,nmol,nroots,nov),
                         where amplitude_store[0,...] are the X amplitudes, and amplitude_store[1,...] are the Y amplitudes
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
    
    maxSubspacesize = getMaxSubspacesize(dtype,device,nov,num_big_matrices=3) # TODO: User-defined

    V = torch.zeros(nmol,maxSubspacesize,nov,device=device,dtype=dtype)
    AV = torch.empty_like(V)
    BV = torch.empty_like(V)

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

    C = mol.molecular_orbitals
    Cocc = C[:,:,:nocc]
    Cvirt = C[:,:,nocc:norb]

    e_val_n = torch.empty(nmol,nroots,dtype=dtype,device=device)
    amplitude_store = torch.empty(2,nmol,nroots,nov,dtype=dtype,device=device)

    n_collapses = torch.zeros_like(vstart)
    n_iters = torch.zeros_like(vstart)
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
        AV_batch, BV_batch = matrix_vector_product_batched(mol, V_batched, w, ea_ei, Cocc, Cvirt,makeB=True)
        AV[batch_idx[mask], abs_idx[mask], :] = AV_batch[mask]
        BV[batch_idx[mask], abs_idx[mask], :] = BV_batch[mask]

        # Make H by multiplying V.T * HV
        vend_max = int(torch.max(vend).item())
        A = torch.einsum('bno,bro->bnr',V[:,:vend_max],AV[:,:vend_max])
        B = torch.einsum('bno,bro->bnr',V[:,:vend_max],BV[:,:vend_max])
        ApB = A
        AmB = A - B
        ApB += B # Make A+B

        davidson_iter = davidson_iter + 1

        # Diagonalize the subspace hamiltonian
        zero_pad = vend_max - vend  # Zero-padding for molecules with smaller subspaces
        X, Y = rpa_subspace_eig(ApB,AmB,nroots, zero_pad, e_val_n, done)

        amplitude_X, amplitude_Y, resid_norm, correction_directionR, correction_directionL = calc_rpa_residue(AV,BV,X,Y,V,vend_max,e_val_n)
        # amplitude_X, amplitude_Y, resid_norm, correction_direction = calc_rpa_residue(AV,BV,X,Y,V,vend_max,e_val_n)

        # correction_direction /= (e_val_n.unsqueeze(2) - approxH.unsqueeze(1)) # Davidson preconditioning
        correction_directionR /= (e_val_n.unsqueeze(2) - approxH.unsqueeze(1)) # Davidson preconditioning
        correction_directionL /= (e_val_n.unsqueeze(2) - approxH.unsqueeze(1)) # Davidson preconditioning

        roots_not_converged = resid_norm > root_tol

        # Mark molecules with all roots converged and store amplitudes
        mol_converged = roots_not_converged.sum(dim=1) == 0
        done_this_loop = (~done) & mol_converged
        done[done_this_loop] = True
        n_iters[done_this_loop] = davidson_iter
        amplitude_store[0,done_this_loop] = amplitude_X[done_this_loop]
        amplitude_store[1,done_this_loop] = amplitude_Y[done_this_loop]

        # Collapse the subspace for those molecules whose subspace will exceed maxSubspacesize
        collapse_condition = ((roots_not_converged.sum(dim=1) + vend > maxSubspacesize)) & (maxSubspacesize != nov)
        collapse_mask = (~done) & (~mol_converged) & collapse_condition 
        if collapse_mask.sum() > 0:
            # collapsing subspace means that the guess space will be reset to 2*nroots vectors. Following that, the roots that didn't converge this cycle 
            # will also be added to the guess space. So we have to check if all these vectors will fit in
            if torch.any(roots_not_converged[collapse_mask].sum(dim=1)+2*nroots > maxSubspacesize):
                raise Exception("Insufficient memory to perform subspace expansion following collapse")

            mols_to_collapse = torch.nonzero(collapse_mask).squeeze(1)
            XY_ = torch.cat((X[collapse_condition],Y[collapse_condition]),dim=2)
            XY_, _ = torch.linalg.qr(XY_, mode='reduced')
            for i in mols_to_collapse:
                vend_i = vend[i]
                V[i,:2*nroots,:] =  torch.einsum('vr,vo->ro',XY_[i],V[i,:vend_i,:])
                AV[i,:2*nroots,:] = torch.einsum('vr,vo->ro',XY_[i], AV[i,:vend_i,:]) 
                BV[i,:2*nroots,:] = torch.einsum('vr,vo->ro',XY_[i], BV[i,:vend_i,:]) 
            V[collapse_mask,2*nroots:] = 0
            AV[collapse_mask,2*nroots:] = 0
            BV[collapse_mask,2*nroots:] = 0
            vstart[collapse_mask] = 0
            vend[collapse_mask] = 2*nroots
            n_collapses[collapse_mask] += 1

        # Orthogonalize the residual vectors for molecules
        orthogonalize_mask = (~done) & (~mol_converged) # & (~collapse_condition)
        mols_to_ortho = torch.nonzero(orthogonalize_mask).squeeze(1)
        for i in mols_to_ortho:
            # newsubspace = correction_direction[i,roots_not_converged[i],:]
            # newsubspace = torch.cat((correction_directionR[i,roots_not_converged[i],:],correction_directionL[i,roots_not_converged[i],:]),dim=0)

            vstart[i] = vend[i]
            # The original 'V' vector is passed by reference to the 'orthogonalize_to_current_subspace' function. 
            # This means changes inside the function will directly modify 'V[i]'
            newsubspace = correction_directionR[i,roots_not_converged[i],:]
            vend[i] = orthogonalize_to_current_subspace(V[i], newsubspace, vend[i], vector_tol)
            newsubspace = correction_directionL[i,roots_not_converged[i],:]
            vend[i] = orthogonalize_to_current_subspace(V[i], newsubspace, vend[i], vector_tol)
            if vend[i] - vstart[i] == 0:
                done[i] = True
                amplitude_store[0,i] = amplitude_X[i]
                amplitude_store[1,i] = amplitude_Y[i]
                n_iters[i] = davidson_iter

            # if davidson_iter % 5 == 0:
                # states_found = f'{nroots-roots_left:3d}/{nroots:3d}'
                # print(f"{davidson_iter:10d} | {states_found:^15} | {total_error[i]:15.4e}")
        # for i in range(nmol):
            # print(f"davidson_iteration {davidson_iter:2}: Found {nroots-roots_not_converged[i].sum().item()}/{nroots} states, Total Error: {torch.sum(resid_norm[i]):.4e}")

        if torch.all(done):
            break
        if davidson_iter > max_iter:
            # for i in range(nmol):
            #     print(f"Mol {i+1} Iterations: {davidson_iter:2}: Found {nroots-roots_not_converged[i].sum().item()}/{nroots} states, Total Error: {torch.sum(resid_norm[i]):.4e}")
            for j in range(nmol):
                print(f"Molecule {j}: Number of davidson iterations: {n_iters[j]}, number of subspace collapses: {n_collapses[j]}")
            raise Exception("Maximum iterations reached but roots have not converged")

    # print("-" * len(header))
    print("\nRPA excited states:")
    for j in range(nmol):
        if nmol>1: print(f"\nMolecule {j+1}")
        print(f"Number of davidson iterations: {n_iters[j]}, number of subspace collapses: {n_collapses[j]}")
        for i, energy in enumerate(e_val_n[j], start=1):
            print(f"State {i:3d}: {energy:.15f} eV")
    print("")

    # Post CIS analysis
    rcis_analysis(mol,e_val_n,amplitude_store,nroots,rpa=True)

    return e_val_n, amplitude_store

def make_sqrt_mat(H,):
    eigenvalues, eigenvectors = torch.linalg.eigh(H)
    # FIXME: Might fail in batch mode when some eigenvalues will be zero, take care of it
    if torch.any(eigenvalues<0.0):
        raise Exception("A-B matrix has negative eigenvalues, cannot compute sqrt of A-B matrix")
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    sqrtH = eigenvectors @ torch.diag_embed(sqrt_eigenvalues) @ eigenvectors.transpose(1,2)
    return sqrtH, eigenvectors, eigenvalues

def rpa_subspace_eig(ApB,AmB,nroots, zero_pad, e_val_n, done):
    AmB_sqrt, AmB_evec, AmB_eval = make_sqrt_mat(AmB[~done],)
    H = AmB_sqrt @ ApB[~done] @ AmB_sqrt
    r_eval, r_evec = torch.linalg.eigh(H) # find the eigenvalues and the eigenvectors

    active_indices = torch.nonzero(~done, as_tuple=False).squeeze(1)
    negative_eigs = r_eval < 0.0
    bad_mask = negative_eigs.any(dim=1)
    if bad_mask.any():
        bad_indices = torch.nonzero(bad_mask, as_tuple=False).squeeze(1)
                
        error_msgs = []
        for idx in bad_indices:
            ev_bad = r_eval[idx,negative_eigs[idx]]
            error_msgs.append(
                f"For molecule {active_indices[idx]+1} we have negative eigenvalues w^2 = {ev_bad.tolist()}. "
                "This leads to imaginary roots. If the negative values are small, consider increasing SCF convergence tolerance; "
                "if they are large, the SCF solution was likely not a minimum."
            )
        raise ValueError("\n".join(error_msgs))

    r_eval = torch.sqrt(r_eval)
    
    nmol, subspacesize = ApB.shape[0], ApB.shape[1]
    e_vec_n = torch.zeros(H.shape[0],subspacesize,nroots,device=H.device,dtype=H.dtype)
    
    # Update eigenvalues and eigenvectors for each active molecule.
    for j, mol_idx in enumerate(active_indices):
        start_idx = int(zero_pad[mol_idx].item())
        end_idx = start_idx + nroots
        eval_current = r_eval[j,start_idx:end_idx]
    
        small_eval = (eval_current < 0.1)
        if torch.any(small_eval):
            ev_bad = eval_current[small_eval]
            error_msg = (
                f"For molecule {mol_idx+1} we have very small eigenvalues w = {ev_bad.tolist()}. "
                "Consider increasing SCF convergence tolerance; "
                "The SCF solution might not be a minimum."
            )
            raise ValueError(error_msg)

        e_val_n[mol_idx] = eval_current
        e_vec_n[j] = r_evec[j, :, start_idx:end_idx]

    # r_evec is (A-B)^(-1/2)|X+Y>
    # |X+Y>  = (A-B)^(1/2) r_evec
    XpY = AmB_sqrt @ e_vec_n

    # # |X-Y> = w*(A-B)^(-1)|X+Y>
    # AmB_inverse = AmB_evec @ torch.diag_embed(1.0/AmB_eval) @ AmB_evec.transpose(1,2)
    # XmY = (AmB_inverse @ XpY)*e_val_n[~done].unsqueeze(1)

    # Alternately, |X-Y> = (A+B)|X+Y>/w. This might be easier than calculating (A-B)^(-1)
    XmY = (ApB[~done] @ XpY)/e_val_n[~done].unsqueeze(1) 

    X = XpY + XmY
    Y = XpY - XmY

    XYnorm = torch.sqrt((X**2).sum(dim=1)-(Y**2).sum(dim=1))
    X /= XYnorm.unsqueeze(1)
    Y /= XYnorm.unsqueeze(1)

    X_vec= torch.zeros(nmol,subspacesize,nroots,device=H.device,dtype=H.dtype)
    Y_vec = torch.zeros_like(X_vec)
    X_vec[~done] = X
    Y_vec[~done] = Y
    return X_vec, Y_vec

def calc_rpa_residue(AV,BV,X,Y,V,vend_max,e_val_n):
    amplitude_X = torch.einsum('bvr,bvo->bro',X,V[:,:vend_max,:])
    amplitude_Y = torch.einsum('bvr,bvo->bro',Y,V[:,:vend_max,:])
    AVX = torch.einsum('bvr,bvo->bro',X, AV[:,:vend_max,:])
    AVY = torch.einsum('bvr,bvo->bro',Y, AV[:,:vend_max,:])
    BVX = torch.einsum('bvr,bvo->bro',X, BV[:,:vend_max,:])
    BVY = torch.einsum('bvr,bvo->bro',Y, BV[:,:vend_max,:])

    # TDHF equation:
    # (A B)(X) = w (1  0)(X)
    # (B A)(Y)     (0 -1)(Y) 
    # Which gives, AX + BY = wX and AY + BX = -wY
    resX = AVX + BVY - amplitude_X*e_val_n.unsqueeze(2)
    resY = AVY + BVX + amplitude_Y*e_val_n.unsqueeze(2)

    # We also have
    # (A+B)(X+Y) = w(X-Y)  => resXmY
    # (A-B)(X-Y) = w(X+Y)  => resXpY
    resXmY = resY
    resXpY = resX - resY
    resXmY += resX

    resR = torch.linalg.norm(resXpY,dim=2,ord=torch.inf)
    resL = torch.linalg.norm(resXmY,dim=2,ord=torch.inf)

    chooseResR = resR > resL
    resid_norm = resL
    resid_norm[chooseResR] = resR[chooseResR]
    # correction_direction = resXmY
    # correction_direction[chooseResR] = resXpY[chooseResR]
    # return amplitude_X, amplitude_Y, resid_norm, correction_direction
    return amplitude_X, amplitude_Y, resid_norm, resXpY, resXmY

import torch
from .constants import a0
import math
from seqm.seqm_functions.pack import packone, unpackone
from .dipole import calc_dipole_matrix
from .rcis_batch import orthogonalize_to_current_subspace, getMaxSubspacesize, getMemUse, print_rcis_analysis, get_occ_virt

def rcis_any_batch(mol, w, e_mo, nroots, root_tol, init_amplitude_guess=None):
    torch.set_printoptions(linewidth=200)
    """Calculate the restricted Configuration Interaction Single (RCIS) excitation energies and amplitudes
       using davidson diagonalization
       This function is called when all the molecules in the batch are NOT the same

    :param mol: Molecule Orbital Coefficients
    :param w: 2-electron integrals
    :param e_mo: Orbital energies
    :param nroots: Number of CIS states requested
    :returns: 

    """

    device = w.device
    dtype = w.dtype

    norb_batch, nocc_batch, nmol = mol.norb, mol.nocc, mol.nmol
    nvirt_batch = norb_batch - nocc_batch
    norb, nocc = int(torch.max(norb_batch).item()), int(torch.max(nocc_batch).item())
    nvirt = int(torch.max(nvirt_batch))
    nov = nocc * nvirt

    nocc, nvirt, Cocc, Cvirt, ea_ei = get_occ_virt(mol, orbital_window=None, e_mo=e_mo)
    nov_batch = nocc_batch * (norb_batch-nocc_batch)
    if nroots > torch.min(nov_batch):
        raise Exception(f"Maximum number of roots for this batch of molecules is {torch.min(nov_batch)}. Reduce the requested number of roots")

    # Precompute energy differences (ea_ei) and form the approximate diagonal of the Hamiltonian (approxH)
    # ea_ei contains the list of orbital energy difference between the virtual and occupied orbitals
    approxH = ea_ei.view(-1,nov)
    
    maxSubspacesize = getMaxSubspacesize(dtype,device,nov,nmol=nmol) # TODO: User-defined

    V = torch.zeros(nmol,maxSubspacesize,nov,device=device,dtype=dtype)
    HV = torch.empty_like(V)

    vector_tol = root_tol*0.05 # Vectors whose norm is smaller than this will be discarded

    if init_amplitude_guess is None:
        occ_idx  = torch.arange(nocc, device=device).view(1, -1, 1)
        virt_idx = torch.arange(nvirt, device=device).view(1, 1, -1)

        valid = (occ_idx < nocc_batch[:, None, None]) & (virt_idx < nvirt_batch[:, None, None])  # [nmol,max_nocc,max_nvirt]

        ediffs = ea_ei.clone().reshape(nmol, nov)
        ediffs.masked_fill_(~valid.view(nmol, nov), float('inf'))
        vend, nroots_per_mol, nroots_max = make_guess_any_batch(ediffs,nroots,maxSubspacesize,V,nmol,nov_batch)
    else:
        raise NotImplementedError
        nstart = nroots
        if nroots>1:
            print("WARNING: Orthogonalizing inital guess")
            init_amplitude_guess, _ = torch.linalg.qr(init_amplitude_guess.transpose(1,2), mode='reduced')
            V[:,:nstart,:] = init_amplitude_guess.transpose(1,2)
        else:
            V[:,:nstart,:] = init_amplitude_guess
            V[:,:nstart,:] /= V[:,:nstart,:].norm(dim=2) 
        # fix signs of the Molecular Orbitals by looking at the MOs from the previous step. 
        # This fails when orbitals are degenerate and switch order
        mol.molecular_orbitals *= torch.sign((torch.einsum('Nmp,Nmp->Np',mol.molecular_orbitals,mol.old_mos))).unsqueeze(1)


    max_iter = 100 # TODO: User-defined
    davidson_iter = 0
    vstart = torch.zeros(nmol,dtype=torch.long,device=device)
    done = torch.zeros(nmol,dtype=torch.bool,device=device)

    # TODO: Test if orthogonal or nonorthogonal version is more efficient
    nonorthogonal = False # TODO: User-defined/fixed

    # C = mol.molecular_orbitals
    # Cocc = torch.zeros(nmol,norb,nocc,device=device,dtype=dtype)
    # Cvirt = torch.zeros(nmol,norb,nvirt,device=device,dtype=dtype)
    # for i in range(nmol):
    #     Cocc[i,:,:nocc_batch[i]] = C[i,:,:nocc_batch[i]]
    #     Cvirt[i,:,:nvirt_batch[i]] = C[i,:,nocc_batch[i]:norb_batch[i]]

    e_val_n = torch.zeros(nmol,nroots_max,dtype=dtype,device=device)
    amplitude_store = torch.zeros(nmol,nroots_max,nov,dtype=dtype,device=device)

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
        HV_batch = matrix_vector_product_any_batched(mol, V_batched, w, ea_ei, Cocc, Cvirt)
        HV[batch_idx[mask], abs_idx[mask], :] = HV_batch[mask]

        # Make H by multiplying V.T * HV
        vend_max = int(torch.max(vend).item())
        H = torch.einsum('bno,bro->bnr',V[:,:vend_max],HV[:,:vend_max])

        davidson_iter = davidson_iter + 1

        # Diagonalize the subspace hamiltonian
        zero_pad = vend_max - vend  # Zero-padding for molecules with smaller subspaces
        e_vec_n = get_subspace_eig_any_batched(H, nroots_per_mol,nroots_max, zero_pad, e_val_n, done, nonorthogonal)

        # Compute CIS amplitudes and the residual
        amplitudes = torch.einsum('bvr,bvo->bro',e_vec_n,V[:,:vend_max,:])
        residual = torch.einsum('bvr,bvo->bro',e_vec_n, HV[:,:vend_max,:]) - amplitudes*e_val_n.unsqueeze(2)
        # resid_norm = torch.norm(residual,dim=2)
        resid_norm = torch.linalg.vector_norm(residual,dim=2,ord=torch.inf)
        roots_not_converged = resid_norm > root_tol

        # Mark molecules with all roots converged and store amplitudes
        mol_converged = roots_not_converged.sum(dim=1) == 0
        done_this_loop = (~done) & mol_converged
        done[done_this_loop] = True
        n_iters[done_this_loop] = davidson_iter
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
            n_collapses[collapse_mask] += 1

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
                n_iters[i] = davidson_iter

            # if davidson_iter % 5 == 0:
                # print(f"davidson_iteration {davidson_iter:2}: Found {nroots-roots_left}/{nroots} states, Total Error: {torch.sum(resid_norm[i]):.4e}")
                # states_found = f'{nroots-roots_left:3d}/{nroots:3d}'
                # print(f"{davidson_iter:10d} | {states_found:^15} | {total_error[i]:15.4e}")

        if torch.all(done):
            break
        if davidson_iter > max_iter:
            for j in range(nmol):
                print(f"Molecule {j}: Number of davidson iterations: {n_iters[j]}, number of subspace collapses: {n_collapses[j]}")
            raise Exception("Maximum iterations reached but roots have not converged")

    # print("-" * len(header))
    # print("\nCIS excited states:")
    # for j in range(nmol):
    #     if nmol>1: print(f"\nMolecule {j+1}")
    #     print(f"Number of davidson iterations: {n_iters[j]}, number of subspace collapses: {n_collapses[j]}")
    #     for i, energy in enumerate(e_val_n[j], start=1):
    #         print(f"State {i:3d}: {energy:.15f} eV")
    # print("")

    # Post CIS analysis
    print(f"Number of davidson iterations: {n_iters}, number of subspace collapses: {n_collapses}")
    rcis_analysis(mol,e_val_n,amplitude_store,nroots)

    return e_val_n, amplitude_store

def rcis_analysis(mol,excitation_energies,amplitudes,nroots,rpa=False):
    dipole_mat = calc_dipole_matrix(mol) 
    transition_dipole, oscillator_strength =  calc_transition_dipoles_any_batch(mol,amplitudes,excitation_energies,nroots,dipole_mat,rpa)
    print_rcis_analysis(excitation_energies,transition_dipole,oscillator_strength)

def matrix_vector_product_any_batched(mol, V, w, ea_ei, Cocc, Cvirt, makeB=False):
    # C: Molecule Orbital Coefficients
    nmol, nNewRoots, _ = V.shape

    nocc = Cocc.shape[2]
    nvirt = Cvirt.shape[2]

    Via = V.view(nmol,nNewRoots, nocc, nvirt)

    # I often run out of memory because I calculate \sum_ia (ia||jb)V_ia in makeA_pi_batched for all the roots in V_ia
    # at once. To avoid this, we first estimate the peak memory usage and then chunk over nNewRoots.
    # TODO: Also chunk over the nmol dimension
    need_to_chunk, chunk_size = getMemUse(V.dtype,V.device,mol,nNewRoots)

    if not need_to_chunk:
        P_xi = torch.einsum('bmi,bria,bna->brmn', Cocc,Via, Cvirt)
        F0 = makeA_pi_any_batched(mol,P_xi,w)
        # why am I multiplying by A 2?
        A = torch.einsum('bmi,brmn,bna->bria', Cocc, F0, Cvirt)*2.0
        if makeB:
            B = torch.einsum('bmi,brnm,bna->bria', Cocc, F0, Cvirt)*2.0
    else:
        # F0 = torch.empty(nmol,nNewRoots,norb,norb,device=V.device,dtype=V.dtype)
        A = torch.empty(nmol,nNewRoots,nocc,nvirt,device=V.device,dtype=V.dtype)
        if makeB:
            B = torch.empty_like(A)
        for start in range(0, nNewRoots, chunk_size):
            end = min(start + chunk_size, nNewRoots)
            P_xi = torch.einsum('bmi,bria,bna->brmn', Cocc,Via[:,start:end], Cvirt)
            # F0[:,start:end,:] = makeA_pi_batched(mol,P_xi,w)
            P_xi = makeA_pi_any_batched(mol,P_xi,w)
            F0 = P_xi
            A[:,start:end,:] = torch.einsum('bmi,brmn,bna->bria', Cocc, F0, Cvirt)*2.0
            if makeB:
                B[:,start:end,:] = torch.einsum('bmi,brnm,bna->bria', Cocc, F0, Cvirt)*2.0

    A += Via*ea_ei.unsqueeze(1)
    A = A.reshape(nmol, nNewRoots, -1)

    if makeB:
        B = B.reshape(nmol,nNewRoots, -1)
        return  A, B

    return A


def makeA_pi_any_batched(mol,P_xi,w_,allSymmetric=False):
    """
    Given the amplitudes in the AO basis (i.e. the transition densities)
    calculates the contraction with two-electron integrals
    In other words, for an amplitude X_jb, this function calculates \sum_jb (\mu\nu||jb)X_jb
    """
    device = P_xi.device

    nmol  = mol.nmol
    mask  = mol.mask
    maskd = mol.maskd
    mask_l = mol.mask_l
    molsize = mol.molsize
    nHeavy = mol.nHeavy
    nHydro = mol.nHydro

    nD = P_xi.shape[1]
    P0 = torch.stack([
        unpackone(P_xi[i,j], 4*nHeavy[i], nHydro[i], molsize * 4)
        for i in range(nmol) for j in range(nD)
    ]).view(nmol,nD, molsize * 4, molsize * 4)
    # P0 = unpackone_batch(P_xi.reshape(nmol*nnewRoots,norb,norb), 4*nHeavy, nHydro, molsize * 4).view(nmol,nnewRoots,4*molsize,4*molsize)
    del P_xi

    # Compute the (ai||jb)X_jb
    F = makeA_pi_symm_any_batch(mol,P0,w_)

    if not allSymmetric:

        P0_antisym = 0.5*(P0 - P0.transpose(2,3))
        P_anti = P0_antisym.reshape(nmol,nD,molsize,4,molsize,4)\
                  .transpose(3,4).reshape(nmol*nD*molsize*molsize,4,4)
        del P0_antisym, P0

        # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
        #   0,     1         2       3       4         5       6      7         8        9
        ind = torch.tensor([[0,1,3,6],
                            [1,2,4,7],
                            [3,4,5,8],
                            [6,7,8,9]],dtype=torch.int64, device=device)
        # Upper-tri (A,B) blocks per density
        npairs_ut = mask.numel()
        base_stride = molsize * molsize                 # blocks per density per molecule
        extra_per_mol = (nD - 1) * base_stride          # extra shift added to each molecule when stacking nD
        dgrid = torch.arange(nD, device=device).view(nD, 1)
        mask_exp = (
            mask.view(1, npairs_ut)
            + mol.pair_molid.view(1, npairs_ut) * extra_per_mol
            + dgrid * base_stride
        )  # (nD, npairs_ut)
        mask_flat = mask_exp.reshape(-1)                # (nD*npairs_ut,)
        Pp = P_anti[mask_flat].view(nD, npairs_ut, 4, 4)  # (nD, npairs_ut, 4,4)
        sumK = torch.empty_like(Pp)
        w_K = w_.unsqueeze(0)  # adjust if you keep a separate tensor for off-diagonal pairs
        for i in range(4):
            Wi = w_K[..., ind[i], :]            # (1, npairs_ut, 4, 10)
            for j in range(4):
                Wij = Wi[..., :, ind[j]]        # (1, npairs_ut, 4, 4)
                # elementwise multiply with Pp and sum over ν,σ (the two middle dims)
                sumK[..., i, j] = -0.5*torch.sum(Pp * Wij, dim=(2, 3))  # (nD, npairs_ut)
        F.index_add_(0, mask_flat, sumK.reshape(-1, 4, 4))
        mask_l_exp = (
            mask_l.view(1, -1)
            + mol.pair_molid.view(1, -1) * (nD - 1) * molsize * molsize
            + torch.arange(nD, device=device).view(nD, 1) * molsize * molsize
        ).reshape(-1)

        F[mask_l_exp] -= sumK.reshape(-1,4,4).transpose(-1, -2)
        del Pp
        del sumK

        gsp = mol.parameters['g_sp'].expand(nD,-1).reshape(-1)
        gpp = mol.parameters['g_pp'].expand(nD,-1).reshape(-1)
        gp2 = mol.parameters['g_p2'].expand(nD,-1).reshape(-1)
        hsp = mol.parameters['h_sp'].expand(nD,-1).reshape(-1)

        nb_diag = maskd.numel()
        maskd_exp = (
            maskd.view(1, nb_diag)                             # (1, nb_diag)
            + mol.atom_molid.view(1, nb_diag) * extra_per_mol      # add (nD-1)*molsize^2 per molecule
            + dgrid * base_stride                              # add d*molsize^2 within molecule
        )  # (nD, nb_diag)
        md_flat = maskd_exp.reshape(-1)                 # (nD*nb_diag,)
        P_md = P_anti[md_flat]                               # (nD*nb_diag, 4,4)
        F2e1c = torch.zeros_like(P_md)
        for i in range(1,4):
            #(s,p) = (p,s) upper triangle
            F2e1c[...,0,i] = P_md[...,0,i]*(0.5*hsp - 0.5*gsp)
        #(p,p*)
        for i,j in [(1,2),(1,3),(2,3)]:
            F2e1c[...,i,j] = P_md[...,i,j]* (0.25*gpp - 0.75*gp2)

        F2e1c.add_(F2e1c.triu(1).transpose(-1,-2),alpha=-1.0)
        F.index_add_(0, md_flat, F2e1c)
        del P_anti
        del F2e1c

    F0 = F.reshape(nmol,nD,molsize,molsize,4,4) \
             .transpose(3,4) \
             .reshape(nmol,nD, 4*molsize, 4*molsize)
    del F

    norb_max = int(torch.max(mol.norb))
    # Alternatively, norb_max = P_xi.shape[2]
    F0 = torch.stack([
        packone(F0[i,j], 4*nHeavy[i], nHydro[i], norb_max)
        for i in range(nmol) for j in range(nD)
    ])

    return F0.view(nmol,nD,norb_max,norb_max)

def makeA_pi_symm_any_batch(mol,P0,w):

    P0_sym = 0.5*(P0 + P0.transpose(2,3))
    nD = P0.shape[1] # number of roots

    molsize = mol.molsize
    dtype = P0.dtype
    device = P0.device

    mask  = mol.mask
    maskd = mol.maskd
    mask_l = mol.mask_l
    idxi  = mol.idxi
    idxj  = mol.idxj
    nmol = mol.nmol

    P = P0_sym.reshape(nmol,nD,molsize,4,molsize,4)\
              .transpose(3,4).reshape(nmol*nD*molsize*molsize,4,4)
    del P0_sym
    F = torch.zeros_like(P)
    base_stride = molsize * molsize                 # blocks per density per molecule
    extra_per_mol = (nD - 1) * base_stride          # extra shift added to each molecule when stacking nD
    nb_diag = maskd.numel()
    npairs_ut = mask.numel()
    # npairs_pairdiag = idxi.numel()
    npairs_pairdiag = npairs_ut
    # ---- Expand indices for ALL densities (no loops over nD) ----
    dgrid = torch.arange(nD, device=device).view(nD, 1)

    # Diagonal (A,A) blocks per density
    # maskd indexes density=0; shift to density d using molecule owner
    maskd_exp = (
        maskd.view(1, nb_diag)                             # (1, nb_diag)
        + mol.atom_molid.view(1, nb_diag) * extra_per_mol      # add (nD-1)*molsize^2 per molecule
        + dgrid * base_stride                              # add d*molsize^2 within molecule
    )  # (nD, nb_diag)

    # Upper-tri (A,B) blocks per density
    mask_exp = (
        mask.view(1, npairs_ut)
        + mol.pair_molid.view(1, npairs_ut) * extra_per_mol
        + dgrid * base_stride
    )  # (nD, npairs_ut)
    
    # For pair-diagonal (A,B) contractions using idxi/idxj into maskd:
    # gather the per-density diagonal indices for A and B
    md_i = maskd_exp[:, idxi]   # (nD, npairs_pairdiag)
    md_j = maskd_exp[:, idxj]   # (nD, npairs_pairdiag)
    # sum px,py,pz

    # ========== 1) One-center two-electron-like terms on diagonal blocks ==========
    gss = mol.parameters['g_ss'].expand(nD,-1).reshape(-1) 
    gsp = mol.parameters['g_sp'].expand(nD,-1).reshape(-1)
    gpp = mol.parameters['g_pp'].expand(nD,-1).reshape(-1)
    gp2 = mol.parameters['g_p2'].expand(nD,-1).reshape(-1)
    hsp = mol.parameters['h_sp'].expand(nD,-1).reshape(-1)
    # Build F2e1c for all densities and all diag blocks in one go
    md_flat = maskd_exp.reshape(-1)                 # (nD*nb_diag,)
    P_md = P[md_flat]                               # (nD*nb_diag, 4,4)
    Pptot_md = P_md[..., 1, 1] + P_md[..., 2, 2] + P_md[..., 3, 3]                  # (nD*nb_diag,)

    F2e1c = torch.zeros_like(P_md)

    # (s,s)
    F2e1c[..., 0, 0] = 0.5 * P_md[..., 0, 0] * gss + Pptot_md * (gsp - 0.5 * hsp)

    # (p,p) diagonals and (s,p) off-diagonals in upper triangle
    for i in range(1, 4):
        F2e1c[..., i, i] = (
            P_md[..., 0, 0] * (gsp - 0.5 * hsp)
            + 0.5 * P_md[..., i, i] * gpp
            + (Pptot_md - P_md[..., i, i]) * (1.25 * gp2 - 0.25 * gpp)
        )
        F2e1c[..., 0, i] = P_md[..., 0, i] * (1.5 * hsp - 0.5 * gsp)

    # (p,p*) off-diagonals
    for i, j in [(1, 2), (1, 3), (2, 3)]:
        F2e1c[..., i, j] = P_md[..., i, j] * (0.75 * gpp - 1.25 * gp2)

    # Scatter-add to F
    F.index_add_(0, md_flat, F2e1c)

    # ========== 2) Two-center Coulomb-like sums over neighbor atoms ==========
    # Pack A and B for all densities × pairs
    tri_i = torch.tensor([0,0,1,0,1,2,0,1,2,3], device=device)
    tri_j = torch.tensor([0,1,1,2,2,2,3,3,3,3], device=device)
    weight = torch.tensor([1.0,
                           2.0, 1.0,
                           2.0, 2.0, 1.0,
                           2.0, 2.0, 2.0, 1.0],dtype=dtype, device=device)
    PA = (P[md_i.reshape(-1)][..., tri_i, tri_j] * weight).view(nD, npairs_pairdiag, 10, 1)
    PB = (P[md_j.reshape(-1)][..., tri_i, tri_j] * weight).view(nD, npairs_pairdiag, 1, 10)

    # Broadcast w over densities (no memory blow-up)
    w_b = w.unsqueeze(0)  # (1, npairs_pairdiag, 10,10)

    # Contractions:
    suma = torch.sum(PA * w_b, dim=2)  # (nD, npairs_pairdiag, 10)   Σ_{μν∈A} P_A(μν) (μν|λσ)
    sumb = torch.sum(PB * w_b, dim=3)  # (nD, npairs_pairdiag, 10)   Σ_{λσ∈B} P_B(λσ) (μν|λσ)

    # Unpack back to 4x4
    sumA = torch.zeros(nD, npairs_pairdiag, 4, 4, dtype=dtype, device=device)
    sumB = torch.zeros_like(sumA)
    sumA[..., tri_i, tri_j] = suma
    sumB[..., tri_i, tri_j] = sumb

    # Scatter-add: add Σ_B to A-diagonals, and Σ_A to B-diagonals
    F.index_add_(0, md_i.reshape(-1), sumB.reshape(-1, 4, 4))
    F.index_add_(0, md_j.reshape(-1), sumA.reshape(-1, 4, 4))

    # ========== 3) Off-diagonal exchange-like AB blocks (K-type) ==========
    # sum[...,i,j] = Σ_{ν∈A} Σ_{σ∈B} [ -0.5 P_{νσ} * (μν | λσ) ]  using your 10x10 pack map
    ind = torch.tensor([[0, 1, 3, 6],
                        [1, 2, 4, 7],
                        [3, 4, 5, 8],
                        [6, 7, 8, 9]], dtype=torch.int64, device=device)

    mask_flat = mask_exp.reshape(-1)                # (nD*npairs_ut,)
    Pp = -0.5 * P[mask_flat].view(nD, npairs_ut, 4, 4)  # (nD, npairs_ut, 4,4)
    sum_K = torch.zeros_like(Pp)

    # We’ll broadcast w (shape (1, npairs_ut, 10, 10)) across densities.
    w_K = w.unsqueeze(0)

    for i in range(4):
        Wi = w_K[..., ind[i], :]            # (1, npairs_ut, 4, 10)
        for j in range(4):
            Wij = Wi[..., :, ind[j]]        # (1, npairs_ut, 4, 4)
            # elementwise multiply with Pp and sum over ν,σ (the two middle dims)
            sum_K[..., i, j] = torch.sum(Pp * Wij, dim=(2, 3))  # (nD, npairs_ut)

    # Scatter-add to F at AB positions
    F.index_add_(0, mask_flat, sum_K.reshape(-1, 4, 4))

    mask_l_exp = (
        mask_l.view(1, -1)
        + mol.pair_molid.view(1, npairs_ut) * extra_per_mol
        + dgrid * base_stride
    ).reshape(-1)

    F[mask_l_exp] = F[mask_flat].transpose(-1, -2)
    F[maskd_exp] += F[maskd_exp].triu(1).transpose(-1,-2)

    return F


def get_subspace_eig_any_batched(H,nroots,max_nroots,zero_pad,e_val_n,done,nonorthogonal):

    if nonorthogonal:
        raise NotImplementedError("Non-orthogonal davidson not yet implemented")

    else:
        r_eval, r_evec = torch.linalg.eigh(H[~done]) # find the eigenvalues and the eigenvectors

        nmol, subspacesize = H.shape[0], H.shape[1]
        e_vec_n = torch.zeros(nmol, subspacesize, max_nroots, device=H.device, dtype=H.dtype)

        active_indices = torch.nonzero(~done, as_tuple=False).squeeze(1)
        
        # Update eigenvalues and eigenvectors for each active molecule.
        for j, mol_idx in enumerate(active_indices):
            k = int(nroots[mol_idx].item())
            s = int(zero_pad[mol_idx].item())
            e_val_n[mol_idx, :k]      = r_eval[j, s : s + k]
            e_vec_n[mol_idx, :, :k]   = r_evec[j, :, s : s + k]
        return e_vec_n

def calc_transition_dipoles_any_batch(mol,amplitudes,excitation_energies,nroots,dipole_mat,rpa=False):

    nocc, nvirt, Cocc, Cvirt = get_occ_virt(mol)
    norb = Cocc.shape[1]

    if rpa:
        amp_ia_X = amplitudes[0].view(mol.nmol,nroots,nocc,nvirt)
        amp_ia_Y = amplitudes[1].view(mol.nmol,nroots,nocc,nvirt)
    else:
        amp_ia_X = amplitudes.view(mol.nmol,nroots,nocc,nvirt)

    nHeavy = mol.nHeavy
    nHydro = mol.nHydro
    dipole_mat_packed = torch.stack([
        packone(dipole_mat[i,j], 4*nHeavy[i], nHydro[i], norb)
        for i in range(mol.nmol) for j in range(3) ]).view(mol.nmol,3,norb,norb)
    # dipole_mat_packed = packone_batch(dipole_mat.view(3*mol.nmol,4*mol.molsize,4*mol.molsize), 4*nHeavy, nHydro, norb).view(mol.nmol,3,norb,norb)


    # CIS transition density R = \sum_ia C_\mu i * t_ia * C_\nu a 
    R = torch.einsum('bmi,bria,bna->brmn',Cocc,amp_ia_X,Cvirt)
    if rpa:
        R += torch.einsum('bma,bria,bni->brmn',Cvirt,amp_ia_Y,Cocc)

    # Transition dipole in AU as calculated in NEXMD
    transition_dipole = torch.einsum('brmn,bdmn->brd',R,dipole_mat_packed)*math.sqrt(2.0)/a0
    hartree = 27.2113962 # value used in NEXMD
    oscillator_strength = 2.0/3.0*excitation_energies/hartree*torch.square(transition_dipole).sum(dim=2)
    return transition_dipole, oscillator_strength


def make_guess_any_batch(ea_ei,nroots,maxSubspacesize,V,nmol,nov_batch):
    # Make the davidson guess vectors
    sorted_ediff, sortedidx = torch.sort(ea_ei, stable=True, descending=False) # stable to preserve the order of degenerate orbitals

    # Per-molecule degeneracy expansion starting from base nroots
    nroots_per_mol = torch.empty(nmol, dtype=torch.long, device=V.device)
    for i in range(nmol):
        nov_i = int(nov_batch[i].item())
        k = min(nroots, nov_i)
        # include all states degenerate with the last chosen one
        while k < nov_i and torch.abs(sorted_ediff[i, k] - sorted_ediff[i, k - 1]) < 1e-5:
            k += 1
        nroots_per_mol[i] = k
        if k > nroots:
            print(f"Increasing the number of states calculated from {nroots} to {k} because of orbital degeneracies")
        # Respect maxSubspacesize; if degeneracy pushed past it, we’ll clamp

    nroots_eff = torch.minimum(nroots_per_mol, torch.tensor(maxSubspacesize, device=V.device))

    # Choose extra subspace per molecule, while keeping 2*nroots + extra <= maxSubspacesize
    # and not exceeding available dimensionality
    extra_by_avail = torch.clamp(nov_batch - nroots_eff, min=0)
    extra = torch.minimum(
        torch.minimum(extra_by_avail, torch.full_like(extra_by_avail, 7)),
        torch.clamp(maxSubspacesize - 2 * nroots_eff, min=0),
    )

    nstart_per_mol = torch.minimum(nroots_eff + extra, torch.tensor(maxSubspacesize, device=V.device))

    # Fill V with one-hot guesses for each molecule up to its own nstart
    for i in range(nmol):
        k = int(nstart_per_mol[i].item())
        V[i, torch.arange(k, device=V.device), sortedidx[i, :k]] = 1.0

    return nstart_per_mol, nroots_per_mol, int(nroots_per_mol.max().item())

def calc_cis_energy_any_batch(mol, w, e_mo, amplitude,rpa=False):

    norb_batch, nocc_batch, nmol = mol.norb, mol.nocc, mol.nmol
    nvirt_batch = norb_batch - nocc_batch
    norb, nocc = int(torch.max(norb_batch).item()), int(torch.max(nocc_batch).item())
    nvirt = int(torch.max(nvirt_batch))

    C = mol.molecular_orbitals
    Cocc = torch.zeros(nmol,norb,nocc,device=w.device,dtype=w.dtype)
    Cvirt = torch.zeros(nmol,norb,nvirt,device=w.device,dtype=w.dtype)
    for i in range(nmol):
        Cocc[i,:,:nocc_batch[i]] = C[i,:,:nocc_batch[i]]
        Cvirt[i,:,:nvirt_batch[i]] = C[i,:,nocc_batch[i]:norb_batch[i]]

    ea_ei = torch.zeros(nmol,nocc,nvirt,device=w.device,dtype=w.dtype)
    for i in range(nmol):
        ea_ei[i,:nocc_batch[i],:nvirt_batch[i]] = e_mo[i,nocc_batch[i]:norb_batch[i]].unsqueeze(0) - e_mo[i,:nocc_batch[i]].unsqueeze(1)

    if not rpa: #CIS: w = XAX
        HV = matrix_vector_product_any_batched(mol, amplitude.unsqueeze(1), w, ea_ei, Cocc, Cvirt)
        E_cis = torch.linalg.vecdot(amplitude,HV.squeeze(1))
    else: # RPA
        #  w = (X Y)(A B)(X) = X(AX+BY) + Y(BX+AY)
        #           (B A)(Y)
        X = amplitude[0,...]
        Y = amplitude[1,...]
        AX, BX= matrix_vector_product_any_batched(mol, X.unsqueeze(1), w, ea_ei, Cocc, Cvirt,makeB=True)
        AY, BY= matrix_vector_product_any_batched(mol, Y.unsqueeze(1), w, ea_ei, Cocc, Cvirt,makeB=True)
        E_cis = torch.linalg.vecdot(X,(AX+BY).squeeze(1)) + torch.linalg.vecdot(Y,(BX+AY).squeeze(1))

    # For calculating excitation energy gradient with backprop
    # L = E_cis.sum()
    # L.backward(create_graph=False,retain_graph=True)
    # force = mol.coordinates.grad.clone()
    # with torch.no_grad(): mol.coordinates.grad.zero_()
    # torch.set_printoptions(precision=15)
    # print(f'E_cis is {E_cis}')
    # print(f'Grad CIS from backprop is\n{force}')

    return E_cis


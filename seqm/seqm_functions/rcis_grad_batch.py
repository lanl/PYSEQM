import torch
from seqm.seqm_functions.anal_grad import overlap_der_finiteDiff, w_der, core_core_der
from seqm.seqm_functions.cg_solver import conjugate_gradient_batch
from seqm.seqm_functions.rcis_batch import makeA_pi_batched, unpackone_batch
from .constants import a0

def rcis_grad_batch(mol, w, e_mo, riXH, ri, P0, zvec_tolerance,gam,method,parnuc,rpa=False,include_ground_state=False):
    """
    amp: tensor of CIS amplitudes of shape [b,nov]. For each of the b molecules, the CIS amplitues of the 
         state for which the gradient is required has to be selected and put together into the amp tensor
    """
    amp = mol.cis_amplitudes[...,mol.active_state-1,:]
    device = amp.device
    dtype = amp.dtype
    norb = mol.norb[0]
    nocc = mol.nocc[0]
    nvirt = norb - nocc
    nov = nocc*nvirt

    # CIS unrelaxed density B = \sum_iab C_\mu a * t_ai * t_bi * C_\nu b - \sum_ija C_\mu i * t_ai * t_aj * C_\nu j 
    C = mol.eig_vec 
    Cocc = C[:,:,:nocc]
    Cvirt = C[:,:,nocc:norb]
    nmol = mol.nmol
    if rpa:
        amp_ia_X = amp[0].view(mol.nmol,nocc,nvirt)
        amp_ia_Y = amp[1].view(mol.nmol,nocc,nvirt)
    else:
        amp_ia_X = amp.view(mol.nmol,nocc,nvirt)

    dens_BR = torch.empty(nmol,2,norb,norb,device=device,dtype=dtype)
    B_virt  = torch.einsum('Nma,Nia->Nmi',Cvirt,amp_ia_X)
    B_occ  = torch.einsum('Nmi,Nia->Nma',Cocc,amp_ia_X)

    dens_BR[:,0] = torch.einsum('Nmi,Nni->Nmn',B_virt,B_virt) - torch.einsum('Nmi,Nni->Nmn',B_occ,B_occ)

    if rpa:
        B_virt_Y  = torch.einsum('Nma,Nia->Nmi',Cvirt,amp_ia_Y)
        B_occ_Y  = torch.einsum('Nmi,Nia->Nma',Cocc,amp_ia_Y)
        dens_BR[:,0] += torch.einsum('Nmi,Nni->Nmn',B_virt_Y,B_virt_Y) - torch.einsum('Nmi,Nni->Nmn',B_occ_Y,B_occ_Y)


    # CIS transition density R = \sum_ia C_\mu i * t_ia * C_\nu a 
    dens_BR[:,1] = torch.einsum('bmi,bia,bna->bmn',Cocc,amp_ia_X,Cvirt)
    if rpa:
        dens_BR[:,1] += torch.einsum('bma,bia,bni->bmn',Cvirt,amp_ia_Y,Cocc)

    # Calculate z-vector
    
    # make RHS of the CPSCF equation:
    BR_pi = makeA_pi_batched(mol,dens_BR,w)*2.0
    RHS = -torch.einsum('Nma,Nmn,Nni->Nai',Cvirt,BR_pi[:,0],Cocc)
    RHS -= torch.einsum('Nma,Nmn,Nni->Nai',Cvirt,BR_pi[:,1],B_virt)
    RHS += torch.einsum('Nma,Nmn,Nni->Nai',B_occ,BR_pi[:,1],Cocc)
    if rpa:
        RHS -= torch.einsum('Nma,Nnm,Nni->Nai',Cvirt,BR_pi[:,1],B_virt_Y)
        RHS += torch.einsum('Nma,Nnm,Nni->Nai',B_occ_Y,BR_pi[:,1],Cocc)

    del B_occ, B_virt

    # debugging:
    RHS = RHS.transpose(1,2).reshape(nmol,nov) # RHS_ia 
    ea_ei = e_mo[:,nocc:norb].unsqueeze(1)-e_mo[:,:nocc].unsqueeze(2)

    # Ad_inv_b = RHS/ea_ei
    # x1 = make_A_times_zvector(mol,Ad_inv_b,w,e_mo)

    def setup_applyA(mol, w, ea_ei, Cocc, Cvirt):
        def applyA(z):
            Az = make_A_times_zvector_batched(mol,z,w,ea_ei, Cocc, Cvirt)
            return Az

        return applyA

    A = setup_applyA(mol,w,ea_ei,Cocc,Cvirt)
    zvec = conjugate_gradient_batch(A,RHS,ea_ei.view(nmol,nocc*nvirt),tol=zvec_tolerance)

    z_ao = torch.einsum('Nmi,Nia,Nna->Nmn',Cocc,zvec.view(nmol,nocc,nvirt),Cvirt)
    dens_BR[:,0] += z_ao + z_ao.transpose(1,2) # Now this contains the relaxed density

    molsize = mol.molsize
    nHeavy = mol.nHeavy[0]
    nHydro = mol.nHydro[0]
    # B0 = torch.stack([ unpackone(dens_BR[i,0], 4*nHeavy, nHydro, molsize * 4)
    #     for i in range(nmol)]).view(nmol,molsize * 4, molsize * 4)
    B0 = unpackone_batch(dens_BR[:,0],4*nHeavy, nHydro, molsize * 4)
    # R0 = torch.stack([ unpackone(dens_BR[i,1], 4*nHeavy, nHydro, molsize * 4)
    #     for i in range(nmol)]).view(nmol,molsize * 4, molsize * 4)
    R0 = unpackone_batch(dens_BR[:,1],4*nHeavy, nHydro, molsize * 4)
    
    del dens_BR

    ###############################
    # Calculate the gradient of CIS energies

    # TODO: instead of repeating the calculation of gradient of the overlap matrix and the 2-e integral matrix w_x, store it and reuse it, while calculating ground state
    # gradients. Alternately, combine ground and excited state gradients
    npairs = mol.rij.shape[0]
    overlap_x = torch.zeros((npairs, 3, 4, 4), dtype=dtype, device=device)
    zeta = torch.cat((mol.parameters['zeta_s'].unsqueeze(1), mol.parameters['zeta_p'].unsqueeze(1)), dim=1)
    Xij = mol.xij * mol.rij.unsqueeze(1) * a0
    overlap_der_finiteDiff(overlap_x, mol.idxi, mol.idxj, mol.rij, Xij, mol.parameters['beta'], mol.ni, mol.nj, zeta, mol.const.qn_int)

    w_x = torch.zeros(mol.rij.shape[0], 3, 10, 10, dtype=dtype, device=device)
    e1b_x, e2a_x = w_der(mol.const, mol.Z, mol.const.tore, mol.ni, mol.nj, w_x, mol.rij, mol.xij, Xij, mol.idxi, mol.idxj, \
                         mol.parameters['g_ss'], mol.parameters['g_pp'], mol.parameters['g_p2'], mol.parameters['h_sp'], mol.parameters['zeta_s'], mol.parameters['zeta_p'], riXH, ri)

    B = B0.reshape(nmol, molsize, 4, molsize, 4).transpose(2, 3).reshape(nmol * molsize * molsize, 4, 4)
    P = P0.reshape(nmol, molsize, 4, molsize, 4).transpose(2, 3).reshape(nmol * molsize * molsize, 4, 4)
    if include_ground_state:
        pair_grad = core_core_der(mol, gam, w_x, method, parnuc)
        B += 0.5*P
        # Typically you add ground state density to excited state density if you want to include gradient of ground state energy in the gradient of excited state energy.
        # But there is a factor of 2 when contracting excited state density with two-electron gradient matrix (not sure why), but not for ground state density. 
        # That's why I add 0.5 times the ground state density to the excited state density. 
        # In doing so, I have to also add 0.5 time the contraction of ground state density with the one-electron gradient matrix. I add 0.5 time the overlap contribution here and 0.5 time core-valence term e1b_x and e2a_x below.
        pair_grad += 0.5*(P[mol.mask].unsqueeze(1) * overlap_x).sum(dim=(2, 3))
    else:
        pair_grad = torch.zeros_like(Xij)
        
    
    # The following logic to form the coulomb and exchange integrals by contracting the two-electron integrals with the density matrix has been cribbed from fock.py

    # Exchange integrals
    # mu, nu in A
    # lambda, sigma in B
    # F_mu_lambda = Hcore - 0.5* \sum_{nu \in A} \sum_{sigma in B} P_{nu, sigma} * (mu nu, lambda, sigma)
    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #   0,     1         2       3       4         5       6      7         8        9
    ind = torch.tensor([[0, 1, 3, 6],\
                        [1, 2, 4, 7],\
                        [3, 4, 5, 8],\
                        [6, 7, 8, 9]], dtype=torch.int64, device=device)
    # mask has the indices of the lower (or upper) triangle blocks of the density matrix. Hence, P[mask] gives
    # us access to P_mu_lambda where mu is on atom A, lambda is on atom B
    overlap_KAB_x = overlap_x
    Pp = P[mol.mask].unsqueeze(1)
    # half_multiply = 1.0 # 0.5
    for i in range(4):
        w_x_i = w_x[..., ind[i], :]
        for j in range(4):
            # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            overlap_KAB_x[..., i, j] -= torch.sum(Pp * (w_x_i[..., :, ind[j]]), dim=(2, 3))

    pair_grad += (B[mol.mask].unsqueeze(1) * overlap_KAB_x).sum(dim=(2, 3))

    # Coulomb integrals -- only on the diagonal
    #F_mu_nv = Hcore + \sum^B \sum_{lambda, sigma} P^B_{lambda, sigma} * (mu nu, lambda sigma)
    #as only upper triangle part is done, and put in order
    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #weight for them are
    #  1       2       1        2        2        1        2       2        2       1
    weight = torch.tensor([1.0,
                           2.0, 1.0,
                           2.0, 2.0, 1.0,
                           2.0, 2.0, 2.0, 1.0],dtype=dtype, device=device).reshape((-1,10))
    # weight *= 0.5  # Multiply the weight by 0.5 because the contribution of coulomb integrals to engergy is calculated as 0.5*P_mu_nu*F_mu_nv

    indices = (0, 0, 1, 0, 1, 2, 0, 1, 2, 3), (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)
    PA = (P[mol.maskd[mol.idxi]][..., indices[0], indices[1]] * weight).unsqueeze(-1)  # Shape: (npairs, 10, 1)
    PB = (P[mol.maskd[mol.idxj]][..., indices[0], indices[1]] * weight).unsqueeze(-2)  # Shape: (npairs, 1, 10)

    suma = torch.sum(PA.unsqueeze(1) * w_x, dim=2)  # Shape: (npairs, 3, 10)

    scale_emat = torch.tensor([ [1.0, 2.0, 2.0, 2.0],
                                [0.0, 1.0, 2.0, 2.0],
                                [0.0, 0.0, 1.0, 2.0],
                                [0.0, 0.0, 0.0, 1.0] ],dtype=dtype, device=device)
    if include_ground_state:
        pair_grad.add_(0.5*(P[mol.maskd[mol.idxj], None, :, :] * e2a_x*scale_emat).sum(dim=(2, 3)) +
                       0.5*(P[mol.maskd[mol.idxi], None, :, :] * e1b_x*scale_emat).sum(dim=(2, 3)))

    # Collect in sumA and sumB tensors
    # reususe overlap_KAB_x here instead of creating new arrays
    # I am going to be alliasing overlap_KAB_x to sumA and then further aliasing it to sumB
    # This seems like bad practice because I'm not allocating new memory but using the same tensor for all operations.
    # In the future, if this code is to be edited, be careful here
    sumA = overlap_KAB_x
    sumA.zero_()
    sumA[..., indices[0], indices[1]] = suma
    e2a_x.add_(sumA)

    sumB = overlap_KAB_x
    sumB.zero_()
    sumb = torch.sum(PB.unsqueeze(1) * w_x, dim=3)  # Shape: (npairs, 3, 10)
    sumB[..., indices[0], indices[1]] = sumb
    del suma, sumb
    e1b_x.add_(sumB)

    e1b_x *= scale_emat
    e2a_x *= scale_emat   
    # e1b_x.add_(e1b_x.triu(1).transpose(2, 3))
    # e2a_x.add_(e2a_x.triu(1).transpose(2, 3))
    pair_grad.add_((B[mol.maskd[mol.idxj], None, :, :] * e2a_x).sum(dim=(2, 3)) +
                   (B[mol.maskd[mol.idxi], None, :, :] * e1b_x).sum(dim=(2, 3)))
    del e1b_x

    ########################################################### 
    
    R_symmetrized = 0.5*(R0+R0.transpose(1,2))
    R_symm = R_symmetrized.reshape(nmol, molsize, 4, molsize, 4).transpose(2, 3).reshape(nmol * molsize * molsize, 4, 4)
    del R_symmetrized

    Rdiag_symmetrized = R_symm[mol.maskd]
    PA = (Rdiag_symmetrized[mol.idxi][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).unsqueeze(-1)
    PB = (Rdiag_symmetrized[mol.idxj][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).unsqueeze(-2)

    suma = torch.sum(PA.unsqueeze(1) * w_x, dim=2)  # Shape: (npairs, 3, 10)
    sumA = overlap_KAB_x
    sumA.zero_()
    sumA[..., indices[0], indices[1]] = suma
    J_x_2a = e2a_x
    J_x_2a[:,:,:] = sumA

    sumB = overlap_KAB_x
    sumB.zero_()
    sumb = torch.sum(PB.unsqueeze(1) * w_x, dim=3)  # Shape: (npairs, 3, 10)
    sumB[..., indices[0], indices[1]] = sumb
    J_x_1b = sumB
    del suma, sumb

    # Core-elecron interaction
    # J_x_1b.add_(J_x_1b.triu(1).transpose(2, 3))
    # J_x_2a.add_(J_x_2a.triu(1).transpose(2, 3))
    J_x_1b *= scale_emat
    J_x_2a *= scale_emat   
    pair_grad.add_((2.0*R_symm[mol.maskd[mol.idxj], None, :, :] * J_x_2a).sum(dim=(2, 3)) +
                   (2.0*R_symm[mol.maskd[mol.idxi], None, :, :] * J_x_1b).sum(dim=(2, 3))) # I can use R_symm instead of R here
    del J_x_2a
    del Rdiag_symmetrized

    Pp = R_symm[mol.mask].unsqueeze(1)
    for i in range(4):
        w_x_i = w_x[..., ind[i], :]
        for j in range(4):
            # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            overlap_KAB_x[..., i, j] = -0.5*torch.sum(Pp * (w_x_i[..., :, ind[j]]), dim=(2, 3))

    pair_grad.add_((4.0*R_symm[mol.mask].unsqueeze(1) * overlap_KAB_x).sum(dim=(2, 3)))
    del R_symm

    R_antisymmetrized = 0.5*(R0-R0.transpose(1,2))
    R_antisymm = R_antisymmetrized.reshape(nmol, molsize, 4, molsize, 4).transpose(2, 3).reshape(nmol * molsize * molsize, 4, 4)
    del R_antisymmetrized
    Pp = R_antisymm[mol.mask].unsqueeze(1)
    for i in range(4):
        w_x_i = w_x[..., ind[i], :]
        for j in range(4):
            # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            overlap_KAB_x[..., i, j] = -0.5*torch.sum(Pp * (w_x_i[..., :, ind[j]]), dim=(2, 3))

    pair_grad.add_((4.0*R_antisymm[mol.mask].unsqueeze(1) * overlap_KAB_x).sum(dim=(2, 3)))

    # Define the gradient tensor
    grad_cis = torch.zeros(nmol * molsize, 3, dtype=dtype, device=device)

    grad_cis.index_add_(0, mol.idxi, pair_grad)
    grad_cis.index_add_(0, mol.idxj, pair_grad, alpha=-1.0)

    grad_cis = grad_cis.view(nmol, molsize, 3)

    # torch.set_printoptions(precision=9)
    # print(f'Analytical CIS gradient is (eV/Angstrom):\n{grad_cis}')

    return grad_cis

def make_A_times_zvector_batched(mol, z, w, ea_ei, Cocc, Cvirt):
    nmol  = mol.nmol
    norb = mol.norb[0]
    nocc = mol.nocc[0]
    nvirt = norb - nocc

    Via = z.view(nmol,nocc,nvirt) 
    P_xi = torch.einsum('Nmi,Nia,Nna->Nmn', Cocc,Via, Cvirt)
    P_xi = P_xi + P_xi.transpose(1,2)
    
    F0 = makeA_pi_batched(mol,P_xi.unsqueeze(1),w,allSymmetric=True)
    A = torch.einsum('Nmi,Nmn,Nna->Nia', Cocc, F0.squeeze(1),Cvirt)*2.0
    A += Via*ea_ei

    return A.view(nmol,-1)

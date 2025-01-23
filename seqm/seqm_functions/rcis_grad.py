import torch
from torch.nn.modules.conv import F
from seqm.seqm_functions.anal_grad import overlap_der_finiteDiff, w_der
from seqm.seqm_functions.rcis import makeA_pi
from seqm.seqm_functions.cg_solver import conjugate_gradient
from seqm.seqm_functions.pack import unpackone
from .constants import a0

def rcis_grad(mol, amp, w, e_mo, riXH, ri, P0):
    device = amp.device
    dtype = amp.dtype
    norb = mol.norb[0]
    nocc = mol.nocc[0]
    nvirt = norb - nocc

    # CIS unrelaxed density B = \sum_iab C_\mu a * t_ai * t_bi * C_\nu b - \sum_ija C_\mu i * t_ai * t_aj * C_\nu j 
    C = mol.eig_vec 
    Cocc = C[0,:,:nocc]
    Cvirt = C[0,:,nocc:norb]
    # occ = slice(None,nocc)
    # virt = slice(nocc,norb)
    amp_ia = amp.view(nocc,nvirt)

    dens_BR = torch.empty(2,norb,norb,device=device,dtype=dtype)

    B_virt  = torch.matmul(Cvirt,amp_ia.T)
    B_occ  = torch.matmul(Cocc,amp_ia)
    dens_BR[0] = torch.matmul(B_virt,B_virt.T) - torch.matmul(B_occ,B_occ.T)
    del B_occ, B_virt

    # CIS transition density R = \sum_ia C_\mu i * t_ia * C_\nu a 
    dens_BR[1] = torch.einsum('mi,ia,na',Cocc,amp_ia,Cvirt)

    # Calculate z-vector
    
    # make RHS of the CPSCF equation:
    # FIXME: Not sure if I should be using BR_pi or transpose of BR_pi because R is not symmetric and hence R.Pi will not be symmetric
    BR_pi = makeA_pi(mol,dens_BR,w)*2.0
    RHS = -torch.einsum('ma,mn,ni->ai',Cvirt,BR_pi[0],Cocc)
    RHS -= torch.einsum('ma,mn,nb,ib->ai',Cvirt,BR_pi[1],Cvirt,amp_ia)
    RHS += torch.einsum('ja,mj,mn,ni->ai',amp_ia,Cocc,BR_pi[1],Cocc)

    # debugging:
    RHS = RHS.T.contiguous() # RHS_ia 
    ea_ei = e_mo[0,nocc:norb].unsqueeze(0)-e_mo[0,:nocc].unsqueeze(1)

    # Ad_inv_b = RHS/ea_ei
    # x1 = make_A_times_zvector(mol,Ad_inv_b,w,e_mo)

    def setup_applyA(mol, w, e_mo):
        def applyA(z):
            Az = make_A_times_zvector(mol,z,w,e_mo)
            return Az

        return applyA

    A = setup_applyA(mol,w,e_mo)
    zvec = conjugate_gradient(A,RHS.view(1,nocc*nvirt),M=ea_ei.view(nocc*nvirt),tol=1e-8)

    z_ao = torch.einsum('mi,ia,na',Cocc,zvec.view(nocc,nvirt),Cvirt)
    dens_BR[0] += z_ao + z_ao.T # Now this contains the relaxed density

    molsize = mol.molsize
    B0 = unpackone(dens_BR[0], 4*mol.nHeavy[0], mol.nHydro[0], molsize * 4)
    R0 = unpackone(dens_BR[1], 4*mol.nHeavy[0], mol.nHydro[0], molsize * 4)



    ###############################
    # Calculate the gradient of CIS energies

    npairs = mol.rij.shape[0]
    overlap_x = torch.zeros((npairs, 3, 4, 4), dtype=dtype, device=device)
    zeta = torch.cat((mol.parameters['zeta_s'].unsqueeze(1), mol.parameters['zeta_p'].unsqueeze(1)), dim=1)
    Xij = mol.xij * mol.rij.unsqueeze(1) * a0
    overlap_der_finiteDiff(overlap_x, mol.idxi, mol.idxj, mol.rij, Xij, mol.parameters['beta'], mol.ni, mol.nj, zeta, mol.const.qn_int)

    w_x = torch.zeros(mol.rij.shape[0], 3, 10, 10, dtype=dtype, device=device)
    e1b_x, e2a_x = w_der(mol.const, mol.Z, mol.const.tore, mol.ni, mol.nj, w_x, mol.rij, mol.xij, Xij, mol.idxi, mol.idxj, \
                         mol.parameters['g_ss'], mol.parameters['g_pp'], mol.parameters['g_p2'], mol.parameters['h_sp'], mol.parameters['zeta_s'], mol.parameters['zeta_p'], riXH, ri)

    B = B0.reshape(1, molsize, 4, molsize, 4).transpose(2, 3).reshape(1 * molsize * molsize, 4, 4)
    P = P0.reshape(1, molsize, 4, molsize, 4).transpose(2, 3).reshape(1 * molsize * molsize, 4, 4)
    
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

    pair_grad = (B[mol.mask].unsqueeze(1) * overlap_KAB_x).sum(dim=(2, 3))

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

    # Core-elecron interaction
    e1b_x.add_(e1b_x.triu(1).transpose(2, 3))
    e2a_x.add_(e2a_x.triu(1).transpose(2, 3))
    pair_grad.add_((B[mol.maskd[mol.idxj], None, :, :] * e2a_x).sum(dim=(2, 3)) +
                   (B[mol.maskd[mol.idxi], None, :, :] * e1b_x).sum(dim=(2, 3)))
    del e1b_x

    ########################################################### 
    
    R_symmetrized = 0.5*(R0+R0.transpose(0,1))
    R_symm = R_symmetrized.reshape(1, molsize, 4, molsize, 4).transpose(2, 3).reshape(1 * molsize * molsize, 4, 4)
    del R_symmetrized

    R_antisymmetrized = 0.5*(R0-R0.transpose(0,1))
    R_antisymm = R_antisymmetrized.reshape(1, molsize, 4, molsize, 4).transpose(2, 3).reshape(1 * molsize * molsize, 4, 4)
    del R_antisymmetrized

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
    J_x_1b.add_(J_x_1b.triu(1).transpose(2, 3))
    J_x_2a.add_(J_x_2a.triu(1).transpose(2, 3))
    R= R0.reshape(1, molsize, 4, molsize, 4).transpose(2, 3).reshape(1 * molsize * molsize, 4, 4)
    pair_grad.add_((2.0*R[mol.maskd[mol.idxj], None, :, :] * J_x_2a).sum(dim=(2, 3)) +
                   (2.0*R[mol.maskd[mol.idxi], None, :, :] * J_x_1b).sum(dim=(2, 3))) # I can use R_symm instead of R here
    del J_x_2a
    del Rdiag_symmetrized

    Pp = R_symm[mol.mask].unsqueeze(1)
    for i in range(4):
        w_x_i = w_x[..., ind[i], :]
        for j in range(4):
            # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            overlap_KAB_x[..., i, j] = -0.5*torch.sum(Pp * (w_x_i[..., :, ind[j]]), dim=(2, 3))

    pair_grad.add_((2.0*R[mol.mask].unsqueeze(1) * overlap_KAB_x).sum(dim=(2, 3)))
    pair_grad.add_((2.0*R[mol.mask_l].unsqueeze(1) * overlap_KAB_x.transpose(2,3)).sum(dim=(2, 3)))

    Pp = R_antisymm[mol.mask].unsqueeze(1)
    for i in range(4):
        w_x_i = w_x[..., ind[i], :]
        for j in range(4):
            # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            overlap_KAB_x[..., i, j] = -0.5*torch.sum(Pp * (w_x_i[..., :, ind[j]]), dim=(2, 3))

    pair_grad.add_((2.0*R[mol.mask].unsqueeze(1) * overlap_KAB_x).sum(dim=(2, 3)))
    pair_grad.add_((2.0*R[mol.mask_l].unsqueeze(1) * overlap_KAB_x.transpose(2,3)).sum(dim=(2, 3)),alpha=-1.0)

    # Define the gradient tensor
    nmol = 1
    grad_cis = torch.zeros(nmol * molsize, 3, dtype=dtype, device=device)

    grad_cis.index_add_(0, mol.idxi, pair_grad)
    grad_cis.index_add_(0, mol.idxj, pair_grad, alpha=-1.0)

    torch.set_printoptions(precision=9)
    print(f'Analytical CIS gradient is:\n{grad_cis.view(nmol,molsize,3)}')
    grad_cis = grad_cis.reshape(nmol, molsize, 3)
    return

def make_A_times_zvector(mol, z, w, e_mo):
    C = mol.eig_vec
    nmol  = mol.nmol
    norb = mol.norb[0]
    nocc = mol.nocc[0]
    nvirt = norb - nocc
    ea_ei = e_mo[0,nocc:norb].unsqueeze(0)-e_mo[0,:nocc].unsqueeze(1)

    # From here I assume there is only one molecule. I'll worry about batching later
    if (nmol != 1):
        raise Exception("Not yet implemented for more than one molecule")

    Via = z.view(-1,nocc,nvirt) 
    P_xi = torch.einsum('mi,ria,na->rmn', C[0, :, :nocc],Via, C[0, :, nocc:])
    P_xi = P_xi + P_xi.transpose(1,2)
    
    F0 = makeA_pi(mol,P_xi,w,allSymmetric=True)
    A = torch.einsum('mi,rmn,na->ria', C[0, :, :nocc], F0,C[0,:, nocc: ])*2.0
    A += Via*ea_ei.unsqueeze(0)

    nnewRoots = z.shape[0]
    return A.view(nnewRoots,-1)

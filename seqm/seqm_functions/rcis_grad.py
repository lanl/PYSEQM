import torch
from seqm.seqm_functions.anal_grad import overlap_der_finiteDiff, w_der
from seqm.seqm_functions.rcis import makeA_pi
from seqm.seqm_functions.cg_solver import conjugate_gradient
from seqm.seqm_functions.pack import unpackone
from .constants import a0, ev

def rcis_grad(mol, amp, w, e_mo, riXH, ri):

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
    zvec = conjugate_gradient(A,RHS.view(1,nocc*nvirt),M=ea_ei.view(nocc*nvirt))

    z_ao = torch.einsum('mi,ia,na',Cocc,zvec.view(nocc,nvirt),Cvirt)
    dens_BR[0] += z_ao + z_ao.T # Now this contains the relaxed density

    molsize = mol.molsize
    B0 = unpackone(dens_BR[0], 4*mol.nHeavy[0], mol.nHydro[0], molsize * 4)
    R0 = unpackone(dens_BR[1], 4*mol.nHeavy[0], mol.nHydro[0], molsize * 4)



    ###############################
    # Calculate the gradient of CIS energies

    grad_cis = torch.zeros(molsize, 3, dtype=dtype, device=device)

    npairs = mol.rij.shape[0]
    overlap_x = torch.zeros((npairs, 3, 4, 4), dtype=dtype, device=device)
    zeta = torch.cat((mol.parameters['zeta_s'].unsqueeze(1), mol.parameters['zeta_p'].unsqueeze(1)), dim=1)
    Xij = mol.xij * mol.rij.unsqueeze(1) * a0
    overlap_der_finiteDiff(overlap_x, mol.idxi, mol.idxj, mol.rij, Xij, mol.parameters['beta'], mol.ni, mol.nj, zeta, mol.const.qn_int)

    
    w_x = torch.zeros(mol.rij.shape[0], 3, 10, 10, dtype=dtype, device=device)
    e1b_x, e2a_x = w_der(mol.const, mol.Z, mol.const.tore, mol.ni, mol.nj, w_x, mol.rij, mol.xij, Xij, mol.idxi, mol.idxj, \
                         mol.parameters['g_ss'], mol.parameters['g_pp'], mol.parameters['g_p2'], mol.parameters['h_sp'], mol.parameters['zeta_s'], mol.parameters['zeta_p'], riXH, ri)

    B = B0.reshape(1, molsize, 4, molsize, 4).transpose(2, 3).reshape(1 * molsize * molsize, 4, 4)
    
    # The following logic to form the coulomb and exchange integrals by contracting the two-electron integrals with the density matrix has been cribbed from fock.py

    # Exchange integrals
    # mu, nu in A
    # lambda, sigma in B
    # F_mu_lambda = Hcore - 0.5* \sum_{nu \in A} \sum_{sigma in B} P_{nu, sigma} * (mu nu, lambda, sigma)
    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #   0,     1         2       3       4         5       6      7         8        9
    ind = torch.tensor([[0, 1, 3, 6], [1, 2, 4, 7], [3, 4, 5, 8], [6, 7, 8, 9]], dtype=torch.int64, device=device)
    # mask has the indices of the lower (or upper) triangle blocks of the density matrix. Hence, P[mask] gives
    # us access to P_mu_lambda where mu is on atom A, lambda is on atom B
    overlap_KAB_x = overlap_x
    Pp = P[mol.mask].unsqueeze(1)
    for i in range(4):
        w_x_i = w_x[..., ind[i], :]
        for j in range(4):
            # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            overlap_KAB_x[..., i, j] -= 0.5 * torch.sum(Pp * (w_x_i[..., :, ind[j]]), dim=(2, 3))

    pair_grad = (Pp * overlap_KAB_x).sum(dim=(2, 3))

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

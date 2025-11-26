import torch
from seqm.seqm_functions.anal_grad import overlap_der_finiteDiff, w_der
from seqm.seqm_functions.rcis_batch import unpackone_batch
from .constants import a0

def calc_nac(mol, amp, e_exc, P0, ri, riXH, state1, state2,rpa=False):
    """
    amp: tensor of CIS amplitudes of shape [b,nov]. For each of the b molecules, the CIS amplitues of the 
         state for which the gradient is required has to be selected and put together into the amp tensor
    """
    if rpa:
        raise NotImplementedError("Nonadiabatic coupling vecotrs not yet implemented for RPA. Use CIS instead.")
    device = amp.device
    dtype = amp.dtype
    norb = mol.norb[0]
    nocc = mol.nocc[0]
    nvirt = norb - nocc

    # CIS unrelaxed density B = \sum_iab C_\mu a * t_ai * t_bi * C_\nu b - \sum_ija C_\mu i * t_ai * t_aj * C_\nu j 
    C = mol.molecular_orbitals 
    Cocc = C[:,:,:nocc]
    Cvirt = C[:,:,nocc:norb]
    nmol = mol.nmol
    nroots = amp.shape[1]
    amp_ia = amp.view(nmol,nroots,nocc,nvirt)

    # print(f"Going to calculate NAC vector between States {state1} and {state2}")
    state1 = state1 - 1
    state2 = state2 - 1

    
    BIJ  = torch.einsum('Nma,Nia,Nib,Nnb->Nmn',Cvirt,amp_ia[:,state1],amp_ia[:,state2],Cvirt) \
              - torch.einsum('Nmi,Nia,Nja,Nnj->Nmn',Cocc,amp_ia[:,state1],amp_ia[:,state2],Cocc)

    BIJ = 0.5*(BIJ+BIJ.transpose(1,2))
    # Now, BIJ is the symmetrized difference density matrix between states 1 and 2

    molsize = mol.molsize
    nHeavy = mol.nHeavy[0]
    nHydro = mol.nHydro[0]
    
    B0 = unpackone_batch(BIJ,4*nHeavy, nHydro, molsize * 4)
    del BIJ

    ###############################
    # Calculate the gradient of CIS energies

    # TODO: instead of repeating the calculation of gradient of the overlap matrix and the 2-e integral matrix w_x, store it and reuse it.
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
    del B0
    
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
    scale_emat = torch.tensor([ [1.0, 2.0, 2.0, 2.0],
                                [0.0, 1.0, 2.0, 2.0],
                                [0.0, 0.0, 1.0, 2.0],
                                [0.0, 0.0, 0.0, 1.0] ],dtype=dtype,device=device)
    e1b_x *= scale_emat
    e2a_x *= scale_emat   
    # e1b_x.add_(e1b_x.triu(1).transpose(2, 3))
    # e2a_x.add_(e2a_x.triu(1).transpose(2, 3))
    pair_grad.add_((B[mol.maskd[mol.idxj], None, :, :] * e2a_x).sum(dim=(2, 3)) +
                   (B[mol.maskd[mol.idxi], None, :, :] * e1b_x).sum(dim=(2, 3)))
    del e1b_x

    # Define the gradient tensor
    nac_cis = torch.zeros(nmol * molsize, 3, dtype=dtype, device=device)

    nac_cis.index_add_(0, mol.idxi, pair_grad)
    nac_cis.index_add_(0, mol.idxj, pair_grad, alpha=-1.0)
    
    nac_cis = nac_cis/(e_exc[:,state2]-e_exc[:,state1]).unsqueeze(1)

    nac_cis = nac_cis.view(nmol, molsize, 3)
    
    torch.set_printoptions(precision=9)
    print(f'NAC vectors between CIS states (Angstrom^-1) {state1+1} and {state2+1} is:\n{nac_cis}')

    return nac_cis

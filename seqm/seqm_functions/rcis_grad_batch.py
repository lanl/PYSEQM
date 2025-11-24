import torch
from seqm.seqm_functions.anal_grad import overlap_der_finiteDiff, w_der, core_core_der
from seqm.seqm_functions.rcis_batch import makeA_pi_batched, unpackone_batch, make_cis_densities
from .constants import a0
from .dispersion_am1_fs1 import dEdisp_dr

def rcis_grad_batch(mol, w, e_mo, riXH, ri, P0, zvec_tolerance,gam,method,parnuc,rpa=False,include_ground_state=False, orbital_window = None, calculate_dipole=False):
    """
    amp: tensor of CIS amplitudes of shape [b,nov]. For each of the b molecules, the CIS amplitues of the 
         state for which the gradient is required has to be selected and put together into the amp tensor
    """
    molsize = mol.molsize
    nHeavy = mol.nHeavy[0]
    nHydro = mol.nHydro[0]
    cis_densities = make_cis_densities(mol, do_transition_denisty=True,do_difference_density=True,
                                       do_relaxed_density=True, orbital_window = orbital_window,
                                       w=w,e_mo=e_mo,zvec_tolerance=zvec_tolerance,rpa=rpa)
    if calculate_dipole:
        make_cis_state_dipole(mol, cis_densities["difference_density"], cis_densities["relaxed_difference_density"], P0)
    # B0 = torch.stack([ unpackone(dens_BR[i,0], 4*nHeavy, nHydro, molsize * 4)
    #     for i in range(nmol)]).view(nmol,molsize * 4, molsize * 4)
    B0 = unpackone_batch(cis_densities["relaxed_difference_density"],4*nHeavy, nHydro, molsize * 4)
    # R0 = torch.stack([ unpackone(dens_BR[i,1], 4*nHeavy, nHydro, molsize * 4)
    #     for i in range(nmol)]).view(nmol,molsize * 4, molsize * 4)
    R0 = unpackone_batch(cis_densities["transition_density"],4*nHeavy, nHydro, molsize * 4)
    
    del cis_densities

    ###############################
    # Calculate the gradient of CIS energies

    # TODO: instead of repeating the calculation of gradient of the overlap matrix and the 2-e integral matrix w_x, store it and reuse it, while calculating ground state
    # gradients. Alternately, combine ground and excited state gradients
    npairs = mol.rij.shape[0]
    dtype = B0.dtype
    device = B0.device
    nmol = mol.nmol
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
        if mol.seqm_parameters.get("dispersion",False) and method == "AM1":
            pair_grad += dEdisp_dr(mol)
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

    # torch.set_printoptions(precision=15)
    # print(f'Analytical CIS gradient is (eV/Angstrom):\n{grad_cis}')

    return grad_cis

from .rcis_batch import packone_batch, calc_dipole_matrix
from .constants import debye_to_AU, to_debye

def make_cis_state_dipole(mol, difference_density, relaxed_difference_density, P0):
    dipole_mat = calc_dipole_matrix(mol) 
    nHeavy = mol.nHeavy[0]
    nHydro = mol.nHydro[0]
    norb = mol.norb[0]
    dipole_mat_packed = packone_batch(dipole_mat.view(3*mol.nmol,4*mol.molsize,4*mol.molsize), 4*nHeavy, nHydro, norb).view(mol.nmol,3,norb,norb)

    mol.cis_state_unrelaxed_dipole = torch.einsum('Nnm,Ndnm->Nd',difference_density,dipole_mat_packed)*to_debye*debye_to_AU + mol.dipole
    mol.cis_state_relaxed_dipole = torch.einsum('Nnm,Ndnm->Nd',relaxed_difference_density,dipole_mat_packed)*to_debye*debye_to_AU + mol.dipole



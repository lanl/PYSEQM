import torch
from torch import pow
from .constants import a0, ev
# from .constants import sto6g_coeff, sto6g_exponent
from .cal_par import *
from .diat_overlap import diatom_overlap_matrix
from .two_elec_two_center_int import two_elec_two_center_int as TETCI
from .energy import pair_nuclear_energy


# @profile
def scf_analytic_grad(P, const, method, mask, maskd, molsize, idxi, idxj, ni, nj, xij, rij, gam, parnuc, Z, gss, gpp,
                      gp2, hsp, beta, zetas, zetap, riXH, ri):
    """
    Calculate the gradient of the ground state SCF energy
    in the units of ev/Angstrom
    The code follows the derivation outlined in 
    Dewar, Michael JS, and Yukio Yamaguchi. "Analytical first derivatives of the energy in MNDO." Computers & Chemistry 2.1 (1978): 25-29.
    https://doi.org/10.1016/0097-8485(78)80005-9
    """
    # torch.set_printoptions(precision=6)
    # torch.set_printoptions(linewidth=110)

    # Xij (= Xj-Xi) is the vector from j to i in Angstroms
    # xij (= xj-xi) is the *unit* vector from j to i
    Xij = xij * rij.unsqueeze(1) * a0
    dtype = Xij.dtype
    device = Xij.device
    nmol = P.shape[0]
    npairs = Xij.shape[0]
    qn_int = const.qn_int  # Principal quantum number of the valence shell

    # Define the gradient tensor
    grad = torch.zeros(nmol * molsize, 3, dtype=dtype, device=device)

    # I will use this tensor to store the gradient of the overlap matrix elements, and then that of the exchange integrals
    overlap_KAB_x = torch.zeros((npairs, 3, 4, 4), dtype=dtype, device=device)

    # overlap_der(overlap_KAB_x,zetas,zetap,qn_int,ni,nj,rij,beta,idxi,idxj,Xij)
    # We will use finite-differnce for the overlap derivative because analytical expression for derivatives of
    # the overlap of slater orbitals is v complicated
    zeta = torch.cat((zetas.unsqueeze(1), zetap.unsqueeze(1)), dim=1)
    overlap_der_finiteDiff(overlap_KAB_x, idxi, idxj, rij, Xij, beta, ni, nj, zeta, qn_int)

    # Core-core repulsion derivatives
    # First, derivative of g_AB
    tore = const.tore  # Charges
    alpha = parnuc[0]
    ZAZB = tore[ni] * tore[nj]

    # Two-center repulsion integral derivatives
    # Core-valence integral derivatives e1b_x and e2a_x also calculated as byproducts
    w_x = torch.zeros(rij.shape[0], 3, 10, 10, dtype=dtype, device=device)
    e1b_x, e2a_x = w_der(const, Z, tore, ni, nj, w_x, rij, xij, Xij, idxi, idxj, \
                         gss, gpp, gp2, hsp, zetas, zetap, riXH, ri)

    # Derivative of pair-nuclear Energy or E_core-core
    # pair_grad = torch.zeros((npairs,3),dtype=dtype, device=device)
    pair_grad = core_core_der(alpha, rij, Xij, ZAZB, ni, nj, idxi, idxj, gam, w_x, method, parameters=parnuc)

    # Assembly
    P0 = P.reshape(nmol, molsize, 4, molsize, 4).transpose(2, 3).reshape(nmol * molsize * molsize, 4, 4)

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
    Pp = -0.5 * P0[mask]
    for i in range(4):
        for j in range(4):
            # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            overlap_KAB_x[..., i, j] += torch.sum(Pp.unsqueeze(1) * (w_x[..., ind[i], :][..., :, ind[j]]), dim=(2, 3))

    pair_grad.add_((P0[mask, None, :, :] * overlap_KAB_x).sum(dim=(2, 3)))

    # Coulomb integrals
    #F_mu_nv = Hcore + \sum^B \sum_{lambda, sigma} P^B_{lambda, sigma} * (mu nu, lambda sigma)
    #as only upper triangle part is done, and put in order
    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #weight for them are
    #  1       2       1        2        2        1        2       2        2       1

    weight = torch.tensor([1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0], dtype=dtype, device=device).reshape(
        (-1, 10))
    weight *= 0.5  # Multiply the weight by 0.5 because the contribution of coulomb integrals to engergy is calculated as 0.5*P_mu_nu*F_mu_nv

    indices = (0, 0, 1, 0, 1, 2, 0, 1, 2, 3), (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)
    PA = (P0[maskd[idxi]][..., indices[0], indices[1]] * weight).unsqueeze(-1)  # Shape: (npairs, 10, 1)
    PB = (P0[maskd[idxj]][..., indices[0], indices[1]] * weight).unsqueeze(-2)  # Shape: (npairs, 1, 10)

    suma = torch.sum(PA.unsqueeze(1) * w_x, dim=2)  # Shape: (npairs, 3, 10)
    sumb = torch.sum(PB.unsqueeze(1) * w_x, dim=3)  # Shape: (npairs, 3, 10)

    # Collect in sumA and sumB tensors
    # reususe overlap_KAB_x here instead of creating new arrays
    # I am going to be alliasing overlap_KAB_x to sumA and then further aliasing it to sumB
    # This seems like bad practice because I'm not allocating new memory but using the same tensor for all operations.
    # In the future if this code is edited be careful here
    sumA = overlap_KAB_x
    sumA.zero_()
    sumA[..., indices[0], indices[1]] = suma
    e2a_x.add_(sumA)

    sumB = overlap_KAB_x
    sumB.zero_()
    sumB[..., indices[0], indices[1]] = sumb
    e1b_x.add_(sumB)

    # Core-elecron interaction
    e1b_x.add_(e1b_x.triu(1).transpose(2, 3))
    e2a_x.add_(e2a_x.triu(1).transpose(2, 3))
    pair_grad.add_((P0[maskd[idxj], None, :, :] * e2a_x).sum(dim=(2, 3)) +
                   (P0[maskd[idxi], None, :, :] * e1b_x).sum(dim=(2, 3)))
    # pair_grad.add_((P0[maskd[idxj],None,:,:]*e2a_x.triu(1)).sum(dim=(2,3)) + (P0[maskd[idxi],None,:,:]*e1b_x.triu(1)).sum(dim=(2,3)))

    grad.index_add_(0, idxi, pair_grad)
    grad.index_add_(0, idxj, pair_grad, alpha=-1.0)

    # print(f'Analytical SCF gradient is:\n{grad.view(nmol,molsize,3)}')
    grad = grad.reshape(nmol, molsize, 3)
    return grad


# @profile
def scf_grad(P, molecule, const, method, mask, maskd, molsize, idxi, idxj, ni, nj, xij, rij, parnuc, Z, gss, gpp, gp2,
             hsp, beta, zetas, zetap):
    """
    Calculate the gradient of the ground state SCF energy
    in the units of ev/Angstrom
    The gradient is calculated in a pseudo-numerical fashion. The derivatives of the overlap, the core-core repulsions and the two-electron integrals in
    the atomic orbital basis are calculated using finite-differnce.
    """
    # torch.set_printoptions(precision=6)
    # torch.set_printoptions(linewidth=110)

    # Xij (= Xj-Xi) is the vector from j to i in Angstroms
    # xij (= xj-xi) is the *unit* vector from j to i
    Xij = xij * rij.unsqueeze(1) * a0
    dtype = Xij.dtype
    device = Xij.device
    nmol = P.shape[0]
    npairs = Xij.shape[0]
    qn_int = const.qn_int  # Principal quantum number of the valence shell

    # Define the gradient tensor
    grad = torch.zeros(nmol * molsize, 3, dtype=dtype, device=device)

    # I will use this tensor to store the gradient of the overlap matrix, and then that of the exchange integrals
    overlap_KAB_x = torch.zeros((npairs, 3, 4, 4), dtype=dtype, device=device)

    # overlap_der(overlap_KAB_x,zetas,zetap,qn_int,ni,nj,rij,beta,idxi,idxj,Xij)
    # We will use finite-differnce for the overlap derivative because analytical expression for derivatives of
    # the overlap of slater orbitals is v complicated
    zeta = torch.cat((zetas.unsqueeze(1), zetap.unsqueeze(1)), dim=1)
    overlap_der_finiteDiff(overlap_KAB_x, idxi, idxj, rij, Xij, beta, ni, nj, zeta, qn_int)

    # verify with finite difference
    delta = 1e-5
    e1b_x = torch.zeros(rij.shape[0], 3, 4, 4, device=device, dtype=dtype)
    e2a_x = torch.zeros(rij.shape[0], 3, 4, 4, device=device, dtype=dtype)
    w_x = torch.zeros(rij.shape[0], 3, 10, 10, device=device, dtype=dtype)
    pair_grad = torch.zeros(rij.shape[0], 3, device=device, dtype=dtype)
    for coord in range(3):
        # since Xij = Xj-Xi, when I want to do Xi+delta, I have to subtract delta from from Xij
        Xij[:, coord] -= delta
        rij_ = torch.norm(Xij, dim=1)
        xij_ = Xij / rij_.unsqueeze(1)
        rij_ = rij_ / a0
        w_plus, e1b_plus, e2a_plus, rho0xi,rho0xj, _, _ = TETCI(const, idxi, idxj, ni, nj, xij_, rij_, Z, \
                                        molecule.parameters['zeta_s'], molecule.parameters['zeta_p'], molecule.parameters['zeta_d'],\
                                        molecule.parameters['s_orb_exp_tail'], molecule.parameters['p_orb_exp_tail'], molecule.parameters['d_orb_exp_tail'],\
                                        molecule.parameters['g_ss'], molecule.parameters['g_pp'], molecule.parameters['g_p2'], molecule.parameters['h_sp'],\
                                        molecule.parameters['F0SD'], molecule.parameters['G2SD'], molecule.parameters['rho_core'],\
                                        molecule.alp, molecule.chi, molecule.method)
        gam_ = w_plus[:, 0, 0]
        EnucAB_plus = pair_nuclear_energy(Z, const, nmol, ni, nj, idxi, idxj, rij_, rho0xi, rho0xj, \
                                          molecule.alp, molecule.chi, gam=gam_, method=method, parameters=parnuc)
        Xij[:, coord] += 2.0 * delta
        rij_ = torch.norm(Xij, dim=1)
        xij_ = Xij / rij_.unsqueeze(1)
        rij_ = rij_ / a0
        w_minus, e1b_minus, e2a_minus, rho0xi,rho0xj, _, _ = TETCI(const, idxi, idxj, ni, nj, xij_, rij_, Z,\
                                        molecule.parameters['zeta_s'], molecule.parameters['zeta_p'], molecule.parameters['zeta_d'],\
                                        molecule.parameters['s_orb_exp_tail'], molecule.parameters['p_orb_exp_tail'], molecule.parameters['d_orb_exp_tail'],\
                                        molecule.parameters['g_ss'], molecule.parameters['g_pp'], molecule.parameters['g_p2'], molecule.parameters['h_sp'],\
                                        molecule.parameters['F0SD'], molecule.parameters['G2SD'], molecule.parameters['rho_core'],\
                                        molecule.alp, molecule.chi, molecule.method)
        gam_ = w_minus[:, 0, 0]
        EnucAB_minus = pair_nuclear_energy(Z, const, nmol, ni, nj, idxi, idxj, rij_, rho0xi, rho0xj, \
                                           molecule.alp, molecule.chi, gam=gam_, method=method, parameters=parnuc)
        Xij[:, coord] -= delta

        e1b_x[:, coord, ...] = (e1b_plus - e1b_minus) / (2.0 * delta)
        e2a_x[:, coord, ...] = (e2a_plus - e2a_minus) / (2.0 * delta)
        w_x[:, coord, ...] = (w_plus - w_minus) / (2.0 * delta)
        pair_grad[:, coord] = (EnucAB_plus - EnucAB_minus) / (2.0 * delta)

    # Assembly
    P0 = P.reshape(nmol, molsize, 4, molsize, 4).transpose(2, 3).reshape(nmol * molsize * molsize, 4, 4)

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
    Pp = -0.5 * P0[mask]
    for i in range(4):
        for j in range(4):
            # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            overlap_KAB_x[..., i, j] += torch.sum(Pp.unsqueeze(1) * (w_x[..., ind[i], :][..., :, ind[j]]), dim=(2, 3))

    pair_grad.add_((P0[mask, None, :, :] * overlap_KAB_x).sum(dim=(2, 3)))

    # Coulomb integrals
    #F_mu_nv = Hcore + \sum^B \sum_{lambda, sigma} P^B_{lambda, sigma} * (mu nu, lambda sigma)
    #as only upper triangle part is done, and put in order
    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #weight for them are
    #  1       2       1        2        2        1        2       2        2       1

    weight = torch.tensor([1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0], dtype=dtype, device=device).reshape(
        (-1, 10))
    weight *= 0.5  # Multiply the weight by 0.5 because the contribution of coulomb integrals to engergy is calculated as 0.5*P_mu_nu*F_mu_nv

    indices = (0, 0, 1, 0, 1, 2, 0, 1, 2, 3), (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)
    PA = (P0[maskd[idxi]][..., indices[0], indices[1]] * weight).unsqueeze(-1)  # Shape: (npairs, 10, 1)
    PB = (P0[maskd[idxj]][..., indices[0], indices[1]] * weight).unsqueeze(-2)  # Shape: (npairs, 1, 10)

    suma = torch.sum(PA.unsqueeze(1) * w_x, dim=2)  # Shape: (npairs, 3, 10)
    sumb = torch.sum(PB.unsqueeze(1) * w_x, dim=3)  # Shape: (npairs, 3, 10)

    # Collect in sumA and sumB tensors
    # reususe overlap_KAB_x here instead of creating new arrays
    # I am going to be alliasing overlap_KAB_x to sumA and then further aliasing it to sumB
    # This seems like bad practice because I'm not allocating new memory but using the same tensor for all operations.
    # In the future if this code is edited be careful here
    sumA = overlap_KAB_x
    sumA.zero_()
    sumA[..., indices[0], indices[1]] = suma
    e2a_x.add_(sumA)

    sumB = overlap_KAB_x
    sumB.zero_()
    sumB[..., indices[0], indices[1]] = sumb
    e1b_x.add_(sumB)

    # Core-elecron interaction
    e1b_x.add_(e1b_x.triu(1).transpose(2, 3))
    e2a_x.add_(e2a_x.triu(1).transpose(2, 3))
    pair_grad.add_((P0[maskd[idxj], None, :, :] * e2a_x).sum(dim=(2, 3)) +
                   (P0[maskd[idxi], None, :, :] * e1b_x).sum(dim=(2, 3)))
    # pair_grad.add_((P0[maskd[idxj],None,:,:]*e2a_x.triu(1)).sum(dim=(2,3)) + (P0[maskd[idxi],None,:,:]*e1b_x.triu(1)).sum(dim=(2,3)))

    grad.index_add_(0, idxi, pair_grad)
    grad.index_add_(0, idxj, pair_grad, alpha=-1.0)

    # print(f'SCF gradient is:\n{grad.view(nmol,molsize,3)}')
    grad = grad.reshape(nmol, molsize, 3)
    return grad


# def overlap_der(overlap_KAB_x,zetas,zetap,qn_int,ni,nj,rij,beta,idxi,idxj,Xij):
#     if torch.any(qn_int[ni]>1):
#         raise Exception("Not yet implemented for molecules with non-Hydrogen atoms")
#     a0_sq = a0*a0
#
#     # (sA|sB) overlap
#     C_times_C = torch.einsum('bi,bj->bij',sto6g_coeff[qn_int[ni]-1,0],sto6g_coeff[qn_int[nj]-1,0])
#
#     alpha1 = sto6g_exponent[qn_int[ni]-1,0,:]*(zetas[idxi].unsqueeze(1)**2)
#     alpha2 = sto6g_exponent[qn_int[nj]-1,0,:]*(zetas[idxj].unsqueeze(1)**2)
#
#     # alpha_i*alpha_j/(alpha_i+alpha_j)
#     alpha_product = alpha1.unsqueeze(2) * alpha2.unsqueeze(1)  # Shape: (batch_size, vector_size, vector_size)
#     alpha_sum = alpha1[..., None] + alpha2[..., None, :]
#     alphas_1 = alpha_product / alpha_sum  # Shape: (batch_size, vector_size, vector_size)
#
#     # <sA|sB>ij
#     # From MOPAC the arugment of the exponential is not allowed to exceed -35 (presumably because exp(-35) ~ double precision minimum)
#     sij = ((2.0*torch.div(torch.sqrt(alpha_product),alpha_sum))**(3/2))*torch.exp(-1.0*(alphas_1*(rij[:,None,None]**2)).clamp_(max=35.0))
#
#     # d/dx of <sA|sB>
#     sAsB = 2.0*alphas_1*sij
#
#     # Dividing with a0^2 beacuse we want gradients in ev/ang. Remember, alpha(gaussian exponent) has units of (bohr)^-2
#     # There is no dividing beta_mu+beta_nu by 2. Found this out during debugging.
#     # Possibly because here we're only going over unique pairs, but in the total energy
#     # expression the overlap term appears on the upper and lower triangle of Hcore
#     # and hence needs to be multiplied by 2.
#     overlap_KAB_x[:,:,0,0] = ((beta[idxi,0]+beta[idxj,0])*torch.sum(C_times_C*sAsB,dim=(1,2))).unsqueeze(1)*Xij[:,:]/a0_sq
#
#     '''
#     #(px|s)
#     C_times_C = torch.einsum('bi,bj->bij',sto6g_coeff[qn_int[ni]-1,1],sto6g_coeff[qn_int[nj]-1,0])
#
#     alpha1 = sto6g_exponent[qn_int[ni]-1,1,:]*(zetas[idxi].unsqueeze(1)**2)
#     alpha2 = sto6g_exponent[qn_int[nj]-1,0,:]*(zetas[idxj].unsqueeze(1)**2)
#
#     # alpha_i*alpha_j/(alpha_i+alpha_j)
#     alpha_product = alpha1.unsqueeze(2) * alpha2.unsqueeze(1)  # Shape: (batch_size, vector_size, vector_size)
#     alpha_sum = alpha1[..., None] + alpha2[..., None, :]
#     alphas_1 = alpha_product / alpha_sum  # Shape: (batch_size, vector_size, vector_size)
#
#     # <sA|sB>ij
#     # From MOPAC the arugment of the exponential is not allowed to exceed -35 (presumably because exp(-35) ~ double precision minimum)
#     sij = ((2.0*torch.div(torch.sqrt(alpha_product),alpha_sum))**(3/2))*torch.exp(-1.0*(alphas_1*(rij[:,None,None]**2)).clamp_(max=35.0))
#
#     # d/dx of <sA|sB>
#     sAsB = 2.0*alphas_1*sij
#
#     # Dividing with a0^2 beacuse we want gradients in ev/ang. Remember, alpha(gaussian exponent) has units of (bohr)^-2
#     # There is no dividing beta_mu+beta_nu by 2. Found this out during debugging.
#     # Possibly because here we're only going over unique pairs, but in the total energy
#     # expression the overlap term appears on the upper and lower triangle of Hcore
#     # and hence needs to be multiplied by 2.
#     overlap_KAB_x[:,:,0,0] = ((beta[idxi,0]+beta[idxj,0])*torch.sum(C_times_C*sAsB,dim=(1,2))).unsqueeze(1)*Xij[:,:]/a0_sq
#     '''
#     print(f'overlap_x from gaussians is \n{overlap_KAB_x}')


def core_core_der(alpha, rij, Xij, ZAZB, ni, nj, idxi, idxj, gam, w_x, method, parameters):
    rija = rij * a0
    # special case for N-H and O-H
    XH = ((ni == 7) | (ni == 8)) & (nj == 1)
    t2 = torch.zeros_like(rij)
    tmp = torch.exp(-alpha[idxi] * rija)
    t2[~XH] = tmp[~XH]
    t2[XH] = tmp[XH] * rija[XH]
    t3 = torch.exp(-alpha[idxj] * rija)
    g = 1.0 + t2 + t3

    # For MNDO, core-core term is ZAZB*(SASA|SBSB)*g, where g=1+exp(-alpha_A*RAB)+exp(-alpha_B*RAB)
    prefactor = alpha[idxi]
    prefactor[XH] = prefactor[XH] * rija[XH] - 1.0
    t3 = alpha[idxj] * torch.exp(-alpha[idxj] * rija)
    coreTerm = ZAZB * gam / rija * (prefactor * tmp + t3)
    # The derivative of the core-core term is ZAZB*(SASA|SBSB)*dg/dx + ZAZB*g*d(SASA|SBSB)/dx
    # Here we calculate the first term, i.e., ZAZB*(SASA|SBSB)*dg/dx
    pair_grad = coreTerm.unsqueeze(1) * Xij
    # Here we add the second term in the derivative for MNDO, i.e., ZAZB*g*d(SASA|SBSB)/dx
    pair_grad.add_((ZAZB * g).unsqueeze(1) * w_x[:, :, 0, 0])
    if method == 'MNDO':
        return pair_grad

    # if method=='PM6':
    #     # Here we don't have the MNDO term, so pair_grad has to be reinitialized
    #     pair_grad.zero_()

    # For AM1 and PM3, in addition to the MNDO term we also have
    #two gaussian terms for PM3
    # 3~4 terms for AM1
    _, K, L, M = parameters
    #K, L , M shape (natoms,2 or 4)
    t4 = ZAZB / rija
    t5 = torch.sum(K[idxi] * torch.exp(-L[idxi] * (rija[:, None] - M[idxi])**2), dim=1)
    t6 = torch.sum(K[idxj] * torch.exp(-L[idxj] * (rija[:, None] - M[idxj])**2), dim=1)
    pair_grad.add_((ZAZB * torch.pow(rija, -3) * (t5 + t6)).unsqueeze(1) * Xij)
    t5_der = torch.sum(K[idxi] * torch.exp(-L[idxi] * (rija[:, None] - M[idxi])**2) * L[idxi] *
                       (rija[:, None] - M[idxi]),
                       dim=1)
    t6_der = torch.sum(K[idxj] * torch.exp(-L[idxj] * (rija[:, None] - M[idxj])**2) * L[idxj] *
                       (rija[:, None] - M[idxj]),
                       dim=1)
    pair_grad.add_((2.0 * t4 / rija * (t5_der + t6_der)).unsqueeze(1) * Xij)
    if method == 'PM3' or method == 'AM1':
        return pair_grad
    # Put PM6 specific grad here
    # if method=='PM6':
    #       return pair_grad
    else:
        raise ValueError("Supported Method: MNDO, AM1, PM3")


def w_der(const, Z, tore, ni, nj, w_x, rij, xij, Xij, idxi, idxj, gss, gpp, gp2, hsp, zetas, zetap, riXH, ri):
    # Two-center repulsion integral derivatives
    HH = (ni == 1) & (nj == 1)
    XH = (ni > 1) & (nj == 1)
    XX = (ni > 1) & (nj > 1)
    qn = const.qn
    hpp = 0.5 * (gpp - gp2)
    qn0 = qn[Z]
    isH = Z == 1  # Hydrogen
    isX = Z > 2  # Heavy atom
    rho_0 = torch.zeros_like(qn0)
    rho_1 = torch.zeros_like(qn0)
    rho_2 = torch.zeros_like(qn0)
    dd = torch.zeros_like(qn0)
    qq = torch.zeros_like(qn0)
    rho1 = additive_term_rho1.apply
    rho2 = additive_term_rho2.apply

    dd[isX], qq[isX] = dd_qq(qn0[isX], zetas[isX], zetap[isX])
    rho_0[isH] = 0.5 * ev / gss[isH]
    rho_0[isX] = 0.5 * ev / gss[isX]
    if torch.sum(isX) > 0:
        rho_1[isX] = rho1(hsp[isX], dd[isX])
        rho_2[isX] = rho2(hpp[isX], qq[isX])

    der_TETCILF(w_x, const, ni, nj, xij, Xij, rij, dd[idxi], dd[idxj], qq[idxi], qq[idxj], rho_0[idxi], rho_0[idxj],
                rho_1[idxi], rho_1[idxj], rho_2[idxi], rho_2[idxj], tore, riXH, ri)

    # # Why is rij in bohr? It should be in angstrom right? Ans: OpenMopac website seems to suggest using bohr as well
    # # for the 2-e integrals.

    # Core-elecron interaction
    e1b_x = torch.zeros((rij.shape[0], 3, 4, 4), dtype=w_x.dtype, device=w_x.device)
    e2a_x = torch.zeros((rij.shape[0], 3, 4, 4), dtype=w_x.dtype, device=w_x.device)
    nonHH = ~HH
    e1b_x[:, :, 0, 0] = -tore[nj].unsqueeze(1) * w_x[:, :, 0, 0]
    e2a_x[:, :, 0, 0] = -tore[ni].unsqueeze(1) * w_x[:, :, 0, 0]
    e1b_x[nonHH, :, 0, 1] = -tore[nj[nonHH]].unsqueeze(1) * w_x[nonHH, :, 1, 0]
    e1b_x[nonHH, :, 1, 1] = -tore[nj[nonHH]].unsqueeze(1) * w_x[nonHH, :, 2, 0]
    e1b_x[nonHH, :, 0, 2] = -tore[nj[nonHH]].unsqueeze(1) * w_x[nonHH, :, 3, 0]
    e1b_x[nonHH, :, 1, 2] = -tore[nj[nonHH]].unsqueeze(1) * w_x[nonHH, :, 4, 0]
    e1b_x[nonHH, :, 2, 2] = -tore[nj[nonHH]].unsqueeze(1) * w_x[nonHH, :, 5, 0]
    e1b_x[nonHH, :, 0, 3] = -tore[nj[nonHH]].unsqueeze(1) * w_x[nonHH, :, 6, 0]
    e1b_x[nonHH, :, 1, 3] = -tore[nj[nonHH]].unsqueeze(1) * w_x[nonHH, :, 7, 0]
    e1b_x[nonHH, :, 2, 3] = -tore[nj[nonHH]].unsqueeze(1) * w_x[nonHH, :, 8, 0]
    e1b_x[nonHH, :, 3, 3] = -tore[nj[nonHH]].unsqueeze(1) * w_x[nonHH, :, 9, 0]

    e2a_x[XX, :, 0, 1] = -tore[ni[XX]].unsqueeze(1) * w_x[XX, :, 0, 1]
    e2a_x[XX, :, 1, 1] = -tore[ni[XX]].unsqueeze(1) * w_x[XX, :, 0, 2]
    e2a_x[XX, :, 0, 2] = -tore[ni[XX]].unsqueeze(1) * w_x[XX, :, 0, 3]
    e2a_x[XX, :, 1, 2] = -tore[ni[XX]].unsqueeze(1) * w_x[XX, :, 0, 4]
    e2a_x[XX, :, 2, 2] = -tore[ni[XX]].unsqueeze(1) * w_x[XX, :, 0, 5]
    e2a_x[XX, :, 0, 3] = -tore[ni[XX]].unsqueeze(1) * w_x[XX, :, 0, 6]
    e2a_x[XX, :, 1, 3] = -tore[ni[XX]].unsqueeze(1) * w_x[XX, :, 0, 7]
    e2a_x[XX, :, 2, 3] = -tore[ni[XX]].unsqueeze(1) * w_x[XX, :, 0, 8]
    e2a_x[XX, :, 3, 3] = -tore[ni[XX]].unsqueeze(1) * w_x[XX, :, 0, 9]

    return e1b_x, e2a_x


from .constants import overlap_cutoff


def overlap_der_finiteDiff(overlap_KAB_x, idxi, idxj, rij, Xij, beta, ni, nj, zeta, qn_int):
    overlap_pairs = rij <= overlap_cutoff
    delta = 1e-5  # TODO: Make sure this is a good delta (small enough, but still doesnt cause numerical instabilities)
    di_plus = torch.zeros(Xij.shape[0], 4, 4, dtype=Xij.dtype, device=Xij.device)
    di_minus = torch.clone(di_plus)
    for coord in range(3):
        # since Xij = Xj-Xi, when I want to do Xi+delta, I have to subtract delta from from Xij
        Xij[:, coord] -= delta
        rij_ = torch.norm(Xij, dim=1)
        xij_ = Xij / rij_.unsqueeze(1)
        rij_ = rij_ / a0
        di_plus[overlap_pairs] = diatom_overlap_matrix(
            ni[overlap_pairs],
            nj[overlap_pairs],
            xij_[overlap_pairs],
            rij_[overlap_pairs],
            zeta[idxi][overlap_pairs],
            zeta[idxj][overlap_pairs],
            qn_int,
        )
        Xij[:, coord] += 2.0 * delta
        rij_ = torch.norm(Xij, dim=1)
        xij_ = Xij / rij_.unsqueeze(1)
        rij_ = rij_ / a0

        di_minus[overlap_pairs] = diatom_overlap_matrix(
            ni[overlap_pairs],
            nj[overlap_pairs],
            xij_[overlap_pairs],
            rij_[overlap_pairs],
            zeta[idxi][overlap_pairs],
            zeta[idxj][overlap_pairs],
            qn_int,
        )
        Xij[:, coord] -= delta
        overlap_KAB_x[:, coord, :, :] = (di_plus - di_minus) / (2.0 * delta)

    overlap_KAB_x[..., 0, 0] *= (beta[idxi, 0] + beta[idxj, 0]).unsqueeze(1)
    overlap_KAB_x[..., 0, 1:] *= (beta[idxi, 0:1] + beta[idxj, 1:2]).unsqueeze(1)
    overlap_KAB_x[..., 1:, 0] *= (beta[idxi, 1:2] + beta[idxj, 0:1]).unsqueeze(1)
    overlap_KAB_x[..., 1:, 1:] *= (beta[idxi, 1:2, None] + beta[idxj, 1:2, None]).unsqueeze(1)


def der_TETCILF(w_x_final, const, ni, nj, xij, Xij, r0, da0, db0, qa0, qb0, rho0a, rho0b, rho1a, rho1b, rho2a, rho2b,
                tore, riXH, ri):

    dtype = r0.dtype
    device = r0.device

    HH = (ni == 1) & (nj == 1)
    XH = (ni > 1) & (nj == 1)
    XX = (ni > 1) & (nj > 1)

    # Hydrogen - Hydrogen
    # aeeHH = (rho0a[HH]+rho0b[HH])**2
    # # Dividing by a0^2 for gradient in eV/ang
    term = -ev / a0 / a0 / r0.unsqueeze(1) * Xij
    ee = -r0 * pow((r0**2 + (rho0a + rho0b)**2), -1.5)
    ee_x = term * ee.unsqueeze(1)
    riHH_x = ee_x[HH, :]

    # Heavy atom - Hydrogen
    # aeeXH = (rho0a[XH]+rho0b[XH])**2
    rXH = r0[XH]
    daXH = da0[XH]
    qaXH = qa0[XH] * 2.0
    adeXH = (rho1a[XH] + rho0b[XH])**2
    aqeXH = (rho2a[XH] + rho0b[XH])**2
    dsqr6XH = 2.0 * rXH * pow(rXH**2 + aqeXH, -1.5)
    riXH_x = torch.zeros(XH.sum(), 3, 4, dtype=dtype, device=device)
    eeXH = ee[XH]
    riXH_x[..., 1 - 1] = ee_x[XH, :]
    riXH_x[...,2-1] = -0.5*term[XH]*((rXH+daXH)*pow((rXH+daXH)**2+adeXH,-1.5) \
                   - (rXH-daXH)*pow((rXH-daXH)**2+adeXH,-1.5)).unsqueeze(1)
    riXH_x[...,3-1] = term[XH]*(eeXH + 0.25*(-(rXH+qaXH)*pow((rXH+qaXH)**2+aqeXH,-1.5) \
                       - (rXH-qaXH)*pow((rXH-qaXH)**2+aqeXH,-1.5) \
                       + dsqr6XH)).unsqueeze(1)
    riXH_x[...,
           4 - 1] = term[XH] * (eeXH + 0.25 * (-2.0 * rXH * pow(rXH**2 + qaXH**2 + aqeXH, -1.5) + dsqr6XH)).unsqueeze(1)

    # Heavy atom - Heavy atom
    term = term[XX]
    r = r0[XX]
    da = da0[XX]
    db = db0[XX]
    qa = qa0[XX] * 2.0
    qb = qb0[XX] * 2.0
    qa1 = qa0[XX]
    qb1 = qb0[XX]
    # sqr(54)-sqr(72) use qa1 and qb1
    ri_x = torch.zeros(XX.sum(), 3, 22, dtype=dtype, device=device)

    # only the repeated terms are listed here
    ade = (rho1a[XX] + rho0b[XX])**2
    aqe = (rho2a[XX] + rho0b[XX])**2
    aed = (rho0a[XX] + rho1b[XX])**2
    aeq = (rho0a[XX] + rho2b[XX])**2
    axx = (rho1a[XX] + rho1b[XX])**2
    adq = (rho1a[XX] + rho2b[XX])**2
    aqd = (rho2a[XX] + rho1b[XX])**2
    aqq = (rho2a[XX] + rho2b[XX])**2
    ee = ee[XX]
    dze = ((r+da)*pow((r+da)**2+ade,-1.5) \
                   - (r-da)*pow((r-da)**2+ade,-1.5))
    dsqr6 = 2.0 * r * pow(r**2 + aqe, -1.5)
    qzze = -(r - qa) * pow((r - qa)**2 + aqe, -1.5) - (r + qa) * pow((r + qa)**2 + aqe, -1.5) + dsqr6
    qxxe = -2.0 * r * pow(r**2 + qa**2 + aqe, -1.5) + dsqr6
    edz = (r - db) * pow((r - db)**2 + aed, -1.5) - (r + db) * pow((r + db)**2 + aed, -1.5)
    dsqr12 = 2.0 * r * pow(r**2 + aeq, -1.5)
    eqzz = -(r - qb) * pow((r - qb)**2 + aeq, -1.5) - (r + qb) * pow((r + qb)**2 + aeq, -1.5) + dsqr12
    eqxx = -2.0 * r * pow(r**2 + qb**2 + aeq, -1.5) + dsqr12
    dsqr20 = 2.0 * (r + da) * pow((r + da)**2 + adq, -1.5)
    dsqr22 = 2.0 * (r - da) * pow((r - da)**2 + adq, -1.5)
    dsqr24 = 2.0 * (r - db) * pow((r - db)**2 + aqd, -1.5)
    dsqr26 = 2.0 * (r + db) * pow((r + db)**2 + aqd, -1.5)
    dsqr36 = 4.0 * (r) * pow(r**2 + aqq, -1.5)
    dsqr39 = 4.0 * (r) * pow(r**2 + qa**2 + aqq, -1.5)
    dsqr40 = 4.0 * (r) * pow(r**2 + qb**2 + aqq, -1.5)
    dsqr42 = 2.0 * (r - qb) * pow((r - qb)**2 + aqq, -1.5)
    dsqr44 = 2.0 * (r + qb) * pow((r + qb)**2 + aqq, -1.5)
    dsqr46 = 2.0 * (r + qa) * pow((r + qa)**2 + aqq, -1.5)
    dsqr48 = 2.0 * (r - qa) * pow((r - qa)**2 + aqq, -1.5)
    # all the index for ri is shfited by 1 to save space
    # C     (SS/SS)=1,   (SO/SS)=2,   (OO/SS)=3,   (PP/SS)=4,   (SS/OS)=5,
    # C     (SO/SO)=6,   (SP/SP)=7,   (OO/SO)=8,   (PP/SO)=9,   (PO/SP)=10,
    # C     (SS/OO)=11,  (SS/PP)=12,  (SO/OO)=13,  (SO/PP)=14,  (SP/OP)=15,
    # C     (OO/OO)=16,  (PP/OO)=17,  (OO/PP)=18,  (PP/PP)=19,  (PO/PO)=20,
    # C     (PP/P*P*)=21,   (P*P/P*P)=22
    ri_x[..., 1 - 1] = ee_x[XX, :]
    ri_x[..., 2 - 1] = -0.5 * term * dze.unsqueeze(1)
    ri_x[..., 3 - 1] = term * (ee + 0.25 * qzze).unsqueeze(1)
    ri_x[..., 4 - 1] = term * (ee + 0.25 * qxxe).unsqueeze(1)
    ri_x[..., 5 - 1] = -0.5 * term * edz.unsqueeze(1)
    # RI(6) = DZDZ = EV2/SQR(16) + EV2/SQR(17) - EV2/SQR(18) - EV2/SQR(19)
    ri_x[...,6-1] = 0.25*term*(-(r+da-db)*pow((r+da-db)**2 + axx,-1.5) - (r-da+db)*pow((r-da+db)**2 + axx,-1.5) \
                  + (r-da-db)*pow((r-da-db)**2 + axx,-1.5) + (r+da+db)*pow((r+da+db)**2 + axx,-1.5)).unsqueeze(1)
    # RI(7) = DXDX = EV1/SQR(14) - EV1/SQR(15)
    ri_x[..., 7 -
         1] = 0.5 * term * (-r * pow(r**2 + (da - db)**2 + axx, -1.5) + r * pow(r**2 +
                                                                                (da + db)**2 + axx, -1.5)).unsqueeze(1)
    # RI(8) = -EDZ -QZZDZ
    # QZZDZ = -EV3/SQR(32) + EV3/SQR(33) - EV3/SQR(34) + EV3/SQR(35)
    # + EV2/SQR(24) - EV2/SQR(26)
    ri_x[...,8-1] = -term*(0.5*edz + 0.125*((r+qa-db)*pow((r+qa-db)**2 + aqd,-1.5) - (r+qa+db)*pow((r+qa+db)**2 + aqd,-1.5) \
                + (r-qa-db)*pow((r-qa-db)**2 + aqd,-1.5) - (r-qa+db)*pow((r-qa+db)**2 + aqd,-1.5) \
                - dsqr24 + dsqr26)).unsqueeze(1)
    # RI(9) = -EDZ -QXXDZ
    # QXXDZ =  EV2/SQR(24) - EV2/SQR(25) - EV2/SQR(26) + EV2/SQR(27)
    ri_x[...,9-1] = -term*(0.5*edz + 0.125*(-dsqr24 + 2.0*(r-db)*pow((r-db)**2 + qa**2 + aqd,-1.5) \
                + dsqr26 - 2.0*(r+db)*pow((r+db)**2 + qa**2 + aqd,-1.5))).unsqueeze(1)

    # sqr(54)-sqr(72) use qa1 and qb1
    # RI(10) = -QXZDX
    # QXZDX = -EV2/SQR(58) + EV2/SQR(59) + EV2/SQR(60) - EV2/SQR(61)
    ri_x[...,10-1] = -0.25*term*((r+qa1)*pow((qa1-db)**2 + (r+qa1)**2 + aqd,-1.5) \
                   - (r-qa1)*pow((qa1-db)**2 + (r-qa1)**2 + aqd,-1.5) \
                   - (r+qa1)*pow((qa1+db)**2 + (r+qa1)**2 + aqd,-1.5) \
                   + (r-qa1)*pow((qa1+db)**2 + (r-qa1)**2 + aqd,-1.5)).unsqueeze(1)
    # RI(11) =  EE + EQZZ
    ri_x[..., 11 - 1] = term * (ee + 0.25 * eqzz).unsqueeze(1)
    # RI(12) =  EE + EQXX
    ri_x[..., 12 - 1] = term * (ee + 0.25 * eqxx).unsqueeze(1)
    # RI(13) = -DZE -DZQZZ
    # DZQZZ = -EV3/SQR(28) + EV3/SQR(29) - EV3/SQR(30) + EV3/SQR(31)
    #  - EV2/SQR(22) + EV2/SQR(20)
    ri_x[...,13-1] = -term*(0.5*dze + 0.125*(\
                 + (r+da-qb)*pow((r+da-qb)**2 + adq,-1.5) \
                 - (r-da-qb)*pow((r-da-qb)**2 + adq,-1.5) \
                 + (r+da+qb)*pow((r+da+qb)**2 + adq,-1.5) \
                 - (r-da+qb)*pow((r-da+qb)**2 + adq,-1.5) \
                 + dsqr22 - dsqr20)).unsqueeze(1)
    #
    # RI(14) = -DZE -DZQXX
    # DZQXX =  EV2/SQR(20) - EV2/SQR(21) - EV2/SQR(22) + EV2/SQR(23)
    ri_x[...,14-1] = -term*(0.5*dze + 0.125*(- dsqr20 + dsqr22 \
                 + 2.0*(r+da)*pow((r+da)**2 + qb**2 + adq,-1.5) \
                 - 2.0*(r-da)*pow((r-da)**2 + qb**2 + adq,-1.5))).unsqueeze(1)
    # RI(15) = -DXQXZ
    # DXQXZ = -EV2/SQR(54) + EV2/SQR(55) + EV2/SQR(56) - EV2/SQR(57)
    # sqr(54)-sqr(72) use qa1 and qb1
    ri_x[...,15-1] = -0.25*term*((r-qb1)*pow((da-qb1)**2 + (r-qb1)**2 + adq,-1.5) \
                   - (r+qb1)*pow((da-qb1)**2 + (r+qb1)**2 + adq,-1.5) \
                   - (r-qb1)*pow((da+qb1)**2 + (r-qb1)**2 + adq,-1.5) \
                   + (r+qb1)*pow((da+qb1)**2 + (r+qb1)**2 + adq,-1.5)).unsqueeze(1)
    # RI(16) = EE +EQZZ +QZZE +QZZQZZ
    # QZZQZZ = EV4/SQR(50) + EV4/SQR(51) + EV4/SQR(52) + EV4/SQR(53)
    # - EV3/SQR(48) - EV3/SQR(46) - EV3/SQR(42) - EV3/SQR(44)
    # + EV2/SQR(36)
    ri_x[...,16-1] = term*(ee + 0.25*eqzz + 0.25*qzze \
                 + 0.0625*(-(r+qa-qb)*pow((r+qa-qb)**2 + aqq,-1.5) \
                 - (r+qa+qb)*pow((r+qa+qb)**2 + aqq,-1.5) \
                 - (r-qa-qb)*pow((r-qa-qb)**2 + aqq,-1.5) \
                 - (r-qa+qb)*pow((r-qa+qb)**2 + aqq,-1.5) \
                 + dsqr48 + dsqr46 +dsqr42 + dsqr44 - dsqr36)).unsqueeze(1)
    # RI(17) = EE +EQZZ +QXXE +QXXQZZ
    # QXXQZZ = EV3/SQR(43) + EV3/SQR(45) - EV3/SQR(42) - EV3/SQR(44)
    #  - EV2/SQR(39) + EV2/SQR(36)
    ri_x[...,17-1] = term*(ee + 0.25*eqzz + 0.25*qxxe \
                 +0.0625*( -2.0*(r-qb)*pow((r-qb)**2 + qa**2 + aqq,-1.5) \
                 -2.0*(r+qb)*pow((r+qb)**2 + qa**2 + aqq,-1.5) \
                 + dsqr42 + dsqr44 + dsqr39 - dsqr36)).unsqueeze(1)
    # RI(18) = EE +EQXX +QZZE +QZZQXX
    # QZZQXX = EV3/SQR(47) + EV3/SQR(49) - EV3/SQR(46) - EV3/SQR(48)
    #  - EV2/SQR(40) + EV2/SQR(36)
    ri_x[...,18-1] = term*(ee + 0.25*eqxx + 0.25*qzze \
                 + 0.0625*(-2.0*(r+qa)*pow((r+qa)**2 + qb**2 + aqq,-1.5) \
                 -2.0*(r-qa)*pow((r-qa)**2 + qb**2 + aqq,-1.5) \
                 + dsqr46 + dsqr48 + dsqr40 - dsqr36)).unsqueeze(1)
    # RI(19) = EE +EQXX +QXXE +QXXQXX
    # QXXQXX = EV3/SQR(37) + EV3/SQR(38) - EV2/SQR(39) - EV2/SQR(40)
    # + EV2/SQR(36)
    qxxqxx = -2.0*r*pow(r**2 + (qa-qb)**2 + aqq,-1.5) \
            -2.0*r*pow(r**2 + (qa+qb)**2 + aqq,-1.5) \
           + dsqr39 + dsqr40 - dsqr36
    ri_x[..., 19 - 1] = term * (ee + 0.25 * eqxx + 0.25 * qxxe + 0.0625 * qxxqxx).unsqueeze(1)
    # RI(20) = QXZQXZ
    # QXZQXZ = EV3/SQR(65) - EV3/SQR(67) - EV3/SQR(69) + EV3/SQR(71)
    # - EV3/SQR(66) + EV3/SQR(68) + EV3/SQR(70) - EV3/SQR(72)
    # sqr(54)-sqr(72) use qa1 and qb1
    ri_x[...,20-1] = 0.125*term*(-(r+qa1-qb1)*pow((r+qa1-qb1)**2 + (qa1-qb1)**2 + aqq,-1.5) \
                   + (r+qa1+qb1)*pow((r+qa1+qb1)**2 + (qa1-qb1)**2 + aqq,-1.5) \
                   + (r-qa1-qb1)*pow((r-qa1-qb1)**2 + (qa1-qb1)**2 + aqq,-1.5) \
                   - (r-qa1+qb1)*pow((r-qa1+qb1)**2 + (qa1-qb1)**2 + aqq,-1.5) \
                   + (r+qa1-qb1)*pow((r+qa1-qb1)**2 + (qa1+qb1)**2 + aqq,-1.5) \
                   - (r+qa1+qb1)*pow((r+qa1+qb1)**2 + (qa1+qb1)**2 + aqq,-1.5) \
                   - (r-qa1-qb1)*pow((r-qa1-qb1)**2 + (qa1+qb1)**2 + aqq,-1.5) \
                   + (r-qa1+qb1)*pow((r-qa1+qb1)**2 + (qa1+qb1)**2 + aqq,-1.5)).unsqueeze(1)
    # RI(21) = EE +EQXX +QXXE +QXXQYY
    # QXXQYY = EV2/SQR(41) - EV2/SQR(39) - EV2/SQR(40) + EV2/SQR(36)
    qxxqyy = -4.0*r*pow(r**2 + qa**2 + qb**2 + aqq,-1.5) \
           + dsqr39 + dsqr40 - dsqr36
    ri_x[..., 21 - 1] = term * (ee + 0.25 * eqxx + 0.25 * qxxe + 0.0625 * qxxqyy).unsqueeze(1)
    # RI(22) = PP * (QXXQXX -QXXQYY)
    ri_x[..., 22 - 1] = 0.03125 * term * (qxxqxx - qxxqyy).unsqueeze(1)

    # We have the derivatives of the 2-center-2-elec integrals in the local frame
    # In the local frame for p-orbitals we have p-sigma (along the axis) ,p-pi,p-pi' (perpendicular to the axis)
    # But in the molecular frame we have px,py,pz which are rotations of p-sigma, p-pi, p-pi'
    # The p orbitals rotate just like the coordinate frame, so the rotation matrix is easy to express
    # We now make the rotation matrix and its derivative for the p-orbitals
    rot = torch.zeros(r0.shape[0], 3, 3, device=device, dtype=dtype)
    rot_der = torch.zeros(r0.shape[0], 3, 3, 3, device=device, dtype=dtype)

    rxy2 = torch.square(Xij[:, 0]) + torch.square(Xij[:, 1])
    ryz2 = torch.square(Xij[:, 1]) + torch.square(Xij[:, 2])
    rxz2 = torch.square(Xij[:, 0]) + torch.square(Xij[:, 2])
    axis_tolerance = 1e-8
    onerij = 1.0 / a0 / r0

    Xalign = ryz2 < axis_tolerance
    Yalign = rxz2 < axis_tolerance
    Zalign = rxy2 < axis_tolerance
    Noalign = ~(Xalign | Yalign | Zalign)

    xij_ = -xij[Noalign, ...]
    rot[Noalign, 0, :] = xij_
    onerxy = 1.0 / torch.sqrt(rxy2[Noalign])
    rxy_over_rab = (torch.sqrt(rxy2) / r0)[Noalign] / a0
    rab_over_rxy = a0 * r0[Noalign] * onerxy
    rab_over_rxy_sq = torch.square(rab_over_rxy)

    # The (1,0) element of the rotation matrix is -Y/sqrt(X^2+Y^2)*sign(X). If X (=xi-xj) is zero then there is a discontinuity in the sign function
    # and hence the derivative will not exist. So I'm printing a warning that there might be numerical errors here
    # Similaryly the (1,1) element of the rotation matrix is abs(X/sqrt(X^2+Y^2)). Again, the derivative of abs(X) will not exist when X=0, and hence this
    # will lead to errors.
    if (xij_[:, 0].any() == 0):
        print(
            "WARNING: The x component of the pair distance is zero. This could lead to instabilities in the derivative of the rotation matrix"
        )

    # As a quick-fix, I will add a small number (eps) when calculating sign(X) to avoid the aforementioned instability
    signcorrect = torch.sign(xij_[:, 0] + torch.finfo(dtype).eps)
    rot[Noalign, 1, 0] = -xij_[:, 1] * rab_over_rxy * signcorrect
    rot[Noalign, 1, 1] = torch.abs(xij_[:, 0] * rab_over_rxy)

    rot[Noalign, 2, 0] = -xij_[:, 0] * xij_[:, 2] * rab_over_rxy
    rot[Noalign, 2, 1] = -xij_[:, 1] * xij_[:, 2] * rab_over_rxy
    rot[Noalign, 2, 2] = rxy_over_rab

    # Derivative of the rotation matrix
    termX = xij_[:, 0] * onerij[Noalign]
    termY = xij_[:, 1] * onerij[Noalign]
    termZ = xij_[:, 2] * onerij[Noalign]
    # term = Xij[Noalign,:]*onerij.unsqueeze(1)
    rot_der[Noalign, 0, 0, 0] = onerij[Noalign] - xij_[:, 0] * termX
    rot_der[Noalign, 0, 0, 1] = -xij_[:, 0] * termY
    rot_der[Noalign, 0, 0, 2] = -xij_[:, 0] * termZ

    rot_der[Noalign, 1, 0, 0] = -xij_[:, 1] * termX
    rot_der[Noalign, 1, 0, 1] = onerij[Noalign] - xij_[:, 1] * termY
    rot_der[Noalign, 1, 0, 2] = -xij_[:, 1] * termZ

    rot_der[Noalign, 2, 0, 0] = -xij_[:, 2] * termX
    rot_der[Noalign, 2, 0, 1] = -xij_[:, 2] * termY
    rot_der[Noalign, 2, 0, 2] = onerij[Noalign] - xij_[:, 2] * termZ

    rot_der[Noalign, 0, 2, 2] = xij_[:, 0] * onerxy - rot[Noalign, 2, 2] * termX
    rot_der[Noalign, 1, 2, 2] = xij_[:, 1] * onerxy - rot[Noalign, 2, 2] * termY
    rot_der[Noalign, 2, 2, 2] = -rot[Noalign, 2, 2] * termZ

    rot_der[Noalign, 0, 1, 0] = -rot[Noalign, 1, 1] * rot[Noalign, 1, 0] * onerxy
    rot_der[Noalign, 1, 1, 0] = -torch.square(rot[Noalign, 1, 1]) * onerxy
    # # Sanity check because openmopac (and hence NEXMD) do this differently. I want to make sure our expressions give the same result
    # tolerance = 1e-8
    # assert torch.allclose(rot_der[Noalign,0,1,0],-rot_der[Noalign,1,0,0]*rab_over_rxy+rot[Noalign,0,1]*rot_der[Noalign,0,2,2]*rab_over_rxy_sq,atol=tolerance)
    # assert torch.allclose(rot_der[Noalign,1,1,0],-rot_der[Noalign,1,0,1]*rab_over_rxy+rot[Noalign,0,1]*rot_der[Noalign,1,2,2]*rab_over_rxy_sq,atol=tolerance)
    # assert torch.all(torch.abs(-rot_der[Noalign,1,0,2]*rab_over_rxy+rot[Noalign,0,1]*rot_der[Noalign,2,2,2]*rab_over_rxy_sq)<tolerance)

    rot_der[Noalign, 0:2, 1, 0] *= signcorrect.unsqueeze(1)

    rot_der[Noalign, 0, 1, 1] = torch.square(rot[Noalign, 1, 0]) * onerxy
    rot_der[Noalign, 1, 1, 1] = rot[Noalign, 1, 1] * rot[Noalign, 1, 0] * onerxy
    # # Sanity check because openmopac (and hence NEXMD) do this differently. I want to make sure our expressions give the same result
    # tolerance = 1e-8
    # mopacs = rot_der[Noalign,0,0,0]*rab_over_rxy-rot[Noalign,0,0]*rot_der[Noalign,0,2,2]*rab_over_rxy_sq
    # mine = rot_der[Noalign,0,1,1]
    # assert torch.allclose(mine,mopacs,atol=tolerance)
    # assert torch.allclose(rot_der[Noalign,1,1,1],rot_der[Noalign,0,0,1]*rab_over_rxy-rot[Noalign,0,0]*rot_der[Noalign,1,2,2]*rab_over_rxy_sq,atol=tolerance)
    # assert torch.all(torch.abs(rot_der[Noalign,0,0,2]*rab_over_rxy-rot[Noalign,0,0]*rot_der[Noalign,2,2,2]*rab_over_rxy_sq)<tolerance)

    rot_der[Noalign, 0:2, 1, 1] *= signcorrect.unsqueeze(1)

    rot_der[Noalign, 0, 2, 0] = -xij_[:, 2] * rot_der[Noalign, 0, 0, 0] * rab_over_rxy - xij_[:, 0] * rot_der[
        Noalign, 2, 0, 0] * rab_over_rxy + xij_[:, 0] * xij_[:, 2] * rot_der[Noalign, 0, 2, 2] * rab_over_rxy_sq
    rot_der[Noalign, 1, 2, 0] = torch.prod(xij_, dim=1) * (onerxy + rab_over_rxy_sq * onerxy)
    rot_der[Noalign, 2, 2, 0] = -termX * rxy_over_rab

    rot_der[Noalign, 0, 2, 1] = rot_der[Noalign, 1, 2, 0]
    rot_der[Noalign, 1, 2, 1] = -xij_[:, 2] * rot_der[Noalign, 1, 0, 1] * rab_over_rxy - xij_[:, 1] * rot_der[
        Noalign, 2, 0, 1] * rab_over_rxy + xij_[:, 1] * xij_[:, 2] * rot_der[Noalign, 1, 2, 2] * rab_over_rxy_sq
    rot_der[Noalign, 2, 2, 1] = -termY * rxy_over_rab

    rot[Zalign, 0, 2] = torch.sign(-xij[Zalign, 2])
    rot[Zalign, 1, 1] = 1.0
    rot[Zalign, 2, 0] = rot[Zalign, 0, 2]
    rot_der[Zalign, 0, 0, 0] = onerij[Zalign]
    rot_der[Zalign, 0, 2, 2] = -onerij[Zalign]
    rot_der[Zalign, 1, 0, 1] = onerij[Zalign]
    rot_der[Zalign, 1, 1, 2] = -rot[Zalign, 0, 2] * onerij[Zalign]

    rot[Xalign, 0, 0] = torch.sign(-xij[Xalign, 0])
    rot[Xalign, 1, 1] = rot[Xalign, 0, 0]
    rot[Xalign, 2, 2] = 1.0
    rot_der[Xalign, 1, 0, 1] = onerij[Xalign]
    rot_der[Xalign, 1, 1, 0] = -onerij[Xalign]
    rot_der[Xalign, 2, 0, 2] = onerij[Xalign]
    rot_der[Xalign, 2, 2, 0] = -rot[Xalign, 0, 0] * onerij[Xalign]

    rot[Yalign, 0, 1] = torch.sign(-xij[Yalign, 1])
    rot[Yalign, 1, 0] = -rot[Yalign, 0, 1]
    rot[Yalign, 2, 2] = 1.0
    rot_der[Yalign, 0, 0, 0] = onerij[Yalign]
    rot_der[Yalign, 0, 1, 1] = onerij[Yalign]
    rot_der[Yalign, 2, 0, 2] = onerij[Yalign]
    rot_der[Yalign, 2, 2, 1] = -rot[Yalign, 0, 1] * onerij[Yalign]

    rotXH = rot[XH, ...]
    rot = rot[XX, ...]
    rot_derXH = rot_der[XH, ...]
    rot_der = rot_der[XX, ...]

    w_x = torch.zeros(ri.shape[0], 3, 100, device=device, dtype=dtype)
    wXH_x = torch.zeros(XH.sum(), 3, 10, device=device, dtype=dtype)

    idx = -1
    idxXH = 0
    for kk in range(0, 4):
        k = kk - 1
        for ll in range(0, kk + 1):
            l = ll - 1
            for mm in range(0, 4):
                m = mm - 1
                for nn in range(0, mm + 1):
                    n = nn - 1
                    idx = idx + 1
                    if kk == 0:
                        if mm == 0:
                            # (ss|ss)
                            w_x[..., idx] = ri_x[..., 0]
                            wXH_x[..., idxXH] = riXH_x[..., 0]
                            idxXH = idxXH + 1
                        elif nn == 0:
                            # (ss|ps)
                            w_x[..., idx] = ri_x[..., 4] * rot[:, None, 0, m] + ri[:, None, 4] * rot_der[:, :, 0, m]
                        else:
                            # (ss|pp)
                            w_x[...,idx] = ri_x[...,10]*(rot[:,0,m]*rot[:,0,n]).unsqueeze(1) +\
                                    ri[:,None,10]*(rot_der[:,:,0,m]*rot[:,None,0,n]+rot[:,None,0,m]*rot_der[:,:,0,n]) + \
                                           ri_x[...,11]*(rot[:,1,m]*rot[:,1,n]+rot[:,2,m]*rot[:,2,n]).unsqueeze(1) +\
                                           ri[:,None,11]*(rot_der[:,:,1,m]*rot[:,None,1,n]+rot_der[:,:,2,m]*rot[:,None,2,n]+
                                                          rot[:,None,1,m]*rot_der[:,:,1,n]+rot[:,None,2,m]*rot_der[:,:,2,n])

                    elif ll == 0:
                        if mm == 0:
                            # (ps|ss)
                            w_x[..., idx] = ri_x[..., 1] * rot[:, None, 0, k] + ri[:, None, 1] * rot_der[:, :, 0, k]
                            wXH_x[..., idxXH] = riXH_x[..., 1] * rotXH[:, None, 0, k] + riXH[:, None,
                                                                                             1] * rot_derXH[:, :, 0, k]
                            idxXH = idxXH + 1
                        elif nn == 0:
                            # (ps|ps)
                            w_x[...,idx] = ri_x[...,5]*(rot[:,0,k]*rot[:,0,m]).unsqueeze(1) +\
                                    ri[:,None,5]*(rot_der[:,:,0,k]*rot[:,None,0,m]+rot[:,None,0,k]*rot_der[:,:,0,m]) + \
                                           ri_x[...,6]*(rot[:,1,k]*rot[:,1,m]+rot[:,2,k]*rot[:,2,m]).unsqueeze(1) +\
                                           ri[:,None,6]*(rot_der[:,:,1,k]*rot[:,None,1,m]+rot_der[:,:,2,k]*rot[:,None,2,m]+
                                                          rot[:,None,1,k]*rot_der[:,:,1,m]+rot[:,None,2,k]*rot_der[:,:,2,m])
                        else:
                            #(ps|pp)
                            w_x[...,idx] = ri_x[...,12]*(rot[:,0,k]*rot[:,0,n]*rot[:,0,m]).unsqueeze(1) +\
                                    ri[:,None,12]*(rot_der[:,:,0,k]*(rot[:,0,n]*rot[:,0,m]).unsqueeze(1) +
                                     rot_der[:,:,0,n]*(rot[:,0,k]*rot[:,0,m]).unsqueeze(1)+rot_der[:,:,0,m]*(rot[:,0,n]*rot[:,0,k]).unsqueeze(1)) +\
                                    ri_x[...,13]*((rot[:,1,m]*rot[:,1,n]+rot[:,2,m]*rot[:,2,n])*rot[:,0,k]).unsqueeze(1) +\
                                    ri[:,None,13]*((rot[:,1,m]*rot[:,1,n]+rot[:,2,m]*rot[:,2,n]).unsqueeze(1)*rot_der[:,:,0,k]+
                                                  (rot_der[:,:,1,m]*rot[:,None,1,n]+rot[:,None,1,m]*rot_der[:,:,1,n]+
                                                   rot_der[:,:,2,m]*rot[:,None,2,n]+rot[:,None,2,m]*rot_der[:,:,2,n])*rot[:,None,0,k]) +\
                                    ri_x[...,14]*(rot[:,1,k]*(rot[:,1,n]*rot[:,0,m]+rot[:,1,m]*rot[:,0,n])+
                                                 rot[:,2,k]*(rot[:,2,m]*rot[:,0,n]+rot[:,2,n]*rot[:,0,m])).unsqueeze(1) +\
                                    ri[:,None,14]*(rot_der[:,:,1,k]*(rot[:,1,n]*rot[:,0,m]+rot[:,1,m]*rot[:,0,n]).unsqueeze(1)+
                                                  rot[:,None,1,k]*(rot_der[:,:,1,m]*rot[:,None,0,n]+rot[:,None,1,m]*rot_der[:,:,0,n]+
                                                                    rot_der[:,:,1,n]*rot[:,None,0,m]+rot[:,None,1,n]*rot_der[:,:,0,m])+
                                                  rot_der[:,:,2,k]*(rot[:,2,n]*rot[:,0,m]+rot[:,2,m]*rot[:,0,n]).unsqueeze(1)+
                                                  rot[:,None,2,k]*(rot_der[:,:,2,n]*rot[:,None,0,m]+rot[:,None,2,n]*rot_der[:,:,0,m]+
                                                                    rot_der[:,:,2,m]*rot[:,None,0,n]+rot[:,None,2,m]*rot_der[:,:,0,n]))
                            pass
                    else:
                        if mm == 0:
                            # (pp|ss)
                            w_x[...,idx] = ri_x[...,2]*(rot[:,0,k]*rot[:,0,l]).unsqueeze(1) +\
                                    ri[:,None,2]*(rot_der[:,:,0,k]*rot[:,None,0,l]+rot[:,None,0,k]*rot_der[:,:,0,l]) + \
                                           ri_x[...,3]*(rot[:,1,k]*rot[:,1,l]+rot[:,2,k]*rot[:,2,l]).unsqueeze(1) +\
                                           ri[:,None,3]*(rot_der[:,:,1,k]*rot[:,None,1,l]+rot_der[:,:,2,k]*rot[:,None,2,l]+
                                                          rot[:,None,1,k]*rot_der[:,:,1,l]+rot[:,None,2,k]*rot_der[:,:,2,l])
                            wXH_x[...,idxXH] = riXH_x[...,2]*(rotXH[:,0,k]*rotXH[:,0,l]).unsqueeze(1) +\
                                    riXH[:,None,2]*(rot_derXH[:,:,0,k]*rotXH[:,None,0,l]+rotXH[:,None,0,k]*rot_derXH[:,:,0,l]) + \
                                           riXH_x[...,3]*(rotXH[:,1,k]*rotXH[:,1,l]+rotXH[:,2,k]*rotXH[:,2,l]).unsqueeze(1) +\
                                           riXH[:,None,3]*(rot_derXH[:,:,1,k]*rotXH[:,None,1,l]+rot_derXH[:,:,2,k]*rotXH[:,None,2,l]+
                                                          rotXH[:,None,1,k]*rot_derXH[:,:,1,l]+rotXH[:,None,2,k]*rot_derXH[:,:,2,l])
                            idxXH = idxXH + 1
                        elif nn == 0:
                            # (pp|ps)
                            w_x[...,idx] = ri_x[...,7]*(rot[:,0,k]*rot[:,0,l]*rot[:,0,m]).unsqueeze(1) +\
                                    ri[:,None,7]*(rot_der[:,:,0,k]*(rot[:,0,l]*rot[:,0,m]).unsqueeze(1) +
                                     rot_der[:,:,0,l]*(rot[:,0,k]*rot[:,0,m]).unsqueeze(1)+rot_der[:,:,0,m]*(rot[:,0,l]*rot[:,0,k]).unsqueeze(1)) +\
                                    ri_x[...,8]*((rot[:,1,k]*rot[:,1,l]+rot[:,2,k]*rot[:,2,l])*rot[:,0,m]).unsqueeze(1) +\
                                    ri[:,None,8]*((rot[:,1,k]*rot[:,1,l]+rot[:,2,k]*rot[:,2,l]).unsqueeze(1)*rot_der[:,:,0,m]+
                                                  (rot_der[:,:,1,k]*rot[:,None,1,l]+rot[:,None,1,k]*rot_der[:,:,1,l]+
                                                   rot_der[:,:,2,k]*rot[:,None,2,l]+rot[:,None,2,k]*rot_der[:,:,2,l])*rot[:,None,0,m]) +\
                                    ri_x[...,9]*(rot[:,0,k]*(rot[:,1,l]*rot[:,1,m]+rot[:,2,l]*rot[:,2,m])+
                                                 rot[:,0,l]*(rot[:,1,k]*rot[:,1,m]+rot[:,2,k]*rot[:,2,m])).unsqueeze(1) +\
                                    ri[:,None,9]*(rot_der[:,:,0,k]*(rot[:,1,l]*rot[:,1,m]+rot[:,2,l]*rot[:,2,m]).unsqueeze(1)+
                                                  rot[:,None,0,k]*(rot_der[:,:,1,l]*rot[:,None,1,m]+rot[:,None,2,l]*rot_der[:,:,2,m]+
                                                                    rot_der[:,:,1,m]*rot[:,None,1,l]+rot[:,None,2,m]*rot_der[:,:,2,l])+
                                                  rot_der[:,:,0,l]*(rot[:,1,k]*rot[:,1,m]+rot[:,2,k]*rot[:,2,m]).unsqueeze(1)+
                                                  rot[:,None,0,l]*(rot_der[:,:,1,k]*rot[:,None,1,m]+rot[:,None,2,k]*rot_der[:,:,2,m]+
                                                                    rot_der[:,:,1,m]*rot[:,None,1,k]+rot[:,None,2,m]*rot_der[:,:,2,k]))

                        else:
                            #(pp|pp)
                            w_x[...,idx] = ri_x[...,16-1] * (rot[:,0,k] * rot[:,0,l] * rot[:,0,m] * rot[:,0,n]).unsqueeze(1) +  \
             ri[:,None,16-1] * (rot_der[:,:,0,k]*(rot[:,0,l]*rot[:,0,m]*rot[:,0,n]).unsqueeze(1)+\
             rot_der[:,:,0,l]*(rot[:,0,k]*rot[:,0,m]*rot[:,0,n]).unsqueeze(1)+rot_der[:,:,0,m]*(rot[:,0,k]*rot[:,0,l]*rot[:,0,n]).unsqueeze(1)+\
             (rot[:,0,k]*rot[:,0,l]*rot[:,0,m]).unsqueeze(1)*rot_der[:,:,0,n]) + ri_x[...,17-1] * ((rot[:,1,k]*rot[:,1,l]+\
             rot[:,2,k]*rot[:,2,l]) * rot[:,0,m] * rot[:,0,n]).unsqueeze(1) + ri[:,None,17-1] * ((rot_der[:,:,1,k]*rot[:,None,1,l]+\
             rot[:,None,1,k]*rot_der[:,:,1,l]+rot_der[:,:,2,k]*rot[:,None,2,l]+rot[:,None,2,k]*rot_der[:,:,2,l])*(rot[:,0,m]*rot[:,0,n]).unsqueeze(1)+\
             (rot[:,1,k]*rot[:,1,l]+rot[:,2,k]*rot[:,2,l]).unsqueeze(1)*(rot_der[:,:,0,m]*rot[:,None,0,n]+rot[:,None,0,m]*rot_der[:,:,0,n])) +\
              ri_x[...,18-1] * (rot[:,0,k] * rot[:,0,l] * (rot[:,1,m]*rot[:,1,n]+rot[:,2,m]*rot[:,2,n])).unsqueeze(1) + \
             ri[:,None,18-1] * ((rot_der[:,:,0,k]*rot[:,None,0,l]+rot[:,None,0,k]*rot_der[:,:,0,l])*(rot[:,1,m]*rot[:,1,n]+\
             rot[:,2,m]*rot[:,2,n]).unsqueeze(1)+(rot[:,0,k]*rot[:,0,l]).unsqueeze(1)*(rot_der[:,:,1,m]*rot[:,None,1,n]+rot[:,None,1,m]*rot_der[:,:,1,n]+\
             rot_der[:,:,2,m]*rot[:,None,2,n]+rot[:,None,2,m]*rot_der[:,:,2,n]))
                            w_x[...,idx] += ri_x[...,19-1] * (rot[:,1,k]*rot[:,1,l]*rot[:,1,m]*rot[:,1,n]+\
             rot[:,2,k]*rot[:,2,l]*rot[:,2,m]*rot[:,2,n]).unsqueeze(1) + ri[:,None,19-1] * \
             (rot_der[:,:,1,k]*(rot[:,1,l]*rot[:,1,m]*rot[:,1,n]).unsqueeze(1)+rot_der[:,:,1,l]*(rot[:,1,k]*rot[:,1,m]*rot[:,1,n]).unsqueeze(1)+\
             rot_der[:,:,1,m]*(rot[:,1,k]*rot[:,1,l]*rot[:,1,n]).unsqueeze(1)+(rot[:,1,k]*rot[:,1,l]*rot[:,1,m]).unsqueeze(1)*rot_der[:,:,1,n]+\
             rot_der[:,:,2,k]*(rot[:,2,l]*rot[:,2,m]*rot[:,2,n]).unsqueeze(1)+rot_der[:,:,2,l]*(rot[:,2,k]*rot[:,2,m]*rot[:,2,n]).unsqueeze(1)+\
             rot_der[:,:,2,m]*(rot[:,2,k]*rot[:,2,l]*rot[:,2,n]).unsqueeze(1)+(rot[:,2,k]*rot[:,2,l]*rot[:,2,m]).unsqueeze(1)*rot_der[:,:,2,n]) + \
             ri_x[...,20-1] * (rot[:,0,k]*(rot[:,0,m]*(rot[:,1,l]*rot[:,1,n]+rot[:,2,l]*rot[:,2,n])+\
             rot[:,0,n]*(rot[:,1,l]*rot[:,1,m]+rot[:,2,l]*rot[:,2,m]))+\
             rot[:,0,l]*(rot[:,0,m]*(rot[:,1,k]*rot[:,1,n]+rot[:,2,k]*rot[:,2,n])+\
             rot[:,0,n]*(rot[:,1,k]*rot[:,1,m]+rot[:,2,k]*rot[:,2,m]))).unsqueeze(1)
                            #      TO AVOID COMPILER DIFFICULTIES THIS IS DIVIDED
                            temp1 = rot_der[:,:,0,k] * (rot[:,0,m]*(rot[:,1,l]*rot[:,1,n]+rot[:,2,l]*rot[:,2,n])+\
             rot[:,0,n]*(rot[:,1,l]*rot[:,1,m]+rot[:,2,l]*rot[:,2,m])).unsqueeze(1) + rot_der[:,:,0,l] * \
             (rot[:,0,m]*(rot[:,1,k]*rot[:,1,n]+rot[:,2,k]*rot[:,2,n])+rot[:,0,n]*(rot[:,1,k]*rot[:,1,m]+\
             rot[:,2,k]*rot[:,2,m])).unsqueeze(1) + rot[:,None,0,k] * (rot_der[:,:,0,m]*(rot[:,1,l]*rot[:,1,n]+\
             rot[:,2,l]*rot[:,2,n]).unsqueeze(1)+rot_der[:,:,0,n]*(rot[:,1,l]*rot[:,1,m]+rot[:,2,l]*rot[:,2,m]).unsqueeze(1)) + rot[:,None,0,l] \
             * (rot_der[:,:,0,m]*(rot[:,1,k]*rot[:,1,n]+rot[:,2,k]*rot[:,2,n]).unsqueeze(1)+rot_der[:,:,0,n]*(rot[:,1,k]*rot[:,1,m]+\
             rot[:,2,k]*rot[:,2,m]).unsqueeze(1))
                            temp2 = rot[:,None,0,k] * (rot[:,None,0,m]*(rot_der[:,:,1,l]*rot[:,None,1,n]+rot[:,None,1,l]*rot_der[:,:,1,n]+\
             rot_der[:,:,2,l]*rot[:,None,2,n]+rot[:,None,2,l]*rot_der[:,:,2,n])+rot[:,None,0,n]*(rot_der[:,:,1,l]*rot[:,None,1,m]+\
             rot[:,None,1,l]*rot_der[:,:,1,m]+rot_der[:,:,2,l]*rot[:,None,2,m]+rot[:,None,2,l]*rot_der[:,:,2,m])) + rot[:,None,0,l] * \
             (rot[:,None,0,m]*(rot_der[:,:,1,k]*rot[:,None,1,n]+rot[:,None,1,k]*rot_der[:,:,1,n]+rot_der[:,:,2,k]*rot[:,None,2,n]+\
             rot[:,None,2,k]*rot_der[:,:,2,n])+rot[:,None,0,n]*(rot_der[:,:,1,k]*rot[:,None,1,m]+rot[:,None,1,k]*rot_der[:,:,1,m]+\
             rot_der[:,:,2,k]*rot[:,None,2,m]+rot[:,None,2,k]*rot_der[:,:,2,m]))
                            w_x[..., idx] += ri[:, None, 20 - 1] * (temp1 + temp2)
                            w_x[...,idx] += ri_x[...,21-1] * (rot[:,1,k]*rot[:,1,l]*rot[:,2,m]*rot[:,2,n]+\
             rot[:,2,k]*rot[:,2,l]*rot[:,1,m]*rot[:,1,n]).unsqueeze(1) + ri[:,None,21-1] * \
             (rot_der[:,:,1,k]*(rot[:,1,l]*rot[:,2,m]*rot[:,2,n]).unsqueeze(1)+rot_der[:,:,1,l]*(rot[:,1,k]*rot[:,2,m]*rot[:,2,n]).unsqueeze(1)+\
             rot_der[:,:,2,m]*(rot[:,1,k]*rot[:,1,l]*rot[:,2,n]).unsqueeze(1)+(rot[:,1,k]*rot[:,1,l]*rot[:,2,m]).unsqueeze(1)*rot_der[:,:,2,n]+\
             rot_der[:,:,2,k]*(rot[:,2,l]*rot[:,1,m]*rot[:,1,n]).unsqueeze(1)+rot_der[:,:,2,l]*(rot[:,2,k]*rot[:,1,m]*rot[:,1,n]).unsqueeze(1)+\
             rot_der[:,:,1,m]*(rot[:,2,k]*rot[:,2,l]*rot[:,1,n]).unsqueeze(1)+(rot[:,2,k]*rot[:,2,l]*rot[:,1,m]).unsqueeze(1)*rot_der[:,:,1,n])
                            w_x[...,idx] += ri_x[...,22-1] * ((rot[:,1,k]*rot[:,2,l]+rot[:,2,k]*rot[:,1,l]) * \
             (rot[:,1,m]*rot[:,2,n]+rot[:,2,m]*rot[:,1,n])).unsqueeze(1) + ri[:,None,22-1] * ((rot_der[:,:,1,k]*rot[:,None,2,l]+\
             rot[:,None,1,k]*rot_der[:,:,2,l]+rot_der[:,:,2,k]*rot[:,None,1,l]+rot[:,None,2,k]*rot_der[:,:,1,l])*(rot[:,1,m]*rot[:,2,n]+\
             rot[:,2,m]*rot[:,1,n]).unsqueeze(1)+ (rot[:,1,k]*rot[:,2,l]+rot[:,2,k]*rot[:,1,l]).unsqueeze(1)*(rot_der[:,:,1,m]*rot[:,None,2,n]+\
             rot[:,None,1,m]*rot_der[:,:,2,n]+rot_der[:,:,2,m]*rot[:,None,1,n]+rot[:,None,2,m]*rot_der[:,:,1,n]))

    w_x_final[HH, :, 0, 0] = riHH_x
    w_x_final[XH, :, :, 0] = wXH_x
    w_x_final[XX, ...] = w_x.reshape(ri.shape[0], 3, 10, 10)

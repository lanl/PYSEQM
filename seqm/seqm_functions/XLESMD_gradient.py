import torch

from .anal_grad import core_core_der, overlap_der_finiteDiff, w_der
from .cg_solver import conjugate_gradient_batch
from .constants import a0
from .dispersion_am1_fs1 import dEdisp_dr
from .rcis_batch import get_occ_virt, make_A_times_zvector_batched, makeA_pi_batched, unpackone_batch
from .rcis_grad_batch import make_cis_state_dipole


def xlesmd_rcis_grad_batch(
    amp_Xbar,
    amp_X,
    mol,
    w,
    e_mo,
    riXH,
    ri,
    P0,
    zvec_tolerance,
    gam,
    method,
    parnuc,
    rpa=False,
    include_ground_state=False,
    orbital_window=None,
    calculate_dipole=False,
):
    """
    amp: tensor of CIS amplitudes of shape [b,nov]. For each of the b molecules, the CIS amplitues of the
         state for which the gradient is required has to be selected and put together into the amp tensor
    """
    molsize = mol.molsize
    nHeavy = mol.nHeavy[0]
    nHydro = mol.nHydro[0]
    nocc, nvirt, Cocc, Cvirt = get_occ_virt(mol, orbital_window=orbital_window)
    norb = mol.norb[0]
    nmol = mol.nmol
    if rpa:
        raise NotImplementedError("XLESMD RPA gradients not implemented yet")

    def difference_density(amp_ia_X, Cocc, Cvirt, amp_ia_Y=None, rpa=False):
        B_virt = torch.einsum("Nma,Nia->Nmi", Cvirt, amp_ia_X)
        B_occ = torch.einsum("Nmi,Nia->Nma", Cocc, amp_ia_X)

        B = torch.einsum("Nmi,Nni->Nmn", B_virt, B_virt) - torch.einsum("Nmi,Nni->Nmn", B_occ, B_occ)

        # if rpa:
        #     B_virt_Y = torch.einsum("Nma,Nia->Nmi", Cvirt, amp_ia_Y)
        #     B_occ_Y = torch.einsum("Nmi,Nia->Nma", Cocc, amp_ia_Y)
        #     B += torch.einsum("Nmi,Nni->Nmn", B_virt_Y, B_virt_Y) - torch.einsum(
        #         "Nmi,Nni->Nmn", B_occ_Y, B_occ_Y
        #     )

        return B, B_virt, B_occ

    amp_ia_X = torch.einsum("bmi,brmn,bna->bria", Cocc, amp_X, Cvirt).squeeze(1)
    amp_ia_Xbar = torch.einsum("bmi,brmn,bna->bria", Cocc, amp_Xbar, Cvirt).squeeze(1)
    cis_densities = {}
    # cis_densities["transition_density"] = amp_X.squeeze(1)
    cis_densities["transition_density"] = torch.einsum("bmi,bia,bna->bmn", Cocc, amp_ia_X, Cvirt)
    cis_densities["transition_density_bar"] = torch.einsum("bmi,bia,bna->bmn", Cocc, amp_ia_Xbar, Cvirt)
    cis_densities["transition_density_2XXbar"] = (
        2.0 * cis_densities["transition_density"] - cis_densities["transition_density_bar"]
    )
    cis_densities["difference_density"], _, _ = difference_density(amp_ia_X, Cocc, Cvirt)
    _, B_virt, B_occ = difference_density(amp_ia_Xbar, Cocc, Cvirt)
    amp_ia_Xbar = 2.0 * amp_ia_X - amp_ia_Xbar
    _, Bbar_virt, Bbar_occ = difference_density(amp_ia_Xbar, Cocc, Cvirt)

    B = cis_densities["difference_density"]
    R = cis_densities["transition_density_bar"]
    Rbar = cis_densities["transition_density_2XXbar"]
    # Calculate z-vector
    # make RHS of the CPSCF equation:
    B_pi = makeA_pi_batched(mol, B.unsqueeze(1), w).squeeze(1) * 2.0
    R_pi = makeA_pi_batched(mol, R.unsqueeze(1), w).squeeze(1)
    Rbar_pi = makeA_pi_batched(mol, Rbar.unsqueeze(1), w).squeeze(1)
    RHS = -torch.einsum("Nni,Nmn,Nma->Nia", Cocc, B_pi, Cvirt)
    RHS -= torch.einsum("Nni,Nmn,Nma->Nia", Bbar_virt, R_pi, Cvirt)
    RHS += torch.einsum("Nni,Nmn,Nma->Nia", Cocc, R_pi, Bbar_occ)
    RHS -= torch.einsum("Nni,Nmn,Nma->Nia", B_virt, Rbar_pi, Cvirt)
    RHS += torch.einsum("Nni,Nmn,Nma->Nia", Cocc, Rbar_pi, B_occ)

    # if rpa:
    #     RHS -= torch.einsum("Nni,Nnm,Nma->Nia", B_virt_Y, R_pi, Cvirt)
    #     RHS += torch.einsum("Nni,Nnm,Nma->Nia", Cocc, R_pi, B_occ_Y)

    RHS = RHS.reshape(nmol, nocc * nvirt)
    ea_ei = e_mo[:, nocc:norb].unsqueeze(1) - e_mo[:, :nocc].unsqueeze(2)

    # C = mol.molecular_orbitals
    # debugE = (B * (C @ torch.diag_embed(e_mo[0, :norb]).unsqueeze(0) @ C.transpose(-2, -1))).sum(
    #     dim=(1, 2)
    # ) + 2.0 * (R * Rbar_pi).sum(dim=(1, 2))
    # print(f"DEBUG: XLESMD Gradient: energy is {debugE.item():.15f}")

    # Ad_inv_b = RHS/ea_ei
    # x1 = make_A_times_zvector(mol,Ad_inv_b,w,e_mo)

    def setup_applyA(mol, w, ea_ei, Cocc, Cvirt):
        def applyA(z):
            Az = make_A_times_zvector_batched(mol, z, w, ea_ei, Cocc, Cvirt)
            return Az

        return applyA

    A = setup_applyA(mol, w, ea_ei, Cocc, Cvirt)
    zvec = conjugate_gradient_batch(A, RHS, ea_ei.view(nmol, nocc * nvirt), tol=zvec_tolerance)

    z_ao = torch.einsum("Nmi,Nia,Nna->Nmn", Cocc, zvec.view(nmol, nocc, nvirt), Cvirt)
    cis_densities["relaxed_difference_density"] = (
        B + z_ao + z_ao.transpose(1, 2)
    )  # Now this contains the relaxed density

    del B_occ, B_virt, Bbar_occ, Bbar_virt, z_ao, zvec, RHS, B_pi, R_pi, Rbar_pi

    if calculate_dipole:
        make_cis_state_dipole(
            mol, cis_densities["difference_density"], cis_densities["relaxed_difference_density"], P0
        )
    # B0 = torch.stack([ unpackone(dens_BR[i,0], 4*nHeavy, nHydro, molsize * 4)
    #     for i in range(nmol)]).view(nmol,molsize * 4, molsize * 4)
    B0 = unpackone_batch(cis_densities["relaxed_difference_density"], 4 * nHeavy, nHydro, molsize * 4)
    # R0 = torch.stack([ unpackone(dens_BR[i,1], 4*nHeavy, nHydro, molsize * 4)
    #     for i in range(nmol)]).view(nmol,molsize * 4, molsize * 4)
    R0 = unpackone_batch(cis_densities["transition_density_bar"], 4 * nHeavy, nHydro, molsize * 4)
    Rbar0 = unpackone_batch(cis_densities["transition_density_2XXbar"], 4 * nHeavy, nHydro, molsize * 4)

    del cis_densities

    ###############################
    # Calculate the gradient of CIS energies

    npairs = mol.rij.shape[0]
    dtype = B0.dtype
    device = B0.device
    nmol = mol.nmol
    overlap_x = torch.zeros((npairs, 3, 4, 4), dtype=dtype, device=device)
    zeta = torch.cat((mol.parameters["zeta_s"].unsqueeze(1), mol.parameters["zeta_p"].unsqueeze(1)), dim=1)
    Xij = mol.xij * mol.rij.unsqueeze(1) * a0
    overlap_der_finiteDiff(
        overlap_x,
        mol.idxi,
        mol.idxj,
        mol.rij,
        Xij,
        mol.parameters["beta"],
        mol.ni,
        mol.nj,
        zeta,
        mol.const.qn_int,
    )

    w_x = torch.zeros(mol.rij.shape[0], 3, 10, 10, dtype=dtype, device=device)
    e1b_x, e2a_x = w_der(
        mol.const,
        mol.Z,
        mol.const.tore,
        mol.ni,
        mol.nj,
        w_x,
        mol.rij,
        mol.xij,
        Xij,
        mol.idxi,
        mol.idxj,
        mol.parameters["g_ss"],
        mol.parameters["g_pp"],
        mol.parameters["g_p2"],
        mol.parameters["h_sp"],
        mol.parameters["zeta_s"],
        mol.parameters["zeta_p"],
        riXH,
        ri,
    )

    B = B0.reshape(nmol, molsize, 4, molsize, 4).transpose(2, 3).reshape(nmol * molsize * molsize, 4, 4)
    P = P0.reshape(nmol, molsize, 4, molsize, 4).transpose(2, 3).reshape(nmol * molsize * molsize, 4, 4)
    if include_ground_state:
        pair_grad = core_core_der(mol, gam, w_x, method, parnuc)
        if mol.seqm_parameters.get("dispersion", False) and method == "AM1":
            pair_grad += dEdisp_dr(mol)
        B += 0.5 * P
        # Typically you add ground state density to excited state density if you want to include gradient of ground state energy in the gradient of excited state energy.
        # But there is a factor of 2 when contracting excited state density with two-electron gradient matrix (not sure why), but not for ground state density.
        # That's why I add 0.5 times the ground state density to the excited state density.
        # In doing so, I have to also add 0.5 time the contraction of ground state density with the one-electron gradient matrix. I add 0.5 time the overlap contribution here and 0.5 time core-valence term e1b_x and e2a_x below.
        pair_grad += 0.5 * (P[mol.mask].unsqueeze(1) * overlap_x).sum(dim=(2, 3))
    else:
        pair_grad = torch.zeros_like(Xij)

    # The following logic to form the coulomb and exchange integrals by contracting the two-electron integrals with the density matrix has been cribbed from fock.py

    # Exchange integrals
    # mu, nu in A
    # lambda, sigma in B
    # F_mu_lambda = Hcore - 0.5* \sum_{nu \in A} \sum_{sigma in B} P_{nu, sigma} * (mu nu, lambda, sigma)
    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #   0,     1         2       3       4         5       6      7         8        9
    ind = torch.tensor(
        [[0, 1, 3, 6], [1, 2, 4, 7], [3, 4, 5, 8], [6, 7, 8, 9]], dtype=torch.int64, device=device
    )
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
    # F_mu_nv = Hcore + \sum^B \sum_{lambda, sigma} P^B_{lambda, sigma} * (mu nu, lambda sigma)
    # as only upper triangle part is done, and put in order
    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    # weight for them are
    #  1       2       1        2        2        1        2       2        2       1
    weight = torch.tensor(
        [1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0], dtype=dtype, device=device
    ).reshape((-1, 10))
    # weight *= 0.5  # Multiply the weight by 0.5 because the contribution of coulomb integrals to engergy is calculated as 0.5*P_mu_nu*F_mu_nv

    indices = (0, 0, 1, 0, 1, 2, 0, 1, 2, 3), (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)
    PA = (P[mol.maskd[mol.idxi]][..., indices[0], indices[1]] * weight).unsqueeze(
        -1
    )  # Shape: (npairs, 10, 1)
    PB = (P[mol.maskd[mol.idxj]][..., indices[0], indices[1]] * weight).unsqueeze(
        -2
    )  # Shape: (npairs, 1, 10)

    suma = torch.sum(PA.unsqueeze(1) * w_x, dim=2)  # Shape: (npairs, 3, 10)

    scale_emat = torch.tensor(
        [[1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    if include_ground_state:
        pair_grad.add_(
            0.5 * (P[mol.maskd[mol.idxj], None, :, :] * e2a_x * scale_emat).sum(dim=(2, 3))
            + 0.5 * (P[mol.maskd[mol.idxi], None, :, :] * e1b_x * scale_emat).sum(dim=(2, 3))
        )

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
    pair_grad.add_(
        (B[mol.maskd[mol.idxj], None, :, :] * e2a_x).sum(dim=(2, 3))
        + (B[mol.maskd[mol.idxi], None, :, :] * e1b_x).sum(dim=(2, 3))
    )
    del e1b_x

    ###########################################################

    R_symmetrized = 0.5 * (R0 + R0.transpose(1, 2))
    R_symm = (
        R_symmetrized.reshape(nmol, molsize, 4, molsize, 4)
        .transpose(2, 3)
        .reshape(nmol * molsize * molsize, 4, 4)
    )
    del R_symmetrized
    Rbar_symmetrized = 0.5 * (Rbar0 + Rbar0.transpose(1, 2))
    # Rbar_symmetrized = Rbar0.clone()
    Rbar_symm = (
        Rbar_symmetrized.reshape(nmol, molsize, 4, molsize, 4)
        .transpose(2, 3)
        .reshape(nmol * molsize * molsize, 4, 4)
    )
    del Rbar_symmetrized

    Rdiag_symmetrized = R_symm[mol.maskd]
    PA = (
        Rdiag_symmetrized[mol.idxi][..., (0, 0, 1, 0, 1, 2, 0, 1, 2, 3), (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)]
        * weight
    ).unsqueeze(-1)
    PB = (
        Rdiag_symmetrized[mol.idxj][..., (0, 0, 1, 0, 1, 2, 0, 1, 2, 3), (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)]
        * weight
    ).unsqueeze(-2)

    suma = torch.sum(PA.unsqueeze(1) * w_x, dim=2)  # Shape: (npairs, 3, 10)
    sumA = overlap_KAB_x
    sumA.zero_()
    sumA[..., indices[0], indices[1]] = suma
    J_x_2a = e2a_x
    J_x_2a[:, :, :] = sumA

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
    pair_grad.add_(
        (2.0 * Rbar_symm[mol.maskd[mol.idxj], None, :, :] * J_x_2a).sum(dim=(2, 3))
        + (2.0 * Rbar_symm[mol.maskd[mol.idxi], None, :, :] * J_x_1b).sum(dim=(2, 3))
    )  # I can use R_symm instead of R here
    del J_x_2a
    del Rdiag_symmetrized

    Pp = R_symm[mol.mask].unsqueeze(1)
    for i in range(4):
        w_x_i = w_x[..., ind[i], :]
        for j in range(4):
            # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            overlap_KAB_x[..., i, j] = -0.5 * torch.sum(Pp * (w_x_i[..., :, ind[j]]), dim=(2, 3))

    pair_grad.add_((4.0 * Rbar_symm[mol.mask].unsqueeze(1) * overlap_KAB_x).sum(dim=(2, 3)))
    del Rbar_symm, R_symm

    R_antisymmetrized = 0.5 * (R0 - R0.transpose(1, 2))
    R_antisymm = (
        R_antisymmetrized.reshape(nmol, molsize, 4, molsize, 4)
        .transpose(2, 3)
        .reshape(nmol * molsize * molsize, 4, 4)
    )
    del R_antisymmetrized
    Rbar_antisymmetrized = 0.5 * (Rbar0 - Rbar0.transpose(1, 2))
    # Rbar_antisymmetrized = Rbar0.clone()
    Rbar_antisymm = (
        Rbar_antisymmetrized.reshape(nmol, molsize, 4, molsize, 4)
        .transpose(2, 3)
        .reshape(nmol * molsize * molsize, 4, 4)
    )
    del Rbar_antisymmetrized

    Pp = R_antisymm[mol.mask].unsqueeze(1)
    for i in range(4):
        w_x_i = w_x[..., ind[i], :]
        for j in range(4):
            # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            overlap_KAB_x[..., i, j] = -0.5 * torch.sum(Pp * (w_x_i[..., :, ind[j]]), dim=(2, 3))

    pair_grad.add_((4.0 * Rbar_antisymm[mol.mask].unsqueeze(1) * overlap_KAB_x).sum(dim=(2, 3)))

    # Define the gradient tensor
    grad_cis = torch.zeros(nmol * molsize, 3, dtype=dtype, device=device)

    # idxi/idxj are assumed to already index the full (nmol*molsize) layout; if padding is present,
    # map packed real-atom indices back to full indices as in anal_grad.contract_ao_derivatives_with_density.
    grad_cis.index_add_(0, mol.idxi, pair_grad)
    grad_cis.index_add_(0, mol.idxj, pair_grad, alpha=-1.0)

    grad_cis = grad_cis.view(nmol, molsize, 3)

    # torch.set_printoptions(precision=15)
    # print(f'Analytical CIS gradient is (eV/Angstrom):\n{grad_cis}')

    return grad_cis

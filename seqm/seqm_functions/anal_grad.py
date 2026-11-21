import torch
from torch import pow

from .cal_par import sp_additive_terms
from .constants import a0, ev
from .diat_overlap_PM6_SP import diatom_overlap_local_terms, diatom_overlap_rotate
from .dispersion import dispersion_pair_grad
from .energy import pair_nuclear_energy
from .fock import EMAT_SCALE_4, UPPER_IDX0_4, UPPER_IDX1_4, WEIGHT_10, K_ind_4, _cached_index, _cached_tensor
from .om1_overlap import diatom_overlap_matrix_OM1, diatom_resonance_matrix_OM1
from .om1_pair_backend import omx_rotate_local_terms
from .om2_hcore import build_omx_ortho_cache, build_omx_radial_context, omx_cor_difference, pair_directions
from .omx_utils import blocks_to_dense, dense_to_blocks
from .two_elec_two_center_int import (
    _pair_rotation_matrix,
    block_matvec,
    block_sandwich,
    local_sp_integral_matrix,
    local_xh_integral_vector,
    pair_rotation_matrix_derivative,
    rotate_sp_local,
    rotate_with_quaternion,
    two_elec_two_center_local_sp,
)

delta = 5e-6  # delta for finite difference calcs
_VECTORIZED_SP_DERIVATIVES = True


def _rotate_sp_derivatives(rotXH, rot, rot_derXH, rot_der, riXH, ri, riXH_x, ri_x):
    T = _pair_rotation_matrix(rot)
    dT = pair_rotation_matrix_derivative(rot, rot_der)
    W = local_sp_integral_matrix(ri)
    dW = local_sp_integral_matrix(ri_x)
    T3, W3 = T[:, None], W[:, None]
    # d(T W T^T) = dT W T^T + T dW T^T + T W dT^T; T and dT are block diagonal in the AO-pair ranks.
    w_x = block_sandwich(dT, W3, T3) + block_sandwich(T3, dW, T3) + block_sandwich(T3, W3, dT)

    T, dT = _pair_rotation_matrix(rotXH), pair_rotation_matrix_derivative(rotXH, rot_derXH)
    W, dW = local_xh_integral_vector(riXH), local_xh_integral_vector(riXH_x)
    wXH_x = block_matvec(dT, W[:, None]) + block_matvec(T[:, None], dW)
    return wXH_x, w_x


# @profile
def scf_analytic_grad(
    P0,
    molecule,
    const,
    method,
    mask,
    maskd,
    molsize,
    idxi,
    idxj,
    ni,
    nj,
    xij,
    rij,
    gam,
    parnuc,
    Z,
    gss,
    gpp,
    gp2,
    hsp,
    beta,
    zetas,
    zetap,
    riXH,
    ri,
):
    """
    Calculate the gradient of the ground state SCF energy
    in the units of ev/Angstrom
    The code follows the derivation outlined in
    Dewar, Michael JS, and Yukio Yamaguchi. "Analytical first derivatives of the energy in MNDO." Computers & Chemistry 2.1 (1978): 25-29.
    https://doi.org/10.1016/0097-8485(78)80005-9
    """
    if molecule.has_d or method not in {"PM3", "AM1", "MNDO", "PM6", "PM6_SP"}:
        raise Exception("Analytical gradients implented only for MNDO, AM1, PM3 and PM6 without d orbitals")

    # torch.set_printoptions(precision=6)
    # torch.set_printoptions(linewidth=110)

    # Xij (= Xj-Xi) is the vector from j to i in Angstroms
    # xij (= xj-xi) is the *unit* vector from j to i
    Xij = xij * rij.unsqueeze(1) * a0
    dtype = Xij.dtype
    device = Xij.device
    npairs = Xij.shape[0]
    qn_int = const.qn_int  # Principal quantum number of the valence shell

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

    # Two-center repulsion integral derivatives
    # Core-valence integral derivatives e1b_x and e2a_x also calculated as byproducts
    w_x = torch.zeros(rij.shape[0], 3, 10, 10, dtype=dtype, device=device)
    e1b_x, e2a_x = w_der(
        const, Z, tore, ni, nj, w_x, rij, xij, Xij, idxi, idxj, gss, gpp, gp2, hsp, zetas, zetap, riXH, ri
    )

    # Derivative of pair-nuclear Energy or E_core-core
    # pair_grad = torch.zeros((npairs,3),dtype=dtype, device=device)
    # pair_grad = core_core_der(alpha, rij, Xij, ZAZB, ni, nj, idxi, idxj, gam, w_x, method, parameters=parnuc)
    pair_grad = core_core_der(molecule, gam, w_x, method, parnuc)
    pair_grad += dispersion_pair_grad(molecule)
    return contract_ao_derivatives_with_density(
        P0, molecule, molsize, overlap_KAB_x, e1b_x, e2a_x, w_x, pair_grad, mask, maskd, idxi, idxj
    )


def contract_ao_derivatives_with_density(
    P0, molecule, molsize, overlap_KAB_x, e1b_x, e2a_x, w_x, pair_grad, mask, maskd, idxi, idxj
):
    dtype = P0.dtype
    device = P0.device
    nmol = molecule.species.shape[0]

    unrestricted = True if P0.dim() == 4 else False

    # Assembly
    if unrestricted:
        P = (
            (P0[:, 0] + P0[:, 1])
            .reshape((nmol, molsize, 4, molsize, 4))
            .transpose(2, 3)
            .reshape(nmol * molsize * molsize, 4, 4)
        )

        PAlpha_ = (
            P0.transpose(0, 1)
            .reshape((2, nmol, molsize, 4, molsize, 4))
            .transpose(3, 4)
            .reshape(2, nmol * molsize * molsize, 4, 4)
        )
    else:
        P = P0.reshape(nmol, molsize, 4, molsize, 4).transpose(2, 3).reshape(nmol * molsize * molsize, 4, 4)

    # The following logic to form the coulomb and exchange integrals by contracting the two-electron integrals with the density matrix has been cribbed from fock.py

    # Exchange integrals
    # mu, nu in A
    # lambda, sigma in B
    # F_mu_lambda = Hcore - 0.5* \sum_{nu \in A} \sum_{sigma in B} P_{nu, sigma} * (mu nu, lambda, sigma)
    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #   0,     1         2       3       4         5       6      7         8        9
    ind = _cached_index(K_ind_4, device)
    # mask has the indices of the lower (or upper) triangle blocks of the density matrix. Hence, P[mask] gives
    # us access to P_mu_lambda where mu is on atom A, lambda is on atom B
    if unrestricted:
        # sum_ = torch.empty(2,npairs,3,4,4,dtype=dtype,device=device)
        sum_ = overlap_KAB_x.unsqueeze(0).expand(2, *overlap_KAB_x.shape).clone()
        Pp = PAlpha_[:, mask].unsqueeze(2)
        for i in range(4):
            for j in range(4):
                # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
                sum_[..., i, j] -= torch.sum(Pp * w_x[None, ..., ind[i], :][..., :, ind[j]], dim=(3, 4))

        pair_grad.add_((Pp * sum_).sum(dim=(0, 3, 4)))
        del sum_
        # pair_grad.add_((P[mask].unsqueeze(1) * overlap_KAB_x).sum(dim=(2, 3)))

    else:
        Pp = P[mask].unsqueeze(1)
        for i in range(4):
            w_x_i = w_x[..., ind[i], :]
            for j in range(4):
                # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
                overlap_KAB_x[..., i, j] -= 0.5 * torch.sum(Pp * (w_x_i[..., :, ind[j]]), dim=(2, 3))

        pair_grad.add_((Pp * overlap_KAB_x).sum(dim=(2, 3)))

    # Coulomb integrals -- only on the diagonal
    # F_mu_nv = Hcore + \sum^B \sum_{lambda, sigma} P^B_{lambda, sigma} * (mu nu, lambda sigma)
    # as only upper triangle part is done, and put in order
    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    # weight for them are
    #  1       2       1        2        2        1        2       2        2       1

    # Multiply the weight by 0.5 because the contribution of coulomb integrals to energy is 0.5*P_mu_nu*F_mu_nv.
    weight = (_cached_tensor(WEIGHT_10, device, dtype) * 0.5).reshape((-1, 10))

    idx0 = _cached_index(UPPER_IDX0_4, device)
    idx1 = _cached_index(UPPER_IDX1_4, device)
    PA = (P[maskd[idxi]][..., idx0, idx1] * weight).unsqueeze(-1)  # Shape: (npairs, 10, 1)
    PB = (P[maskd[idxj]][..., idx0, idx1] * weight).unsqueeze(-2)  # Shape: (npairs, 1, 10)

    suma = torch.sum(PA.unsqueeze(1) * w_x, dim=2)  # Shape: (npairs, 3, 10)

    # Collect in sumA and sumB tensors
    # reususe overlap_KAB_x here instead of creating new arrays
    # I am going to be alliasing overlap_KAB_x to sumA and then further aliasing it to sumB
    # This seems like bad practice because I'm not allocating new memory but using the same tensor for all operations.
    # In the future, if this code is to be edited, be careful here
    sumA = overlap_KAB_x
    sumA.zero_()
    sumA[..., idx0, idx1] = suma
    e2a_x.add_(sumA)

    sumB = overlap_KAB_x
    sumB.zero_()
    sumb = torch.sum(PB.unsqueeze(1) * w_x, dim=3)  # Shape: (npairs, 3, 10)
    sumB[..., idx0, idx1] = sumb
    e1b_x.add_(sumB)

    # Core-elecron interaction
    scale_emat = _cached_tensor(EMAT_SCALE_4, device, dtype)
    e1b_x *= scale_emat
    e2a_x *= scale_emat
    # e1b_x.add_(e1b_x.triu(1).transpose(2, 3))
    # e2a_x.add_(e2a_x.triu(1).transpose(2, 3))
    pair_grad.add_(
        (P[maskd[idxj], None, :, :] * e2a_x).sum(dim=(2, 3))
        + (P[maskd[idxi], None, :, :] * e1b_x).sum(dim=(2, 3))
    )
    # pair_grad.add_((P[maskd[idxj],None,:,:]*e2a_x.triu(1)).sum(dim=(2,3)) + (P[maskd[idxi],None,:,:]*e1b_x.triu(1)).sum(dim=(2,3)))

    # Define the gradient tensor
    grad = torch.zeros(nmol * molsize, 3, dtype=dtype, device=device)

    # idxi/idxj are packed indices over real atoms; map them back to full atom indices.
    real_atoms = torch.arange(nmol * molsize, device=device, dtype=torch.int64)[
        molecule.species.reshape(-1) > 0
    ]
    grad.index_add_(0, real_atoms[idxi], pair_grad)
    grad.index_add_(0, real_atoms[idxj], pair_grad, alpha=-1.0)

    # print(f'Analytical SCF gradient is:\n{grad.view(nmol,molsize,3)}')
    grad = grad.view(nmol, molsize, 3)
    return grad


# @profile
def scf_grad(
    P0,
    molecule,
    const,
    method,
    mask,
    maskd,
    molsize,
    idxi,
    idxj,
    ni,
    nj,
    xij,
    rij,
    parnuc,
    Z,
    beta,
    gam,
    zetas,
    zetap,
):
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
    npairs = Xij.shape[0]
    qn_int = const.qn_int  # Principal quantum number of the valence shell

    # I will use this tensor to store the gradient of the overlap matrix, and then that of the exchange integrals
    overlap_KAB_x = torch.zeros((npairs, 3, 4, 4), dtype=dtype, device=device)
    w_x_new = torch.zeros(rij.shape[0], 3, 10, 10, device=device, dtype=dtype)

    if method in {"OM1", "OM2", "OM3"}:
        e1b_x_new, e2a_x_new, fko_x, omx_orthogonalization_grad, _ = omx_fd(
            molecule, overlap_KAB_x, w_x_new, Xij, ni, nj, idxi, idxj, method, P0
        )

        tore = const.tore  # Charges
        ZAZB = tore[ni] * tore[nj]
        fko = gam
        pair_grad = (
            ZAZB.unsqueeze(1)
            * ev
            * (fko_x / rij.unsqueeze(1) + fko.unsqueeze(1) * Xij / (a0 * a0 * torch.pow(rij, 3)).unsqueeze(1))
        )

    else:
        # overlap_der(overlap_KAB_x,zetas,zetap,qn_int,ni,nj,rij,beta,idxi,idxj,Xij)
        # We will use finite-differnce for the overlap derivative because analytical expression for derivatives of
        # the overlap of slater orbitals is v complicated
        zeta = torch.cat((zetas.unsqueeze(1), zetap.unsqueeze(1)), dim=1)
        overlap_der_finiteDiff(overlap_KAB_x, idxi, idxj, rij, Xij, beta, ni, nj, zeta, qn_int)

        e1b_x_new, e2a_x_new = w_derivative_numerical(molecule, Xij, w_x_new)
        pair_grad = core_core_der(molecule, gam, w_x_new, method, parnuc)

    pair_grad += dispersion_pair_grad(molecule)
    grad = contract_ao_derivatives_with_density(
        P0,
        molecule,
        molsize,
        overlap_KAB_x,
        e1b_x_new,
        e2a_x_new,
        w_x_new,
        pair_grad,
        mask,
        maskd,
        idxi,
        idxj,
    )

    if method in {"OM2", "OM3"}:
        grad += omx_orthogonalization_grad

    return grad


def _density_to_ortho_blocks(P0, molecule, unrestricted=False):
    molsize = molecule.molsize
    if unrestricted:
        Ptot = P0[:, 0] + P0[:, 1]
    else:
        Ptot = P0
    if Ptot.dim() == 3:
        Ptot = Ptot.unsqueeze(1)
    ndensity = Ptot.shape[1]
    return (
        Ptot.reshape(molecule.nmol, ndensity, molsize, 4, molsize, 4)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(molecule.nmol * molsize * molsize, ndensity, 4, 4)
    )


def omx_threebody_ortho_grad(
    molecule, P0, S_x, B_x, pair_core_semi_x, ortho_cache=None, unrestricted=False, vel_eff=None
):
    """Return the OMx orthogonalization correction without squeezing batch axes."""
    cache = build_omx_ortho_cache(molecule) if ortho_cache is None else ortho_cache
    pi, pj = cache["pi"], cache["pj"]
    P_blocks = _density_to_ortho_blocks(P0, molecule, unrestricted=unrestricted)
    method = molecule.method
    if method == "OM3":
        grad = betor_grad_dense(
            P_blocks,
            cache["S5"],
            cache["B5"],
            S_x,
            B_x,
            molecule,
            (pi["gval1"], pj["gval1"]),
            vel_eff=vel_eff,
        )
    else:
        COR = molecule.om2_COR
        grad = betor_grad_dense(
            P_blocks,
            cache["S5"],
            cache["B5"],
            S_x,
            B_x,
            molecule,
            (pi["gval1"], pj["gval1"]),
            COR=COR,
            gval2=(pi["gval2"], pj["gval2"]),
            pair_core_semi_x=pair_core_semi_x,
            vel_eff=vel_eff,
        )
    return grad


repeat_tensor = lambda x: torch.cat([x, x])


def w_derivative_numerical(mol, Xij, w_x_new):
    """Finite-difference derivative of the s/p two-center integrals and core attractions (fills ``w_x_new``)."""
    if mol.has_d:
        raise NotImplementedError("Numerical two-center derivatives are only implemented for s/p methods")
    npairs = Xij.shape[0]
    p = mol.parameters
    idxi2, idxj2, ni2, nj2 = (t.repeat(2) for t in (mol.idxi, mol.idxj, mol.ni, mol.nj))

    def local_at(r):
        return two_elec_two_center_local_sp(
            mol.const,
            idxi2,
            idxj2,
            ni2,
            nj2,
            r,
            mol.Z,
            p["zeta_s"],
            p["zeta_p"],
            p["g_ss"],
            p["g_pp"],
            p["g_p2"],
            p["h_sp"],
            mol.method,
        )

    def rotate(local, units):
        copies = units.shape[0] // npairs
        return rotate_sp_local(local, mol.const.tore, mol.ni.repeat(copies), mol.nj.repeat(copies), units)

    derivatives = _radial_rotation_derivative(local_at, rotate, Xij)
    w_x_new.copy_(derivatives["w"])
    return derivatives["e1b"], derivatives["e2a"]


def core_core_der_fd(mol, method, gam, parameters):
    """
    Core-core repulsion derivative by finite differences. The energy depends on the pair distance only, so
    only the radial term of ``_radial_rotation_derivative`` is nonzero (``gam`` is held fixed).
    """
    idxi_ = repeat_tensor(mol.idxi)
    idxj_ = repeat_tensor(mol.idxj)
    rho_0 = 0.5 * ev / mol.parameters["g_ss"]
    rho0xi = rho_0[idxi_]
    rho0xj = rho_0[idxj_]
    rho_core = mol.parameters["rho_core"]
    A = rho_core[idxi_] != 0.000
    B = rho_core[idxj_] != 0.000
    rho0xi[A] = rho_core[idxi_][A]
    rho0xj[B] = rho_core[idxj_][B]
    ni_, nj_, alp, chi, gam_ = (repeat_tensor(t) for t in (mol.ni, mol.nj, mol.alp, mol.chi, gam))

    def local_at(r):
        energy = pair_nuclear_energy(
            None,
            mol.const,
            mol.nmol,
            ni_,
            nj_,
            idxi_,
            idxj_,
            r,
            rho0xi,
            rho0xj,
            alp,
            chi,
            gam_,
            method=method,
            parameters=parameters,
        )
        return {"energy": energy}

    Xij = mol.xij * mol.rij.unsqueeze(1) * a0
    return _radial_rotation_derivative(local_at, lambda local, units: local, Xij)["energy"]


def core_core_der(mol, gam, w_x, method, parameters):
    if method == "PM6" or method == "PM6_SP" or method == "PM6_SP_STAR":
        return core_core_der_fd(mol, method, gam, parameters)
    ni = mol.ni
    nj = mol.nj
    idxi = mol.idxi
    idxj = mol.idxj
    xij = mol.xij
    rij = mol.rij
    rija = rij * a0
    Xij = xij * rij.unsqueeze(1) * a0
    alpha = parameters[0]
    const = mol.const
    tore = const.tore  # Charges
    ZAZB = tore[ni] * tore[nj]
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
    t3 = alpha[idxj] * t3
    coreTerm = ZAZB * gam / rija * (prefactor * tmp + t3)
    # The derivative of the core-core term is ZAZB*(SASA|SBSB)*dg/dx + ZAZB*g*d(SASA|SBSB)/dx
    # Here we calculate the first term, i.e., ZAZB*(SASA|SBSB)*dg/dx
    pair_grad = coreTerm.unsqueeze(1) * Xij
    # Here we add the second term in the derivative for MNDO, i.e., ZAZB*g*d(SASA|SBSB)/dx
    pair_grad.add_((ZAZB * g).unsqueeze(1) * w_x[:, :, 0, 0])
    if method == "MNDO":
        return pair_grad

    # if method=='PM6':
    #     # Here we don't have the MNDO term, so pair_grad has to be reinitialized
    #     pair_grad.zero_()

    # For AM1 and PM3, in addition to the MNDO term we also have
    # two gaussian terms for PM3
    # 3~4 terms for AM1
    _, K, L, M = parameters
    # K, L , M shape (natoms,2 or 4)
    inv_rija = rija.reciprocal()
    inv_rija3 = torch.pow(inv_rija, 3)
    t4 = ZAZB * inv_rija
    t5 = torch.sum(K[idxi] * torch.exp(-L[idxi] * (rija[:, None] - M[idxi]) ** 2), dim=1)
    t6 = torch.sum(K[idxj] * torch.exp(-L[idxj] * (rija[:, None] - M[idxj]) ** 2), dim=1)
    pair_grad.add_((ZAZB * inv_rija3 * (t5 + t6)).unsqueeze(1) * Xij)
    t5_der = torch.sum(
        K[idxi] * torch.exp(-L[idxi] * (rija[:, None] - M[idxi]) ** 2) * L[idxi] * (rija[:, None] - M[idxi]),
        dim=1,
    )
    t6_der = torch.sum(
        K[idxj] * torch.exp(-L[idxj] * (rija[:, None] - M[idxj]) ** 2) * L[idxj] * (rija[:, None] - M[idxj]),
        dim=1,
    )
    pair_grad.add_((2.0 * t4 * inv_rija * (t5_der + t6_der)).unsqueeze(1) * Xij)
    if method == "PM3" or method == "AM1":
        return pair_grad
    # Put PM6 specific grad here
    # if method=='PM6':
    #       return pair_grad
    else:
        raise ValueError("Supported Method: MNDO, AM1, PM3")


def w_der(const, Z, tore, ni, nj, w_x, rij, xij, Xij, idxi, idxj, gss, gpp, gp2, hsp, zetas, zetap, riXH, ri):
    # Two-center repulsion integral derivatives
    HH = (ni == 1) & (nj == 1)
    XX = (ni > 1) & (nj > 1)
    dd, qq, rho_0, rho_1, rho_2 = sp_additive_terms(const, Z, zetas, zetap, gss, gpp, gp2, hsp)

    der_TETCILF(
        w_x,
        ni,
        nj,
        xij,
        Xij,
        rij,
        dd[idxi],
        dd[idxj],
        qq[idxi],
        qq[idxj],
        rho_0[idxi],
        rho_0[idxj],
        rho_1[idxi],
        rho_1[idxj],
        rho_2[idxi],
        rho_2[idxj],
        riXH,
        ri,
    )

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
    """Finite-difference derivative of the beta-weighted overlap blocks (only pairs within the overlap cutoff)."""
    pairs = (rij <= overlap_cutoff).nonzero(as_tuple=False).squeeze(1)
    if pairs.numel() > 0:
        ni_p, nj_p = ni[pairs].repeat(2), nj[pairs].repeat(2)
        zeta_i, zeta_j = zeta[idxi[pairs]].repeat(2, 1), zeta[idxj[pairs]].repeat(2, 1)

        def local_at(r):
            return {"terms": diatom_overlap_local_terms(ni_p, nj_p, r, zeta_i, zeta_j, qn_int)}

        def rotate(local, units):
            return {"overlap": diatom_overlap_rotate(local["terms"], units)}

        overlap_KAB_x[pairs] = _radial_rotation_derivative(local_at, rotate, Xij[pairs])["overlap"]

    overlap_KAB_x[..., 0, 0] *= (beta[idxi, 0] + beta[idxj, 0]).unsqueeze(1)
    overlap_KAB_x[..., 0, 1:] *= (beta[idxi, 0:1] + beta[idxj, 1:2]).unsqueeze(1)
    overlap_KAB_x[..., 1:, 0] *= (beta[idxi, 1:2] + beta[idxj, 0:1]).unsqueeze(1)
    overlap_KAB_x[..., 1:, 1:] *= (beta[idxi, 1:2, None] + beta[idxj, 1:2, None]).unsqueeze(1)


def der_TETCILF(
    w_x_final, ni, nj, xij, Xij, r0, da0, db0, qa0, qb0, rho0a, rho0b, rho1a, rho1b, rho2a, rho2b, riXH, ri
):
    dtype = r0.dtype
    device = r0.device

    HH = (ni == 1) & (nj == 1)
    XH = (ni > 1) & (nj == 1)
    XX = (ni > 1) & (nj > 1)

    # Hydrogen - Hydrogen
    # aeeHH = (rho0a[HH]+rho0b[HH])**2
    # # Dividing by a0^2 for gradient in eV/ang
    term = -ev / (a0 * a0) / r0.unsqueeze(1) * Xij
    ee = -r0 * pow((r0**2 + (rho0a + rho0b) ** 2), -1.5)
    ee_x = term * ee.unsqueeze(1)
    riHH_x = ee_x[HH, :]

    # Heavy atom - Hydrogen
    # aeeXH = (rho0a[XH]+rho0b[XH])**2
    rXH = r0[XH]
    daXH = da0[XH]
    qaXH = qa0[XH] * 2.0
    adeXH = (rho1a[XH] + rho0b[XH]) ** 2
    aqeXH = (rho2a[XH] + rho0b[XH]) ** 2
    dsqr6XH = 2.0 * rXH * pow(rXH**2 + aqeXH, -1.5)
    riXH_x = torch.zeros(XH.sum(), 3, 4, dtype=dtype, device=device)
    eeXH = ee[XH]
    riXH_x[..., 1 - 1] = ee_x[XH, :]
    riXH_x[..., 2 - 1] = (
        -0.5
        * term[XH]
        * (
            (rXH + daXH) * pow((rXH + daXH) ** 2 + adeXH, -1.5)
            - (rXH - daXH) * pow((rXH - daXH) ** 2 + adeXH, -1.5)
        ).unsqueeze(1)
    )
    riXH_x[..., 3 - 1] = term[XH] * (
        eeXH
        + 0.25
        * (
            -(rXH + qaXH) * pow((rXH + qaXH) ** 2 + aqeXH, -1.5)
            - (rXH - qaXH) * pow((rXH - qaXH) ** 2 + aqeXH, -1.5)
            + dsqr6XH
        )
    ).unsqueeze(1)
    riXH_x[..., 4 - 1] = term[XH] * (
        eeXH + 0.25 * (-2.0 * rXH * pow(rXH**2 + qaXH**2 + aqeXH, -1.5) + dsqr6XH)
    ).unsqueeze(1)

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
    ade = (rho1a[XX] + rho0b[XX]) ** 2
    aqe = (rho2a[XX] + rho0b[XX]) ** 2
    aed = (rho0a[XX] + rho1b[XX]) ** 2
    aeq = (rho0a[XX] + rho2b[XX]) ** 2
    axx = (rho1a[XX] + rho1b[XX]) ** 2
    adq = (rho1a[XX] + rho2b[XX]) ** 2
    aqd = (rho2a[XX] + rho1b[XX]) ** 2
    aqq = (rho2a[XX] + rho2b[XX]) ** 2
    ee = ee[XX]
    dze = (r + da) * pow((r + da) ** 2 + ade, -1.5) - (r - da) * pow((r - da) ** 2 + ade, -1.5)
    dsqr6 = 2.0 * r * pow(r**2 + aqe, -1.5)
    qzze = -(r - qa) * pow((r - qa) ** 2 + aqe, -1.5) - (r + qa) * pow((r + qa) ** 2 + aqe, -1.5) + dsqr6
    qxxe = -2.0 * r * pow(r**2 + qa**2 + aqe, -1.5) + dsqr6
    edz = (r - db) * pow((r - db) ** 2 + aed, -1.5) - (r + db) * pow((r + db) ** 2 + aed, -1.5)
    dsqr12 = 2.0 * r * pow(r**2 + aeq, -1.5)
    eqzz = -(r - qb) * pow((r - qb) ** 2 + aeq, -1.5) - (r + qb) * pow((r + qb) ** 2 + aeq, -1.5) + dsqr12
    eqxx = -2.0 * r * pow(r**2 + qb**2 + aeq, -1.5) + dsqr12
    dsqr20 = 2.0 * (r + da) * pow((r + da) ** 2 + adq, -1.5)
    dsqr22 = 2.0 * (r - da) * pow((r - da) ** 2 + adq, -1.5)
    dsqr24 = 2.0 * (r - db) * pow((r - db) ** 2 + aqd, -1.5)
    dsqr26 = 2.0 * (r + db) * pow((r + db) ** 2 + aqd, -1.5)
    dsqr36 = 4.0 * (r) * pow(r**2 + aqq, -1.5)
    dsqr39 = 4.0 * (r) * pow(r**2 + qa**2 + aqq, -1.5)
    dsqr40 = 4.0 * (r) * pow(r**2 + qb**2 + aqq, -1.5)
    dsqr42 = 2.0 * (r - qb) * pow((r - qb) ** 2 + aqq, -1.5)
    dsqr44 = 2.0 * (r + qb) * pow((r + qb) ** 2 + aqq, -1.5)
    dsqr46 = 2.0 * (r + qa) * pow((r + qa) ** 2 + aqq, -1.5)
    dsqr48 = 2.0 * (r - qa) * pow((r - qa) ** 2 + aqq, -1.5)
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
    ri_x[..., 6 - 1] = (
        0.25
        * term
        * (
            -(r + da - db) * pow((r + da - db) ** 2 + axx, -1.5)
            - (r - da + db) * pow((r - da + db) ** 2 + axx, -1.5)
            + (r - da - db) * pow((r - da - db) ** 2 + axx, -1.5)
            + (r + da + db) * pow((r + da + db) ** 2 + axx, -1.5)
        ).unsqueeze(1)
    )
    # RI(7) = DXDX = EV1/SQR(14) - EV1/SQR(15)
    ri_x[..., 7 - 1] = (
        0.5
        * term
        * (
            -r * pow(r**2 + (da - db) ** 2 + axx, -1.5) + r * pow(r**2 + (da + db) ** 2 + axx, -1.5)
        ).unsqueeze(1)
    )
    # RI(8) = -EDZ -QZZDZ
    # QZZDZ = -EV3/SQR(32) + EV3/SQR(33) - EV3/SQR(34) + EV3/SQR(35)
    # + EV2/SQR(24) - EV2/SQR(26)
    ri_x[..., 8 - 1] = -term * (
        0.5 * edz
        + 0.125
        * (
            (r + qa - db) * pow((r + qa - db) ** 2 + aqd, -1.5)
            - (r + qa + db) * pow((r + qa + db) ** 2 + aqd, -1.5)
            + (r - qa - db) * pow((r - qa - db) ** 2 + aqd, -1.5)
            - (r - qa + db) * pow((r - qa + db) ** 2 + aqd, -1.5)
            - dsqr24
            + dsqr26
        )
    ).unsqueeze(1)
    # RI(9) = -EDZ -QXXDZ
    # QXXDZ =  EV2/SQR(24) - EV2/SQR(25) - EV2/SQR(26) + EV2/SQR(27)
    ri_x[..., 9 - 1] = -term * (
        0.5 * edz
        + 0.125
        * (
            -dsqr24
            + 2.0 * (r - db) * pow((r - db) ** 2 + qa**2 + aqd, -1.5)
            + dsqr26
            - 2.0 * (r + db) * pow((r + db) ** 2 + qa**2 + aqd, -1.5)
        )
    ).unsqueeze(1)

    # sqr(54)-sqr(72) use qa1 and qb1
    # RI(10) = -QXZDX
    # QXZDX = -EV2/SQR(58) + EV2/SQR(59) + EV2/SQR(60) - EV2/SQR(61)
    ri_x[..., 10 - 1] = (
        -0.25
        * term
        * (
            (r + qa1) * pow((qa1 - db) ** 2 + (r + qa1) ** 2 + aqd, -1.5)
            - (r - qa1) * pow((qa1 - db) ** 2 + (r - qa1) ** 2 + aqd, -1.5)
            - (r + qa1) * pow((qa1 + db) ** 2 + (r + qa1) ** 2 + aqd, -1.5)
            + (r - qa1) * pow((qa1 + db) ** 2 + (r - qa1) ** 2 + aqd, -1.5)
        ).unsqueeze(1)
    )
    # RI(11) =  EE + EQZZ
    ri_x[..., 11 - 1] = term * (ee + 0.25 * eqzz).unsqueeze(1)
    # RI(12) =  EE + EQXX
    ri_x[..., 12 - 1] = term * (ee + 0.25 * eqxx).unsqueeze(1)
    # RI(13) = -DZE -DZQZZ
    # DZQZZ = -EV3/SQR(28) + EV3/SQR(29) - EV3/SQR(30) + EV3/SQR(31)
    #  - EV2/SQR(22) + EV2/SQR(20)
    ri_x[..., 13 - 1] = -term * (
        0.5 * dze
        + 0.125
        * (
            +(r + da - qb) * pow((r + da - qb) ** 2 + adq, -1.5)
            - (r - da - qb) * pow((r - da - qb) ** 2 + adq, -1.5)
            + (r + da + qb) * pow((r + da + qb) ** 2 + adq, -1.5)
            - (r - da + qb) * pow((r - da + qb) ** 2 + adq, -1.5)
            + dsqr22
            - dsqr20
        )
    ).unsqueeze(1)
    #
    # RI(14) = -DZE -DZQXX
    # DZQXX =  EV2/SQR(20) - EV2/SQR(21) - EV2/SQR(22) + EV2/SQR(23)
    ri_x[..., 14 - 1] = -term * (
        0.5 * dze
        + 0.125
        * (
            -dsqr20
            + dsqr22
            + 2.0 * (r + da) * pow((r + da) ** 2 + qb**2 + adq, -1.5)
            - 2.0 * (r - da) * pow((r - da) ** 2 + qb**2 + adq, -1.5)
        )
    ).unsqueeze(1)
    # RI(15) = -DXQXZ
    # DXQXZ = -EV2/SQR(54) + EV2/SQR(55) + EV2/SQR(56) - EV2/SQR(57)
    # sqr(54)-sqr(72) use qa1 and qb1
    ri_x[..., 15 - 1] = (
        -0.25
        * term
        * (
            (r - qb1) * pow((da - qb1) ** 2 + (r - qb1) ** 2 + adq, -1.5)
            - (r + qb1) * pow((da - qb1) ** 2 + (r + qb1) ** 2 + adq, -1.5)
            - (r - qb1) * pow((da + qb1) ** 2 + (r - qb1) ** 2 + adq, -1.5)
            + (r + qb1) * pow((da + qb1) ** 2 + (r + qb1) ** 2 + adq, -1.5)
        ).unsqueeze(1)
    )
    # RI(16) = EE +EQZZ +QZZE +QZZQZZ
    # QZZQZZ = EV4/SQR(50) + EV4/SQR(51) + EV4/SQR(52) + EV4/SQR(53)
    # - EV3/SQR(48) - EV3/SQR(46) - EV3/SQR(42) - EV3/SQR(44)
    # + EV2/SQR(36)
    ri_x[..., 16 - 1] = term * (
        ee
        + 0.25 * eqzz
        + 0.25 * qzze
        + 0.0625
        * (
            -(r + qa - qb) * pow((r + qa - qb) ** 2 + aqq, -1.5)
            - (r + qa + qb) * pow((r + qa + qb) ** 2 + aqq, -1.5)
            - (r - qa - qb) * pow((r - qa - qb) ** 2 + aqq, -1.5)
            - (r - qa + qb) * pow((r - qa + qb) ** 2 + aqq, -1.5)
            + dsqr48
            + dsqr46
            + dsqr42
            + dsqr44
            - dsqr36
        )
    ).unsqueeze(1)
    # RI(17) = EE +EQZZ +QXXE +QXXQZZ
    # QXXQZZ = EV3/SQR(43) + EV3/SQR(45) - EV3/SQR(42) - EV3/SQR(44)
    #  - EV2/SQR(39) + EV2/SQR(36)
    ri_x[..., 17 - 1] = term * (
        ee
        + 0.25 * eqzz
        + 0.25 * qxxe
        + 0.0625
        * (
            -2.0 * (r - qb) * pow((r - qb) ** 2 + qa**2 + aqq, -1.5)
            - 2.0 * (r + qb) * pow((r + qb) ** 2 + qa**2 + aqq, -1.5)
            + dsqr42
            + dsqr44
            + dsqr39
            - dsqr36
        )
    ).unsqueeze(1)
    # RI(18) = EE +EQXX +QZZE +QZZQXX
    # QZZQXX = EV3/SQR(47) + EV3/SQR(49) - EV3/SQR(46) - EV3/SQR(48)
    #  - EV2/SQR(40) + EV2/SQR(36)
    ri_x[..., 18 - 1] = term * (
        ee
        + 0.25 * eqxx
        + 0.25 * qzze
        + 0.0625
        * (
            -2.0 * (r + qa) * pow((r + qa) ** 2 + qb**2 + aqq, -1.5)
            - 2.0 * (r - qa) * pow((r - qa) ** 2 + qb**2 + aqq, -1.5)
            + dsqr46
            + dsqr48
            + dsqr40
            - dsqr36
        )
    ).unsqueeze(1)
    # RI(19) = EE +EQXX +QXXE +QXXQXX
    # QXXQXX = EV3/SQR(37) + EV3/SQR(38) - EV2/SQR(39) - EV2/SQR(40)
    # + EV2/SQR(36)
    qxxqxx = (
        -2.0 * r * pow(r**2 + (qa - qb) ** 2 + aqq, -1.5)
        - 2.0 * r * pow(r**2 + (qa + qb) ** 2 + aqq, -1.5)
        + dsqr39
        + dsqr40
        - dsqr36
    )
    ri_x[..., 19 - 1] = term * (ee + 0.25 * eqxx + 0.25 * qxxe + 0.0625 * qxxqxx).unsqueeze(1)
    # RI(20) = QXZQXZ
    # QXZQXZ = EV3/SQR(65) - EV3/SQR(67) - EV3/SQR(69) + EV3/SQR(71)
    # - EV3/SQR(66) + EV3/SQR(68) + EV3/SQR(70) - EV3/SQR(72)
    # sqr(54)-sqr(72) use qa1 and qb1
    ri_x[..., 20 - 1] = (
        0.125
        * term
        * (
            -(r + qa1 - qb1) * pow((r + qa1 - qb1) ** 2 + (qa1 - qb1) ** 2 + aqq, -1.5)
            + (r + qa1 + qb1) * pow((r + qa1 + qb1) ** 2 + (qa1 - qb1) ** 2 + aqq, -1.5)
            + (r - qa1 - qb1) * pow((r - qa1 - qb1) ** 2 + (qa1 - qb1) ** 2 + aqq, -1.5)
            - (r - qa1 + qb1) * pow((r - qa1 + qb1) ** 2 + (qa1 - qb1) ** 2 + aqq, -1.5)
            + (r + qa1 - qb1) * pow((r + qa1 - qb1) ** 2 + (qa1 + qb1) ** 2 + aqq, -1.5)
            - (r + qa1 + qb1) * pow((r + qa1 + qb1) ** 2 + (qa1 + qb1) ** 2 + aqq, -1.5)
            - (r - qa1 - qb1) * pow((r - qa1 - qb1) ** 2 + (qa1 + qb1) ** 2 + aqq, -1.5)
            + (r - qa1 + qb1) * pow((r - qa1 + qb1) ** 2 + (qa1 + qb1) ** 2 + aqq, -1.5)
        ).unsqueeze(1)
    )
    # RI(21) = EE +EQXX +QXXE +QXXQYY
    # QXXQYY = EV2/SQR(41) - EV2/SQR(39) - EV2/SQR(40) + EV2/SQR(36)
    qxxqyy = -4.0 * r * pow(r**2 + qa**2 + qb**2 + aqq, -1.5) + dsqr39 + dsqr40 - dsqr36
    ri_x[..., 21 - 1] = term * (ee + 0.25 * eqxx + 0.25 * qxxe + 0.0625 * qxxqyy).unsqueeze(1)
    # RI(22) = PP * (QXXQXX -QXXQYY)
    ri_x[..., 22 - 1] = 0.03125 * term * (qxxqxx - qxxqyy).unsqueeze(1)

    # We have the derivatives of the 2-center-2-elec integrals in the local frame
    # In the local frame for p-orbitals we have p-sigma (along the axis) ,p-pi,p-pi' (perpendicular to the axis)
    # But in the molecular frame we have px,py,pz which are rotations of p-sigma, p-pi, p-pi'
    # The p orbitals rotate just like the coordinate frame, so the rotation matrix is easy to express
    # We now make the rotation matrix and its derivative for the p-orbitals

    # rot = torch.zeros(r0.shape[0], 3, 3, device=device, dtype=dtype)
    # rot_der = torch.zeros(r0.shape[0], 3, 3, 3, device=device, dtype=dtype)
    #
    # rxy2 = torch.square(Xij[:, 0]) + torch.square(Xij[:, 1])
    # # ryz2 = torch.square(Xij[:, 1]) + torch.square(Xij[:, 2])
    # # rxz2 = torch.square(Xij[:, 0]) + torch.square(Xij[:, 2])
    # axis_tolerance = 1e-12
    # onerij = 1.0 / a0 / r0
    #
    # # Xalign = ryz2 < axis_tolerance
    # # Yalign = rxz2 < axis_tolerance
    # Zalign = rxy2 < axis_tolerance
    # # Noalign = ~(Xalign | Yalign | Zalign)
    # Noalign = ~(Zalign)
    # # if torch.any(Zalign):
    # #     print(f"Unfortunately z-axes align for {xij.shape[0]+ ~Noalign.sum() + 1}/{xij.shape[0]} pairs. This may cause some numerical instabilities in the derivative of two-electron integrals")
    #
    # xij_ = -xij[Noalign, ...]
    # rot[Noalign, 0, :] = xij_
    # onerxy = 1.0 / torch.sqrt(rxy2[Noalign])
    # rxy_over_rab = (torch.sqrt(rxy2) / r0)[Noalign] / a0
    # rab_over_rxy = a0 * r0[Noalign] * onerxy
    # rab_over_rxy_sq = torch.square(rab_over_rxy)
    #
    # # The (1,0) element of the rotation matrix is -Y/sqrt(X^2+Y^2)*sign(X). If X (=xi-xj) is zero then there is a discontinuity in the sign function
    # # and hence the derivative will not exist. So I'm printing a warning that there might be numerical errors here
    # # Similaryly the (1,1) element of the rotation matrix is abs(X/sqrt(X^2+Y^2)). Again, the derivative of abs(X) will not exist when X=0, and hence this
    # # will lead to errors.
    #
    # # if torch.any(xij_[:, 0] == 0):
    # #     print(
    # #         "WARNING: The x component of the pair distance is zero. This could lead to instabilities in the derivative of the rotation matrix becuase it is discontinuous at this point"
    # #     )
    #
    # # As a quick-fix, I will add a small number (eps) when calculating sign(X) to avoid the aforementioned instability
    # rot[Noalign, 1, 0] = -xij_[:, 1] * rab_over_rxy
    # rot[Noalign, 1, 1] = xij_[:, 0] * rab_over_rxy
    #
    # rot[Noalign, 2, 0] = xij_[:, 0] * xij_[:, 2] * rab_over_rxy
    # rot[Noalign, 2, 1] = xij_[:, 1] * xij_[:, 2] * rab_over_rxy
    # rot[Noalign, 2, 2] = -rxy_over_rab
    #
    # # Derivative of the rotation matrix
    # termX = xij_[:, 0] * onerij[Noalign]
    # termY = xij_[:, 1] * onerij[Noalign]
    # termZ = xij_[:, 2] * onerij[Noalign]
    # # term = Xij[Noalign,:]*onerij.unsqueeze(1)
    # rot_der[Noalign, 0, 0, 0] = onerij[Noalign] - xij_[:, 0] * termX
    # rot_der[Noalign, 0, 0, 1] = -xij_[:, 0] * termY
    # rot_der[Noalign, 0, 0, 2] = -xij_[:, 0] * termZ
    #
    # rot_der[Noalign, 1, 0, 0] = -xij_[:, 1] * termX
    # rot_der[Noalign, 1, 0, 1] = onerij[Noalign] - xij_[:, 1] * termY
    # rot_der[Noalign, 1, 0, 2] = -xij_[:, 1] * termZ
    #
    # rot_der[Noalign, 2, 0, 0] = -xij_[:, 2] * termX
    # rot_der[Noalign, 2, 0, 1] = -xij_[:, 2] * termY
    # rot_der[Noalign, 2, 0, 2] = onerij[Noalign] - xij_[:, 2] * termZ
    #
    # rot_der[Noalign, 0, 2, 2] = -xij_[:, 0] * onerxy - rot[Noalign, 2, 2] * termX
    # rot_der[Noalign, 1, 2, 2] = -xij_[:, 1] * onerxy - rot[Noalign, 2, 2] * termY
    # rot_der[Noalign, 2, 2, 2] = -rot[Noalign, 2, 2] * termZ
    #
    # rot_der[Noalign, 0, 1, 0] = -rot[Noalign, 1, 1] * rot[Noalign, 1, 0] * onerxy
    # rot_der[Noalign, 1, 1, 0] = -torch.square(rot[Noalign, 1, 1]) * onerxy
    # # # Sanity check because openmopac (and hence NEXMD) do this differently. I want to make sure our expressions give the same result
    # # tolerance = 1e-8
    # # assert torch.allclose(rot_der[Noalign,0,1,0],-rot_der[Noalign,1,0,0]*rab_over_rxy+rot[Noalign,0,1]*rot_der[Noalign,0,2,2]*rab_over_rxy_sq,atol=tolerance)
    # # assert torch.allclose(rot_der[Noalign,1,1,0],-rot_der[Noalign,1,0,1]*rab_over_rxy+rot[Noalign,0,1]*rot_der[Noalign,1,2,2]*rab_over_rxy_sq,atol=tolerance)
    # # assert torch.all(torch.abs(-rot_der[Noalign,1,0,2]*rab_over_rxy+rot[Noalign,0,1]*rot_der[Noalign,2,2,2]*rab_over_rxy_sq)<tolerance)
    #
    # rot_der[Noalign, 0, 1, 1] = torch.square(rot[Noalign, 1, 0]) * onerxy
    # rot_der[Noalign, 1, 1, 1] = rot[Noalign, 1, 1] * rot[Noalign, 1, 0] * onerxy
    # # # Sanity check because openmopac (and hence NEXMD) do this differently. I want to make sure our expressions give the same result
    # # tolerance = 1e-8
    # # mopacs = rot_der[Noalign,0,0,0]*rab_over_rxy-rot[Noalign,0,0]*rot_der[Noalign,0,2,2]*rab_over_rxy_sq
    # # mine = rot_der[Noalign,0,1,1]
    # # assert torch.allclose(mine,mopacs,atol=tolerance)
    # # assert torch.allclose(rot_der[Noalign,1,1,1],rot_der[Noalign,0,0,1]*rab_over_rxy-rot[Noalign,0,0]*rot_der[Noalign,1,2,2]*rab_over_rxy_sq,atol=tolerance)
    # # assert torch.all(torch.abs(rot_der[Noalign,0,0,2]*rab_over_rxy-rot[Noalign,0,0]*rot_der[Noalign,2,2,2]*rab_over_rxy_sq)<tolerance)
    #
    # rot_der[Noalign, 0, 2, 0] = xij_[:, 2] * rot_der[Noalign, 0, 0, 0] * rab_over_rxy + xij_[:, 0] * rot_der[
    #     Noalign, 2, 0, 0] * rab_over_rxy + xij_[:, 0] * xij_[:, 2] * rot_der[Noalign, 0, 2, 2] * rab_over_rxy_sq
    # rot_der[Noalign, 1, 2, 0] = -torch.prod(xij_, dim=1) * (onerxy + rab_over_rxy_sq * onerxy)
    # rot_der[Noalign, 2, 2, 0] = termX * rxy_over_rab
    #
    # rot_der[Noalign, 0, 2, 1] = rot_der[Noalign, 1, 2, 0]
    # rot_der[Noalign, 1, 2, 1] = xij_[:, 2] * rot_der[Noalign, 1, 0, 1] * rab_over_rxy + xij_[:, 1] * rot_der[
    #     Noalign, 2, 0, 1] * rab_over_rxy + xij_[:, 1] * xij_[:, 2] * rot_der[Noalign, 1, 2, 2] * rab_over_rxy_sq
    # rot_der[Noalign, 2, 2, 1] = termY * rxy_over_rab
    #
    # rot[Zalign, 0, 2] = torch.sign(-xij[Zalign, 2])
    # rot[Zalign, 1, 1] = rot[Zalign, 0, 2]
    # rot[Zalign, 2, 0] = 1.0
    # # rot_der[Zalign, 0, 0, 0] = onerij[Zalign]
    # # rot_der[Zalign, 0, 2, 2] = -onerij[Zalign]
    # # rot_der[Zalign, 1, 0, 1] = onerij[Zalign]
    # # rot_der[Zalign, 1, 1, 2] = -rot[Zalign, 0, 2] * onerij[Zalign]
    #
    # # rot[Xalign, 0, 0] = torch.sign(-xij[Xalign, 0])
    # # rot[Xalign, 1, 1] = rot[Xalign, 0, 0]
    # # rot[Xalign, 2, 2] = 1.0
    # # rot_der[Xalign, 1, 0, 1] = onerij[Xalign]
    # # rot_der[Xalign, 1, 1, 0] = -onerij[Xalign]
    # # rot_der[Xalign, 2, 0, 2] = onerij[Xalign]
    # # rot_der[Xalign, 2, 2, 0] = -rot[Xalign, 0, 0] * onerij[Xalign]
    #
    # # rot[Yalign, 0, 1] = torch.sign(-xij[Yalign, 1])
    # # rot[Yalign, 1, 0] = -rot[Yalign, 0, 1]
    # # rot[Yalign, 2, 2] = 1.0
    # # rot_der[Yalign, 0, 0, 0] = onerij[Yalign]
    # # rot_der[Yalign, 0, 1, 1] = onerij[Yalign]
    # # rot_der[Yalign, 2, 0, 2] = onerij[Yalign]
    # # rot_der[Yalign, 2, 2, 1] = -rot[Yalign, 0, 1] * onerij[Yalign]

    # v = -xij
    # v.requires_grad_()
    # rot = rotate_with_quaternion(v)
    # rot_der = torch.zeros(r0.shape[0], 3, 3, 3, device=device, dtype=dtype)
    # for j in range(3):
    #     # build a batched “direction” d with 1’s in coordinate j
    #     d = torch.zeros_like(v)
    #     d[:, j] = 1.0
    #
    #     # jvp returns (r, dr) where dr = (∂r/∂v) ⋅ d, shape = (n,3,3)
    #     _, dr = jvp(rotate_with_quaternion, (v,), (d,), create_graph=True)
    #
    #     # dr[i,a,b] == ∂r[i,a,b] / ∂v[i,j]
    #     rot_der[:, j, :, :] = dr

    v = -xij
    rot, rot_der = rotate_with_quaternion(v, calculate_gradient=True)

    # rot_der is dR/dv. But I want dR/dx (x are components of -Xij). We have, dR_ij/dx_a = \sum_b dR_ij/dv_b * dv_b/dx_a
    # We first need the Jacobian J, where J_ba = dv_b/dx_a.
    # v is -Xij/norm(Xij)
    # J_{ba} = ∂u_b/∂v_a = (δ_ba - v_b v_a/||v||) / ||Xij||
    I = torch.eye(3, device=device, dtype=dtype)  # (3,3)
    inv_r0_a0 = (r0 * a0).reciprocal().unsqueeze(-1).unsqueeze(-1)
    J_uv = (I - v.unsqueeze(-1) * v.unsqueeze(-2)) * inv_r0_a0
    rot_der = torch.einsum("nbij,nba->naij", rot_der, J_uv)  # (n,4,3)

    # print(f"rot mat orthogonality: {torch.sum(rot@rot.transpose(1,2))}, with 3*natoms is {rot.shape[0]*3}")
    #
    # print(f"rot der check zero: {torch.sum(rot_der[:,0,...]@rot.transpose(1,2)+rot@rot_der[:,0,...].transpose(1,2))}")
    # print(f"rot der check zero: {torch.sum(rot_der[:,1,...]@rot.transpose(1,2)+rot@rot_der[:,1,...].transpose(1,2))}")
    # print(f"rot der check zero: {torch.sum(rot_der[:,2,...]@rot.transpose(1,2)+rot@rot_der[:,2,...].transpose(1,2))}")

    rotXH = rot[XH, ...]
    rot = rot[XX, ...]
    rot_derXH = rot_der[XH, ...]
    rot_der = rot_der[XX, ...]

    if _VECTORIZED_SP_DERIVATIVES:
        wXH_x, w_x = _rotate_sp_derivatives(rotXH, rot, rot_derXH, rot_der, riXH, ri, riXH_x, ri_x)

        w_x_final[HH, :, 0, 0] = riHH_x
        w_x_final[XH, :, :, 0] = wXH_x
        w_x_final[XX, ...] = w_x
        return

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
                            w_x[..., idx] = (
                                ri_x[..., 4] * rot[:, None, 0, m] + ri[:, None, 4] * rot_der[:, :, 0, m]
                            )
                        else:
                            # (ss|pp)
                            w_x[..., idx] = (
                                ri_x[..., 10] * (rot[:, 0, m] * rot[:, 0, n]).unsqueeze(1)
                                + ri[:, None, 10]
                                * (
                                    rot_der[:, :, 0, m] * rot[:, None, 0, n]
                                    + rot[:, None, 0, m] * rot_der[:, :, 0, n]
                                )
                                + ri_x[..., 11]
                                * (rot[:, 1, m] * rot[:, 1, n] + rot[:, 2, m] * rot[:, 2, n]).unsqueeze(1)
                                + ri[:, None, 11]
                                * (
                                    rot_der[:, :, 1, m] * rot[:, None, 1, n]
                                    + rot_der[:, :, 2, m] * rot[:, None, 2, n]
                                    + rot[:, None, 1, m] * rot_der[:, :, 1, n]
                                    + rot[:, None, 2, m] * rot_der[:, :, 2, n]
                                )
                            )

                    elif ll == 0:
                        if mm == 0:
                            # (ps|ss)
                            w_x[..., idx] = (
                                ri_x[..., 1] * rot[:, None, 0, k] + ri[:, None, 1] * rot_der[:, :, 0, k]
                            )
                            wXH_x[..., idxXH] = (
                                riXH_x[..., 1] * rotXH[:, None, 0, k]
                                + riXH[:, None, 1] * rot_derXH[:, :, 0, k]
                            )
                            idxXH = idxXH + 1
                        elif nn == 0:
                            # (ps|ps)
                            w_x[..., idx] = (
                                ri_x[..., 5] * (rot[:, 0, k] * rot[:, 0, m]).unsqueeze(1)
                                + ri[:, None, 5]
                                * (
                                    rot_der[:, :, 0, k] * rot[:, None, 0, m]
                                    + rot[:, None, 0, k] * rot_der[:, :, 0, m]
                                )
                                + ri_x[..., 6]
                                * (rot[:, 1, k] * rot[:, 1, m] + rot[:, 2, k] * rot[:, 2, m]).unsqueeze(1)
                                + ri[:, None, 6]
                                * (
                                    rot_der[:, :, 1, k] * rot[:, None, 1, m]
                                    + rot_der[:, :, 2, k] * rot[:, None, 2, m]
                                    + rot[:, None, 1, k] * rot_der[:, :, 1, m]
                                    + rot[:, None, 2, k] * rot_der[:, :, 2, m]
                                )
                            )
                        else:
                            # (ps|pp)
                            w_x[..., idx] = (
                                ri_x[..., 12] * (rot[:, 0, k] * rot[:, 0, n] * rot[:, 0, m]).unsqueeze(1)
                                + ri[:, None, 12]
                                * (
                                    rot_der[:, :, 0, k] * (rot[:, 0, n] * rot[:, 0, m]).unsqueeze(1)
                                    + rot_der[:, :, 0, n] * (rot[:, 0, k] * rot[:, 0, m]).unsqueeze(1)
                                    + rot_der[:, :, 0, m] * (rot[:, 0, n] * rot[:, 0, k]).unsqueeze(1)
                                )
                                + ri_x[..., 13]
                                * (
                                    (rot[:, 1, m] * rot[:, 1, n] + rot[:, 2, m] * rot[:, 2, n]) * rot[:, 0, k]
                                ).unsqueeze(1)
                                + ri[:, None, 13]
                                * (
                                    (rot[:, 1, m] * rot[:, 1, n] + rot[:, 2, m] * rot[:, 2, n]).unsqueeze(1)
                                    * rot_der[:, :, 0, k]
                                    + (
                                        rot_der[:, :, 1, m] * rot[:, None, 1, n]
                                        + rot[:, None, 1, m] * rot_der[:, :, 1, n]
                                        + rot_der[:, :, 2, m] * rot[:, None, 2, n]
                                        + rot[:, None, 2, m] * rot_der[:, :, 2, n]
                                    )
                                    * rot[:, None, 0, k]
                                )
                                + ri_x[..., 14]
                                * (
                                    rot[:, 1, k] * (rot[:, 1, n] * rot[:, 0, m] + rot[:, 1, m] * rot[:, 0, n])
                                    + rot[:, 2, k]
                                    * (rot[:, 2, m] * rot[:, 0, n] + rot[:, 2, n] * rot[:, 0, m])
                                ).unsqueeze(1)
                                + ri[:, None, 14]
                                * (
                                    rot_der[:, :, 1, k]
                                    * (rot[:, 1, n] * rot[:, 0, m] + rot[:, 1, m] * rot[:, 0, n]).unsqueeze(1)
                                    + rot[:, None, 1, k]
                                    * (
                                        rot_der[:, :, 1, m] * rot[:, None, 0, n]
                                        + rot[:, None, 1, m] * rot_der[:, :, 0, n]
                                        + rot_der[:, :, 1, n] * rot[:, None, 0, m]
                                        + rot[:, None, 1, n] * rot_der[:, :, 0, m]
                                    )
                                    + rot_der[:, :, 2, k]
                                    * (rot[:, 2, n] * rot[:, 0, m] + rot[:, 2, m] * rot[:, 0, n]).unsqueeze(1)
                                    + rot[:, None, 2, k]
                                    * (
                                        rot_der[:, :, 2, n] * rot[:, None, 0, m]
                                        + rot[:, None, 2, n] * rot_der[:, :, 0, m]
                                        + rot_der[:, :, 2, m] * rot[:, None, 0, n]
                                        + rot[:, None, 2, m] * rot_der[:, :, 0, n]
                                    )
                                )
                            )
                            pass
                    else:
                        if mm == 0:
                            # (pp|ss)
                            w_x[..., idx] = (
                                ri_x[..., 2] * (rot[:, 0, k] * rot[:, 0, l]).unsqueeze(1)
                                + ri[:, None, 2]
                                * (
                                    rot_der[:, :, 0, k] * rot[:, None, 0, l]
                                    + rot[:, None, 0, k] * rot_der[:, :, 0, l]
                                )
                                + ri_x[..., 3]
                                * (rot[:, 1, k] * rot[:, 1, l] + rot[:, 2, k] * rot[:, 2, l]).unsqueeze(1)
                                + ri[:, None, 3]
                                * (
                                    rot_der[:, :, 1, k] * rot[:, None, 1, l]
                                    + rot_der[:, :, 2, k] * rot[:, None, 2, l]
                                    + rot[:, None, 1, k] * rot_der[:, :, 1, l]
                                    + rot[:, None, 2, k] * rot_der[:, :, 2, l]
                                )
                            )
                            wXH_x[..., idxXH] = (
                                riXH_x[..., 2] * (rotXH[:, 0, k] * rotXH[:, 0, l]).unsqueeze(1)
                                + riXH[:, None, 2]
                                * (
                                    rot_derXH[:, :, 0, k] * rotXH[:, None, 0, l]
                                    + rotXH[:, None, 0, k] * rot_derXH[:, :, 0, l]
                                )
                                + riXH_x[..., 3]
                                * (
                                    rotXH[:, 1, k] * rotXH[:, 1, l] + rotXH[:, 2, k] * rotXH[:, 2, l]
                                ).unsqueeze(1)
                                + riXH[:, None, 3]
                                * (
                                    rot_derXH[:, :, 1, k] * rotXH[:, None, 1, l]
                                    + rot_derXH[:, :, 2, k] * rotXH[:, None, 2, l]
                                    + rotXH[:, None, 1, k] * rot_derXH[:, :, 1, l]
                                    + rotXH[:, None, 2, k] * rot_derXH[:, :, 2, l]
                                )
                            )
                            idxXH = idxXH + 1
                        elif nn == 0:
                            # (pp|ps)
                            w_x[..., idx] = (
                                ri_x[..., 7] * (rot[:, 0, k] * rot[:, 0, l] * rot[:, 0, m]).unsqueeze(1)
                                + ri[:, None, 7]
                                * (
                                    rot_der[:, :, 0, k] * (rot[:, 0, l] * rot[:, 0, m]).unsqueeze(1)
                                    + rot_der[:, :, 0, l] * (rot[:, 0, k] * rot[:, 0, m]).unsqueeze(1)
                                    + rot_der[:, :, 0, m] * (rot[:, 0, l] * rot[:, 0, k]).unsqueeze(1)
                                )
                                + ri_x[..., 8]
                                * (
                                    (rot[:, 1, k] * rot[:, 1, l] + rot[:, 2, k] * rot[:, 2, l]) * rot[:, 0, m]
                                ).unsqueeze(1)
                                + ri[:, None, 8]
                                * (
                                    (rot[:, 1, k] * rot[:, 1, l] + rot[:, 2, k] * rot[:, 2, l]).unsqueeze(1)
                                    * rot_der[:, :, 0, m]
                                    + (
                                        rot_der[:, :, 1, k] * rot[:, None, 1, l]
                                        + rot[:, None, 1, k] * rot_der[:, :, 1, l]
                                        + rot_der[:, :, 2, k] * rot[:, None, 2, l]
                                        + rot[:, None, 2, k] * rot_der[:, :, 2, l]
                                    )
                                    * rot[:, None, 0, m]
                                )
                                + ri_x[..., 9]
                                * (
                                    rot[:, 0, k] * (rot[:, 1, l] * rot[:, 1, m] + rot[:, 2, l] * rot[:, 2, m])
                                    + rot[:, 0, l]
                                    * (rot[:, 1, k] * rot[:, 1, m] + rot[:, 2, k] * rot[:, 2, m])
                                ).unsqueeze(1)
                                + ri[:, None, 9]
                                * (
                                    rot_der[:, :, 0, k]
                                    * (rot[:, 1, l] * rot[:, 1, m] + rot[:, 2, l] * rot[:, 2, m]).unsqueeze(1)
                                    + rot[:, None, 0, k]
                                    * (
                                        rot_der[:, :, 1, l] * rot[:, None, 1, m]
                                        + rot[:, None, 2, l] * rot_der[:, :, 2, m]
                                        + rot_der[:, :, 1, m] * rot[:, None, 1, l]
                                        + rot[:, None, 2, m] * rot_der[:, :, 2, l]
                                    )
                                    + rot_der[:, :, 0, l]
                                    * (rot[:, 1, k] * rot[:, 1, m] + rot[:, 2, k] * rot[:, 2, m]).unsqueeze(1)
                                    + rot[:, None, 0, l]
                                    * (
                                        rot_der[:, :, 1, k] * rot[:, None, 1, m]
                                        + rot[:, None, 2, k] * rot_der[:, :, 2, m]
                                        + rot_der[:, :, 1, m] * rot[:, None, 1, k]
                                        + rot[:, None, 2, m] * rot_der[:, :, 2, k]
                                    )
                                )
                            )

                        else:
                            # (pp|pp)
                            w_x[..., idx] = (
                                ri_x[..., 16 - 1]
                                * (rot[:, 0, k] * rot[:, 0, l] * rot[:, 0, m] * rot[:, 0, n]).unsqueeze(1)
                                + ri[:, None, 16 - 1]
                                * (
                                    rot_der[:, :, 0, k]
                                    * (rot[:, 0, l] * rot[:, 0, m] * rot[:, 0, n]).unsqueeze(1)
                                    + rot_der[:, :, 0, l]
                                    * (rot[:, 0, k] * rot[:, 0, m] * rot[:, 0, n]).unsqueeze(1)
                                    + rot_der[:, :, 0, m]
                                    * (rot[:, 0, k] * rot[:, 0, l] * rot[:, 0, n]).unsqueeze(1)
                                    + (rot[:, 0, k] * rot[:, 0, l] * rot[:, 0, m]).unsqueeze(1)
                                    * rot_der[:, :, 0, n]
                                )
                                + ri_x[..., 17 - 1]
                                * (
                                    (rot[:, 1, k] * rot[:, 1, l] + rot[:, 2, k] * rot[:, 2, l])
                                    * rot[:, 0, m]
                                    * rot[:, 0, n]
                                ).unsqueeze(1)
                                + ri[:, None, 17 - 1]
                                * (
                                    (
                                        rot_der[:, :, 1, k] * rot[:, None, 1, l]
                                        + rot[:, None, 1, k] * rot_der[:, :, 1, l]
                                        + rot_der[:, :, 2, k] * rot[:, None, 2, l]
                                        + rot[:, None, 2, k] * rot_der[:, :, 2, l]
                                    )
                                    * (rot[:, 0, m] * rot[:, 0, n]).unsqueeze(1)
                                    + (rot[:, 1, k] * rot[:, 1, l] + rot[:, 2, k] * rot[:, 2, l]).unsqueeze(1)
                                    * (
                                        rot_der[:, :, 0, m] * rot[:, None, 0, n]
                                        + rot[:, None, 0, m] * rot_der[:, :, 0, n]
                                    )
                                )
                                + ri_x[..., 18 - 1]
                                * (
                                    rot[:, 0, k]
                                    * rot[:, 0, l]
                                    * (rot[:, 1, m] * rot[:, 1, n] + rot[:, 2, m] * rot[:, 2, n])
                                ).unsqueeze(1)
                                + ri[:, None, 18 - 1]
                                * (
                                    (
                                        rot_der[:, :, 0, k] * rot[:, None, 0, l]
                                        + rot[:, None, 0, k] * rot_der[:, :, 0, l]
                                    )
                                    * (rot[:, 1, m] * rot[:, 1, n] + rot[:, 2, m] * rot[:, 2, n]).unsqueeze(1)
                                    + (rot[:, 0, k] * rot[:, 0, l]).unsqueeze(1)
                                    * (
                                        rot_der[:, :, 1, m] * rot[:, None, 1, n]
                                        + rot[:, None, 1, m] * rot_der[:, :, 1, n]
                                        + rot_der[:, :, 2, m] * rot[:, None, 2, n]
                                        + rot[:, None, 2, m] * rot_der[:, :, 2, n]
                                    )
                                )
                            )
                            w_x[..., idx] += (
                                ri_x[..., 19 - 1]
                                * (
                                    rot[:, 1, k] * rot[:, 1, l] * rot[:, 1, m] * rot[:, 1, n]
                                    + rot[:, 2, k] * rot[:, 2, l] * rot[:, 2, m] * rot[:, 2, n]
                                ).unsqueeze(1)
                                + ri[:, None, 19 - 1]
                                * (
                                    rot_der[:, :, 1, k]
                                    * (rot[:, 1, l] * rot[:, 1, m] * rot[:, 1, n]).unsqueeze(1)
                                    + rot_der[:, :, 1, l]
                                    * (rot[:, 1, k] * rot[:, 1, m] * rot[:, 1, n]).unsqueeze(1)
                                    + rot_der[:, :, 1, m]
                                    * (rot[:, 1, k] * rot[:, 1, l] * rot[:, 1, n]).unsqueeze(1)
                                    + (rot[:, 1, k] * rot[:, 1, l] * rot[:, 1, m]).unsqueeze(1)
                                    * rot_der[:, :, 1, n]
                                    + rot_der[:, :, 2, k]
                                    * (rot[:, 2, l] * rot[:, 2, m] * rot[:, 2, n]).unsqueeze(1)
                                    + rot_der[:, :, 2, l]
                                    * (rot[:, 2, k] * rot[:, 2, m] * rot[:, 2, n]).unsqueeze(1)
                                    + rot_der[:, :, 2, m]
                                    * (rot[:, 2, k] * rot[:, 2, l] * rot[:, 2, n]).unsqueeze(1)
                                    + (rot[:, 2, k] * rot[:, 2, l] * rot[:, 2, m]).unsqueeze(1)
                                    * rot_der[:, :, 2, n]
                                )
                                + ri_x[..., 20 - 1]
                                * (
                                    rot[:, 0, k]
                                    * (
                                        rot[:, 0, m]
                                        * (rot[:, 1, l] * rot[:, 1, n] + rot[:, 2, l] * rot[:, 2, n])
                                        + rot[:, 0, n]
                                        * (rot[:, 1, l] * rot[:, 1, m] + rot[:, 2, l] * rot[:, 2, m])
                                    )
                                    + rot[:, 0, l]
                                    * (
                                        rot[:, 0, m]
                                        * (rot[:, 1, k] * rot[:, 1, n] + rot[:, 2, k] * rot[:, 2, n])
                                        + rot[:, 0, n]
                                        * (rot[:, 1, k] * rot[:, 1, m] + rot[:, 2, k] * rot[:, 2, m])
                                    )
                                ).unsqueeze(1)
                            )
                            #      TO AVOID COMPILER DIFFICULTIES THIS IS DIVIDED
                            temp1 = (
                                rot_der[:, :, 0, k]
                                * (
                                    rot[:, 0, m] * (rot[:, 1, l] * rot[:, 1, n] + rot[:, 2, l] * rot[:, 2, n])
                                    + rot[:, 0, n]
                                    * (rot[:, 1, l] * rot[:, 1, m] + rot[:, 2, l] * rot[:, 2, m])
                                ).unsqueeze(1)
                                + rot_der[:, :, 0, l]
                                * (
                                    rot[:, 0, m] * (rot[:, 1, k] * rot[:, 1, n] + rot[:, 2, k] * rot[:, 2, n])
                                    + rot[:, 0, n]
                                    * (rot[:, 1, k] * rot[:, 1, m] + rot[:, 2, k] * rot[:, 2, m])
                                ).unsqueeze(1)
                                + rot[:, None, 0, k]
                                * (
                                    rot_der[:, :, 0, m]
                                    * (rot[:, 1, l] * rot[:, 1, n] + rot[:, 2, l] * rot[:, 2, n]).unsqueeze(1)
                                    + rot_der[:, :, 0, n]
                                    * (rot[:, 1, l] * rot[:, 1, m] + rot[:, 2, l] * rot[:, 2, m]).unsqueeze(1)
                                )
                                + rot[:, None, 0, l]
                                * (
                                    rot_der[:, :, 0, m]
                                    * (rot[:, 1, k] * rot[:, 1, n] + rot[:, 2, k] * rot[:, 2, n]).unsqueeze(1)
                                    + rot_der[:, :, 0, n]
                                    * (rot[:, 1, k] * rot[:, 1, m] + rot[:, 2, k] * rot[:, 2, m]).unsqueeze(1)
                                )
                            )
                            temp2 = rot[:, None, 0, k] * (
                                rot[:, None, 0, m]
                                * (
                                    rot_der[:, :, 1, l] * rot[:, None, 1, n]
                                    + rot[:, None, 1, l] * rot_der[:, :, 1, n]
                                    + rot_der[:, :, 2, l] * rot[:, None, 2, n]
                                    + rot[:, None, 2, l] * rot_der[:, :, 2, n]
                                )
                                + rot[:, None, 0, n]
                                * (
                                    rot_der[:, :, 1, l] * rot[:, None, 1, m]
                                    + rot[:, None, 1, l] * rot_der[:, :, 1, m]
                                    + rot_der[:, :, 2, l] * rot[:, None, 2, m]
                                    + rot[:, None, 2, l] * rot_der[:, :, 2, m]
                                )
                            ) + rot[:, None, 0, l] * (
                                rot[:, None, 0, m]
                                * (
                                    rot_der[:, :, 1, k] * rot[:, None, 1, n]
                                    + rot[:, None, 1, k] * rot_der[:, :, 1, n]
                                    + rot_der[:, :, 2, k] * rot[:, None, 2, n]
                                    + rot[:, None, 2, k] * rot_der[:, :, 2, n]
                                )
                                + rot[:, None, 0, n]
                                * (
                                    rot_der[:, :, 1, k] * rot[:, None, 1, m]
                                    + rot[:, None, 1, k] * rot_der[:, :, 1, m]
                                    + rot_der[:, :, 2, k] * rot[:, None, 2, m]
                                    + rot[:, None, 2, k] * rot_der[:, :, 2, m]
                                )
                            )
                            w_x[..., idx] += ri[:, None, 20 - 1] * (temp1 + temp2)
                            w_x[..., idx] += ri_x[..., 21 - 1] * (
                                rot[:, 1, k] * rot[:, 1, l] * rot[:, 2, m] * rot[:, 2, n]
                                + rot[:, 2, k] * rot[:, 2, l] * rot[:, 1, m] * rot[:, 1, n]
                            ).unsqueeze(1) + ri[:, None, 21 - 1] * (
                                rot_der[:, :, 1, k]
                                * (rot[:, 1, l] * rot[:, 2, m] * rot[:, 2, n]).unsqueeze(1)
                                + rot_der[:, :, 1, l]
                                * (rot[:, 1, k] * rot[:, 2, m] * rot[:, 2, n]).unsqueeze(1)
                                + rot_der[:, :, 2, m]
                                * (rot[:, 1, k] * rot[:, 1, l] * rot[:, 2, n]).unsqueeze(1)
                                + (rot[:, 1, k] * rot[:, 1, l] * rot[:, 2, m]).unsqueeze(1)
                                * rot_der[:, :, 2, n]
                                + rot_der[:, :, 2, k]
                                * (rot[:, 2, l] * rot[:, 1, m] * rot[:, 1, n]).unsqueeze(1)
                                + rot_der[:, :, 2, l]
                                * (rot[:, 2, k] * rot[:, 1, m] * rot[:, 1, n]).unsqueeze(1)
                                + rot_der[:, :, 1, m]
                                * (rot[:, 2, k] * rot[:, 2, l] * rot[:, 1, n]).unsqueeze(1)
                                + (rot[:, 2, k] * rot[:, 2, l] * rot[:, 1, m]).unsqueeze(1)
                                * rot_der[:, :, 1, n]
                            )
                            w_x[..., idx] += ri_x[..., 22 - 1] * (
                                (rot[:, 1, k] * rot[:, 2, l] + rot[:, 2, k] * rot[:, 1, l])
                                * (rot[:, 1, m] * rot[:, 2, n] + rot[:, 2, m] * rot[:, 1, n])
                            ).unsqueeze(1) + ri[:, None, 22 - 1] * (
                                (
                                    rot_der[:, :, 1, k] * rot[:, None, 2, l]
                                    + rot[:, None, 1, k] * rot_der[:, :, 2, l]
                                    + rot_der[:, :, 2, k] * rot[:, None, 1, l]
                                    + rot[:, None, 2, k] * rot_der[:, :, 1, l]
                                )
                                * (rot[:, 1, m] * rot[:, 2, n] + rot[:, 2, m] * rot[:, 1, n]).unsqueeze(1)
                                + (rot[:, 1, k] * rot[:, 2, l] + rot[:, 2, k] * rot[:, 1, l]).unsqueeze(1)
                                * (
                                    rot_der[:, :, 1, m] * rot[:, None, 2, n]
                                    + rot[:, None, 1, m] * rot_der[:, :, 2, n]
                                    + rot_der[:, :, 2, m] * rot[:, None, 1, n]
                                    + rot[:, None, 2, m] * rot_der[:, :, 1, n]
                                )
                            )

    w_x_final[HH, :, 0, 0] = riHH_x
    w_x_final[XH, :, :, 0] = wXH_x
    w_x_final[XX, ...] = w_x.reshape(ri.shape[0], 3, 10, 10)


def _radial_rotation_derivative(local_at, rotate, Xij):
    """
    Derivative with respect to the position of atom i of quantities ``q = rotate(L(r), direction)``, where
    ``L`` is a dict of local-frame tensors that depend on the pair distance only and ``rotate`` is linear in
    ``L`` and cheap. Moving atom i changes both the distance and the direction, so

        dq/dx_c = [q(dir+, L) - q(dir-, L)] / 2 delta  -  (dr/dX_c) * q(dir, dL/dr)

    The expensive ``L`` is evaluated only at ``r +- eps`` (once, shared by the three components); the
    displaced geometries only enter through ``rotate``. This equals a finite difference of the fully rebuilt
    quantity at displaced geometries to O(delta^2).

    Parameters
    ----------
    local_at(r) : callable
        ``r`` (bohr), shape (m,) -> dict of local tensors. The pair axis of the result is either all m pairs
        or a subset chosen by pair class (the same subset for every half of a [plus; minus] batch).
    rotate(local, units) : callable
        ``units`` are pair unit vectors (m, 3); returns a dict of tensors with a leading pair axis of size m.
        Entries that do not depend on the direction may simply be returned as they are.
    Xij : (npairs, 3) displacement of atom j from atom i, in Angstrom.

    Returns a dict of tensors shaped (npairs, 3, ...).
    """
    npairs = Xij.shape[0]
    norm = torch.norm(Xij, dim=1)
    unit = Xij / norm.unsqueeze(1)
    dr_dX = Xij / (norm * a0).unsqueeze(1)  # d r / d X_c, with r in bohr and X in Angstrom
    eps = delta / a0

    local = local_at(torch.cat([norm / a0 + eps, norm / a0 - eps]))
    mid, slope = {}, {}
    for key, value in local.items():
        half = value.shape[0] // 2
        mid[key] = 0.5 * (value[:half] + value[half:])
        slope[key] = (value[:half] - value[half:]) / (2.0 * eps)
    mid_twice = {key: torch.cat([value, value]) for key, value in mid.items()}

    slope_rotated = rotate(slope, unit)

    results = {}
    for c in range(3):
        step = torch.zeros_like(Xij)
        step[:, c] = delta
        # Xij = Xj - Xi, so moving atom i by +delta means subtracting delta from Xij.
        displaced = torch.cat([Xij - step, Xij + step])
        rotated = rotate(mid_twice, displaced / torch.norm(displaced, dim=1, keepdim=True))
        g = dr_dX[:, c]
        for key, value in rotated.items():
            radial = g.view(-1, *([1] * (value.dim() - 1))) * slope_rotated[key]
            results.setdefault(key, [None] * 3)[c] = _central_diff(value, npairs) - radial
    return {key: torch.stack(values, dim=1) for key, values in results.items()}


def _omx_pair_derivatives(molecule, Xij, ni, nj, idxi, idxj, method, with_pair=True, with_overlap=False):
    """
    Central-difference derivatives of the OMx pair terms with respect to the position of atom i.

    Returns a dict of tensors shaped (npairs, 3, ...): ``B`` (resonance blocks) and, with ``with_pair``,
    ``w``, ``e1b``, ``e2a``, ``fko``, ``core_semi``; with ``with_overlap`` also ``S`` (overlap blocks).
    See ``_radial_rotation_derivative``.
    """

    def local_at(r):
        radial = build_omx_radial_context(
            molecule,
            method,
            idxi.repeat(2),
            idxj.repeat(2),
            ni.repeat(2),
            nj.repeat(2),
            r,
            with_pair=with_pair,
        )
        local = {"s_local": radial["s_local"], "t_local": radial["t_local"]}
        if with_pair:
            local.update(radial["local"])
        return local

    def rotate(local, units):
        directions = pair_directions(units)
        out = {"B": diatom_resonance_matrix_OM1(units, local["t_local"], directions["direction"])}
        if with_overlap:
            out["S"] = diatom_overlap_matrix_OM1(
                units, None, directions["direction"], None, None, local["s_local"]
            )
        if with_pair:
            out.update(omx_rotate_local_terms(local, directions["rot"], directions["rot_t"]))
            out["fko"] = local["fko"]  # radial only
            out["core_semi"] = local["core_semi"]
        return out

    return _radial_rotation_derivative(local_at, rotate, Xij)


def _central_diff(t, npairs):
    """Central difference of a [plus; minus] batched tensor."""
    return (t[:npairs] - t[npairs:]) * (1.0 / (2.0 * delta))


def _build_omx_ortho_fd_cache(molecule, Xij, ni, nj, idxi, idxj, method, dtype, device):
    if method not in {"OM2", "OM3"}:
        return None

    # Only OM2 needs the two-center terms (core_semi); OM3 needs just overlap and resonance.
    d = _omx_pair_derivatives(
        molecule, Xij, ni, nj, idxi, idxj, method, with_pair=method == "OM2", with_overlap=True
    )
    return {
        "S_x": d["S"],
        "B_x": d["B"],
        "pair_core_semi_x": d["core_semi"] if method == "OM2" else None,
        **build_omx_ortho_cache(molecule),
    }


def omx_fd(molecule, overlap_KAB_x, w_x, Xij, ni, nj, idxi, idxj, method, P0=None, return_ortho_cache=False):
    need_ortho_terms = method in {"OM2", "OM3"} and (torch.is_tensor(P0) or return_ortho_cache)
    need_threebody_grad = method in {"OM2", "OM3"} and torch.is_tensor(P0)
    ortho_cache = build_omx_ortho_cache(molecule) if need_ortho_terms else None

    d = _omx_pair_derivatives(
        molecule, Xij, ni, nj, idxi, idxj, method, with_pair=True, with_overlap=need_ortho_terms
    )
    B_x = overlap_KAB_x
    B_x.copy_(d["B"])
    w_x.copy_(d["w"])
    e1b_x, e2a_x, fko_x = d["e1b"], d["e2a"], d["fko"]
    S_x = d.get("S")
    pair_core_semi_x = d["core_semi"] if method == "OM2" and need_ortho_terms else None

    omx_orthogonalization_grad = None
    if need_threebody_grad:
        omx_orthogonalization_grad = omx_threebody_ortho_grad(
            molecule, P0, S_x, B_x, pair_core_semi_x, ortho_cache=ortho_cache, unrestricted=P0.dim() == 4
        )
        # Legacy callers expect a single density-gradient slice here.
        omx_orthogonalization_grad = omx_orthogonalization_grad[:, 0]
    if need_ortho_terms:
        ortho_cache = {"S_x": S_x, "B_x": B_x.clone(), "pair_core_semi_x": pair_core_semi_x, **ortho_cache}

    # Hcore derivative needs to have upper and lower triangle contribution, multiply by 2.0, since Hcore is symmetric and will be contracted with a symmetric P0.
    overlap_KAB_x *= 2.0

    return e1b_x, e2a_x, fko_x, omx_orthogonalization_grad, ortho_cache


def _dense_from_pairs(M, N, mol, i, j, blocks):
    if blocks.dim() == 3:
        W = blocks.new_zeros((M, 4 * N, 4 * N))
        W5 = W.reshape(M, N, 4, N, 4).permute(0, 1, 3, 2, 4)
        W5[mol, i, j] = blocks
        return W
    K = blocks.shape[1]
    W = blocks.new_zeros((M, K, N, N, 4, 4))
    W[mol, :, i, j] = blocks
    return blocks_to_dense(W)


def betor_grad_dense(
    P_blocks, S5, B5, S_x, B_x, molecule, gval1, COR=None, gval2=None, pair_core_semi_x=None, vel_eff=None
):
    M, N = S5.shape[:2]
    dtype, device = S5.dtype, S5.device

    mask = molecule.mask
    mol = molecule.pair_molid
    a = (mask // N) % N
    b = mask % N
    P = mask.numel()

    if P_blocks.dim() == 3:
        P_blocks = P_blocks.unsqueeze(1)
    K = P_blocks.shape[1]
    W = 2.0 * P_blocks[mask]

    S2 = blocks_to_dense(S5)
    B2 = blocks_to_dense(B5)

    ft1 = 0.25 * (gval1[0] + gval1[1])
    G1 = -ft1[:, None, None, None] * W

    W1 = _dense_from_pairs(M, N, mol, a, b, G1)
    W1 = W1 + W1.transpose(-1, -2)

    adjS2 = torch.matmul(W1, B2[:, None])
    adjB2 = torch.matmul(W1, S2[:, None])

    if COR is not None:
        ft2 = 0.0625 * (gval2[0] + gval2[1])
        G2 = ft2[:, None, None, None] * W

        W2 = _dense_from_pairs(M, N, mol, a, b, G2)
        W2 = W2 + W2.transpose(-1, -2)

        F = omx_cor_difference(COR)

        D2 = blocks_to_dense(S5 * F)

        adjD2 = torch.matmul(W2, S2[:, None])
        adjS2 = adjS2 + torch.matmul(W2, D2[:, None])

        adjD = dense_to_blocks(adjD2, N)
        adjS = dense_to_blocks(adjS2, N) + adjD * F[:, None]
        adjB = dense_to_blocks(adjB2, N)

        adjF = adjD * S5[:, None]
        adjCOR = adjF.sum(-1) - adjF.sum(-2).transpose(-3, -2)
    else:
        adjS = dense_to_blocks(adjS2, N)
        adjB = dense_to_blocks(adjB2, N)
        adjCOR = None

    S_x_t = -S_x.transpose(-1, -2)
    B_x_t = -B_x.transpose(-1, -2)
    adjS_ab = adjS[mol, :, a, b]
    adjB_ab = adjB[mol, :, a, b]
    adjS_ba = adjS[mol, :, b, a]
    adjB_ba = adjB[mol, :, b, a]

    q_ab = (adjS_ab[:, :, None] * S_x[:, None]).sum((-1, -2)) + (adjB_ab[:, :, None] * B_x[:, None]).sum(
        (-1, -2)
    )
    q_ba = (adjS_ba[:, :, None] * S_x_t[:, None]).sum((-1, -2)) + (adjB_ba[:, :, None] * B_x_t[:, None]).sum(
        (-1, -2)
    )

    if COR is not None:
        Cx0 = torch.zeros((P, 3, 4), dtype=dtype, device=device)
        Cx1 = torch.zeros_like(Cx0)

        Cx0[:, :, 0] = pair_core_semi_x[:, :, 0, 0]
        Cx1[:, :, 0] = pair_core_semi_x[:, :, 0, 1]

        Cx0p = (pair_core_semi_x[:, :, 2, 0] + 2 * pair_core_semi_x[:, :, 3, 0]) / 3
        Cx1p = (pair_core_semi_x[:, :, 2, 1] + 2 * pair_core_semi_x[:, :, 3, 1]) / 3

        ha = (molecule.species[mol, a] > 1).to(dtype)[:, None]
        hb = (molecule.species[mol, b] > 1).to(dtype)[:, None]

        Cx0[:, :, 1:] = Cx0p[:, :, None] * ha[:, :, None]
        Cx1[:, :, 1:] = Cx1p[:, :, None] * hb[:, :, None]

        adjCOR_ab = adjCOR[mol, :, a, b]
        adjCOR_ba = adjCOR[mol, :, b, a]
        q_ab = q_ab + (adjCOR_ab[:, :, None, :] * Cx0[:, None, :, :]).sum(-1)
        q_ba = q_ba + (adjCOR_ba[:, :, None, :] * (-Cx1[:, None, :, :])).sum(-1)

    if vel_eff is not None:
        # Fast path for TD-NAC: the caller only needs grad · vel, so skip the
        # dense [M, K, N, 3] gradient materialization and contract per-pair.
        v_ab = vel_eff[mol, a] - vel_eff[mol, b]
        pair_val = ((q_ab - q_ba) * v_ab[:, None, :]).sum(dim=-1)
        grad = torch.zeros((M, K), dtype=dtype, device=device)
        grad.index_add_(0, mol, pair_val)
        return grad

    grad = torch.zeros((M * N, K, 3), dtype=dtype, device=device)
    fa = mol * N + a
    fb = mol * N + b

    grad.index_add_(0, fa, q_ab)
    grad.index_add_(0, fb, -q_ab)

    grad.index_add_(0, fb, q_ba)
    grad.index_add_(0, fa, -q_ba)

    return grad.reshape(M, N, K, 3).permute(0, 2, 1, 3)

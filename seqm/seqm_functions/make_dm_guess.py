# This file is required for SEDACS
import torch

from seqm.basics import Pack_Parameters

from .build_two_elec_one_center_int_D import calc_integral  # , calc_integral_os
from .diag import (
    DEGEN_EIGENSOLVER,
    _apply_padding_eigen_shifts,
    _zero_padding_eigenvalues,
    construct_P,
    degen_symeig,
    pytorch_symeig,
)  # sym_eig_trunc, sym_eig_trunc1, pseudo_diag
from .fock_u_batch import fock_u_batch
from .hcore import hcore
from .omx_utils import get_orbital_zetas
from .pack import pack, unpack
from .packd import packd, unpackd

CHECK_DEGENERACY = False


def _initial_density_guess(molecule, tore, nmol):
    orb_dim = 9 if molecule.method == "PM6" else 4
    P0 = torch.zeros(
        molecule.nmol * molecule.molsize * molecule.molsize,
        orb_dim,
        orb_dim,
        dtype=molecule.coordinates.dtype,
        device=molecule.coordinates.device,
    )
    heavy = molecule.Z > 1
    P0[molecule.maskd[heavy], 0, 0] = tore[molecule.Z[heavy]] / 4.0
    P0[molecule.maskd, 1, 1] = P0[molecule.maskd, 0, 0]
    P0[molecule.maskd, 2, 2] = P0[molecule.maskd, 0, 0]
    P0[molecule.maskd, 3, 3] = P0[molecule.maskd, 0, 0]
    P0[molecule.maskd[molecule.Z == 1], 0, 0] = 1.0
    return (
        P0.reshape(nmol, molecule.molsize, molecule.molsize, orb_dim, orb_dim)
        .transpose(2, 3)
        .reshape(nmol, orb_dim * molecule.molsize, orb_dim * molecule.molsize)
    )


def make_dm_guess(
    molecule,
    seqm_parameters,
    mix_homo_lumo=False,
    mix_coeff=0.4,
    learned_parameters=dict(),
    overwrite_existing_dm=False,
    assignDM=True,
):
    sym_eigh = degen_symeig.apply if DEGEN_EIGENSOLVER else pytorch_symeig
    packpar = Pack_Parameters(seqm_parameters).to(molecule.coordinates.device)

    if callable(learned_parameters):
        adict = learned_parameters(molecule.species, molecule.coordinates)
        parameters, alp, chi = packpar(molecule.Z, learned_params=adict)
    else:
        parameters, alp, chi = packpar(molecule.Z, learned_params=learned_parameters)

    if molecule.method == "PM6":
        zetas = parameters["zeta_s"]
        zetap = parameters["zeta_p"]
        zetad = parameters["zeta_d"]
        zs = parameters["s_orb_exp_tail"]
        zp = parameters["p_orb_exp_tail"]
        zd = parameters["d_orb_exp_tail"]
        gss = parameters["g_ss"]
        gsp = parameters["g_sp"]
        gpp = parameters["g_pp"]
        gp2 = parameters["g_p2"]
        hsp = parameters["h_sp"]
        F0SD = parameters["F0SD"]
        G2SD = parameters["G2SD"]
    else:
        zetas, zetap = get_orbital_zetas(parameters, molecule.method)
        zetad = torch.zeros_like(zetas)
        zs = torch.zeros_like(zetas)
        zp = torch.zeros_like(zetas)
        zd = torch.zeros_like(zetas)
        gss = parameters["g_ss"]
        gsp = parameters["g_sp"]
        gpp = parameters["g_pp"]
        gp2 = parameters["g_p2"]
        hsp = parameters["h_sp"]
        F0SD = torch.zeros_like(parameters["U_ss"])
        G2SD = torch.zeros_like(parameters["U_ss"])

    nmol = molecule.nHeavy.shape[0]
    tore = molecule.const.tore

    if not torch.is_tensor(molecule.dm) or overwrite_existing_dm:
        # print('Reinitializing DM')
        P = _initial_density_guess(molecule, tore, nmol)

        if molecule.nocc.dim() == 2:
            P = torch.stack((0.5 * P, 0.5 * P), dim=1)

        if assignDM:
            molecule.dm = P

    if molecule.method == "PM6":
        if molecule.nocc.dim() == 2:  # open shell
            raise RuntimeError("Openshell PM6 not tested")
        else:
            W = calc_integral(
                zs,
                zp,
                zd,
                molecule.Z,
                nmol * molecule.molsize * molecule.molsize,
                molecule.maskd,
                P,
                F0SD,
                G2SD,
            )
            # W_exch = torch.tensor([0], device=molecule.nocc.device)
    else:
        W = torch.tensor([0], device=molecule.nocc.device)
        # W_exch = torch.tensor([0], device=molecule.nocc.device)

    if molecule.nocc.dim() == 2:
        P = molecule.dm
        if mix_homo_lumo:
            M, w, rho0xi, rho0xj, _, _ = hcore(molecule)
            if molecule.method == "PM6":
                x = fock_u_batch(
                    nmol,
                    molecule.molsize,
                    P,
                    M,
                    molecule.maskd,
                    molecule.mask,
                    molecule.idxi,
                    molecule.idxj,
                    w,
                    W,
                    gss,
                    gpp,
                    gsp,
                    gp2,
                    hsp,
                    molecule.method,
                    zetas,
                    zetap,
                    zetad,
                    molecule.Z,
                    F0SD,
                    G2SD,
                )

                # modified sym_eig_trunc below:
                dtype = x.dtype
                device = x.device

                nSuperHeavy = molecule.nSuperHeavy.repeat_interleave(2)
                nheavyatom = molecule.nHeavy.repeat_interleave(2)
                nH = molecule.nHydro.repeat_interleave(2)
                nocc = molecule.nocc.flatten()
                x_orig_shape = x.size()
                x0 = packd(x, nSuperHeavy, nheavyatom, nH)

                size = x0.shape[1]
                norb = nheavyatom * 4 + nH + nSuperHeavy * 9
                has_padding = _apply_padding_eigen_shifts(x0, norb)
                try:
                    e0, v = sym_eigh(x0)
                except:
                    if torch.isnan(x0).any():
                        print("isnan(x0) #1 in DM guess", x0)
                    e0, v = sym_eigh(x0)
                nmol = x0.shape[0]
                e = torch.zeros((nmol, x.shape[-1]), dtype=dtype, device=device)
                e[..., :size] = e0
                e = _zero_padding_eigenvalues(e, norb, has_padding)

                # $$$ the code below can and SHOULD be optimized. Too many reshapes

                e = e.reshape(x_orig_shape[0:3])
                v = v.reshape(int(v.shape[0] / 2), 2, v.shape[1], v.shape[2])

                v_lumo = v[:, 0].gather(
                    2, molecule.nocc[:, 0].unsqueeze(0).unsqueeze(0).T.repeat(1, v.shape[-1], 1)
                )
                v_homo = v[:, 0].gather(
                    2, molecule.nocc[:, 0].unsqueeze(0).unsqueeze(0).T.repeat(1, v.shape[-1], 1) - 1
                )

                mix_coeff = torch.tensor([mix_coeff], device=device)

                v_a_homo = (1 - mix_coeff) * v_homo + (mix_coeff) * v_lumo
                # v_a_lumo = -(mix_coeff)*v_homo + (1-mix_coeff)*v_lumo

                # v_b_homo = (1-mix_coeff)*v_homo - torch.sin(mix_coeff)*v_lumo
                # v_b_lumo =  (mix_coeff)*v_homo + (1-mix_coeff)*v_lumo

                v[:, 0].scatter_(
                    2, molecule.nocc[:, 0].unsqueeze(0).unsqueeze(0).T.repeat(1, v.shape[-1], 1) - 1, v_a_homo
                )
                # v[:,0].scatter_(2, molecule.nocc[:,0].unsqueeze(0).unsqueeze(0).T.repeat(1,v.shape[-1],1), v_a_lumo)

                # v[:,1].scatter_(2, molecule.nocc[:,1].unsqueeze(0).unsqueeze(0).T.repeat(1,v.shape[-1],1)-1, v_b_homo)
                # v[:,1].scatter_(2, molecule.nocc[:,1].unsqueeze(0).unsqueeze(0).T.repeat(1,v.shape[-1],1), v_b_lumo)

                v = v.reshape(int(v.shape[0] * 2), v.shape[2], v.shape[3])

                if CHECK_DEGENERACY:
                    t = torch.stack(list(map(lambda a, b, n: construct_P(a, b, n), e, v, nocc)))
                else:
                    t = 2.0 * torch.stack(
                        list(map(lambda a, n: torch.matmul(a[:, :n], a[:, :n].transpose(0, 1)), v, nocc))
                    )

                P = unpackd(t, nSuperHeavy, nheavyatom, nH, x.shape[-1])

                v = v.reshape(int(v.shape[0] / 2), 2, v.shape[1], v.shape[2])
                P = P.reshape(x_orig_shape)
                if assignDM:
                    molecule.dm = P
                return P, v

            else:
                x = fock_u_batch(
                    nmol,
                    molecule.molsize,
                    P,
                    M,
                    molecule.maskd,
                    molecule.mask,
                    molecule.idxi,
                    molecule.idxj,
                    w,
                    W,
                    gss,
                    gpp,
                    gsp,
                    gp2,
                    hsp,
                    molecule.method,
                    zetas,
                    zetap,
                    zetad,
                    molecule.Z,
                    F0SD,
                    G2SD,
                )

                # modified sym_eig_trunc below:
                dtype = x.dtype
                device = x.device

                nheavyatom = molecule.nHeavy.repeat_interleave(2)
                nH = molecule.nHydro.repeat_interleave(2)
                nocc = molecule.nocc.flatten()
                x_orig_shape = x.size()
                x0 = pack(x, nheavyatom, nH)

                size = x0.shape[1]
                norb = nheavyatom * 4 + nH
                has_padding = _apply_padding_eigen_shifts(x0, norb)
                try:
                    e0, v = sym_eigh(x0)
                except:
                    if torch.isnan(x0).any():
                        print("isnan(x0) #2 in DM guess", x0)
                    e0, v = sym_eigh(x0)
                nmol = x0.shape[0]
                e = torch.zeros((nmol, x.shape[-1]), dtype=dtype, device=device)
                e[..., :size] = e0
                e = _zero_padding_eigenvalues(e, norb, has_padding)

                # $$$ the code below can and SHOULD be optimized. Too many reshapes

                e = e.reshape(x_orig_shape[0:3])
                v = v.reshape(int(v.shape[0] / 2), 2, v.shape[1], v.shape[2])

                v_lumo = v[:, 0].gather(
                    2, molecule.nocc[:, 0].unsqueeze(0).unsqueeze(0).T.repeat(1, v.shape[-1], 1)
                )
                v_homo = v[:, 0].gather(
                    2, molecule.nocc[:, 0].unsqueeze(0).unsqueeze(0).T.repeat(1, v.shape[-1], 1) - 1
                )

                mix_coeff = torch.tensor([mix_coeff], device=device)

                v_a_homo = (1 - mix_coeff) * v_homo + (mix_coeff) * v_lumo
                # v_a_lumo = -(mix_coeff)*v_homo + (1-mix_coeff)*v_lumo

                # v_b_homo = (1-mix_coeff)*v_homo - torch.sin(mix_coeff)*v_lumo
                # v_b_lumo =  (mix_coeff)*v_homo + (1-mix_coeff)*v_lumo

                v[:, 0].scatter_(
                    2, molecule.nocc[:, 0].unsqueeze(0).unsqueeze(0).T.repeat(1, v.shape[-1], 1) - 1, v_a_homo
                )
                # v[:,0].scatter_(2, molecule.nocc[:,0].unsqueeze(0).unsqueeze(0).T.repeat(1,v.shape[-1],1), v_a_lumo)

                # v[:,1].scatter_(2, molecule.nocc[:,1].unsqueeze(0).unsqueeze(0).T.repeat(1,v.shape[-1],1)-1, v_b_homo)
                # v[:,1].scatter_(2, molecule.nocc[:,1].unsqueeze(0).unsqueeze(0).T.repeat(1,v.shape[-1],1), v_b_lumo)

                v = v.reshape(int(v.shape[0] * 2), v.shape[2], v.shape[3])

                if CHECK_DEGENERACY:
                    t = torch.stack(list(map(lambda a, b, n: construct_P(a, b, n), e, v, nocc)))
                else:
                    # list(map(lambda a,n : print('norm', torch.norm(v, dim=0), n), v, nocc))
                    # print(torch.norm())
                    t = 2.0 * torch.stack(
                        list(map(lambda a, n: torch.matmul(a[:, :n], a[:, :n].transpose(0, 1)), v, nocc))
                    )

                P = unpack(t, nheavyatom, nH, x.shape[-1])
                v = v.reshape(int(v.shape[0] / 2), 2, v.shape[1], v.shape[2])
                P = P.reshape(x_orig_shape) / 2
                if assignDM:
                    molecule.dm = P
                return P, v
        else:
            return P, None
    else:
        return P, None

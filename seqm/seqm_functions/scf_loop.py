import torch
from torch.autograd import grad as agrad
from .fock import fock as fock_restricted
from .fock_u_batch import fock_u_batch
from .hcore import hcore
from .energy import elec_energy
from .SP2 import SP2
from .fermi_q import Fermi_Q
from .G_XL_LR import G
from seqm.seqm_functions.canon_dm_prt import Canon_DM_PRT
from .pack import *
from .packd import *
from .diag import sym_eig_trunc, sym_eig_trunc1
from .diag_d import sym_eig_truncd, sym_eig_trunc1d
import warnings
import time
from .build_two_elec_one_center_int_D import calc_integral #, calc_integral_os
from .cg_solver import conjugate_gradient_batch
#from .check import check

#scf_backward==0: ignore the gradient on density matrix
#scf_backward==1: use recursive formula/implicit autodiff
#scf_backward==2: go backward scf loop directly. If the density matrix converges too fast then the gradients are wrong because not enough info about gradient of density is built up

debug = False
MAX_ITER = 2000
RAISE_ERROR_IF_SCF_FORWARD_FAILS = False
#if true, raise error rather than ignore those non-convered molecules

SCF_BACKWARD_MAX_ITER = 10
MAX_ITER_TO_STOP_IF_SCF_BACKWARD_DIVERGE = 5
RAISE_ERROR_IF_SCF_BACKWARD_FAILS = False
# if true, raise error rather than truncate gradient
# (set the gradient on non-converged molecules as 0.0)

# use memory-efficient evaluation of gradients in implicit autodiff
SCF_IMPLICIT_BACKWARD = True
# SCF_IMPLICIT_BACKWARD = False
# tolerance, max no. iteration, and history size for Anderson acceleration
# in solving fixed point problem in SCF_IMPLICIT_BACKWARD


# number of iterations in canon_dm_prt.py (m)
CANON_DM_PRT_ITER = 8

def make_Pnew_factory(method, sp2, molsize,
                      backward, scf_converger, openshell):
    """
    Returns a function Pnew = inner(F, nSH, nH, nHyd, nOcc) that
    applies exactly the right algorithm for your chosen flags.
    """
    if openshell and method == "PM6":
        raise ValueError("PM6 + open‐shell is not yet supported")

    if openshell and sp2[0]:
        raise ValueError("SP2 + open‐shell is not yet supported")
    # 1) pick the “core” step function
    if sp2[0]:
        # SP2-based
        if method == "PM6":
            packer   = lambda F, nsh, nh, nhy: packd(F, nsh, nh, nhy)
            unpacker = lambda D, nsh, nh, nhy: unpackd(D, nsh, nh, nhy, 9*molsize)
        else:
            packer   = lambda F, nh, nhy: pack(F, nh, nhy)
            unpacker = lambda D, nh, nhy: unpack(D, nh, nhy, 4*molsize)

        def _forward(F, nsh, nh, nhy, nOcc):
            D = packer(F, *( (nsh,nh,nhy) if method=="PM6" else (nh,nhy) ))
            D2 = SP2(D, nOcc, sp2[1])
            return unpacker(D2, *( (nsh,nh,nhy) if method=="PM6" else (nh,nhy) ))

        core_step = _forward

    else:
        # diagonalization of symmetric Fock matrix
        if method == "PM6":
            core_step = lambda F, nsh, nh, nhy, nOcc: sym_eig_truncd(F, nsh, nh, nhy, nOcc)[1]
        else:
            if scf_converger and "T_el" in scf_converger:
                core_step =  lambda F, nsh, nh, nhy, nOcc: Fermi_Q(F, scf_converger[3], nOcc, nh, nhy, 8.61739e-5,
                               False)[0]
            else:
                core_step = lambda F, nsh, nh, nhy, nOcc: sym_eig_trunc(F, nh, nhy, nOcc)[1]

    # 2) if backward, override core_step entirely
    if backward:
        if method == "PM6":
            core_step = lambda F, nsh, nh, nhy, nOcc: sym_eig_trunc1d(F, nsh, nh, nhy, nOcc)[1]
        else:
            core_step = lambda F, nsh, nh, nhy, nOcc: sym_eig_trunc1(F, nh, nhy, nOcc)[1]

    # 3) return the final inner function
    def inner(F, nSuperHeavy, nHeavy, nHydro, nOccMO):
        # Note: for SP2-packer we only pulled nsH/nh/nhy arguments from the
        # outer scope correctly above, so we can always pass the 5 args in.
        return core_step(F, nSuperHeavy, nHeavy, nHydro, nOccMO)

    return inner

def get_error(Pold, P, notconverged, matrix_size_sqrt, dm_err, dm_element_err, Eelec_new, err, Eelec, eps, diis_error=None, unrestricted=False):
    dP = (P[notconverged] - Pold[notconverged]).sum(dim=1) if unrestricted else (P[notconverged] - Pold[notconverged])
    dm_err[notconverged] = torch.norm(dP, dim = (1,2)) \
                             / matrix_size_sqrt[notconverged]
    dm_element_err[notconverged] = torch.amax(torch.abs(dP), dim=(1,2))
    max_dm_element_err = torch.max(dm_element_err)
    max_dm_err = torch.max(dm_err)
    
    err[notconverged] = Eelec_new[notconverged]-Eelec[notconverged]
    
    notconverged = (err.abs() > eps) | (dm_err > eps*2) | (dm_element_err > eps*15)
    if diis_error is not None:
        notconverged |= (diis_error > 50*eps)

    return notconverged, max_dm_err, max_dm_element_err

# constant mixing
def scf_forward0(M, w, W, gss, gpp, gsp, gp2, hsp, \
                nHydro, nHeavy, nSuperHeavy, nOccMO, \
                nmol, molsize, \
                maskd, mask, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD, sp2=[False], scf_converger=[0, 0.5], unrestricted=False, backward=False, verbose=True):
    """
    alpha : mixing parameters, alpha=0.0, directly take the new density matrix
    backward is for testing purpose, default is False
    if want to test scf backward directly through the loop in this function, turn backward to be True
    
    INPUT:
    M: core part of Fock matrix
    w: 2c-2e integrals (s,p)
    W: some integrals in PM6. zero in PM3 and PM6_SP.)
    """
    alpha=scf_converger[1]

    Pnew = torch.zeros_like(P)
    Pold = torch.zeros_like(P)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    dm_err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    dm_element_err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    fock = fock_u_batch if unrestricted else fock_restricted
    F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp,themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
    num_orbitals = 9 if themethod == 'PM6' else 4
    Hcore = M.reshape(nmol,molsize,molsize,num_orbitals,num_orbitals) \
             .transpose(2,3) \
             .reshape(nmol, num_orbitals*molsize, num_orbitals*molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)
    matrix_size_sqrt = torch.sqrt((nSuperHeavy * 9 + nHeavy * 4 + nHydro * 4)**2)
    one_minus_alpha = 1.0 - alpha

    make_Pnew = make_Pnew_factory(themethod,sp2,molsize,backward,scf_converger,unrestricted)

    for k in range(MAX_ITER+1):
        Pnew[notconverged] = make_Pnew(F[notconverged], nSuperHeavy[notconverged],nHeavy[notconverged], nHydro[notconverged], nOccMO[notconverged])

        if unrestricted:
            Pnew[notconverged] = Pnew[notconverged] / 2
    
        if backward:
            Pold = P.clone()
            P = torch.lerp(Pnew,P,alpha) # alpha * P + (1.0 - alpha) * Pnew
        else:
            Pold[notconverged] = P[notconverged]
            P[notconverged] = alpha * P[notconverged] + one_minus_alpha * Pnew[notconverged]
        
        F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
        Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])

        notconverged, max_dm_err, max_dm_element_err =  get_error(Pold, P, notconverged, matrix_size_sqrt, dm_err, dm_element_err, Eelec_new, err, Eelec, eps, unrestricted=unrestricted)
        
        Eelec[notconverged] = Eelec_new[notconverged]

        Nnot = int(notconverged.sum())
        if debug:
            print("scf direct step  : {:>3d} | E[{:>4d}]: {:>12.8f} | MAX \u0394E[{:>4d}]: {:>12.8f} | MAX \u0394DM[{:>4d}]: {:>12.7f} | MAX \u0394DM_ij[{:>4d}]: {:>10.7f}".format(
                        k, torch.argmax(abs(err)), Eelec_new[torch.argmax(abs(err))], torch.argmax(abs(err)), err[torch.argmax(abs(err))], torch.argmax(dm_err), max_dm_err, torch.argmax(dm_element_err), max_dm_element_err), " | N not converged:", Nnot)
            
        if Nnot == 0: 
            if verbose:
                print("scf direct step  : {:>3d} | E[{:>4d}]: {:>12.8f} | MAX \u0394E[{:>4d}]: {:>12.8f} | MAX \u0394DM[{:>4d}]: {:>12.7f} | MAX \u0394DM_ij[{:>4d}]: {:>10.7f}".format(
                        k, torch.argmax(abs(err)), Eelec_new[torch.argmax(abs(err))], torch.argmax(abs(err)), err[torch.argmax(abs(err))], torch.argmax(dm_err), max_dm_err, torch.argmax(dm_element_err), max_dm_element_err), " | N not converged:", Nnot)
            break
    return P, notconverged

def compute_fac(Pnew_diag, P_diag, Pold_diag):
    diff1 = Pnew_diag - P_diag
    diff1 = (Pnew_diag - P_diag)
    diff2 = (Pnew_diag - 2.0 * P_diag + Pold_diag)
    num = diff1.square().sum(dim=1)
    den = diff2.square().sum(dim=1)
    valid = (den > 0) & (num < 100.0 * den)
    FAC = torch.zeros_like(num)
    FAC[valid] = torch.sqrt(num[valid] / den[valid])
    return FAC

def adaptive_mix(scf_iteration, P_prev, P_cur, Pold2_diag, unrestricted):
    # We treat each molecule separately (different active sizes)
    is_third = (scf_iteration % 3 == 0)
    # DAMP per Fortran
    DAMP = 0.05 if scf_iteration > 4 else 1.0e10

    diag_prev = torch.diagonal(P_prev, dim1=1, dim2=2)  # [B, nbas]
    diag_cur  = torch.diagonal(P_cur,  dim1=1, dim2=2)  # [B, nbas]
    diag_old2 = Pold2_diag                          # [B, nbas]

    occ_number = 1.0 if unrestricted else 2.0
    if is_third:
        with torch.no_grad():
            FAC = compute_fac(diag_cur,diag_prev,diag_old2)
    
        # Off-diagonal extrapolation: Pmix = Pcur + FAC * (Pcur - Pprev)
        # TODO: unrestricted
        f_b = FAC.view(-1, 1, 1) # [B,1,1]
        Pmix = (1.0+ f_b) * P_cur - (f_b * P_prev)
    else:
        FAC = torch.zeros(P_cur.shape[0], dtype=P_cur.dtype, device=P_cur.device)
        Pmix = P_cur

    # Diagonal capped/extrapolated vs old2
    delta = diag_cur - diag_prev

    # piecewise: cap if |delta|>DAMP, else prev + FAC*delta
    # NOTE: DAMP is scalar per-iteration (same for all molecules)
    cap_mask = delta.abs() > DAMP
    diag_new = torch.where(cap_mask,
                           diag_prev + delta.sign() * DAMP,
                           diag_cur + FAC.view(-1, 1) * delta)
    diag_new.clamp_(0.0, occ_number)

    # --- Renormalize Σ diag to match Σ cur_diag  ---
    SUM0 = diag_cur.sum(dim=1)                                   # [B]
    di = diag_new
    full_di_occ = di.new_full((), occ_number)
    for _ in range(20):
        SUM2 = di.sum(dim=1)                        # sum of *current* di
        large = SUM2 > 1.0e-3
        SUM3 = torch.zeros_like(SUM2)
        SUM3[large] = SUM0[large] / SUM2[large]

        # if already normalized, stop
        done = (~large) | (torch.abs(SUM3 - 1.0) <= 1.0e-5)
        if torch.all(done):
            break

        # scale first, then clamp/split full vs partial 
        scaled = di.mul(SUM3.view(-1,1)).clamp_(min=0.0)
        new_full = scaled > occ_number
        di = torch.where(new_full, full_di_occ, scaled)

        # adjust SUM0 (the remaining electrons to distribute to partials) for next round
        # SUM0_next = SUM0 − 2*(#full); reuse SUM0 var name like Fortran
        SUM0 = SUM0 - new_full.to(di.dtype).sum(dim=1) * occ_number

    # Pmix[:, ar, ar] = di
    diag_view = torch.diagonal(Pmix, dim1=1, dim2=2)
    diag_view.copy_(di)                        # in-place write
    return Pmix, diag_prev

#adaptive mixing
def scf_forward1(M, w, W, gss, gpp, gsp, gp2, hsp, \
                nHydro, nHeavy, nSuperHeavy, nOccMO, \
                nmol, molsize, \
                maskd, mask, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD, sp2=[False], scf_converger=[1, 0.0, 0.0, 1], unrestricted=False, backward=False, verbose=True):
    Pnew = torch.zeros_like(P)
    Pold = torch.zeros_like(P)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    dm_err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    dm_element_err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    notconverged = torch.ones(nmol, dtype=torch.bool, device=M.device)

    fock = fock_u_batch if unrestricted else fock_restricted
    F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)

    num_orbitals = 9 if themethod == 'PM6' else 4
    Hcore = M.reshape(nmol, molsize, molsize, num_orbitals, num_orbitals) \
             .transpose(2, 3) \
             .reshape(nmol, num_orbitals * molsize, num_orbitals * molsize)

    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)

    # Workspace for diagonal of P from two steps before
    Pold2_diag = torch.zeros(P.shape[:-1], dtype=P.dtype, device=P.device)

    nbas = P.shape[-1]
    matrix_size_sqrt = torch.sqrt((nSuperHeavy * 9 + nHeavy * 4 + nHydro * 4)**2)

    make_Pnew = make_Pnew_factory(themethod, sp2, molsize, backward, scf_converger, unrestricted)



    for k in range(1,MAX_ITER + 1):
        # Build current density from current Fock
        Pnew[notconverged] = make_Pnew(F[notconverged],
                                       nSuperHeavy[notconverged],
                                       nHeavy[notconverged],
                                       nHydro[notconverged],
                                       nOccMO[notconverged])

        if unrestricted:
            Pnew[notconverged] = Pnew[notconverged] / 2.0

        nz = notconverged.nonzero(as_tuple=False).squeeze(-1)
        P_prev = P[nz]            # [B, nbas, nbas]
        P_cur  = Pnew[nz]         # [B, nbas, nbas]
        if unrestricted:
            Pmix_0, diag_prev_0 = adaptive_mix(k, P_prev[:,0], P_cur[:,0], Pold2_diag[nz,0], unrestricted)
            Pmix_1, diag_prev_1 = adaptive_mix(k, P_prev[:,1], P_cur[:,1], Pold2_diag[nz,1], unrestricted)
            if backward:
                # Build full-batch P via masked where (keeps autograd graph intact)
                Pold = P + 0.0
                P = P.clone()
                P[notconverged,0] = Pmix_0
                P[notconverged,1] = Pmix_1
            else:
                # Only touch notconverged rows (leanest path)
                Pold[notconverged] = P[notconverged]
                P[notconverged,0] = Pmix_0
                P[notconverged,1] = Pmix_1
            del Pmix_0, Pmix_1
            # Update old2 only for notconverged rows
            Pold2_diag[notconverged,0] = diag_prev_0
            Pold2_diag[notconverged,1] = diag_prev_1
        else:
            Pmix, diag_prev = adaptive_mix(k, P_prev, P_cur, Pold2_diag[nz], unrestricted)
            if backward:
                # Build full-batch P via masked where (keeps autograd graph intact)
                Pold = P + 0.0
                P = P.clone()
                P[notconverged] = Pmix
            else:
                # Only touch notconverged rows (leanest path)
                Pold[notconverged] = P[notconverged]
                P[notconverged] = Pmix
            del Pmix

            # Update old2 only for notconverged rows
            Pold2_diag[notconverged] = diag_prev

        # Rebuild Fock with mixed density
        F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
        Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])

        notconverged, max_dm_err, max_dm_element_err =  get_error(Pold, P, notconverged, matrix_size_sqrt, dm_err, dm_element_err, Eelec_new, err, Eelec, eps, unrestricted=unrestricted)

        Eelec[notconverged] = Eelec_new[notconverged]

        Nnot = int(notconverged.sum())
        if debug:
            print("scf adaptive step    : {:>3d} | E[{:>4d}]: {:>12.8f} | MAX ΔE[{:>4d}]: {:>12.8f} | MAX ΔDM[{:>4d}]: {:>12.7f} | MAX ΔDM_ij[{:>4d}]: {:>10.7f}".format(
                k, torch.argmax(abs(err)), Eelec_new[torch.argmax(abs(err))],
                torch.argmax(abs(err)), err[torch.argmax(abs(err))],
                torch.argmax(dm_err), max_dm_err,
                torch.argmax(dm_element_err), max_dm_element_err
            ), " | N not converged:", Nnot)

        if Nnot == 0:
            if verbose:
                print("scf adaptive step    : {:>3d} | E[{:>4d}]: {:>12.8f} | MAX ΔE[{:>4d}]: {:>12.8f} | MAX ΔDM[{:>4d}]: {:>12.7f} | MAX ΔDM_ij[{:>4d}]: {:>10.7f}".format(
                    k, torch.argmax(abs(err)), Eelec_new[torch.argmax(abs(err))],
                    torch.argmax(abs(err)), err[torch.argmax(abs(err))],
                    torch.argmax(dm_err), max_dm_err,
                    torch.argmax(dm_element_err), max_dm_element_err
                ), " | N not converged:", Nnot)
            break

    return P, notconverged

#adaptive mixing, pulay
def scf_forward2(M, w, W, gss, gpp, gsp, gp2, hsp, \
                nHydro, nHeavy, nSuperHeavy, nOccMO, \
                nmol, molsize, \
                maskd, mask, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD, sp2=[False], backward=False, verbose=True):

    """
    adaptive mixing algorithm, see cnvg.f
    combine with pulay converger
    #check mopac for which P is stored: P constructed from fock subroutine, or P from pulay algorithm
    """
    dtype=M.dtype
    device=M.device
    #procedure
    #nDirect1 steps of directly taking new density
    #nAdapt steps of adaptive mixing (start preparing for pulay)
    #nFock-nAdapt steps of directly taking new density
    #pulay

    nDirect1 = 5
    alpha_direct = 0.7

    nAdapt = 2
    num_orbitals = 9 if themethod == 'PM6' else 4
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    k = 0
    fock = fock_restricted
    F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    Pnew = torch.zeros_like(P)
    Pold = torch.zeros_like(P)
    dm_err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    dm_element_err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    Hcore = M.reshape(nmol,molsize,molsize,num_orbitals,num_orbitals) \
             .transpose(2,3) \
             .reshape(nmol, num_orbitals*molsize, num_orbitals*molsize)
        
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)

    matrix_size_sqrt = torch.sqrt((nSuperHeavy * 9 + nHeavy * 4 + nHydro * 4)**2)
    make_Pnew = make_Pnew_factory(themethod,sp2,molsize,backward,scf_converger=[2],openshell=False)

    Nnot = nmol
    for i in range(nDirect1):

        if Nnot > 0:
            Pnew[notconverged] = make_Pnew(F[notconverged], nSuperHeavy[notconverged],nHeavy[notconverged], nHydro[notconverged], nOccMO[notconverged])
            if backward:
                Pold = P.clone()
                P = alpha_direct * P + (1.0 - alpha_direct) * Pnew
            else:
                Pold[notconverged] = P[notconverged]
                P[notconverged] = alpha_direct * P[notconverged] + (1.0 - alpha_direct) * Pnew[notconverged]

            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
            
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])

            notconverged, max_dm_err, max_dm_element_err =  get_error(Pold, P, notconverged, matrix_size_sqrt, dm_err, dm_element_err, Eelec_new, err, Eelec, eps)
            
            Eelec[notconverged] = Eelec_new[notconverged]

            Nnot = int(notconverged.sum())
            if debug:
                print("scf direct step  : {:>3d} | MAX \u0394E[{:>4d}]: {:>12.7f} | MAX \u0394DM[{:>4d}]: {:>12.7f} | MAX \u0394DM_ij[{:>4d}]: {:>10.7f}".format(
                        k, torch.argmax(err), torch.max(err), torch.argmax(dm_err), max_dm_err, torch.argmax(dm_element_err), max_dm_element_err), " | N not converged:", Nnot)
            k = k + 1
        else:
            if verbose:
                print("scf direct step  : {:>3d} | E[{:>4d}]: {:>12.8f} | MAX \u0394E[{:>4d}]: {:>12.8f} | MAX \u0394DM[{:>4d}]: {:>12.7f} | MAX \u0394DM_ij[{:>4d}]: {:>10.7f}".format(
                        k, torch.argmax(abs(err)), Eelec_new[torch.argmax(abs(err))], torch.argmax(abs(err)), err[torch.argmax(abs(err))], torch.argmax(dm_err), max_dm_err, torch.argmax(dm_element_err), max_dm_element_err), " | N not converged:", Nnot)
            return P, notconverged
    
    
    """
    cFock = cFock + 1 if cFock < nFock else nFock
    #store fock matrix
    FOCK[...,counter,:,:] = F
    FPPF[...,counter,:,:] = (F.matmul(P) - P.matmul(F)).triu()
    # in mopac7 interp.f pulay subroutine, only lower triangle of FPPF is used to construct EMAT
    #only compute lower triangle as Emat are symmetric
    EMAT[...,counter,:cFock] = torch.sum(FPPF[..., counter:(counter+1),:,:]*FPPF[...,:cFock,:,:], dim=(2,3))
    """
    for i in range(nAdapt):
        if Nnot > 0:
            Pnew[notconverged] = make_Pnew(F[notconverged], nSuperHeavy[notconverged],nHeavy[notconverged], nHydro[notconverged], nOccMO[notconverged])
                    
            #fac = sqrt( \sum_i (P_ii^(k) - P_ii^(k-1))**2 / \sum_i (P_ii^(k) - 2*P_ii^(k-1) + P_ii^(k-2))**2 )
            with torch.no_grad():
                Dn   = Pnew[notconverged].diagonal(dim1=1, dim2=2)
                D    = P[notconverged].diagonal(dim1=1, dim2=2)
                Do   = Pold[notconverged].diagonal(dim1=1, dim2=2)
                num  = (Dn - D).pow(2).sum(1, keepdim=True)
                den  = (Dn - 2.0*D + Do).pow(2).sum(1, keepdim=True)
                fac = torch.sqrt(num/den).view(-1,1,1)
                if backward:
                    fac_register = torch.zeros((P.shape[0], 1, 1), dtype=P.dtype, device=P.device)
                    fac_register[notconverged] = fac
                
            if backward:
                Pold = P.clone()
                P = (1.0+fac_register)*Pnew - fac_register*P
            else:
                Pold[notconverged] = P[notconverged]
                P[notconverged] = (1.0+fac)*Pnew[notconverged] - fac*P[notconverged]
            
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])

            notconverged, max_dm_err, max_dm_element_err =  get_error(Pold, P, notconverged, matrix_size_sqrt, dm_err, dm_element_err, Eelec_new, err, Eelec, eps)
            
            Eelec[notconverged] = Eelec_new[notconverged]

            Nnot = int(notconverged.sum())
            if debug: print("scf adaptive step: {:>3d} | MAX \u0394E[{:>4d}]: {:>12.7f} | MAX \u0394DM[{:>4d}]: {:>12.7f} | MAX \u0394DM_ij[{:>4d}]: {:>10.7f}".format(
                            k, torch.argmax(err), torch.max(err), torch.argmax(dm_err), max_dm_err, torch.argmax(dm_element_err), max_dm_element_err), " | N not converged:", Nnot)

            k = k + 1
        else:
            if verbose:
                print("scf adaptive step  : {:>3d} | E[{:>4d}]: {:>12.8f} | MAX \u0394E[{:>4d}]: {:>12.8f} | MAX \u0394DM[{:>4d}]: {:>12.7f} | MAX \u0394DM_ij[{:>4d}]: {:>10.7f}".format(
                        k, torch.argmax(abs(err)), Eelec_new[torch.argmax(abs(err))], torch.argmax(abs(err)), err[torch.argmax(abs(err))], torch.argmax(dm_err), max_dm_err, torch.argmax(dm_element_err), max_dm_element_err), " | N not converged:", Nnot)
            return P, notconverged
    #del Pold, Pnew

    # number of maximal fock matrixes used
    nFock = 10

    """
    *      Emat is matrix with form
    *      |<E(1)*E(1)>  <E(1)*E(2)> ...   -1.0|
    *      |<E(2)*E(1)>  <E(2)*E(2)> ...   -1.0|
    *      |<E(3)*E(1)>  <E(3)*E(2)> ...   -1.0|
    *      |<E(4)*E(1)>  <E(4)*E(2)> ...   -1.0|
    *      |     .            .      ...     . |
    *      |   -1.0         -1.0     ...    0. |
    *
    *   WHERE <E(I)*E(J)> IS THE SCALAR PRODUCT OF [F*P] FOR ITERATION I
    *   TIMES [F*P] FOR ITERATION J.
    """
    # F*P - P*F = [F*P]
    FPPF = torch.zeros(nmol, nFock, molsize*num_orbitals, molsize*num_orbitals, dtype=dtype,device=device)
        
    EMAT = (torch.eye(nFock+1, nFock+1, dtype=dtype, device=device) - 1.0).expand(nmol,nFock+1,nFock+1).tril().clone()
    FOCK = torch.zeros_like(FPPF) # store last n=nFock number of Fock matrixes
    #start prepare for pulay algorithm
    counter = -1 # index of stored FPPF for current iteration: 0, 1, ..., cFock-1
    cFock = 0 # in current iteraction, number of fock matrixes stored, cFock <= nFock
    #Pulay algorithm needs at least two previous stored density and Fock matrixes to start
    alpha_direct = 0.5
    diis_error = torch.empty_like(Eelec).fill_(torch.finfo(dtype).max)

    reset_diis = False
    if dtype==torch.float64:
        eval_eps = 1e-13
        clamp_eps = 1e-15
    elif dtype==torch.float32:
        eval_eps = 1e-5
        clamp_eps = 1e-7
    else:
        raise RuntimeError

    while (1):

        if Nnot>0:
            cFock = cFock + 1 if cFock < nFock else nFock
            #store fock matrix
            counter = (counter + 1)%nFock
            FOCK[notconverged, counter, :, :] = F[notconverged]
            with torch.no_grad():
                FPPF[notconverged, counter, :, :] = (F[notconverged].matmul(P[notconverged]) - P[notconverged].matmul(F[notconverged])).triu()
                diis_error[notconverged] = torch.amax(FPPF[notconverged,counter].abs(),dim=(1,2))
                # in mopac7 interp.f pulay subroutine, only lower triangle of FPPF is used to construct EMAT
                # only compute lower triangle as Emat are symmetric
                EMAT[notconverged, counter, :cFock] = torch.sum(FPPF[notconverged, counter:(counter+1),:,:] * FPPF[notconverged,:cFock,:,:], dim=(2,3))

            if cFock>=2:
                with torch.no_grad():
                    EVEC = EMAT[notconverged,:(cFock+1),:(cFock+1)]
                    # EVEC += EVEC.tril(-1).transpose(1,2) # no need to symmetrize because torch.linalg.eigh uses only the lower triangle
                    denom = EVEC[:, counter, counter].clamp(clamp_eps)
                    EVEC[:,:cFock,:cFock] /= denom.view(-1,1,1)

                    # Calculate the pseudo-inverse to get the DIIS mixing coeffients
                    L, Q = torch.linalg.eigh(EVEC)
                    absvals = L.abs()
                    # calculate the condition-number to see if EVEC is ill-conditioned and hence DIIS needs to be reset
                    cond =  torch.amax(absvals,dim=-1) / torch.amin(absvals,dim=-1)#.clamp(min=1e-15)
                    reset_diis = torch.any(cond > 1e7)
                    valid_eigs = absvals > eval_eps
                    inv_eig = torch.zeros_like(L)
                    inv_eig[valid_eigs] = L[valid_eigs].reciprocal()
                    coeff = -torch.einsum('bki,bi,bi', Q[:, :cFock, :],inv_eig,Q[:,-1,:])  # (b', cFock)

                    # rhs = torch.zeros(Nnot, cFock+1, device=device, dtype=dtype)
                    # rhs[:,-1] = 1.0
                    # # x = torch.linalg.solve(EVEC, rhs.unsqueeze(-1)).squeeze(-1)  # [B, cFock+1]
                    # x = torch.linalg.lstsq(EVEC, rhs.unsqueeze(-1)).solution.squeeze(-1)  # [B, cFock+1]
                    # coeff = -x[:, :cFock]  # drop the last entry, apply “–” sign
                F[notconverged] = torch.sum(FOCK[notconverged,:cFock,:,:]*coeff.unsqueeze(-1).unsqueeze(-1), dim=1)

            Pnew[notconverged] = make_Pnew(F[notconverged], nSuperHeavy[notconverged],nHeavy[notconverged], nHydro[notconverged], nOccMO[notconverged])

            if backward:
                Pold = P.clone()
                if cFock<2:
                    P = alpha_direct * P + (1.0 - alpha_direct) * Pnew
                else:
                    P = Pnew.clone()
            else:
                Pold[notconverged] = P[notconverged]
                if cFock<2:
                    P[notconverged] = alpha_direct * P[notconverged] + (1.0 - alpha_direct) * Pnew[notconverged]
                else:
                    P[notconverged] = Pnew[notconverged]

            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)

            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])

            notconverged, max_dm_err, max_dm_element_err =  get_error(Pold, P, notconverged, matrix_size_sqrt, dm_err, dm_element_err, Eelec_new, err, Eelec, eps, diis_error=diis_error)

            Eelec[notconverged] = Eelec_new[notconverged]

            Nnot = int(notconverged.sum())
            if debug: print("scf pulay seeding step   : {:>3d} | MAX \u0394E[{:>4d}]: {:>12.7f} | MAX \u0394DM[{:>4d}]: {:>12.7f} | MAX \u0394DM_ij[{:>4d}]: {:>10.7f}".format(
                            k, torch.argmax(err), torch.max(err), torch.argmax(dm_err), max_dm_err, torch.argmax(dm_element_err), max_dm_element_err), " | N not converged:", Nnot)

            k = k + 1
            if reset_diis:
                reset_diis = False
                if debug: print(f"Resetting DIIS at k = {k}, cFock is {cFock}, counter is {counter}")
                counter = -1
                cFock = 0
                FPPF.zero_()
                EMAT = (torch.eye(nFock+1, nFock+1, dtype=dtype, device=device) - 1.0).expand(nmol,nFock+1,nFock+1).tril().clone()
                FOCK.zero_()

        else:
            if verbose:
                print("scf pulay diis   : {:>3d} | E[{:>4d}]: {:>12.8f} | MAX \u0394E[{:>4d}]: {:>12.8f} | MAX \u0394DM[{:>4d}]: {:>12.7f} | MAX \u0394DM_ij[{:>4d}]: {:>10.7f}".format(
                        k, torch.argmax(abs(err)), Eelec_new[torch.argmax(abs(err))], torch.argmax(abs(err)), err[torch.argmax(abs(err))], torch.argmax(dm_err), max_dm_err, torch.argmax(dm_element_err), max_dm_element_err), " | N not converged:", Nnot)
            return P, notconverged

        
def scf_forward3(M, w, W_pm6, gss, gpp, gsp, gp2, hsp, \
                nHydro, nHeavy, nSuperHeavy, nOccMO, \
                nmol, molsize, \
                maskd, mask, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD, xl_bomd_params, backward=False,verbose=False):
    """
    DM scf optimization using KSA
    $$$ probably, not properly optimized for batches. 
    backward is for testing purpose, default is False
    if want to test scf backward directly through the loop in this function, turn backward to be True
    """
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    dm_err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    Hcore = M.reshape(nmol,molsize,molsize,4,4) \
         .transpose(2,3) \
         .reshape(nmol, 4*molsize, 4*molsize)
    
    Temp = xl_bomd_params['T_el']
    kB = 8.61739e-5 # eV/K, kB = 6.33366256e-6 Ry/K, kB = 3.166811429e-6 Ha/K, #kB = 3.166811429e-6 #Ha/K
    COUNTER = 0
    
    Rank = xl_bomd_params['max_rank']
    V = torch.zeros((P.shape[0], P.shape[1], P.shape[2], Rank), dtype=P.dtype, device=P.device)
    W = torch.zeros((P.shape[0], P.shape[1], P.shape[2], Rank), dtype=P.dtype, device=P.device) # here W if from XL-BOMD, not PM6 2c-2e ints
    
    K0 = 1.0
    F = fock_restricted(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W_pm6, gss, gpp, gsp, gp2, hsp,themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
    D, S_Ent, QQ, e, Fe_occ, mu0, Occ_mask = Fermi_Q(F, Temp, nOccMO, nHeavy, nHydro, kB, scf_backward=0)
    dDS = K0 * (D - P)
    dW = dDS
    
    Eelec = torch.zeros((nmol), dtype=P.dtype, device=P.device)
    Eelec_new = torch.zeros_like(Eelec)
    
    while (1):
        start_time = time.time()
        if notconverged.any():
            COUNTER = COUNTER + 1
            D[notconverged], S_Ent[notconverged], QQ[notconverged], e[notconverged], \
                Fe_occ[notconverged], mu0[notconverged], Occ_mask[notconverged] = \
                Fermi_Q(F[notconverged], Temp, nOccMO[notconverged], nHeavy[notconverged], 
                        nHydro[notconverged], kB, scf_backward = 0)
                        
            dDS = K0 * (D - P)
            dW = dDS
            k = -1
            Error = torch.tensor([10], dtype=D.dtype, device=D.device)
            while k < Rank-1 and torch.max(Error) > xl_bomd_params['err_threshold']:
                k = k + 1
                V[:,:,:,k] = dW
                for j in range(0,k): #Orthogonalized Krylov vectors (Arnoldi)
                    V[:,:,:,k] = V[:,:,:,k] - torch.sum(V[:,:,:,k].transpose(1,2) * V[:,:,:,j], dim=(1,2)).view(-1, 1, 1) * V[:,:,:,j]
                V[:,:,:,k] = V[:,:,:,k] / torch.sqrt(torch.sum(V[:,:,:,k].transpose(1,2) * V[:,:,:,k], dim=(1,2))).view(-1, 1, 1)
                d_D = V[:,:,:,k]
                FO1 = G(nmol, molsize, d_D, M, maskd, mask, idxi, idxj, w, W_pm6, gss, gpp, gsp, gp2, hsp,themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
                
                PO1 = Canon_DM_PRT(FO1, Temp, nHeavy, nHydro, QQ, e, mu0, CANON_DM_PRT_ITER, kB, Occ_mask)
                W[:,:,:,k] = K0 * (PO1 - V[:,:,:,k])
                dW = W[:,:,:,k]
                Rank_m = k + 1
                O = torch.zeros((D.shape[0], Rank_m, Rank_m), dtype=D.dtype, device=D.device)
                for I in range(0, Rank_m):
                    for J in range(I, Rank_m):
                        O[:,I,J] = torch.sum(W[:,:,:,I].transpose(1,2) * W[:,:,:,J], dim=(1,2))
                        O[:,J,I] = O[:,I,J]
                
                MM = torch.inverse(O)
                IdentRes = torch.zeros(D.shape, dtype=D.dtype, device=D.device)
                for I in range(0,Rank_m):
                    for J in range(0, Rank_m):
                        IdentRes = IdentRes + \
                            MM[:,I,J].view(-1, 1, 1) * torch.sum(W[:,:,:,J].transpose(1,2) * dDS, dim=(1,2)).view(-1, 1, 1) * W[:,:,:,I]
                Error = torch.linalg.norm(IdentRes - dDS, ord='fro', dim=(1,2))/torch.linalg.norm(dDS, ord='fro', dim=(1,2))
            
            #print(MM)
            for I in range(0, Rank_m):
                for J in range(0, Rank_m):
                    P[notconverged] = P[notconverged] - \
                            MM[notconverged,I,J].view(-1, 1, 1) *torch.sum(W[notconverged,:,:,J].transpose(1,2)*dDS[notconverged], dim=(1,2)).view(-1, 1, 1) * V[notconverged,:,:,I]
            
            F = fock_restricted(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W_pm6, gss, gpp, gsp, gp2, hsp,themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged] - Eelec[notconverged])
            dm_err[notconverged] = torch.linalg.norm(dDS[notconverged], ord='fro', dim=(1,2))
            max_dm_err = torch.max(dm_err)
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err)
            Nnot = torch.sum(notconverged).item()
            

            if debug: print("scf KSA step     : {:>3d} | MAX \u0394E[{:>4d}]: {:>12.7f} | MAX \u0394DM[{:>4d}]: {:>12.7f}".format(
                        COUNTER, torch.argmax(err), max_err, torch.argmax(dm_err), max_dm_err), " | N not converged:", Nnot)

            if COUNTER >= MAX_ITER: return P, notconverged
        else:
            if verbose: 
                print("scf KSA step     : {:>3d} | MAX \u0394E[{:>4d}]: {:>12.7f} | MAX \u0394DM[{:>4d}]: {:>12.7f}".format(
                        COUNTER, torch.argmax(err), max_err, torch.argmax(dm_err), max_dm_err), " | N not converged:", Nnot)
            return P, notconverged
        

SCF_BACKWARD_ANDERSON_MAXITER = 200      # sufficient for all test cases
SCF_BACKWARD_ANDERSON_HISTSIZE = 3      # seems reasonable, but TODO!
SCF_BACKWARD_LAMBDA_MIN   = 1e-6          # floor for λ
SCF_BACKWARD_BETA0        = 0.20          # mixing for the first 3 steps
SCF_BACKWARD_BETA         = 0.95          # mixing afterwards
def fixed_point_anderson(fp_fun, u0, tol, lam=1e-3, 
                         beta0=SCF_BACKWARD_BETA0,
                         beta=SCF_BACKWARD_BETA,
                         lam_min=SCF_BACKWARD_LAMBDA_MIN):
    """
    Anderson acceleration for fixed point solver adapted from
    http://implicit-layers-tutorial.org/deep_equilibrium_models
    
    Parameters
    ----------
    fp_fun : callable
        function defining fixed point solution, i.e., ``fp_fun(u) = u``
    u0 : array-/Tensor-like
        initial guess for fixed point
    lam : float
        regularization for solving linear system
    beta : float
        mixing parameter
    tol : float
        convergence tolerance
    
    Returns
    -------
    u_conv : array-/Tensor-like
        fixed point solution satisfying ``fp_fun(u_conv) = u_conv``
    
    """
    # handle UHF/RHF
    if u0.dim() == 4:
        nmol, nsp, norb, morb = u0.shape
    else:
        nmol, norb, morb = u0.shape
        nsp = 1
    # init history of solver
    m = SCF_BACKWARD_ANDERSON_HISTSIZE
    X = torch.zeros((nmol, m, nsp*norb*morb), dtype=u0.dtype, device=u0.device)
    F = torch.zeros((nmol, m, nsp*norb*morb), dtype=u0.dtype, device=u0.device)
    X[:,0], F[:,0] = u0.view(nmol, -1), fp_fun(u0).view(nmol, -1)
    X[:,1], F[:,1] = F[:,0], fp_fun(F[:,0].view_as(u0)).view(nmol, -1)
    # set up linear system
    H = torch.zeros((nmol, m+1, m+1), dtype=u0.dtype, device=u0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros((nmol, m+1, 1), dtype=u0.dtype, device=u0.device)
    y[:,0] = 1
    cond, resid = False, 100.
    # solve iteratively
    max_iters = SCF_BACKWARD_ANDERSON_MAXITER
    for k in range(2, max_iters):
        n = min(k, m)
        G = F[:,:n] - X[:,:n]
        GTG = torch.bmm(G, G.transpose(1,2))
        E_n = torch.eye(n, dtype=u0.dtype, device=u0.device)
        H[:,1:n+1,1:n+1] = GTG + lam * E_n[None]
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:,1:n+1,0]
        beta_k = beta if k > 10 else beta0
        Xold = (1 - beta_k) * (alpha[:,None] @ X[:,:n])[:,0]
        X[:,k%m] = beta_k * (alpha[:,None] @ F[:,:n])[:,0] + Xold
        F[:,k%m] = fp_fun(X[:,k%m].view_as(u0)).view(nmol, -1)
        resid1 = (F[:,k%m] - X[:,k%m]).norm()
        resid = resid1 / (1e-5 + F[:,k%m].norm())
        cond = resid.item() < tol # SCF_BACKWARD_ANDERSON_TOLERANCE
        if cond: break   # solver converged
        # λ exponential decay (once we are in the basin)
        lam = max(lam * 0.5, lam_min)
    if not cond:
        msg = "Anderson solver in SCF backward did not converge for some molecule(s)"
        if RAISE_ERROR_IF_SCF_BACKWARD_FAILS: raise ValueError(msg)
        print(msg)
    u_conv = X[:,k%m].view_as(u0)
    # print(f"Anderson took {k} iterations")
    return u_conv
    

def fixed_point_picard(fp_fun, u0, tol, maxiter=SCF_BACKWARD_ANDERSON_MAXITER):
    u = u0.clone()
    for k in range(maxiter):
        u_new = fp_fun(u)
        # relative residual
        resid = (u_new - u).norm() / (1e-6 + u_new.norm())
        if resid < tol:
            # print(f"Picard convered in {k} iters")
            return u_new
        u = u_new
    raise RuntimeError(f"Picard did not converge in {maxiter} steps (resid={resid:.2e})")

class SCF(torch.autograd.Function):
    """
    A custom autograd Function that wraps scf loop
    
    Because density matrix P* is solved for self-consistently in SCF, P* satisfies:
        P* = g(P*; θ)
    where θ = (M, w, gss, gpp, gsp, gp2, hsp,...) are the semiempirical parameters.

    Forward:
      - Runs an SCF loop to convergence:
          P_{new} = g(P_old; θ)
      - Takes inputs θ 
      - Returns the converged density P* and a mask indicating any molecules that failed.

    Backward:
      We receive `grad_P = ∂L/∂P*` from upstream.  We must produce `∂L/∂θ` for each
      parameter tensors θ = M, w, W, gss, gpp, gsp, gp2, hsp, etc.

      By the chain rule,
          ∂L/∂θ = (∂L/∂P*) · (∂P*/∂θ) = vᵀ·J (vector-Jacobian product) 

      Hence we need the jacobian (∂P*/∂θ). 
      Implicit differentiation of P* = g(P*;θ) gives
          (I − ∂g/∂P) · (∂P*/∂θ) = ∂g/∂θ,
      so formally
          ∂P*/∂θ = [I − ∂g/∂P]⁻¹ · ∂g/∂θ.

      We never formally invert F = [I − ∂g/∂P]. Let B = ∂g/∂θ
      We need to calculate the vector-Jacobian product:
      vᵀ · J = vᵀ· F⁻¹ · B = zᵀ · B,
      where zᵀ = vᵀ· F⁻¹. So we only need to solve for z. 

      1. if SCF_IMPLICIT_BACKWARD is False, we expand the inverse via the
      Neumann series (valid when spectral radius(∂g/∂P)<1):
          [I − A]⁻¹ = I + A + A² + ⋯ ,  A = ∂g/∂P.

      Concretely, we unroll:
          G₀ = ∂L/∂P*,
          G_{k+1} = (∂g/∂P)ᵀ · G_k,
      and accumulate
          ∂L/∂θ = Σ_k G_kᵀ · (∂g/∂θ).

      2. if SCF_IMPLICIT_BACKWARD is True, we do Implicit‐adjoint solve
      We have 
        zᵀ = vᵀ· F⁻¹ = vᵀ· [I-A]⁻¹  
        or zᵀ = vᵀ + zᵀ · A or z = v + Aᵀ · z 

      Instead of unrolling a Neumann series for (I−A)⁻¹,
      we solve the affine equation for z: z = v + Aᵀ · z  directly in fixed_point_picard

      However if fixed_point_picard does not converge we can accelerate the convergence 
      using Anderson acceleration, which is the same as DIIS. This is done like so: 
        1. Initialize z⁽⁰⁾ = 0.
        2. For k = 0,1,2,… until convergence:
           a) f(z⁽ᵏ⁾) = v + Aᵀ · z⁽ᵏ⁾    # vector‐Jacobian product via autograd
           b) r⁽ᵏ⁾ = f(z⁽ᵏ⁾) − z⁽ᵏ⁾        # residual
           c) Keep the last m pairs {(z⁽ᵏ⁻ʲ⁾, r⁽ᵏ⁻ʲ⁾)} for j=0…m−1
           d) Solve the small least‐squares problem
                 min_{α_j, ∑α_j=1} ‖∑_j α_j r⁽ᵏ⁻ʲ⁾‖
           e) update
                 z⁽ᵏ⁺¹⁾ = ∑_j α_j f(z⁽ᵏ⁻ʲ⁾)
        3. Stop when ‖z⁽ᵏ⁺¹⁾ − z⁽ᵏ⁾‖ < tol.

      Once converged, the final z approximates (I − Aᵀ)⁻¹ g_P.
      With some testing it seems that using Anderson acceleration is slower than directly solving for z!
      I have commented out the call to fixed_point_anderson for now

      We then compute all parameter gradients in one shot:
          grads = torch.autograd.grad(Pout, gv, grad_outputs=u)
      so that each grads[i] = z · (∂g/∂θ_i) = ∂L/∂θ_i.

      Notes:
        - We wrap the Anderson solve in torch.no_grad(), so this path does not
          support double‐backward (second derivatives).
        - Anderson acceleration seems to converges in fewer iterations than
          Neumann‐series unrolling.
    """
    def __init__(self, scf_converger=[2], use_sp2=[False], scf_backward_eps=1.0e-2):
        SCF.sp2 = use_sp2
        SCF.converger = scf_converger
        SCF.scf_backward_eps = scf_backward_eps
    
    
    @staticmethod
    def forward(ctx, \
                M, w, W, gss, gpp, gsp, gp2, hsp, \
                nHydro, nHeavy, nSuperHeavy,  nOccMO, \
                nmol, molsize, \
                maskd, mask, atom_molid, pair_molid, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD,verbose):
        
        SCF.scf_backward_eps = eps # set the convergence tolerance for backprop the same as the scf tolerance
        SCF.themethod = themethod
        unrestricted=(P.dim()==4)
        if SCF.converger[0] == 0:
            P, notconverged =  scf_forward0(M, w, W, gss, gpp, gsp, gp2, hsp, \
                               nHydro, nHeavy, nSuperHeavy, nOccMO, \
                               nmol, molsize, \
                               maskd, mask, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD, sp2=SCF.sp2, scf_converger=SCF.converger,unrestricted=unrestricted, verbose=verbose )
        elif SCF.converger[0] == 3: # KSA
            if unrestricted:
                raise NotImplementedError("scf_converger = [3] (KSA) not yet implemented for unrestricted calculations. Set UHF = False")
            P, notconverged =      scf_forward3(M, w, W, gss, gpp, gsp, gp2, hsp, \
                                   nHydro, nHeavy, nSuperHeavy, nOccMO, \
                                   nmol, molsize, \
                                   maskd, mask, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD, SCF.converger[1], verbose=verbose)
        else:
            if SCF.converger[0] == 1: # adaptive mixing
                P, notconverged = scf_forward1(M, w, W, gss, gpp, gsp, gp2, hsp, \
                                            nHydro, nHeavy, nSuperHeavy, nOccMO, \
                                            nmol, molsize, \
                                            maskd, mask, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD, sp2=SCF.sp2, scf_converger=SCF.converger,unrestricted=unrestricted, verbose=verbose)
            elif SCF.converger[0] == 2: # adaptive mixing, then pulay
                if unrestricted:
                    raise NotImplementedError("scf_converger = [2] (Pulay DIIS) not yet implemented for unrestricted calculations. Set UHF = False")
                P, notconverged = scf_forward2(M, w, W, gss, gpp, gsp, gp2, hsp, \
                           nHydro, nHeavy, nSuperHeavy, nOccMO, \
                           nmol, molsize, \
                           maskd, mask, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD, sp2=SCF.sp2, verbose=verbose)

        eps = torch.as_tensor(eps, dtype=M.dtype, device=M.device)
        ctx.save_for_backward(P, M, w, W, gss, gpp, gsp, gp2, hsp, \
                              nHydro, nHeavy, nSuperHeavy, nOccMO, \
                              maskd, mask, idxi, idxj, eps, zetas, zetap, zetad, Z, F0SD, G2SD, notconverged, \
                              atom_molid, pair_molid)

        return P, notconverged


    @staticmethod
    def backward(ctx, grad_P, grad1):
        """
        Compute ∂L/∂θ given grad_P = ∂L/∂P*.
        We must return one tensor ∂L/∂θ (or None) for each of the forward inputs θ 

        
        CURRENTLY DOES NOT SUPPORT DOUBLE-BACKWARD!
        (irrelevant for dE^2/dx^2 or dy/dparam, if y is no derivative itself like forces)
        FOR CORRECT SECOND DERIVATIVES OF DENSITY MATRIX, USE DIRECT BACKPROP.
        """
        #TODO: clean up when fully switching to implicit autodiff
        Pin, M, w, W, gss, gpp, gsp, gp2, hsp, \
        nHydro, nHeavy, nSuperHeavy, nOccMO, \
        maskd, mask, idxi, idxj, eps, zetas, zetap, zetad, Z, F0SD, G2SD, notconverged, \
        atom_molid, pair_molid = ctx.saved_tensors
        nmol = Pin.shape[0]
        themethod = SCF.themethod
        
        if SCF.themethod == 'PM6':
            molsize = Pin.shape[1]//9
        else:
            molsize = Pin.shape[-1]//4
        grads, gvind = {}, []
        gv = [] if SCF_IMPLICIT_BACKWARD else [Pin]
        # Build the list gv = [Pin, θ₁=M, θ₂=w, θ₃=W, θ₄=gss, …]
        for i, st in enumerate([M, w, W, gss, gpp, gsp, gp2, hsp]):
            if st.requires_grad:
                gv.append(st)
                gvind.append(i+1)
            else:
                grads[i+1] = None
        unrestricted = (Pin.dim()==4)
        fock = fock_u_batch if unrestricted else fock_restricted
        with torch.enable_grad():
            Pin.requires_grad_(True)
            F = fock(nmol, molsize, Pin, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp,\
                             themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
            if(themethod == 'PM6'):
                Pout = sym_eig_trunc1d(F, nSuperHeavy, nHeavy, nHydro, nOccMO)[1]
            else:
                Pout = sym_eig_trunc1(F, nHeavy, nHydro, nOccMO)[1]
            if unrestricted:
                Pout = Pout/2
                
        backward_eps = SCF.scf_backward_eps.to(Pin.device)
        converged = ~notconverged.detach() # scf forward converged
        diverged = None # scf backward diverged
        
         # Depending on flag, choose implicit vs Neumann‐series backward
        if SCF_IMPLICIT_BACKWARD:
            ## THIS DOES NOT SUPPORT DOUBLE BACKWARD. MAY AS WELL STOP AUTOGRAD TAPE
            ## TODO: INCLUDING THIS PART IN GRAPH CAUSES MEMORY LEAK.
            ##       RESOLVE THIS WHEN IMPLEMENTING DOUBLE-BACKWARD!
            with torch.no_grad():
                def affine_eq(u): return grad_P + agrad(Pout, Pin, grad_outputs=u, retain_graph=True)[0] # agrad is torch.autograd.grad
                u_init = torch.zeros_like(Pin)  
                u = fixed_point_anderson(affine_eq, u_init, backward_eps*10)
                u = fixed_point_picard(affine_eq, u, backward_eps)
                # u = fixed_point_picard(affine_eq, u_init,backward_eps)
                # def A_matvec(u): return u - agrad(Pout, Pin, grad_outputs=u, retain_graph=True)[0]
                # u = conjugate_gradient_batch(A_matvec,grad_P,tol=backward_eps*100)

                gradients = agrad(Pout, gv, grad_outputs=u, retain_graph=True)
                for t, i in enumerate(gvind): grads[i] = gradients[t]
        else:
            # — Neumann‐series unrolling —
            gradients = [(grad_P,)]
            for k in range(SCF_BACKWARD_MAX_ITER+1):
                grad0_max_prev = gradients[-1][0].abs().max(dim=-1)[0].max(dim=-1)[0]
                gradients.append(agrad(Pout, gv, grad_outputs=gradients[-1][0], create_graph=True))
                grad0_max =  gradients[-1][0].abs().max(dim=-1)[0].max(dim=-1)[0]
                if converged.any():
                    err = torch.max(grad0_max[converged])
                else:
                    err = torch.tensor(0.0, device=grad_P.device)
                if debug:
                    t = grad0_max[converged] > backward_eps
                    print('backward scf: ', k, err.item(), t.sum().item())
        
                if (err < backward_eps): break
                diverged = (grad0_max > grad0_max_prev) * (grad0_max>=1.0)
                if diverged.any() and k>=MAX_ITER_TO_STOP_IF_SCF_BACKWARD_DIVERGE:
                    print("SCF backward diverges for %d molecules, stop after %d iterations" %
                          (diverged.sum().item(), MAX_ITER_TO_STOP_IF_SCF_BACKWARD_DIVERGE))
                    break
            ln = len(gradients)
            # Accumulate ∂L/∂θ = Σ_k G_kᵀ · (∂g/∂θ)
            for t, i in enumerate(gvind):
                grads[i] = torch.sum(torch.stack([gradients[l][t+1] for l in range(1, ln)]), dim=0)
        
            with torch.no_grad():
                #one way is to check grad0.abs().max if they are smaller than backward_eps
                #another way is to compare grad0.abs().max with previous grad0.abs().max, if
                #      they are increasing, they are diverging
                notconverged1 = (grad0_max>backward_eps) + (~torch.isfinite(grad0_max))
                if notconverged.any():
                    print("SCF forward       : %d/%d not converged" % (notconverged.sum().item(),nmol))
                if notconverged1.any():
                    print("SCF backward      : %d/%d not converged" % (notconverged1.sum().item(),nmol))
                    if RAISE_ERROR_IF_SCF_BACKWARD_FAILS:
                        raise ValueError("SCF backward doesn't converged for some molecules")
                notconverged = notconverged + notconverged1
            
        with torch.no_grad():
            if notconverged.any():
                print("SCF for/back-ward : %d/%d not converged" % (notconverged.sum().item(),nmol))
                cond = notconverged.detach()
                #M, w, gss, gpp, gsp, gp2, hsp
                #M shape(nmol*molsizes*molsize, 4, 4)
                if torch.is_tensor(grads[1]):
                    if(themethod == 'PM6'):
                        grads[1] = grads[1].reshape(nmol, molsize*molsize, 9, 9)
                        grads[1][cond] = 0.0
                        grads[1] = grads[1].reshape(nmol*molsize*molsize, 9, 9)
                    else:
                        grads[1] = grads[1].reshape(nmol, molsize*molsize, 4, 4)
                        grads[1][cond] = 0.0
                        grads[1] = grads[1].reshape(nmol*molsize*molsize, 4, 4)
                #w shape (npairs, 10, 10)
                if torch.is_tensor(grads[2]): grads[2][cond[pair_molid]] = 0.0
                #gss, gpp, gsp, gp2, hsp shape (natoms,)
                for i in range(3,8):
                    if torch.is_tensor(grads[i]): grads[i][cond[atom_molid]] = 0.0
        
        return grads[1], grads[2], grads[3], grads[4], grads[5], grads[6], grads[7], \
               grads[8], None, None, \
               None, None, \
               None, None, None, None, None, None, None, None, \
               None, None, None, None, None, None, None, None, None, None
        
    

class SCF0(SCF):
    @staticmethod
    def backward(ctx, grad_P, grad1):
        # ignore gradient on density matrix and eigenvectors/-values
        return None, None, None, None, None, None, None, \
               None, None, None, \
               None, None, \
               None, None, None, None, None, None, None, None, \
               None, None, None, None, None, None, None, None, None, None


def scf_loop(molecule, \
            eps = 1.0e-4, P=None, sp2=[False], scf_converger=[1], eig=False, scf_backward=0, scf_backward_eps=1.0e-2):
    """
    SCF loop
    # check hcore.py for the details of arguments
    eps : convergence criteria for density matrix on density matrix
    P : if provided, will be used as initial density matrix in scf loop
    return : F, e, P, Hcore, w, v
    """
    device = molecule.xij.device
    nmol = molecule.nHeavy.shape[0]
    tore = molecule.const.tore
    if molecule.const.do_timing: t0 = time.time()
    M, w,rho0xi,rho0xj, riXH, ri = hcore(molecule)

    if molecule.const.do_timing:
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.time()
        molecule.const.timing["Hcore + STO Integrals"].append(t1 - t0)
        t0 = time.time()
    #if scf_backward == 2 or (not torch.is_tensor(P)):
    num_orbitals = 9 if molecule.method == 'PM6' else 4
    unrestricted = (molecule.nocc.dim() == 2)
    if (not torch.is_tensor(P)): # $$$ I'm not sure if it is okay to use DM initialized from make_dm_guess which corresponds to 1 SCF iteration + HOMO-LUMO mix.
        P0 = torch.zeros_like(M)  # density matrix
        P0[molecule.maskd[molecule.Z>1],0,0] = tore[molecule.Z[molecule.Z>1]]/4.0
        P0[molecule.maskd,1,1] = P0[molecule.maskd,0,0]
        P0[molecule.maskd,2,2] = P0[molecule.maskd,0,0]
        P0[molecule.maskd,3,3] = P0[molecule.maskd,0,0]

        P0[molecule.maskd[molecule.Z==1],0,0] = 1.0

        P = P0.reshape(nmol,molecule.molsize,molecule.molsize,num_orbitals,num_orbitals) \
            .transpose(2,3) \
            .reshape(nmol, num_orbitals*molecule.molsize, num_orbitals*molecule.molsize)
        
        if unrestricted: # alpha and beta dm for open shell
            #print('DOING UHF!!!!!!')
            P = torch.stack((0.5*P, 0.5*P), dim=1)
    #print('GRAD P',P.requires_grad)
    if(molecule.method == 'PM6'): # PM6 does not work. ignore this part
        if unrestricted: # open shell
            
            W, W_exch = calc_integral_os(molecule.parameters['s_orb_exp_tail'], molecule.parameters['p_orb_exp_tail'], molecule.parameters['d_orb_exp_tail'],\
                                         molecule.Z, nmol*molecule.molsize*molecule.molsize, molecule.maskd, P, molecule.parameters['F0SD'], molecule.parameters['G2SD'])
            W = torch.stack((W, W_exch))
            #print(W_exch)
        else:
            W = calc_integral(molecule.parameters['s_orb_exp_tail'], molecule.parameters['p_orb_exp_tail'], molecule.parameters['d_orb_exp_tail'],\
                              molecule.Z, nmol*molecule.molsize*molecule.molsize, molecule.maskd, P, molecule.parameters['F0SD'], molecule.parameters['G2SD'])
            W_exch = torch.tensor([0], device=molecule.nocc.device)
    else:
        W = torch.tensor([0], device=molecule.nocc.device)
        W_exch = torch.tensor([0], device=molecule.nocc.device)
    
    #"""
    #scf_backward == 2, directly backward through scf loop
    #             can't reuse P, so put P=None and initial P above
    #"""
    verbose = molecule.verbose

    if scf_backward == 2:
        if sp2[0]:
            warnings.warn('SP2 is not used for direct backpropagation through scf loop')
            sp2[0] = False
        if scf_converger[0] == 0:
            Pconv, notconverged =  scf_forward0  (M, w, W, molecule.parameters['g_ss'], molecule.parameters['g_pp'], molecule.parameters['g_sp'], molecule.parameters['g_p2'], molecule.parameters['h_sp'], \
                     molecule.nHydro, molecule.nHeavy, molecule.nSuperHeavy, molecule.nocc, \
                     nmol, molecule.molsize, \
                     molecule.maskd, molecule.mask, molecule.idxi, molecule.idxj, P, eps, molecule.method, molecule.parameters['s_orb_exp_tail'], molecule.parameters['p_orb_exp_tail'], molecule.parameters['d_orb_exp_tail'], molecule.Z, molecule.parameters['F0SD'], molecule.parameters['G2SD'], sp2=sp2, scf_converger=scf_converger, unrestricted=unrestricted, backward=True, verbose=verbose)
        elif scf_converger[0] == 1:
            Pconv, notconverged =  scf_forward1(  M, w, W, molecule.parameters['g_ss'], molecule.parameters['g_pp'], molecule.parameters['g_sp'], molecule.parameters['g_p2'], molecule.parameters['h_sp'], \
                     molecule.nHydro, molecule.nHeavy, molecule.nSuperHeavy, molecule.nocc, \
                     nmol, molecule.molsize, \
                     molecule.maskd, molecule.mask, molecule.idxi, molecule.idxj, P, eps, molecule.method, molecule.parameters['s_orb_exp_tail'], molecule.parameters['p_orb_exp_tail'], molecule.parameters['d_orb_exp_tail'], molecule.Z, molecule.parameters['F0SD'], molecule.parameters['G2SD'], sp2=sp2, scf_converger = scf_converger, backward=True, unrestricted = unrestricted, verbose=verbose)
        elif scf_converger[0] == 2:
            if unrestricted:
                raise NotImplementedError("scf_converger = [2] (Pulay DIIS) not yet implemented for unrestricted calculations. Set UHF = False")
            else:
                Pconv, notconverged =  scf_forward2(  M, w, W, molecule.parameters['g_ss'], molecule.parameters['g_pp'], molecule.parameters['g_sp'], molecule.parameters['g_p2'], molecule.parameters['h_sp'], \
                         molecule.nHydro, molecule.nHeavy, molecule.nSuperHeavy, molecule.nocc, \
                         nmol, molecule.molsize, \
                         molecule.maskd, molecule.mask, molecule.idxi, molecule.idxj, P, eps, molecule.method, molecule.parameters['s_orb_exp_tail'], molecule.parameters['p_orb_exp_tail'], molecule.parameters['d_orb_exp_tail'], molecule.Z, molecule.parameters['F0SD'], molecule.parameters['G2SD'], sp2=sp2,backward=True, verbose=verbose)
        else:
            raise ValueError("""For direct backpropagation through scf,
                                must set scf_converger=[0, alpha] or [1,...] or [2]\n""")
    #scf_backward 1, use recursive formula/implicit autodiff
    elif scf_backward == 1:
        scfapply = SCF(use_sp2=sp2, scf_converger=scf_converger, scf_backward_eps=scf_backward_eps).apply
    #scf_backward 0: ignore the gradient on density matrix
    elif scf_backward == 0:
        scfapply = SCF0(use_sp2=sp2, scf_converger=scf_converger).apply

    # apply_params = {k:pp[k] for k in apply_param_map}
    if scf_backward==0 or scf_backward==1:
        with torch.set_grad_enabled(scf_backward==1): # no grad if scf_backward==0
            Pconv, notconverged = scfapply(M, w, W, molecule.parameters['g_ss'], molecule.parameters['g_pp'], molecule.parameters['g_sp'], molecule.parameters['g_p2'], molecule.parameters['h_sp'], \
                molecule.nHydro, molecule.nHeavy, molecule.nSuperHeavy, molecule.nocc, \
                nmol, molecule.molsize, \
                molecule.maskd, molecule.mask, molecule.atom_molid, molecule.pair_molid, molecule.idxi, molecule.idxj, P, eps, molecule.method, molecule.parameters['s_orb_exp_tail'], molecule.parameters['p_orb_exp_tail'], molecule.parameters['d_orb_exp_tail'], molecule.Z, molecule.parameters['F0SD'], molecule.parameters['G2SD'], verbose )

    if notconverged.any():
        nnot = notconverged.type(torch.int).sum().data.item()
        print('did not converge', nnot)
        print('not converged: ', notconverged.nonzero().squeeze())
        warnings.warn("SCF for %d/%d molecules doesn't converge after %d iterations" % (nnot, nmol, MAX_ITER))
        if RAISE_ERROR_IF_SCF_FORWARD_FAILS:
            raise ValueError("SCF for some the molecules in the batch doesn't converge")

    if molecule.const.do_timing:
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.time()
        molecule.const.timing["SCF"].append(t1-t0)
    
    fock = fock_u_batch if unrestricted else fock_restricted
    F = fock(nmol, molecule.molsize, Pconv, M, molecule.maskd, molecule.mask, molecule.idxi, molecule.idxj, w, W,\
                         molecule.parameters['g_ss'], molecule.parameters['g_pp'], molecule.parameters['g_sp'], molecule.parameters['g_p2'], molecule.parameters['h_sp'], molecule.method, molecule.parameters['s_orb_exp_tail'], molecule.parameters['p_orb_exp_tail'], molecule.parameters['d_orb_exp_tail'], molecule.Z, molecule.parameters['F0SD'], molecule.parameters['G2SD'])
        
    Hcore = M.reshape(nmol,molecule.molsize,molecule.molsize,num_orbitals,num_orbitals) \
             .transpose(2,3) \
             .reshape(nmol, num_orbitals*molecule.molsize, num_orbitals*molecule.molsize)
    #
    #return Fock matrix, eigenvalues, density matrix, Hcore,  2 electron 2 center integrals, eigenvectors
    if eig: #
        if scf_backward >= 1:
            if molecule.method == 'PM6':
                e, v = sym_eig_trunc1d(F, molecule.nSuperHeavy, molecule.nHeavy, molecule.nHydro, molecule.nocc, eig_only=True)
            else:
                e, v = sym_eig_trunc1(F, molecule.nHeavy, molecule.nHydro, molecule.nocc, eig_only=True)
        else:
            if molecule.method == 'PM6':
                e, v = sym_eig_truncd(F, molecule.nSuperHeavy, molecule.nHeavy, molecule.nHydro, molecule.nocc, eig_only=True)
            else:
                e, v = sym_eig_trunc( F, molecule.nHeavy, molecule.nHydro, molecule.nocc, eig_only=True)

        #get charge of each orbital on each atom
        charge = torch.zeros(nmol, molecule.molsize*4, molecule.molsize, device=e.device, dtype=e.dtype)
        v2 = [x**2 for x in v]
        norb = 4 * molecule.nHeavy + molecule.nHydro

        if molecule.method != 'PM6': # no orbital chargges for PM6 for now  
            if unrestricted: # open shell ($$$ not sure if orbital charges properly work for open shell. TEST!)
                # $$$
                for i in range(nmol):
                    q1 = v2[i][0,:norb[i],:(4*molecule.nHeavy[i])].reshape(norb[i],4,molecule.nHeavy[i]).sum(dim=1)
                    q1 = q1 + v2[i][1,:norb[i],:(4*molecule.nHeavy[i])].reshape(norb[i],4,molecule.nHeavy[i]).sum(dim=1)
                    charge[i,:norb[i],:molecule.nHeavy[i]] = q1
                    q2 = v2[i][0,:norb[i],(4*molecule.nHeavy[i]):(4*molecule.nHeavy[i]+molecule.nHydro[i])]
                    q2 = q2 + v2[i][1,:norb[i],(4*molecule.nHeavy[i]):(4*molecule.nHeavy[i]+molecule.nHydro[i])]
                    charge[i,:norb[i],molecule.nHeavy[i]:(molecule.nHeavy[i]+molecule.nHydro[i])] = q2
                charge = charge / 2
            else: # closed shell
                for i in range(nmol):
                    charge[i,:norb[i],:molecule.nHeavy[i]] = v2[i][:norb[i],:(4*molecule.nHeavy[i])].reshape(norb[i],4,molecule.nHeavy[i]).sum(dim=1)
                    charge[i,:norb[i],molecule.nHeavy[i]:(molecule.nHeavy[i]+molecule.nHydro[i])] = v2[i][:norb[i],(4*molecule.nHeavy[i]):(4*molecule.nHeavy[i]+molecule.nHydro[i])]
                
        else:
            charge = None
        
        return F, e,    Pconv, Hcore, w, charge, rho0xi, rho0xj, riXH, ri, notconverged, v
    else:
        return F, None, Pconv, Hcore, w, None,   rho0xi, rho0xj, riXH, ri, notconverged, None

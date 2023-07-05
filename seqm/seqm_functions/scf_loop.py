import torch
from torch.autograd import grad as agrad
from .fock import fock
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
#from .check import check
#scf_backward==0: ignore the gradient on density matrix
#scf_backward==1: use recursive formula/implicit autodiff
#scf_backward==2: go backward scf loop directly

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
# tolerance, max no. iteration, and history size for Anderson acceleration
# in solving fixed point problem in SCF_IMPLICIT_BACKWARD
SCF_BACKWARD_ANDERSON_TOLERANCE = 1e-4  # this seems stable enough, but TODO!
SCF_BACKWARD_ANDERSON_MAXITER = 50      # sufficient for all test cases
SCF_BACKWARD_ANDERSON_HISTSIZE = 5      # seems reasonable, but TODO!



# constant mixing
def scf_forward0(M, w, W, gss, gpp, gsp, gp2, hsp, \
                nHydro, nHeavy, nSuperHeavy, nOccMO, \
                nmol, molsize, \
                maskd, mask, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD, sp2=[False], alpha=0.0, backward=False):
    """
    alpha : mixing parameters, alpha=0.0, directly take the new density matrix
    backward is for testing purpose, default is False
    if want to test scf backward directly through the loop in this function, turn backward to be True
    
    INPUT:
    M: core part of Fock matrix
    w: 2c-2e integrals (s,p)
    W: some integrals in PM6. zero in PM3 and PM6_SP.)
    """
    Pnew = torch.zeros_like(P)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp,themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
    if(themethod == 'PM6'):
        Hcore = M.reshape(nmol,molsize,molsize,9,9) \
                 .transpose(2,3) \
                 .reshape(nmol, 9*molsize, 9*molsize)
    else:
        Hcore = M.reshape(nmol,molsize,molsize,4,4) \
                 .transpose(2,3) \
                 .reshape(nmol, 4*molsize, 4*molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)
    for k in range(MAX_ITER+1):
        start_time = time.time()
        if backward:
            if(themethod == 'PM6'):
                Pnew[notconverged] = sym_eig_trunc1d(F[notconverged],
                                                   nSuperHeavy[notconverged],
                                                   nHeavy[notconverged],
                                                   nHydro[notconverged],
                                                   nOccMO[notconverged])[1]
            else:
                e, Pnew[notconverged], v = sym_eig_trunc1(F[notconverged],
                                                        nHeavy[notconverged],
                                                        nHydro[notconverged],
                                                        nOccMO[notconverged])
        elif sp2[0]:
            if(themethod == 'PM6'):
                Pnew[notconverged] = unpackd(
                                        SP2(
                                            packd(F[notconverged],nSuperHeavy[notconverged],  nHeavy[notconverged], nHydro[notconverged]),
                                            nOccMO[notconverged], sp2[1]
                                            ),
                                        nSuperHeavy[notconverged], nHeavy[notconverged], nHydro[notconverged], 9*molsize)
            else:
                Pnew[notconverged] = unpack(
                                        SP2(
                                            pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                            nOccMO[notconverged], sp2[1]
                                            ),
                                        nHeavy[notconverged], nHydro[notconverged], 4*molsize)
        else:
            if(themethod == 'PM6'):
                Pnew[notconverged] = sym_eig_truncd(F[notconverged],
                                                   nSuperHeavy[notconverged],
                                                   nHeavy[notconverged],
                                                   nHydro[notconverged],
                                                   nOccMO[notconverged])[1]
            else:
                e, Pnew[notconverged], v = sym_eig_trunc(F[notconverged],
                                                        nHeavy[notconverged],
                                                        nHydro[notconverged],
                                                        nOccMO[notconverged])
        if backward:
            P = alpha * P + (1.0 - alpha) * Pnew
        else:
            P[notconverged] = alpha * P[notconverged] + (1.0 - alpha) * Pnew[notconverged]
        
        F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
        Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
        err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
        
        Eelec[notconverged] = Eelec_new[notconverged]
        notconverged = err > eps
        if debug:
            end_time = time.time()
            print("scf ", k, torch.max(err).item(), torch.sum(notconverged).item(),  end_time-start_time )
        if not notconverged.any(): break
    return P, notconverged


#use constant mixing, open shell
def scf_forward0_u(M, w, W, gss, gpp, gsp, gp2, hsp, \
                nHydro, nHeavy, nSuperHeavy, nOccMO, \
                nmol, molsize, \
                maskd, mask, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD, sp2=[False], alpha=0.0, backward=False):

    """
    alpha : mixing parameters, alpha=0.0, directly take the new density matrix
    backward is for testing purpose, default is False
    if want to test scf backward directly through the loop in this function, turn backward to be True
    """
    P_ab = torch.zeros_like(P)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    F = fock_u_batch(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp,themethod, zetas, zetap, zetad, Z, F0SD, G2SD)

    if(themethod == 'PM6'):
        Hcore = M.reshape(nmol,molsize,molsize,9,9) \
                 .transpose(2,3) \
                 .reshape(nmol, 9*molsize, 9*molsize)
    else:
        Hcore = M.reshape(nmol, molsize, molsize, 4, 4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)

    for k in range(MAX_ITER+1):
        start_time = time.time()
        if backward:
            if(themethod == 'PM6'):
                raise ValueError('DO NOT use PM6 now')
            else:
                e, P_ab[notconverged], v = sym_eig_trunc1(F[notconverged],
                            nHeavy[notconverged], nHydro[notconverged],
                            nOccMO[notconverged])
            P_ab[notconverged] = P_ab[notconverged] / 2
        elif sp2[0]:
            if(themethod == 'PM6'):
                raise ValueError('DO NOT use PM6 now')
            else:
                P_ab[notconverged] = unpack(
                                        SP2(
                                            pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                            nOccMO[notconverged], sp2[1]
                                            ),
                                        nHeavy[notconverged], nHydro[notconverged], 4*molsize)
        else:
            if(themethod == 'PM6'):
                raise ValueError('DO NOT use PM6 now')
            else:
                e, P_ab[notconverged], v = sym_eig_trunc(F[notconverged],
                            nHeavy[notconverged], nHydro[notconverged],
                            nOccMO[notconverged])
            P_ab[notconverged] = P_ab[notconverged] / 2
        if backward:
            P = alpha * P + (1.0 - alpha) * P_ab
        else:
            P[notconverged] = alpha * P[notconverged] + (1.0 - alpha) * P_ab[notconverged]
            
        F = fock_u_batch(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
        Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
        err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
        if k == 0:
            err_N_steps_back = err.clone()
        Eelec[notconverged] = Eelec_new[notconverged]
        notconverged = err > eps
        if debug:
            end_time = time.time()
            print("scf ", k, torch.max(err).item(), torch.sum(notconverged).item(),  end_time-start_time)
        
        # if (k+1)%5 == 0:
        #     if torch.max(err)/torch.max(err_N_steps_back) > 0.9:
        #         alpha = alpha + (1-alpha)/3
        #     err_N_steps_back = err.clone()


        if not notconverged.any(): break
    return P, notconverged

        
#adaptive mixing
def scf_forward1(M, w, W, gss, gpp, gsp, gp2, hsp, \
                nHydro, nHeavy, nSuperHeavy, nOccMO, \
                nmol, molsize, \
                maskd, mask, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD, sp2=[False], backward=False):
    """
    adaptive mixing algorithm, see cnvg.f
    """
    nDirect1 = 120
    alpha_direct = 0.8
    alpha_direct_increment = (0.95-alpha_direct)/nDirect1
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    
    k = 0
    F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z,  F0SD, G2SD)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    Pnew = torch.zeros_like(P)
    Pold = torch.zeros_like(P)
    if(themethod=='PM6'):
        Hcore = M.reshape(nmol,molsize,molsize,9,9) \
                 .transpose(2,3) \
                 .reshape(nmol, 9*molsize, 9*molsize)
    else:
        Hcore = M.reshape(nmol,molsize,molsize,4,4) \
                 .transpose(2,3) \
                 .reshape(nmol, 4*molsize, 4*molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)
    
    for i in range(nDirect1):
        if notconverged.any():
            if backward:
                if(themethod=='PM6'):
                    raise ValueError('DO NOT use PM6 now')
                else:
                    Pnew[notconverged] = sym_eig_trunc1(F[notconverged],
                                                    nHeavy[notconverged],
                                                    nHydro[notconverged],
                                                    nOccMO[notconverged])[1]
            elif sp2[0]:
                if(themethod=='PM6'):
                    raise ValueError('DO NOT use PM6 now')
                else:
                    Pnew[notconverged] = unpack(
                                            SP2(
                                                pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                                nOccMO[notconverged], sp2[1]
                                                ),
                                            nHeavy[notconverged], nHydro[notconverged], 4*molsize)
            else:
                if(themethod=='PM6'):
                    Pnew[notconverged] = sym_eig_truncd(F[notconverged],
                                                   nSuperHeavy[notconverged],
                                                   nHeavy[notconverged],
                                                   nHydro[notconverged],
                                                   nOccMO[notconverged])[1]
                else:
                    Pnew[notconverged] = sym_eig_trunc(F[notconverged],
                                                   nHeavy[notconverged],
                                                   nHydro[notconverged],
                                                   nOccMO[notconverged])[1]
            if backward:
                Pold = P + 0.0 # ???
                P = alpha_direct * P + (1.0 - alpha_direct) * Pnew
            else:
                Pold[notconverged] = P[notconverged]
                P[notconverged] = alpha_direct * P[notconverged] + (1.0 - alpha_direct) * Pnew[notconverged]
            alpha_direct += alpha_direct_increment
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf direct", k, max_err, Nnot)
            k = k + 1
        else:
            return P, notconverged
    if backward: fac_register = []
    while(1):
        if notconverged.any():
            if backward:
                if(themethod=='PM6'):
                    raise ValueError('DO NOT use PM6 now')
                else:
                    Pnew[notconverged] = sym_eig_trunc1(F[notconverged],
                             nHeavy[notconverged], nHydro[notconverged],
                             nOccMO[notconverged])[1]
            elif sp2[0]:
                if(themethod=='PM6'):
                    raise ValueError('DO NOT use PM6 now')
                else:
                    Pnew[notconverged] = unpack(
                                            SP2(
                                                pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                                nOccMO[notconverged], sp2[1]
                                                ),
                                            nHeavy[notconverged], nHydro[notconverged], 4*molsize)
            else:
                if(themethod=='PM6'):
                    Pnew[notconverged] = sym_eig_truncd(F[notconverged],
                                                   nSuperHeavy[notconverged],
                                                   nHeavy[notconverged],
                                                   nHydro[notconverged],
                                                   nOccMO[notconverged])[1]
                else:
                    Pnew[notconverged] = sym_eig_trunc(F[notconverged],
                             nHeavy[notconverged], nHydro[notconverged],
                             nOccMO[notconverged])[1]
            if backward:
                with torch.no_grad():
                    f = torch.zeros((P.shape[0], 1, 1), dtype=P.dtype, device=P.device)
                    
                    f[notconverged] = torch.sqrt( torch.sum( (   Pnew[notconverged].diagonal(dim1=1,dim2=2)
                                               - P[notconverged].diagonal(dim1=1,dim2=2) \
                                             )**2, dim=1 ) / \
                                  torch.sum( (   Pnew[notconverged].diagonal(dim1=1,dim2=2)
                                              - P[notconverged].diagonal(dim1=1,dim2=2)*2.0
                                              + Pold[notconverged].diagonal(dim1=1,dim2=2)
                                          )**2, dim=1 ) ).reshape(-1,1,1)
                    fac_register.append(f)
            else:
                fac = torch.sqrt( torch.sum( (   Pnew[notconverged].diagonal(dim1=1,dim2=2)
                                           - P[notconverged].diagonal(dim1=1,dim2=2) \
                                         )**2, dim=1 ) / \
                                  torch.sum( (   Pnew[notconverged].diagonal(dim1=1,dim2=2)
                                           - P[notconverged].diagonal(dim1=1,dim2=2)*2.0
                                           + Pold[notconverged].diagonal(dim1=1,dim2=2)
                                         )**2, dim=1 ) ).reshape(-1,1,1)
                #print(fac)
            #
            if backward:
                Pold = P + 0.0  # ???
                P = (1. + fac_register[-1]) * Pnew - fac_register[-1] * P
            else:
                Pold[notconverged] = P[notconverged]
                P[notconverged] = (1. + fac) * Pnew[notconverged] - fac * P[notconverged]
            
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf adapt", k, max_err, Nnot)
            k = k + 1
            if k >= MAX_ITER: return P, notconverged
        else:
            return P, notconverged

def scf_forward1_u(M, w, W, gss, gpp, gsp, gp2, hsp, \
                nHydro, nHeavy, nSuperHeavy, nOccMO, \
                nmol, molsize, \
                maskd, mask, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD, sp2=[False], backward=False):
    """
    adaptive mixing algorithm, see cnvg.f
    """
    nDirect1 = 60
    alpha_direct = 0.6
    alpha_direct_increment = (0.91-alpha_direct)/nDirect1
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    
    k = 0
    F = fock_u_batch(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp,themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    ###Pnew = torch.zeros_like(P)
    P_ab_new = torch.zeros_like(P)
    P_ab_old = torch.zeros_like(P)
    if(themethod=='PM6'):
        Hcore = M.reshape(nmol,molsize,molsize,9,9) \
                 .transpose(2,3) \
                 .reshape(nmol, 9*molsize, 9*molsize)
    else:
        Hcore = M.reshape(nmol,molsize,molsize,4,4) \
                 .transpose(2,3) \
                 .reshape(nmol, 4*molsize, 4*molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)
    
    for i in range(nDirect1):
        if notconverged.any():
            if backward:
                if(themethod=='PM6'):
                    raise ValueError('DO NOT use PM6 now')
                else:
                    P_ab_new[notconverged] = sym_eig_trunc1(F[notconverged],
                                                    nHeavy[notconverged],
                                                    nHydro[notconverged],
                                                    nOccMO[notconverged])[1]
                    P_ab_new[notconverged] = P_ab_new[notconverged] / 2
            elif sp2[0]:
                if(themethod=='PM6'):
                    raise ValueError('DO NOT use PM6 now')
                else:
                    P_ab_new[notconverged] = unpack(
                                            SP2(
                                                pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                                nOccMO[notconverged], sp2[1]
                                                ),
                                            nHeavy[notconverged], nHydro[notconverged], 4*molsize)
            else:
                if(themethod=='PM6'):
                    raise ValueError('DO NOT use PM6 now')
                else:
                    P_ab_new[notconverged] = sym_eig_trunc(F[notconverged],
                                                   nHeavy[notconverged],
                                                   nHydro[notconverged],
                                                   nOccMO[notconverged])[1]
                    P_ab_new[notconverged] = P_ab_new[notconverged] / 2
            if backward:
                P_ab_old = P + 0.0 # ???
                P = alpha_direct * P + (1.0 - alpha_direct) * P_ab_new

            else:
                P_ab_old[notconverged] = P[notconverged]
                #P[notconverged] = P_ab_new[notconverged]
                P[notconverged] = alpha_direct * P[notconverged] + (1.0 - alpha_direct) * P_ab_new[notconverged]
            if i >= 7:
                alpha_direct += alpha_direct_increment
            F = fock_u_batch(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf direct ", k, max_err, Nnot)
            k = k + 1
        else:
            return P, notconverged
    if backward: fac_register = []
    init_fac = True
    while(1):
        if notconverged.any():
            if backward:
                if(themethod=='PM6'):
                    raise ValueError('DO NOT use PM6 now')
                else:
                    P_ab_new[notconverged] = sym_eig_trunc1(F[notconverged],
                             nHeavy[notconverged], nHydro[notconverged],
                             nOccMO[notconverged])[1]
                    P_ab_new[notconverged] = P_ab_new[notconverged] / 2
            elif sp2[0]:
                if(themethod=='PM6'):
                    raise ValueError('DO NOT use PM6 now')
                else:
                    P_ab_new[notconverged] = unpack(
                                            SP2(
                                                pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                                nOccMO[notconverged], sp2[1]
                                                ),
                                            nHeavy[notconverged], nHydro[notconverged], 4*molsize)
            else:
                if(themethod=='PM6'):
                    raise ValueError('DO NOT use PM6 now')
                else:
                    P_ab_new[notconverged] = sym_eig_trunc(F[notconverged],
                             nHeavy[notconverged], nHydro[notconverged],
                             nOccMO[notconverged])[1]
                    P_ab_new[notconverged] = P_ab_new[notconverged] / 2
            if backward:
                with torch.no_grad():
                    f = torch.zeros((P.shape[0], 2, 1, 1), dtype=P.dtype, device=P.device)
                    
                    f[notconverged] = torch.sqrt( torch.sum( (   P_ab_new[notconverged].diagonal(dim1=2,dim2=3)
                                               - P[notconverged].diagonal(dim1=2,dim2=3) \
                                             )**2, dim=2 ) / \
                                  torch.sum( (   P_ab_new[notconverged].diagonal(dim1=2,dim2=3)
                                              - P[notconverged].diagonal(dim1=2,dim2=3)*2.0
                                              + P_ab_old[notconverged].diagonal(dim1=2,dim2=3)
                                          )**2, dim=2 ) ).reshape(-1,2,1,1)
                    fac_register.append(f)
            else:
                fac = torch.sqrt( torch.sum( (   P_ab_new[notconverged].diagonal(dim1=2,dim2=3)
                                           - P[notconverged].diagonal(dim1=2,dim2=3) \
                                         )**2, dim=2 ) / \
                                  torch.sum( (   P_ab_new[notconverged].diagonal(dim1=2,dim2=3)
                                           - P[notconverged].diagonal(dim1=2,dim2=3)*2.0
                                           + P_ab_old[notconverged].diagonal(dim1=2,dim2=3)
                                         )**2, dim=2 ) ).reshape(-1,2,1,1)
            #
            if backward:
                P_ab_old = P + 0.0  # ???
                P = (1. + fac_register[-1]) * P_ab_new - fac_register[-1] * P
                print(fac_register[-1])
            else:
                P_ab_old[notconverged] = P[notconverged]
                # if init_fac:
                #     fac = fac*0.0 + 0.9
                #     init_fac = False
                P[notconverged] = ((1. + fac) * P_ab_new[notconverged] - fac * P[notconverged])
                #P[notconverged] = (1. - fac) * P_ab_new[notconverged] + fac * P[notconverged]
            
            F = fock_u_batch(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf adaptive ", k, max_err, Nnot)
            k = k + 1
            if k >= MAX_ITER: return P, notconverged
        else:
            return P, notconverged

#adaptive mixing, pulay
def scf_forward2(M, w, W, gss, gpp, gsp, gp2, hsp, \
                nHydro, nHeavy, nSuperHeavy, nOccMO, \
                nmol, molsize, \
                maskd, mask, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD, sp2=[False], backward=False):

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

    nDirect1 = 50
    alpha_direct = 0.9

    nAdapt = 1
    # number of maximal fock matrixes used
    nFock = 8

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
    if(themethod == 'PM6'):
        FPPF = torch.zeros(nmol, nFock, molsize*9, molsize*9, dtype=dtype,device=device)
    else:
        FPPF = torch.zeros(nmol, nFock, molsize*4, molsize*4, dtype=dtype,device=device)
        
    EMAT = (torch.eye(nFock+1, nFock+1, dtype=dtype, device=device) - 1.0).expand(nmol,nFock+1,nFock+1).tril().clone()
    EVEC = torch.zeros_like(EMAT) # EVEC is <E(i)*E(j)> scaled by a constant
    FOCK = torch.zeros_like(FPPF) # store last n=nFock number of Fock matrixes
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    k = 0
    F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    Pnew = torch.zeros_like(P)
    Pold = torch.zeros_like(P)
    if(themethod == 'PM6'):
        Hcore = M.reshape(nmol,molsize,molsize,9,9) \
                 .transpose(2,3) \
                 .reshape(nmol, 9*molsize, 9*molsize)
    else:
        Hcore = M.reshape(nmol,molsize,molsize,4,4) \
                 .transpose(2,3) \
                 .reshape(nmol, 4*molsize, 4*molsize)
        
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)

    for i in range(nDirect1):

        if notconverged.any():
            if backward:
                if(themethod == 'PM6'):
                    Pnew[notconverged] = sym_eig_trunc1d(F[notconverged],
                                                       nSuperHeavy[notconverged],
                                                       nHeavy[notconverged],
                                                       nHydro[notconverged],
                                                       nOccMO[notconverged])[1]
                else:
                    Pnew[notconverged] = sym_eig_trunc1(F[notconverged],
                                                       nHeavy[notconverged],
                                                       nHydro[notconverged],
                                                       nOccMO[notconverged])[1]
            elif sp2[0]:
                if(themethod == 'PM6'):                     
                    Pnew[notconverged] = unpackd(
                                                SP2(
                                                    packd(F[notconverged], nSuperHeavy[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                                    nOccMO[notconverged], sp2[1]
                                                    ),
                                                nSuperHeavy[notconverged],nHeavy[notconverged], nHydro[notconverged], 9*molsize)
                else:
                    Pnew[notconverged] = unpack(
                                                SP2(
                                                    pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                                    nOccMO[notconverged], sp2[1]
                                                    ),
                                                nHeavy[notconverged], nHydro[notconverged], 4*molsize)
            else:
                if(themethod == 'PM6'):
                    Pnew[notconverged] = sym_eig_truncd(F[notconverged],
                                                       nSuperHeavy[notconverged],
                                                       nHeavy[notconverged],
                                                       nHydro[notconverged],
                                                       nOccMO[notconverged])[1]
                else:
                    Pnew[notconverged] = sym_eig_trunc(F[notconverged],
                                                       nHeavy[notconverged],
                                                       nHydro[notconverged],
                                                       nOccMO[notconverged])[1]
            if backward:
                #P=Pnew.clone()
                Pold = P+0.0
                P = Pnew+0.0
            else:
                Pold[notconverged] = P[notconverged]
                #P[notconverged] = Pnew[notconverged]
                P[notconverged] = alpha_direct * P[notconverged] + (1.0 - alpha_direct) * Pnew[notconverged]

            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf direct", k, max_err, Nnot)
            k = k + 1
        else:
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
    if backward:
        fac_register = []
    for i in range(nAdapt):
        if notconverged.any():

            if backward:
                if(themethod == 'PM6'):
                    Pnew[notconverged] = sym_eig_trunc1d(F[notconverged],
                                                       nSuperHeavy[notconverged],
                                                       nHeavy[notconverged],
                                                       nHydro[notconverged],
                                                       nOccMO[notconverged])[1]
                else:
                    Pnew[notconverged] = sym_eig_trunc1(F[notconverged],
                                                       nHeavy[notconverged],
                                                       nHydro[notconverged],
                                                       nOccMO[notconverged])[1]
            elif sp2[0]:
                if(themethod == 'PM6'):
                    Pnew[notconverged] = unpackd(
                                                SP2(
                                                    packd(F[notconverged], nSuperHeavy[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                                    nOccMO[notconverged], sp2[1]
                                                    ),
                                                nSuperHeavy[notconverged], nHeavy[notconverged], nHydro[notconverged], 9*molsize)
                else:
                    Pnew[notconverged] = unpack(
                                                SP2(
                                                    pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                                    nOccMO[notconverged], sp2[1]
                                                    ),
                                                nHeavy[notconverged], nHydro[notconverged], 4*molsize)
            else:
                if(themethod == 'PM6'):
                    Pnew[notconverged] = sym_eig_truncd(F[notconverged],
                                                   nSuperHeavy[notconverged],
                                                   nHeavy[notconverged],
                                                   nHydro[notconverged],
                                                   nOccMO[notconverged])[1]
                else:
                    Pnew[notconverged] = sym_eig_trunc(F[notconverged],
                                                   nHeavy[notconverged],
                                                   nHydro[notconverged],
                                                   nOccMO[notconverged])[1]
                    
            #fac = sqrt( \sum_i (P_ii^(k) - P_ii^(k-1))**2 / \sum_i (P_ii^(k) - 2*P_ii^(k-1) + P_ii^(k-2))**2 )
            if backward:
                with torch.no_grad():
                    f = torch.zeros((P.shape[0], 1, 1), dtype=P.dtype, device=P.device)
                    f[notconverged] = torch.sqrt( torch.sum( (   Pnew[notconverged].diagonal(dim1=1,dim2=2)
                                               - P[notconverged].diagonal(dim1=1,dim2=2) \
                                             )**2, dim=1 ) / \
                                  torch.sum( (   Pnew[notconverged].diagonal(dim1=1,dim2=2)
                                              - P[notconverged].diagonal(dim1=1,dim2=2)*2.0
                                              + Pold[notconverged].diagonal(dim1=1,dim2=2)
                                          )**2, dim=1 ) ).reshape(-1,1,1)
                    fac_register.append(f)
            else:
                fac = torch.sqrt( torch.sum( (   Pnew[notconverged].diagonal(dim1=1,dim2=2)
                                           - P[notconverged].diagonal(dim1=1,dim2=2) \
                                         )**2, dim=1 ) / \
                                  torch.sum( (   Pnew[notconverged].diagonal(dim1=1,dim2=2)
                                          - P[notconverged].diagonal(dim1=1,dim2=2)*2.0
                                          + Pold[notconverged].diagonal(dim1=1,dim2=2)
                                      )**2, dim=1 ) ).reshape(-1,1,1)
                
            if backward:
                Pold = P+0.0
                P = (1.0+fac_register[-1])*Pnew - fac_register[-1]*P
            else:
                Pold[notconverged] = P[notconverged]
                P[notconverged] = (1.0+fac)*Pnew[notconverged] - fac*P[notconverged]
            
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf adaptive", k, max_err, Nnot)
            k = k + 1
        else:
            return P, notconverged
    del Pold, Pnew

    #start prepare for pulay algorithm
    counter = -1 # index of stored FPPF for current iteration: 0, 1, ..., cFock-1
    cFock = 0 # in current iteraction, number of fock matrixes stored, cFock <= nFock
    #Pulay algorithm needs at least two previous stored density and Fock matrixes to start
    while (cFock<2):

        if notconverged.any():
            cFock = cFock + 1 if cFock < nFock else nFock
            #store fock matrix
            counter = (counter + 1)%nFock
            FOCK[notconverged, counter, :, :] = F[notconverged]
            FPPF[notconverged, counter, :, :] = (F[notconverged].matmul(P[notconverged]) - P[notconverged].matmul(F[notconverged])).triu()
            # in mopac7 interp.f pulay subroutine, only lower triangle of FPPF is used to construct EMAT
            #only compute lower triangle as Emat are symmetric
            EMAT[notconverged, counter, :cFock] = torch.sum(FPPF[notconverged, counter:(counter+1),:,:] * FPPF[notconverged,:cFock,:,:], dim=(2,3))
            if backward:
                if(themethod == 'PM6'):
                    P[notconverged] = sym_eig_trunc1d(F[notconverged],
                                                       nSuperHeavy[notconverged],
                                                       nHeavy[notconverged],
                                                       nHydro[notconverged],
                                                       nOccMO[notconverged])[1]
                else:
                    P[notconverged] = sym_eig_trunc1(F[notconverged],
                                                       nHeavy[notconverged],
                                                       nHydro[notconverged],
                                                       nOccMO[notconverged])[1]
            elif sp2[0]:
                if(themethod == 'PM6'):
                    P[notconverged] = unpackd(
                                                SP2(
                                                    packd(F[notconverged], nSuperHeavy[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                                    nOccMO[notconverged], sp2[1]
                                                    ),
                                                nSuperHeavy[notconverged], nHeavy[notconverged], nHydro[notconverged], 9*molsize)
                else:
                    P[notconverged] = unpack(
                                                SP2(
                                                    pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                                    nOccMO[notconverged], sp2[1]
                                                    ),
                                                nHeavy[notconverged], nHydro[notconverged], 4*molsize)
            else:
                if(themethod == 'PM6'):
                    P[notconverged] = sym_eig_truncd(F[notconverged],
                                                       nSuperHeavy[notconverged],
                                                       nHeavy[notconverged],
                                                       nHydro[notconverged],
                                                       nOccMO[notconverged])[1]
                else:
                    P[notconverged] = sym_eig_trunc(F[notconverged],
                                                       nHeavy[notconverged],
                                                       nHydro[notconverged],
                                                       nOccMO[notconverged])[1]
                    
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            if debug:
                print("scf pulay", k, torch.max(err).item(), torch.sum(notconverged).item())
            k = k + 1
        else:
            return P, notconverged
    
    #start pulay algorithm
    while(1):

        if notconverged.any():
            EVEC[notconverged] = EMAT[notconverged] + EMAT[notconverged].tril(-1).transpose(1,2)
#            EVEC[notconverged,:cFock,:cFock] /= EVEC[notconverged,counter:(counter+1),counter:(counter+1)]
            # work-around for in-place operation (more elegant solution?)
            EVcF = EVEC[notconverged,:cFock,:cFock].clone()
            EVnorm = EVEC[notconverged,counter:(counter+1),counter:(counter+1)].clone()
            EVEC[notconverged,:cFock,:cFock] = EVcF / EVnorm
            coeff = -torch.inverse(EVEC[notconverged,:(cFock+1),:(cFock+1)])[...,:-1,-1]
            F[notconverged] = torch.sum(FOCK[notconverged,:cFock,:,:]*coeff.unsqueeze(-1).unsqueeze(-1), dim=1)
            if backward:
                if(themethod == 'PM6'):
                    P[notconverged] = sym_eig_trunc1d(F[notconverged],
                                                       nSuperHeavy[notconverged],
                                                       nHeavy[notconverged],
                                                       nHydro[notconverged],
                                                       nOccMO[notconverged])[1]
                else:
                    P[notconverged] = sym_eig_trunc1(F[notconverged],
                                                       nHeavy[notconverged],
                                                       nHydro[notconverged],
                                                       nOccMO[notconverged])[1]
            elif sp2[0]:
                if(themethod == 'PM6'):
                    P[notconverged] = unpackd(
                                                SP2(
                                                    packd(F[notconverged], nSuperHeavy[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                                    nOccMO[notconverged], sp2[1]
                                                    ),
                                                nSuperHeavy[notconverged], nHeavy[notconverged], nHydro[notconverged], 9*molsize)
                else:
                    P[notconverged] = unpack(
                                                SP2(
                                                    pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                                    nOccMO[notconverged], sp2[1]
                                                    ),
                                                nHeavy[notconverged], nHydro[notconverged], 4*molsize)
            else:
                if(themethod == 'PM6'):
                    P[notconverged] = sym_eig_truncd(F[notconverged],
                                                       nSuperHeavy[notconverged],
                                                       nHeavy[notconverged],
                                                       nHydro[notconverged],
                                                       nOccMO[notconverged])[1]
                else:
                    P[notconverged] = sym_eig_trunc(F[notconverged],
                                                       nHeavy[notconverged],
                                                       nHydro[notconverged],
                                                       nOccMO[notconverged])[1]
            #
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
            
            cFock = cFock + 1 if cFock < nFock else nFock
            counter = (counter + 1)%nFock
            FOCK[notconverged,counter,:,:] = F[notconverged]
            FPPF[notconverged,counter,:,:] = (F[notconverged].matmul(P[notconverged]) - P[notconverged].matmul(F[notconverged])).triu()
            # in mopac7 interp.f pulay subroutine, only lower triangle of FPPF is used to construct EMAT
            #only compute lower triangle as Emat are symmetric
            EMAT[notconverged,counter,:cFock] = torch.sum(FPPF[notconverged, counter:(counter+1),:,:]*FPPF[notconverged,:cFock,:,:], dim=(2,3))
            
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf pulay", k, max_err, Nnot)
            k = k + 1
            if k >= MAX_ITER: return P, notconverged
        else:
            return P, notconverged

        
def scf_forward3(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO, 
                 nmol, molsize, maskd, mask, idxi, idxj, P, eps, 
                 xl_bomd_params, backward=False):
    """
    DM scf optimization using KSA
    $$$ probably, not properly optimized for batches. 
    backward is for testing purpose, default is False
    if want to test scf backward directly through the loop in this function, turn backward to be True
    """
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    Hcore = M.reshape(nmol,molsize,molsize,4,4) \
         .transpose(2,3) \
         .reshape(nmol, 4*molsize, 4*molsize)
    
    Temp = xl_bomd_params['T_el']
    kB = 8.61739e-5 # eV/K, kB = 6.33366256e-6 Ry/K, kB = 3.166811429e-6 Ha/K, #kB = 3.166811429e-6 #Ha/K
    SCF_err = torch.tensor([1.0], dtype=P.dtype, device=P.device)
    COUNTER = 0
    
    Rank = xl_bomd_params['max_rank']
    V = torch.zeros((P.shape[0], P.shape[1], P.shape[2], Rank), dtype=P.dtype, device=P.device)
    W = torch.zeros((P.shape[0], P.shape[1], P.shape[2], Rank), dtype=P.dtype, device=P.device)
    
    K0 = 1.0
    F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    D, S_Ent, QQ, e, Fe_occ, mu0, Occ_mask = Fermi_Q(F, Temp, nOccMO, nHeavy, nHydro, kB, scf_backward=0)
    dDS = K0 * (D - P)
    dW = dDS
    
    Eelec = torch.zeros((nmol), dtype=P.dtype, device=P.device)
    Eelec_new = torch.zeros_like(Eelec)
    if debug: print("step, DM rmse, dE, number of not converged")
    
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
                FO1 = G(nmol, molsize, d_D, M, maskd, mask, idxi, idxj, w, \
                        gss=gss, gpp=gpp, gsp=gsp, gp2=gsp, hsp=gsp)
                
                PO1 = Canon_DM_PRT(FO1, Temp, nHeavy, nHydro, QQ, e, mu0, 8, kB, Occ_mask)
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
            
            for I in range(0, Rank_m):
                for J in range(0, Rank_m):
                    P[notconverged] = P[notconverged] - \
                            MM[notconverged,I,J].view(-1, 1, 1) *torch.sum(W[notconverged,:,:,J].transpose(1,2)*dDS[notconverged], dim=(1,2)).view(-1, 1, 1) * V[notconverged,:,:,I]
            
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged] - Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            SCF_err = torch.linalg.norm(dDS[notconverged], ord='fro', dim=(1,2))
            if debug:
                end_time = time.time()
                print(COUNTER, SCF_err.cpu().numpy(), err.cpu().numpy(), torch.sum(notconverged).item(), end_time - start_time)
            if COUNTER >= MAX_ITER: return P, notconverged
        else:
            return P, notconverged
        

def fixed_point_anderson(fp_fun, u0, lam=1e-4, beta=1.0):
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
    for k in range(2, SCF_BACKWARD_ANDERSON_MAXITER):
        n = min(k, m)
        G = F[:,:n] - X[:,:n]
        GTG = torch.bmm(G, G.transpose(1,2))
        E_n = torch.eye(n, dtype=u0.dtype, device=u0.device)
        H[:,1:n+1,1:n+1] = GTG + lam * E_n[None]
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:,1:n+1,0]
        Xold = (1 - beta) * (alpha[:,None] @ X[:,:n])[:,0]
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + Xold
        F[:,k%m] = fp_fun(X[:,k%m].view_as(u0)).view(nmol, -1)
        resid1 = (F[:,k%m] - X[:,k%m]).norm().item()
        resid = resid1 / (1e-5 + F[:,k%m].norm().item())
        cond = resid < SCF_BACKWARD_ANDERSON_TOLERANCE
        if cond: break   # solver converged
    if not cond:
        msg = "Anderson solver in SCF backward did not converge for some molecule(s)"
        if RAISE_ERROR_IF_SCF_BACKWARD_FAILS: raise ValueError(msg)
        print(msg)
    u_conv = X[:,k%m].view_as(u0)
    return u_conv
    

class SCF(torch.autograd.Function):
    """
    scf loop
    forward and backward
    check function scf_loop for details
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
                maskd, mask, atom_molid, pair_molid, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD):
        
        SCF.themethod = themethod
        if SCF.converger[0] == 0:
            if P.dim() == 4:
                P, notconverged = scf_forward0_u(M, w, W, gss, gpp, gsp, gp2, hsp, \
                                   nHydro, nHeavy, nSuperHeavy, nOccMO, \
                                   nmol, molsize, \
                                   maskd, mask, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD, sp2=SCF.sp2, alpha=SCF.converger[1])
            else:
                P, notconverged = scf_forward0(M, w, W, gss, gpp, gsp, gp2, hsp, \
                                   nHydro, nHeavy, nSuperHeavy, nOccMO, \
                                   nmol, molsize, \
                                   maskd, mask, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD, sp2=SCF.sp2, alpha=SCF.converger[1])
        elif SCF.converger[0] == 3: # KSA
            P, notconverged = scf_forward3(M, w, gss, gpp, gsp, gp2, hsp,
                                   nHydro, nHeavy, nOccMO, nmol, molsize,
                                   maskd, mask, idxi, idxj, P, eps, SCF.converger[1])
        else:
            if SCF.converger[0] == 1: # adaptive mixing
                if P.dim() == 4:
                    scf_forward = scf_forward1_u
                else:
                    scf_forward = scf_forward1
            elif SCF.converger[0] == 2: # adaptive mixing, then pulay
                scf_forward = scf_forward2
            P, notconverged = scf_forward(M, w, W, gss, gpp, gsp, gp2, hsp, \
                           nHydro, nHeavy, nSuperHeavy, nOccMO, \
                           nmol, molsize, \
                           maskd, mask, idxi, idxj, P, eps, themethod, zetas, zetap, zetad, Z, F0SD, G2SD, sp2=SCF.sp2)
        eps = torch.as_tensor(eps, dtype=M.dtype, device=M.device)
        ctx.save_for_backward(P, M, w, W, gss, gpp, gsp, gp2, hsp, \
                              nHydro, nHeavy, nSuperHeavy, nOccMO, \
                              maskd, mask, idxi, idxj, eps, zetas, zetap, zetad, Z, F0SD, G2SD, notconverged, \
                              atom_molid, pair_molid)

        return P, notconverged


    @staticmethod
    def backward(ctx, grad_P, grad1):
        """
        custom backward of SCF loop
        
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
        for i, st in enumerate([M, w, gss, gpp, gsp, gp2, hsp]):
            if st.requires_grad:
                gv.append(st)
                gvind.append(i+1)
            else:
                grads[i+1] = None
        with torch.enable_grad():
            Pin.requires_grad_(True)
            if Pin.dim() == 4:
                F = fock_u_batch(nmol, molsize, Pin, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp,\
                                 themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
                
                if(themethod == 'PM6'):
                    raise ValueError('DO NOT use PM6 now')
                else:
                    Pout = sym_eig_trunc1(F, nHeavy, nHydro, nOccMO)[1] / 2
            else:
                F = F = fock(nmol, molsize, Pin, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp,\
                             themethod, zetas, zetap, zetad, Z, F0SD, G2SD)
                
                if(themethod == 'PM6'):
                    raise ValueError('DO NOT use PM6 now')
                else:
                    Pout = sym_eig_trunc1(F, nHeavy, nHydro, nOccMO)[1]
        backward_eps = SCF.scf_backward_eps.to(Pin.device)
        converged = ~notconverged.detach() # scf forward converged
        diverged = None # scf backward diverged
        
        if SCF_IMPLICIT_BACKWARD:
            ## THIS DOES NOT SUPPORT DOUBLE BACKWARD. MAY AS WELL STOP AUTOGRAD TAPE
            ## TODO: INCLUDING THIS PART IN GRAPH CAUSES MEMORY LEAK.
            ##       RESOLVE THIS WHEN IMPLEMENTING DOUBLE-BACKWARD!
            with torch.no_grad():
                def affine_eq(u): return grad_P + agrad(Pout, Pin, grad_outputs=u, retain_graph=True)[0]
                u_init = torch.zeros_like(Pin)  #TODO: better initial guess?
                u = fixed_point_anderson(affine_eq, u_init)
                gradients = agrad(Pout, gv, grad_outputs=u, retain_graph=True)
                for t, i in enumerate(gvind): grads[i] = gradients[t]
        else:
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
               None, None, None, \
               None, None, \
               None, None, None, None, None, None, None, None, \
               None, None, None, None, None, None, None, None, None
        
    

class SCF0(SCF):
    @staticmethod
    def backward(ctx, grad_P, grad1):
        # ignore gradient on density matrix and eigenvectors/-values
        return None, None, None, None, None, None, None, \
               None, None, None, \
               None, None, \
               None, None, None, None, None, None, None, None, \
               None, None, None, None, None, None, None, None, None


def scf_loop(const, molsize, \
            nSuperHeavy, nHeavy, nHydro, nOccMO, \
            maskd, mask, atom_molid, pair_molid, idxi, idxj, ni,nj,xij,rij, Z, \
            zetas, zetap, zetad, zs, zp, zd, uss, upp , udd,  gss, gsp, gpp, gp2, hsp, F0SD, G2SD, rho_core, alpha, chi, themethod, beta, \
            Kbeta=None, eps = 1.0e-4, P=None, sp2=[False], scf_converger=[1], eig=False, scf_backward=0, scf_backward_eps=1.0e-2):
    """
    SCF loop
    # check hcore.py for the details of arguments
    eps : convergence criteria for density matrix on density matrix
    P : if provided, will be used as initial density matrix in scf loop
    return : F, e, P, Hcore, w, v
    """
    device = xij.device
    nmol = nHeavy.shape[0]
    tore = const.tore
    if const.do_timing: t0 = time.time()
    M, w,rho0xi,rho0xj = hcore(const, nmol, molsize, maskd, mask, idxi, idxj, ni,nj,xij,rij, Z, \
                     zetas,zetap, zetad, zs, zp, zd,  uss, upp , udd, gss, gpp, gp2, hsp,F0SD, G2SD, rho_core, alpha, chi, themethod,  beta, Kbeta=Kbeta)

    if const.do_timing:
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.time()
        const.timing["Hcore + STO Integrals"].append(t1 - t0)
        t0 = time.time()
    #if scf_backward == 2 or (not torch.is_tensor(P)):
    if (not torch.is_tensor(P)): # $$$ I'm not sure if it is okay to use DM initialized from make_dm_guess which corresponds to 1 SCF iteration + HOMO-LUMO mix.
        if(themethod == 'PM6'):
            P0 = torch.zeros_like(M)  # density matrix
            P0[maskd[Z>1],0,0] = tore[Z[Z>1]]/4.0
            P0[maskd,1,1] = P0[maskd,0,0]
            P0[maskd,2,2] = P0[maskd,0,0]
            P0[maskd,3,3] = P0[maskd,0,0]

            P0[maskd[Z==1],0,0] = 1.0
        #P0 += torch.randn(P0.shape,dtype=P0.dtype, device=P0.device)*0.01
            P = P0.reshape(nmol,molsize,molsize,9,9) \
                .transpose(2,3) \
                .reshape(nmol, 9*molsize, 9*molsize)
            ##print(P.shape)
            ##sys.exit()


        else:
            P0 = torch.zeros_like(M)  # density matrix
            P0[maskd[Z>1],0,0] = tore[Z[Z>1]]/4.0
            P0[maskd,1,1] = P0[maskd,0,0]
            P0[maskd,2,2] = P0[maskd,0,0]
            P0[maskd,3,3] = P0[maskd,0,0]
            P0[maskd[Z==1],0,0] = 1.0
            P = P0.reshape(nmol,molsize,molsize,4,4) \
                .transpose(2,3) \
                .reshape(nmol, 4*molsize, 4*molsize)
        
        if nOccMO.dim() == 2: # alpha and beta dm for open shell
            #print('DOING UHF!!!!!!')
            P = torch.stack((0.55*P, 0.45*P), dim=1)
    #print('GRAD P',P.requires_grad)
    if(themethod == 'PM6'): # PM6 does not work. ignore this part
        if nOccMO.dim() == 2: # open shell
            
            W, W_exch = calc_integral_os(zs, zp, zd, Z, nmol*molsize*molsize, maskd, P, F0SD, G2SD)
            W = torch.stack((W, W_exch))
            #print(W_exch)
        else:
            W = calc_integral(zs, zp, zd, Z, nmol*molsize*molsize, maskd, P, F0SD, G2SD)
            W_exch = torch.tensor([0], device=nOccMO.device)
    else:
        W = torch.tensor([0], device=nOccMO.device)
        W_exch = torch.tensor([0], device=nOccMO.device)
    
    #"""
    #scf_backward == 2, directly backward through scf loop
    #             can't reuse P, so put P=None and initial P above
    #"""
    if scf_backward == 2:
        if sp2[0]:
            warnings.warn('SP2 is not used for direct backpropagation through scf loop')
            sp2[0] = False
        if scf_converger[0] == 0:
            if P.dim() == 4:
                Pconv, notconverged =  scf_forward0_u(M, w, W, gss, gpp, gsp, gp2, hsp, \
                         nHydro, nHeavy, nSuperHeavy, nOccMO, \
                         nmol, molsize, \
                         maskd, mask, idxi, idxj, P, eps, themethod, zs, zp, zd, Z, F0SD, G2SD, sp2=sp2, alpha=scf_converger[1], backward=True)
            else:
                Pconv, notconverged =  scf_forward0  (M, w, W, gss, gpp, gsp, gp2, hsp, \
                         nHydro, nHeavy, nSuperHeavy, nOccMO, \
                         nmol, molsize, \
                         maskd, mask, idxi, idxj, P, eps, themethod, zs, zp, zd, Z, F0SD, G2SD, sp2=sp2, alpha=scf_converger[1], backward=True)
        elif scf_converger[0] == 1:
            if P.dim() == 4:
                Pconv, notconverged =  scf_forward1_u(  M, w, W, gss, gpp, gsp, gp2, hsp, \
                         nHydro, nHeavy, nSuperHeavy, nOccMO, \
                         nmol, molsize, \
                         maskd, mask, idxi, idxj, P, eps, themethod, zs, zp, zd, Z, F0SD, G2SD, sp2=sp2, backward=True)
            else:
                Pconv, notconverged =  scf_forward1(  M, w, W, gss, gpp, gsp, gp2, hsp, \
                         nHydro, nHeavy, nSuperHeavy, nOccMO, \
                         nmol, molsize, \
                         maskd, mask, idxi, idxj, P, eps, themethod, zs, zp, zd, Z, F0SD, G2SD, sp2=sp2, backward=True)
        else:
            raise ValueError("""For direct backpropagation through scf,
                                must use constant mixing at this moment\n
                                set scf_converger=[0, alpha] or [1]\n""")
    #scf_backward 1, use recursive formula/implicit autodiff
    if scf_backward == 1:
        scfapply = SCF(use_sp2=sp2, scf_converger=scf_converger, scf_backward_eps=scf_backward_eps).apply
    #scf_backward 0: ignore the gradient on density matrix
    elif scf_backward == 0:
        scfapply = SCF0(use_sp2=sp2, scf_converger=scf_converger).apply

    # apply_params = {k:pp[k] for k in apply_param_map}
    if scf_backward==0 or scf_backward==1:
        Pconv, notconverged = scfapply(M, w, W, gss, gpp, gsp, gp2, hsp, \
            nHydro, nHeavy, nSuperHeavy, nOccMO, \
            nmol, molsize, \
            maskd, mask, atom_molid, pair_molid, idxi, idxj, P, eps, themethod, zs, zp, zd, Z, F0SD, G2SD )
    if notconverged.any():
        nnot = notconverged.type(torch.int).sum().data.item()
        print('did not converge', nnot)
        warnings.warn("SCF for %d/%d molecules doesn't converge after %d iterations" % (nnot, nmol, MAX_ITER))
        if RAISE_ERROR_IF_SCF_FORWARD_FAILS:
            raise ValueError("SCF for some the molecules in the batch doesn't converge")

    if const.do_timing:
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.time()
        const.timing["SCF"].append(t1-t0)
    
    if Pconv.dim() == 4:
        F = fock_u_batch(nmol, molsize, Pconv, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zs, zp, zd, Z, F0SD, G2SD)
    else:
        F = fock        (nmol, molsize, Pconv, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zs, zp, zd, Z, F0SD, G2SD)
        
    if(themethod == 'PM6'):
         Hcore = M.reshape(nmol,molsize,molsize,9,9) \
                 .transpose(2,3) \
                 .reshape(nmol, 9*molsize, 9*molsize)
    else:
         Hcore = M.reshape(nmol,molsize,molsize,4,4) \
                 .transpose(2,3) \
                 .reshape(nmol, 4*molsize, 4*molsize)
    #
    #return Fock matrix, eigenvalues, density matrix, Hcore,  2 electron 2 center integrals, eigenvectors
    if eig and themethod != 'PM6': #
        if scf_backward >= 1:
            e, v = sym_eig_trunc1(F, nHeavy, nHydro, nOccMO, eig_only=True)
        else:
            e, v = sym_eig_trunc(F, nHeavy, nHydro, nOccMO, eig_only=True)

        #get charge of each orbital on each atom
        charge = torch.zeros(nmol, molsize*4, molsize, device=e.device, dtype=e.dtype)
        v2 = [x**2 for x in v]
        norb = 4 * nHeavy + nHydro
                
        if F.dim() == 4: # open shell ($$$ not sure if orbital charges properly work for open shell. TEST!)
            # $$$
            for i in range(nmol):
                q1 = v2[i][0,:norb[i],:(4*nHeavy[i])].reshape(norb[i],4,nHeavy[i]).sum(dim=1)
                q1 = q1 + v2[i][1,:norb[i],:(4*nHeavy[i])].reshape(norb[i],4,nHeavy[i]).sum(dim=1)
                charge[i,:norb[i],:nHeavy[i]] = q1
                q2 = v2[i][0,:norb[i],(4*nHeavy[i]):(4*nHeavy[i]+nHydro[i])]
                q2 = q2 + v2[i][1,:norb[i],(4*nHeavy[i]):(4*nHeavy[i]+nHydro[i])]
                charge[i,:norb[i],nHeavy[i]:(nHeavy[i]+nHydro[i])] = q2
            charge = charge / 2
        else: # closed shell
            for i in range(nmol):
                charge[i,:norb[i],:nHeavy[i]] = v2[i][:norb[i],:(4*nHeavy[i])].reshape(norb[i],4,nHeavy[i]).sum(dim=1)
                charge[i,:norb[i],nHeavy[i]:(nHeavy[i]+nHydro[i])] = v2[i][:norb[i],(4*nHeavy[i]):(4*nHeavy[i]+nHydro[i])]
        
        return F, e,    Pconv, Hcore, w, charge, rho0xi, rho0xj, notconverged
    else:
        return F, None, Pconv, Hcore, w, None,   rho0xi, rho0xj, notconverged

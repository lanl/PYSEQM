import torch
from torch.autograd import grad
from .fock import fock
from .hcore import hcore
from .energy import elec_energy
from .SP2 import SP2
from .pack import pack, unpack
from .diag import sym_eig_trunc, sym_eig_trunc1
import warnings

# scf_backward==0: ignore the gradient on density matrix
# scf_backward==1: use recursive formula
# scf_backward==2: go backward scf loop directly

debug = False
# debug=True

MAX_ITER = 1000
SCF_BACKWARD_MAX_ITER = 10
MAX_ITER_TO_STOP_IF_SCF_BACKWARD_DIVERGE = 5

RAISE_ERROR_IF_SCF_BACKWARD_FAILS = False
# if true, raise error rather than
# truncate gradient(set the gradient on non-converged molecules as 0.0)

RAISE_ERROR_IF_SCF_FORWARD_FAILS = False
# if true, raise error rather than ignore those non-converged molecules


# use constant mixing
def scf_forward0(M, w, gss, gpp, gsp, gp2, hsp,
                 nHydro, nHeavy, nOccMO,
                 nmol, molsize,
                 maskd, mask, idxi, idxj, P, eps, sp2=[False], alpha=0.0, backward=False):
    """
    alpha : mixing parameters, alpha=0.0, directly take the new density matrix
    backward is for testing purpose, default is False
    if want to test scf backward directly through the loop in this function, turn backward to be True
    """
    Pnew = torch.zeros_like(P)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    notconverged = torch.ones(nmol, dtype=torch.bool, device=M.device)
    F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4) \
             .transpose(2, 3) \
             .reshape(nmol, 4 * molsize, 4 * molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)
    k = 0
    while (1):
        if notconverged.any():
            if backward:
                Pnew[notconverged] = sym_eig_trunc1(F[notconverged],
                                                    nHeavy[notconverged],
                                                    nHydro[notconverged],
                                                    nOccMO[notconverged])[1]
            elif sp2[0]:
                # Pnew[notconverged] = SP2(F[notconverged], nOccMO[notconverged], sp2[1])
                Pnew[notconverged] = unpack(

                    SP2(
                        pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                        nOccMO[notconverged], sp2[1]
                    ),
                    nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
            else:
                """
                Pnew[notconverged] = torch.stack(list(map(lambda x,nX,nH,nocc: sym_eig_trunc(x,nX,nH,nocc)[1], \
                              F[notconverged],nHeavy[notconverged], nHydro[notconverged], nOccMO[notconverged])))
                """
                Pnew[notconverged] = sym_eig_trunc(F[notconverged],
                                                   nHeavy[notconverged],
                                                   nHydro[notconverged],
                                                   nOccMO[notconverged])[1]
            #
            if backward:
                # P=Pnew.clone()
                P = alpha * P + (1.0 - alpha) * Pnew
            else:
                P[notconverged] = alpha * P[notconverged] + (1.0 - alpha) * Pnew[notconverged]
                # Pnew[notconverged] = P[notconverged]
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            # print(torch.max(Eelec_new-Eelec)) #make sure energy is decreasing
            err[notconverged] = torch.abs(Eelec_new[notconverged] - Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]

            """
            err[notconverged] = torch.max(torch.abs( P[notconverged].matmul(F[notconverged]) \
                                                    -F[notconverged].matmul(P[notconverged])) \
                                                    .reshape(torch.sum(notconverged),-1), dim=1)[0]
            #"""
            notconverged = err > eps
            if debug:
                print("scf ", k, torch.max(err).item(), torch.sum(notconverged).item())
            k += 1
            if k >= MAX_ITER:
                return P, notconverged
        else:
            return P, notconverged

# adaptive mixing


def scf_forward1(M, w, gss, gpp, gsp, gp2, hsp,
                 nHydro, nHeavy, nOccMO,
                 nmol, molsize,
                 maskd, mask, idxi, idxj, P, eps, sp2=[False], backward=False):
    """
    adaptive mixing algorithm, see cnvg.f
    """
    nDirect1 = 2
    notconverged = torch.ones(nmol, dtype=torch.bool, device=M.device)

    k = 0
    F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    Pnew = torch.zeros_like(P)
    Pold = torch.zeros_like(P)
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4) \
             .transpose(2, 3) \
             .reshape(nmol, 4 * molsize, 4 * molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)

    for i in range(nDirect1):
        if notconverged.any():
            if backward:
                Pnew[notconverged] = sym_eig_trunc1(F[notconverged],
                                                    nHeavy[notconverged],
                                                    nHydro[notconverged],
                                                    nOccMO[notconverged])[1]
            elif sp2[0]:
                # Pnew[notconverged] = SP2(F[notconverged], nOccMO[notconverged], sp2[1])
                Pnew[notconverged] = unpack(
                    SP2(
                        pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                        nOccMO[notconverged], sp2[1]
                    ),
                    nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
            else:
                """
                Pnew[notconverged] = torch.stack(list(map(lambda x,nX,nH,nocc: sym_eig_trunc(x,nX,nH,nocc)[1], \
                              F[notconverged],nHeavy[notconverged], nHydro[notconverged], nOccMO[notconverged])))
                """
                Pnew[notconverged] = sym_eig_trunc(F[notconverged],
                                                   nHeavy[notconverged],
                                                   nHydro[notconverged],
                                                   nOccMO[notconverged])[1]
            if backward:
                Pold = P + 0.0
                P = Pnew + 0.0
            else:
                Pold[notconverged] = P[notconverged]
                P[notconverged] = Pnew[notconverged]
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            # print(torch.max(Eelec_new-Eelec)) #make sure energy is decreasing
            err[notconverged] = torch.abs(Eelec_new[notconverged] - Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            """
            err[notconverged] = torch.max(torch.abs( P[notconverged].matmul(F[notconverged]) \
                                                    -F[notconverged].matmul(P[notconverged])) \
                                                    .reshape(torch.sum(notconverged),-1), dim=1)[0]
            #"""
            notconverged = err > eps
            if debug:
                print("scf ", k, torch.max(err).item(), torch.sum(notconverged).item())
            k += 1
        else:
            return P, notconverged
    if backward:
        fac_register = []
    while (1):
        if notconverged.any():
            if backward:
                Pnew[notconverged] = sym_eig_trunc1(F[notconverged],
                                                    nHeavy[notconverged],
                                                    nHydro[notconverged],
                                                    nOccMO[notconverged])[1]
            elif sp2[0]:
                # Pnew[notconverged] = SP2(F[notconverged], nOccMO[notconverged], sp2[1])
                Pnew[notconverged] = unpack(
                    SP2(
                        pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                        nOccMO[notconverged], sp2[1]
                    ),
                    nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
            else:
                """
                Pnew[notconverged] = torch.stack(list(map(lambda x,nX,nH,nocc: sym_eig_trunc(x,nX,nH,nocc)[1], \
                              F[notconverged],nHeavy[notconverged], nHydro[notconverged], nOccMO[notconverged])))
                """
                Pnew[notconverged] = sym_eig_trunc(F[notconverged],
                                                   nHeavy[notconverged],
                                                   nHydro[notconverged],
                                                   nOccMO[notconverged])[1]
            # fac = sqrt( \sum_i (P_ii^(k) - P_ii^(k-1))**2 / \sum_i (P_ii^(k) - 2*P_ii^(k-1) + P_ii^(k-2))**2 )
            if backward:
                with torch.no_grad():
                    f = torch.zeros((P.shape[0], 1, 1), dtype=P.dtype, device=P.device)
                    f[notconverged] = torch.sqrt(torch.sum((Pnew[notconverged].diagonal(dim1=1, dim2=2)
                                                            - P[notconverged].diagonal(dim1=1, dim2=2)
                                                            )**2, dim=1)
                                                 / torch.sum((Pnew[notconverged].diagonal(dim1=1, dim2=2)
                                                              - P[notconverged].diagonal(dim1=1, dim2=2) * 2.0
                                                              + Pold[notconverged].diagonal(dim1=1, dim2=2)
                                                              )**2, dim=1)).reshape(-1, 1, 1)
                    fac_register.append(f)
            else:
                fac = torch.sqrt(torch.sum((Pnew[notconverged].diagonal(dim1=1, dim2=2)
                                            - P[notconverged].diagonal(dim1=1, dim2=2)
                                            )**2, dim=1)
                                 / torch.sum((Pnew[notconverged].diagonal(dim1=1, dim2=2)
                                              - P[notconverged].diagonal(dim1=1, dim2=2) * 2.0
                                              + Pold[notconverged].diagonal(dim1=1, dim2=2)
                                              )**2, dim=1)).reshape(-1, 1, 1)
            #
            if backward:
                Pold = P + 0.0
                P = (1.0 + fac_register[-1]) * Pnew - fac_register[-1] * P
            else:
                Pold[notconverged] = P[notconverged]
                P[notconverged] = (1.0 + fac) * Pnew[notconverged] - fac * P[notconverged]

            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            # print(torch.max(Eelec_new-Eelec)) #make sure energy is decreasing
            err[notconverged] = torch.abs(Eelec_new[notconverged] - Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            """
            #err = torch.max(torch.abs(F.matmul(P)-P.matmul(F)).reshape(nmol,-1),dim=1)[0]
            err[notconverged] = torch.max(torch.abs( P[notconverged].matmul(F[notconverged]) \
                                                    -F[notconverged].matmul(P[notconverged])) \
                                                    .reshape(torch.sum(notconverged),-1), dim=1)[0]
            #err = torch.max(torch.abs((Pnew-P).reshape(Pnew.shape[0],-1)), dim=1)[0]
            #"""
            notconverged = err > eps
            if debug:
                print("scf ", k, torch.max(err).item(), torch.sum(notconverged).item())
            k += 1
            if k >= MAX_ITER:
                return P, notconverged
        else:
            return P, notconverged

# adaptive mixing, pulay


def scf_forward2(M, w, gss, gpp, gsp, gp2, hsp,
                 nHydro, nHeavy, nOccMO,
                 nmol, molsize,
                 maskd, mask, idxi, idxj, P, eps, sp2=[False]):
    """
    adaptive mixing algorithm, see cnvg.f
    combine with pulay converger
    #check mopac for which P is stored: P constructed from fock subroutine, or P from pulay algorithm
    """
    dtype = M.dtype
    device = M.device
    # procedure
    # nDirect1 steps of directly taking new density
    # nAdapt steps of adaptive mixing (start preparing for pulay)
    # nFock-nAdapt steps of directly taking new density
    # pulay

    nDirect1 = 2
    nAdapt = 1

    # number of maximal fock matrixes used
    nFock = 5

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
    FPPF = torch.zeros(nmol, nFock, molsize * 4, molsize * 4, dtype=dtype, device=device)
    EMAT = (torch.eye(nFock + 1, nFock + 1, dtype=dtype, device=device) - 1.0).expand(nmol, nFock + 1, nFock + 1).tril().clone()
    EVEC = torch.zeros_like(EMAT)  # EVEC is <E(i)*E(j)> scaled by a constant
    FOCK = torch.zeros_like(FPPF)  # store last n=nFock number of Fock matrixes

    notconverged = torch.ones(nmol, dtype=torch.bool, device=M.device)

    k = 0
    F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    Pnew = torch.zeros_like(P)
    Pold = torch.zeros_like(P)
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4) \
             .transpose(2, 3) \
             .reshape(nmol, 4 * molsize, 4 * molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)

    for i in range(nDirect1):
        if notconverged.any():
            if sp2[0]:
                # Pnew[notconverged] = SP2(F[notconverged], nOccMO[notconverged], sp2[1])
                Pnew[notconverged] = unpack(
                    SP2(
                        pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                        nOccMO[notconverged], sp2[1]
                    ),
                    nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
            else:
                """
                Pnew[notconverged] = torch.stack(list(map(lambda x,nX,nH,nocc: sym_eig_trunc(x,nX,nH,nocc)[1], \
                              F[notconverged],nHeavy[notconverged], nHydro[notconverged], nOccMO[notconverged])))
                """
                Pnew[notconverged] = sym_eig_trunc(F[notconverged],
                                                   nHeavy[notconverged],
                                                   nHydro[notconverged],
                                                   nOccMO[notconverged])[1]
            Pold[notconverged] = P[notconverged]
            P[notconverged] = Pnew[notconverged]
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            # print(torch.max(Eelec_new-Eelec)) #make sure energy is decreasing
            err[notconverged] = torch.abs(Eelec_new[notconverged] - Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            """
            err[notconverged] = torch.max(torch.abs( P[notconverged].matmul(F[notconverged]) \
                                                    -F[notconverged].matmul(P[notconverged])) \
                                                    .reshape(torch.sum(notconverged),-1), dim=1)[0]
            #"""
            notconverged = err > eps
            if debug:
                print("scf ", k, torch.max(err).item(), torch.sum(notconverged).item())
            k += 1
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

    for i in range(nAdapt):
        if notconverged.any():
            if sp2[0]:
                # Pnew[notconverged] = SP2(F[notconverged], nOccMO[notconverged], sp2[1])
                Pnew[notconverged] = unpack(
                    SP2(
                        pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                        nOccMO[notconverged], sp2[1]
                    ),
                    nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
            else:
                """
                Pnew[notconverged] = torch.stack(list(map(lambda x,nX,nH,nocc: sym_eig_trunc(x,nX,nH,nocc)[1], \
                              F[notconverged],nHeavy[notconverged], nHydro[notconverged], nOccMO[notconverged])))
                """
                Pnew[notconverged] = sym_eig_trunc(F[notconverged],
                                                   nHeavy[notconverged],
                                                   nHydro[notconverged],
                                                   nOccMO[notconverged])[1]
            # fac = sqrt( \sum_i (P_ii^(k) - P_ii^(k-1))**2 / \sum_i (P_ii^(k) - 2*P_ii^(k-1) + P_ii^(k-2))**2 )
            fac = torch.sqrt(torch.sum((Pnew[notconverged].diagonal(dim1=1, dim2=2)
                                        - P[notconverged].diagonal(dim1=1, dim2=2)
                                        )**2, dim=1)
                             / torch.sum((Pnew[notconverged].diagonal(dim1=1, dim2=2)
                                          - P[notconverged].diagonal(dim1=1, dim2=2) * 2.0
                                          + Pold[notconverged].diagonal(dim1=1, dim2=2)
                                          )**2, dim=1)).reshape(-1, 1, 1)
            #
            Pold[notconverged] = P[notconverged]
            P[notconverged] = (1.0 + fac) * Pnew[notconverged] - fac * P[notconverged]

            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            # print(torch.max(Eelec_new-Eelec)) #make sure energy is decreasing
            err[notconverged] = torch.abs(Eelec_new[notconverged] - Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            """
            #err = torch.max(torch.abs(F.matmul(P)-P.matmul(F)).reshape(nmol,-1),dim=1)[0]
            err[notconverged] = torch.max(torch.abs( P[notconverged].matmul(F[notconverged]) \
                                                    -F[notconverged].matmul(P[notconverged])) \
                                                    .reshape(torch.sum(notconverged),-1), dim=1)[0]
            #err = torch.max(torch.abs((Pnew-P).reshape(Pnew.shape[0],-1)), dim=1)[0]
            """
            notconverged = err > eps
            if debug:
                print("scf ", k, torch.max(err).item(), torch.sum(notconverged).item())
            k += 1
        else:
            return P, notconverged
    #
    del Pold, Pnew

    # start prepare for pulay algorithm
    counter = -1  # index of stored FPPF for current iteration: 0, 1, ..., cFock-1
    cFock = 0  # in current iteration, number of fock matrixes stored, cFock <= nFock
    # Pulay algorithm needs at least two previous stored density and Fock matrixes to start
    while (cFock < 2):
        if notconverged.any():
            cFock = cFock + 1 if cFock < nFock else nFock
            # store fock matrix
            counter = (counter + 1) % nFock
            FOCK[notconverged, counter, :, :] = F[notconverged]
            FPPF[notconverged, counter, :, :] = (F[notconverged].matmul(P[notconverged]) - P[notconverged].matmul(F[notconverged])).triu()
            # in mopac7 interp.f pulay subroutine, only lower triangle of FPPF is used to construct EMAT
            # only compute lower triangle as Emat are symmetric
            EMAT[notconverged, counter, :cFock] = torch.sum(FPPF[notconverged, counter:(counter + 1), :, :] * FPPF[notconverged, :cFock, :, :], dim=(2, 3))
            if sp2[0]:
                # P[notconverged] = SP2(F[notconverged], nOccMO[notconverged], sp2[1])
                P[notconverged] = unpack(
                    SP2(
                        pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                        nOccMO[notconverged], sp2[1]
                    ),
                    nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
            else:
                """
                P[notconverged] = torch.stack(list(map(lambda x,nX,nH,nocc: sym_eig_trunc(x,nX,nH,nocc)[1], \
                              F[notconverged],nHeavy[notconverged], nHydro[notconverged], nOccMO[notconverged])))
                """
                P[notconverged] = sym_eig_trunc(F[notconverged],
                                                nHeavy[notconverged],
                                                nHydro[notconverged],
                                                nOccMO[notconverged])[1]
            #
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            # print(torch.max(Eelec_new-Eelec)) #make sure energy is decreasing
            err[notconverged] = torch.abs(Eelec_new[notconverged] - Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            """
            #err = torch.max(torch.abs(F.matmul(P)-P.matmul(F)).reshape(nmol,-1),dim=1)[0]
            err[notconverged] = torch.max(torch.abs( P[notconverged].matmul(F[notconverged]) \
                                                    -F[notconverged].matmul(P[notconverged])) \
                                                    .reshape(torch.sum(notconverged),-1), dim=1)[0]
            #err = torch.max(torch.abs((Pnew-P).reshape(Pnew.shape[0],-1)), dim=1)[0]
            """
            notconverged = err > eps
            if debug:
                print("scf ", k, torch.max(err).item(), torch.sum(notconverged).item())
            k += 1
        else:
            return P, notconverged

    # start pulay algorithm
    while (1):
        if notconverged.any():
            EVEC[notconverged] = EMAT[notconverged] + EMAT[notconverged].tril(-1).transpose(1, 2)
            EVEC[notconverged, :cFock, :cFock] /= EVEC[notconverged, counter:(counter + 1), counter:(counter + 1)]
            coeff = -torch.inverse(EVEC[notconverged, :(cFock + 1), :(cFock + 1)])[..., :-1, -1]
            F[notconverged] = torch.sum(FOCK[notconverged, :cFock, :, :] * coeff.unsqueeze(-1).unsqueeze(-1), dim=1)
            if sp2[0]:
                # P[notconverged] = SP2(F[notconverged], nOccMO[notconverged], sp2[1])
                P[notconverged] = unpack(
                    SP2(
                        pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                        nOccMO[notconverged], sp2[1]
                    ),
                    nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
            else:
                """
                P[notconverged] = torch.stack(list(map(lambda x,nX,nH,nocc: sym_eig_trunc(x,nX,nH,nocc)[1], \
                              F[notconverged],nHeavy[notconverged], nHydro[notconverged], nOccMO[notconverged])))
                """
                P[notconverged] = sym_eig_trunc(F[notconverged],
                                                nHeavy[notconverged],
                                                nHydro[notconverged],
                                                nOccMO[notconverged])[1]
            #
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)

            cFock = cFock + 1 if cFock < nFock else nFock
            counter = (counter + 1) % nFock
            FOCK[notconverged, counter, :, :] = F[notconverged]
            FPPF[notconverged, counter, :, :] = (F[notconverged].matmul(P[notconverged]) - P[notconverged].matmul(F[notconverged])).triu()
            # in mopac7 interp.f pulay subroutine, only lower triangle of FPPF is used to construct EMAT
            # only compute lower triangle as Emat are symmetric
            EMAT[notconverged, counter, :cFock] = torch.sum(FPPF[notconverged, counter:(counter + 1), :, :] * FPPF[notconverged, :cFock, :, :], dim=(2, 3))

            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            # print(torch.max(Eelec_new-Eelec)) #make sure energy is decreasing
            err[notconverged] = torch.abs(Eelec_new[notconverged] - Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            """
            err[notconverged] = torch.max(torch.abs( P[notconverged].matmul(F[notconverged]) \
                                                    -F[notconverged].matmul(P[notconverged])) \
                                                    .reshape(torch.sum(notconverged),-1), dim=1)[0]
            #err = torch.max(torch.abs((Pnew-P).reshape(Pnew.shape[0],-1)), dim=1)[0]
            """
            notconverged = err > eps

            if debug:
                print("scf ", k, torch.max(err).item(), torch.sum(notconverged).item())
            k += 1
            if k >= MAX_ITER:
                return P, notconverged
        else:
            return P, notconverged


class SCF(torch.autograd.Function):
    """
    scf loop
    forward and backward
    check function scf_loop for details
    """
    sp2 = [False]
    converger = [2]
    scf_backward_eps = 1.0e-2

    def __init__(self, scf_converger=[2], use_sp2=[False], scf_backward_eps=1.0e-2):
        SCF.sp2 = use_sp2
        SCF.converger = scf_converger
        SCF.scf_backward_eps = scf_backward_eps

    @staticmethod
    def forward(ctx,
                M, w, gss, gpp, gsp, gp2, hsp,
                nHydro, nHeavy, nOccMO,
                nmol, molsize,
                maskd, mask, atom_molid, pair_molid, idxi, idxj, P, eps):
        #
        if SCF.converger[0] == 0:
            P, notconverged = scf_forward0(M, w, gss, gpp, gsp, gp2, hsp,
                                           nHydro, nHeavy, nOccMO,
                                           nmol, molsize,
                                           maskd, mask, idxi, idxj, P, eps, sp2=SCF.sp2, alpha=SCF.converger[1])
        else:
            if SCF.converger[0] == 1:  # adaptive mixing
                scf_forward = scf_forward1
            elif SCF.converger[0] == 2:  # adaptive mixing, then pulay
                scf_forward = scf_forward2
            P, notconverged = scf_forward(M, w, gss, gpp, gsp, gp2, hsp,
                                          nHydro, nHeavy, nOccMO,
                                          nmol, molsize,
                                          maskd, mask, idxi, idxj, P, eps, sp2=SCF.sp2)
        eps = torch.as_tensor(eps, dtype=M.dtype, device=M.device)
        ctx.save_for_backward(P, M, w, gss, gpp, gsp, gp2, hsp,
                              nHydro, nHeavy, nOccMO,
                              maskd, mask, idxi, idxj, eps, notconverged,
                              atom_molid, pair_molid)
        #

        return P, notconverged

    @staticmethod
    def backward(ctx, grad0, grad1):
        # use recursive formula
        Pin, M, w, gss, gpp, gsp, gp2, hsp, \
            nHydro, nHeavy, nOccMO, \
            maskd, mask, idxi, idxj, eps, notconverged, \
            atom_molid, pair_molid = ctx.saved_tensors
        nmol = Pin.shape[0]
        molsize = Pin.shape[1] // 4
        grads = {}
        # print('initialization')
        gv = [Pin]
        gvind = []
        for i in range(1, 8):
            if ctx.saved_tensors[i].requires_grad:
                gv.append(ctx.saved_tensors[i])
                gvind.append(i)
                # grads[i] = torch.zeros_like(ctx.saved_tensors[i])
            else:
                grads[i] = None
        with torch.enable_grad():
            Pin.requires_grad_(True)
            F = fock(nmol, molsize, Pin, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            """
            Pout = torch.stack(list(map(lambda x,nX,nH,nocc: sym_eig_trunc(x,nX,nH,nocc)[1], F, nHeavy, nHydro, nOccMO)))
            """
            Pout = sym_eig_trunc1(F, nHeavy, nHydro, nOccMO)[1]

        k = 0
        backward_eps = SCF.scf_backward_eps.to(Pin.device)
        converged = ~notconverged.detach()  # scf forward converged
        diverged = None  # scf backward diverged
        gradients = [(grad0,)]
        while (1):
            grad0_max_prev = gradients[-1][0].abs().max(dim=-1)[0].max(dim=-1)[0]
            gradients.append(grad(Pout, gv, grad_outputs=gradients[-1][0], create_graph=True, retain_graph=True))

            grad0_max = gradients[-1][0].abs().max(dim=-1)[0].max(dim=-1)[0]
            if converged.any():
                err = torch.max(grad0_max[converged])
            else:
                err = torch.tensor(0.0, device=grad0.device)

            if debug:
                t = grad0_max[converged] > backward_eps
                print('backward scf: ', k, err.item(), t.sum().item())
            k += 1

            if err < backward_eps:
                break
            if k >= SCF_BACKWARD_MAX_ITER:
                break
            diverged = (grad0_max > grad0_max_prev) * (grad0_max >= 1.0)
            if diverged.any() and k >= MAX_ITER_TO_STOP_IF_SCF_BACKWARD_DIVERGE:
                print("SCF backward diverges for %d molecules, stop after %d iterations" %
                      (diverged.sum().item(), MAX_ITER_TO_STOP_IF_SCF_BACKWARD_DIVERGE))
                break
        t = 1
        ln = len(gradients)
        for i in gvind:
            grads[i] = torch.sum(torch.stack([gradients[k][t] for k in range(1, ln)]), dim=0)
            # for l in range(1, ln):
            #     grads[i].add_(gradients[l][t])
            t += 1
        notconverged2 = notconverged.detach().clone()
        with torch.no_grad():
            # one way is to check grad0.abs().max if they are smaller than backward_eps
            # another way is to compare grad0.abs().max with previous grad0.abs().max, if
            #      they are increasing, they are diverging
            notconverged1 = (grad0_max > backward_eps) + (~torch.isfinite(grad0_max))
            if notconverged.any():
                print("SCF forward       : %d/%d not converged" % (notconverged.sum().item(), nmol))
            if notconverged1.any():
                print("SCF backward      : %d/%d not converged" % (notconverged1.sum().item(), nmol))
                if RAISE_ERROR_IF_SCF_BACKWARD_FAILS:
                    raise ValueError("SCF backward doesn't converged for some molecules")
            notconverged2 = notconverged2 + notconverged1
            if notconverged2.any():
                print("SCF for/back-ward : %d/%d not converged" % (notconverged2.sum().item(), nmol))
                cond = notconverged2.detach()
                # M, w, gss, gpp, gsp, gp2, hsp
                # M shape(nmol*molsizes*molsize, 4, 4)
                if torch.is_tensor(grads[1]):
                    grads[1] = grads[1].reshape(nmol, molsize * molsize, 4, 4)
                    grads[1][cond] = 0.0
                    grads[1] = grads[1].reshape(nmol * molsize * molsize, 4, 4)
                # w shape (npairs, 10, 10)
                if torch.is_tensor(grads[2]):
                    grads[2][cond[pair_molid]] = 0.0
                # gss, gpp, gsp, gp2, hsp shape (natoms,)
                for i in range(3, 8):
                    if torch.is_tensor(grads[i]):
                        grads[i][cond[atom_molid]] = 0.0

        return grads[1], grads[2], grads[3], grads[4], grads[5], grads[6], grads[7], \
            None, None, None, \
            None, None, \
            None, None, None, None, None, None, None, None


class SCF0(SCF):
    @staticmethod
    def backward(ctx, grad0, grad1):
        # ignore the gradient on density matrix
        return None, None, None, None, None, None, None, \
            None, None, None, \
            None, None, \
            None, None, None, None, None, None, None, None


def scf_loop(const, molsize,
             nHeavy, nHydro, nOccMO,
             maskd, mask, atom_molid, pair_molid, idxi, idxj, ni, nj, xij, rij, Z,
             zetas, zetap, uss, upp, gss, gsp, gpp, gp2, hsp, beta, Kbeta=None,
             eps=1.0e-4, P=None, sp2=[False], scf_converger=[1], eig=False, scf_backward=0, scf_backward_eps=1.0e-2):
    """
    SCF loop
    # check hcore.py for the details of arguments
    eps : convergence criteria for density matrix on density matrix
    P : if provided, will be used as initial density matrix in scf loop
    return : F, e, P, Hcore, w, v
    """
    # device = xij.device
    # pp = paraemeters

    nmol = nOccMO.shape[0]
    tore = const.tore

    M, w = hcore(const, nmol, molsize, maskd, mask, idxi, idxj, ni, nj, xij, rij, Z,
                 zetas, zetap, uss, upp, gss, gpp, gp2, hsp, beta, Kbeta=Kbeta)

    if scf_backward == 2:
        P = None
    if not torch.is_tensor(P):
        P0 = torch.zeros_like(M)  # density matrix
        P0[maskd[Z > 1], 0, 0] = tore[Z[Z > 1]] / 4.0
        P0[maskd, 1, 1] = P0[maskd, 0, 0]
        P0[maskd, 2, 2] = P0[maskd, 0, 0]
        P0[maskd, 3, 3] = P0[maskd, 0, 0]
        P0[maskd[Z == 1], 0, 0] = 1.0
        # P0 += torch.randn(P0.shape,dtype=P0.dtype, device=P0.device)*0.01
        P = P0.reshape(nmol, molsize, molsize, 4, 4) \
            .transpose(2, 3) \
            .reshape(nmol, 4 * molsize, 4 * molsize)
        # P.add_(torch.randn(P.shape,dtype=P.dtype,device=P.device)*0.001)
    # else:
    #    P.add_(torch.randn(P.shape,dtype=P.dtype,device=P.device)*0.001)

    # t1 = time.time()
    # print('Hcore : %f sec' % (t1-t0))
    # t0 = time.time()
    # """
    # scf_backward == 2, directly backward through scf loop
    #             can't reuse P, so put P=None and initial P above
    # """
    if scf_backward == 2:
        if sp2[0]:
            warnings.warn('SP2 is not used for direct back propagation through scf loop')
            sp2[0] = False
        if scf_converger[0] == 0:
            Pconv, notconverged = scf_forward0(M, w, gss, gpp, gsp, gp2, hsp,
                                               nHydro, nHeavy, nOccMO,
                                               nmol, molsize,
                                               maskd, mask, idxi, idxj, P, eps, sp2=sp2, alpha=scf_converger[1], backward=True)
        elif scf_converger[0] == 1:
            Pconv, notconverged = scf_forward1(M, w, gss, gpp, gsp, gp2, hsp,
                                               nHydro, nHeavy, nOccMO,
                                               nmol, molsize,
                                               maskd, mask, idxi, idxj, P, eps, sp2=sp2, backward=True)
        else:
            raise ValueError("""For direct back propagation through scf,
                                must use constant mixing at this moment\n
                                set scf_converger=[0, alpha] or [1]\n""")
    # """
    # """
    # scf_backward 1, use recursive formula, uncomment following line
    if scf_backward == 1:
        scfapply = SCF(use_sp2=sp2, scf_converger=scf_converger, scf_backward_eps=scf_backward_eps).apply

    # scf_backward 0: ignore the gradient on density matrix
    if scf_backward == 0:
        scfapply = SCF0(use_sp2=sp2, scf_converger=scf_converger).apply

    # """
    # apply_params = {k:pp[k] for k in apply_param_map}
    if scf_backward == 0 or scf_backward == 1:
        Pconv, notconverged = scfapply(M, w, gss, gpp, gsp, gp2, hsp,
                                       nHydro, nHeavy, nOccMO,
                                       nmol, molsize,
                                       maskd, mask, atom_molid, pair_molid, idxi, idxj, P, eps)
    # """
    if notconverged.any():
        nnot = notconverged.type(torch.int).sum().data.item()
        warnings.warn("SCF for %d/%d molecules doesn't converge after %d iterations" % (nnot, nmol, MAX_ITER))
        if RAISE_ERROR_IF_SCF_FORWARD_FAILS:
            raise ValueError("SCF for some the molecules in the batch doesn't converge")

    # if notconverged.all():
    #    raise ValueError("SCF for all the molecules in the batch doesn't converge, try to increase MAX_ITER")

    F = fock(nmol, molsize, Pconv, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4) \
             .transpose(2, 3) \
             .reshape(nmol, 4 * molsize, 4 * molsize)

    if eig:
        """
        e, v = list(zip(*list(map(
                        lambda x,nX,nH,nocc: sym_eig_trunc(x,nX,nH,nocc,eig_only=True),
                        F,nHeavy, nHydro, nOccMO))))
        e = torch.stack(e)
        """
        if scf_backward >= 1:
            e, v = sym_eig_trunc1(F, nHeavy, nHydro, nOccMO, eig_only=True)
        else:
            e, v = sym_eig_trunc(F, nHeavy, nHydro, nOccMO, eig_only=True)

        # t1 = time.time()
        # print('Diag : %f sec' % (t1-t0))

        # get charge of each orbital on each atom
        charge = torch.zeros(nmol, molsize * 4, molsize, device=e.device, dtype=e.dtype)
        v2 = [x**2 for x in v]
        norb = 4 * nHeavy + nHydro
        for i in range(nmol):
            charge[i, :norb[i], :nHeavy[i]] = v2[i][:norb[i], :(4 * nHeavy[i])].reshape(norb[i], 4, nHeavy[i]).sum(dim=1)
            charge[i, :norb[i], nHeavy[i]:(nHeavy[i] + nHydro[i])] = v2[i][:norb[i], (4 * nHeavy[i]):(4 * nHeavy[i] + nHydro[i])]

        return F, e, Pconv, Hcore, w, charge, notconverged
    else:
        return F, None, Pconv, Hcore, w, None, notconverged

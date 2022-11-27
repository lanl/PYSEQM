import torch
from .pack import *

def Canon_DM_PRT(FO1,T,nHeavy, nHydro,Q,ev,mu_0,m, kB, Occ_mask):
    '''
    canonical density matrix perturbation theory
    Alg.2 from https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00264
    '''

    mu0 = mu_0 #Intial guess
    h0 = ev    # H0 Diagonal in eigenbasis
    FO1_initial_shape = FO1.shape
    FO1 = pack(FO1, nHeavy, nHydro)
    FO1 = Q.transpose(1,2)@(FO1@Q)    # Main cost I = O(N^3) -> GPU
    beta = 1/(kB*T)    # Temp in Kelvin
    cnst = (2**(-2-m))*beta

    # $$$ maybe Occ_mask is redundant ??
    p0 = (0.5 - cnst*(h0-mu0))*Occ_mask #################################
    p0 = p0.unsqueeze(2)                                                #
    P1 = -cnst*FO1  #{P1 = -cnst*(FO1-mu1*I);}                          #
    for i in range(0,m):                                                #
        p02 = p0*p0                                                     #
        dX1 = p0*P1+P1*p0.transpose(1,2)        # Cost O(N^2)           #
        iD0 = 1./(2*(p02-p0)+1)                                         #
        p0 = iD0*p02                                                    #
        P1 = iD0*(dX1+2*(P1-dX1)*p0.transpose(1,2)) #Cost O(N^2) <#mask##
    dpdmu=beta*p0*(1-p0)
    dmu1 = -( P1.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1).view(-1, 1, 1) )/dpdmu.sum(dim=1).view(-1, 1, 1)
    # Adjust occupation
    P1 = P1 + torch.diag_embed(dpdmu[:,:,0])*dmu1
    P1 = Q@P1@Q.transpose(1,2)  # Main cost II = O(N^3) -> GPU
    P1 = unpack(P1, nHeavy, nHydro, FO1_initial_shape[-1])
    
    return P1
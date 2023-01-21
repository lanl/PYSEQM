"""
Extended Lagrangian BOMD

XL_BOMD

D: Density Matrix from SP2 algorithm, i.e. given F => D
P: Dynamic Density Matrix (field tensor)
#t=0
#use SCF loop to construct D, and P(0) = D

P = P(t)
F(P) = Hcore + G(P) (Directly construct)
D <= F (diagonization or just use SP2)

#electronic energy
E(D,P) = (2*tr(Hcore*D) + tr((2D-P)*G(P)))/2.0
here Tr(D) = 2*Nocc
#the formula without XL is E(D) = 0.5 * tr(D(Hcore+F)), (Tr(D)==2*Nocc)
#agree if P==D

# in A.M.N. Niklasson code, SP2.m return D, tr(D)=Nocc
# and E(D,P) = 2*tr(Hcore*D) + tr((2D-P)*G(P))

#in the seqm_functions/scf_loop.py, P means Density matrix
"""

import torch
from .seqm_functions.energy import total_energy, pair_nuclear_energy, elec_energy_isolated_atom, heat_formation, elec_energy_xl
from .seqm_functions.SP2 import SP2
from .basics import Parser, Parser_For_Ovr, Pack_Parameters
from .seqm_functions.fock import fock
from .seqm_functions.G_XL_LR import G
from .seqm_functions.fermi_q import Fermi_Q
from .seqm_functions.canon_dm_prt import Canon_DM_PRT
from .seqm_functions.hcore import hcore
from .seqm_functions.diag import sym_eig_trunc
from .seqm_functions.pack import *
from .basics import Force
from torch.autograd import grad
import time

from .tools import attach_profile_range

class EnergyXL(torch.nn.Module):
    def __init__(self, seqm_parameters):
        """
        Constructor
        """
        super().__init__()
        self.seqm_parameters = seqm_parameters
        self.method = seqm_parameters['method']
        self.parser = Parser(seqm_parameters)
        self.packpar = Pack_Parameters(seqm_parameters)
        self.Hf_flag = True
        if "Hf_flag" in seqm_parameters:
            self.Hf_flag = seqm_parameters["Hf_flag"] # True: Heat of formation, False: Etot-Eiso

         
    def forward(self, molecule, P, learned_parameters=dict(), xl_bomd_params=dict(), all_terms=False, *args, **kwargs):
        """
        get the energy terms
        D: Density Matrix, F=>D  (SP2)
        P: Dynamics Field Tensor
        """
        nmol, molsize, \
        nHeavy, nHydro, nocc, \
        Z, maskd, atom_molid, \
        mask, mask_l, pair_molid, ni, nj, idxi, idxj, xij, rij = self.parser(molecule, return_mask_l=True, *args, **kwargs)
        
        if callable(learned_parameters):
            adict = learned_parameters(molecule.species, molecule.coordinates)
            parameters = self.packpar(Z, learned_params = adict)    
        else:
            parameters = self.packpar(Z, learned_params = learned_parameters)
        beta = torch.cat((parameters['beta_s'].unsqueeze(1), parameters['beta_p'].unsqueeze(1)),dim=1)
        if "Kbeta" in parameters:
            Kbeta = parameters["Kbeta"]
        else:
            Kbeta = None
        if molecule.const.do_timing:
            t0 = time.time()

        M, w = hcore(molecule.const, nmol, molsize, maskd, mask, idxi, idxj, ni,nj,xij,rij, Z, \
                     zetas=parameters['zeta_s'],
                     zetap=parameters['zeta_p'],
                     uss=parameters['U_ss'],
                     upp=parameters['U_pp'],
                     gss=parameters['g_ss'],
                     gpp=parameters['g_pp'],
                     gp2=parameters['g_p2'],
                     hsp=parameters['h_sp'],
                     beta=beta,
                     Kbeta=Kbeta)

        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            molecule.const.timing["Hcore + STO Integrals"].append(t1-t0)

        Hcore = M.reshape(nmol,molsize,molsize,4,4) \
                 .transpose(2,3) \
                 .reshape(nmol, 4*molsize, 4*molsize)

        F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, \
                 gss=parameters['g_ss'],
                 gpp=parameters['g_pp'],
                 gsp=parameters['g_sp'],
                 gp2=parameters['g_p2'],
                 hsp=parameters['h_sp'])

        
        if 'max_rank' in xl_bomd_params: # Krylov
            if 'scf_backward' in self.seqm_parameters:
                self.scf_backward = self.seqm_parameters['scf_backward']
            else:
                self.scf_backward = 0

            Temp = xl_bomd_params['T_el']
            kB = 8.61739e-5 # eV/K, kB = 6.33366256e-6 Ry/K, kB = 3.166811429e-6 Ha/K, #kB = 3.166811429e-6 #Ha/K
            with torch.no_grad():
                D,S_Ent,QQ,e,Fe_occ,mu0, Occ_mask = Fermi_Q(F, Temp, nocc, nHeavy, nHydro, kB, scf_backward = self.scf_backward) # Fermi operator expansion, eigenapirs [QQ,e], and entropy S_Ent

                EEnt = -2.*Temp*S_Ent
                e_gap = e.gather(1, nocc.unsqueeze(0).T) - e.gather(1, nocc.unsqueeze(0).T-1)

                Rank = xl_bomd_params['max_rank']
                K0 = 1.0
                dDS = K0*(D - P) # tr2  #W0 = K0*(DS - X) from J. Chem. Theory Comput. 2020, 16, 6, 3628–3640, alg 3

                #  Rank-m Kernel approximation of dP2dt2 %%%-
                V = torch.zeros((D.shape[0], D.shape[1], D.shape[2], Rank), dtype=D.dtype, device=D.device)
                W = torch.zeros((D.shape[0], D.shape[1], D.shape[2], Rank), dtype=D.dtype, device=D.device)
                dW = dDS # tr2
                k = -1
                Error = torch.tensor([10], dtype=D.dtype, device=D.device)

                while k < Rank-1 and torch.max(Error) > xl_bomd_params['err_threshold']:
                #while k < Rank-1:
                    k = k + 1
                    V[:,:,:,k] = dW

                    for j in range(0,k): #Orthogonalized Krylov vectors (Arnoldi)
                        #  J. Chem. Theory Comput. 2020, 16, 6, 3628–3640, alg 3
                        ### if not symmetric use this:
                        ### V[:,:,:,k] = V[:,:,:,k] - (V[:,:,:,k].transpose(1,2)@V[:,:,:,j]).diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1).view(-1, 1, 1)*V[:,:,:,j]
                        V[:,:,:,k] = V[:,:,:,k] -  torch.sum(V[:,:,:,k].transpose(1,2)*V[:,:,:,j], dim=(1,2)).view(-1, 1, 1) * V[:,:,:,j]
                    V[:,:,:,k] = V[:,:,:,k]/torch.linalg.norm(V[:,:,:,k], ord='fro', dim=(1,2)).view(-1, 1, 1)

                    d_D = V[:,:,:,k]
                    FO1 = G(nmol, molsize, d_D, M, maskd, mask, idxi, idxj, w, \
                                    gss=parameters['g_ss'],
                                    gpp=parameters['g_pp'],
                                    gsp=parameters['g_sp'],
                                    gp2=parameters['g_p2'],
                                    hsp=parameters['h_sp'])

                    # $$$ multiply by 2 ???
                    PO1 = Canon_DM_PRT(FO1,Temp,nHeavy,nHydro,QQ,e,mu0,8, kB, Occ_mask)

                    W[:,:,:,k] = K0*(PO1 - V[:,:,:,k])
                    dW = W[:,:,:,k]
                    Rank_m = k+1

                    O = torch.zeros((D.shape[0], Rank_m, Rank_m), dtype=D.dtype, device=D.device)   
                    for I in range(0,Rank_m):
                        for J in range(I,Rank_m):
                            O[:,I,J] = torch.sum(W[:,:,:,I].transpose(1,2)*W[:,:,:,J], dim=(1,2))
                            O[:,J,I] = O[:,I,J]

                    MM = torch.inverse(O)

                    IdentRes = torch.zeros(D.shape, dtype=D.dtype, device=D.device)
                    for I in range(0,Rank_m):
                        for J in range(0,Rank_m):
                            IdentRes = IdentRes + \
                                MM[:,I,J].view(-1, 1, 1) * torch.sum(W[:,:,:,J].transpose(1,2)*dDS, dim=(1,2)).view(-1, 1, 1) * W[:,:,:,I]            
                    Error = torch.linalg.norm(IdentRes - dDS, ord='fro', dim=(1,2))/torch.linalg.norm(dDS, ord='fro', dim=(1,2))

                dP2dt2 = torch.zeros(D.shape, dtype=D.dtype, device=D.device)

                # $$$ room for optimization
                for I in range(0,Rank_m):
                    for J in range(0,Rank_m):
                        dP2dt2 = dP2dt2 - \
                            MM[:,I,J].view(-1, 1, 1) * torch.sum(W[:,:,:,J].transpose(1,2)*dDS, dim=(1,2)).view(-1, 1, 1) * V[:,:,:,I]

            del V, PO1, QQ, d_D, dDS, W, dW, FO1, O, MM            
            
        else:
            sp2 = self.seqm_parameters['sp2']
            if molecule.const.do_timing:
                t0 = time.time()
            with torch.no_grad():
                if sp2[0]:
                    D = unpack(SP2(pack(F, nHeavy, nHydro), nocc, sp2[1]), nHeavy, nHydro, F.shape[-1])
                    e_gap = torch.zeros(molecule.species.shape[0],1)
                    e = torch.zeros(molecule.species.shape[0], molecule.nocc)
                else:
                    e, D = sym_eig_trunc(F,nHeavy, nHydro, nocc)[0:2]
                    e_gap = e.gather(1, nocc.unsqueeze(0).T) - e.gather(1, nocc.unsqueeze(0).T-1)
                    #print('adfdfa',F, '\nfg', sym_eig_trunc(F,nHeavy, nHydro, nocc))
                    
            EEnt = torch.zeros(molecule.species.shape[0], device=molecule.coordinates.device)
            dP2dt2 = None
            Error = None
            Fe_occ = None
            if molecule.const.do_timing:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.time()
                molecule.const.timing["D*"].append(t1-t0)

            
            

        #nuclear energy
        alpha = parameters['alpha']
        if self.method=='MNDO':
            parnuc = (alpha,)
        elif self.method=='AM1':
            K = torch.stack((parameters['Gaussian1_K'],
                             parameters['Gaussian2_K'],
                             parameters['Gaussian3_K'],
                             parameters['Gaussian4_K']),dim=1)
            #
            L = torch.stack((parameters['Gaussian1_L'],
                             parameters['Gaussian2_L'],
                             parameters['Gaussian3_L'],
                             parameters['Gaussian4_L']),dim=1)
            #
            M = torch.stack((parameters['Gaussian1_M'],
                             parameters['Gaussian2_M'],
                             parameters['Gaussian3_M'],
                             parameters['Gaussian4_M']),dim=1)
            #
            parnuc = (alpha, K, L, M)
        elif self.method=='PM3':
            K = torch.stack((parameters['Gaussian1_K'],
                             parameters['Gaussian2_K']),dim=1)
            #
            L = torch.stack((parameters['Gaussian1_L'],
                             parameters['Gaussian2_L']),dim=1)
            #
            M = torch.stack((parameters['Gaussian1_M'],
                             parameters['Gaussian2_M']),dim=1)
            #
            parnuc = (alpha, K, L, M)

        EnucAB = pair_nuclear_energy(molecule.const, nmol, ni, nj, idxi, idxj, rij, gam=w[...,0,0], method=self.method, parameters=parnuc)
        Eelec = elec_energy_xl(D,P,F,Hcore)
        if all_terms:
            Etot, Enuc = total_energy(nmol, pair_molid,EnucAB, Eelec)
            Eiso = elec_energy_isolated_atom(molecule.const, Z,
                                         uss=parameters['U_ss'],
                                         upp=parameters['U_pp'],
                                         gss=parameters['g_ss'],
                                         gpp=parameters['g_pp'],
                                         gsp=parameters['g_sp'],
                                         gp2=parameters['g_p2'],
                                         hsp=parameters['h_sp'])
            Hf, Eiso_sum = heat_formation(molecule.const, nmol,atom_molid, Z, Etot, Eiso, flag=self.Hf_flag)
            return Hf, Etot, Eelec, EEnt, Enuc, Eiso_sum, EnucAB, D, dP2dt2, Error, e_gap, e, Fe_occ
        else:
            #for computing force, Eelec.sum()+EnucAB.sum() and backward is enough
            #index_add is used in total_energy and heat_formation function
            # P can be used as the initialization
            return Eelec, EnucAB, D, e_gap, e


class ForceXL(torch.nn.Module):
    """
    get force for XL-BOMD
    """
    def __init__(self, seqm_parameters):
        super().__init__()
        self.energy = EnergyXL(seqm_parameters)
        self.seqm_parameters = seqm_parameters

    def forward(self, molecule, P, learned_parameters=dict(), xl_bomd_params=dict(), *args, **kwargs):

        molecule.coordinates.requires_grad_(True)
        Hf, Etot, Eelec, EEnt, Enuc, Eiso, EnucAB, D, dP2dt2, Error, e_gap, e, Fe_occ = \
            self.energy(molecule, P, learned_parameters, xl_bomd_params=xl_bomd_params, all_terms=True, *args, **kwargs)
        L = Hf.sum()
        if molecule.const.do_timing:
            t0 = time.time()
        gv = [molecule.coordinates]
        gradients  = grad(L, gv)
        molecule.coordinates.grad = gradients[0]
        #"""
        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            molecule.const.timing["Force"].append(t1-t0)
        with torch.no_grad():
            force = -molecule.coordinates.grad.clone()
            molecule.coordinates.grad.zero_()
        #return force, Etot, D.detach()
        del EnucAB, L
        return force.detach(), D.detach(), Hf, Etot.detach(), Eelec.detach(), Enuc.detach(), Eiso.detach(), e, e_gap, EEnt, dP2dt2, Error, Fe_occ

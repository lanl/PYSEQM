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
from .basics import Parser, Pack_Parameters
from .seqm_functions.fock import fock
from .seqm_functions.G_XL_LR import G
from .seqm_functions.fermi_q import Fermi_Q
from .seqm_functions.build_two_elec_one_center_int_D import calc_integral #, calc_integral_os

from .seqm_functions.canon_dm_prt import Canon_DM_PRT
from .seqm_functions.hcore import hcore
from .seqm_functions.diag import sym_eig_trunc
from .seqm_functions.pack import *
from .basics import Force
from torch.autograd import grad
import time

from .tools import attach_profile_range


# number of iterations in canon_dm_prt.py (m)
CANON_DM_PRT_ITER = 10


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

        molecule.nmol, molecule.molsize, \
        molecule.nSuperHeavy, molecule.nHeavy, molecule.nHydro, molecule.nocc, \
        molecule.Z, molecule.maskd, molecule.atom_molid, \
        molecule.mask, molecule.pair_molid, molecule.ni, molecule.nj, molecule.idxi, molecule.idxj, molecule.xij, molecule.rij = self.parser(molecule, self.method, *args, **kwargs)

        
        if callable(learned_parameters):
            adict = learned_parameters(molecule.species, molecule.coordinates)
            molecule.parameters, molecule.alp, molecule.chi = self.packpar(molecule.Z, learned_params = adict)   
        else:
            molecule.parameters, molecule.alp, molecule.chi = self.packpar(molecule.Z, learned_params = learned_parameters)

        
        if(molecule.method == 'PM6'): # PM6 not implemented yet. Only PM6_SP
            molecule.parameters['beta'] = torch.cat((molecule.parameters['beta_s'].unsqueeze(1), molecule.parameters['beta_p'].unsqueeze(1), molecule.parameters['beta_d'].unsqueeze(1)),dim=1)
        else:
            molecule.parameters['beta'] = torch.cat((molecule.parameters['beta_s'].unsqueeze(1), molecule.parameters['beta_p'].unsqueeze(1)),dim=1)        
            molecule.parameters['zeta_d'] = torch.zeros_like(molecule.parameters['zeta_s'])
            molecule.parameters['s_orb_exp_tail'] = torch.zeros_like(molecule.parameters['zeta_s'])
            molecule.parameters['p_orb_exp_tail'] = torch.zeros_like(molecule.parameters['zeta_s'])
            molecule.parameters['d_orb_exp_tail'] = torch.zeros_like(molecule.parameters['zeta_s'])

            molecule.parameters['U_dd'] = torch.zeros_like(molecule.parameters['U_ss'])
            molecule.parameters['F0SD'] = torch.zeros_like(molecule.parameters['U_ss'])
            molecule.parameters['G2SD'] = torch.zeros_like(molecule.parameters['U_ss'])
            molecule.parameters['rho_core'] = torch.zeros_like(molecule.parameters['U_ss'])

        
        molecule.parameters['Kbeta'] = molecule.parameters.get('Kbeta', None)


        if molecule.const.do_timing:
            t0 = time.time()

        M, w, rho0xi, rho0xj, _, _ = hcore(molecule)

        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            molecule.const.timing["Hcore + STO Integrals"].append(t1-t0)

        Hcore = M.reshape(molecule.nmol,molecule.molsize,molecule.molsize,4,4) \
                 .transpose(2,3) \
                 .reshape(molecule.nmol, 4*molecule.molsize, 4*molecule.molsize)

        if(molecule.method == 'PM6'): # PM6 does not work. ignore this part
            if molecule.nocc.dim() == 2: # open shell
                
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
            
        
        F = fock(molecule.nmol, molecule.molsize, P, M, molecule.maskd, molecule.mask, molecule.idxi, molecule.idxj, w, W, \
                 molecule.parameters['g_ss'],
                 molecule.parameters['g_pp'],
                 molecule.parameters['g_sp'],
                 molecule.parameters['g_p2'],
                 molecule.parameters['h_sp'],
                 molecule.method,
                 molecule.parameters['s_orb_exp_tail'],
                 molecule.parameters['p_orb_exp_tail'],
                 molecule.parameters['d_orb_exp_tail'],
                 molecule.Z,
                 molecule.parameters['F0SD'],
                 molecule.parameters['G2SD'])


        
        if 'max_rank' in xl_bomd_params: # Krylov
            if 'scf_backward' in self.seqm_parameters:
                self.scf_backward = self.seqm_parameters['scf_backward']
            else:
                self.scf_backward = 0

            Temp = xl_bomd_params['T_el']
            kB = 8.61739e-5 # eV/K, kB = 6.33366256e-6 Ry/K, kB = 3.166811429e-6 Ha/K, #kB = 3.166811429e-6 #Ha/K
            with torch.no_grad():
                D,S_Ent,QQ,e,Fe_occ,mu0, Occ_mask = Fermi_Q(F, Temp, molecule.nocc, molecule.nHeavy, molecule.nHydro, kB, scf_backward = self.scf_backward) # Fermi operator expansion, eigenapirs [QQ,e], and entropy S_Ent

                EEnt = -2.*Temp*S_Ent
                lumo = molecule.nocc.unsqueeze(0).T
                e_gap = (e.gather(1, lumo) - e.gather(1, lumo-1)).reshape(-1)
                #e_gap = e.gather(1, molecule.nocc.unsqueeze(0).T) - e.gather(1, molecule.nocc.unsqueeze(0).T-1)

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
                    
                    FO1 = G(molecule.nmol, molecule.molsize, d_D, M, molecule.maskd, molecule.mask, molecule.idxi, molecule.idxj, w, W, \
                            molecule.parameters['g_ss'],
                            molecule.parameters['g_pp'],
                            molecule.parameters['g_sp'],
                            molecule.parameters['g_p2'],
                            molecule.parameters['h_sp'],
                            molecule.method,
                            molecule.parameters['s_orb_exp_tail'],
                            molecule.parameters['p_orb_exp_tail'],
                            molecule.parameters['d_orb_exp_tail'],
                            molecule.Z,
                            molecule.parameters['F0SD'],
                            molecule.parameters['G2SD'])

                    # $$$ multiply by 2 ???
                    PO1 = Canon_DM_PRT(FO1,Temp,molecule.nHeavy,molecule.nHydro,QQ,e,mu0,CANON_DM_PRT_ITER, kB, Occ_mask)

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
                    D = unpack(SP2(pack(F, molecule.nHeavy, molecule.nHydro), molecule.nocc, sp2[1]), molecule.nHeavy, molecule.nHydro, F.shape[-1])
                    e_gap = torch.zeros(molecule.species.shape[0],1)
                    e = torch.zeros(molecule.species.shape[0], molecule.molecule.nocc)
                else:
                    e, D = sym_eig_trunc(F,molecule.nHeavy, molecule.nHydro, molecule.nocc)[0:2]
                    lumo = molecule.nocc.unsqueeze(0).T
                    e_gap = (e.gather(1, lumo) - e.gather(1, lumo-1)).reshape(-1)
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
        alpha = molecule.parameters['alpha']
        if self.method=='MNDO':
            parnuc = (alpha,)
        elif self.method=='AM1' or self.method=='PM6' or self.method=='PM6_SP':
            K = torch.stack((molecule.parameters['Gaussian1_K'],
                             molecule.parameters['Gaussian2_K'],
                             molecule.parameters['Gaussian3_K'],
                             molecule.parameters['Gaussian4_K']),dim=1)
            #
            L = torch.stack((molecule.parameters['Gaussian1_L'],
                             molecule.parameters['Gaussian2_L'],
                             molecule.parameters['Gaussian3_L'],
                             molecule.parameters['Gaussian4_L']),dim=1)
            #molecule.
            M = torch.stack((molecule.parameters['Gaussian1_M'],
                             molecule.parameters['Gaussian2_M'],
                             molecule.parameters['Gaussian3_M'],
                             molecule.parameters['Gaussian4_M']),dim=1)
            #
            parnuc = (alpha, K, L, M)
        elif self.method=='PM3':
            K = torch.stack((molecule.parameters['Gaussian1_K'],
                             molecule.parameters['Gaussian2_K']),dim=1)
            #
            L = torch.stack((molecule.parameters['Gaussian1_L'],
                             molecule.parameters['Gaussian2_L']),dim=1)
            #
            M = torch.stack((molecule.parameters['Gaussian1_M'],
                             molecule.parameters['Gaussian2_M']),dim=1)
            #
            parnuc = (alpha, K, L, M)

        if 'g_ss_nuc' in molecule.parameters:
            g = molecule.parameters['g_ss_nuc']
            rho0a = 0.5 * ev / g[molecule.idxi]
            rho0b = 0.5 * ev / g[molecule.idxj]
            gam = ev / torch.sqrt(molecule.rij**2 + (rho0a + rho0b)**2)
        else:
            gam = w[...,0,0]
        
        
        EnucAB = pair_nuclear_energy(molecule.Z, molecule.const, molecule.nmol, molecule.ni, molecule.nj, molecule.idxi, molecule.idxj, molecule.rij, \
                                     rho0xi,rho0xj,molecule.alp, molecule.chi, gam=gam, method=self.method, parameters=parnuc)
        Eelec = elec_energy_xl(D,P,F,Hcore)
        if all_terms:
            Etot, Enuc = total_energy(molecule.nmol, molecule.pair_molid,EnucAB, Eelec)
            Eiso = elec_energy_isolated_atom(molecule.const, molecule.Z,
                                         uss=molecule.parameters['U_ss'],
                                         upp=molecule.parameters['U_pp'],
                                         gss=molecule.parameters['g_ss'],
                                         gpp=molecule.parameters['g_pp'],
                                         gsp=molecule.parameters['g_sp'],
                                         gp2=molecule.parameters['g_p2'],
                                         hsp=molecule.parameters['h_sp'])
            Hf, Eiso_sum = heat_formation(molecule.const, molecule.nmol,molecule.atom_molid, molecule.Z, Etot, Eiso, flag=self.Hf_flag)
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
        self.create_graph = seqm_parameters.get('2nd_grad', False)

    def forward(self, molecule, P, learned_parameters=dict(), xl_bomd_params=dict(), *args, **kwargs):

        molecule.coordinates.requires_grad_(True)
        Hf, Etot, Eelec, EEnt, Enuc, Eiso, EnucAB, D, dP2dt2, Error, e_gap, e, Fe_occ = \
            self.energy(molecule, P, learned_parameters, xl_bomd_params=xl_bomd_params, all_terms=True, *args, **kwargs)
        L = Hf.sum()
        if molecule.const.do_timing:
            t0 = time.time()
        L.backward(create_graph=self.create_graph)
        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            molecule.const.timing["Force"].append(t1-t0)
        if self.create_graph:
            force = -molecule.coordinates.grad.clone()
            with torch.no_grad(): molecule.coordinates.grad.zero_()
        else:
            force = -molecule.coordinates.grad.detach()
            molecule.coordinates.grad.zero_()
        del EnucAB, L
        return force.detach(), D.detach(), Hf, Etot.detach(), Eelec.detach(), Enuc.detach(), Eiso.detach(), e, e_gap, EEnt, dP2dt2, Error, Fe_occ

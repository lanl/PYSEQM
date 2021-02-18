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
from .seqm_functions.energy import total_energy, pair_nuclear_energy, elec_energy_isolated_atom, heat_formation
from .seqm_functions.SP2 import SP2
from .basics import Parser, Pack_Parameters
from .seqm_functions.fock import fock
from .seqm_functions.hcore import hcore
from .seqm_functions.diag import sym_eig_trunc
from .seqm_functions.pack import *
from .basics import Force
from .MolecularDynamics import Molecular_Dynamics_Basic
from torch.autograd import grad
import time

def elec_energy_xl(D,P,F,Hcore):
    """
    XL_BOMD
    electrionic energy is defined as:
    E(D,P) = (2*tr(Hcore*D) + tr((2D-P)*G(P)))/2.0
           = tr(D*F)-0.5*Tr((F-Hcore)*P)
    """
    #Hcore : only have upper triangle as constructed from hcore.py
    h = Hcore.triu()+Hcore.triu(1).transpose(1,2)

    Eelec = torch.sum(D*F-0.5*(F-h)*P,dim=(1,2))

    return Eelec

class EnergyXL(torch.nn.Module):
    def __init__(self, seqm_parameters):
        """
        Constructor
        """
        super().__init__()

        self.method = seqm_parameters['method']
        self.seqm_parameters = seqm_parameters

        self.parser = Parser(seqm_parameters)
        self.packpar = Pack_Parameters(seqm_parameters)
        self.Hf_flag = True
        if "Hf_flag" in seqm_parameters:
            self.Hf_flag = seqm_parameters["Hf_flag"] # True: Heat of formation, False: Etot-Eiso

    def forward(self, const, coordinates, species, P, learned_parameters=dict(), all_terms=False, step=0):
        """
        get the energy terms
        D: Density Matrix, F=>D  (SP2)
        P: Dynamics Field Tensor
        """
        nmol, molsize, \
        nHeavy, nHydro, nocc, \
        Z, maskd, atom_molid, \
        mask, pair_molid, ni, nj, idxi, idxj, xij, rij = self.parser(const, species, coordinates)
        if callable(learned_parameters):
            adict = learned_parameters(species, coordinates)
            parameters = self.packpar(Z, learned_params = adict)    
        else:
            parameters = self.packpar(Z, learned_params = learned_parameters)
        beta = torch.cat((parameters['beta_s'].unsqueeze(1), parameters['beta_p'].unsqueeze(1)),dim=1)
        if "Kbeta" in parameters:
            Kbeta = parameters["Kbeta"]
        else:
            Kbeta = None
        if const.do_timing:
            t0 = time.time()
        M, w = hcore(const, nmol, molsize, maskd, mask, idxi, idxj, ni,nj,xij,rij, Z, \
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
        #
        if const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            const.timing["Hcore + STO Integrals"].append(t1-t0)

        Hcore = M.reshape(nmol,molsize,molsize,4,4) \
                 .transpose(2,3) \
                 .reshape(nmol, 4*molsize, 4*molsize)
        #
        F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, \
                 gss=parameters['g_ss'],
                 gpp=parameters['g_pp'],
                 gsp=parameters['g_sp'],
                 gp2=parameters['g_p2'],
                 hsp=parameters['h_sp'])
        #
        sp2 = self.seqm_parameters['sp2']
        if const.do_timing:
            t0 = time.time()
        with torch.no_grad():
            if sp2[0]:
                D = unpack(SP2(pack(F, nHeavy, nHydro), nocc, sp2[1]), nHeavy, nHydro, F.shape[-1])
            else:
                D = sym_eig_trunc(F,nHeavy, nHydro, nocc)[1]
        if const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            const.timing["D*"].append(t1-t0)
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
        #

        EnucAB = pair_nuclear_energy(const, nmol, ni, nj, idxi, idxj, rij, gam=w[...,0,0], method=self.method, parameters=parnuc)
        Eelec = elec_energy_xl(D,P,F,Hcore)
        if all_terms:
            Etot, Enuc = total_energy(nmol, pair_molid,EnucAB, Eelec)
            Eiso = elec_energy_isolated_atom(const, Z,
                                         uss=parameters['U_ss'],
                                         upp=parameters['U_pp'],
                                         gss=parameters['g_ss'],
                                         gpp=parameters['g_pp'],
                                         gsp=parameters['g_sp'],
                                         gp2=parameters['g_p2'],
                                         hsp=parameters['h_sp'])
            Hf, Eiso_sum = heat_formation(const, nmol,atom_molid, Z, Etot, Eiso, flag=self.Hf_flag)
            return Hf, Etot, Eelec, Enuc, Eiso_sum, EnucAB, D
        else:
            #for computing force, Eelec.sum()+EnucAB.sum() and backward is enough
            #index_add is used in total_energy and heat_formation function
            # P can be used as the initialization
            return Eelec, EnucAB, D


class ForceXL(torch.nn.Module):
    """
    get force for XL-BOMD
    """
    def __init__(self, seqm_parameters):
        super().__init__()
        self.energy = EnergyXL(seqm_parameters)
        self.seqm_parameters = seqm_parameters

    def forward(self, const, coordinates, species, P, learned_parameters=dict(), step=0):

        coordinates.requires_grad_(True)
        Hf, Etot, Eelec, Enuc, Eiso, EnucAB, D = \
            self.energy(const, coordinates, species, P, learned_parameters=learned_parameters, all_terms=True, step=step)
        L = Hf.sum()
        if const.do_timing:
            t0 = time.time()
        gv = [coordinates]
        gradients  = grad(L, gv)
        coordinates.grad = gradients[0]
        #"""
        if const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            const.timing["Force"].append(t1-t0)
        with torch.no_grad():
            force = -coordinates.grad.clone()
            coordinates.grad.zero_()
        #return force, Etot, D.detach()
        return force, Hf, D.detach()



class XL_BOMD(Molecular_Dynamics_Basic):
    """
    perform basic moleculer dynamics with verlocity_verlet algorithm, and in NVE ensemble
    separate get force, run one step, and run n steps is to make it easier to implement thermostats
    """
    def __init__(self,seqm_parameters, timestep=1.0, k=5, output={'molid':[0], 'thermo':1, 'dump':10, 'prefix':'md'}):
        """
        unit for timestep is femtosecond
        """
        super().__init__(seqm_parameters, timestep=1.0, output=output)
        self.seqm_parameters = seqm_parameters
        self.timestep = timestep
        self.conservative_force = ForceXL(seqm_parameters)
        self.force0 = Force(seqm_parameters)
        self.acc_scale = 0.009648532800137615
        self.vel_scale = 0.9118367323190634e-3
        self.kinetic_energy_scale = 1.0364270099032438e2
        #check Niklasson et al JCP 130, 214109 (2009)
        #coeff: kappa, alpha, c0, c1, ..., c9
        self.output = output
        self.coeffs = {3: [1.69,  150e-3,   -2.0,   3.0,    0.0,  -1.0],
                       4: [1.75,   57e-3,   -3.0,   6.0,   -2.0,  -2.0,   1.0],
                       5: [1.82,   18e-3,   -6.0,  14.0,   -8.0,  -3.0,   4.0,   -1.0],
                       6: [1.84,  5.5e-3,  -14.0,  36.0,  -27.0,  -2.0,  12.0,   -6.0,   1.0],
                       7: [1.86,  1.6e-3,  -36.0,  99.0,  -88.0,  11.0,  32.0,  -25.0,   8.0,  -1.0],
                       8: [1.88, 0.44e-3,  -99.0, 286.0, -286.0,  78.0,  78.0,  -90.0,  42.0, -10.0,  1.0],
                       9: [1.89, 0.12e-3, -286.0, 858.0, -936.0, 364.0, 168.0, -300.0, 184.0, -63.0, 12.0, -1.0]
                      }
        #
        self.k = k
        self.m = k+1
        self.kappa = self.coeffs[k][0]
        self.alpha = self.coeffs[k][1]
        cc = 1.00 # Scaled delta function approximation of Kernel coefficient cc = [0,1]
        tmp = torch.as_tensor(self.coeffs[k][2:])*self.alpha
        #P(n+1) = 2*P(n) - P(n-1) + cc*kappa*(D(n)-P(n)) + alpha*(c0*P(n) + c1*P(n-1) + ... ck*P(n-k))
        #       =  cc*kappa*D(n)
        #        + (2 - cc*kappa + alpha*c0)*P(n)
        #        + (alpha*c1 - 1) * P(n-1)
        #        + alpha*c2*P(n-2)
        #        + ...
        self.coeff_D = cc*self.kappa
        tmp[0] += (2.0 - cc*self.kappa)
        tmp[1] -= 1.0
        self.coeff = torch.nn.Parameter(tmp.repeat(2), requires_grad=False)

    def initialize(self, const, mass, coordinates, species, learned_parameters=dict()):
        #t=0, just use normal way
        f, D, _ = self.force0(const, coordinates, species, learned_parameters=learned_parameters)[:3]
        acc = f/mass*self.acc_scale
        return acc, D.detach()

    def get_force(self):
        """
        don't use the parent class get_force method
        """
        pass

    def one_step(self, const, step, mass, coordinates, velocities, species, acc, D, P, Pt, learned_parameters=dict()):
        #cindx: show in Pt, which is the latest P
        dt = self.timestep
        if const.do_timing:
            t0 = time.time()

        with torch.no_grad():
            velocities.add_(0.5*acc*dt)
            coordinates.add_(velocities*dt)

        #cindx = step%self.m
        #e.g k=5, m=6
        #coeff: c0, c1, c2, c3, c4, c5, c0, c1, c2, c3, c4, c5
        #Pt (0,1,2,3,4,5), step=6n  , cindx = 0, coeff[0:6]
        #Pt (1,2,3,4,5,0), step=6n+1, cindx = 1, coeff[1:7]
        #Pt (2,3,4,5,0,1), step=6n+2
        cindx = step%self.m
        P = self.coeff_D*D + torch.sum(self.coeff[cindx:(cindx+self.m)].reshape(-1,1,1,1)*Pt, dim=0)
        Pt[(self.m-1-cindx)] = P

        force, Etot, D = self.conservative_force(const, coordinates, species, P, learned_parameters=learned_parameters, step=step)
        D = D.detach()
        acc = force/mass*self.acc_scale
        with torch.no_grad():
            velocities.add_(0.5*acc*dt)
        if const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            const.timing["MD"].append(t1-t0)
        return coordinates, velocities, acc, D, P, Pt, Etot

    def run(self, const, steps, coordinates, velocities, species, learned_parameters=dict(), Pt=None):
        MASS = torch.as_tensor(const.mass)
        # put the padding virtual atom mass finite as for accelaration, F/m evaluation.
        MASS[0] = 1.0
        mass = MASS[species].unsqueeze(2)

        acc, D = self.initialize(const, mass, coordinates, species, learned_parameters=learned_parameters)
        if not torch.is_tensor(Pt):
            Pt = D.unsqueeze(0).expand((self.m,)+D.shape).clone()
        P = D.clone()

        #output={'molid':[0], 'thermo':1, 'dump':10, 'prefix':'md'}
        """
        print("#step(dt=%.2f) " % self.timestep, end='')
        for mol in self.output['molid']:
            print("T(%d)  Etot(%d)  " % (mol, mol), end="")
        print()
        """
        q0 = const.tore[species]

        for i in range(steps):
            coordinates, velocities, acc, D, P, Pt, Etot = self.one_step(const, i, mass, coordinates, velocities, species, \
                                                         acc, D, P, Pt, learned_parameters=learned_parameters)
            Ek, T = self.kinetic_energy(const, mass, species, velocities)
            q = q0 - self.atomic_charges(P) # unit +e, i.e. electron: -1.0
            if (i+1)%self.output['thermo']==0:
                print("md  %6d" % (i+1), end="")
                for mol in self.output['molid']:
                    print(" %f %f " % (T[mol], Etot[mol]+Ek[mol]), end="")
                print()
            if (i+1)%self.output['dump']==0:
                for mol in self.output['molid']:
                    fn = self.output['prefix'] + "." + str(mol) + ".xyz"
                    f = open(fn,'a+')
                    f.write("%d\nstep: %d\n" % (torch.sum(species[mol]>0), i+1))
                    for atom in range(coordinates.shape[1]):
                        if species[mol,atom]>0:
                            f.write("%2s %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e\n" % 
                                                    (const.label[species[mol,atom].item()],
                                                        coordinates[mol,atom,0],
                                                        coordinates[mol,atom,1],
                                                        coordinates[mol,atom,2], 
                                                        velocities[mol,atom,0],
                                                        velocities[mol,atom,1],
                                                        velocities[mol,atom,2],
                                                        q[mol,atom]))

                    f.close()
        return coordinates, velocities, acc, P, Pt

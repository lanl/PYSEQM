import torch
from .basics import *
import time
from seqm.ElectronicStructure import Electronic_Structure as esdriver
from .basics import Parser
from .seqm_functions.G_XL_LR import G
from seqm.seqm_functions.spherical_pot_force import Spherical_Pot_Force
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

from .tools import attach_profile_range
#not finished

debug = False

class Geometry_Optimization_SD(torch.nn.Module):
    """
    steepest descent algorithm for geometry optimization
    pass in function for Eelec and EnucAB and current coordinates
    use line search to choose best alpha
    """
    def __init__(self, seqm_parameters, alpha=0.01, force_tol=1.0e-4, max_evl=1000):
        """
        Constructor
        alpha : steepest descent mixing paramters, coordinates_new =  coordinates_old + alpha*force
        force_tol : force tolerance, stop criteria when all force components are less then this
        engery_tol : energy tolerance, stop criteria when delta \sum_{molecules} Etot / nmol <= engery_tot
                     i.e. stop when the difference of the total energy for the whole batch of molecules is smaller than this
        mex_evl : maximal number of evaluations/iterations
        """
        super().__init__()
        self.seqm_parameters = seqm_parameters
        self.esdriver = esdriver(self.seqm_parameters)
        self.alpha = alpha
        self.force_tol = force_tol
        self.max_evl = max_evl
        self.force = Force(seqm_parameters)

    def onestep(self, molecule, learned_parameters=dict(), *args, **kwargs):

        self.esdriver(molecule, learned_parameters=learned_parameters, P0=molecule.dm, dm_prop='SCF', *args, **kwargs)
        force = molecule.force
        with torch.no_grad():
            molecule.coordinates.add_(self.alpha*force)
        return force, molecule.Hf

    def run(self, molecule, learned_parameters=dict(), log=True, *args, **kwargs):
        dtype = molecule.coordinates.dtype
        device = molecule.coordinates.device
        nmol = molecule.coordinates.shape[0]
        molecule.coordinates.requires_grad_(True)
        Lold = torch.zeros(nmol,dtype=dtype,device=device)
        print("Step,  Max_Force,      Etot(eV),     dE(eV)")
        for i in range(self.max_evl):
            force, Lnew = self.onestep(molecule, learned_parameters=learned_parameters, *args, **kwargs)
            if torch.is_tensor(molecule.coordinates.grad):
                with torch.no_grad():
                    molecule.coordinates.grad.zero_()
            force_err = torch.max(torch.abs(force))
            energy_err = (Lnew-Lold).sum()/nmol
            if log:

                print("%d      " % (i+1), end="")
                print("%e " % force_err.item(), end="")

                """
                dis = torch.norm(coordinates[...,0,:]-coordinates[...,1,:], dim=1)
                for k in range(coordinates.shape[0]):
                    print("%e " % dis[k], end="")
                #"""
                for k in range(molecule.coordinates.shape[0]):
                    print("||%e %e " % (Lnew[k], Lnew[k]-Lold[k]), end="")
                print("")

            if (force_err>self.force_tol):
                Lold = Lnew
                continue
            else:
                break
        if i==(self.max_evl-1):
            print('not converged within %d step' % self.max_evl)
        else:
            if log:
                print("converged with %d step, Max Force = %e (eV/Ang), dE = %e (eV)" % (i+1, force_err.item(), energy_err.item()))

        return force_err, energy_err


class Molecular_Dynamics_Basic(torch.nn.Module):
    """
    perform basic moleculer dynamics with verlocity_verlet algorithm, and in NVE ensemble
    separate get force, run one step, and run n steps is to make it easier to implement thermostats
    """
    def __init__(self, seqm_parameters, Temp, timestep=1.0,  output={'molid':[0], 'thermo':1, 'dump':10, 'prefix':'md'}, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.seqm_parameters = seqm_parameters
        self.Temp = Temp
        self.timestep = timestep
        self.esdriver = esdriver(self.seqm_parameters)
        self.acc_scale = 0.009648532800137615
        self.vel_scale = 0.9118367323190634e-3
        self.kinetic_energy_scale = 1.0364270099032438e2
        self.output = output

    def initialize_velocity(self, molecule, vel_com=True):

        dtype = molecule.coordinates.dtype
        device = molecule.coordinates.device
        MASS = torch.as_tensor(molecule.const.mass)
        MASS[0] = 1.0
        mass = MASS[molecule.species].unsqueeze(2)
        scale = torch.sqrt(self.Temp/mass)*self.vel_scale
        molecule.velocities = torch.randn(molecule.coordinates.shape, dtype=dtype, device=device)*scale
        molecule.velocities[molecule.species==0,:] = 0.0
        if vel_com:
            #remove center of mass velocity
            print('Initialize velocities: zero_com')
            self.zero_com(molecule)
        return molecule.velocities

    def zero_com(self, molecule):
        """
        remove center of mass
        remove center of mass velocity
        remove angular momentum
        scale temperature back
        mass shape (nmol, molsize, 1)
        velocities shape (nmol, molsize, 3)
        """
        #mass for padding atom is 0.0
        mass = molecule.const.mass[molecule.species].unsqueeze(2)
        M = torch.sum(mass,dim=1,keepdim=True)
        _, T0 = self.kinetic_energy(molecule)
        with torch.no_grad():
            r_com = torch.sum(mass*molecule.coordinates,dim=1,keepdim=True)/M
            molecule.coordinates.sub_(r_com)
            v_com = torch.sum(mass*molecule.velocities,dim=1,keepdim=True)/M
            molecule.velocities.sub_(v_com)
            L = torch.sum(mass * torch.cross(molecule.coordinates, molecule.velocities, dim=2), dim=1)
            I = torch.sum(mass * torch.norm(molecule.coordinates, dim=2, keepdim=True)**2, dim=1, keepdim=True) \
                * torch.eye(3, dtype=molecule.coordinates.dtype, device=molecule.coordinates.device).reshape(1,3,3) \
                - torch.sum( mass.unsqueeze(3) * molecule.coordinates.unsqueeze(3) * molecule.coordinates.unsqueeze(2), dim=1)
            omega = torch.linalg.solve(I, L.unsqueeze(2))
            molecule.velocities.add_( torch.cross(molecule.coordinates, omega.reshape(-1,1,3).repeat(1,molecule.coordinates.shape[1],1)) )
            _, T1 = self.kinetic_energy(molecule)
            alpha = torch.sqrt(T0/T1)
            molecule.velocities.mul_(alpha.reshape(-1,1,1))
            #Lnew = torch.sum(mass * torch.cross(coordinates, velocities), dim=1)
            #_, Tnew = self.kinetic_energy(const, mass, species, velocities)
            #print("After remove angular momentum: ", Lnew, T0-Tnew)
            del r_com, v_com, L, I, omega, T0, T1, alpha #, Lnew, Tnew


    def kinetic_energy(self, molecule):
        Ek = torch.sum(0.5*molecule.mass*molecule.velocities**2,dim=(1,2))*self.kinetic_energy_scale
        #Ek = torch.sum(0.5*mass*velocities**2)*self.kinetic_energy_scale
        Temp = Ek*1.160451812e4/(1.5*torch.sum(molecule.species>0,dim=1).type(Ek.dtype))
        return Ek, Temp
    
    @staticmethod
    def atomic_charges(P, n_orbital=4):
        """
        get atomic charge based on single-particle density matrix P
        n_orbital : number of orbitals for each atom, default is 4
        """
        n_molecule = P.shape[0]
        n_atom = P.shape[1]//n_orbital
        q = P.diagonal(dim1=1,dim2=2).reshape(n_molecule, n_atom, n_orbital).sum(axis=2)
        return q
    
    @staticmethod
    def dipole(q, coordinates):
        return torch.sum(q.unsqueeze(2)*coordinates, axis=1)

    
    def screen_output(self, i, T, Ek, L):
        if i==0:
            print("Step,    Temp,    E(kinetic),  E(potential),  E(total)")
        if (i+1)%self.output['thermo']==0:
                print("%6d" % (i+1), end="")
                for mol in self.output['molid']:
                    print(" %8.2f   %e %e %e || " % (T[mol],   Ek[mol], L[mol], L[mol]+Ek[mol]), end="")
                print()
    
    def dump(self, i, molecule, velocities, q, T, Ek, L, forces, e_gap, Err=None, **kwargs):
        
        if Err == None:
            Err=np.zeros(T.shape)

        if (i+1)%self.output['dump']==0:
            for mol in self.output['molid']:
                fn = self.output['prefix'] + "." + str(mol) + ".xyz"
                f = open(fn,'a+')
                f.write("{}\nstep: {}  T= {:12.3f}K  Ek= {:12.6f}  Ep= {:12.6f}  E_gap= {:12.6f}  Err= {:24.16f}  time_stamp= {:.4f}\n".format(torch.sum(molecule.species[mol]>0), i+1, T[mol], Ek[mol], L[mol], e_gap[mol,0], Err[mol], time.time()))
                    
                for atom in range(molecule.coordinates.shape[1]):
                    if molecule.species[mol,atom]>0:
                        f.write("{} {:15.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f}\n".format(
                                                    molecule.const.label[molecule.species[mol,atom].item()],
                                                    molecule.coordinates[mol,atom,0],
                                                    molecule.coordinates[mol,atom,1],
                                                    molecule.coordinates[mol,atom,2], 
                                                    velocities[mol,atom,0],
                                                    velocities[mol,atom,1],
                                                    velocities[mol,atom,2],
                                                    forces[mol,atom,0],
                                                    forces[mol,atom,1],
                                                    forces[mol,atom,2],
                                                    q[mol,atom]))
                f.close()
                
            if 'Info_log' in kwargs:
                for mol in self.output['molid']:
                    fn = self.output['prefix'] + "." + str(mol) + ".Info.txt"
                    f = open(fn,'a+')
                    f.write("\nstep: {}\n".format(i+1))
                    for log in kwargs['Info_log']:
                        f.write("  {}{} \n".format(log[0], log[1][mol]))
                    f.close()
                    

    def scale_velocities(self, i, velocities, T, scale_vel):
        #freq, T0 = scale_vel
        if (i)%scale_vel[0]==0:
            alpha = torch.sqrt(scale_vel[1]/T)
            velocities.mul_(alpha.reshape(-1,1,1))
            return True
        return False

    def control_shift(self, velocities, Ek, Eshift):
        alpha = torch.sqrt((Ek-Eshift)/Ek)
        alpha[~torch.isfinite(alpha)]=0.0
        velocities.mul_(alpha.reshape(-1,1,1))

    def one_step(self, molecule, learned_parameters=dict(), *args, **kwargs):
        dt = self.timestep
        """
        MASS = torch.as_tensor(const.mass)
        # put the padding virtual atom mass finite as for accelaration, F/m evaluation.
        MASS[0] = 1.0
        mass = MASS[species].unsqueeze(2)
        """ 
        if molecule.const.do_timing:
            t0 = time.time()
        if not torch.is_tensor(molecule.acc):
            self.esdriver(molecule, learned_parameters=learned_parameters, P0=molecule.dm, dm_prop='SCF', *args, **kwargs)
            with torch.no_grad():
                molecule.acc = molecule.force/molecule.mass*self.acc_scale
        with torch.no_grad():
            molecule.velocities.add_(0.5*molecule.acc*dt)
            molecule.coordinates.add_(molecule.velocities*dt)
        self.esdriver(molecule, learned_parameters=learned_parameters, P0=molecule.dm, dm_prop='SCF', *args, **kwargs)

        with torch.no_grad():
            molecule.acc = molecule.force/molecule.mass*self.acc_scale
            molecule.velocities.add_(0.5*molecule.acc*dt)
        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            molecule.const.timing["MD"].append(t1-t0)

    def run(self, molecule, steps, learned_parameters=dict(), reuse_P=True, remove_com=[False,1000], *args, **kwargs):

        q0 = molecule.const.tore[molecule.species]
        E0 = None

        for i in range(steps):
            
            start_time = time.time()
            self.one_step(molecule, learned_parameters=learned_parameters, P=molecule.dm, *args, **kwargs)

            with torch.no_grad():
                if torch.is_tensor(molecule.coordinates.grad):
                    molecule.coordinates.grad.zero_()
                
                if not reuse_P:
                    molecule.dm = None
                if remove_com[0]:
                    if i%remove_com[1]==0:
                        self.zero_com(molecule)

                Ek, T = self.kinetic_energy(molecule)
                if not torch.is_tensor(E0):
                    E0 = molecule.Hf+Ek
                
                if 'scale_vel' in kwargs and 'control_energy_shift' in kwargs:
                    raise ValueError("Can't scale velocities to fix temperature and fix energy shift at same time")
                
                #scale velocities to control temperature
                if 'scale_vel' in kwargs:
                    # kwargs["scale_vel"] = [freq, T(target)]
                    flag = self.scale_velocities(i, molecule.velocities, T, kwargs["scale_vel"])
                    if flag:
                        Ek, T = self.kinetic_energy(molecule)
                
                #control energy shift
                if 'control_energy_shift' in kwargs and kwargs['control_energy_shift']:
                    #scale velocities to adjust kinetic energy and compenstate the energy shift
                    Eshift = Ek + molecule.Hf - E0
                    self.control_shift(molecule.velocities, Ek, Eshift)
                    Ek, T = self.kinetic_energy(molecule)
                    del Eshift
                
                
                self.screen_output(i, T, Ek, molecule.Hf)
                dump_kwargs = {}
                if 'Info_log' in kwargs and kwargs['Info_log']:
                    dump_kwargs['Info_log'] = [
                        ['Orbital energies:\n', ['    Occupied:\n      ' + str(x[0: i])[1:-1].replace('\n', '\n     ') + '\n    Virtual:\n      ' + str(x[i:])[1:-1].replace('\n', '\n     ') for x, i in zip(np.round(molecule.e_mo.cpu().numpy(), 5), molecule.nocc)]],
                        
                        ['dipole(x,y,z): ', [str(x)[1:-1] for x in np.round(molecule.d.cpu().numpy(), 6)]],

                                              ]
                    
                self.dump(i, molecule, molecule.velocities, molecule.q, T, Ek, molecule.Hf, molecule.force, molecule.e_gap, **dump_kwargs)
            del Ek, T
            if i%1000==0:
                torch.cuda.empty_cache()
            
            if debug:
                print(time.time() - start_time)

        return molecule.coordinates, molecule.velocities, molecule.acc


class Molecular_Dynamics_Langevin(Molecular_Dynamics_Basic):
    """
    molecular dynamics with langevin thermostat
    #same formula as in lammps
    """

    def __init__(self, damp=1.0, *args, **kwargs):
        """
        damp is damping factor in unit of time (fs)
        Temp : temperature in unit of Kelvin
        F = Fc + Ff + Fr
        Ff = - (m / damp) v
        Fr = sqrt(2 Kb T m / (dt damp))*R(t)
        each component of R(t) ~ N(0,1)
        <R_ij(t)>=0
        <R_ij(t) * R_ik(t) > = delta_jk
        """
        
        self.damp = damp
        super().__init__(*args, **kwargs)
        #self.T = Temp
        # Fr = sqrt(2 Kb T m / (dt damp))*R(t)
        #    = sqrt(2.0) * sqrt(kb T/m) * m/sqrt(dt damp) * R(t)
        # self.vel_scale change sqrt(kb T/m) into Angstrom/fs
        self.Fr_scale = 0.09450522179973914

    def one_step(self, molecule, learned_parameters=dict(), *args, **kwargs):
        dt = self.timestep
        """
        MASS = torch.as_tensor(const.mass)
        # put the padding virtual atom mass finite as for accelaration, F/m evaluation.
        MASS[0] = 1.0
        mass = MASS[species].unsqueeze(2)
        """ 
        if molecule.const.do_timing:
            t0 = time.time()
        if not torch.is_tensor(molecule.acc):
            self.esdriver(molecule, learned_parameters=learned_parameters, P0=molecule.dm, dm_prop='SCF', *args, **kwargs)
            Fc = molecule.force
            Ff = -molecule.mass * molecule.velocities / self.damp / self.acc_scale
            Fr = self.Fr_scale * torch.sqrt(2.0*self.Temp*molecule.mass/self.timestep/self.damp)*torch.randn(Fc.shape, dtype=Fc.dtype, device=Fc.device)
            molecule.force = Fc+Ff+Fr
            molecule.force[molecule.species==0,:] = 0.0

            with torch.no_grad():
                molecule.acc = molecule.force/molecule.mass*self.acc_scale
        with torch.no_grad():
            molecule.velocities.add_(0.5*molecule.acc*dt)
            molecule.coordinates.add_(molecule.velocities*dt)
        self.esdriver(molecule, learned_parameters=learned_parameters, P0=molecule.dm, dm_prop='SCF', *args, **kwargs)
        Fc = molecule.force
        Ff = -molecule.mass * molecule.velocities / self.damp / self.acc_scale
        Fr = self.Fr_scale * torch.sqrt(2.0*self.Temp*molecule.mass/self.timestep/self.damp)*torch.randn(Fc.shape, dtype=Fc.dtype, device=Fc.device)
        molecule.force = Fc+Ff+Fr
        molecule.force[molecule.species==0,:] = 0.0

        with torch.no_grad():
            molecule.acc = molecule.force/molecule.mass*self.acc_scale
            molecule.velocities.add_(0.5*molecule.acc*dt)
        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            molecule.const.timing["MD"].append(t1-t0)


            
            
            
class XL_BOMD(Molecular_Dynamics_Basic):
    """
    perform basic moleculer dynamics with verlocity_verlet algorithm, and in NVE ensemble
    separate get force, run one step, and run n steps is to make it easier to implement thermostats
    """
    def __init__(self, xl_bomd_params=dict(), damp=False, *args, **kwargs):
        """
        unit for timestep is femtosecond
        """
        self.k = xl_bomd_params['k']
        self.xl_bomd_params = xl_bomd_params

        self.damp = damp
        self.Fr_scale = 0.09450522179973914
        super().__init__(*args, **kwargs)
        self.esdriver = esdriver(self.seqm_parameters)
        #check Niklasson et al JCP 130, 214109 (2009)
        #coeff: kappa, alpha, c0, c1, ..., c9
        self.coeffs = {3: [1.69,  150e-3,   -2.0,   3.0,    0.0,  -1.0],
                       4: [1.75,   57e-3,   -3.0,   6.0,   -2.0,  -2.0,   1.0],
                       5: [1.82,   18e-3,   -6.0,  14.0,   -8.0,  -3.0,   4.0,   -1.0],
                       6: [1.84,  5.5e-3,  -14.0,  36.0,  -27.0,  -2.0,  12.0,   -6.0,   1.0],
                       7: [1.86,  1.6e-3,  -36.0,  99.0,  -88.0,  11.0,  32.0,  -25.0,   8.0,  -1.0],
                       8: [1.88, 0.44e-3,  -99.0, 286.0, -286.0,  78.0,  78.0,  -90.0,  42.0, -10.0,  1.0],
                       9: [1.89, 0.12e-3, -286.0, 858.0, -936.0, 364.0, 168.0, -300.0, 184.0, -63.0, 12.0, -1.0]
                      }
        #
        self.m = xl_bomd_params['k']+1
        self.kappa = self.coeffs[xl_bomd_params['k']][0]
        self.alpha = self.coeffs[xl_bomd_params['k']][1]
        cc = 1.00 # Scaled delta function approximation of Kernel coefficient cc = [0,1]
        tmp = torch.as_tensor(self.coeffs[xl_bomd_params['k']][2:])*self.alpha
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

    def initialize(self, molecule, learned_parameters=dict(), *args, **kwargs):
        #t=0, just use normal way

        if not torch.is_tensor(molecule.dm):
            print('Doing initialization')
            self.esdriver(molecule, learned_parameters=learned_parameters, *args, **kwargs)
            if self.damp:
                Ff = -molecule.mass * molecule.velocities / self.damp / self.acc_scale
                Fr = self.Fr_scale * torch.sqrt(2.0*self.Temp*molecule.mass/self.timestep/self.damp)*torch.randn(molecule.force.shape,
                                                                                                                 dtype=molecule.force.dtype, device=molecule.force.device)
                molecule.force += Ff+Fr
                molecule.force[molecule.species==0,:] = 0.0
        else:
            print('Already initialized')

        with torch.no_grad():
            molecule.acc = molecule.force/molecule.mass*self.acc_scale
            #print('acc: ',molecule.acc)


    def one_step(self, molecule, step, P, Pt, learned_parameters=dict(), *args, **kwargs):
        #cindx: show in Pt, which is the latest P 
        dt = self.timestep
        if molecule.const.do_timing:
            t0 = time.time()

        with torch.no_grad():
            # leapfrog velocity Verlet
            molecule.velocities.add_(0.5*molecule.acc*dt)
            molecule.coordinates.add_(molecule.velocities*dt)                

            #cindx = step%self.m
            #e.g k=5, m=6
            #coeff: c0, c1, c2, c3, c4, c5, c0, c1, c2, c3, c4, c5
            #Pt (0,1,2,3,4,5), step=6n  , cindx = 0, coeff[0:6]
            #Pt (1,2,3,4,5,0), step=6n+1, cindx = 1, coeff[1:7]
            #Pt (2,3,4,5,0,1), step=6n+2
            cindx = step%self.m
            # eq. 22 in https://doi.org/10.1063/1.3148075
            
            #### Scaling delta function. Use eq with c if stability problems occur.
            c = 0.9
            P = self.coeff_D*c*molecule.dm + self.coeff_D*(1-c)*P + torch.sum(self.coeff[cindx:(cindx+self.m)].reshape(-1,1,1,1)*Pt, dim=0)
            
            #P = self.coeff_D*molecule.dm + torch.sum(self.coeff[cindx:(cindx+self.m)].reshape(-1,1,1,1)*Pt, dim=0)
            Pt[(self.m-1-cindx)] = P
    
        self.esdriver(molecule, learned_parameters=learned_parameters, xl_bomd_params = self.xl_bomd_params, P0=P, dm_prop='XL-BOMD', *args, **kwargs)
        if self.damp:
            #print('DAMPING')
            Ff = -molecule.mass * molecule.velocities / self.damp / self.acc_scale
            Fr = self.Fr_scale * torch.sqrt(2.0*self.Temp*molecule.mass/self.timestep/self.damp)*torch.randn(molecule.force.shape,
                                                                                                             dtype=molecule.force.dtype, device=molecule.force.device)
            molecule.force += Ff+Fr
            molecule.force[molecule.species==0,:] = 0.0
        
        with torch.no_grad():
            molecule.acc = molecule.force/molecule.mass*self.acc_scale
            molecule.velocities.add_(0.5*molecule.acc*dt)
        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            molecule.const.timing["MD"].append(t1-t0)
        return P, Pt

    def run(self, molecule, steps, learned_parameters=dict(), Pt=None, remove_com=[False,1000], *args, **kwargs):
        
        self.initialize(molecule, learned_parameters=learned_parameters, *args, **kwargs)
        with torch.no_grad():
            if not torch.is_tensor(Pt):
                Pt = molecule.dm.unsqueeze(0).expand((self.m,)+molecule.dm.shape).clone()
                if 'max_rank' in self.xl_bomd_params:
                    molecule.dP2dt2 = torch.zeros(molecule.dm.shape, dtype=molecule.dm.dtype, device=molecule.dm.device)
            P = molecule.dm.clone()

        E0 = None

        for i in range(steps):
            start_time = time.time()

            P, Pt = self.one_step(molecule, i, P, Pt, learned_parameters=learned_parameters, *args, **kwargs)

            with torch.no_grad():
                if torch.is_tensor(molecule.coordinates.grad):
                    molecule.coordinates.grad.zero_()
                
                if remove_com[0]:
                    if i%remove_com[1]==0:
                        self.zero_com(molecule)
                
                Ek, T = self.kinetic_energy(molecule)
                if not torch.is_tensor(E0):
                    E0 = molecule.Hf + molecule.Electronic_entropy + Ek
                
                if 'scale_vel' in kwargs and 'control_energy_shift' in kwargs:
                    raise ValueError("Can't scale velocities to fix temperature and fix energy shift at same time")
                
                #scale velocities to control temperature
                if 'scale_vel' in kwargs:
                    # kwargs["scale_vel"] = [freq, T(target)]
                    flag = self.scale_velocities(i, molecule.velocities, T, kwargs["scale_vel"])
                    if flag:
                        Ek, T = self.kinetic_energy(molecule)
                
                #control energy shift
                if 'control_energy_shift' in kwargs and kwargs['control_energy_shift']:
                    #scale velocities to adjust kinetic energy and compenstate the energy shift
                    Eshift = Ek + molecule.Hf + molecule.Electronic_entropy - E0
                    self.control_shift(molecule.velocities, Ek, Eshift)
                    Ek, T = self.kinetic_energy(molecule)
                self.screen_output(i, T, Ek, molecule.Hf + molecule.Electronic_entropy)
                dump_kwargs = {}
                if 'Info_log' in kwargs and kwargs['Info_log']:
                    if 'max_rank' in self.xl_bomd_params:
                        dump_kwargs['Info_log'] = [
                            ['Orbital energies (eV):\n', ['    Occupied:\n      ' + str(x[0: i])[1:-1].replace('\n', '\n     ') +\
                                                          '\n    Virtual:\n      ' + str(x[i:])[1:-1].replace('\n', '\n     ') for x, i in \
                                                          zip(np.round(molecule.e_mo.cpu().numpy(), 5), molecule.nocc)] ],
                            ['dipole(x,y,z): ', [str(x)[1:-1] for x in np.round(molecule.d.cpu().numpy(), 6)] ],

                            ['Electronic entropy contribution (eV): ', molecule.Electronic_entropy],

                            ['Fermi occupancies:\n', ['    Occupied:\n      ' + str(x[0: i])[1:-1].replace('\n', '\n     ') + \
                                                      '\n    Virtual:\n      ' + str(x[i:])[1:-1].replace('\n', '\n     ') for x, i in \
                                                      zip(np.round(molecule.Fermi_occ.cpu().numpy(), 6), molecule.nocc)] ],

                            ['Rank-m Krylov subspace approximation error: ', molecule.Krylov_Error], ]
                    else:
                        dump_kwargs['Info_log'] = [
                            ['Orbital energies (eV):\n', ['    Occupied:\n      ' + str(x[0: i])[1:-1].replace('\n', '\n     ') +\
                                                          '\n    Virtual:\n      ' + str(x[i:])[1:-1].replace('\n', '\n     ') for x, i in \
                                                          zip(np.round(molecule.e_mo.cpu().numpy(), 5), molecule.nocc)] ],
                            ['dipole(x,y,z): ', [str(x)[1:-1] for x in np.round(molecule.d.cpu().numpy(), 6)] ], ]

                self.dump(i, molecule, molecule.velocities, molecule.q, T, Ek, molecule.Hf + molecule.Electronic_entropy, molecule.force,
                          molecule.e_gap, molecule.Krylov_Error, **dump_kwargs)
            del T, Ek, dump_kwargs
            if i%1000==0:
                torch.cuda.empty_cache()
            
            if debug:
                print(time.time() - start_time)

        return molecule.coordinates, molecule.velocities, molecule.acc



class KSA_XL_BOMD(XL_BOMD):
    """
    perform basic moleculer dynamics with verlocity_verlet algorithm, and in NVE ensemble
    separate get force, run one step, and run n steps is to make it easier to implement thermostats
    """
    def __init__(self, *args, **kwargs):
        """
        unit for timestep is femtosecond
        """
        super().__init__(*args, **kwargs)

    @attach_profile_range("KSA_XL_BOMD_ONESTEP")
    def one_step(self, molecule, step, P, Pt, learned_parameters=dict(), *args, **kwargs):
        #cindx: show in Pt, which is the latest P 
        dt = self.timestep
        if molecule.const.do_timing:
            t0 = time.time()

        with torch.no_grad():
            # leapfrog velocity Verlet
            molecule.velocities.add_(0.5*molecule.acc*dt)
            
            cindx = step%self.m
            # eq. 22 in https://doi.org/10.1063/1.3148075
            P = self.coeff_D*molecule.dP2dt2 + self.coeff_D*P + torch.sum(self.coeff[cindx:(cindx+self.m)].reshape(-1,1,1,1)*Pt, dim=0)
            Pt[(self.m-1-cindx)] = P

            molecule.coordinates.add_(molecule.velocities*dt)

        self.esdriver(molecule, learned_parameters=learned_parameters, P0=P, xl_bomd_params=self.xl_bomd_params, dm_prop='XL-BOMD', *args, **kwargs)
        if self.damp:
            #print('DAMPING')
            Ff = -molecule.mass * molecule.velocities / self.damp / self.acc_scale
            Fr = self.Fr_scale * torch.sqrt(2.0*self.Temp*molecule.mass/self.timestep/self.damp)*torch.randn(molecule.force.shape,
                                                                                                             dtype=molecule.force.dtype, device=molecule.force.device)
            molecule.force += Ff+Fr
            molecule.force[molecule.species==0,:] = 0.0
        
        spherical_pot = False ########################################################################################==========
        if spherical_pot:
            with torch.no_grad():
                spherical_pot_E, spherical_pot_force = Spherical_Pot_Force(molecule, 5.9, k=1.1)
                molecule.Hf += spherical_pot_E
                molecule.force = molecule.force + spherical_pot_force

        with torch.no_grad():
            molecule.acc = molecule.force/molecule.mass*self.acc_scale
            molecule.velocities.add_(0.5*molecule.acc*dt)
        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            molecule.const.timing["MD"].append(t1-t0)
        return P, Pt


"""
1 eV = 1.602176565e-19 J
1 Angstrom = 1.0e-10 m
1 eV/Angstrom = 1.602176565e-09 N
1 AU = 1.66053906660e-27 kg
1 femtosecond = 1.0e-15 second
1 Angstrom/fs = 1.0e5 m/s

# accelaration scale
1 eV/Angstroms / (grams/mol) = 1.602176565e-09 N / 1.66053906660e-27 kg
 = 1.602176565e-09/1.66053906660e-27 m/s^2 = 1.602176565e-09/1.66053906660e-27 * 1.0e-20 Angstrom/fs^2
 = 1.602176565/1.66053906660*0.01 Angstrom/fs^2 = 0.009648532800137615 Angstrom/fs^2

1 eV = 1.160451812e4 Kelvin

# vel_scale = sqrt(kb*T/m)
kb*T/m: 1 Kelvin/ AU = 1.0/1.160451812e4 * 1.602176565e-19 / 1.66053906660e-27 * m^2/s^2
= 1.0/1.160451812e4 * 1.602176565e-19 / 1.66053906660e-27 * 1.0e-10 Angstrom^2/fs^2
= 1.0/1.160451812 * 1.602176565 / 1.6605390666 * 1.0e-6 Angstrom^2/fs^2
= 0.8314462264063073e-6 Angstrom^2/fs^2

sqrt(Kelvin/ AU) = 0.9118367323190634e-3 Angstrom/fs

# kinetic energy scale
AU*(Angstrom/fs)^2 = 1.66053906660e-27 kg * 1.0e10 m^2/s^2 = 1.66053906660e-17 J
= 1.66053906660e-17/1.602176565e-19 eV = 1.0364270099032438e2 eV

# random force scale
random force unit coversion
Fr = sqrt(2 Kb T m / (dt damp))*R(t)
Fr unit: sqrt(Kelvin*AU/(fs^2)) ==> eV/Angstrom
1 sqrt(Kelvin*AU/(fs^2)) = sqrt(1.0/1.160451812e4 eV * 1.66053906660e-27 kg)/fs
= sqrt(1.0/1.160451812e4 * 1.602176565e-19 J * 1.66053906660e-27 kg)/fs
= sqrt(1.0/1.160451812e4 * 1.602176565e-19 * 1.66053906660e-27)/1.0e-15 kg*m/s^2
= sqrt(1.0/1.160451812e4 * 1.602176565e-19 * 1.66053906660e-27)/1.0e-15 J/m
= sqrt(1.0/1.160451812e4 * 1.602176565e-19 * 1.66053906660e-27)/1.0e-15 /1.602176565e-19 / 1.0e10  eV/Angstrom
= sqrt(1.0/1.160451812e4 / 1.602176565e-19 * 1.66053906660e-27) * 1.0e5 eV/Angstrom
= sqrt(1.0/1.160451812e4 / 1.602176565e-19 * 1.66053906660e-27) * 1.0e5 eV/Angstrom
= sqrt(1.0/1.160451812 / 1.602176565 * 1.66053906660) * 0.1 eV/Angstrom
= 0.09450522179973914 eV/Angstrom
"""
#acc ==> Angstrom/fs^2

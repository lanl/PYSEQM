import torch
from .basics import *
import time
#not finished
class Geometry_Optimization_SD_LS(torch.nn.Module):
    """
    steepest descent algorithm for geometry optimization
    pass in function for Eelec and EnucAB and current coordinates
    use line search to choose best alpha
    """
    def __init__(self, seqm_parameters, alpha=0.01, force_tol=1.0e-4, energy_tol=1.0e-4, max_evl=1000):
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
        self.alpha = alpha
        self.force_tol = force_tol
        self.energy_tol = energy_tol
        self.max_evl = max_evl
        self.force = Force(seqm_parameters)

    def onestep(self, const, coordinates, species, alpha, learned_parameters=dict(), P0=None):
        force, P, L = self.force(const, coordinates, species, learned_parameters=learned_parameters, P0=P0)[:3]
        P = P.detach()
        alphas = alpha.unsqueeze(1) * torch.tensor([0.5, 0.75, 1.0, 1.25, 1.5], dtype=coordinates.dtype, device=coordinates.device).reshape((1,-1))
        with torch.no_grad():
            eng = torch.zeros_like(alphas)
            for k in range(alphas.shape[1]):
                eng[...,k] = self.force.energy(const, coordinates+force*alphas[...,k].reshape(-1,1,1), species, learned_parameters=learned_parameters, all_terms=True, P0=P0)[1]
            #print(eng)
            index1 = torch.arange(alphas.shape[0], dtype=torch.int64, device=coordinates.device)
            index2 = torch.argmin(eng,dim=1)
            alpha = alphas[index1, index2]
            coordinates.add_(alpha.reshape(-1,1,1)*force)
        return coordinates, force, P, L, alpha

    def run(self, const, coordinates, species, learned_parameters=dict(), P=None, log=True):
        dtype = coordinates.dtype
        device = coordinates.device
        nmol=coordinates.shape[0]
        coordinates.requires_grad_(True)
        alpha = torch.zeros(nmol,dtype=coordinates.dtype, device=coordinates.device)+self.alpha

        Lold = torch.zeros(nmol,dtype=dtype,device=device)
        for i in range(self.max_evl):
            coordinates, force, P, Lnew, alpha = self.onestep(const, coordinates, species, alpha, learned_parameters=dict(), P0=P)
            force_err = torch.max(torch.abs(force))
            energy_err = (Lnew-Lold).sum()/nmol
            if log:

                print("%d " % (i+1), end="")
                print("%e " % force_err.item(), end="")

                """
                dis = torch.norm(coordinates[...,0,:]-coordinates[...,1,:], dim=1)
                for k in range(coordinates.shape[0]):
                    print("%e " % dis[k], end="")
                #"""
                for k in range(coordinates.shape[0]):
                    print("%e %e %e " % (Lnew[k], Lold[k], Lnew[k]-Lold[k]), end="")
                print("")

            #if (force_err>self.force_tol and torch.abs(energy_err)>self.energy_tol):
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

        return coordinates,force_err, energy_err

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
        self.alpha = alpha
        self.force_tol = force_tol
        self.max_evl = max_evl
        self.force = Force(seqm_parameters)

    def onestep(self, const, coordinates, species, learned_parameters=dict(), P0=None):
        force, P, L = self.force(const, coordinates, species, learned_parameters=learned_parameters, P0=P0)[:3]
        P = P.detach()
        with torch.no_grad():
            coordinates.add_(self.alpha*force)
        return coordinates, force, P, L

    def run(self, const, coordinates, species, learned_parameters=dict(), P=None, log=True):
        dtype = coordinates.dtype
        device = coordinates.device
        nmol=coordinates.shape[0]
        coordinates.requires_grad_(True)
        Lold = torch.zeros(nmol,dtype=dtype,device=device)
        for i in range(self.max_evl):
            coordinates, force, P, Lnew = self.onestep(const, coordinates, species, learned_parameters=dict(), P0=P)
            force_err = torch.max(torch.abs(force))
            energy_err = (Lnew-Lold).sum()/nmol
            if log:

                print("%d " % (i+1), end="")
                print("%e " % force_err.item(), end="")

                """
                dis = torch.norm(coordinates[...,0,:]-coordinates[...,1,:], dim=1)
                for k in range(coordinates.shape[0]):
                    print("%e " % dis[k], end="")
                #"""
                for k in range(coordinates.shape[0]):
                    print("%e %e %e " % (Lnew[k], Lold[k], Lnew[k]-Lold[k]), end="")
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

        return coordinates,force_err, energy_err

class Molecular_Dynamics_Basic(torch.nn.Module):
    """
    perform basic moleculer dynamics with verlocity_verlet algorithm, and in NVE ensemble
    separate get force, run one step, and run n steps is to make it easier to implement thermostats
    """
    def __init__(self,seqm_parameters, timestep=1.0, output={'molid':[0], 'thermo':1, 'dump':10, 'prefix':'md'}):
        """
        unit for timestep is femtosecond
        output: [molecule id list, frequency N, prefix]
            molecule id in the list are output, staring from 0 to nmol-1
            geometry is writted every dump step to the file with name prefix + molid + .xyz
            step, temp, and total energy is print to screens for select molecules every thermo
        """
        super().__init__()
        self.seqm_parameters = seqm_parameters
        self.timestep = timestep
        self.conservative_force = Force(seqm_parameters)
        self.acc_scale = 0.009648532800137615
        self.vel_scale = 0.9118367323190634e-3
        self.kinetic_energy_scale = 1.0364270099032438e2
        self.output = output


    def initialize_velocity(self, const, coordinates, species, Temp=300.0, vel_com=True):
        dtype = coordinates.dtype
        device = coordinates.device
        MASS = torch.as_tensor(const.mass)
        MASS[0] = 1.0
        mass = MASS[species].unsqueeze(2)
        scale = torch.sqrt(Temp/mass)*self.vel_scale
        velocities = torch.randn(coordinates.shape, dtype=dtype, device=device)*scale
        velocities[species==0,:] = 0.0
        if vel_com:
            #remove center of mass velocity
            self.zero_com(const, species, velocities)
        return velocities

    def zero_com(self, const, species, velocities):

        """
        remove center of mass velocity
        mass shape (nmol, molsize, 1)
        velocities shape (nmol, molsize, 3)
        """
        #mass for padding atom is 0.0
        mass = const.mass[species].unsqueeze(2)
        com = torch.sum(mass*velocities,dim=1,keepdim=True)/torch.sum(mass,dim=1,keepdim=True)
        velocities.sub_(com)

    def kinetic_energy(self, const, mass, species, velocities):
        Ek = torch.sum(0.5*mass*velocities**2,dim=(1,2))*self.kinetic_energy_scale
        #Ek = torch.sum(0.5*mass*velocities**2)*self.kinetic_energy_scale
        Temp = Ek*1.160451812e4/(1.5*torch.sum(species>0,dim=1).type(Ek.dtype))
        return Ek, Temp

    def get_force(self, const, mass, coordinates, velocities, species, learned_parameters=dict(), P0=None, step=0):
        """
        return force in unit of eV/Angstrom
        return force, density matrix, total energy of this batch
        """
        F, P, Etot, Hf = self.conservative_force(const, coordinates, species, learned_parameters=learned_parameters, P0=P0, step=step)[:4]
        L = Hf
        P = P.detach()
        return F, P, L

    def one_step(self, const, mass, coordinates, velocities, species, acc=None, learned_parameters=dict(), P=None, step=0):
        dt = self.timestep
        """
        MASS = torch.as_tensor(const.mass)
        # put the padding virtual atom mass finite as for accelaration, F/m evaluation.
        MASS[0] = 1.0
        mass = MASS[species].unsqueeze(2)
        """

        if not torch.is_tensor(acc):
            force, P, _ = self.get_force(const, mass, coordinates, velocities, species, learned_parameters=learned_parameters, P0=P, step=step)
            acc = force/mass*self.acc_scale
        if const.do_timing:
            t0 = time.time()
        with torch.no_grad():
            velocities.add_(0.5*acc*dt)
            coordinates.add_(velocities*dt)

        force, P, L = self.get_force(const, mass, coordinates, velocities, species, learned_parameters=learned_parameters, P0=P, step=step)
        acc = force/mass*self.acc_scale
        with torch.no_grad():
            velocities.add_(0.5*acc*dt)
        if const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            const.timing["MD"].append(t1-t0)

        return coordinates, velocities, acc, P, L

    def run(self, const, steps, coordinates, velocities, species, learned_parameters=dict(), reuse_P=True):

        MASS = torch.as_tensor(const.mass)
        # put the padding virtual atom mass finite as for accelaration, F/m evaluation.
        MASS[0] = 1.0
        mass = MASS[species].unsqueeze(2)

        acc = None
        P = None
        #output={'molid':[0], 'thermo':1, 'dump':10, 'prefix':'md'}
        """
        print("#step(dt=%.2f) " % self.timestep, end='')
        for mol in self.output['molid']:
            print("T(%d)  Etot(%d)  " % (mol, mol), end="")
        print()
        """

        for i in range(steps):
            coordinates, velocities, acc, P, L = self.one_step(const, mass, coordinates, velocities, species, \
                                                         acc=acc, learned_parameters=learned_parameters, P=P, step=i)
            #
            if not reuse_P:
                P = None
            Ek, T = self.kinetic_energy(const, mass, species, velocities)
            if (i+1)%self.output['thermo']==0:
                print("md  %6d" % (i+1), end="")
                for mol in self.output['molid']:
                    print(" %f %f " % (T[mol], L[mol]+Ek[mol]), end="")
                print()

            if (i+1)%self.output['dump']==0:
                for mol in self.output['molid']:
                    fn = self.output['prefix'] + "." + str(mol) + ".xyz"
                    f = open(fn,'a+')
                    f.write("%d\nstep: %d\n" % (torch.sum(species[mol]>0), i+1))
                    for atom in range(coordinates.shape[1]):
                        if species[mol,atom]>0:
                            f.write("%s %f %f %f\n" % (const.label[species[mol,atom].item()],
                                                       coordinates[mol,atom,0],
                                                       coordinates[mol,atom,1],
                                                       coordinates[mol,atom,2]))

                    f.close()
        return coordinates, velocities, acc

class Molecular_Dynamics_Langevin(Molecular_Dynamics_Basic):
    """
    molecular dynamics with langevin thermostat
    #same formula as in lammps
    """

    def __init__(self, seqm_parameters, timestep=1.0, damp=1.0, T=300.0, output={'molid':[0], 'thermo':1, 'dump':10, 'prefix':'md'}):
        """
        damp is damping factor in unit of time (fs)
        T : temperature in unit of Kelvin
        F = Fc + Ff + Fr
        Ff = - (m / damp) v
        Fr = sqrt(2 Kb T m / (dt damp))*R(t)
        each component of R(t) ~ N(0,1)
        <R_ij(t)>=0
        <R_ij(t) * R_ik(t) > = delta_jk
        """
        super().__init__(seqm_parameters, timestep=1.0, output=output)
        self.damp = damp
        self.T = T
        # Fr = sqrt(2 Kb T m / (dt damp))*R(t)
        #    = sqrt(2.0) * sqrt(kb T/m) * m/sqrt(dt damp) * R(t)
        # self.vel_scale change sqrt(kb T/m) into Angstrom/fs
        self.Fr_scale = 0.09450522179973914

    def get_force(self, const, masses, coordinates, velocities, species, learned_parameters=dict(), P0=None):
        """
        return force in unit of eV/Angstrom
        return force, density matrix, total energy of this batch (from conservative force)
        """
        Fc, P, L = self.conservative_force(const, coordinates, species, learned_parameters=learned_parameters, P0=P0)[:3]
        P = P.detach()
        Ff = -masses * velocities / self.damp / self.acc_scale
        Fr = self.Fr_scale * torch.sqrt(2.0*self.T*masses/self.timestep/self.damp)*torch.randn(Fc.shape, dtype=Fc.dtype, device=Fc.device)
        F = Fc+Ff+Fr
        F[species==0,:] = 0.0

        return F, P, L

#not complete
class Molecular_Dynamics_Nose_Hoover(Molecular_Dynamics_Basic):
    pass

#unit conversion note

# energy unit: eV
# length unit : Angstroms
# force unit : eV/Angstroms
# mass unit : grams/mol
# velocity unit : Angstrom/fs
# time unit: femtoseconds
# temperature : Kelvin
# accelaration : eV/Angstroms / (grams/mol)

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

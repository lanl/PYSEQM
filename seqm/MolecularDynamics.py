import torch
from .basics import *
import time
from seqm.ElectronicStructure import Electronic_Structure as esdriver
from .basics import Parser
from .seqm_functions.G_XL_LR import G
from seqm.seqm_functions.spherical_pot_force import Spherical_Pot_Force
import numpy as np
import sys
from io import StringIO
import h5py
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

    def onestep(self, molecule, learned_parameters=dict()):

        self.esdriver(molecule, learned_parameters=learned_parameters, P0=molecule.dm, dm_prop='SCF')
        force = molecule.force
        with torch.no_grad():
            molecule.coordinates.add_(self.alpha*force)
        return force, molecule.Hf

    def run(self, molecule, learned_parameters=dict(), log=True):
        dtype = molecule.coordinates.dtype
        device = molecule.coordinates.device
        nmol = molecule.coordinates.shape[0]
        molecule.coordinates.requires_grad_(True)
        Lold = torch.zeros(nmol,dtype=dtype,device=device)
        print("Step,  Max_Force,      Etot(eV),     dE(eV)")
        for i in range(self.max_evl):
            force, Lnew = self.onestep(molecule, learned_parameters=learned_parameters)
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
    def __init__(self, seqm_parameters, timestep=1.0,  output=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.seqm_parameters = seqm_parameters
        self.timestep = timestep
        self.esdriver = esdriver(self.seqm_parameters)
        self.acc_scale = 0.009648532800137615
        self.vel_scale = 0.9118367323190634e-3
        self.kinetic_energy_scale = 1.0364270099032438e2
        self.output = (output or {'molid':[0], 'thermo':1, 'dump':10, 'prefix':'md'}).copy()


    def initialize_velocity(self, molecule, Temperature, vel_com=True):

        dtype = molecule.coordinates.dtype
        device = molecule.coordinates.device
        MASS = torch.as_tensor(molecule.const.mass)
        MASS[0] = 1.0
        mass = MASS[molecule.species].unsqueeze(2)
        scale = torch.sqrt(Temperature/mass)*self.vel_scale
        molecule.velocities = torch.randn(molecule.coordinates.shape, dtype=dtype, device=device)*scale
        molecule.velocities[molecule.species==0,:] = 0.0
        if vel_com and Temperature > 0.0:
            #remove center of mass velocity
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
            L = torch.sum(mass * torch.linalg.cross(molecule.coordinates, molecule.velocities, dim=2), dim=1)
            I = torch.sum(mass * torch.norm(molecule.coordinates, dim=2, keepdim=True)**2, dim=1, keepdim=True) \
                * torch.eye(3, dtype=molecule.coordinates.dtype, device=molecule.coordinates.device).reshape(1,3,3) \
                - torch.sum( mass.unsqueeze(3) * molecule.coordinates.unsqueeze(3) * molecule.coordinates.unsqueeze(2), dim=1)
            omega = torch.linalg.solve(I, L.unsqueeze(2))
            molecule.velocities.add_( torch.linalg.cross(molecule.coordinates, omega.reshape(-1,1,3).repeat(1,molecule.coordinates.shape[1],1)) )
            # Rescale speeds so the kinetic energy (temperature) returns to what it was before the removal of COM velocity
            _, T1 = self.kinetic_energy(molecule)
            bad = (~torch.isfinite(T1)) | (T1 == 0.0)
            if bad.any():
                idx = torch.nonzero(bad.squeeze(1), as_tuple=False).squeeze(1).tolist()
                raise ValueError(
                    f"Post-correction kinetic energy T1 is zero or non-finite for batch indices {idx}. "
                    "Velocities have no remaining DOF after COM/AM removal."
                )
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
                    Tm  = float(T[mol].detach().cpu())
                    Ekm = float(Ek[mol].detach().cpu())
                    Lm  = float(L[mol].detach().cpu())
                    print(" %8.2f   %e %e %e || " % (Tm,   Ekm, Lm, Lm+Ekm), end="")
                print()
    
    def dump(self, i, molecule, velocities, q, T, Ek, L, forces, e_gap, Err=None, **kwargs):
        
        # Only dump on schedule
        if (i + 1) % self.output['dump'] != 0:
            return

        if Err is None:
            Err=np.zeros(T.shape)

        restricted = e_gap.dim()==1
        timestamp = time.time()

        for mol in self.output['molid']:
            n_atoms = int(torch.sum(molecule.species[mol] > 0))
            s = StringIO()
            s.write(f"{n_atoms}\n")
            if restricted:
                eg = float(e_gap[mol].detach().cpu())
                e_gap_str = f"{eg:12.6f}"
            else:
                eg0 = float(e_gap[mol,0].detach().cpu())
                eg1 = float(e_gap[mol,1].detach().cpu())
                e_gap_str = f"{eg0:12.6f}/{eg1:12.6f}"
            Tm  = float(T[mol].detach().cpu())
            Ekm = float(Ek[mol].detach().cpu())
            Lm  = float(L[mol].detach().cpu())
            Em  = float(Err[mol])
            s.write(f"step: {i+1}  T= {Tm:12.3f}K  Ek= {Ekm:12.9f}  Ep= {Lm:12.9f}  "
                    f"E_gap= {e_gap_str}  Err= {Em:24.16f}  time_stamp= {timestamp:.4f}\n")
            xyz = molecule.coordinates[mol]
            Z   = molecule.species[mol]
            for a in range(n_atoms):
                label = molecule.const.label[Z[a].item()]  # ensure .labels everywhere
                x, y, z = (xyz[a,0].item(), xyz[a,1].item(), xyz[a,2].item())
                s.write(f"{label} {x:15.5f} {y:15.5f} {z:15.5f}\n")
            self._xyz_files[mol].write(s.getvalue())
            
        if 'Info_log' in kwargs:
            for mol in self.output['molid']:
                info_lines = StringIO()
                info_lines.write(f"\nstep: {i+1}\n")
                # Each entry: [label, values], where values is list-of-strings per mol
                for label, values in kwargs['Info_log']:
                    info_lines.write(f"  {label}{values[mol]} \n")
                self._info_files[mol].write(info_lines.getvalue())


    def _open_dump_files(self):
        # 1 MB buffers; keep handles for reuse
        bufsize = 1_048_576
        self._xyz_files = {}
        self._info_files = {}
        for mol in self.output['molid']:
            xyz_fn  = f"{self.output['prefix']}.{mol}.xyz"
            info_fn = f"{self.output['prefix']}.{mol}.Info.txt"
            self._xyz_files[mol]  = open(xyz_fn,  'a+', buffering=bufsize)
            self._info_files[mol] = open(info_fn, 'a+', buffering=bufsize)

    def _close_dump_files(self):
        for d in (getattr(self, "_xyz_files", {}), getattr(self, "_info_files", {})):
            for fh in d.values():
                try: fh.close()
                except: pass
                    

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
        self.esdriver(molecule, learned_parameters=learned_parameters, P0=molecule.dm, dm_prop='SCF', cis_amp=molecule.cis_amplitudes, *args, **kwargs)

        with torch.no_grad():
            molecule.acc = molecule.force/molecule.mass*self.acc_scale
            molecule.velocities.add_(0.5*molecule.acc*dt)
        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            molecule.const.timing["MD"].append(t1-t0)

    def run(self, molecule, steps, learned_parameters=dict(), reuse_P=True, remove_com=[False,1000], *args, **kwargs):

        E0 = None

        h5_path     = f"{self.output['prefix']}.h5"                 # e.g. "run_001.h5" to enable
        h5_stride   = self.output['dump']                 # write every N steps
        excited_states_params = self.seqm_parameters.get('excited_states')
        n_roots = excited_states_params["n_states"] if excited_states_params is not None else 0
        write_mo    = bool(kwargs.get("h5_write_mo",  False))

        try:
            if h5_path:
                self._h5_open(molecule, h5_path,
                              steps=steps, h5_stride=h5_stride,
                              excited_states=n_roots, write_mo=write_mo)

            self._open_dump_files()

            for i in range(steps):
                
                start_time = time.time()
                self.one_step(molecule, learned_parameters=learned_parameters, P=molecule.dm, *args, **kwargs)

                with torch.no_grad():
                    if torch.is_tensor(molecule.coordinates.grad):
                        molecule.coordinates.grad.zero_()
                    
                    if not reuse_P:
                        molecule.dm = None
                        molecule.cis_amplitudes = None
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
                    if h5_path and ((i + 1) % h5_stride == 0):
                        self._h5_append_step(i + 1, molecule, T, Ek, molecule.Hf, molecule.e_gap)
                    dump_kwargs = {}
                    if kwargs.get("Info_log"):
                        dump_kwargs["Info_log"] = build_info_log(molecule)
                        
                    self.dump(i, molecule, molecule.velocities, molecule.q, T, Ek, molecule.Hf, molecule.force, molecule.e_gap, **dump_kwargs)
                del Ek, T
                if i%1000==0:
                    torch.cuda.empty_cache()
                
                if debug:
                    print(time.time() - start_time)
        finally:
            self._close_dump_files()
            if h5_path:
                self._h5_close()
        return molecule.coordinates, molecule.velocities, molecule.acc

    def _h5_open(self, molecule, h5_path, *, steps, h5_stride=1,
                 excited_states=0, write_mo=False,
                 compression="gzip", complvl=1):
        self._h5 = h5py.File(h5_path, "w")
        g = self._h5

        # derive constants
        B    = int(molecule.coordinates.shape[0])     # batch (molecules)
        Nat  = int(molecule.coordinates.shape[1])
        restricted = not bool(self.seqm_parameters.get('UHF', False))
        Norb = int(molecule.norb.max())

        Twrites = (int(steps) + int(h5_stride) - 1) // int(h5_stride)  # number of rows along time axis

        # If excitations enabled, get number of excited states
        write_excitations = excited_states > 0
        if write_excitations:
            R = excited_states

        # convenience
        def create(path, shape, dtype=np.float64):
            return g.create_dataset(path, shape=shape, dtype=dtype,
                                    chunks=True, compression=compression, compression_opts=complvl)

        # metadata
        mg = g.create_group("meta")
        mg.attrs.update({
            "units_E": "eV",
            "units_dipole": "a.u.",
            "units_coords": "Angstrom",
            "stride": int(h5_stride),
            "Twrites": int(Twrites),
        })

        # static (per run)
        g.create_dataset("atoms", data=_to_np(molecule.species))

        # time-major datasets with fixed shapes (Twrites, B, ...)
        create("steps",          (Twrites,),            np.int64)
        create("thermo/T",       (Twrites, B))
        create("thermo/Ek",      (Twrites, B))
        create("thermo/Ep",      (Twrites, B))
        create("homo_lumo_gap",      (Twrites, B, 1 if restricted else 2))

        # coords / vel / forces every write
        # create("coords/x",       (Twrites, B, Nat, 3))
        create("vel",          (Twrites, B, Nat, 3))
        # create("forces",       (Twrites, B, Nat, 3))

        # optional: excitations
        if write_excitations:
            create("excitation/excitation_energy",    (Twrites, B, R))
            create("excitation/transition_dipole",  (Twrites, B, R, 3))
            create("excitation/oscillator_strength", (Twrites, B, R))
            create("excitation/ground_dipole", (Twrites, B,3))
            create("excitation/active_state", (Twrites,),np.int64)
            create("excitation/relaxed_dipole", (Twrites, B, 3))
            create("excitation/unrelaxed_dipole", (Twrites, B, 3))

        # optional: MO energies
        if write_mo:
            if restricted:
                create("mo/e_orb", (Twrites, B, Norb))
                create("mo/nocc",  (Twrites, B))
            else:
                create("mo/e_orb", (Twrites, B, 2, Norb))
                create("mo/nocc",  (Twrites, B, 2))

        # internal write index (0..Twrites-1)
        self._h5_i = 0
        self._h5_flags = {
            "n_excited_states": excited_states,
            "write_mo":  bool(write_mo),
        }

    def _h5_append_step(self, step_idx, molecule, T, Ek, Ep, e_gap):
        """
        Write into row self._h5_i; caller must ensure this is a 'write step'
        (i.e., (step_idx % stride) == 0). No resizing here.
        """
        g = self._h5
        i = self._h5_i

        # basic fields
        g["steps"][i]          = int(step_idx)
        g["thermo/T"][i, ...]  = _to_np(T)
        g["thermo/Ek"][i, ...] = _to_np(Ek)
        g["thermo/Ep"][i, ...] = _to_np(Ep)

        gap_np = _to_np(e_gap)
        restricted = gap_np.ndim == 1
        if restricted:
            gap_np = gap_np[:, None]  # (B,1)
        g["homo_lumo_gap"][i, ...] = gap_np

        # vectors & arrays
        # g["coords/x"][i, ...]       = _to_np(molecule.coordinates)  # (B,N,3)
        g["vel"][i, ...]          = _to_np(molecule.velocities)
        # g["forces/x"][i, ...]       = _to_np(molecule.force)

        # optional: excitations
        
        R = self._h5_flags.get("n_excited_states", 0)
        if R > 0:
            g["excitation/excitation_energy"][i, ...]    = _to_np(molecule.cis_energies[:,:R])    # (B,R)
            g["excitation/transition_dipole"][i, ...]  = _to_np(molecule.transition_dipole[:,:R])    # (B,R,3)
            g["excitation/oscillator_strength"][i, ...] = _to_np(molecule.oscillator_strength[:,:R])  # (B,R)
            g["excitation/active_state"][i] = int(molecule.active_state)
            g["excitation/relaxed_dipole"][i, ...] = _to_np(molecule.cis_state_relaxed_dipole)            # (B,3)
            g["excitation/unrelaxed_dipole"][i, ...] = _to_np(molecule.cis_state_unrelaxed_dipole)            # (B,3)
            g["excitation/ground_dipole"][i, ...] = _to_np(molecule.dipole)            # (B,3)

        # optional: MO energies (can be large)
        if self._h5_flags.get("write_mo", False):
            g["mo/e_orb"][i, ...] = _to_np(molecule.e_mo)   # (B,Norb) or (B,2,Norb)
            g["mo/nocc"][i, ...]  = _to_np(molecule.nocc)   # (B,) or (B,2)

        self._h5_i += 1
        # optional: flush periodically, not every step
        if (self._h5_i % 100) == 0: g.flush()

    def _h5_close(self):
        h = getattr(self, "_h5", None)
        if h is not None:
            try: h.flush()
            finally:
                h.close()
                self._h5 = None
                self._h5_i = 0
                self._h5_flags = {}

def _to_np(x):
    return x.detach().cpu().numpy() #if hasattr(x, "detach") else (x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x))

class Molecular_Dynamics_Langevin(Molecular_Dynamics_Basic):
    """
    molecular dynamics with langevin thermostat
    #same formula as in lammps
    """

    def __init__(self, damp=1.0, Temp = 0.0, *args, **kwargs):
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
        self.Temp = Temp
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
    def __init__(self, xl_bomd_params=dict(), damp=False, Temp=0.0, *args, **kwargs):
        """
        unit for timestep is femtosecond
        """
        self.k = xl_bomd_params['k']
        self.xl_bomd_params = xl_bomd_params

        self.damp = damp
        self.Temp = Temp
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

        if not torch.is_tensor(molecule.force):
            print('Doing initialization')
            self.esdriver(molecule, learned_parameters=learned_parameters, P0=molecule.dm, *args, **kwargs)
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
        h5_path     = f"{self.output['prefix']}.h5"                 # e.g. "run_001.h5" to enable
        h5_stride   = self.output['dump']                 # write every N steps
        excited_states_params = self.seqm_parameters.get('excited_states')
        n_roots = excited_states_params["n_states"] if excited_states_params is not None else 0
        write_mo    = bool(kwargs.get("h5_write_mo",  False))

        try:
            if h5_path:
                self._h5_open(molecule, h5_path,
                              steps=steps, h5_stride=h5_stride,
                              excited_states=n_roots, write_mo=write_mo)

            self._open_dump_files()

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
                    if h5_path and ((i + 1) % h5_stride == 0):
                        self._h5_append_step(i + 1, molecule, T, Ek, molecule.Hf, molecule.e_gap)
                    dump_kwargs = {}
                    if kwargs.get("Info_log"):
                        dump_kwargs["Info_log"] = build_info_log(molecule)

                    self.dump(i, molecule, molecule.velocities, molecule.q, T, Ek, molecule.Hf + molecule.Electronic_entropy, molecule.force,
                              molecule.e_gap, molecule.Krylov_Error, **dump_kwargs)
                del T, Ek, dump_kwargs
                if i%1000==0:
                    torch.cuda.empty_cache()
                
                if debug:
                    print(time.time() - start_time)
        finally:
            self._close_dump_files()
            if h5_path:
                self._h5_close()

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

    #@attach_profile_range("KSA_XL_BOMD_ONESTEP")
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
        
        spherical_pot = True ########################################################################################==========
        if spherical_pot:
            with torch.no_grad():
                spherical_pot_E, spherical_pot_force = Spherical_Pot_Force(molecule, 14.85, k=0.1)
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

class ESMD(Molecular_Dynamics_Basic):
    def __init__(self, *args, **kwargs):
        """
        unit for timestep is femtosecond
        """
        super().__init__(*args, **kwargs)

    def run(self, molecule, steps, active_state, learned_parameters=dict(), reuse_P=True, remove_com=[False,1000], *args, **kwargs):
        molecule.active_state = active_state
        super().run(molecule=molecule,steps=steps,learned_parameters=learned_parameters, reuse_P=reuse_P, remove_com=remove_com, *args, **kwargs)


class XL_ESMD(XL_BOMD):
    def __init__(self, *args, **kwargs):
        """
        unit for timestep is femtosecond
        """
        super().__init__(*args, **kwargs)

    def one_step(self, molecule, step, P, Pt, amp, amp_t, ground_state_SCF=True, learned_parameters=dict(), *args, **kwargs):
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
            amp = self.coeff_D*c*molecule.cis_amplitudes + self.coeff_D*(1-c)*amp + torch.sum(self.coeff[cindx:(cindx+self.m)].reshape(-1,1,1,1)*amp_t, dim=0)
            
            #P = self.coeff_D*molecule.dm + torch.sum(self.coeff[cindx:(cindx+self.m)].reshape(-1,1,1,1)*Pt, dim=0)
            Pt[(self.m-1-cindx)] = P
            amp_t[(self.m-1-cindx)] = amp
    
        calc_type = 'SCF' if ground_state_SCF else 'XL-BOMD'
        self.esdriver(molecule, learned_parameters=learned_parameters, P0=P, xl_bomd_params=self.xl_bomd_params, dm_prop=calc_type, cis_amp=amp, *args, **kwargs)
        molecule.Electronic_entropy = torch.zeros(molecule.species.shape[0], device=molecule.coordinates.device)

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
        return P, Pt, amp, amp_t

    def run(self, molecule, steps, active_state, ground_state_SCF = True, learned_parameters=dict(), Pt=None, amp_t=None, xi_t=None, remove_com=[False,1000], *args, **kwargs):
        
        molecule.active_state = active_state
        self.initialize(molecule, learned_parameters=learned_parameters, *args, **kwargs)
        with torch.no_grad():
            if not torch.is_tensor(Pt):
                Pt = molecule.dm.unsqueeze(0).expand((self.m,)+molecule.dm.shape).clone()
                if 'max_rank' in self.xl_bomd_params:
                    molecule.dP2dt2 = torch.zeros(molecule.dm.shape, dtype=molecule.dm.dtype, device=molecule.dm.device)
            P = molecule.dm.clone()
            if not torch.is_tensor(amp_t):
                amp_t = molecule.cis_amplitudes.unsqueeze(0).expand((self.m,)+molecule.cis_amplitudes.shape).clone()
            amp = molecule.cis_amplitudes.clone()

        E0 = None

        for i in range(steps):
            start_time = time.time()

            P, Pt, amp, amp_t = self.one_step(molecule, i, P, Pt, amp, amp_t, learned_parameters=learned_parameters, ground_state_SCF=ground_state_SCF, *args, **kwargs)

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

class KSA_XL_ESMD(XL_ESMD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def one_step(self, molecule, step, P, Pt, amp, amp_t, learned_parameters=dict(), *args, **kwargs):
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
            P = self.coeff_D*molecule.dP2dt2 + self.coeff_D*P + torch.sum(self.coeff[cindx:(cindx+self.m)].reshape(-1,1,1,1)*Pt, dim=0)
            amp = self.coeff_D*c*molecule.cis_amplitudes + self.coeff_D*(1-c)*amp + torch.sum(self.coeff[cindx:(cindx+self.m)].reshape(-1,1,1,1)*amp_t, dim=0)
            
            #P = self.coeff_D*molecule.dm + torch.sum(self.coeff[cindx:(cindx+self.m)].reshape(-1,1,1,1)*Pt, dim=0)
            Pt[(self.m-1-cindx)] = P
            amp_t[(self.m-1-cindx)] = amp
    
        self.esdriver(molecule, learned_parameters=learned_parameters, P0=P, xl_bomd_params=self.xl_bomd_params, dm_prop='XL-BOMD', cis_amp=amp, *args, **kwargs)
        molecule.Electronic_entropy = torch.zeros(molecule.species.shape[0], device=molecule.coordinates.device)

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
        return P, Pt, amp, amp_t

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

def build_info_log(molecule, xl_bomd=False):
    # single compact helper that works for torch/np/python lists
    to_np = lambda x: x.detach().cpu().numpy()

    # --- Fetch inputs once ---
    e = to_np(molecule.e_mo)          # MO energies (B, n_orb) or (B, 2, n_orb)
    nocc = to_np(molecule.nocc)       # number of occupied orbitals (B,) or (B, 2)
    dvec = to_np(molecule.d).ravel()  # dipole vector (B,3)

    def fmt_vec(arr, prec):
        return np.array2string(
            arr,
            separator=", ",
            formatter={"float_kind": (lambda x: f"{x:.{prec}f}")},
            threshold=10**6,  # switch to summary form
            max_line_width=10**9  # keep on one logical line
        ).strip("[]")

    def format_mo_energies(e_mol, nocc):
        nocc_i = int(nocc)
        occ = fmt_vec(e_mol[:nocc_i], 5)
        virt = fmt_vec(e_mol[nocc_i:], 5)
        return f"    Occupied:\n      {occ}\n    Virtual:\n      {virt}"

    info = []

    
    restricted = nocc.ndim == 1
    if restricted:
        # e: (B, n_orb)     nocc: (B,)
        # For each molecule in the batch, produce one formatted string
        entries = [format_mo_energies(e_mol, nocc_mol) for e_mol, nocc_mol in zip(e, nocc)]
        info.append(["Orbital energies:\n", entries])
    else:
        # e: (B, 2, n_orb)  nocc: (B, 2)
        e_alpha, e_beta = e[:, 0], e[:, 1]
        n_alpha, n_beta = nocc[:, 0], nocc[:, 1]

        entries_a = [format_mo_energies(e_mol, nocc_mol) for e_mol, nocc_mol in zip(e_alpha, n_alpha)]
        entries_b = [format_mo_energies(e_mol, nocc_mol) for e_mol, nocc_mol in zip(e_beta,  n_beta)]

        info.append(["Orbital energies alpha:\n", entries_a])
        info.append(["Orbital energies beta:\n",  entries_b])

    info.append(["dipole(x,y,z): ", [f"{v:.6f}" for v in dvec]])

    if xl_bomd:
        info.append(['Electronic entropy contribution (eV): ', molecule.Electronic_entropy])
        info.append(['Fermi occupancies:\n', ['    Occupied:\n      ' + str(x[0: i])[1:-1].replace('\n', '\n     ') + \
                                  '\n    Virtual:\n      ' + str(x[i:])[1:-1].replace('\n', '\n     ') for x, i in \
                                  zip(np.round(molecule.Fermi_occ.cpu().numpy(), 6), molecule.nocc)] ])
        info.append(['Rank-m Krylov subspace approximation error: ', molecule.Krylov_Error])

    return info

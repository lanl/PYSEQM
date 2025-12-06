import os
import sys
import tempfile
import time
import math
from datetime import datetime
from io import StringIO

import h5py
import numpy as np
import torch

from seqm.basics import Force
from seqm.ElectronicStructure import Electronic_Structure as esdriver
from seqm.seqm_functions.rcis_batch import orthogonalize_to_current_subspace, make_cis_densities

# from .seqm_functions.G_XL_LR import G
from seqm.seqm_functions.spherical_pot_force import Spherical_Pot_Force

np.set_printoptions(threshold=sys.maxsize)

# from .tools import attach_profile_range
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
            molecule.coordinates.add_(self.alpha * force)
        return force, molecule.Etot

    def run(self, molecule, learned_parameters=dict(), log=True):
        dtype = molecule.coordinates.dtype
        device = molecule.coordinates.device
        nmol = molecule.coordinates.shape[0]
        molecule.verbose = False
        Lold = torch.zeros(nmol, dtype=dtype, device=device)
        print("Step,  Max_Force,      Etot(eV),     dE(eV)")
        for i in range(self.max_evl):
            force, Lnew = self.onestep(molecule, learned_parameters=learned_parameters)
            if torch.is_tensor(molecule.coordinates.grad):
                with torch.no_grad():
                    molecule.coordinates.grad.zero_()
            force_err = torch.max(torch.abs(force))
            energy_err = (Lnew - Lold).sum() / nmol
            if log:

                print("%d      " % (i + 1), end="")
                print("%e " % force_err.item(), end="")
                """
                dis = torch.norm(coordinates[...,0,:]-coordinates[...,1,:], dim=1)
                for k in range(coordinates.shape[0]):
                    print("%e " % dis[k], end="")
                #"""
                for k in range(molecule.coordinates.shape[0]):
                    print("||%e %e " % (Lnew[k], Lnew[k] - Lold[k]), end="")
                print("")

            if (force_err > self.force_tol):
                Lold = Lnew
                continue
            else:
                break
        if i == (self.max_evl - 1):
            print('not converged within %d step' % self.max_evl)
        else:
            if log:
                print("converged with %d step, Max Force = %e (eV/Ang), dE = %e (eV)" %
                      (i + 1, force_err.item(), energy_err.item()))

        return force_err, energy_err


class Molecular_Dynamics_Basic(torch.nn.Module):
    """
    perform basic moleculer dynamics with verlocity_verlet algorithm, and in NVE ensemble
    separate get force, run one step, and run n steps is to make it easier to implement thermostats
    """

    def __init__(self, seqm_parameters, timestep=1.0, Temp=0.0, step_offset=0, output=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.seqm_parameters = seqm_parameters
        self.timestep = timestep
        self.esdriver = esdriver(self.seqm_parameters)
        self.Temp = Temp
        self.acc_scale = 0.009648532800137615
        self.vel_scale = 0.9118367323190634e-3
        self.kinetic_energy_scale = 1.0364270099032438e2
        self.temperature_scale = 1.160451812e4  # Kelvin / eV
        self.step_offset = step_offset

        self.output = (output or {'molid': [0], 'print every': 1, 'prefix': 'md'}).copy()
        self._normalize_output()

        self.n_dof = None  # number of degrees of freedom
        self.remove_com_angular = False
        self.start_time = None

    def _normalize_output(self):
        o = self.output or {}

        # For backward compatibility
        if "thermo" in o and "print_every" not in o:
            o["print every"] = int(o["thermo"])
        if "dump" in o:
            dump = int(o["dump"])
            o.setdefault("xyz", dump)
            o.setdefault("h5", {}).setdefault("data", dump)

        # Defaults
        o.setdefault("molid", [0])
        o.setdefault("print every", 1)
        o.setdefault("checkpoint every", 100)
        o.setdefault("prefix", "md")
        o.setdefault("h5", {})

        # XYZ cadence
        o.setdefault("xyz", 0)
        self._print_every = int(o["print every"])
        self._xyz_every = int(o["xyz"])

        # HDF5: intuitive cadences (numbers mean "every N steps"; 0 = off)
        h5 = o["h5"]
        self._cadence = {
            "coordinates": int(h5.get("coordinates", 0)),
            "velocities": int(h5.get("velocities", 0)),
            "forces": int(h5.get("forces", 0)),
        }
        self._any_vectors = any(self._cadence.values())
        self._h5_data_every = int(h5.get("data", 0))  # scalars/thermo cadence; 0 = off
        self._h5_write_mo = bool(h5.get("write_mo", False))

    def initialize_velocity(self, molecule, vel_com=True):

        # check if the MD object is on the correct device
        md_dev = self.esdriver.conservative_force.energy.packpar.p.device
        mol_dev = molecule.coordinates.device
        if md_dev != mol_dev:
            raise RuntimeError(
                f"Please move the MD object (on {md_dev}) to the same device as the molecule object (on {mol_dev})")
        if self.n_dof is None:
            raise RuntimeError(
                "n_dof is not set. Call `initialize(molecule, ...)` or `set_dof(molecule, ...)` "
                "before initializing velocities."
            )

        if torch.is_tensor(molecule.velocities):
            return molecule.velocities

        Temperature = self.Temp
        dtype = molecule.coordinates.dtype
        device = molecule.coordinates.device
        if Temperature == 0.0:
            molecule.velocities = torch.zeros(molecule.coordinates.shape, dtype=dtype, device=device)
            return molecule.velocities

        scale = torch.sqrt(Temperature * molecule.mass_inverse) * self.vel_scale
        molecule.velocities = torch.randn(molecule.coordinates.shape, dtype=dtype, device=device) * scale

        # Rescale speeds to reach the target temperature
        Ek = self.kinetic_energy(molecule)
        T1 = self.calc_temperature(Ek)
        alpha = torch.sqrt(Temperature / T1)
        molecule.velocities.mul_(alpha.reshape(-1, 1, 1))

        if vel_com:
            #remove center of mass velocity
            self.zero_com(molecule, translate_to_origin=True)
        return molecule.velocities

    def zero_com(self, molecule, remove_angular=True, translate_to_origin=False):
        """
        remove center of mass velocity
        remove angular momentum
        mass shape (nmol, molsize, 1)
        velocities shape (nmol, molsize, 3)
        """
        #mass for padding atom is 0.0
        mass = molecule.mass
        M = torch.sum(mass, dim=1, keepdim=True)
        Ek_initial = self.kinetic_energy(molecule)

        with torch.no_grad():
            r_com = torch.sum(mass * molecule.coordinates, dim=1, keepdim=True) / M
            if translate_to_origin:
                molecule.coordinates.sub_(r_com)
                r_rel = molecule.coordinates
            else:
                r_rel = molecule.coordinates - r_com
            v_com = torch.sum(mass * molecule.velocities, dim=1, keepdim=True) / M
            molecule.velocities.sub_(v_com)
            if remove_angular:
                L = torch.sum(mass * torch.linalg.cross(r_rel, molecule.velocities, dim=2), dim=1)
                I = torch.sum(mass * (r_rel * r_rel).sum(dim=2, keepdim=True), dim=1, keepdim=True) \
                    * torch.eye(3, dtype=molecule.coordinates.dtype, device=molecule.coordinates.device).reshape(1,3,3) \
                    - torch.sum( mass.unsqueeze(3) * r_rel.unsqueeze(3) * r_rel.unsqueeze(2), dim=1)
                # Compute angular velocity omega = I^{-1} L (use pinv for robustness)
                I_pinv = torch.linalg.pinv(I)
                omega = (I_pinv @ L.unsqueeze(2)).squeeze(-1)
                molecule.velocities.sub_(torch.linalg.cross(omega.unsqueeze(1).expand_as(r_rel), r_rel, dim=2))
                del L, I, omega, I_pinv

            # Rescale speeds so the kinetic energy returns to what it was before the removal of COM velocity
            Ek_after = self.kinetic_energy(molecule)
            if torch.any(Ek_after < 1e-12):
                raise RuntimeError("Kinetic energy is zero after removing centre of mass momentum")
            alpha = torch.sqrt(Ek_initial / Ek_after)
            molecule.velocities.mul_(alpha.reshape(-1, 1, 1))

            del r_com, v_com

    def kinetic_energy(self, molecule):
        Ek = torch.sum(0.5 * molecule.mass * molecule.velocities**2, dim=(1, 2)) * self.kinetic_energy_scale
        return Ek

    def calc_temperature(self, kinetic_energy):
        return kinetic_energy * self.temperature_scale / (0.5 * self.n_dof)

    def screen_output(self, i, T, Ek, L):
        if i == 0:
            print("Step,    Temp,    E(kinetic),  E(potential),  E(total)")
        print("%6d" % (i + 1), end="")
        for mol in self.output['molid']:
            Tm = float(T[mol].detach().cpu())
            Ekm = float(Ek[mol].detach().cpu())
            Lm = float(L[mol].detach().cpu())
            print(" %8.2f   %e %e %e || " % (Tm, Ekm, Lm, Lm + Ekm), end="")
        print()

    def dump(self, i, molecule, velocities, q, T, Ek, L, forces, e_gap, Err=None, **kwargs):
        # if Err is None:
        #     Err=np.zeros(T.shape)

        # restricted = e_gap.dim()==1
        # timestamp = time.time()

        Et_ = (Ek + L).detach().cpu()
        # dipole_ = molecule.dipole.detach().cpu()

        for mol in self.output['molid']:
            n_atoms = int(torch.sum(molecule.species[mol] > 0))
            s = StringIO()
            s.write(f"{n_atoms}\n")

            # if restricted:
            #     eg = float(e_gap[mol].detach().cpu())
            #     e_gap_str = f"{eg:12.6f}"
            # else:
            #     eg0 = float(e_gap[mol,0].detach().cpu())
            #     eg1 = float(e_gap[mol,1].detach().cpu())
            #     e_gap_str = f"{eg0:12.6f}/{eg1:12.6f}"
            # Tm  = float(T[mol].detach().cpu())
            # Ekm = float(Ek[mol].detach().cpu())
            # Lm  = float(L[mol].detach().cpu())
            # Em  = float(Err[mol])
            # s.write(f"step: {i+1}  T= {Tm:12.3f}K  Ek= {Ekm:12.9f}  Ep= {Lm:12.9f}  "
            #         f"E_gap= {e_gap_str}  Err= {Em:24.16f}  time_stamp= {timestamp:.4f}\n")

            Et_mol = float(Et_[mol])
            s.write(f"step: {i+1}  E_total = {Et_mol:12.9f}  \n")

            xyz = molecule.coordinates[mol]
            Z = molecule.species[mol]
            for a in range(n_atoms):
                label = molecule.const.label[Z[a].item()]  # ensure .labels everywhere
                x, y, z = (xyz[a, 0].item(), xyz[a, 1].item(), xyz[a, 2].item())
                s.write(f"{label} {x:15.5f} {y:15.5f} {z:15.5f}\n")
            self._xyz_files[mol].write(s.getvalue())

        # if 'Info_log' in kwargs:
        #     for mol in self.output['molid']:
        #         info_lines = StringIO()
        #         info_lines.write(f"\nstep: {i+1}\n")
        #         # Each entry: [label, values], where values is list-of-strings per mol
        #         for label, values in kwargs['Info_log']:
        #             info_lines.write(f"  {label}{values[mol]} \n")
        #         self._info_files[mol].write(info_lines.getvalue())

    def _open_dump_files(self):
        # 1 MB buffers; keep handles for reuse
        bufsize = 1_048_576
        self._xyz_files = {}
        # self._info_files = {}
        for mol in self.output['molid']:
            xyz_fn = f"{self.output['prefix']}.{mol}.xyz"
            if self.step_offset == 0:
                _rotate_existing(xyz_fn)
            self._xyz_files[mol] = open(xyz_fn, 'a+', buffering=bufsize)
            # info_fn = f"{self.output['prefix']}.{mol}.Info.txt"
            # _rotate_existing(info_fn)
            # self._info_files[mol] = open(info_fn, 'a+', buffering=bufsize)

    def _close_dump_files(self):
        # for d in (getattr(self, "_xyz_files", {}), getattr(self, "_info_files", {})):
        for d in (getattr(self, "_xyz_files", {}), ):
            for fh in d.values():
                try:
                    fh.close()
                except:
                    pass

    def control_shift(self, velocities, Ek, Eshift):
        alpha = torch.sqrt((Ek - Eshift) / Ek)
        alpha[~torch.isfinite(alpha)] = 0.0
        velocities.mul_(alpha.reshape(-1, 1, 1))

    def one_step(self, molecule, learned_parameters=dict(), *args, **kwargs):
        dt = self.timestep
        if molecule.const.do_timing:
            t0 = time.time()
        with torch.no_grad():
            molecule.velocities.add_(0.5 * molecule.acc * dt)
            molecule.coordinates.add_(molecule.velocities * dt)
        self.esdriver(molecule,
                      learned_parameters=learned_parameters,
                      P0=molecule.dm,
                      dm_prop='SCF',
                      cis_amp=molecule.cis_amplitudes,
                      *args,
                      **kwargs)

        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * self.acc_scale
            molecule.velocities.add_(0.5 * molecule.acc * dt)
        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            molecule.const.timing["MD"].append(t1 - t0)

    def set_dof(self, molecule, constraints=0.0):
        self.n_dof = 3.0 * molecule.num_atoms - constraints

    def initialize(self, molecule, remove_com=None, learned_parameters=dict(), *args, **kwargs):
        molecule.verbose = False  # Dont print SCF and CIS/RPA results
        self.do_remove_com = remove_com is not None
        constraints = 0.0
        # remove_com is a tuple of (mode,stride), where mode='linear' or 'angular'
        # and stride is the number of steps after which com motion is removed
        if self.do_remove_com:
            mode, self.remove_com_stride = remove_com
            mode = str(mode).lower().strip()
            if mode not in ("linear", "angular"):
                raise ValueError(f"Invalid COM motion removal mode '{mode}'. "
                                 "Expected 'linear' or 'angular'. "
                                 "Usage: remove_com=('linear', N) or ('angular', N).")
            self.remove_com_angular = (mode == "angular")
            constraints = 6.0 if self.remove_com_angular else 3.0  # TODO: check if the molecule is linear

        self.set_dof(molecule, constraints)

        self.initialize_velocity(molecule)

        # Calculate accelearation at t=0
        if not torch.is_tensor(molecule.force):
            self.esdriver(molecule,
                          learned_parameters=learned_parameters,
                          P0=molecule.dm,
                          cis_amp=molecule.cis_amplitudes,
                          *args,
                          **kwargs)

        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * self.acc_scale

    def _thermo_potential(self, molecule):
        """Potential-energy contribution to show/log/write with Ek. Basic: Etot."""
        return molecule.Etot

    def _do_integrator_step(self, i, molecule, learned_parameters, **kwargs):
        """Advance one MD step. Basic uses the standard one_step signature."""
        return self.one_step(molecule, learned_parameters=learned_parameters, **kwargs)

    def run(self, molecule, steps, learned_parameters=dict(), reuse_P=True, remove_com=None, seed=None, *args, **kwargs):
        self.start_time = datetime.now() 
        print(f"MD run began at {self.start_time}",flush=True)
        if seed is not None:
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))

        # if doing XL-BOMD reuse_P has to be True
        if getattr(self,"k",None) is not None:
            reuse_P = True

        self.initialize(molecule, remove_com=remove_com, learned_parameters=learned_parameters, *args, **kwargs)
        if not reuse_P:
            molecule.dm = None
            molecule.cis_amplitudes = None

        E0 = None

        do_scale_vel = ("scale_vel" in kwargs)
        if do_scale_vel:
            scale_freq, T_target = kwargs["scale_vel"]
            scale_freq = int(scale_freq)
            T_target = torch.as_tensor(T_target, dtype=molecule.coordinates.dtype, device=molecule.coordinates.device)

        do_energy_shift = bool(kwargs.get("control_energy_shift", False))
        if do_scale_vel and do_energy_shift:
            raise ValueError("Can't scale velocities to fix temperature and fix energy shift at the same time.")

        do_screen = (self._print_every > 0)
        do_xyz = (self._xyz_every > 0)
        do_h5 = (self._h5_data_every > 0) or self._any_vectors

        checkpoint_every = self.output["checkpoint every"]
        checkpoint_path = f'{self.output["prefix"]}.restart.pt'

        try:
            if do_h5:
                h5_prefix = f"{self.output['prefix']}"
                excited_states_params = self.seqm_parameters.get('excited_states')
                n_roots = excited_states_params["n_states"] if excited_states_params else 0

                self._h5_open(
                    molecule,
                    h5_prefix,
                    steps=steps,
                    data_stride=self._h5_data_every,
                    excited_states=n_roots,
                    write_mo=self._h5_write_mo,
                    resume=self.step_offset > 0,
                )

            if do_xyz:
                self._open_dump_files()
            for i in range(self.step_offset, steps):

                # start_time = time.time()
                self._do_integrator_step(i, molecule, learned_parameters, *args, **kwargs)

                with torch.no_grad():
                    if torch.is_tensor(molecule.coordinates.grad):
                        molecule.coordinates.grad.zero_()

                    if not reuse_P:
                        molecule.dm = None
                        molecule.cis_amplitudes = None

                    if self.do_remove_com:
                        if i % self.remove_com_stride == 0:
                            self.zero_com(molecule, remove_angular=self.remove_com_angular)

                    Ek = self.kinetic_energy(molecule)
                    T = self.calc_temperature(Ek)
                    V = self._thermo_potential(molecule)
                    if E0 is None:
                        E0 = V + Ek

                    #scale velocities to control temperature
                    if do_scale_vel and ((i + 1) % scale_freq == 0):
                        # rescale velocities to T_target
                        alpha = torch.sqrt(torch.clamp(T_target / T, min=0.0))
                        molecule.velocities.mul_(alpha.reshape(-1, 1, 1))
                        Ek = self.kinetic_energy(molecule)
                        T = self.calc_temperature(Ek)

                    #control energy shift
                    if do_energy_shift:
                        #scale velocities to adjust kinetic energy and compenstate the energy shift
                        Eshift = Ek + V - E0
                        self.control_shift(molecule.velocities, Ek, Eshift)
                        Ek = self.kinetic_energy(molecule)
                        T = self.calc_temperature(Ek)

                    if do_screen and ((i + 1) % self._print_every == 0):
                        self.screen_output(i, T, Ek, V)

                    if do_h5:
                        if self._h5_data_every > 0 and ((i + 1) % self._h5_data_every == 0):
                            self._h5_append_data_step(i + 1, molecule, T, Ek, V, molecule.e_gap)
                        if self._any_vectors:
                            self._h5_append_vectors_step(i + 1, molecule)

                    if do_xyz and ((i + 1) % self._xyz_every == 0):
                        # dump_kwargs = {}
                        # if kwargs.get("Info_log"):
                        #     dump_kwargs["Info_log"] = build_info_log(molecule)
                        self.dump(i, molecule, molecule.velocities, molecule.q, T, Ek, V, molecule.force,
                                  molecule.e_gap, molecule.Krylov_Error)  #, **dump_kwargs)

                    if checkpoint_every > 0 and ((i + 1) % checkpoint_every == 0):
                        self._flush_all()
                        self.save_checkpoint(molecule, steps, reuse_P, remove_com, step_done=i + 1, path=checkpoint_path)

                del Ek, T
                if i % 1000 == 0:
                    torch.cuda.empty_cache()

                # if debug:
                #     print(time.time() - start_time)
        finally:
            if do_xyz:
                self._close_dump_files()
            if do_h5:
                self._h5_close()

        now = datetime.now()
        print(f"MD run ended at {now}")
        print(f"Time elapsed since the beginning of MD run: {now-self.start_time}",flush=True)
        return molecule.coordinates, molecule.velocities, molecule.acc

    def _h5_open(self,
                 molecule,
                 h5_prefix,
                 *,
                 steps,
                 data_stride=1,
                 excited_states=0,
                 write_mo=False,
                 resume=False,
                 compression="gzip",
                 complvl=4):

        self._h5_handles = {}
        self._h5_i_data = {}
        self._h5_i_vec = {}
        self._h5_flags = {}

        def n_timepoints(stride):
            s = int(stride)
            return (int(steps) + s - 1) // s if s > 0 else 0

        Tw_data = n_timepoints(int(data_stride))
        Tw_vec = {k: n_timepoints(v) for k, v in self._cadence.items()}

        restricted = not bool(self.seqm_parameters.get('UHF', False))

        # if doing excited state gradients with autograd, then relaxed_dipole is not calculated
        self._h5_save_relaxed_dipole = excited_states > 0 and molecule.seqm_parameters.get('scf_backward',0) == 0

        def create_row_chunked(g, path, shape, dtype=np.float64):
            chunks = (1, ) + tuple(shape[1:])
            return g.create_dataset(path,
                                    shape=shape,
                                    dtype=dtype,
                                    chunks=chunks,
                                    compression=compression,
                                    compression_opts=complvl)

        for mol in self.output['molid']:
            Nat_mol = int(torch.sum(molecule.species[mol] > 0))
            S = slice(0, Nat_mol)
            Norb_mol = int(molecule.norb[mol])
            R = int(excited_states) if excited_states > 0 else 0

            h5_path = f"{h5_prefix}.{mol}.h5"
            self._h5_flags[mol] = {
                "restricted": restricted,
                "Norb": Norb_mol,
                "Nat": Nat_mol,
                "active_slice": S,
                "n_excited_states": R,
                "write_mo": bool(write_mo),
                "Tw_data": int(Tw_data),
                "Tw_vec": {
                    k: int(v)
                    for k, v in Tw_vec.items()
                },
                "stride": dict(self._cadence),  # cadence per stream
            }

            if resume:
                # --- RESUME: open existing file, read shapes, set indices from current content
                h5 = h5py.File(h5_path, "r+")
                self._h5_handles[mol] = h5

                # Use existing dataset shapes as truth for capacity
                Tw_data_exist = h5["data/steps"].shape[0] if ("data" in h5 and "steps" in h5["data"]) else 0
                Tw_vec_exist = {
                    "coordinates": h5["coordinates/steps"].shape[0] if "coordinates" in h5 else 0,
                    "velocities": h5["velocities/steps"].shape[0] if "velocities" in h5 else 0,
                    "forces": h5["forces/steps"].shape[0] if "forces" in h5 else 0,
                }
                # Set write cursors from absolute step offset
                i_data = (self.step_offset // self._h5_data_every) if self._h5_data_every > 0 else 0
                i_vec = {
                    "coordinates":
                    (self.step_offset // self._cadence["coordinates"]) if self._cadence["coordinates"] > 0 else 0,
                    "velocities":
                    (self.step_offset // self._cadence["velocities"]) if self._cadence["velocities"] > 0 else 0,
                    "forces": (self.step_offset // self._cadence["forces"]) if self._cadence["forces"] > 0 else 0,
                }
                # Defensive: if cadence>0 but group missing, fail loudly
                if self._h5_data_every > 0 and Tw_data_exist == 0:
                    raise RuntimeError("Resume requested but /data group not present in HDF5.")
                for k, cad in self._cadence.items():
                    if cad > 0 and Tw_vec_exist[k] == 0:
                        raise RuntimeError(f"Resume requested but /{k} group not present in HDF5.")
                self._h5_i_data[mol] = i_data
                self._h5_i_vec[mol] = i_vec
                # Use existing capacities in flags
                self._h5_flags[mol].update({
                    "Tw_data": int(Tw_data_exist),
                    "Tw_vec": {
                        kk: int(vv)
                        for kk, vv in Tw_vec_exist.items()
                    },
                })

                continue

            _rotate_existing(h5_path)
            h5 = h5py.File(h5_path, "w")
            self._h5_handles[mol] = h5
            self._h5_i_data[mol] = 0
            self._h5_i_vec[mol] = {"coordinates": 0, "velocities": 0, "forces": 0}

            # Store atoms (active only)
            h5.create_dataset("atoms", data=_to_np(molecule.species[mol, S]))

            # ---- /data group (thermo, MO, excitations...)
            if Tw_data > 0:
                gd = h5.create_group("data")
                create_row_chunked(gd, "steps", (Tw_data, ), np.int64)
                create_row_chunked(gd, "thermo/T", (Tw_data, ))
                create_row_chunked(gd, "thermo/Ek", (Tw_data, ))
                create_row_chunked(gd, "thermo/Ep", (Tw_data, ))

                create_row_chunked(gd, "properties/ground_dipole", (Tw_data, 3))

                # excitations (optional)
                if R > 0:
                    gd.create_dataset("excitation/active_state", data=int(molecule.active_state))
                    create_row_chunked(gd, "excitation/excitation_energy", (Tw_data, R))
                    create_row_chunked(gd, "excitation/transition_dipole", (Tw_data, R, 3))
                    create_row_chunked(gd, "excitation/oscillator_strength", (Tw_data, R))
                    if self._h5_save_relaxed_dipole:
                        create_row_chunked(gd, "excitation/unrelaxed_dipole", (Tw_data, 3))
                        create_row_chunked(gd, "excitation/relaxed_dipole", (Tw_data, 3))

                # MO (optional)
                if write_mo:
                    if restricted:
                        # create_row_chunked(gd, "mo/e_orb",        (Tw_data, Norb_mol))
                        create_row_chunked(gd, "mo/homo_lumo_gap", (Tw_data, 1))
                        gd.create_dataset("mo/nocc", data=int(molecule.nocc[mol].item()))
                    else:
                        # create_row_chunked(gd, "mo/e_orb",        (Tw_data, 2, Norb_mol))
                        create_row_chunked(gd, "mo/homo_lumo_gap", (Tw_data, 2))
                        gd.create_dataset("mo/nocc", data=_to_np(molecule.nocc[mol]))

            # ---- /vectors group (vel/forces[/coords])
            def make_stream_group(name, Tlen):
                if Tlen <= 0:
                    return
                g = h5.create_group(name)  # "/coordinates", "/velocities", "/forces"
                create_row_chunked(g, "steps", (Tlen, ), np.int64)
                create_row_chunked(g, "values", (Tlen, Nat_mol, 3))

            make_stream_group("coordinates", Tw_vec["coordinates"])
            make_stream_group("velocities", Tw_vec["velocities"])
            make_stream_group("forces", Tw_vec["forces"])

    def _h5_append_data_step(self, step_idx, molecule, T, Ek, Ep, e_gap):
        for mol in self.output['molid']:
            h5 = self._h5_handles[mol]
            i = self._h5_i_data[mol]
            if i is None:
                continue
            flags = self._h5_flags[mol]
            if flags["Tw_data"] == 0:
                continue

            gd = h5["data"]

            gd["steps"][i] = int(step_idx)
            gd["thermo/T"][i] = float(T[mol].detach().cpu())
            gd["thermo/Ek"][i] = float(Ek[mol].detach().cpu())
            gd["thermo/Ep"][i] = float(Ep[mol].detach().cpu())
            gd["properties/ground_dipole"][i, ...] = _to_np(molecule.dipole[mol])

            R = flags["n_excited_states"]
            if R > 0:
                gd["excitation/excitation_energy"][i, ...] = _to_np(molecule.cis_energies[mol, :R])
                gd["excitation/transition_dipole"][i, ...] = _to_np(molecule.transition_dipole[mol, :R])
                gd["excitation/oscillator_strength"][i, ...] = _to_np(molecule.oscillator_strength[mol, :R])
                if self._h5_save_relaxed_dipole:
                    gd["excitation/unrelaxed_dipole"][i, ...] = _to_np(molecule.cis_state_unrelaxed_dipole[mol])
                    gd["excitation/relaxed_dipole"][i, ...] = _to_np(molecule.cis_state_relaxed_dipole[mol])

            if flags.get("write_mo", False):
                Norb = flags["Norb"]
                if flags["restricted"]:
                    # gd["mo/e_orb"][i, ...] = _to_np(molecule.e_mo[mol, :Norb])
                    gd["mo/homo_lumo_gap"][i, ...] = _to_np(e_gap[mol, None])  # (1,1)
                else:
                    # gd["mo/e_orb"][i, ...] = _to_np(molecule.e_mo[mol, :, :Norb])
                    gd["mo/homo_lumo_gap"][i, ...] = _to_np(e_gap[mol])  # (2,)

            self._h5_i_data[mol] = i + 1
            if (self._h5_i_data[mol] % 100) == 0:
                h5.flush()

    def _h5_append_vectors_step(self, step_idx, molecule):
        for mol in self.output['molid']:
            h5 = self._h5_handles[mol]
            f = self._h5_flags[mol]
            S = f["active_slice"]
            stride, Tw, i_vec = f["stride"], f["Tw_vec"], self._h5_i_vec[mol]

            tensors = {
                "coordinates": molecule.coordinates[mol, S, :],
                "velocities": molecule.velocities[mol, S, :],
                "forces": molecule.force[mol, S, :],
            }

            did_write = False
            for name, arr in tensors.items():
                k = stride[name]
                if k <= 0 or (step_idx % k) != 0:
                    continue
                i = i_vec[name]
                if i >= Tw[name]:
                    continue
                g = h5[name]
                g["steps"][i] = int(step_idx)
                g["values"][i] = _to_np(arr)
                i_vec[name] = i + 1
                did_write = did_write or (i_vec[name] % 100 == 0)

            if did_write:
                h5.flush()

    def _h5_close(self):
        for h in getattr(self, "_h5_handles", {}).values():
            try:
                h.flush()
            finally:
                h.close()
        self._h5_handles = {}
        self._h5_i_data = {}
        self._h5_i_vec = {}
        self._h5_flags = {}

    def _flush_all(self):
        # Flush HDF5 library buffers
        for h in getattr(self, "_h5_handles", {}).values():
            try:
                h.flush()
            except Exception:
                pass
        for fh in getattr(self, "_xyz_files", {}).values():
            try:
                fh.flush()
            except Exception:
                pass

    def save_checkpoint(self, molecule, steps: int, reuse_P, remove_com, *, step_done: int, path: str):
        """Atomically save a restart snapshot."""

        def _tensor_cpu(x):
            return x.detach().cpu() if torch.is_tensor(x) else x

        ckpt = {
            # run-level
            "MD_type": self.__class__.__name__,
            "device": molecule.coordinates.device,
            "step_done": int(step_done),
            "steps": int(steps),
            "reuse_P": bool(reuse_P),
            "timestep": float(self.timestep),
            "Temp": float(self.Temp),
            "damp": getattr(self, "damp", None),
            "xl_bomd_params": getattr(self, "xl_bomd_params", None),
            "seqm_parameters": self.seqm_parameters,
            "remove_com": remove_com,
            "output": self.output,
            # RNG states
            "rng": {
                "torch_cpu": torch.random.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "molecules": {
                "species": _tensor_cpu(molecule.species),
                "coordinates": _tensor_cpu(molecule.coordinates),
                "velocities": _tensor_cpu(molecule.velocities),
                "forces": _tensor_cpu(molecule.force),
                "dm": _tensor_cpu(molecule.dm) if reuse_P else None,
                "cis_amplitudes": _tensor_cpu(molecule.cis_amplitudes) if reuse_P else None,
                "constants": molecule.const,
                "old_mos": _tensor_cpu(molecule.old_mos),
            }
        }
        if hasattr(self, "m"):  # XL-BOMD and KSA-XL-BOMD
            ckpt["xl_ctx"] = {
                "Pt": _tensor_cpu(self._xl_ctx["Pt"]),
                "es_amp_t": _tensor_cpu(self._xl_ctx.get("es_amp_t"))
            }
            ckpt["dP2dt2"] = _tensor_cpu(molecule.dP2dt2) 

        # atomic write
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_ckpt_", suffix=".pt")
        os.close(tmp_fd)
        try:
            torch.save(ckpt, tmp_path)
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except:
                pass
        now = datetime.now()
        print(f"Saved checkpoint at {now}")
        print(f"Time elapsed since the beginning of MD run: {now-self.start_time}",flush=True)

    @staticmethod
    def run_from_checkpoint(path: str, device=None):
        """Load restart and reconstruct Molecule + MD objects."""
        ckpt = torch.load(path, map_location="cpu",weights_only=False)
        # rehydrate RNGs first (optional; or do after constructing objects)
        torch.random.set_rng_state(ckpt["rng"]["torch_cpu"])
        if torch.cuda.is_available() and ckpt["rng"]["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["torch_cuda"])

        # rebuild objects
        from seqm.Molecule import Molecule

        torch.set_default_dtype(ckpt["molecules"]["coordinates"].dtype)
        if device is None:
            device = ckpt["device"]
        const = ckpt["molecules"]["constants"].to(device)

        molecule = Molecule(const, ckpt["seqm_parameters"], ckpt["molecules"]["coordinates"].to(device),
                            ckpt["molecules"]["species"].to(device)).to(device)

        # restore dynamics/electronic state
        reuse_P = ckpt["reuse_P"]
        with torch.no_grad():
            molecule.velocities = ckpt["molecules"]["velocities"].to(device)
            molecule.force = ckpt["molecules"]["forces"].to(device)
            molecule.dm = ckpt["molecules"]["dm"].to(device) if reuse_P else None
            molecule.cis_amplitudes = ckpt["molecules"]["cis_amplitudes"]
            if isinstance(molecule.cis_amplitudes, torch.Tensor):
                molecule.cis_amplitudes = molecule.cis_amplitudes.to(device)
                molecule.old_mos = ckpt["molecules"]["old_mos"].to(device)
            dP2dt2 = ckpt.get("dP2dt2")
            if dP2dt2 is not None:
                molecule.dP2dt2 = dP2dt2.to(device)

        Temp = ckpt["Temp"]
        seqm_parameters = ckpt["seqm_parameters"]
        timestep = ckpt["timestep"]
        output = ckpt["output"]
        damp = ckpt["damp"]
        xl_bomd_params = ckpt["xl_bomd_params"]

        md_type = ckpt["MD_type"]

        # rebuild MD object with same knobs
        if md_type == "Molecular_Dynamics_Basic":
            md = Molecular_Dynamics_Basic(seqm_parameters=seqm_parameters,
                                          timestep=timestep,
                                          Temp=Temp,
                                          output=output,
                                          step_offset=ckpt["step_done"]).to(device)
        elif md_type == "Molecular_Dynamics_Langevin":
            md = Molecular_Dynamics_Langevin(seqm_parameters=seqm_parameters,
                                             timestep=timestep,
                                             Temp=Temp,
                                             output=output,
                                             damp=damp,
                                             step_offset=ckpt["step_done"]).to(device)
        elif md_type == "XL_BOMD":
            md = XL_BOMD(seqm_parameters=seqm_parameters,
                         timestep=timestep,
                         Temp=Temp,
                         output=output,
                         damp=damp,
                         xl_bomd_params=xl_bomd_params,
                         step_offset=ckpt["step_done"]).to(device)
        elif md_type == "KSA_XL_BOMD":
            md = KSA_XL_BOMD(seqm_parameters=seqm_parameters,
                             timestep=timestep,
                             Temp=Temp,
                             output=output,
                             damp=damp,
                             xl_bomd_params=xl_bomd_params,
                             step_offset=ckpt["step_done"]).to(device)
        else:
            raise RuntimeError(f"Unrecognized MD type '{md_type}' in the saved checkpoint file")

        if md_type in ("XL_BOMD", "KSA_XL_BOMD"):
            xl = ckpt["xl_ctx"]
            Pt = xl["Pt"].to(device)
            es_amp_t = xl.get("es_amp_t")
            xl_m = ckpt["xl_bomd_params"]['k'] + 1
            cindx = (ckpt["step_done"] - 1) % xl_m # subtract one because step_done is advanced by one step
            P = Pt[(xl_m - 1 - cindx)].clone()
            if  isinstance(es_amp_t, torch.Tensor):
                es_amp_t = es_amp_t.to(device)
                es_amp = es_amp_t[(xl_m - 1 - cindx)].clone()
            else:
                es_amp = None

            md._xl_ctx = {"P": P, "Pt": Pt, "es_amp": es_amp, "es_amp_t": es_amp_t}

        md.run(molecule=molecule, steps=ckpt["steps"], reuse_P=reuse_P, remove_com=ckpt["remove_com"])

def _to_np(x):
    return x.detach().cpu().numpy()


class Molecular_Dynamics_Langevin(Molecular_Dynamics_Basic):
    """
    molecular dynamics with langevin thermostat
    """

    def __init__(self, damp=50.0, *args, **kwargs):
        """
        damp is damping factor in unit of time (fs)
        Temp : temperature in unit of Kelvin

        F = Fc + Ff + Fr
        Ff = - (m / damp) v
        Fr = sqrt(2 Kb T m / (dt damp))*R(t)
        each component of R(t) ~ N(0,1)
        <R_ij(t)>=0
        <R_ij(t) * R_ik(t) > = delta_jk

        Integration scheme for Langevin dynamics is from 
        Bussi, G., & Parrinello, M. (2007). Accurate sampling using Langevin dynamics. Physical Review E, 75(5), 056707.
        DOI: https://doi.org/10.1103/PhysRevE.75.056707
        """

        self.damp = damp
        super().__init__(*args, **kwargs)

    def set_dof(self, molecule, constraints=0.0):
        # For langevin thermostat dont reduce degrees of freedom even if centre of mass momentum is zeroed out
        # because the thermostat gives energy into all 3N degrees of freedom
        # See: https://nwchemgit.github.io/Special_AWCforum/st/id2509/Langevin_thermostat_for_Gaussian....html
        self.n_dof = 3.0 * molecule.num_atoms

    def apply_langevin_thermostat(self, molecule):
        # dt = self.timestep
        # s    = -dt / self.damp                 # = -γ dt
        # c1 = torch.exp(torch.as_tensor(0.5*s,dtype=molecule.coordinates.dtype, device=molecule.coordinates.device))
        # 1 - e^{-γ dt} (stable)
        # one_me = -torch.expm1(torch.as_tensor(s,dtype=molecule.coordinates.dtype, device=molecule.coordinates.device))
        # c2 = torch.sqrt(one_me*self.Temp*molecule.mass_inverse)*self.vel_scale
        with torch.no_grad():
            molecule.velocities.mul_(self.langevin_c1)
            molecule.velocities.add_(self.langevin_c2 * torch.randn_like(molecule.velocities))

    def one_step(self, molecule, learned_parameters=dict(), *args, **kwargs):
        dt = self.timestep
        if molecule.const.do_timing:
            t0 = time.time()

        self.apply_langevin_thermostat(molecule)

        with torch.no_grad():
            molecule.velocities.add_(0.5 * molecule.acc * dt)
            molecule.coordinates.add_(molecule.velocities * dt)
        self.esdriver(molecule,
                      learned_parameters=learned_parameters,
                      P0=molecule.dm,
                      dm_prop='SCF',
                      cis_amp=molecule.cis_amplitudes,
                      *args,
                      **kwargs)

        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * self.acc_scale
            molecule.velocities.add_(0.5 * molecule.acc * dt)

        self.apply_langevin_thermostat(molecule)

        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            molecule.const.timing["MD"].append(t1 - t0)

    def initialize(self, molecule, remove_com=None, learned_parameters=dict(), *args, **kwargs):

        if self.damp is not None:
            dt = self.timestep
            # s = -γ dt
            s = torch.as_tensor(-dt / self.damp, dtype=molecule.coordinates.dtype, device=molecule.coordinates.device)
            # c1 = exp{-γ dt/2}
            self.langevin_c1 = torch.exp(0.5 * s)
            # c2 = 1 - c1^2 = 1 - e^{-γ dt}
            one_me = -torch.expm1(s)
            self.langevin_c2 = torch.sqrt(one_me * self.Temp * molecule.mass_inverse) * self.vel_scale
        return super().initialize(molecule, remove_com, learned_parameters, *args, **kwargs)


class XL_BOMD(Molecular_Dynamics_Langevin):
    """
    perform basic moleculer dynamics with verlocity_verlet algorithm, and in NVE ensemble
    separate get force, run one step, and run n steps is to make it easier to implement thermostats
    """

    def __init__(self, damp=None, xl_bomd_params=dict(), *args, **kwargs):
        """
        unit for timestep is femtosecond
        """
        self.k = xl_bomd_params['k']
        self.xl_bomd_params = xl_bomd_params
        super().__init__(damp, *args, **kwargs)
        #check Niklasson et al JCP 130, 214109 (2009)
        #coeff: kappa, alpha, c0, c1, ..., c9
        self.coeffs = {
            3: [1.69, 150e-3, -2.0, 3.0, 0.0, -1.0],
            4: [1.75, 57e-3, -3.0, 6.0, -2.0, -2.0, 1.0],
            5: [1.82, 18e-3, -6.0, 14.0, -8.0, -3.0, 4.0, -1.0],
            6: [1.84, 5.5e-3, -14.0, 36.0, -27.0, -2.0, 12.0, -6.0, 1.0],
            7: [1.86, 1.6e-3, -36.0, 99.0, -88.0, 11.0, 32.0, -25.0, 8.0, -1.0],
            8: [1.88, 0.44e-3, -99.0, 286.0, -286.0, 78.0, 78.0, -90.0, 42.0, -10.0, 1.0],
            9: [1.89, 0.12e-3, -286.0, 858.0, -936.0, 364.0, 168.0, -300.0, 184.0, -63.0, 12.0, -1.0]
        }
        #
        self.m = xl_bomd_params['k'] + 1
        self.kappa = self.coeffs[xl_bomd_params['k']][0]
        self.alpha = self.coeffs[xl_bomd_params['k']][1]
        cc = 1.00  # Scaled delta function approximation of Kernel coefficient cc = [0,1]
        tmp = torch.as_tensor(self.coeffs[xl_bomd_params['k']][2:]) * self.alpha
        #P(n+1) = 2*P(n) - P(n-1) + cc*kappa*(D(n)-P(n)) + alpha*(c0*P(n) + c1*P(n-1) + ... ck*P(n-k))
        #       =  cc*kappa*D(n)
        #        + (2 - cc*kappa + alpha*c0)*P(n)
        #        + (alpha*c1 - 1) * P(n-1)
        #        + alpha*c2*P(n-2)
        #        + ...
        self.coeff_D = cc * self.kappa
        tmp[0] += (2.0 - cc * self.kappa)
        tmp[1] -= 1.0
        self.coeff = torch.nn.Parameter(tmp.repeat(2), requires_grad=False)
        self.add_spherical_potential = False  # Spherical force to prevent atoms from flying off?
        self.do_scf = False
        self.move_on_excited_state = False

    def set_dof(self, molecule, constraints=0.0):
        # For langevin thermostat dont reduce degrees of freedom even if centre of mass momentum is zeroed out
        # because the thermostat gives energy into all 3N degrees of freedom
        # See: https://nwchemgit.github.io/Special_AWCforum/st/id2509/Langevin_thermostat_for_Gaussian....html
        if self.damp is not None:  #do Langevin dynamics
            constraints = 0.0

        self.n_dof = 3.0 * molecule.num_atoms - constraints

    def propagate_P(self, P, Pt, cindx, molecule):
        # eq. 22 in https://doi.org/10.1063/1.3148075
        #### Scaling delta function. Use eq with c if stability problems occur.
        # P(n+1) = coeff_D * [ c*D(n) + (1-c)*P(n) ] + sum_j coeff[j] * Pt[j]
        c = 0.95
        P_new = self.coeff_D * ( c * molecule.dm + (1.0 - c) * P )  \
            + torch.sum(self.coeff[cindx:(cindx+self.m)].reshape(-1,1,1,1)*Pt, dim=0)
        return P_new

    def propagate_excited_state(self, es_amp, es_amp_t, cindx, molecule):
        c = 0.95
        es_new = self.coeff_D * ( c * molecule.transition_density_matrices + (1.0 - c) * es_amp )  \
            + torch.sum(self.coeff[cindx:(cindx+self.m)].reshape(-1,1,1,1,1)*es_amp_t, dim=0)
        return es_new

    def one_step(self, molecule, step, P, Pt, es_amp=None, es_amp_t=None, learned_parameters=dict(), *args, **kwargs):
        #cindx: show in Pt, which is the latest P
        dt = self.timestep
        if molecule.const.do_timing:
            t0 = time.time()

        if self.damp:
            self.apply_langevin_thermostat(molecule)

        with torch.no_grad():
            # leapfrog velocity Verlet
            molecule.velocities.add_(0.5 * molecule.acc * dt)
            molecule.coordinates.add_(molecule.velocities * dt)

            #cindx = step%self.m
            #e.g k=5, m=6
            #coeff: c0, c1, c2, c3, c4, c5, c0, c1, c2, c3, c4, c5
            #Pt (0,1,2,3,4,5), step=6n  , cindx = 0, coeff[0:6]
            #Pt (1,2,3,4,5,0), step=6n+1, cindx = 1, coeff[1:7]
            #Pt (2,3,4,5,0,1), step=6n+2
            cindx = step % self.m
            # eq. 22 in https://doi.org/10.1063/1.3148075
            P = self.propagate_P(P, Pt, cindx, molecule)
            Pt[(self.m - 1 - cindx)] = P

            if molecule.active_state > 0:
                es_amp = self.propagate_excited_state(es_amp, es_amp_t, cindx, molecule)
                es_amp_t[(self.m - 1 - cindx)] = es_amp
                es_amp_ortho = es_amp

                # # Orthonormalize the propagated excited state amplitudes
                # # TODO: Will not work the same for RPA
                # es_amp_ortho, R_ = torch.linalg.qr(es_amp.transpose(-2, -1),
                #                                    mode="reduced")  # Q: (b, m, n), R_: (b, n, n)
                # # If any diagonal of R_ is ~0 in a batch, some input row was zero → fail that batch.
                # diag = torch.abs(torch.diagonal(R_, dim1=-2, dim2=-1))  # (b, n)
                # bad = (diag < self.vec_eps*math.sqrt(es_amp.shape[-1])).any(dim=-1)  # (b,)
                # if bad.any():
                #     idx = torch.nonzero(bad, as_tuple=False).flatten().tolist()
                #     raise ValueError(f"Rank-deficient/zero vector detected in batch indices {idx}.")
                # es_amp_ortho = es_amp_ortho.transpose(-2, -1)

                # n_roots = molecule.cis_amplitudes.shape[1]
                # es_amp_ortho = es_amp.clone()
                # # Normalize the first CIS amplitude
                # es_amp_ortho[:,0] /=  torch.linalg.vector_norm(es_amp_ortho[:,0],dim=1,keepdim=True)
                # if  n_roots > 1: # if more than one root, orthonormalize them
                #     for i in range(molecule.nmol):
                #         n_new = orthogonalize_to_current_subspace(es_amp_ortho[i], es_amp[i,1:], 1, tol=1e-8)
                #         if n_new < n_roots:
                #             raise RuntimeError("Some roots were lost while orthogonalizing, cannot proceed")
            else:
                es_amp_ortho = es_amp  # will be set to None

            if self.do_scf:
                calc_type = 'SCF'

                # Purify with McWeeny polynomial since P may not be idempotent
                # 3P^2 - 2P^3
                # For restricted density matrix (spin summed) D = 2P. So to purify, D0 = 3/2 D^2 - 1/2 D^3
                # TODO: Make it work for unrestricted P
                P2 = P @ P
                P0 = torch.baddbmm(P2, P2, P, beta=1.5, alpha=-0.5)

            else:
                P0 = P
                calc_type = 'XL-BOMD'

        self.esdriver(molecule,
                      learned_parameters=learned_parameters,
                      xl_bomd_params=self.xl_bomd_params,
                      P0=P0,
                      cis_amp=es_amp_ortho,
                      dm_prop=calc_type,
                      *args,
                      **kwargs)

        if self.add_spherical_potential:
            with torch.no_grad():
                dE, dF = Spherical_Pot_Force(molecule, radius=14.85, k=0.1)
                molecule.Etot = molecule.Etot + dE
                molecule.force = molecule.force + dF

        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * self.acc_scale
            molecule.velocities.add_(0.5 * molecule.acc * dt)

        if self.damp:
            self.apply_langevin_thermostat(molecule)

        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            molecule.const.timing["MD"].append(t1 - t0)
        return P, Pt, es_amp, es_amp_t

    def _thermo_potential(self, molecule):
        # XL-BOMD potential energy includes Electronic_entropy (why?)
        return molecule.Etot + molecule.Electronic_entropy

    def _do_integrator_step(self, i, molecule, learned_parameters, **kwargs):
        # Keep & update short history state across steps
        # Expect self._xl_ctx holding P, Pt, es_amp, es_amp_t initialized in initialize()
        P, Pt = self._xl_ctx["P"], self._xl_ctx["Pt"]
        es_amp, es_amp_t = self._xl_ctx.get("es_amp"), self._xl_ctx.get("es_amp_t")
        P, Pt, es_amp, es_amp_t = self.one_step(molecule,
                                                i,
                                                P,
                                                Pt,
                                                es_amp,
                                                es_amp_t,
                                                learned_parameters=learned_parameters,
                                                **kwargs)
        self._xl_ctx.update(P=P, Pt=Pt, es_amp=es_amp, es_amp_t=es_amp_t)

    def initialize(self, molecule, remove_com=None, learned_parameters=dict(), do_xl_esmd=False, *args, **kwargs):
        if molecule.active_state > 0:
            self.move_on_excited_state = True
            self.esdriver.conservative_force.energy.excited_states['save_tdm'] = True

        super().initialize(molecule, remove_com=remove_com, learned_parameters=learned_parameters, *args, **kwargs)

        if self.move_on_excited_state and not do_xl_esmd:
            scf_eps = self.xl_bomd_params.setdefault("scf_eps", 1e-5)
            es_eps = self.xl_bomd_params.setdefault("es_eps", 1e-4)
            dtype = molecule.coordinates.dtype
            if dtype==torch.float32:
                    self.vec_eps = 5.0e-5
            elif dtype==torch.float64:
                    self.vec_eps = 1.0e-8
            else:
                raise RuntimeError("Set dtype to float64 or float32")

            if 'max_rank' in self.xl_bomd_params:
                raise ValueError("KSA-XL-BOMD does not yet work for excited state dynamics")
            self.do_scf = True
            self.esdriver.conservative_force.energy.hamiltonian.eps = \
                torch.nn.Parameter(torch.as_tensor(scf_eps), requires_grad=False)
            self.esdriver.conservative_force.energy.excited_states['tolerance'] = es_eps
            molecule.Electronic_entropy = torch.zeros(molecule.species.shape[0], device=molecule.coordinates.device)
            self.esdriver.conservative_force.energy.excited_states['make_best_guess'] = False
            if self.esdriver.conservative_force.energy.excited_states['method'].lower() == "rpa":
                raise ValueError(
                    "XL-BOMD with excited states not tested for RPA. "
                    "Currently only works for CIS. "
                    "Will have to change one_step function to make it work for RPA."
                )

        if self.step_offset > 0:
            # resuming: expect caller/loader to have restored dm / cis_amplitudes
            return

        # Set up P/Pt (and es_amp/es_amp_t if on excited state) for a fresh run (not resuming)
        with torch.no_grad():
            P = molecule.dm.clone()
            Pt = molecule.dm.unsqueeze(0).expand((self.m, ) + molecule.dm.shape).clone()
            if 'max_rank' in self.xl_bomd_params:
                molecule.dP2dt2 = torch.zeros(molecule.dm.shape, dtype=molecule.dm.dtype, device=molecule.dm.device)
            ctx = {"P": P, "Pt": Pt}

            # for xl-esmd since we propagate the transition_density for only the active state, save only that
            if do_xl_esmd:
                molecule.transition_density_matrices = molecule.transition_density_matrices[:,molecule.active_state-1].unsqueeze(1)

            if self.move_on_excited_state:
                es_amp = molecule.transition_density_matrices.clone()
                es_amp_t = es_amp.unsqueeze(0).expand((self.m, ) + es_amp.shape).clone()
                ctx.update(es_amp=es_amp, es_amp_t=es_amp_t)
            self._xl_ctx = ctx


class KSA_XL_BOMD(XL_BOMD):

    def __init__(self, damp=None, xl_bomd_params=dict(), *args, **kwargs):
        super().__init__(damp, xl_bomd_params, *args, **kwargs)
        self.add_spherical_potential = False  # Spherical force to prevent atoms from flying off?

    def propagate_P(self, P, Pt, cindx, molecule):
        P_new = self.coeff_D * (molecule.dP2dt2 + P) \
                + torch.sum(self.coeff[cindx:(cindx+self.m)].reshape(-1,1,1,1)*Pt, dim=0)
        return P_new

class XL_ESMD(XL_BOMD):
    def one_step(self, molecule, step, P, Pt, es_amp=None, es_amp_t=None, learned_parameters=dict(), *args, **kwargs):
        #cindx: show in Pt, which is the latest P
        dt = self.timestep
        if molecule.const.do_timing:
            t0 = time.time()

        if self.damp:
            self.apply_langevin_thermostat(molecule)

        with torch.no_grad():
            # leapfrog velocity Verlet
            molecule.velocities.add_(0.5 * molecule.acc * dt)
            molecule.coordinates.add_(molecule.velocities * dt)

            #cindx = step%self.m
            #e.g k=5, m=6
            #coeff: c0, c1, c2, c3, c4, c5, c0, c1, c2, c3, c4, c5
            #Pt (0,1,2,3,4,5), step=6n  , cindx = 0, coeff[0:6]
            #Pt (1,2,3,4,5,0), step=6n+1, cindx = 1, coeff[1:7]
            #Pt (2,3,4,5,0,1), step=6n+2
            cindx = step % self.m
            # eq. 22 in https://doi.org/10.1063/1.3148075
            P = self.propagate_P(P, Pt, cindx, molecule)
            Pt[(self.m - 1 - cindx)] = P

            es_amp = self.propagate_excited_state(es_amp, es_amp_t, cindx, molecule)
            es_amp_t[(self.m - 1 - cindx)] = es_amp

            # Purify with McWeeny polynomial since P may not be idempotent
            # 3P^2 - 2P^3
            # For restricted density matrix (spin summed) D = 2P. So to purify, D0 = 3/2 D^2 - 1/2 D^3
            # TODO: Make it work for unrestricted P
            P2 = P @ P
            P0 = torch.baddbmm(P2, P2, P, beta=1.5, alpha=-0.5)

        self.esdriver.conservative_force.energy.xlesmd_transition_density = es_amp
        # active_state = molecule.active_state
        # molecule.active_state = 0
        self.esdriver(molecule,
                      learned_parameters=learned_parameters,
                      xl_bomd_params=self.xl_bomd_params,
                      P0=P0,
                      cis_amp=None,
                      dm_prop='SCF',
                      *args,
                      **kwargs)
        # molecule.active_state = active_state

        if self.add_spherical_potential:
            with torch.no_grad():
                dE, dF = Spherical_Pot_Force(molecule, radius=14.85, k=0.1)
                molecule.Etot = molecule.Etot + dE
                molecule.force = molecule.force + dF

        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * self.acc_scale
            molecule.velocities.add_(0.5 * molecule.acc * dt)

        if self.damp:
            self.apply_langevin_thermostat(molecule)

        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            molecule.const.timing["MD"].append(t1 - t0)
        return P, Pt, es_amp, es_amp_t

    def initialize(self, molecule, remove_com=None, learned_parameters=dict(), *args, **kwargs):
        super().initialize(molecule, remove_com=remove_com, learned_parameters=learned_parameters, do_xl_esmd=True, *args, **kwargs)

        molecule.Electronic_entropy = torch.zeros(molecule.species.shape[0], device=molecule.coordinates.device)
        self.esdriver.conservative_force.energy.excited_states = None

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
kb = 0.8314462264063073e-6 a.m.u. Angstrom^2/fs^2/K

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
    # --- Fetch inputs once ---
    e = _to_np(molecule.e_mo)  # MO energies (B, n_orb) or (B, 2, n_orb)
    nocc = _to_np(molecule.nocc)  # number of occupied orbitals (B,) or (B, 2)
    dvec = _to_np(molecule.dipole)  # dipole vector (B,3)

    def fmt_vec(arr, prec):
        return np.array2string(
            arr,
            separator=", ",
            formatter={
                "float_kind": (lambda x: f"{x:.{prec}f}")
            },
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
        entries_b = [format_mo_energies(e_mol, nocc_mol) for e_mol, nocc_mol in zip(e_beta, n_beta)]

        info.append(["Orbital energies alpha:\n", entries_a])
        info.append(["Orbital energies beta:\n", entries_b])

    info.append(["dipole(x,y,z): ", [fmt_vec(dvec[i], 6) for i in range(molecule.nmol)]])

    if xl_bomd:
        info.append(['Electronic entropy contribution (eV): ', molecule.Electronic_entropy])
        info.append(['Fermi occupancies:\n', ['    Occupied:\n      ' + str(x[0: i])[1:-1].replace('\n', '\n     ') + \
                                  '\n    Virtual:\n      ' + str(x[i:])[1:-1].replace('\n', '\n     ') for x, i in \
                                  zip(np.round(molecule.Fermi_occ.cpu().numpy(), 6), molecule.nocc)] ])
        info.append(['Rank-m Krylov subspace approximation error: ', molecule.Krylov_Error])

    return info


def _rotate_existing(path, start=1, max_tries=99, error_on_max=True):
    """
    If `path` exists, rename it to the first free variant like:
      base.bak.1, base.bak.2, ...
    """
    if not os.path.exists(path):
        return
    base, ext = os.path.splitext(path)
    n = start
    while n <= max_tries:
        backup = f"{base}.bak.{n}{ext}"
        if not os.path.exists(backup):
            os.rename(path, backup)
            return
        n += 1

    if error_on_max:
        raise RuntimeError(f"Unable to backup: could not rotate existing file: {path}")

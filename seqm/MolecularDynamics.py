import os
import sys
import tempfile
import time
import math
from datetime import datetime
from io import StringIO
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from seqm.seqm_functions.spherical_pot_force import Spherical_Pot_Force

import h5py
import numpy as np
import torch

from seqm.basics import Force
from seqm.ElectronicStructure import Electronic_Structure as esdriver

np.set_printoptions(threshold=sys.maxsize)

# Physical constants and conversion factors
@dataclass
class PhysicalConstants:
    """Physical constants and unit conversion factors."""
    ACC_SCALE: float = 0.009648532800137615  # eV/Å / (g/mol) -> Å/fs²
    VEL_SCALE: float = 0.9118367323190634e-3  # sqrt(K/amu) -> Å/fs
    KINETIC_ENERGY_SCALE: float = 1.0364270099032438e2  # amu*(Å/fs)² -> eV
    TEMPERATURE_SCALE: float = 1.160451812e4  # K/eV

CONSTANTS = PhysicalConstants()


@dataclass
class OutputConfig:
    """Configuration for MD output files and frequencies."""
    molid: List[int] = field(default_factory=lambda: [0])
    prefix: str = "md"
    print_every: int = 1
    checkpoint_every: int = 100
    xyz_every: int = 0
    h5_config: Dict[str, Any] = field(default_factory=dict)
    h5_vectors_every: Optional[int] = None
    
    @classmethod
    def from_dict(cls, config: Optional[Dict] = None) -> 'OutputConfig':
        """Create OutputConfig from dictionary with backward compatibility."""
        if config is None:
            config = {}
        
        # Backward compatibility
        if "thermo" in config and "print_every" not in config:
            config["print_every"] = int(config["thermo"])
        if "dump" in config:
            dump = int(config["dump"])
            config.setdefault("xyz", dump)
            config.setdefault("h5", {}).setdefault("data", dump)
            
        wanted_vectors = {"coordinates", "velocities", "forces"}
        vals = [
            v for k, v in config['h5'].items()
            if k in wanted_vectors and isinstance(v, int) and v > 0
        ]
        vectors_every = min(vals, default=None) 
        
        return cls(
            molid=config.get("molid", [0]),
            prefix=config.get("prefix", "md"),
            print_every=int(config.get("print every", 1)),
            checkpoint_every=int(config.get("checkpoint every", 100)),
            xyz_every=int(config.get("xyz", 0)),
            h5_config=config.get("h5", {}),
            h5_vectors_every=vectors_every
        )
    
    def get_h5_cadence(self) -> Dict[str, int]:
        """Extract HDF5 writing cadences."""
        return {
            "coordinates": int(self.h5_config.get("coordinates", 0)),
            "velocities": int(self.h5_config.get("velocities", 0)),
            "forces": int(self.h5_config.get("forces", 0)),
        }
    
    def get_h5_data_every(self) -> int:
        return int(self.h5_config.get("data", 0))
    
    def get_h5_write_mo(self) -> bool:
        return bool(self.h5_config.get("write_mo", False))
    
    def get_h5_write_tdm(self) -> int:
        return int(self.h5_config.get("transition_density_matrices", 0))


class HDF5Writer:
    """Manages HDF5 file writing for MD trajectories."""
    
    def __init__(self, output_config: OutputConfig, seqm_parameters: Dict):
        self.config = output_config
        self.seqm_parameters = seqm_parameters
        self.handles: Dict[int, h5py.File] = {}
        self.i_data: Dict[int, int] = {}
        self.i_vec: Dict[int, Dict[str, int]] = {}
        self.i_tdm: Dict[int, int] = {}
        self.flags: Dict[int, Dict] = {}
        
        self._cadence = output_config.get_h5_cadence()
        self._data_every = output_config.get_h5_data_every()
        self._write_mo = output_config.get_h5_write_mo()
        self._write_tdm = output_config.get_h5_write_tdm()
        self._save_relaxed_dipole = False
    
    @staticmethod
    def _n_timepoints(steps: int, stride: int) -> int:
        """Calculate number of timepoints for given stride."""
        return (steps + stride - 1) // stride if stride > 0 else 0
    
    @staticmethod
    def _create_row_chunked(group: h5py.Group, path: str, shape: Tuple, 
                           dtype=np.float64, compression="gzip", complvl=4) -> h5py.Dataset:
        """Create chunked dataset optimized for row-wise writing."""
        chunks = (1,) + tuple(shape[1:])
        return group.create_dataset(
            path, shape=shape, dtype=dtype, chunks=chunks,
            compression=compression, compression_opts=complvl
        )
    
    def open(self, molecule, prefix: str, steps: int, excited_states: int = 0,
             resume: bool = False, step_offset: int = 0):
        """Open HDF5 files for writing."""
        Tw_data = self._n_timepoints(steps, self._data_every)
        Tw_vec = {k: self._n_timepoints(steps, v) for k, v in self._cadence.items()}
        Tw_tdm = self._n_timepoints(steps, self._write_tdm)
        
        restricted = not bool(self.seqm_parameters.get('UHF', False))
        self._save_relaxed_dipole = (excited_states > 0 and 
                                     molecule.seqm_parameters.get('scf_backward', 0) == 0)
        
        for mol in self.config.molid:
            h5_path = f"{prefix}.{mol}.h5"
            Nat_mol = int(torch.sum(molecule.species[mol] > 0))
            Norb_mol = int(molecule.norb[mol])
            R = int(excited_states) if excited_states > 0 else 0
            
            self.flags[mol] = {
                "restricted": restricted,
                "Norb": Norb_mol,
                "Nat": Nat_mol,
                "active_slice": slice(0, Nat_mol),
                "n_excited_states": R,
                "write_mo": self._write_mo,
                "write_tdm": bool(Tw_tdm > 0),
                "Tw_data": Tw_data,
                "Tw_tdm": Tw_tdm,
                "Tw_vec": Tw_vec.copy(),
                "stride": self._cadence.copy(),
                "tdm_stride": self._write_tdm,
            }
            
            if resume:
                self._open_resume(h5_path, mol, step_offset)
            else:
                self._create_new(h5_path, mol, molecule, Nat_mol, Norb_mol, R, Tw_data, Tw_vec, Tw_tdm)
    
    def _open_resume(self, h5_path: str, mol: int, step_offset: int):
        """Open existing HDF5 file for resuming."""
        h5 = h5py.File(h5_path, "r+")
        self.handles[mol] = h5
        
        # Read existing capacities
        Tw_data_exist = h5["data/steps"].shape[0] if ("data" in h5 and "steps" in h5["data"]) else 0
        Tw_vec_exist = {
            k: h5[f"{k}/steps"].shape[0] if k in h5 else 0
            for k in ["coordinates", "velocities", "forces"]
        }
        Tw_tdm_exist = (h5["data/excitation/transition_density_matrices/steps"].shape[0]
                       if ("data" in h5 and "excitation" in h5["data"] and
                           "transition_density_matrices" in h5["data/excitation"]) else 0)
        
        # Validate
        if self._data_every > 0 and Tw_data_exist == 0:
            raise RuntimeError("Resume requested but /data group not present in HDF5.")
        for k, cad in self._cadence.items():
            if cad > 0 and Tw_vec_exist[k] == 0:
                raise RuntimeError(f"Resume requested but /{k} group not present in HDF5.")
        if self._write_tdm > 0 and Tw_tdm_exist == 0:
            raise RuntimeError("Resume: /data/excitation/transition_density_matrices not present.")
        
        # Set indices
        self.i_data[mol] = (step_offset // self._data_every) if self._data_every > 0 else 0
        self.i_vec[mol] = {
            k: (step_offset // v) if v > 0 else 0 for k, v in self._cadence.items()
        }
        self.i_tdm[mol] = (step_offset // self._write_tdm) if self._write_tdm > 0 else 0
        
        # Update flags with existing capacities
        self.flags[mol].update({
            "Tw_data": Tw_data_exist,
            "Tw_tdm": Tw_tdm_exist,
            "Tw_vec": Tw_vec_exist,
        })
    
    def _create_new(self, h5_path: str, mol: int, molecule, Nat_mol: int, 
                   Norb_mol: int, R: int, Tw_data: int, Tw_vec: Dict, Tw_tdm: int):
        """Create new HDF5 file."""
        _rotate_existing(h5_path)
        h5 = h5py.File(h5_path, "w")
        self.handles[mol] = h5
        self.i_data[mol] = 0
        self.i_vec[mol] = {"coordinates": 0, "velocities": 0, "forces": 0}
        self.i_tdm[mol] = 0
        
        S = slice(0, Nat_mol)
        h5.create_dataset("atoms", data=_to_np(molecule.species[mol, S]))
        
        # Create /data group
        if Tw_data > 0:
            gd = h5.create_group("data")
            self._create_row_chunked(gd, "steps", (Tw_data,), np.int64)
            self._create_row_chunked(gd, "thermo/T", (Tw_data,))
            self._create_row_chunked(gd, "thermo/Ek", (Tw_data,))
            self._create_row_chunked(gd, "thermo/Ep", (Tw_data,))
            self._create_row_chunked(gd, "properties/ground_dipole", (Tw_data, 3))
            
            if R > 0:
                gd.create_dataset("excitation/active_state", data=int(molecule.active_state))
                self._create_row_chunked(gd, "excitation/excitation_energy", (Tw_data, R))
                self._create_row_chunked(gd, "excitation/transition_dipole", (Tw_data, R, 3))
                self._create_row_chunked(gd, "excitation/oscillator_strength", (Tw_data, R))
                
                if self._save_relaxed_dipole:
                    self._create_row_chunked(gd, "excitation/unrelaxed_dipole", (Tw_data, 3))
                    self._create_row_chunked(gd, "excitation/relaxed_dipole", (Tw_data, 3))
                
                if Tw_tdm > 0:
                    gtdm = gd["excitation"].create_group("transition_density_matrices")
                    self._create_row_chunked(gtdm, "steps", (Tw_tdm,), np.int64)
                    self._create_row_chunked(gtdm, "values", (Tw_tdm, R, Norb_mol, Norb_mol))
            
            if self._write_mo:
                restricted = self.flags[mol]["restricted"]
                shape = (Tw_data, 1) if restricted else (Tw_data, 2)
                self._create_row_chunked(gd, "mo/homo_lumo_gap", shape)
                nocc_data = int(molecule.nocc[mol].item()) if restricted else _to_np(molecule.nocc[mol])
                gd.create_dataset("mo/nocc", data=nocc_data)
        
        # Create vector groups
        for name, Tlen in Tw_vec.items():
            if Tlen > 0:
                g = h5.create_group(name)
                self._create_row_chunked(g, "steps", (Tlen,), np.int64)
                self._create_row_chunked(g, "values", (Tlen, Nat_mol, 3))
    
    def append_data(self, step_idx: int, molecule, T, Ek, Ep, e_gap):
        """Append scalar data (thermo, MO, excitations)."""
        for mol in self.config.molid:
            i = self.i_data.get(mol)
            if i is None or self.flags[mol]["Tw_data"] == 0:
                continue
            
            h5 = self.handles[mol]
            gd = h5["data"]
            flags = self.flags[mol]
            
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
                
                if self._save_relaxed_dipole:
                    gd["excitation/unrelaxed_dipole"][i, ...] = _to_np(molecule.cis_state_unrelaxed_dipole[mol])
                    gd["excitation/relaxed_dipole"][i, ...] = _to_np(molecule.cis_state_relaxed_dipole[mol])
                
                if flags.get("write_tdm") and (step_idx % flags["tdm_stride"]) == 0:
                    i_tdm = self.i_tdm[mol]
                    if i_tdm < flags["Tw_tdm"]:
                        gtdm = gd["excitation/transition_density_matrices"]
                        gtdm["steps"][i_tdm] = int(step_idx)
                        Norb = flags["Norb"]
                        gtdm["values"][i_tdm, ...] = _to_np(molecule.transition_density_matrices[mol, :R, :Norb, :Norb])
                        self.i_tdm[mol] = i_tdm + 1
            
            if flags.get("write_mo"):
                Norb = flags["Norb"]
                if flags["restricted"]:
                    gd["mo/homo_lumo_gap"][i, ...] = _to_np(e_gap[mol, None])
                else:
                    gd["mo/homo_lumo_gap"][i, ...] = _to_np(e_gap[mol])
            
            self.i_data[mol] = i + 1
            if (i + 1) % 100 == 0:
                h5.flush()
    
    def append_vectors(self, step_idx: int, molecule):
        """Append vector data (coordinates, velocities, forces)."""
        for mol in self.config.molid:
            h5 = self.handles[mol]
            f = self.flags[mol]
            S = f["active_slice"]
            
            tensors = {
                "coordinates": molecule.coordinates[mol, S, :],
                "velocities": molecule.velocities[mol, S, :],
                "forces": molecule.force[mol, S, :],
            }
            
            did_write = False
            for name, arr in tensors.items():
                stride = f["stride"][name]
                if stride <= 0 or (step_idx % stride) != 0:
                    continue
                
                i = self.i_vec[mol][name]
                if i >= f["Tw_vec"][name]:
                    continue
                
                g = h5[name]
                g["steps"][i] = int(step_idx)
                g["values"][i] = _to_np(arr)
                self.i_vec[mol][name] = i + 1
                did_write = did_write or (self.i_vec[mol][name] % 100 == 0)
            
            if did_write:
                h5.flush()
    
    def flush(self):
        """Flush all HDF5 buffers."""
        for h in self.handles.values():
            try:
                h.flush()
            except Exception:
                pass
    
    def close(self):
        """Close all HDF5 files."""
        self.flush()
        for h in self.handles.values():
            try:
                h.close()
            except Exception:
                pass
        self.handles.clear()
        self.i_data.clear()
        self.i_vec.clear()
        self.i_tdm.clear()
        self.flags.clear()


class XYZWriter:
    """Manages XYZ file writing for MD trajectories."""
    
    def __init__(self, output_config: OutputConfig, step_offset: int = 0):
        self.config = output_config
        self.step_offset = step_offset
        self.files: Dict[int, Any] = {}
    
    def open(self):
        """Open XYZ files for writing."""
        bufsize = 1_048_576  # 1 MB buffer
        for mol in self.config.molid:
            xyz_fn = f"{self.config.prefix}.{mol}.xyz"
            if self.step_offset == 0:
                _rotate_existing(xyz_fn)
            self.files[mol] = open(xyz_fn, 'a+', buffering=bufsize)
    
    def write(self, step: int, molecule, Ek, L):
        """Write XYZ frame for current step."""
        Et = (Ek + L).detach().cpu()
        
        for mol in self.config.molid:
            n_atoms = int(torch.sum(molecule.species[mol] > 0))
            s = StringIO()
            s.write(f"{n_atoms}\n")
            s.write(f"step: {step+1}  E_total = {float(Et[mol]):12.9f}  \n")
            
            xyz = molecule.coordinates[mol]
            Z = molecule.species[mol]
            for a in range(n_atoms):
                label = molecule.const.label[Z[a].item()]
                x, y, z = xyz[a, 0].item(), xyz[a, 1].item(), xyz[a, 2].item()
                s.write(f"{label} {x:15.5f} {y:15.5f} {z:15.5f}\n")
            
            self.files[mol].write(s.getvalue())
    
    def flush(self):
        """Flush all file buffers."""
        for fh in self.files.values():
            try:
                fh.flush()
            except Exception:
                pass
    
    def close(self):
        """Close all XYZ files."""
        for fh in self.files.values():
            try:
                fh.close()
            except Exception:
                pass
        self.files.clear()


class Geometry_Optimization_SD(torch.nn.Module):
    """
    steepest descent algorithm for geometry optimization
    pass in function for Eelec and EnucAB and current coordinates
    use line search to choose best alpha
    """

    def __init__(self, seqm_parameters, alpha=0.01, force_tol=1.0e-4, max_evl=1000):
        r"""
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
    """Base class for molecular dynamics simulations."""
    
    def __init__(self, seqm_parameters, timestep=1.0, Temp=0.0, step_offset=0, 
                 output=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seqm_parameters = seqm_parameters
        self.timestep = timestep
        self.esdriver = esdriver(self.seqm_parameters)
        self.Temp = Temp
        self.step_offset = step_offset
        
        self.output_config = OutputConfig.from_dict(output)
        self.n_dof = None
        self.remove_com_angular = False
        self.do_remove_com = False
        self.remove_com_stride = 0
        self.start_time = None
        
        self._h5_writer: Optional[HDF5Writer] = None
        self._xyz_writer: Optional[XYZWriter] = None
    
    @property
    def output(self):
        """Backward compatibility property."""
        return {
            'molid': self.output_config.molid,
            'prefix': self.output_config.prefix,
            'print every': self.output_config.print_every,
            'checkpoint every': self.output_config.checkpoint_every,
            'xyz': self.output_config.xyz_every,
            'h5': self.output_config.h5_config,
        }
    
    def initialize_velocity(self, molecule, vel_com=True):
        """Initialize velocities from Maxwell-Boltzmann distribution."""
        # Check device compatibility
        md_dev = self.esdriver.conservative_force.energy.packpar.p.device
        mol_dev = molecule.coordinates.device
        if md_dev != mol_dev:
            raise RuntimeError(f"MD object on {md_dev}, molecule on {mol_dev}")
        
        if self.n_dof is None:
            raise RuntimeError("n_dof not set. Call initialize() first")
        
        if torch.is_tensor(molecule.velocities):
            return molecule.velocities
        
        if self.Temp == 0.0:
            molecule.velocities = torch.zeros_like(molecule.coordinates)
            return molecule.velocities
        
        # Sample from Maxwell-Boltzmann
        scale = torch.sqrt(self.Temp * molecule.mass_inverse) * CONSTANTS.VEL_SCALE
        molecule.velocities = torch.randn_like(molecule.coordinates) * scale
        
        # Rescale to exact temperature
        Ek = self._kinetic_energy(molecule)
        T1 = self._calc_temperature(Ek)
        alpha = torch.sqrt(self.Temp / T1)
        molecule.velocities.mul_(alpha.reshape(-1, 1, 1))
        
        if vel_com:
            self._zero_com(molecule, translate_to_origin=True)
        
        return molecule.velocities
    
    def _zero_com(self, molecule, remove_angular=True, translate_to_origin=False):
        """Remove center of mass motion."""
        mass = molecule.mass
        M = torch.sum(mass, dim=1, keepdim=True)
        Ek_initial = self._kinetic_energy(molecule)
        
        with torch.no_grad():
            r_com = torch.sum(mass * molecule.coordinates, dim=1, keepdim=True) / M
            r_rel = molecule.coordinates - r_com
            if translate_to_origin:
                molecule.coordinates.copy_(r_rel)
            
            v_com = torch.sum(mass * molecule.velocities, dim=1, keepdim=True) / M
            molecule.velocities.sub_(v_com)
            
            if remove_angular:
                L = torch.sum(mass * torch.linalg.cross(r_rel, molecule.velocities, dim=2), dim=1)
                eye = torch.eye(3, dtype=molecule.coordinates.dtype, device=molecule.coordinates.device)
                I = (torch.sum(mass * (r_rel * r_rel).sum(dim=2, keepdim=True), dim=1, keepdim=True) * eye.reshape(1,3,3)
                     - torch.sum(mass.unsqueeze(3) * r_rel.unsqueeze(3) * r_rel.unsqueeze(2), dim=1))
                omega = (torch.linalg.pinv(I) @ L.unsqueeze(2)).squeeze(-1)
                molecule.velocities.sub_(torch.linalg.cross(omega.unsqueeze(1).expand_as(r_rel), r_rel, dim=2))
            
            # Restore kinetic energy
            Ek_after = self._kinetic_energy(molecule)
            if torch.any(Ek_after < 1e-12):
                raise RuntimeError("Zero kinetic energy after removing COM momentum")
            alpha = torch.sqrt(Ek_initial / Ek_after)
            molecule.velocities.mul_(alpha.reshape(-1, 1, 1))
    
    def _kinetic_energy(self, molecule):
        """Calculate kinetic energy."""
        return torch.sum(0.5 * molecule.mass * molecule.velocities**2, dim=(1, 2)) * CONSTANTS.KINETIC_ENERGY_SCALE
    
    def _calc_temperature(self, kinetic_energy):
        """Calculate temperature from kinetic energy."""
        return kinetic_energy * CONSTANTS.TEMPERATURE_SCALE / (0.5 * self.n_dof)
    
    def _output_to_screen(self, step: int, T, Ek, V):
        """Print MD data to screen."""
        if step == 0:
            print("Step,    Temp,    E(kinetic),  E(potential),  E(total)")
        print(f"{step+1:6d}", end="")
        for mol in self.output_config.molid:
            Tm = float(T[mol].detach().cpu())
            Ekm = float(Ek[mol].detach().cpu())
            Vm = float(V[mol].detach().cpu())
            print(f" {Tm:8.2f}   {Ekm:e} {Vm:e} {Vm+Ekm:e} || ", end="")
        print()
    
    def set_dof(self, molecule, constraints=0.0):
        """Set degrees of freedom."""
        self.n_dof = 3.0 * molecule.num_atoms - constraints
    
    def _thermo_potential(self, molecule):
        """Potential energy for thermodynamics (override in subclasses)."""
        return molecule.Etot
    
    def one_step(self, molecule, learned_parameters=dict(), *args, **kwargs):
        """Perform one velocity Verlet integration step."""
        dt = self.timestep
        if molecule.const.do_timing:
            t0 = time.time()
        
        with torch.no_grad():
            molecule.velocities.add_(0.5 * molecule.acc * dt)
            molecule.coordinates.add_(molecule.velocities * dt)
        
        self.esdriver(molecule, learned_parameters=learned_parameters, P0=molecule.dm,
                     dm_prop='SCF', cis_amp=molecule.cis_amplitudes, *args, **kwargs)
        
        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE
            molecule.velocities.add_(0.5 * molecule.acc * dt)
        
        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            molecule.const.timing["MD"].append(time.time() - t0)
    
    def _do_integrator_step(self, i, molecule, learned_parameters, **kwargs):
        """Hook for subclasses to override integration."""
        return self.one_step(molecule, learned_parameters=learned_parameters, **kwargs)
    
    def initialize(self, molecule, remove_com=None, learned_parameters=dict(), *args, **kwargs):
        """Initialize MD simulation."""
        molecule.verbose = False # Dont print SCF and CIS/RPA results
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
            self.esdriver(molecule, learned_parameters=learned_parameters, 
                         P0=molecule.dm, cis_amp=molecule.cis_amplitudes, *args, **kwargs)
        
        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE
    
    def run(self, molecule, steps, learned_parameters=dict(), reuse_P=True, 
            remove_com=None, seed=None, *args, **kwargs):
        """Run molecular dynamics simulation."""
        self.start_time = datetime.now()
        print(f"MD run began at {self.start_time}", flush=True)
        
        if seed is not None:
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
        
        if getattr(self, "k", None) is not None:
            reuse_P = True
        
        self.initialize(molecule, remove_com=remove_com, 
                       learned_parameters=learned_parameters, *args, **kwargs)
        
        if not reuse_P:
            molecule.dm = None
            molecule.cis_amplitudes = None
        
        # Setup output
        do_screen = self.output_config.print_every > 0
        do_xyz = self.output_config.xyz_every > 0
        do_h5 = (self.output_config.get_h5_data_every() > 0 or 
                any(self.output_config.get_h5_cadence().values()))
        
        # Velocity scaling / energy shift
        do_scale_vel = ("scale_vel" in kwargs)
        if do_scale_vel:
            scale_freq, T_target = kwargs["scale_vel"]
            scale_freq = int(scale_freq)
            T_target = torch.as_tensor(T_target, dtype=molecule.coordinates.dtype, 
                                      device=molecule.coordinates.device)
        
        do_energy_shift = bool(kwargs.get("control_energy_shift", False))
        if do_scale_vel and do_energy_shift:
            raise ValueError("Cannot scale velocities and fix energy shift simultaneously.")
        
        E0 = None
        checkpoint_every = self.output_config.checkpoint_every
        checkpoint_path = f'{self.output_config.prefix}.restart.pt'
        
        try:
            if do_h5:
                excited_states_params = self.seqm_parameters.get('excited_states')
                n_roots = excited_states_params["n_states"] if excited_states_params else 0
                # With XL-ESMD, only the active state is computed
                if isinstance(self, XL_ESMD):
                    n_roots = 1
                self._h5_writer = HDF5Writer(self.output_config, self.seqm_parameters)
                self._h5_writer.open(molecule, self.output_config.prefix, steps, 
                                    n_roots, resume=(self.step_offset > 0), 
                                    step_offset=self.step_offset)
            
            if do_xyz:
                self._xyz_writer = XYZWriter(self.output_config, self.step_offset)
                self._xyz_writer.open()
            
            for i in range(self.step_offset, steps):
                self._do_integrator_step(i, molecule, learned_parameters, *args, **kwargs)
                
                with torch.no_grad():
                    if torch.is_tensor(molecule.coordinates.grad):
                        molecule.coordinates.grad.zero_()
                    
                    if not reuse_P:
                        molecule.dm = None
                        molecule.cis_amplitudes = None
                    
                    if self.do_remove_com and (i % self.remove_com_stride == 0):
                        self._zero_com(molecule, remove_angular=self.remove_com_angular)
                    
                    Ek = self._kinetic_energy(molecule)
                    T = self._calc_temperature(Ek)
                    V = self._thermo_potential(molecule)
                    if E0 is None:
                        E0 = V + Ek

                    # if scaling velocities to control temperature
                    if do_scale_vel and ((i + 1) % scale_freq == 0):
                        alpha = torch.sqrt(torch.clamp(T_target / T, min=0.0))
                        molecule.velocities.mul_(alpha.reshape(-1, 1, 1))
                        Ek = self._kinetic_energy(molecule)
                        T = self._calc_temperature(Ek)
                    
                    if do_energy_shift:
                        #scale velocities to adjust kinetic energy and compenstate the energy shift
                        Eshift = Ek + V - E0
                        alpha = torch.sqrt((Ek - Eshift) / Ek)
                        alpha[~torch.isfinite(alpha)] = 0.0
                        molecule.velocities.mul_(alpha.reshape(-1, 1, 1))
                        Ek = self._kinetic_energy(molecule)
                        T = self._calc_temperature(Ek)
                    
                    if do_screen and ((i + 1) % self.output_config.print_every == 0):
                        self._output_to_screen(i, T, Ek, V)
                    
                    if do_h5:
                        if (self.output_config.get_h5_data_every() > 0 and 
                            (i + 1) % self.output_config.get_h5_data_every() == 0):
                            self._h5_writer.append_data(i + 1, molecule, T, Ek, V, molecule.e_gap)
                        if (self.output_config.h5_vectors_every and 
                            (i + 1) % self.output_config.h5_vectors_every == 0): 
                            self._h5_writer.append_vectors(i + 1, molecule)
                    
                    if do_xyz and ((i + 1) % self.output_config.xyz_every == 0):
                        self._xyz_writer.write(i, molecule, Ek, V)
                    
                    if checkpoint_every > 0 and ((i + 1) % checkpoint_every == 0):
                        self._flush_all()
                        self.save_checkpoint(molecule, steps, reuse_P, remove_com, 
                                           step_done=i + 1, path=checkpoint_path)
                
                del Ek, T
                if i % 1000 == 0:
                    torch.cuda.empty_cache()
        
        finally:
            if self._xyz_writer:
                self._xyz_writer.close()
            if self._h5_writer:
                self._h5_writer.close()
        
        now = datetime.now()
        print(f"MD run ended at {now}")
        print(f"Time elapsed since the beginning of MD run: {now-self.start_time}",flush=True)
        return molecule.coordinates, molecule.velocities, molecule.acc
    
    def _flush_all(self):
        """Flush all output buffers."""
        if self._h5_writer:
            self._h5_writer.flush()
        if self._xyz_writer:
            self._xyz_writer.flush()
    
    def save_checkpoint(self, molecule, steps: int, reuse_P, remove_com, 
                       *, step_done: int, path: str):
        """Save checkpoint for restart."""
        def _tensor_cpu(x):
            return x.detach().cpu() if torch.is_tensor(x) else x
        
        ckpt = {
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
        
        if hasattr(self, "m"):  # XL-BOMD variants
            ckpt["xl_ctx"] = {
                "Pt": _tensor_cpu(self._xl_ctx["Pt"]),
                "es_amp_t": _tensor_cpu(self._xl_ctx.get("es_amp_t"))
            }
            ckpt["dP2dt2"] = _tensor_cpu(molecule.dP2dt2)
        
        # Atomic write
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), 
                                           prefix=".tmp_ckpt_", suffix=".pt")
        os.close(tmp_fd)
        try:
            torch.save(ckpt, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
        
        now = datetime.now()
        print(f"Saved checkpoint at {now}")
        print(f"Time elapsed since the beginning of MD run: {now-self.start_time}",flush=True)

    @staticmethod
    def run_from_checkpoint(path: str, device=None):
        """Load and resume from checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        
        torch.random.set_rng_state(ckpt["rng"]["torch_cpu"])
        if torch.cuda.is_available() and ckpt["rng"]["torch_cuda"]:
            torch.cuda.set_rng_state_all(ckpt["rng"]["torch_cuda"])

        # rebuild objects
        from seqm.Molecule import Molecule
        
        torch.set_default_dtype(ckpt["molecules"]["coordinates"].dtype)
        device = device or ckpt["device"]
        const = ckpt["molecules"]["constants"].to(device)
        
        molecule = Molecule(const, ckpt["seqm_parameters"], 
                          ckpt["molecules"]["coordinates"].to(device),
                          ckpt["molecules"]["species"].to(device)).to(device)
        
        reuse_P = ckpt["reuse_P"]
        with torch.no_grad():
            molecule.velocities = ckpt["molecules"]["velocities"].to(device)
            molecule.force = ckpt["molecules"]["forces"].to(device)
            molecule.dm = ckpt["molecules"]["dm"].to(device) if reuse_P else None
            molecule.cis_amplitudes = ckpt["molecules"]["cis_amplitudes"]
            if isinstance(molecule.cis_amplitudes, torch.Tensor):
                molecule.cis_amplitudes = molecule.cis_amplitudes.to(device)
                molecule.old_mos = ckpt["molecules"]["old_mos"].to(device)
            if "dP2dt2" in ckpt:
                molecule.dP2dt2 = ckpt["dP2dt2"].to(device)
        
        md_type = ckpt["MD_type"]
        md_classes = {
            "Molecular_Dynamics_Basic": Molecular_Dynamics_Basic,
            "Molecular_Dynamics_Langevin": Molecular_Dynamics_Langevin,
            "XL_BOMD": XL_BOMD,
            "KSA_XL_BOMD": KSA_XL_BOMD,
        }
        
        if md_type not in md_classes:
            raise RuntimeError(f"Unknown MD type '{md_type}' in checkpoint")
        
        md_cls = md_classes[md_type]
        kwargs = {
            "seqm_parameters": ckpt["seqm_parameters"],
            "timestep": ckpt["timestep"],
            "Temp": ckpt["Temp"],
            "output": ckpt["output"],
            "step_offset": ckpt["step_done"],
        }
        
        if md_type in ("Molecular_Dynamics_Langevin", "XL_BOMD", "KSA_XL_BOMD"):
            kwargs["damp"] = ckpt["damp"]
        if md_type in ("XL_BOMD", "KSA_XL_BOMD"):
            kwargs["xl_bomd_params"] = ckpt["xl_bomd_params"]
        
        md = md_cls(**kwargs).to(device)
        
        if md_type in ("XL_BOMD", "KSA_XL_BOMD"):
            xl = ckpt["xl_ctx"]
            Pt = xl["Pt"].to(device)
            es_amp_t = xl.get("es_amp_t")
            xl_m = ckpt["xl_bomd_params"]['k'] + 1
            cindx = (ckpt["step_done"] - 1) % xl_m # subtract one because step_done is advanced by one step
            P = Pt[(xl_m - 1 - cindx)].clone()
            es_amp = None
            if isinstance(es_amp_t, torch.Tensor):
                es_amp_t = es_amp_t.to(device)
                es_amp = es_amp_t[(xl_m - 1 - cindx)].clone()
            md._xl_ctx = {"P": P, "Pt": Pt, "es_amp": es_amp, "es_amp_t": es_amp_t}
        
        md.run(molecule=molecule, steps=ckpt["steps"], 
              reuse_P=reuse_P, remove_com=ckpt["remove_com"])


def _to_np(x):
    """Convert tensor to numpy array."""
    return x.detach().cpu().numpy()


class Molecular_Dynamics_Langevin(Molecular_Dynamics_Basic):
    """MD with Langevin thermostat."""
    
    def __init__(self, damp=50.0, *args, **kwargs):
        """
        damp is damping factor in unit of time (fs)
        Temp : temperature in unit of Kelvin

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
    
    def _apply_langevin_thermostat(self, molecule):
        """Apply Langevin thermostat."""
        with torch.no_grad():
            molecule.velocities.mul_(self.langevin_c1)
            molecule.velocities.add_(self.langevin_c2 * torch.randn_like(molecule.velocities))
    
    def one_step(self, molecule, learned_parameters=dict(), *args, **kwargs):
        """Velocity Verlet with Langevin thermostat."""
        dt = self.timestep
        if molecule.const.do_timing:
            t0 = time.time()
        
        self._apply_langevin_thermostat(molecule)
        
        with torch.no_grad():
            molecule.velocities.add_(0.5 * molecule.acc * dt)
            molecule.coordinates.add_(molecule.velocities * dt)
        
        self.esdriver(molecule, learned_parameters=learned_parameters, P0=molecule.dm,
                     dm_prop='SCF', cis_amp=molecule.cis_amplitudes, *args, **kwargs)
        
        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE
            molecule.velocities.add_(0.5 * molecule.acc * dt)
        
        self._apply_langevin_thermostat(molecule)
        
        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            molecule.const.timing["MD"].append(time.time() - t0)
    
    def initialize(self, molecule, remove_com=None, learned_parameters=dict(), *args, **kwargs):
        if self.damp is not None:
            dt = self.timestep
            # s = -γ dt
            s = torch.as_tensor(-dt / self.damp, dtype=molecule.coordinates.dtype, device=molecule.coordinates.device)
            # c1 = exp{-γ dt/2}
            self.langevin_c1 = torch.exp(0.5 * s)
            # c2 = 1 - c1^2 = 1 - e^{-γ dt}
            one_me = -torch.expm1(s)
            self.langevin_c2 = torch.sqrt(one_me * self.Temp * molecule.mass_inverse) * CONSTANTS.VEL_SCALE
        return super().initialize(molecule, remove_com, learned_parameters, *args, **kwargs)


class XL_BOMD(Molecular_Dynamics_Langevin):
    """Extended Lagrangian Born-Oppenheimer MD."""
    
    def __init__(self, damp=None, xl_bomd_params=dict(), *args, **kwargs):
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
        
        self.m = self.k + 1
        self.kappa = self.coeffs[self.k][0]
        self.alpha = self.coeffs[self.k][1]
        cc = 1.00
        tmp = torch.as_tensor(self.coeffs[self.k][2:]) * self.alpha
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
        self.add_spherical_potential = False  # Spherical force to prevent atoms from flying off beyond a certain radius
        self.do_scf = False
        self.move_on_excited_state = False
    
    def set_dof(self, molecule, constraints=0.0):
        if self.damp is not None:
            constraints = 0.0
        self.n_dof = 3.0 * molecule.num_atoms - constraints
    
    def _propagate_P(self, P, Pt, cindx, molecule):
        """Propagate density matrix."""
        # eq. 22 in https://doi.org/10.1063/1.3148075
        #### Scaling delta function. Use eq with c if stability problems occur.
        # P(n+1) = coeff_D * [ c*D(n) + (1-c)*P(n) ] + sum_j coeff[j] * Pt[j]
        c = 0.95
        P_new = (self.coeff_D * (c * molecule.dm + (1.0 - c) * P) + 
                torch.sum(self.coeff[cindx:(cindx+self.m)].reshape(-1,1,1,1) * Pt, dim=0))
        return P_new
    
    def _propagate_excited_state(self, es_amp, es_amp_t, cindx, molecule):
        """Propagate excited state transition density matrices."""
        c = 0.95
        es_new = (self.coeff_D * (c * molecule.transition_density_matrices + (1.0 - c) * es_amp) +
                 torch.sum(self.coeff[cindx:(cindx+self.m)].reshape(-1,1,1,1,1) * es_amp_t, dim=0))
        return es_new
    
    def one_step(self, molecule, step, P, Pt, es_amp=None, es_amp_t=None, 
                learned_parameters=dict(), *args, **kwargs):
        """XL-BOMD integration step."""
        dt = self.timestep
        if molecule.const.do_timing:
            t0 = time.time()
        
        if self.damp:
            self._apply_langevin_thermostat(molecule)
        
        with torch.no_grad():
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
            P = self._propagate_P(P, Pt, cindx, molecule)
            Pt[(self.m - 1 - cindx)] = P
            
            if molecule.active_state > 0:
                es_amp = self._propagate_excited_state(es_amp, es_amp_t, cindx, molecule)
                es_amp_t[(self.m - 1 - cindx)] = es_amp
                es_amp_ortho = es_amp
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
        
        self.esdriver(molecule, learned_parameters=learned_parameters,
                     xl_bomd_params=self.xl_bomd_params, P0=P0, cis_amp=es_amp_ortho,
                     dm_prop=calc_type, *args, **kwargs)
        
        if self.add_spherical_potential:  # don't do this unless necessary
            with torch.no_grad():
                dE, dF = Spherical_Pot_Force(molecule, radius=14.85, k=0.1)
                molecule.Etot = molecule.Etot + dE
                molecule.force = molecule.force + dF
        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE
            molecule.velocities.add_(0.5 * molecule.acc * dt)
        
        if self.damp:
            self._apply_langevin_thermostat(molecule)
        
        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            molecule.const.timing["MD"].append(time.time() - t0)
        
        return P, Pt, es_amp, es_amp_t
    
    def _thermo_potential(self, molecule):
        # XL-BOMD potential energy includes Electronic_entropy (why?)
        return molecule.Etot + molecule.Electronic_entropy
    
    def _do_integrator_step(self, i, molecule, learned_parameters, **kwargs):
        # Expect self._xl_ctx holding P, Pt, es_amp, es_amp_t initialized in initialize()
        P, Pt = self._xl_ctx["P"], self._xl_ctx["Pt"]
        es_amp, es_amp_t = self._xl_ctx.get("es_amp"), self._xl_ctx.get("es_amp_t")
        P, Pt, es_amp, es_amp_t = self.one_step(molecule, i, P, Pt, es_amp, es_amp_t,
                                                learned_parameters=learned_parameters, **kwargs)
        self._xl_ctx.update(P=P, Pt=Pt, es_amp=es_amp, es_amp_t=es_amp_t)
    
    def initialize(self, molecule, remove_com=None, learned_parameters=dict(), 
                  do_xl_esmd=False, *args, **kwargs):
        if molecule.active_state > 0:
            self.move_on_excited_state = True
            self.esdriver.conservative_force.energy.excited_states['save_tdm'] = True
        
        super().initialize(molecule, remove_com=remove_com, 
                          learned_parameters=learned_parameters, *args, **kwargs)
        
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
                raise ValueError("KSA-XL-BOMD not supported for excited state dynamics")
            
            self.do_scf = True
            self.esdriver.conservative_force.energy.hamiltonian.eps = torch.nn.Parameter(
                torch.as_tensor(scf_eps), requires_grad=False)
            self.esdriver.conservative_force.energy.excited_states['tolerance'] = es_eps
            molecule.Electronic_entropy = torch.zeros(molecule.species.shape[0], 
                                                     device=molecule.coordinates.device)
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
        
        with torch.no_grad():
            P = molecule.dm.clone()
            Pt = molecule.dm.unsqueeze(0).expand((self.m,) + molecule.dm.shape).clone()
            if 'max_rank' in self.xl_bomd_params:
                molecule.dP2dt2 = torch.zeros_like(molecule.dm)
            ctx = {"P": P, "Pt": Pt}

            # for xl-esmd since we propagate the transition_density for only the active state, save only that
            if do_xl_esmd:
                active_idx = molecule.active_state - 1  # Convert 1-indexed to 0-indexed
                molecule.transition_density_matrices = molecule.transition_density_matrices[:,active_idx:active_idx+1]
                # For XL_ESMD, keep only the active state data for all excited state properties
                # These are the properties that get saved to HDF5
                if torch.is_tensor(molecule.cis_energies) and molecule.cis_energies.shape[1] > 1:
                    molecule.cis_energies = molecule.cis_energies[:, active_idx:active_idx+1]
                if torch.is_tensor(molecule.transition_dipole) and molecule.transition_dipole.shape[1] > 1:
                    molecule.transition_dipole = molecule.transition_dipole[:, active_idx:active_idx+1, :]
                if torch.is_tensor(molecule.oscillator_strength) and molecule.oscillator_strength.shape[1] > 1:
                    molecule.oscillator_strength = molecule.oscillator_strength[:, active_idx:active_idx+1]
            
            if self.move_on_excited_state:
                es_amp = molecule.transition_density_matrices.clone()
                es_amp_t = es_amp.unsqueeze(0).expand((self.m,) + es_amp.shape).clone()
                ctx.update(es_amp=es_amp, es_amp_t=es_amp_t)
            self._xl_ctx = ctx


class KSA_XL_BOMD(XL_BOMD):
    """Krylov Subspace Approximation XL-BOMD."""
    
    def __init__(self, damp=None, xl_bomd_params=dict(), *args, **kwargs):
        super().__init__(damp, xl_bomd_params, *args, **kwargs)
        self.add_spherical_potential = False
    
    def _propagate_P(self, P, Pt, cindx, molecule):
        P_new = (self.coeff_D * (molecule.dP2dt2 + P) +
                torch.sum(self.coeff[cindx:(cindx+self.m)].reshape(-1,1,1,1) * Pt, dim=0))
        return P_new


class XL_ESMD(XL_BOMD):
    """XL-BOMD for excited state MD."""
    
    def one_step(self, molecule, step, P, Pt, es_amp=None, es_amp_t=None,
                learned_parameters=dict(), *args, **kwargs):
        dt = self.timestep
        if molecule.const.do_timing:
            t0 = time.time()
        
        if self.damp:
            self._apply_langevin_thermostat(molecule)
        
        with torch.no_grad():
            molecule.velocities.add_(0.5 * molecule.acc * dt)
            molecule.coordinates.add_(molecule.velocities * dt)
            
            cindx = step % self.m
            P = self._propagate_P(P, Pt, cindx, molecule)
            Pt[(self.m - 1 - cindx)] = P
            
            es_amp = self._propagate_excited_state(es_amp, es_amp_t, cindx, molecule)
            es_amp_t[(self.m - 1 - cindx)] = es_amp

            # Purify with McWeeny polynomial since P may not be idempotent
            # 3P^2 - 2P^3
            # For restricted density matrix (spin summed) D = 2P. So to purify, D0 = 3/2 D^2 - 1/2 D^3
            # TODO: Make it work for unrestricted P
            # P2 = P @ P
            # P0 = torch.baddbmm(P2, P2, P, beta=1.5, alpha=-0.5)
            P0 = P
        
        # dm_prop = 'SCF'
        dm_prop = 'XL-BOMD'
        self.esdriver(molecule, learned_parameters=learned_parameters,
                     xl_bomd_params=self.xl_bomd_params, P0=P0, cis_amp=es_amp,
                     dm_prop=dm_prop, *args, **kwargs)
        
        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE
            molecule.velocities.add_(0.5 * molecule.acc * dt)
        
        if self.damp:
            self._apply_langevin_thermostat(molecule)
        
        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            molecule.const.timing["MD"].append(time.time() - t0)
        
        return P, Pt, es_amp, es_amp_t
    
    def initialize(self, molecule, remove_com=None, learned_parameters=dict(), *args, **kwargs):
        super().initialize(molecule, remove_com=remove_com, 
                          learned_parameters=learned_parameters, do_xl_esmd=True, *args, **kwargs)
        molecule.Electronic_entropy = torch.zeros(molecule.species.shape[0], 
                                                 device=molecule.coordinates.device)
        self.esdriver.conservative_force.energy.excited_states = None
        self.esdriver.conservative_force.energy.xlesmd = True

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
= sqrt(1.0/1.160451812e4 * 1.602176565e-19 * 1.66053906660e-27)/1.0e-15 /1.602176565e-19 / 1.0e10  eV/Angstrom
= sqrt(1.0/1.160451812e4 / 1.602176565 * 1.66053906660) * 0.1 eV/Angstrom
= 0.09450522179973914 eV/Angstrom
"""
#acc ==> Angstrom/fs^2

def _rotate_existing(path, start=1, max_tries=99, error_on_max=True):
    """Rotate existing file to .bak.N"""
    if not os.path.exists(path):
        return
    base, ext = os.path.splitext(path)
    for n in range(start, max_tries + 1):
        backup = f"{base}.bak.{n}{ext}"
        if not os.path.exists(backup):
            os.rename(path, backup)
            return
    if error_on_max:
        raise RuntimeError(f"Unable to backup: {path}")

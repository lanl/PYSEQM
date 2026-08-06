import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from typing import Any, Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
import torch

from seqm.basics import Force
from seqm.dynamics.active_state import active_state_tensor
from seqm.ElectronicStructure import Electronic_Structure as esdriver
from seqm.seqm_functions.fock import enable_fock_compile
from seqm.seqm_functions.omx_utils import OMX_METHODS
from seqm.seqm_functions.rcis_batch import enable_rcis_compile
from seqm.seqm_functions.spherical_pot_force import Spherical_Pot_Force
from seqm.seqm_functions.two_elec_two_center_int import enable_two_center_compile
from seqm.utils.torch_compile import normalize_torch_compile_config

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
    checkpoint_every: int = 0
    xyz_every: int = 0
    h5_config: Dict[str, Any] = field(default_factory=dict)
    h5_vectors_every: Optional[int] = None

    @classmethod
    def from_dict(cls, config: Optional[Dict] = None) -> "OutputConfig":
        """Create OutputConfig from dictionary with backward compatibility."""
        if not config:
            return cls(molid=[], print_every=0, checkpoint_every=0, xyz_every=0)
        config = dict(config)

        # Backward compatibility
        if "thermo" in config and "print_every" not in config:
            config["print_every"] = int(config["thermo"])
        if "dump" in config:
            dump = int(config["dump"])
            config.setdefault("xyz", dump)
            config.setdefault("h5", {}).setdefault("data", dump)

        wanted_vectors = {"coordinates", "velocities", "forces"}
        h5 = config.get("h5", {}) or {}
        vals = [v for k, v in h5.items() if k in wanted_vectors and isinstance(v, int) and v > 0]
        vectors_every = min(vals, default=None)

        return cls(
            molid=config.get("molid", [0]),
            prefix=config.get("prefix", "md"),
            print_every=int(config.get("print every", 1)),
            checkpoint_every=int(config.get("checkpoint every", 100)),
            xyz_every=int(config.get("xyz", 0)),
            h5_config=dict(config.get("h5", {}) or {}),
            h5_vectors_every=vectors_every,
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

    def get_h5_tdm_mode(self) -> str:
        mode = str(self.h5_config.get("transition_density_matrices_mode", "full")).strip().lower()
        if mode not in ("full", "diag"):
            raise ValueError("output.h5.transition_density_matrices_mode only supports 'full' and 'diag'.")
        return mode

    def get_h5_transition_properties(self) -> bool:
        return bool(self.h5_config.get("transition_properties", False))

    def get_h5_write_nonadiabatic(self) -> int:
        return int(self.h5_config.get("nonadiabatic", 0))


@dataclass
class _H5Track:
    """A fixed-capacity stepped group, optionally with a values dataset."""

    group: h5py.Group
    steps: h5py.Dataset
    count: int = 0
    values: Optional[h5py.Dataset] = None

    @property
    def capacity(self) -> int:
        return int(self.steps.shape[0])

    def commit(self) -> None:
        self.group.attrs["n_written"] = np.int64(self.count)


@dataclass
class _MoleculeState:
    h5: h5py.File
    nat: int
    norb: int
    restricted: bool
    data: Optional[_H5Track] = None
    nonadiabatic: Optional[_H5Track] = None
    series: Dict[str, _H5Track] = field(default_factory=dict)

    def tracks(self) -> Iterator[_H5Track]:
        if self.data is not None:
            yield self.data
        if self.nonadiabatic is not None:
            yield self.nonadiabatic
        yield from self.series.values()


@dataclass(frozen=True)
class _H5Layout:
    """Fixed capacities for all HDF5 output streams in one MD run."""

    data: int
    tdm: int
    nonadiabatic: int
    vectors: Dict[str, int]


class HDF5Writer:
    """Write fixed-size HDF5 molecular-dynamics trajectories."""

    _SCHEMA_VERSION = 2
    _CHUNK_TARGET_BYTES = 512 << 10

    def __init__(self, output_config: "OutputConfig", seqm_parameters: Dict, timestep: float):
        self.config = output_config
        self.seqm_parameters = seqm_parameters
        self.timestep = float(timestep)
        self._states: Dict[int, _MoleculeState] = {}

        self._cadence = {str(k): int(v) for k, v in output_config.get_h5_cadence().items()}
        self._data_every = int(output_config.get_h5_data_every())
        self._write_mo = bool(output_config.get_h5_write_mo())
        self._write_tdm = int(output_config.get_h5_write_tdm())
        self._tdm_mode = str(output_config.get_h5_tdm_mode())
        self._write_transition_properties = bool(output_config.get_h5_transition_properties())
        self._write_nonadiabatic = int(output_config.get_h5_write_nonadiabatic())

        self._steps = 0
        self._nstates = 0
        self._include_initial = False

    @staticmethod
    def _n_timepoints(steps: int, stride: int, include_initial: bool) -> int:
        return 0 if stride <= 0 else steps // stride + int(include_initial)

    def _due(self, step: int, stride: int) -> bool:
        return (
            stride > 0
            and 0 <= step <= self._steps
            and step % stride == 0
            and (step != 0 or self._include_initial)
        )

    def _layout(self) -> _H5Layout:
        return _H5Layout(
            data=self._n_timepoints(self._steps, self._data_every, self._include_initial),
            tdm=self._n_timepoints(self._steps, self._write_tdm, self._include_initial),
            nonadiabatic=self._n_timepoints(self._steps, self._write_nonadiabatic, self._include_initial),
            vectors={
                name: self._n_timepoints(self._steps, stride, self._include_initial)
                for name, stride in self._cadence.items()
            },
        )

    def _iter_live(self, live_mask) -> Iterator[Tuple[int, _MoleculeState]]:
        if live_mask is None:
            yield from self._states.items()
            return
        mask = np.asarray(_to_np(live_mask), dtype=bool)
        yield from ((mol, state) for mol, state in self._states.items() if mask[mol])

    @classmethod
    def _chunk_shape(cls, shape: Tuple[int, ...], dtype) -> Tuple[int, ...]:
        frame_items = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
        frame_bytes = max(1, frame_items * np.dtype(dtype).itemsize)
        rows = max(1, min(shape[0], cls._CHUNK_TARGET_BYTES // frame_bytes))
        return (rows,) + shape[1:]

    @classmethod
    def _create_dataset(
        cls, group: h5py.Group, path: str, shape: Tuple[int, ...], dtype=np.float64, *, compress: bool = False
    ) -> h5py.Dataset:
        kwargs: Dict[str, Any] = {"shape": shape, "dtype": dtype}
        if compress:
            kwargs.update(
                chunks=cls._chunk_shape(shape, dtype), compression="gzip", compression_opts=1, shuffle=True
            )
        return group.create_dataset(path, **kwargs)

    @staticmethod
    def _require_group(parent: h5py.Group, path: str) -> h5py.Group:
        obj = parent.get(path)
        if not isinstance(obj, h5py.Group):
            raise RuntimeError(f"Required HDF5 group is missing: {path}")
        return obj

    @staticmethod
    def _require_dataset(parent: h5py.Group, path: str, shape: Tuple[int, ...], dtype=None) -> h5py.Dataset:
        obj = parent.get(path)
        if not isinstance(obj, h5py.Dataset):
            raise RuntimeError(f"Required HDF5 dataset is missing: {path}")
        if obj.shape != shape:
            raise RuntimeError(f"Shape mismatch for {obj.name}: found {obj.shape}, expected {shape}.")
        if dtype is not None and obj.dtype != np.dtype(dtype):
            raise RuntimeError(
                f"Dtype mismatch for {obj.name}: found {obj.dtype}, expected {np.dtype(dtype)}."
            )
        return obj

    @classmethod
    def _create_fields(cls, group: h5py.Group, fields: Dict[str, Tuple[Tuple[int, ...], Any]]) -> None:
        for path, (shape, dtype) in fields.items():
            cls._create_dataset(group, path, shape, dtype)

    def _config_signature(self, *, nat: int, norb: int, restricted: bool) -> str:
        config = {
            "schema": self._SCHEMA_VERSION,
            "timestep_fs": self.timestep,
            "n_atoms": nat,
            "n_orbitals": norb,
            "n_excited_states": self._nstates,
            "restricted": restricted,
            "include_initial": self._include_initial,
            "total_steps": self._steps,
            "data_stride": self._data_every,
            "vector_strides": self._cadence,
            "tdm_stride": self._write_tdm,
            "tdm_mode": self._tdm_mode,
            "nonadiabatic_stride": self._write_nonadiabatic,
            "write_mo": self._write_mo,
            "write_transition_properties": self._write_transition_properties,
        }
        return json.dumps(config, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _load_count(group: h5py.Group) -> int:
        if "n_written" not in group.attrs:
            raise RuntimeError(f"Missing n_written on {group.name}.")
        count = int(group.attrs["n_written"])
        capacity = int(group["steps"].shape[0])
        if not 0 <= count <= capacity:
            raise RuntimeError(f"Invalid n_written={count} on {group.name}; capacity={capacity}.")
        return count

    @classmethod
    def _new_track(
        cls,
        group: h5py.Group,
        length: int,
        stride: int,
        values_shape: Optional[Tuple[int, ...]] = None,
        *,
        compress: bool = False,
    ) -> _H5Track:
        group.attrs["stride"] = np.int64(stride)
        group.attrs["n_written"] = np.int64(0)
        steps = cls._create_dataset(group, "steps", (length,), np.int64)
        values = (
            None
            if values_shape is None
            else cls._create_dataset(group, "values", values_shape, compress=compress)
        )
        return _H5Track(group, steps, values=values)

    @classmethod
    def _load_track(
        cls, group: h5py.Group, length: int, stride: int, values_shape: Optional[Tuple[int, ...]] = None
    ) -> _H5Track:
        if int(group.attrs.get("stride", -1)) != stride:
            raise RuntimeError(f"Stride mismatch for {group.name}.")
        steps = cls._require_dataset(group, "steps", (length,), np.int64)
        values = (
            None if values_shape is None else cls._require_dataset(group, "values", values_shape, np.float64)
        )
        return _H5Track(group, steps, cls._load_count(group), values)

    @staticmethod
    def _start_row(track: _H5Track, step: int, *, mol: int) -> int:
        i = track.count
        if i >= track.capacity:
            raise RuntimeError(f"HDF5 {track.group.name} capacity exceeded for molecule {mol}.")
        if i and step <= int(track.steps[i - 1]):
            raise RuntimeError(
                f"Non-increasing step for {track.group.name}: previous={int(track.steps[i - 1])}, new={step}."
            )
        track.steps[i] = np.int64(step)
        return i

    @classmethod
    def _append_series(cls, track: _H5Track, step: int, value, *, mol: int) -> None:
        i = cls._start_row(track, step, mol=mol)
        track.values[i] = value
        track.count = i + 1

    def open(
        self,
        molecule,
        prefix: str,
        steps: int,
        excited_states: int = 0,
        resume: bool = False,
        step_offset: int = 0,
        include_initial: bool = False,
    ) -> None:
        if self._states:
            raise RuntimeError("HDF5Writer is already open.")
        if steps < 0:
            raise ValueError("steps must be non-negative.")

        self._steps = int(steps)
        self._nstates = max(0, int(excited_states))
        self._include_initial = bool(include_initial)
        if self._nstates == 0 and (
            self._write_tdm > 0 or self._write_transition_properties or self._write_nonadiabatic > 0
        ):
            raise ValueError("Excited-state output requires at least one excited state.")

        restricted = not bool(self.seqm_parameters.get("UHF", False))
        layout = None
        active_states = None
        if not resume:
            layout = self._layout()
            active_states = active_state_tensor(
                molecule.active_state, int(molecule.nmol), molecule.coordinates.device
            )

        try:
            for mol in self.config.molid:
                nat = int(torch.sum(molecule.species[mol] > 0))
                norb = int(molecule.norb[mol])
                path = f"{prefix}.{mol}.h5"
                if resume:
                    state = self._open_resume(
                        path, mol, molecule, nat, norb, restricted, step_offset=step_offset
                    )
                else:
                    state = self._create_new(
                        path, mol, molecule, nat, norb, restricted, layout, active_states
                    )
                self._states[mol] = state
        except BaseException:
            self._close_all()
            raise

    def _open_resume(
        self, path: str, mol: int, molecule, nat: int, norb: int, restricted: bool, *, step_offset: int
    ) -> _MoleculeState:
        h5 = h5py.File(path, "r+")
        state = _MoleculeState(h5, nat, norb, restricted)
        try:
            actual = h5.attrs.get("writer_config")
            if isinstance(actual, bytes):
                actual = actual.decode()
            try:
                stored = json.loads(actual)
                self._include_initial = bool(stored["include_initial"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Invalid writer configuration in {path}.") from exc

            expected = self._config_signature(nat=nat, norb=norb, restricted=restricted)
            if actual != expected:
                raise RuntimeError(f"Writer configuration does not match {path}.")

            layout = self._layout()

            atoms = self._require_dataset(h5, "atoms", (nat,))
            if not np.array_equal(atoms[...], _to_np(molecule.species[mol, :nat])):
                raise RuntimeError(f"Atom identities do not match in {path}.")

            data_len = layout.data
            if ("data/steps" in h5) != bool(data_len):
                raise RuntimeError(f"Data-output configuration does not match {path}.")
            if data_len:
                gd = self._require_group(h5, "data")
                state.data = self._load_track(gd, data_len, self._data_every)

            for name, stride in self._cadence.items():
                length = layout.vectors[name]
                if (name in h5) != bool(length):
                    raise RuntimeError(f"Vector configuration for {name!r} does not match {path}.")
                if length:
                    group = self._require_group(h5, name)
                    state.series[name] = self._load_track(group, length, stride, (length, nat, 3))

            tdm_len = layout.tdm
            tdm_path = "data/excitation/transition_density_matrices"
            has_tdm = bool(tdm_len and self._nstates)
            if (tdm_path in h5) != has_tdm:
                raise RuntimeError(f"TDM configuration does not match {path}.")
            if has_tdm:
                group = self._require_group(h5, tdm_path)
                state.series["tdm"] = self._load_track(
                    group, tdm_len, self._write_tdm, self._tdm_shape(tdm_len, norb)
                )

            na_len = layout.nonadiabatic
            na_path = "data/nonadiabatic"
            has_na = bool(na_len and self._nstates)
            if (na_path in h5) != has_na:
                raise RuntimeError(f"Nonadiabatic configuration does not match {path}.")
            if has_na:
                group = self._require_group(h5, na_path)
                state.nonadiabatic = self._load_track(group, na_len, self._write_nonadiabatic)

            for track in state.tracks():
                if track.count and int(track.steps[track.count - 1]) > step_offset:
                    raise RuntimeError(
                        f"{track.group.name} contains data beyond checkpoint step {step_offset}."
                    )
            return state
        except BaseException:
            h5.close()
            raise

    def _create_new(
        self,
        path: str,
        mol: int,
        molecule,
        nat: int,
        norb: int,
        restricted: bool,
        layout: _H5Layout,
        active_states,
    ) -> _MoleculeState:
        _rotate_existing(path)
        h5 = h5py.File(path, "w")
        state = _MoleculeState(h5, nat, norb, restricted)
        try:
            h5.attrs["writer_config"] = self._config_signature(nat=nat, norb=norb, restricted=restricted)
            h5.create_dataset("atoms", data=_to_np(molecule.species[mol, :nat]))
            gd = h5.create_group("data") if any((layout.data, layout.tdm, layout.nonadiabatic)) else None

            data_len = layout.data
            if data_len:
                state.data = self._new_track(gd, data_len, self._data_every)
                self._create_fields(gd, self._data_fields(data_len, restricted))
                if self._nstates:
                    gd.create_dataset("excitation/active_state", data=int(active_states[mol].item()))
                if self._write_mo:
                    nocc = int(molecule.nocc[mol].item()) if restricted else _to_np(molecule.nocc[mol])
                    gd.create_dataset("mo/nocc", data=nocc)

            for name, length in layout.vectors.items():
                if length:
                    group = h5.create_group(name)
                    state.series[name] = self._new_track(
                        group, length, self._cadence[name], (length, nat, 3), compress=True
                    )

            tdm_len = layout.tdm
            if tdm_len and self._nstates:
                group = gd.require_group("excitation").create_group("transition_density_matrices")
                state.series["tdm"] = self._new_track(
                    group, tdm_len, self._write_tdm, self._tdm_shape(tdm_len, norb), compress=True
                )

            na_len = layout.nonadiabatic
            if na_len and self._nstates:
                group = gd.create_group("nonadiabatic")
                state.nonadiabatic = self._new_track(group, na_len, self._write_nonadiabatic)
                self._create_fields(group, self._nonadiabatic_fields(na_len))
            return state
        except BaseException:
            h5.close()
            raise

    def _data_fields(self, length: int, restricted: bool):
        fields = {
            "thermo/T": ((length,), np.float64),
            "thermo/Ek": ((length,), np.float64),
            "thermo/Ep": ((length,), np.float64),
            "properties/ground_dipole": ((length, 3), np.float64),
        }
        if self._nstates:
            fields["excitation/state_energies"] = ((length, self._nstates + 1), np.float64)
            if self._write_transition_properties:
                fields["excitation/transition_dipole"] = ((length, self._nstates, 3), np.float64)
                fields["excitation/oscillator_strength"] = ((length, self._nstates), np.float64)
        if self._write_mo:
            fields["mo/homo_lumo_gap"] = ((length, 1 if restricted else 2), np.float64)
        return fields

    def _nonadiabatic_fields(self, length: int):
        return {
            "active_surface": ((length,), np.int64),
            "electronic_amplitudes": ((length, self._nstates, 2), np.float64),
            "NACT": ((length, self._nstates, self._nstates), np.float64),
        }

    def _tdm_shape(self, length: int, norb: int) -> Tuple[int, ...]:
        shape = (length, self._nstates, norb)
        return shape + (norb,) if self._tdm_mode == "full" else shape

    def append_data(self, step: int, molecule, T, Ek, Ep, e_gap) -> None:
        T_np, Ek_np, Ep_np = map(_to_np, (T, Ek, Ep))
        dipole_np = _to_np(molecule.dipole)
        active_np = (
            _to_np(molecule.active_state) if torch.is_tensor(molecule.active_state) else molecule.active_state
        )
        etot_np = _to_np(molecule.Etot) if self._nstates else None
        cis_np = _to_np(molecule.cis_energies[:, : self._nstates]) if self._nstates else None
        td_np = (
            _to_np(molecule.transition_dipole[:, : self._nstates])
            if self._write_transition_properties
            else None
        )
        osc_np = (
            _to_np(molecule.oscillator_strength[:, : self._nstates])
            if self._write_transition_properties
            else None
        )
        gap_np = _to_np(e_gap) if self._write_mo else None

        for mol, state in self._iter_live(getattr(molecule, "_trajectory_live_mask", None)):
            track = state.data
            i = self._start_row(track, step, mol=mol)
            gd = track.group
            gd["thermo/T"][i], gd["thermo/Ek"][i], gd["thermo/Ep"][i] = T_np[mol], Ek_np[mol], Ep_np[mol]
            gd["properties/ground_dipole"][i] = dipole_np[mol]

            if self._nstates:
                active = int(active_np[mol]) if np.ndim(active_np) else int(active_np)
                e0 = etot_np[mol] - (cis_np[mol, active - 1] if active > 0 else 0.0)
                gd["excitation/state_energies"][i] = np.r_[e0, e0 + cis_np[mol]]
                if self._write_transition_properties:
                    gd["excitation/transition_dipole"][i] = td_np[mol]
                    gd["excitation/oscillator_strength"][i] = osc_np[mol]

            if self._write_mo:
                gd["mo/homo_lumo_gap"][i] = gap_np[mol, None] if state.restricted else gap_np[mol]
            track.count = i + 1

    def append_tdm(self, step: int, molecule) -> None:
        for mol, state in self._iter_live(getattr(molecule, "_trajectory_live_mask", None)):
            tdm = molecule.transition_density_matrices[mol, : self._nstates]
            if self._tdm_mode == "diag":
                tdm = (
                    torch.diagonal(tdm[:, : state.norb, : state.norb], dim1=-2, dim2=-1)
                    if tdm.dim() == 3
                    else tdm[:, : state.norb]
                )
            else:
                tdm = tdm[:, : state.norb, : state.norb]
            self._append_series(state.series["tdm"], step, _to_np(tdm), mol=mol)

    def append_vectors(self, step: int, molecule) -> None:
        names = [name for name, stride in self._cadence.items() if self._due(step, stride)]
        if not names:
            return
        arrays = {name: _to_np(getattr(molecule, "force" if name == "forces" else name)) for name in names}
        for mol, state in self._iter_live(getattr(molecule, "_trajectory_live_mask", None)):
            for name in names:
                self._append_series(state.series[name], step, arrays[name][mol, : state.nat], mol=mol)

    @staticmethod
    def _amplitudes_array(amplitudes):
        if amplitudes is None:
            return None
        array = np.asarray(_to_np(amplitudes) if torch.is_tensor(amplitudes) else amplitudes)
        if np.iscomplexobj(array):
            return np.stack((array.real, array.imag), axis=-1)
        return (
            array if array.ndim and array.shape[-1] == 2 else np.stack((array, np.zeros_like(array)), axis=-1)
        )

    def append_nonadiabatic(self, step: int, active_states, amplitudes, nac_dot, live_mask=None) -> None:
        active_np = None if active_states is None else np.asarray(_to_np(active_states))
        amp_np = self._amplitudes_array(amplitudes)
        nac_np = None if nac_dot is None else np.asarray(_to_np(nac_dot))

        for mol, state in self._iter_live(live_mask):
            track = state.nonadiabatic
            i = self._start_row(track, step, mol=mol)
            group = track.group
            group["active_surface"][i] = (
                -1 if active_np is None else int(active_np[mol] if active_np.ndim else active_np)
            )
            group["electronic_amplitudes"][i] = (
                np.full((self._nstates, 2), np.nan) if amp_np is None else amp_np[mol]
            )
            group["NACT"][i] = np.nan if nac_np is None else nac_np[mol, : self._nstates, : self._nstates]
            track.count = i + 1

    def flush(self) -> None:
        """Persist values first, then publish committed row counts."""
        for state in self._states.values():
            state.h5.flush()
        for state in self._states.values():
            for track in state.tracks():
                track.commit()
            state.h5.flush()

    def _close_all(self):
        error = None
        for state in self._states.values():
            try:
                state.h5.close()
            except BaseException as exc:
                error = error or exc
        self._states.clear()
        return error

    def close(self) -> None:
        error = None
        try:
            self.flush()
        except BaseException as exc:
            error = exc
        error = error or self._close_all()
        if error is not None:
            raise error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            self._close_all()
        return False


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
            self.files[mol] = open(xyz_fn, "a+", buffering=bufsize)

    def write(self, step: int, molecule, Ek, L):
        """Write XYZ frame for current step."""
        Et = (Ek + L).detach().cpu()

        for mol in self.config.molid:
            species = molecule.species[mol].detach().cpu().tolist()
            xyz = molecule.coordinates[mol].detach().cpu().numpy()
            n_atoms = sum(z > 0 for z in species)
            s = StringIO()
            s.write(f"{n_atoms}\n")
            s.write(f"step: {step + 1}  E_total = {float(Et[mol]):12.9f}  \n")

            for a in range(n_atoms):
                label = molecule.const.label[species[a]]
                x, y, z = xyz[a]
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
    !!! This class uses steepest descent algorithm for geometry optimization, which is not recommended for production use due to its inefficiency.
    It is provided here for demonstration and testing purposes.
    For practical geometry optimization with PYSEQM use the geomeTRIC optimizer instead.
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

    def onestep(self, molecule, learned_parameters=None):
        learned_parameters = {} if learned_parameters is None else learned_parameters
        self.esdriver(molecule, learned_parameters=learned_parameters, P0=molecule.dm, dm_prop="SCF")
        force = molecule.force
        with torch.no_grad():
            molecule.coordinates.add_(self.alpha * force)
        return force, molecule.Etot

    def run(self, molecule, learned_parameters=None, log=True):
        learned_parameters = {} if learned_parameters is None else learned_parameters
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

            if force_err > self.force_tol:
                Lold = Lnew
                continue
            else:
                break
        if i == (self.max_evl - 1):
            print("not converged within %d step" % self.max_evl)
        else:
            if log:
                print(
                    "converged with %d step, Max Force = %e (eV/Ang), dE = %e (eV)"
                    % (i + 1, force_err.item(), energy_err.item())
                )

        return force_err, energy_err


class Molecular_Dynamics_Basic(torch.nn.Module):
    """Base class for molecular dynamics simulations."""

    def __init__(
        self,
        seqm_parameters,
        timestep=1.0,
        Temp=0.0,
        step_offset=0,
        output=None,
        torch_compile=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.seqm_parameters = seqm_parameters
        self._torch_compile_config = normalize_torch_compile_config(seqm_parameters, torch_compile)
        self._torch_compile_applied = False
        self.timestep = timestep
        self.output_config = OutputConfig.from_dict(output)
        self._sync_excited_state_output_flags()
        self.esdriver = esdriver(self.seqm_parameters)
        self.Temp = Temp
        self.step_offset = step_offset

        self.n_dof = None
        self.remove_com_angular = False
        self.do_remove_com = False
        self.remove_com_stride = 0
        self.start_time = None

        self._h5_writer: Optional[HDF5Writer] = None
        self._xyz_writer: Optional[XYZWriter] = None
        self._do_screen = False
        self._do_xyz = False
        self._do_h5 = False

    def _sync_excited_state_output_flags(self):
        exc = self.seqm_parameters.get("excited_states")
        if not isinstance(exc, dict):
            return
        h5 = self.output_config.h5_config if isinstance(self.output_config.h5_config, dict) else {}
        if int(h5.get("transition_density_matrices", 0)) > 0:
            exc["save_tdm_output"] = True
            exc["transition_density_matrices_mode"] = self.output_config.get_h5_tdm_mode()
        if int(h5.get("data", 0)) > 0 and bool(h5.get("transition_properties", False)):
            exc["compute_transition_properties"] = True

    def _validate_h5_output_config(self):
        h5 = self.output_config.h5_config if isinstance(self.output_config.h5_config, dict) else {}
        data_every = int(h5.get("data", 0))
        if (
            int(h5.get("transition_density_matrices", 0)) > 0 or bool(h5.get("transition_properties", False))
        ) and not isinstance(self.seqm_parameters.get("excited_states"), dict):
            raise ValueError("Excited-state HDF5 output requires excited_states.")
        if data_every > 0:
            return
        if bool(h5.get("transition_properties", False)):
            raise ValueError(
                "output.h5.transition_properties requires output.h5.data > 0 "
                "(transition dipoles/oscillator strengths are written through append_data cadence)."
            )
        if bool(h5.get("write_mo", False)):
            raise ValueError(
                "output.h5.write_mo requires output.h5.data > 0 "
                "(MO gaps are written through append_data cadence)."
            )

    @property
    def output(self):
        """Backward compatibility property."""
        return {
            "molid": self.output_config.molid,
            "prefix": self.output_config.prefix,
            "print every": self.output_config.print_every,
            "checkpoint every": self.output_config.checkpoint_every,
            "xyz": self.output_config.xyz_every,
            "h5": self.output_config.h5_config,
        }

    def initialize_velocity(self, molecule, vel_com=True):
        """Initialize velocities from Maxwell-Boltzmann distribution."""
        if self.n_dof is None:
            raise RuntimeError("n_dof not set. Call initialize() first")

        if torch.is_tensor(molecule.velocities):
            if vel_com:
                self._zero_com(molecule, translate_to_origin=False, restore_kinetic_energy=False)
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

    def _zero_com(
        self, molecule, remove_angular=True, translate_to_origin=False, restore_kinetic_energy=True
    ):
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
                I = torch.sum(
                    mass * (r_rel * r_rel).sum(dim=2, keepdim=True), dim=1, keepdim=True
                ) * eye.reshape(1, 3, 3) - torch.sum(
                    mass.unsqueeze(3) * r_rel.unsqueeze(3) * r_rel.unsqueeze(2), dim=1
                )
                omega = (torch.linalg.pinv(I, hermitian=True, atol=1e-10) @ L.unsqueeze(2)).squeeze(-1)
                molecule.velocities.sub_(
                    torch.linalg.cross(omega.unsqueeze(1).expand_as(r_rel), r_rel, dim=2)
                )

            # Restore kinetic energy
            Ek_after = self._kinetic_energy(molecule)
            if torch.any(Ek_after < 1e-12):
                raise RuntimeError("Zero kinetic energy after removing COM momentum")
            if restore_kinetic_energy:
                alpha = torch.sqrt(Ek_initial / Ek_after)
                molecule.velocities.mul_(alpha.reshape(-1, 1, 1))

    def _kinetic_energy(self, molecule):
        """Calculate kinetic energy."""
        return (
            torch.sum(0.5 * molecule.mass * molecule.velocities**2, dim=(1, 2))
            * CONSTANTS.KINETIC_ENERGY_SCALE
        )

    def _calc_temperature(self, kinetic_energy):
        """Calculate temperature from kinetic energy."""
        return kinetic_energy * CONSTANTS.TEMPERATURE_SCALE / (0.5 * self.n_dof)

    def _output_to_screen(self, step: int, T, Ek, V):
        """Print MD data to screen."""
        if step == 0:
            print("Step,    Temp,    E(kinetic),  E(potential),  E(total)")
        print(f"{step + 1:6d}", end="")
        for mol in self.output_config.molid:
            Tm = float(T[mol].detach().cpu())
            Ekm = float(Ek[mol].detach().cpu())
            Vm = float(V[mol].detach().cpu())
            print(f" {Tm:8.2f}   {Ekm:e} {Vm:e} {Vm + Ekm:e} || ", end="")
        print()

    def set_dof(self, molecule, constraints=0.0):
        """Set degrees of freedom."""
        self.n_dof = 3.0 * molecule.num_atoms - constraints

    def _enable_torch_compile_if_requested(self, molecule):
        """Enable optional compilation for repeated dynamics force evaluations."""
        cfg = self._torch_compile_config
        if self._torch_compile_applied or not cfg["enabled"]:
            return
        self._torch_compile_applied = True

        # XL-BOMD has a separate density propagation path; leave it eager for now.
        if getattr(self, "k", None) is not None:
            return

        method = str(self.seqm_parameters.get("method", "")).upper()
        kernel_options = dict(cfg["options"])
        kernel_mode = kernel_options.pop("mode", None)
        excited = bool(self.seqm_parameters.get("excited_states"))

        if excited:
            enable_rcis_compile(mode=kernel_mode, **kernel_options)
        elif method not in OMX_METHODS and int(molecule.nmol) == 1:
            enable_two_center_compile(mode=kernel_mode, **kernel_options)
            enable_fock_compile(mode=kernel_mode, **kernel_options)

    def _mark_torch_compile_step(self, molecule):
        mark = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
        if self._torch_compile_config["enabled"] and molecule.coordinates.is_cuda and mark:
            mark()

    def _thermo_potential(self, molecule):
        """Potential energy for thermodynamics (override in subclasses)."""
        return molecule.Etot

    def one_step(self, molecule, learned_parameters=None, *args, **kwargs):
        """Perform one velocity Verlet integration step."""
        learned_parameters = {} if learned_parameters is None else learned_parameters
        dt = self.timestep
        if molecule.const.do_timing:
            t0 = time.time()

        with torch.no_grad():
            molecule.velocities.add_(0.5 * molecule.acc * dt)
            molecule.coordinates.add_(molecule.velocities * dt)

        self.esdriver(
            molecule,
            learned_parameters=learned_parameters,
            P0=molecule.dm,
            dm_prop="SCF",
            cis_amp=molecule.cis_amplitudes,
            *args,
            **kwargs,
        )

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

    def initialize(
        self, molecule, remove_com=None, learned_parameters=None, steps: Optional[int] = None, *args, **kwargs
    ):
        """Initialize MD simulation."""
        learned_parameters = {} if learned_parameters is None else learned_parameters
        self._validate_h5_output_config()
        molecule.verbose = False  # Dont print SCF and CIS/RPA results
        self.esdriver.conservative_force.energy.md = True

        # Check device compatibility once at initialization.
        md_dev = getattr(self.esdriver, "device", None)
        if md_dev is None:
            md_dev = next(self.esdriver.parameters()).device
        mol_dev = molecule.coordinates.device
        if md_dev != mol_dev:
            raise RuntimeError(f"MD object on {md_dev}, molecule on {mol_dev}")

        self.do_remove_com = remove_com is not None
        constraints = 0.0
        # remove_com is a tuple of (mode,stride), where mode='linear' or 'angular'
        # and stride is the number of steps after which com motion is removed
        if self.do_remove_com:
            mode, self.remove_com_stride = remove_com
            mode = str(mode).lower().strip()
            if mode not in ("linear", "angular"):
                raise ValueError(
                    f"Invalid COM motion removal mode '{mode}'. "
                    "Expected 'linear' or 'angular'. "
                    "Usage: remove_com=('linear', N) or ('angular', N)."
                )
            self.remove_com_angular = mode == "angular"
            constraints = 6.0 if self.remove_com_angular else 3.0  # TODO: check if the molecule is linear

        self.set_dof(molecule, constraints)
        if self.step_offset == 0 or not torch.is_tensor(molecule.velocities):
            self.initialize_velocity(molecule)

        # Calculate accelearation at t=0
        if not torch.is_tensor(molecule.force):
            self.esdriver(
                molecule,
                learned_parameters=learned_parameters,
                P0=molecule.dm,
                cis_amp=molecule.cis_amplitudes,
                *args,
                **kwargs,
            )

        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE
        self._enable_torch_compile_if_requested(molecule)

        # Setup output
        has_molid = len(self.output_config.molid) > 0
        self._do_screen = self.output_config.print_every > 0 and has_molid
        self._do_xyz = self.output_config.xyz_every > 0 and has_molid
        self._do_h5 = (
            self.output_config.get_h5_data_every() > 0
            or any(self.output_config.get_h5_cadence().values())
            or self.output_config.get_h5_write_tdm() > 0
            or self.output_config.get_h5_write_nonadiabatic() > 0
        ) and has_molid
        h5_data_every = self.output_config.get_h5_data_every()
        h5_tdm_every = self.output_config.get_h5_write_tdm()
        h5_vectors_every = self.output_config.h5_vectors_every

        if steps is None:
            return

        if self._do_h5:
            excited_states_params = self.seqm_parameters.get("excited_states")
            n_roots = getattr(
                self, "_nstates", excited_states_params["n_states"] if excited_states_params else 0
            )
            # n_roots = excited_states_params["n_states"] if excited_states_params else 0
            # # With XL-ESMD, only the active state is computed
            # if isinstance(self, XL_ESMD):
            #     n_roots = 1
            self._h5_writer = HDF5Writer(self.output_config, self.seqm_parameters, self.timestep)
            self._h5_writer.open(
                molecule,
                self.output_config.prefix,
                steps,
                n_roots,
                resume=(self.step_offset > 0),
                step_offset=self.step_offset,
                include_initial=(self.step_offset == 0),
            )

        if self._do_xyz:
            self._xyz_writer = XYZWriter(self.output_config, self.step_offset)
            self._xyz_writer.open()

        if self.step_offset == 0:
            with torch.no_grad():
                Ek0 = self._kinetic_energy(molecule)
                T0 = self._calc_temperature(Ek0)
                V0 = self._thermo_potential(molecule)

                if self._do_h5:
                    if h5_data_every > 0:
                        self._h5_writer.append_data(0, molecule, T0, Ek0, V0, molecule.e_gap)
                    if h5_tdm_every > 0:
                        self._h5_writer.append_tdm(0, molecule)
                    if h5_vectors_every:
                        self._h5_writer.append_vectors(0, molecule)

                if self._do_xyz:
                    # Write an initial snapshot labeled as step 0.
                    self._xyz_writer.write(-1, molecule, Ek0, V0)

    def run(
        self,
        molecule,
        steps,
        learned_parameters=None,
        reuse_P=True,
        remove_com=None,
        seed=None,
        *args,
        **kwargs,
    ):
        """Run molecular dynamics simulation."""
        learned_parameters = {} if learned_parameters is None else learned_parameters
        self.start_time = datetime.now()
        print(f"MD run began at {self.start_time}", flush=True)

        if seed is not None:
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))

        if getattr(self, "k", None) is not None:
            reuse_P = True

        self.initialize(
            molecule,
            remove_com=remove_com,
            learned_parameters=learned_parameters,
            steps=steps,
            *args,
            **kwargs,
        )

        if not reuse_P:
            molecule.dm = None
            molecule.cis_amplitudes = None

        # Velocity scaling / energy shift
        do_scale_vel = "scale_vel" in kwargs
        if do_scale_vel:
            scale_freq, T_target = kwargs["scale_vel"]
            scale_freq = int(scale_freq)
            T_target = torch.as_tensor(
                T_target, dtype=molecule.coordinates.dtype, device=molecule.coordinates.device
            )

        do_energy_shift = bool(kwargs.get("control_energy_shift", False))
        if do_scale_vel and do_energy_shift:
            raise ValueError("Cannot scale velocities and fix energy shift simultaneously.")

        E0 = None
        checkpoint_every = self.output_config.checkpoint_every
        checkpoint_path = f"{self.output_config.prefix}.restart.pt"

        do_screen = self._do_screen
        do_xyz = self._do_xyz
        do_h5 = self._do_h5
        h5_data_every = self.output_config.get_h5_data_every()
        h5_tdm_every = self.output_config.get_h5_write_tdm()
        h5_vectors_every = self.output_config.h5_vectors_every
        print_every = self.output_config.print_every
        xyz_every = self.output_config.xyz_every

        try:
            for i in range(self.step_offset, steps):
                self._mark_torch_compile_step(molecule)
                self._do_integrator_step(i, molecule, learned_parameters, *args, **kwargs)
                if getattr(self, "_terminate_run", False):
                    break

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
                        # scale velocities to adjust kinetic energy and compenstate the energy shift
                        Eshift = Ek + V - E0
                        alpha = torch.sqrt((Ek - Eshift) / Ek)
                        alpha[~torch.isfinite(alpha)] = 0.0
                        molecule.velocities.mul_(alpha.reshape(-1, 1, 1))
                        Ek = self._kinetic_energy(molecule)
                        T = self._calc_temperature(Ek)

                    if do_screen and ((i + 1) % print_every == 0):
                        self._output_to_screen(i, T, Ek, V)

                    if do_h5:
                        if h5_data_every > 0 and (i + 1) % h5_data_every == 0:
                            self._h5_writer.append_data(i + 1, molecule, T, Ek, V, molecule.e_gap)
                        if h5_tdm_every > 0 and (i + 1) % h5_tdm_every == 0:
                            self._h5_writer.append_tdm(i + 1, molecule)
                        if h5_vectors_every and (i + 1) % h5_vectors_every == 0:
                            self._h5_writer.append_vectors(i + 1, molecule)

                    if do_xyz and ((i + 1) % xyz_every == 0):
                        self._xyz_writer.write(i, molecule, Ek, V)

                    if checkpoint_every > 0 and ((i + 1) % checkpoint_every == 0):
                        self._flush_all()
                        self.save_checkpoint(
                            molecule, steps, reuse_P, remove_com, step_done=i + 1, path=checkpoint_path
                        )

                del Ek, T
                if i % 1000 == 0:
                    # Do not remove this. Clearing CUDA cache prevents out of memory issues for long trajectories
                    torch.cuda.empty_cache()

        finally:
            if self._xyz_writer:
                self._xyz_writer.close()
            if self._h5_writer:
                self._h5_writer.close()

        now = datetime.now()
        print(f"MD run ended at {now}")
        print(f"Time elapsed since the beginning of MD run: {now - self.start_time}", flush=True)
        if self.__class__.__name__ == "SurfaceHoppingDynamics" and callable(
            getattr(self, "_print_hop_log", None)
        ):
            self._print_hop_log()
        if callable(getattr(self, "_print_termination_log", None)):
            self._print_termination_log()
        return molecule.coordinates, molecule.velocities, molecule.acc

    def _flush_all(self):
        """Flush all output buffers."""
        if self._h5_writer:
            self._h5_writer.flush()
        if self._xyz_writer:
            self._xyz_writer.flush()

    def save_checkpoint(self, molecule, steps: int, reuse_P, remove_com, *, step_done: int, path: str):
        """Save checkpoint for restart."""
        ckpt = self._build_checkpoint_base(
            molecule, steps, reuse_P, remove_com, step_done=step_done, include_forces=True
        )
        ckpt.update(
            {"MD_type": self.__class__.__name__, "xl_bomd_params": getattr(self, "xl_bomd_params", None)}
        )

        if hasattr(self, "m"):  # XL-BOMD variants
            ckpt["xl_ctx"] = {
                "Pt": self._tensor_cpu(self._xl_ctx["Pt"]),
                "es_amp_t": self._tensor_cpu(self._xl_ctx.get("es_amp_t")),
            }
            if isinstance(molecule.dP2dt2, torch.Tensor):
                ckpt["dP2dt2"] = self._tensor_cpu(molecule.dP2dt2)

        self._save_checkpoint_and_report(ckpt, path)

    def _save_checkpoint_and_report(self, ckpt: Dict, path: str):
        self._atomic_save_checkpoint(ckpt, path)
        now = datetime.now()
        print(f"Saved checkpoint at {now}")
        print(f"Time elapsed since the beginning of MD run: {now - self.start_time}", flush=True)

    @staticmethod
    def run_from_checkpoint(path: str, device=None):
        """Load and resume from checkpoint."""
        ckpt, molecule, device, reuse_P = Molecular_Dynamics_Basic._load_checkpoint_base(path, device=device)
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
        kwargs = Molecular_Dynamics_Basic._checkpoint_init_kwargs(ckpt)

        if md_type in ("Molecular_Dynamics_Langevin", "XL_BOMD", "KSA_XL_BOMD"):
            kwargs["damp"] = ckpt["damp"]
        if md_type in ("XL_BOMD", "KSA_XL_BOMD"):
            kwargs["xl_bomd_params"] = ckpt["xl_bomd_params"]

        md = md_cls(**kwargs).to(device)

        if md_type in ("XL_BOMD", "KSA_XL_BOMD"):
            xl = ckpt["xl_ctx"]
            Pt = xl["Pt"].to(device)
            es_amp_t = xl.get("es_amp_t")
            xl_m = ckpt["xl_bomd_params"]["k"] + 1
            cindx = (ckpt["step_done"] - 1) % xl_m  # subtract one because step_done is advanced by one step
            P = Pt[(xl_m - 1 - cindx)].clone()
            es_amp = None
            if isinstance(es_amp_t, torch.Tensor):
                es_amp_t = es_amp_t.to(device)
                es_amp = es_amp_t[(xl_m - 1 - cindx)].clone()
            md._xl_ctx = {"P": P, "Pt": Pt, "es_amp": es_amp, "es_amp_t": es_amp_t}

        Molecular_Dynamics_Basic._restore_rng(ckpt)
        md.run(molecule=molecule, steps=ckpt["steps"], reuse_P=reuse_P, remove_com=ckpt["remove_com"])

    @staticmethod
    def _tensor_cpu(x):
        return x.detach().cpu() if torch.is_tensor(x) else x

    @staticmethod
    def _checkpoint_init_kwargs(ckpt: Dict) -> Dict:
        return {
            "seqm_parameters": ckpt["seqm_parameters"],
            "timestep": ckpt["timestep"],
            "Temp": ckpt["Temp"],
            "output": ckpt["output"],
            "step_offset": ckpt["step_done"],
        }

    def _build_checkpoint_base(
        self, molecule, steps: int, reuse_P: bool, remove_com, *, step_done: int, include_forces: bool
    ):
        molecules = {
            "species": self._tensor_cpu(molecule.species),
            "coordinates": self._tensor_cpu(molecule.coordinates),
            "velocities": self._tensor_cpu(molecule.velocities),
            "Etot": self._tensor_cpu(getattr(molecule, "Etot", None)),
            "dm": self._tensor_cpu(molecule.dm) if reuse_P else None,
            "cis_amplitudes": self._tensor_cpu(molecule.cis_amplitudes) if reuse_P else None,
            "transition_density_matrices": self._tensor_cpu(
                getattr(molecule, "transition_density_matrices", None)
            ),
            "molecular_orbitals": self._tensor_cpu(getattr(molecule, "molecular_orbitals", None)),
            "constants": molecule.const,
            "old_mos": self._tensor_cpu(molecule.old_mos),
        }
        if include_forces:
            molecules["forces"] = self._tensor_cpu(molecule.force)

        return {
            "device": molecule.coordinates.device,
            "step_done": int(step_done),
            "steps": int(steps),
            "reuse_P": bool(reuse_P),
            "timestep": float(self.timestep),
            "Temp": float(self.Temp),
            "damp": getattr(self, "damp", None),
            "seqm_parameters": self.seqm_parameters,
            "remove_com": remove_com,
            "output": self.output,
            "rng": {
                "torch_cpu": torch.random.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "molecules": molecules,
        }

    @staticmethod
    def _restore_molecule_from_ckpt(mol_ckpt, molecule, reuse_P: bool, device):
        with torch.no_grad():
            molecule.velocities = mol_ckpt["velocities"].to(device)
            if torch.is_tensor(mol_ckpt.get("Etot")):
                molecule.Etot = mol_ckpt["Etot"].to(device)
            if "forces" in mol_ckpt:
                molecule.force = mol_ckpt["forces"].to(device)
            if "molecular_orbitals" in mol_ckpt and torch.is_tensor(mol_ckpt["molecular_orbitals"]):
                molecule.molecular_orbitals = mol_ckpt["molecular_orbitals"].to(device)
            if "cis_energies" in mol_ckpt and torch.is_tensor(mol_ckpt["cis_energies"]):
                molecule.cis_energies = mol_ckpt["cis_energies"].to(device)
            molecule.dm = mol_ckpt["dm"].to(device) if reuse_P else None
            molecule.cis_amplitudes = mol_ckpt["cis_amplitudes"]
            if isinstance(molecule.cis_amplitudes, torch.Tensor):
                molecule.cis_amplitudes = molecule.cis_amplitudes.to(device)
                old_mos = mol_ckpt.get("old_mos")
                if isinstance(old_mos, torch.Tensor):
                    molecule.old_mos = old_mos.to(device)
                tdm = mol_ckpt.get("transition_density_matrices")
                if isinstance(tdm, torch.Tensor):
                    molecule.transition_density_matrices = tdm.to(device)

    @staticmethod
    def _atomic_save_checkpoint(ckpt: Dict, path: str):
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_ckpt_", suffix=".pt")
        os.close(tmp_fd)
        try:
            torch.save(ckpt, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    @staticmethod
    def _load_checkpoint_base(path: str, device=None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        from seqm.Molecule import Molecule

        torch.set_default_dtype(ckpt["molecules"]["coordinates"].dtype)
        device = device or ckpt["device"]
        const = ckpt["molecules"]["constants"].to(device)

        molecule = Molecule(
            const,
            ckpt["seqm_parameters"],
            ckpt["molecules"]["coordinates"].to(device),
            ckpt["molecules"]["species"].to(device),
        ).to(device)

        reuse_P = ckpt["reuse_P"]
        Molecular_Dynamics_Basic._restore_molecule_from_ckpt(ckpt["molecules"], molecule, reuse_P, device)
        return ckpt, molecule, device, reuse_P

    @staticmethod
    def _restore_rng(ckpt: Dict):
        torch.random.set_rng_state(ckpt["rng"]["torch_cpu"])
        if torch.cuda.is_available() and ckpt["rng"]["torch_cuda"]:
            torch.cuda.set_rng_state_all(ckpt["rng"]["torch_cuda"])


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

    def one_step(self, molecule, learned_parameters=None, *args, **kwargs):
        """Velocity Verlet with Langevin thermostat."""
        learned_parameters = {} if learned_parameters is None else learned_parameters
        dt = self.timestep
        if molecule.const.do_timing:
            t0 = time.time()

        self._apply_langevin_thermostat(molecule)

        with torch.no_grad():
            molecule.velocities.add_(0.5 * molecule.acc * dt)
            molecule.coordinates.add_(molecule.velocities * dt)

        self.esdriver(
            molecule,
            learned_parameters=learned_parameters,
            P0=molecule.dm,
            dm_prop="SCF",
            cis_amp=molecule.cis_amplitudes,
            *args,
            **kwargs,
        )

        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE
            molecule.velocities.add_(0.5 * molecule.acc * dt)

        self._apply_langevin_thermostat(molecule)

        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            molecule.const.timing["MD"].append(time.time() - t0)

    def initialize(
        self, molecule, remove_com=None, learned_parameters=None, steps: Optional[int] = None, *args, **kwargs
    ):
        learned_parameters = {} if learned_parameters is None else learned_parameters
        if self.damp is not None:
            dt = self.timestep
            # s = -γ dt
            s = torch.as_tensor(
                -dt / self.damp, dtype=molecule.coordinates.dtype, device=molecule.coordinates.device
            )
            # c1 = exp{-γ dt/2}
            self.langevin_c1 = torch.exp(0.5 * s)
            # c2 = 1 - c1^2 = 1 - e^{-γ dt}
            one_me = -torch.expm1(s)
            self.langevin_c2 = torch.sqrt(one_me * self.Temp * molecule.mass_inverse) * CONSTANTS.VEL_SCALE
        return super().initialize(molecule, remove_com, learned_parameters, steps=steps, *args, **kwargs)


class XL_BOMD(Molecular_Dynamics_Langevin):
    """Extended Lagrangian Born-Oppenheimer MD."""

    def __init__(self, damp=None, xl_bomd_params=None, *args, **kwargs):
        xl_bomd_params = {} if xl_bomd_params is None else dict(xl_bomd_params)
        self.k = xl_bomd_params["k"]
        self.xl_bomd_params = xl_bomd_params
        super().__init__(damp, *args, **kwargs)
        # check Niklasson et al JCP 130, 214109 (2009)
        # coeff: kappa, alpha, c0, c1, ..., c9
        self.coeffs = {
            3: [1.69, 150e-3, -2.0, 3.0, 0.0, -1.0],
            4: [1.75, 57e-3, -3.0, 6.0, -2.0, -2.0, 1.0],
            5: [1.82, 18e-3, -6.0, 14.0, -8.0, -3.0, 4.0, -1.0],
            6: [1.84, 5.5e-3, -14.0, 36.0, -27.0, -2.0, 12.0, -6.0, 1.0],
            7: [1.86, 1.6e-3, -36.0, 99.0, -88.0, 11.0, 32.0, -25.0, 8.0, -1.0],
            8: [1.88, 0.44e-3, -99.0, 286.0, -286.0, 78.0, 78.0, -90.0, 42.0, -10.0, 1.0],
            9: [1.89, 0.12e-3, -286.0, 858.0, -936.0, 364.0, 168.0, -300.0, 184.0, -63.0, 12.0, -1.0],
        }

        self.m = self.k + 1
        self.kappa = self.coeffs[self.k][0]
        self.alpha = self.coeffs[self.k][1]
        cc = 1.00
        tmp = torch.as_tensor(self.coeffs[self.k][2:]) * self.alpha
        # P(n+1) = 2*P(n) - P(n-1) + cc*kappa*(D(n)-P(n)) + alpha*(c0*P(n) + c1*P(n-1) + ... ck*P(n-k))
        #       =  cc*kappa*D(n)
        #        + (2 - cc*kappa + alpha*c0)*P(n)
        #        + (alpha*c1 - 1) * P(n-1)
        #        + alpha*c2*P(n-2)
        #        + ...
        self.coeff_D = cc * self.kappa
        tmp[0] += 2.0 - cc * self.kappa
        tmp[1] -= 1.0
        self.coeff = torch.nn.Parameter(tmp.repeat(2), requires_grad=False)
        self.add_spherical_potential = (
            False  # Spherical force to prevent atoms from flying off beyond a certain radius
        )
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
        P_new = self.coeff_D * (c * molecule.dm + (1.0 - c) * P) + torch.sum(
            self.coeff[cindx : (cindx + self.m)].reshape(-1, 1, 1, 1) * Pt, dim=0
        )
        return P_new

    def _propagate_excited_state(self, es_amp, es_amp_t, cindx, molecule):
        """Propagate excited state transition density matrices."""
        c = 0.95
        es_new = self.coeff_D * (c * molecule.transition_density_matrices + (1.0 - c) * es_amp) + torch.sum(
            self.coeff[cindx : (cindx + self.m)].reshape(-1, 1, 1, 1, 1) * es_amp_t, dim=0
        )
        return es_new

    def one_step(
        self, molecule, step, P, Pt, es_amp=None, es_amp_t=None, learned_parameters=None, *args, **kwargs
    ):
        """XL-BOMD integration step."""
        learned_parameters = {} if learned_parameters is None else learned_parameters
        dt = self.timestep
        if molecule.const.do_timing:
            t0 = time.time()

        if self.damp:
            self._apply_langevin_thermostat(molecule)

        with torch.no_grad():
            molecule.velocities.add_(0.5 * molecule.acc * dt)
            molecule.coordinates.add_(molecule.velocities * dt)

            # cindx = step%self.m
            # e.g k=5, m=6
            # coeff: c0, c1, c2, c3, c4, c5, c0, c1, c2, c3, c4, c5
            # Pt (0,1,2,3,4,5), step=6n  , cindx = 0, coeff[0:6]
            # Pt (1,2,3,4,5,0), step=6n+1, cindx = 1, coeff[1:7]
            # Pt (2,3,4,5,0,1), step=6n+2
            cindx = step % self.m
            # eq. 22 in https://doi.org/10.1063/1.3148075
            P = self._propagate_P(P, Pt, cindx, molecule)
            Pt[(self.m - 1 - cindx)] = P

            if torch.any(
                active_state_tensor(molecule.active_state, int(molecule.nmol), molecule.coordinates.device)
                > 0
            ):
                es_amp = self._propagate_excited_state(es_amp, es_amp_t, cindx, molecule)
                es_amp_t[(self.m - 1 - cindx)] = es_amp
                es_amp_ortho = es_amp
            else:
                es_amp_ortho = es_amp  # will be set to None

            if self.do_scf:
                calc_type = "SCF"

                # Purify with McWeeny polynomial since P may not be idempotent
                # 3P^2 - 2P^3
                # For restricted density matrix (spin summed) D = 2P. So to purify, D0 = 3/2 D^2 - 1/2 D^3
                # TODO: Make it work for unrestricted P
                P2 = P @ P
                P0 = torch.baddbmm(P2, P2, P, beta=1.5, alpha=-0.5)

            else:
                P0 = P
                calc_type = "XL-BOMD"

        self.esdriver(
            molecule,
            learned_parameters=learned_parameters,
            xl_bomd_params=self.xl_bomd_params,
            P0=P0,
            cis_amp=es_amp_ortho,
            dm_prop=calc_type,
            *args,
            **kwargs,
        )

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
        P, Pt, es_amp, es_amp_t = self.one_step(
            molecule, i, P, Pt, es_amp, es_amp_t, learned_parameters=learned_parameters, **kwargs
        )
        self._xl_ctx.update(P=P, Pt=Pt, es_amp=es_amp, es_amp_t=es_amp_t)

    def initialize(
        self,
        molecule,
        remove_com=None,
        learned_parameters=None,
        steps: Optional[int] = None,
        do_xl_esmd=False,
        *args,
        **kwargs,
    ):
        learned_parameters = {} if learned_parameters is None else learned_parameters
        if torch.any(
            active_state_tensor(molecule.active_state, int(molecule.nmol), molecule.coordinates.device) > 0
        ):
            self.move_on_excited_state = True
            self.esdriver.conservative_force.energy.excited_states["save_tdm_xlbomd"] = True

        molecule.Electronic_entropy = torch.zeros(
            molecule.species.shape[0], device=molecule.coordinates.device
        )

        super().initialize(
            molecule,
            remove_com=remove_com,
            learned_parameters=learned_parameters,
            steps=steps,
            *args,
            **kwargs,
        )

        if self.move_on_excited_state and not do_xl_esmd:
            scf_eps = self.xl_bomd_params.setdefault("scf_eps", 1e-5)
            es_eps = self.xl_bomd_params.setdefault("es_eps", 1e-4)
            dtype = molecule.coordinates.dtype
            if dtype == torch.float32:
                self.vec_eps = 5.0e-5
            elif dtype == torch.float64:
                self.vec_eps = 1.0e-8
            else:
                raise RuntimeError("Set dtype to float64 or float32")

            if "max_rank" in self.xl_bomd_params:
                raise ValueError("KSA-XL-BOMD not supported for excited state dynamics")

            self.do_scf = True
            self.esdriver.conservative_force.energy.hamiltonian.eps = torch.nn.Parameter(
                torch.as_tensor(scf_eps), requires_grad=False
            )
            self.esdriver.conservative_force.energy.excited_states["tolerance"] = es_eps
            self.esdriver.conservative_force.energy.excited_states["make_best_guess"] = False
            if self.esdriver.conservative_force.energy.excited_states["method"].lower() == "rpa":
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
            if "max_rank" in self.xl_bomd_params:
                molecule.dP2dt2 = torch.zeros_like(molecule.dm)
            ctx = {"P": P, "Pt": Pt}

            if self.move_on_excited_state:
                if do_xl_esmd:
                    # es_amp = molecule.cis_amplitudes.clone()
                    es_amp = molecule.transition_density_matrices.clone()
                else:
                    es_amp = molecule.transition_density_matrices.clone()
                es_amp_t = es_amp.unsqueeze(0).expand((self.m,) + es_amp.shape).clone()
                ctx.update(es_amp=es_amp, es_amp_t=es_amp_t)
            self._xl_ctx = ctx


class KSA_XL_BOMD(XL_BOMD):
    """Krylov Subspace Approximation XL-BOMD."""

    def __init__(self, damp=None, xl_bomd_params=None, *args, **kwargs):
        super().__init__(damp, xl_bomd_params, *args, **kwargs)
        self.add_spherical_potential = False

    def _propagate_P(self, P, Pt, cindx, molecule):
        P_new = self.coeff_D * (molecule.dP2dt2 + P) + torch.sum(
            self.coeff[cindx : (cindx + self.m)].reshape(-1, 1, 1, 1) * Pt, dim=0
        )
        return P_new


class XL_ESMD(XL_BOMD):
    """XL-BOMD for excited state MD."""

    def _propagate_excited_state(self, es_amp, es_amp_t, cindx, molecule):
        """Propagate excited state transition density matrices."""
        if getattr(molecule, "dxi2dt2", None) is None:
            c = 0.95
            es_new = self.coeff_D * (
                c * molecule.transition_density_matrices + (1.0 - c) * es_amp
            ) + torch.sum(self.coeff[cindx : (cindx + self.m)].reshape(-1, 1, 1, 1, 1) * es_amp_t, dim=0)
        else:
            es_new = self.coeff_D * (molecule.dxi2dt2 + es_amp) + torch.sum(
                self.coeff[cindx : (cindx + self.m)].reshape(-1, 1, 1, 1, 1) * es_amp_t, dim=0
            )
        return es_new

    def _propagate_excited_amp(self, es_amp, es_amp_t, cindx, molecule):
        """Propagate excited state transition density matrices."""
        if getattr(molecule, "dxi2dt2", None) is None:
            c = 0.95
            es_new = self.coeff_D * (c * molecule.cis_amplitudes + (1.0 - c) * es_amp) + torch.sum(
                self.coeff[cindx : (cindx + self.m)].reshape(-1, 1, 1, 1) * es_amp_t, dim=0
            )
        else:
            es_new = self.coeff_D * (molecule.dxi2dt2 + es_amp) + torch.sum(
                self.coeff[cindx : (cindx + self.m)].reshape(-1, 1, 1, 1) * es_amp_t, dim=0
            )
        return es_new

    def one_step(
        self, molecule, step, P, Pt, es_amp=None, es_amp_t=None, learned_parameters=None, *args, **kwargs
    ):
        learned_parameters = {} if learned_parameters is None else learned_parameters
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
            # es_amp = self._propagate_excited_amp(es_amp, es_amp_t, cindx, molecule)
            es_amp_t[(self.m - 1 - cindx)] = es_amp

            dm_prop = self.dmprop

            if dm_prop == "SCF":
                # Purify with McWeeny polynomial since P may not be idempotent
                # 3P^2 - 2P^3
                # For restricted density matrix (spin summed) D = 2P. So to purify, D0 = 3/2 D^2 - 1/2 D^3
                # TODO: Make it work for unrestricted P
                P2 = P @ P
                P0 = torch.baddbmm(P2, P2, P, beta=1.5, alpha=-0.5)
            else:
                P0 = P

        self.esdriver(
            molecule,
            learned_parameters=learned_parameters,
            xl_bomd_params=self.xl_bomd_params,
            P0=P0,
            cis_amp=es_amp,
            dm_prop=dm_prop,
            *args,
            **kwargs,
        )

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

    def initialize(
        self, molecule, remove_com=None, learned_parameters=None, steps: Optional[int] = None, *args, **kwargs
    ):
        learned_parameters = {} if learned_parameters is None else learned_parameters
        super().initialize(
            molecule,
            remove_com=remove_com,
            learned_parameters=learned_parameters,
            steps=steps,
            do_xl_esmd=True,
            *args,
            **kwargs,
        )
        molecule.Electronic_entropy = torch.zeros(
            molecule.species.shape[0], device=molecule.coordinates.device
        )
        self.esdriver.conservative_force.energy.excited_states = None
        self.esdriver.conservative_force.energy.xlesmd = True
        # self.dmprop = kwargs.get("dmprop","XL-BOMD")
        self.dmprop = kwargs.get("dmprop", "SCF")


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
# acc ==> Angstrom/fs^2


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

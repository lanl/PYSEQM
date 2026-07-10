"""Small HiphopNN electronic-structure adapter for PySEQM FSSH."""

import itertools
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = Path(__file__).resolve().parent / "enol_ef_nacr_developer_handoff"
sys.path[:0] = [str(ROOT), str(HANDOFF / "src")]

from hiphop_predictor import HiphopPredictor

from seqm.api import Constants, Molecule, SurfaceHoppingDynamics, read_xyz


def states(spec):
    out = tuple(sorted({int(x) for x in str(spec).replace(" ", "").split(",") if x}))
    if any(x < 0 or x > 5 for x in out) or len([x for x in out if x]) < 2:
        raise ValueError("--states must contain at least two excited labels from 1..5 (optionally S0).")
    return out


def pairs(labels):
    return tuple(f"{i}{j}" for i, j in itertools.combinations([x for x in labels if x], 2))


def molecule(xyz, ntraj, device):
    Z, R = read_xyz([str(Path(xyz).expanduser())], sort=False)
    Z = torch.as_tensor(Z, dtype=torch.long, device=device).repeat(ntraj, 1)
    R = torch.as_tensor(R, dtype=torch.float64, device=device).repeat(ntraj, 1, 1)
    return Molecule(
        Constants().to(device), {"method": "AM1", "scf_eps": 1.0, "scf_converger": [-1]}, R, Z
    ).to(device)


class HiphopSurfaceHopping(SurfaceHoppingDynamics):
    """FSSH that obtains E, F, and NAC vectors from ``HiphopPredictor``."""

    def __init__(
        self,
        model_dir,
        labels,
        initial_state,
        timestep,
        temperature,
        output,
        device,
        friction=None,
        nac_gap_floor=1e-8,
    ):
        self.labels = tuple(x for x in labels if x)
        self.model_states, self.model_pairs = tuple(labels), pairs(labels)
        if initial_state not in self.labels:
            raise ValueError(f"initial state S{initial_state} is outside {self.labels}")
        self.gap_floor, self.previous_nac = float(nac_gap_floor), None
        params = {
            "method": "AM1",
            "elements": [1, 6, 8],
            "scf_eps": 1e-8,
            "scf_converger": [1],
            "analytical_gradient": [True],
            "excited_states": {"n_states": len(self.labels), "method": "cis"},
            "nonadiabatic": {"tdc_method": "nac_dot_v", "detect_crossings": False},
        }
        super().__init__(
            params,
            timestep=timestep,
            Temp=temperature,
            initial_state=self.labels.index(initial_state) + 1,
            damp=None if friction is None else 1 / friction,
            output=output,
        )
        self.predictor = HiphopPredictor(
            model_dir,
            HANDOFF / "src",
            states=self.model_states,
            pairs=self.model_pairs,
            device=device,
            dtype="float64",
        )

    def _setup_states(self, mol):
        self._nstates = len(self.labels)
        self._full_nac_pair_keys = list(itertools.combinations(range(self._nstates), 2))
        self._full_nac_pairs_1based = [(i + 1, j + 1) for i, j in self._full_nac_pair_keys]
        self._full_nac_state_i, self._full_nac_state_j = zip(*self._full_nac_pair_keys)
        self._ensure_active_states(mol.species.shape[0], mol.coordinates.device)

    def initialize(self, mol, remove_com=None, learned_parameters=None, *args, **kwargs):
        self._setup_states(mol)
        self._init_coeffs(mol)
        self.set_dof(mol)
        self.initialize_velocity(mol)  # nac_dot_v needs v at t=0
        self._compute_electronic_structure(mol, {})
        return super().initialize(
            mol, remove_com=remove_com, learned_parameters=learned_parameters, *args, **kwargs
        )

    def initialize_velocity(self, mol, vel_com=True):
        if torch.is_tensor(mol.velocities):
            return mol.velocities
        return super().initialize_velocity(mol, vel_com=vel_com)

    def _phase_match(self, nac):
        if self.previous_nac is None:
            self.previous_nac = {k: v.detach().clone() for k, v in nac.items()}
            return nac
        # Fit one +/- phase per adiabatic state to the prior step; fix the first state to +1.
        candidates = [(1,) + x for x in itertools.product((-1, 1), repeat=self._nstates - 1)]
        for m in range(next(iter(nac.values())).shape[0]):
            phase = max(
                candidates,
                key=lambda q: sum(
                    float((q[i] * q[j] * nac[i, j][m] * self.previous_nac[i, j][m]).sum())
                    for i, j in self._full_nac_pair_keys
                ),
            )
            for i, j in self._full_nac_pair_keys:
                nac[i, j][m].mul_(phase[i] * phase[j])
        self.previous_nac = {k: v.detach().clone() for k, v in nac.items()}
        return nac

    def _compute_electronic_structure(self, mol, learned_parameters, **kwargs):
        del learned_parameters, kwargs
        self._setup_states(mol)
        pred = self.predictor.predict(
            mol.species.cpu().numpy(), mol.coordinates.detach().cpu().numpy(), gdv_source="forces"
        )
        dev, dtype, (n, a) = mol.coordinates.device, mol.coordinates.dtype, mol.coordinates.shape[:2]
        E = {s: torch.as_tensor(pred["E_abs"][s], dtype=dtype, device=dev) for s in self.model_states}
        e0 = E.get(0, E[self.labels[0]])
        cis = torch.stack([E[s] - e0 for s in self.labels], 1)
        mol.cis_energies = cis
        mol.all_forces = torch.zeros((n, self._nstates + 1, a, 3), dtype=dtype, device=dev)
        for i, s in enumerate(self.labels):
            mol.all_forces[:, i + 1] = torch.as_tensor(pred["F"][s], dtype=dtype, device=dev)
        active, idx = self._ensure_active_states(n, dev), self._get_arange(n, device=dev)
        mol.active_state, mol.force = active + 1, mol.all_forces[idx, active + 1]
        mol.Etot = torch.stack([E[s] for s in self.labels], 1)[idx, active]
        mol.Hf = mol.Eelec = mol.Etot.clone()
        mol.Enuc = mol.Eiso = torch.zeros_like(mol.Etot)
        mol.e_gap, mol.dipole, mol.dm, mol.cis_amplitudes = (
            cis[idx, active],
            torch.zeros((n, 3), dtype=dtype, device=dev),
            None,
            None,
        )
        nac = {}
        for i, j in self._full_nac_pair_keys:
            si, sj, name = self.labels[i], self.labels[j], f"{self.labels[i]}{self.labels[j]}"
            gap = np.asarray(pred["sE"][sj]) - np.asarray(pred["sE"][si])
            gap = np.maximum(gap, 1e-8)
            # if self.gap_floor and np.any(np.abs(gap) < self.gap_floor):
            #     raise RuntimeError(f"NAC {name} has |gap| below {self.gap_floor:g} eV")
            nac[i, j] = torch.as_tensor(
                (np.asarray(pred["dENACR"][name]) / gap[:, None]).reshape(n, a, 3), dtype=dtype, device=dev
            )
        mol.nac = self._phase_match(nac)
        mol.nac_dot = self._nac_dot_from_vectors(mol, mol.nac)
        self._cache_new = {"energies": cis, "ground_energy": e0, "nac_vec": mol.nac, "nac_dot": mol.nac_dot}
        return cis

    def _compute_NACR_for_hop(self, mol, nac_pairs):
        out = self._select_nac_pairs(mol.nac, nac_pairs)
        if out is None:
            raise RuntimeError("missing ML NAC vector required for hop rescaling")
        return out

    def _recompute_active_force(self, mol):
        active, idx = (
            self._active_states,
            self._get_arange(len(self._active_states), device=mol.coordinates.device),
        )
        mol.active_state, mol.force = active + 1, mol.all_forces[idx, active + 1]

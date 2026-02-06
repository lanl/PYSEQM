import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch

from seqm.seqm_functions.rcis_batch import packone_batch

from .dynamics.nac_utils import resolve_nac_config
from .MolecularDynamics import CONSTANTS, Molecular_Dynamics_Langevin
from .seqm_functions.hcore import overlap_between_geometries

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    pass

HBAR_EV_FS = 0.6582119514  # Planck's constant (reduced) in eV·fs

# =============================================================================
# Nonadiabatic (mixed quantum–classical) dynamics in the adiabatic electronic basis
#
# Electronic wavefunction expansion (excited-state manifold only):
#     |Ψ(t)⟩ = Σ_i c_i(t) |φ_i(R(t))⟩
#
# Time-dependent electronic amplitudes in adiabatic representation:
#     ẋc_i = -(i/ħ) E_i(R) c_i  -  Σ_j τ_ij(R,Ṙ) c_j
# where
#     τ_ij(t) = ⟨φ_i| d/dt φ_j⟩ = Ṙ · d_ij(R),
#     d_ij(R) = ⟨φ_i(R)| ∇_R φ_j(R)⟩  (NAC vector)
#
# Nuclear motion (classical):
#     M_k R̈_k = F_k
# where the force definition depends on the chosen scheme:
#   - Ehrenfest (mean-field): F = ⟨Ψ| -∇_R Ĥ_el |Ψ⟩
#   - FSSH (surface hopping): F = -∇_R E_a(R) on the active surface a(t),
#     with stochastic hops a→b chosen from fewest-switches probabilities.
# =============================================================================


@dataclass
class HopEvent:
    step: int
    from_state: int
    to_state: int
    accepted: bool
    mol_index: Optional[int] = None
    reason: Optional[str] = None


class NonadiabaticDynamicsBase(Molecular_Dynamics_Langevin):
    """
    Base class for non-adiabatic dynamics over an excited-state manifold.

    Only excited states are included in the propagated electronic wavefunction
    (no ground-state component). Subclasses provide force definitions such as
    Ehrenfest mean-field.
    """

    def __init__(
        self,
        seqm_parameters: Dict,
        timestep: float = 0.1,
        Temp: float = 0.0,
        step_offset: int = 0,
        output: Optional[Dict] = None,
        compute_nac: Optional[bool] = None,
        initial_state: Union[int, torch.Tensor] = 1,
        damp: Optional[float] = None,
        *args,
        **kwargs,
    ):
        if "damp" in kwargs:
            if damp is None:
                damp = kwargs.pop("damp")
            else:
                kwargs.pop("damp")
        params = dict(seqm_parameters)
        # Analytical gradients are required; all-state forces are needed for mean-field dynamics.
        params.setdefault("analytical_gradient", [True])
        params.setdefault("do_all_forces", True)
        na_cfg = dict(params.get("nonadiabatic", {}))
        if compute_nac is not None:
            na_cfg["compute_nac"] = bool(compute_nac)
        params["nonadiabatic"] = na_cfg

        exc_cfg = params.get("excited_states") or {}
        nroots_cfg = exc_cfg.get("n_states")
        target_states = na_cfg.get("states") or []
        n_validate = None
        if target_states:
            try:
                n_validate = max(int(s) for s in target_states)
            except Exception:
                n_validate = None
        if n_validate is None and nroots_cfg is not None:
            n_validate = int(nroots_cfg)
        nac_settings = resolve_nac_config(params, nroots=n_validate, default_enabled=True)
        na_cfg["compute_nac"] = nac_settings.enabled
        if nac_settings.pairs:
            na_cfg["nac_states"] = nac_settings.pairs
        params["nonadiabatic"] = na_cfg
        super().__init__(
            damp=damp,
            seqm_parameters=params,
            timestep=timestep,
            Temp=Temp,
            step_offset=step_offset,
            output=output,
            *args,
            **kwargs,
        )
        self.compute_nac = nac_settings.enabled
        self._nac_pairs = nac_settings.pairs
        self._force_mode = str(na_cfg.get("force_mode", "all")).lower()
        if self._force_mode not in ("all", "active"):
            raise ValueError(f"Invalid nonadiabatic.force_mode '{self._force_mode}'.")
        self._recompute_on_hop = bool(na_cfg.get("recompute_on_hop", False))
        self.initial_state = initial_state
        max_electronic_dt = 0.05  # fs
        electronic_substeps: int = 10
        nsub = max(int(electronic_substeps), math.ceil(self.timestep / max_electronic_dt))
        # force even
        if nsub % 2:
            nsub += 1
        self._electronic_substeps = nsub
        self._nstates: Optional[int] = None
        self._amp_phase: Optional[torch.Tensor] = None  # (nmol, nstates, 3): x, y, theta
        self._current_potential: Optional[torch.Tensor] = None
        self._active_states: Optional[torch.Tensor] = None  # (nmol,)
        self.hop_log: List[HopEvent] = []
        self._cache_old = None
        self._cache_new = None
        self._hop_integral = None
        self._last_esdriver_args = None
        self._decohere_on_hop = params["nonadiabatic"].get("decohere_on_hop", True)
        self._detect_crossings_flag = params["nonadiabatic"].get("detect_crossings", True)
        self._apc_window = int(params["nonadiabatic"].get("apc_window", 2))
        self._trivial_crossing_mask: Optional[torch.Tensor] = None
        # Reusable per-device caches to avoid reallocations and CPU transfers each step
        self._eye_cache: Dict[tuple, torch.Tensor] = {}
        self._arange_cache: Dict[tuple, torch.Tensor] = {}
        self._hop_buffer: Optional[torch.Tensor] = None
        self._state_energy_buffers: Dict[tuple, torch.Tensor] = {}
        self._trivial_zero_buffers: Dict[tuple, torch.Tensor] = {}
        self._trivial_swap_buffers: Dict[tuple, torch.Tensor] = {}
        self._perm_cost_buffers: Dict[tuple, torch.Tensor] = {}

    def _normalize_initial_state(self, nmol: int, device) -> torch.Tensor:
        init = self.initial_state
        if torch.is_tensor(init):
            if init.dim() != 1:
                raise ValueError("initial_state tensor must be 1D with shape (nmol,).")
            if init.numel() != nmol:
                raise ValueError(f"initial_state tensor must have length nmol={nmol}.")
            init_raw = init.to(device=device, dtype=torch.long)
            min_raw = int(init_raw.min().item())
            max_raw = int(init_raw.max().item())
        else:
            try:
                init_val = int(init)
            except Exception as exc:
                raise TypeError("initial_state must be an int or a torch.Tensor of shape (nmol,).") from exc
            init_raw = torch.full((nmol,), init_val, dtype=torch.long, device=device)
            min_raw = init_val
            max_raw = init_val

        if min_raw < 1:
            raise ValueError("initial_state is 1-indexed; values must be >= 1.")
        if self._nstates is not None and max_raw > self._nstates:
            raise ValueError(f"Initial state {max_raw} > available states ({self._nstates}).")
        return init_raw - 1

    def _setup_states(self, molecule):
        # TODO: Calculate more states than required so that lower states are more accurate.
        # So add 2 states to n_states in seqm_parameters,
        # and don't rely on seqm_parameters['excited_states']['n_states'] in the future
        #  (since it will be 2 more than the number of states we are interested in)
        exc_cfg = self.seqm_parameters.get("excited_states")
        if not exc_cfg or "n_states" not in exc_cfg:
            raise RuntimeError(
                "Quantum dynamics requires seqm_parameters['excited_states']['n_states'] to be set."
            )
        self._nstates = int(exc_cfg["n_states"])
        nmol = molecule.species.shape[0]
        device = molecule.coordinates.device
        self._ensure_active_states(nmol, device)

    def _init_coeffs(self, molecule):
        nmol = molecule.species.shape[0]
        device = molecule.coordinates.device
        active = self._ensure_active_states(nmol, device)
        if self._amp_phase is not None and self._amp_phase.shape[0] == nmol:
            return
        amp_phase = torch.zeros((nmol, self._nstates, 3), dtype=molecule.coordinates.dtype, device=device)
        idx = torch.arange(nmol, device=device)
        amp_phase[idx, active, 0] = 1.0
        self._amp_phase = amp_phase

    def _ensure_active_states(self, nmol: int, device):
        """Ensure `_active_states` exists with correct shape/device."""
        if self._active_states is None or self._active_states.shape[0] != nmol:
            self._active_states = self._normalize_initial_state(nmol, device)
        else:
            self._active_states = self._active_states.to(device=device)
        return self._active_states

    def _uniform_active_state(self) -> Optional[int]:
        """Return scalar active state if all molecules share it."""
        if self._active_states is None:
            init = self.initial_state
            if torch.is_tensor(init):
                unique = torch.unique(init)
                if unique.numel() == 1:
                    return int(unique.item()) - 1
                return None
            return int(init) - 1
        if torch.is_tensor(self._active_states):
            unique = torch.unique(self._active_states)
            if unique.numel() == 1:
                return int(unique.item())
            return None
        return int(self._active_states)

    @property
    def populations(self) -> torch.Tensor:
        if self._amp_phase is None:
            raise RuntimeError("Electronic coefficients not initialized.")
        x = self._amp_phase[..., 0]
        y = self._amp_phase[..., 1]
        return x * x + y * y

    def _get_eye(self, n: int, device, dtype=None):
        key = (n, device, dtype)
        eye = self._eye_cache.get(key)
        if eye is None:
            eye = torch.eye(n, device=device, dtype=dtype)
            self._eye_cache[key] = eye
        return eye

    def _get_arange(self, n: int, device, dtype=torch.long):
        key = (n, device, dtype)
        arr = self._arange_cache.get(key)
        if arr is None or arr.numel() != n or arr.dtype != dtype:
            arr = torch.arange(n, device=device, dtype=dtype)
            self._arange_cache[key] = arr
        return arr

    @staticmethod
    def _get_tensor(cache: Dict, key: tuple, shape, device, dtype, fill_value=None):
        """
        Fetch or allocate a tensor in `cache` keyed by shape/device/dtype.
        Optionally fills with a scalar value (0 or given constant).
        """
        t = cache.get(key)
        if t is None or t.shape != tuple(shape) or t.device != device or t.dtype != dtype:
            t = torch.empty(shape, device=device, dtype=dtype)
            cache[key] = t
        if fill_value is not None:
            if fill_value == 0 or fill_value == 0.0:
                t.zero_()
            else:
                t.fill_(fill_value)
        return t

    def _coeffs_complex(self) -> Optional[torch.Tensor]:
        # Optional reconstruction of full coefficients c from (x, y, theta).
        if self._amp_phase is None:
            return None
        x = self._amp_phase[..., 0]
        y = self._amp_phase[..., 1]
        theta = self._amp_phase[..., 2]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        real = x * cos_t - y * sin_t
        imag = x * sin_t + y * cos_t
        return torch.complex(real, imag)

    def _coherence_real(self, i: int, j: int) -> torch.Tensor:
        if self._amp_phase is None:
            raise RuntimeError("Electronic coefficients not initialized.")
        x = self._amp_phase[..., 0]
        y = self._amp_phase[..., 1]
        theta = self._amp_phase[..., 2]
        A = x[:, i] * x[:, j] + y[:, i] * y[:, j]
        B = x[:, i] * y[:, j] - y[:, i] * x[:, j]
        delta = theta[:, j] - theta[:, i]
        return A * torch.cos(delta) - B * torch.sin(delta)

    def _build_state_energies(self, molecule, active_exc_index: Optional[int] = None) -> torch.Tensor:
        if self._nstates is None:
            raise RuntimeError("Number of states not initialized.")
        nmol = molecule.species.shape[0]
        dtype = molecule.coordinates.dtype
        device = molecule.coordinates.device
        if molecule.cis_energies is None:
            raise RuntimeError("Excited-state energies not available for nonadiabatic dynamics.")
        n_exc = min(molecule.cis_energies.shape[1], self._nstates)
        key = (nmol, self._nstates, str(device), dtype)
        state_energies = self._get_tensor(
            self._state_energy_buffers, key, (nmol, self._nstates), device=device, dtype=dtype, fill_value=0.0
        )
        base = molecule.Etot.reshape(nmol)
        if active_exc_index is not None:
            if torch.is_tensor(active_exc_index):
                idx = active_exc_index.to(device=device)
                if idx.dim() == 0:
                    idx = idx.expand(nmol)
                if torch.any((idx < 0) | (idx >= n_exc)):
                    raise ValueError("Active-state index out of range for cis energies.")
                base = base - molecule.cis_energies[torch.arange(nmol, device=device), idx]
            else:
                if active_exc_index < 0 or active_exc_index >= n_exc:
                    raise ValueError("Active-state index out of range for cis energies.")
                base = base - molecule.cis_energies[:, active_exc_index]
        state_energies[:, :n_exc] = base.unsqueeze(1) + molecule.cis_energies[:, :n_exc]
        return state_energies

    def _set_compute_nac(self, enabled: bool, pairs=None):
        """
        Toggle NAC computation in the underlying Energy object.
        Returns previous (enabled, pairs) to allow restore.
        """
        cf = getattr(self.esdriver, "conservative_force", None)
        if cf is None or not hasattr(cf, "energy"):
            return None
        cfg = cf.energy.nac_config
        prev = (cfg.enabled, getattr(cfg, "pairs", None))
        cfg.enabled = enabled
        cfg.pairs = pairs
        return prev

    def _compute_electronic_structure(
        self, molecule, learned_parameters, compute_nac: bool = False, nac_pairs=None, **kwargs
    ):
        # For "all" forces, set active_state = 0 so Etot is the ground state energy.
        # For "active" forces, keep the active excited state so Etot corresponds to that state.
        old_state = molecule.active_state
        if self._force_mode == "active":
            target_state = self._active_states + 1  # 1-based for excited-state gradients
            active_exc_index = self._active_states
        else:
            target_state = 0
            active_exc_index = None
        molecule.active_state = target_state
        prev_nac_cfg = self._set_compute_nac(compute_nac, nac_pairs)
        esdriver_args = kwargs.pop("esdriver_args", ())
        self._last_esdriver_args = {
            "learned_parameters": learned_parameters,
            "esdriver_args": esdriver_args,
            "kwargs": dict(kwargs),
        }
        self.esdriver(
            molecule,
            learned_parameters=learned_parameters,
            P0=molecule.dm,
            dm_prop="SCF",
            cis_amp=molecule.cis_amplitudes,
            *esdriver_args,
            **kwargs,
        )
        molecule.active_state = old_state
        if prev_nac_cfg is not None:
            self._set_compute_nac(prev_nac_cfg[0], prev_nac_cfg[1])
        energies = self._build_state_energies(molecule, active_exc_index=active_exc_index)
        nac_vec = self._get_nac_matrix(molecule)
        self._cache_new = {
            "energies": energies.clone(),
            "nac_vec": None if nac_vec is None else nac_vec.clone(),
            "cis_amp": None if molecule.cis_amplitudes is None else molecule.cis_amplitudes.detach().clone(),
        }
        return energies

    def _get_nac_matrix(self, molecule):
        if self._nstates is None:
            return None
        nac_vec = getattr(molecule, "nac", None)
        if nac_vec is None:
            return None
        nac_vec = nac_vec[:, : self._nstates, : self._nstates]
        return nac_vec

    @staticmethod
    def _permute_cache(cache, perm):
        if cache is None:
            return None
        idx = torch.as_tensor(perm, device=cache["energies"].device)
        new_cache = {
            "energies": cache["energies"][:, idx],
            "nac_dot": None if cache.get("nac_dot") is None else cache["nac_dot"][:, idx, idx],
            "cis_amp": None if cache.get("cis_amp") is None else cache["cis_amp"][:, idx, ...],
        }
        return new_cache

    @staticmethod
    def _hungarian_perm(cost) -> torch.Tensor:
        # cost: 2D torch tensor on CPU
        try:
            cost_np = cost.numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            perm = torch.full((cost_np.shape[0],), -1, dtype=torch.long)
            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                perm[r] = c
            if (perm < 0).any() or torch.unique(perm).numel() != perm.numel():
                raise RuntimeError("Hungarian algorithm failed to find valid permutation.")
            return perm

        except Exception:
            pass

        # Fallback to manual implementation if SciPy is unavailable or failed.
        n = cost.shape[0]
        dtype = cost.dtype

        # 1-based indexing buffers (classic formulation)
        u = torch.zeros(n + 1, dtype=dtype)  # row potentials
        v = torch.zeros(n + 1, dtype=dtype)  # col potentials
        p = torch.zeros(n + 1, dtype=torch.int64)  # p[j] = row assigned to column j (0 means free)
        way = torch.zeros(n + 1, dtype=torch.int64)

        inf = torch.tensor(float("inf"), dtype=dtype)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = torch.full((n + 1,), inf, dtype=dtype)
            used = torch.zeros(n + 1, dtype=torch.bool)

            while True:
                used[j0] = True
                i0 = int(p[j0].item())  # current row (1..n)

                # Reduced costs for all columns 1..n
                cur = cost[i0 - 1, :] - u[i0] - v[1:]  # shape (n,)

                # Relax edges to unused columns (vectorized)
                mask = ~used[1:]  # shape (n,)
                better = (cur < minv[1:]) & mask
                if better.any():
                    minv[1:][better] = cur[better]
                    way[1:][better] = j0

                # Pick next column with smallest minv among unused (vectorized argmin)
                tmp = minv[1:].clone()
                tmp[~mask] = inf
                delta, j1_0 = tmp.min(dim=0)
                j1 = int(j1_0.item()) + 1  # back to 1..n

                # Update potentials
                used_idx = torch.nonzero(used, as_tuple=False).squeeze(1)
                u[p[used_idx]] += delta
                v[used_idx] -= delta
                minv[~used] -= delta

                j0 = j1
                if p[j0].item() == 0:
                    break

            # Augment along the found path
            while True:
                j1 = int(way[j0].item())
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        # Convert p (col->row) into col_for_row
        col_for_row = torch.full((n,), -1, dtype=torch.int64)
        for j in range(1, n + 1):
            r = int(p[j].item())
            if r != 0:
                col_for_row[r - 1] = j - 1

        # Sanity checks
        if (col_for_row < 0).any():
            raise RuntimeError("Assignment failed (unassigned row).")
        if torch.unique(col_for_row).numel() != n:
            raise RuntimeError("Assignment invalid (duplicate columns).")

        return col_for_row

    def _compute_perm_from_overlap(self, ref_or_ovlp, tgt_amp=None):
        """
        Compute optimal permutation per molecule using APC windowed cost.
        Accepts either:
          - ref_or_ovlp: overlap matrix (nmol,n,n) and tgt_amp=None
          - ref_or_ovlp: ref amplitudes, tgt_amp: target amplitudes (both nmol,nstates, ncoeff)
        Returns:
          - list of perms (one per molecule) if nmol>1, else a single perm list.
        """
        if tgt_amp is not None:
            ovlp = torch.square(torch.einsum("nia,nja->nij", ref_or_ovlp, tgt_amp))
        else:
            ovlp = torch.square(ref_or_ovlp)
        if ovlp.dim() != 3:
            raise ValueError("ovlp must have shape (nmol, nstates, nstates)")

        nmol, nstates, _ = ovlp.shape
        big = 1e5
        w = self._apc_window
        ovlp_cpu = ovlp.detach().to("cpu")

        cpu = torch.device("cpu")
        key = (ovlp_cpu.shape, ovlp_cpu.dtype)
        cost_cpu = self._get_tensor(
            self._perm_cost_buffers, key, ovlp_cpu.shape, device=cpu, dtype=ovlp_cpu.dtype, fill_value=big
        )

        # Fill only the window with negative overlap*big
        i_idx = torch.arange(nstates, device=cpu).view(1, nstates, 1)
        j_idx = torch.arange(nstates, device=cpu).view(1, 1, nstates)
        mask = (j_idx >= (i_idx - w)) & (j_idx <= (i_idx + w))
        mask = mask.expand(nmol, -1, -1)
        cost_cpu[mask] = -ovlp_cpu[mask] * big

        # Run Hungarian per molecule (each cost matrix already on CPU)
        perms = [self._hungarian_perm(cost_cpu[m]) for m in range(nmol)]
        return torch.stack(perms, dim=0)

    @staticmethod
    def _time_derivative_coupling(
        molecule,
        coords_prev,
        mos_prev,
        cis_prev,  # CIS: (nmol,nstates,nov) or RPA: (2,nmol,nstates,nov)
        cis_curr,
        dt,
        enforce_antisym=True,
    ):
        """
        Excited–excited time-derivative NAC using finite diff of state overlaps.
        See I. Ryabinkin, J. Nagesh, and F. Izmaylov, J. Phys. Chem. Lett. 6, 4200-4203 (2015)
        The implementation below follows NWCHEM's: J. Chem. Theory Comput. 2020, 16, 6418−6427
        Shapes expected:
          - CIS/TDA: cis_* shape (nmol, nstates, nov)  with nov = nocc*nvirt
          - RPA:     cis_* shape (2, nmol, nstates, nov) with [X,Y]

        Returns:
          nac_dt: (nmol, nstates, nstates)
        """
        if coords_prev is None or mos_prev is None or cis_prev is None or cis_curr is None:
            return None

        if molecule.nocc.dim() != 1:  # restricted closed-shell
            return None

        # TODO: orbital_window logic to limit nocc/nvirt
        nocc = int(molecule.nocc[0].item())
        norb = int(molecule.norb[0].item())
        nvirt = norb - nocc

        nmol = int(molecule.nmol)
        nov = nocc * nvirt

        def parse_amp(amp):
            # CIS/TDA: (nmol, nstates, nov)
            if amp.dim() == 3:
                # keep both flattened and 4D views
                flat = amp
                view = amp.view(nmol, amp.shape[1], nocc, nvirt)
                return ("cis", flat, view)

            # RPA: (2, nmol, nstates, nov)
            if amp.dim() == 4 and amp.shape[0] == 2:
                Xf = amp[0]
                Yf = amp[1]
                Xv = Xf.view(nmol, amp.shape[2], nocc, nvirt)
                Yv = Yf.view(nmol, amp.shape[2], nocc, nvirt)
                return ("rpa", (Xf, Yf), (Xv, Yv))

            return None

        prev = parse_amp(cis_prev)
        curr = parse_amp(cis_curr)
        if prev is None or curr is None:
            return None

        with torch.no_grad():
            # AO overlap between current geometry (rows) and previous geometry (cols)
            # TODO: check if overlap is b/w current-prev or prev-current
            S_ao = overlap_between_geometries(molecule, molecule.coordinates.detach(), coords_prev.detach())
            # S_ao has to be packed (hydrogen blocks with zero-padding have to be removed)
            S_ao = packone_batch(S_ao, 4 * molecule.nHeavy[0], molecule.nHydro[0], norb)

            # MO overlap: S_mo = C(t)^T S_ao(t,t-dt) C(t-dt)
            Cc = molecule.molecular_orbitals  # (nmol, nao, norb)
            Cp = mos_prev  # (nmol, nao, norb)
            S_mo = Cc.transpose(1, 2) @ (S_ao @ Cp)  # (nmol, norb, norb)

            Soo = S_mo[:, :nocc, :nocc]  # (nmol, nocc, nocc)
            Svv = S_mo[:, nocc:, nocc:]  # (nmol, nvirt, nvirt)

            dSoo = Soo.transpose(1, 2) - Soo
            dSvv = Svv.transpose(1, 2) - Svv

            # helper: MO-derivative term on virtual block
            def mo_term_virtual(C_view):
                # C_view: (nmol, nstates, nocc, nvirt)
                Cd = torch.matmul(C_view, dSvv.transpose(1, 2).unsqueeze(1))  # (nmol, nstates, nocc, nvirt)
                Cf = C_view.reshape(nmol, C_view.shape[1], nov)  # (nmol, nstates, nov)
                Cdf = Cd.reshape(nmol, Cd.shape[1], nov)
                return torch.bmm(Cf, Cdf.transpose(1, 2))  # (nmol, nstates, nstates)

            # helper: MO-derivative term on occupied block
            def mo_term_occ(C_view):
                # apply dSoo^T on the occ index
                Ct = C_view.permute(0, 1, 3, 2)  # (nmol, nstates, nvirt, nocc)
                Ctd = torch.matmul(Ct, dSoo.transpose(1, 2).unsqueeze(1))  # (nmol, nstates, nvirt, nocc)
                Cd = Ctd.permute(0, 1, 3, 2)  # (nmol, nstates, nocc, nvirt)
                Cf = C_view.reshape(nmol, C_view.shape[1], nov)
                Cdf = Cd.reshape(nmol, Cd.shape[1], nov)
                return torch.bmm(Cf, Cdf.transpose(1, 2))

            if curr[0] == "cis":
                _, flat_p, view_p = prev
                _, flat_c, view_c = curr

                # CI-derivative term: <p|c> - <c|p>
                ov_pc = torch.bmm(flat_p, flat_c.transpose(1, 2))
                ov_cp = torch.bmm(flat_c, flat_p.transpose(1, 2))
                coup = ov_pc - ov_cp

                # MO-derivative terms
                coup = coup + mo_term_virtual(view_c) + mo_term_occ(view_c)

            else:
                _, (Xp_f, Yp_f), (Xp_v, Yp_v) = prev
                _, (Xc_f, Yc_f), (Xc_v, Yc_v) = curr

                # CI-derivative term in Fortran style:
                # (X+Y)^T(X+Y) + (X-Y)^T(X-Y)  antisymmetrized between steps
                Ap_p = Xp_f + Yp_f
                Ap_c = Xc_f + Yc_f
                Am_p = Xp_f - Yp_f
                Am_c = Xc_f - Yc_f

                ov_pc = torch.bmm(Ap_p, Ap_c.transpose(1, 2)) + torch.bmm(Am_p, Am_c.transpose(1, 2))
                ov_cp = torch.bmm(Ap_c, Ap_p.transpose(1, 2)) + torch.bmm(Am_c, Am_p.transpose(1, 2))
                coup = ov_pc - ov_cp

                # MO-derivative terms: +X part +Y part (note the + sign)
                coup = (
                    coup
                    + mo_term_virtual(Xc_v)
                    + mo_term_occ(Xc_v)
                    + mo_term_virtual(Yc_v)
                    + mo_term_occ(Yc_v)
                )

            if enforce_antisym:
                coup = 0.5 * (coup - coup.transpose(1, 2))
                coup = coup - torch.diag_embed(torch.diagonal(coup, dim1=1, dim2=2))

            nac_dt = coup / (2.0 * dt)

        return nac_dt

    def _detect_crossings(self, cache_old, cache_new):
        # Trivial-crossing (cross==2 in NEXMD) detection (crossed states have overlap >= 0.9)
        # and zero-ing out NACT between trivially swapped states so that hops are not attempted (these states will be swapped manually).
        # Returns swap_to[m, i] = j for states involved in a trivial swap, else -1.

        if (not self._detect_crossings_flag) or (cache_old is None) or (cache_new is None):
            return None

        ref_amp = cache_old.get("cis_amp")
        tgt_amp = cache_new.get("cis_amp")

        if ref_amp is None or tgt_amp is None or ref_amp.shape != tgt_amp.shape:
            return None

        # If RPA, get only the X amplitudes for overlap
        if ref_amp.dim() == 4 and ref_amp.shape[0] == 2:
            ref_amp = ref_amp[0]
            tgt_amp = tgt_amp[0]

        # Overlap matrix |S_ij| between "old" and "new" electronic amplitudes
        overlap = torch.abs(torch.einsum("nia,nja->nij", ref_amp, tgt_amp))  # (nmol, n, n)
        nmol, n_states, _ = overlap.shape
        device = overlap.device

        thr = 0.9  # trivial-crossing threshold (same as later checks)
        diag_idx = self._get_arange(n_states, device=device)

        # respect the same APC/Hungarian window you use in _compute_perm_from_overlap
        w = self._apc_window
        i = diag_idx.view(1, n_states, 1)
        j = diag_idx.view(1, 1, n_states)
        in_win = (j >= (i - w)) & (j <= (i + w))  # (1, n, n)
        ov_win = overlap.masked_fill(~in_win, 0.0)
        # prefilter: if no off-diagonal entry reaches thr, trivial crossing can never occur.
        ov_off = ov_win.clone()
        ov_off[:, diag_idx, diag_idx] = 0.0
        has_strong_offdiag = (ov_off.max(dim=2).values >= thr).any(dim=1)  # (nmol,)

        # "Holdoff" prevents expensive crossing detection for molecules right after a hop
        holdoff = self.post_hop_holdoff > 0
        active = self._active_states  # (nmol,)
        mol_ar = self._get_arange(nmol, device=device)
        active_row = ov_win[mol_ar, active].clone()  # (nmol, n)
        active_row[mol_ar, active] = 0.0
        active_has_partner = active_row.max(dim=1).values >= thr  # (nmol,)

        # Two groups may need assignment (i.e., permutation):
        #  (1) probe group: in holdoff, but we might reset holdoff early (NEXMD conthop reset)
        #  (2) detect group: not in holdoff, and has any strong off-diagonal candidate, do full trivial-cross detection
        probe_mask = holdoff & (self.prev_state >= 0) & active_has_partner
        detect_mask = (~holdoff) & has_strong_offdiag
        need_perm = probe_mask | detect_mask
        if not need_perm.any():
            return None

        need_idx = need_perm.nonzero(as_tuple=False).squeeze(1)  # (n_need,)
        ov_need = ov_win[need_idx]  # (n_need, n, n)
        perm_need = self._compute_perm_from_overlap(ov_need).to(dtype=torch.long, device=device)

        # ---- Probe reset (minimal NEXMD-style reset of holdoff) ----
        # If the active state's trivial-swap partner differs from prev_state, clear holdoff.
        probe_in_need = probe_mask[need_idx]
        if probe_in_need.any():
            probe_mol_idx = need_idx[probe_in_need]  # full molecule indices
            perm_probe = perm_need[probe_in_need]  # (n_probe, n)

            active_probe = self._active_states[probe_mol_idx]  # (n_probe,)
            row = self._get_arange(active_probe.shape[0], device=device)

            partner = perm_probe[row, active_probe]  # partner = p(active_probe)
            partner_ov = ov_win[probe_mol_idx, active_probe, partner]  # |S_active,partner|

            is_trivial_active = (partner != active_probe) & (partner_ov >= thr)
            prev = self.prev_state[probe_mol_idx]
            reset = is_trivial_active & (partner != prev)
            if reset.any():
                self.post_hop_holdoff[probe_mol_idx[reset]] = 0

        # ---- Detect + build swaps for trivial crossings ----
        detect_in_need = detect_mask[need_idx]
        if not detect_in_need.any():
            return None

        detect_mol_idx = need_idx[detect_in_need]  # full molecule indices
        perm = perm_need[detect_in_need]  # (n_det, n)
        overlap_det = ov_win[detect_mol_idx]  # (n_det, n, n)

        # overlap_det[i, p(i)] for all i; trivial if (p(i) != i) and (i < p(i)) and overlap>=0.9
        i = diag_idx.expand(perm.shape[0], n_states)
        row = self._get_arange(perm.shape[0], device=device)
        ov_ip = overlap_det[row[:, None], i, perm]  # (n_det, n)

        trivial = (perm != i) & (i < perm) & (ov_ip >= thr)
        if not trivial.any():
            return None

        # swap_to[m, i] = j for swapped pairs, else -1
        swap_key = (nmol, n_states, str(device))
        swap_to = self._get_tensor(
            self._trivial_swap_buffers,
            swap_key,
            (nmol, n_states),
            device=device,
            dtype=torch.long,
            fill_value=-1,
        )

        det_row, i_sel = trivial.nonzero(as_tuple=True)
        j_sel = perm[det_row, i_sel]
        full_m = detect_mol_idx[det_row]

        swap_to[full_m, i_sel] = j_sel
        swap_to[full_m, j_sel] = i_sel  # symmetric swap

        # Zero time-derivative-couplings for swapped pairs (NEXMD zeros cadiabold/new for cross==2)
        zero_key = (nmol, n_states, str(device))
        zero_mask = self._get_tensor(
            self._trivial_zero_buffers,
            zero_key,
            (nmol, n_states, n_states),
            device=device,
            dtype=torch.bool,
            fill_value=False,
        )
        zero_mask[full_m, i_sel, j_sel] = True
        zero_mask[full_m, j_sel, i_sel] = True

        def _zero_nac_dot(cache):
            nac_dot = cache.get("nac_dot")
            if nac_dot is not None:
                cache["nac_dot"] = nac_dot.masked_fill(zero_mask, 0.0)

        _zero_nac_dot(cache_old)
        _zero_nac_dot(cache_new)

        return swap_to

    def _get_simpson_weights(self, nsub: int, device, dtype) -> torch.Tensor:
        # nsub even; returns (nsub+1,) weights: 1,4,2,4,...,2,4,1
        cache = getattr(self, "_simpson_w_cache", None)
        key = (int(nsub), str(device), str(dtype))
        if cache is None:
            cache = {}
            self._simpson_w_cache = cache
        w = cache.get(key)
        if w is None:
            w = torch.full((nsub + 1,), 2.0, device=device, dtype=dtype)
            w[0] = 1.0
            w[-1] = 1.0
            w[1:-1:2] = 4.0
            cache[key] = w
        return w

    @staticmethod
    def _interp_linear(old, new, tau):
        if old is None or new is None:
            return new if new is not None else old
        return (1.0 - tau) * old + tau * new

    def _propagate_electronic(self, cache_old, cache_new, substeps=None):
        if self._amp_phase is None:
            return
        # Integrating-factor RK4: separate fast dynamical phases so the slow amplitudes
        # evolve without the stiff diagonal term, preserving |c|^2 with larger substeps.
        dt_total = self.timestep
        nsub = int(substeps or self._electronic_substeps)
        dt_sub = dt_total / float(nsub)

        def interp(tau):
            energies = self._interp_linear(cache_old["energies"], cache_new["energies"], tau)
            nd_old = cache_old.get("nac_dot")
            nd_new = cache_new.get("nac_dot")
            nac_dot = None
            if nd_old is not None and nd_new is not None:
                nac_dot = self._interp_linear(nd_old, nd_new, tau)
            # On the first step, NACTs for previous step are not available, so use only the new
            elif nd_new is not None:
                nac_dot = nd_new
            elif nd_old is not None:
                nac_dot = nd_old
            return energies, nac_dot

        def rhs(x, y, theta, energies, nac_proj):
            dtheta = -energies / HBAR_EV_FS
            if nac_proj is None:
                return torch.zeros_like(x), torch.zeros_like(y), dtheta
            delta = theta[:, None, :] - theta[:, :, None]
            cos_d = torch.cos(delta)
            sin_d = torch.sin(delta)
            xj = x[:, None, :]
            yj = y[:, None, :]
            term_x = xj * cos_d - yj * sin_d
            term_y = yj * cos_d + xj * sin_d
            dx = -torch.sum(nac_proj * term_x, dim=2)
            dy = -torch.sum(nac_proj * term_y, dim=2)
            return dx, dy, dtheta

        def hop_numerator(x, y, theta, nac_proj):
            # Returns tensor (nmol, nstates, nstates): 2 Re[c_i* c_j tau_ij]
            if nac_proj is None:
                return None
            delta = theta[:, None, :] - theta[:, :, None]
            cos_d = torch.cos(delta)
            sin_d = torch.sin(delta)
            xi = x[:, :, None]
            yi = y[:, :, None]
            xj = x[:, None, :]
            yj = y[:, None, :]
            A = xi * xj + yi * yj
            B = xi * yj - yi * xj
            return 2.0 * nac_proj * (A * cos_d - B * sin_d)

        device = self._amp_phase.device
        dtype = self._amp_phase.dtype
        hop_shape = (self._amp_phase.shape[0], self._nstates, self._nstates)
        if (
            self._hop_buffer is None
            or self._hop_buffer.shape != hop_shape
            or self._hop_buffer.device != device
        ):
            self._hop_buffer = torch.zeros(hop_shape, dtype=dtype, device=device)
        else:
            self._hop_buffer.zero_()
        hop_int = self._hop_buffer
        w = self._get_simpson_weights(nsub, device=device, dtype=dtype)  # (nsub+1,)

        for s in range(nsub + 1):
            base_tau = s * dt_sub / dt_total
            tau_half = base_tau + 0.5 * dt_sub / dt_total
            tau_full = base_tau + dt_sub / dt_total
            # Integrating-factor form removes fast dynamical phase rotation.
            x0 = self._amp_phase[..., 0]
            y0 = self._amp_phase[..., 1]
            th0 = self._amp_phase[..., 2]

            # Stage 1
            e1, nd1 = interp(base_tau)

            # ------- Integrate the hop numerator separately with Simpson's rule -------
            R = hop_numerator(x0, y0, th0, nd1)
            if R is not None:
                hop_int.add_(R, alpha=float(w[s]))
            if s == nsub:  # For last substep, do not continue to RK4 stages
                break
            # ---------------------------------------------------------------------------

            dx1, dy1, dth1 = rhs(x0, y0, th0, e1, nd1)

            # Stage 2
            x2 = x0 + 0.5 * dt_sub * dx1
            y2 = y0 + 0.5 * dt_sub * dy1
            th2 = th0 + 0.5 * dt_sub * dth1
            e2, nd2 = interp(tau_half)
            dx2, dy2, dth2 = rhs(x2, y2, th2, e2, nd2)

            # Stage 3
            x3 = x0 + 0.5 * dt_sub * dx2
            y3 = y0 + 0.5 * dt_sub * dy2
            th3 = th0 + 0.5 * dt_sub * dth2
            e3, nd3 = interp(tau_half)
            dx3, dy3, dth3 = rhs(x3, y3, th3, e3, nd3)

            # Stage 4
            x4 = x0 + dt_sub * dx3
            y4 = y0 + dt_sub * dy3
            th4 = th0 + dt_sub * dth3
            e4, nd4 = interp(tau_full)
            dx4, dy4, dth4 = rhs(x4, y4, th4, e4, nd4)

            # RK4 coefficient update
            self._amp_phase[..., 0] = x0 + (dt_sub / 6.0) * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
            self._amp_phase[..., 1] = y0 + (dt_sub / 6.0) * (dy1 + 2 * dy2 + 2 * dy3 + dy4)
            self._amp_phase[..., 2] = th0 + (dt_sub / 6.0) * (dth1 + 2 * dth2 + 2 * dth3 + dth4)

        # Wrap self._amp_phase[..., 2] to [-pi, pi]
        self._amp_phase[..., 2] = torch.remainder(self._amp_phase[..., 2] + torch.pi, 2 * torch.pi) - torch.pi

        hop_int.mul_(dt_sub / 3.0)
        hop_int.mul_(1.0 - self._get_eye(self._nstates, device=device, dtype=dtype).unsqueeze(0))
        self._hop_integral = hop_int

    def _thermo_potential(self, molecule):
        if self._current_potential is not None:
            return self._current_potential
        return super()._thermo_potential(molecule)

    def initialize(
        self,
        molecule,
        remove_com=None,
        learned_parameters=dict(),
        steps: Optional[int] = None,
        *args,
        **kwargs,
    ):
        self._setup_states(molecule)
        self._init_coeffs(molecule)
        molecule.active_state = (
            self._active_states + 1
        )  # excited-state index (1-based for downstream grad routines)

        # Inital energies, forces calculated in parent initialize.
        # TODO: For Ehrenfest need NACs too.
        super().initialize(
            molecule,
            remove_com=remove_com,
            learned_parameters=learned_parameters,
            steps=steps,
            *args,
            **kwargs,
        )
        state_energies = self._build_state_energies(molecule, active_exc_index=self._active_states)
        nac_vec = self._get_nac_matrix(molecule)
        self._cache_old = {
            "energies": state_energies,
            "nac_vec": None if nac_vec is None else nac_vec.clone(),
            "cis_amp": None if molecule.cis_amplitudes is None else molecule.cis_amplitudes.detach().clone(),
            "nac_dot": None,
        }

        if self._cache_old["nac_dot"] is None and self._cache_old["nac_vec"] is not None:
            with torch.no_grad():
                v = molecule.velocities.detach()
                nd = torch.sum(v.unsqueeze(1).unsqueeze(1) * self._cache_old["nac_vec"], dim=(3, 4))
                self._cache_old["nac_dot"] = nd

        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE

        nmol = molecule.species.shape[0]
        device = molecule.coordinates.device
        self.post_hop_holdoff = torch.zeros(
            nmol, dtype=torch.int64, device=device
        )  # blocks crossing detection
        self.post_relabel_holdoff = torch.zeros(
            nmol, dtype=torch.int64, device=device
        )  # blocks hop acceptance
        self.prev_state = torch.full((nmol,), -1, dtype=torch.long, device=device)  # like ihopprev

        if self.step_offset == 0 and self._h5_writer is not None:
            na_stride = self.output_config.get_h5_write_nonadiabatic()
            if na_stride > 0:
                active_states = self._active_states + 1
                amplitudes = self._coeffs_complex()
                nac_dot = None
                if self._cache_old is not None:
                    nac_dot = self._cache_old.get("nac_dot")
                self._h5_writer.append_nonadiabatic(
                    0, active_states=active_states, amplitudes=amplitudes, nac_dot=nac_dot
                )

    def _after_electronic_update(
        self, molecule, state_energies, nac_matrix=None, nac_dot=None, step: Optional[int] = None
    ):
        raise NotImplementedError

    def _do_integrator_step(self, i, molecule, learned_parameters, **kwargs):
        dt = self.timestep
        cache_old = self._cache_old or self._cache_new

        coords_prev = molecule.coordinates.detach().clone()
        mos_prev = getattr(molecule, "molecular_orbitals", None)
        if torch.is_tensor(mos_prev):
            mos_prev = mos_prev.detach().clone()
        if self.damp is not None:
            self._apply_langevin_thermostat(molecule)

        # ---- Half kick + drift to t+dt ----
        with torch.no_grad():
            molecule.velocities.add_(0.5 * molecule.acc * dt)
            molecule.coordinates.add_(molecule.velocities * dt)

        _ = self._compute_electronic_structure(molecule, learned_parameters, compute_nac=False, **kwargs)

        # ---- Half kick to t+dt ----
        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE
            molecule.velocities.add_(0.5 * molecule.acc * dt)

        if self.damp is not None:
            self._apply_langevin_thermostat(molecule)

        cache_new = self._cache_new

        nac_dt = self._time_derivative_coupling(
            molecule,
            coords_prev,
            mos_prev,
            cache_old.get("cis_amp") if cache_old is not None else None,
            cache_new.get("cis_amp"),
            dt,
        )

        def _attach_nac_dot(cache, vel):
            if cache is None or cache.get("nac_vec") is None:
                return cache
            v = vel.unsqueeze(1).unsqueeze(1)  # (nmol,1,1,molsize,3)
            nd = torch.sum(v * cache["nac_vec"], dim=(3, 4))
            cache = dict(cache)
            cache["nac_dot"] = nd
            return cache

        if nac_dt is None:
            cache_new = (
                _attach_nac_dot(cache_new, molecule.velocities.detach()) if cache_new is not None else None
            )
        else:
            cache_new = dict(cache_new)
            cache_new["nac_dot"] = nac_dt

        # update cooldown timers used for trivial crossing handling
        self.post_hop_holdoff = (self.post_hop_holdoff - 1).clamp(min=0)
        self.post_relabel_holdoff = (self.post_relabel_holdoff - 1).clamp(min=0)

        # Detect trivial crossings between previous and current ordering
        self._trivial_crossing_mask = self._detect_crossings(cache_old, cache_new)
        # self._trivial_crossing_mask: (n_mol, nstates) tensor of permutations for crossed mols

        self._propagate_electronic(cache_old, cache_new, substeps=self._electronic_substeps)

        self._after_electronic_update(
            molecule,
            state_energies=cache_new["energies"],
            nac_matrix=cache_new.get("nac_vec"),
            nac_dot=cache_new.get("nac_dot"),
            step=i + self.step_offset,
        )

        if self._h5_writer:
            na_stride = self.output_config.get_h5_write_nonadiabatic()
            if na_stride > 0 and ((i + 1) % na_stride == 0):
                active_states = None
                if self._active_states is not None:
                    active_states = self._active_states + 1
                else:
                    uniform = self._uniform_active_state()
                    if uniform is not None:
                        active_states = uniform + 1
                amplitudes = self._coeffs_complex()
                self._h5_writer.append_nonadiabatic(
                    i + 1,
                    active_states=active_states,
                    amplitudes=amplitudes,
                    nac_dot=cache_new.get("nac_dot"),
                )

        # shift caches for next step
        self._cache_old = {
            "energies": cache_new["energies"].clone(),
            "nac_dot": None if cache_new.get("nac_dot") is None else cache_new["nac_dot"].clone(),
            "nac_vec": None if cache_new.get("nac_vec") is None else cache_new["nac_vec"].clone(),
            "cis_amp": None if cache_new.get("cis_amp") is None else cache_new["cis_amp"].clone(),
        }
        if isinstance(self._trivial_crossing_mask, torch.Tensor):
            swap_to = self._trivial_crossing_mask
            crossing_mask = (swap_to >= 0).any(dim=1)
            if crossing_mask.any():
                idx = torch.nonzero(crossing_mask, as_tuple=False).squeeze(1).tolist()
                msg = ", ".join([f"mol{m}" for m in idx])
                print(f"[NonadiabaticDynamics] Trivial crossing detected: {msg}")


class EhrenfestDynamics(NonadiabaticDynamicsBase):
    """Mean-field (Ehrenfest) dynamics."""

    def _after_electronic_update(
        self, molecule, state_energies, nac_matrix=None, nac_dot=None, step: Optional[int] = None
    ):
        if molecule.all_forces is None or molecule.all_forces.shape[1] < self._nstates + 1:
            raise RuntimeError("EhrenfestDynamics requires per-state forces via do_all_forces.")

        pop = self.populations  # shape (nmol, nstates)
        weights = pop.unsqueeze(2).unsqueeze(3)
        # all_forces is [ground + excited]; drop ground
        F_states = molecule.all_forces[:, 1 : 1 + self._nstates]
        force = torch.sum(weights * F_states, dim=1)

        if nac_matrix is not None:
            nac_corr = torch.zeros_like(force)
            for i in range(self._nstates):
                for j in range(i + 1, self._nstates):
                    delta_e = state_energies[:, j] - state_energies[:, i]
                    coh = self._coherence_real(i, j)
                    nac_corr -= (2.0 * coh * delta_e).view(-1, 1, 1) * nac_matrix[:, i, j]
            force = force + nac_corr

        molecule.force = force
        self._current_potential = torch.sum(pop * state_energies, dim=1)
        molecule.Etot = self._current_potential


class SurfaceHoppingDynamics(NonadiabaticDynamicsBase):
    """Fewest Switches Surface Hopping (FSSH) with simple energy rescaling."""

    def __init__(self, seqm_parameters: Dict, *args, **kwargs):
        params = dict(seqm_parameters)
        params.setdefault("do_all_forces", False)
        na_cfg = dict(params.get("nonadiabatic", {}))
        na_cfg.setdefault("force_mode", "active")
        na_cfg.setdefault("recompute_on_hop", True)
        params["nonadiabatic"] = na_cfg
        super().__init__(params, *args, **kwargs)

    def _attempt_hop(self, state_energies, molecule, step: int) -> List[Optional[int]]:
        nmol = state_energies.shape[0]
        if self._hop_integral is None:
            return [None] * nmol
        device = molecule.coordinates.device
        active_states = self._ensure_active_states(nmol, device)
        pop = self.populations  # (nmol, nstates)

        # Probabilities g_ij for each mol from active state i -> j
        arange = self._get_arange(nmol, device=device)
        i_state = active_states
        denom = torch.clamp(pop[arange, i_state], min=1e-10)
        # FSSH: g_ij = max(0, - Δa_ii / a_ii) with Δa_ii ≈ ∫ 2 Re[c_i* c_j τ_ij] dt
        g_rows = self._hop_integral[arange, i_state] / denom.unsqueeze(1)
        g_rows = torch.clamp(g_rows, min=0.0)

        # g_rows = torch.clamp(self._hop_integral[arange, i_state] / denom.unsqueeze(1), min=0.0)

        # Guard against dt so large that Σ_j g_ij > 1
        g_sum = g_rows.sum(dim=1, keepdim=True)
        g_rows = torch.where(g_sum > 1.0, g_rows / g_sum.clamp(min=1e-12), g_rows)

        # Cumulative draw per molecule
        cumsum = torch.cumsum(g_rows, dim=1)
        r = torch.rand(nmol, device=device)
        cmp = cumsum >= r.unsqueeze(1)
        has_hop = cmp.any(dim=1)
        hop_targets: List[Optional[int]] = [None] * nmol

        if has_hop.any():
            cmp_sub = cmp[has_hop]
            tgt = torch.argmax(cmp_sub.to(torch.long), dim=1)

            idx = arange[has_hop]
            for m, to in zip(idx.tolist(), tgt.tolist()):
                hop_targets[m] = int(to)

        return hop_targets

    def _rescale_velocity_along_nac(self, nac_vec, i_state, j_state, molecule, dE, step, mol_index: int):
        if nac_vec is None:
            return False
        dvec = nac_vec[mol_index, i_state, j_state]  # (molsize, 3)
        m_inv = molecule.mass_inverse[mol_index].squeeze(-1)  # (molsize,)
        d2_by_m = torch.sum(m_inv * torch.sum(dvec * dvec, dim=1))
        if d2_by_m <= 1e-12:
            return False
        v_dot_d = torch.sum(molecule.velocities[mol_index] * dvec)
        rad = v_dot_d * v_dot_d - 2.0 * (dE / CONSTANTS.KINETIC_ENERGY_SCALE) * d2_by_m
        if rad <= 0:
            return False
        sqrt_rad = torch.sqrt(rad)
        # choose solution with smaller |alpha|
        alpha = (-v_dot_d + torch.sign(v_dot_d) * sqrt_rad) / d2_by_m
        with torch.no_grad():
            molecule.velocities[mol_index] = molecule.velocities[mol_index] + (
                alpha * dvec * m_inv.unsqueeze(1)
            )
        return True

    def _recompute_active_force(self, molecule):
        if self._last_esdriver_args is None:
            raise RuntimeError("No cached electronic-structure call available for hop recompute.")
        es = self._last_esdriver_args
        active_state_scalar = self._uniform_active_state()
        if self._active_states is not None:
            molecule.active_state = self._active_states + 1
        elif active_state_scalar is not None:
            molecule.active_state = active_state_scalar + 1
        else:
            molecule.active_state = 0

        # Recompute forces on active surfaces; NACs are skipped here for efficiency.
        return self._compute_electronic_structure(
            molecule,
            learned_parameters=es["learned_parameters"],
            compute_nac=False,
            esdriver_args=es["esdriver_args"],
            **es["kwargs"],
        )

    def _after_electronic_update(
        self, molecule, state_energies, nac_matrix=None, nac_dot=None, step: Optional[int] = None
    ):
        nmol = state_energies.shape[0]
        device = molecule.coordinates.device
        self._ensure_active_states(nmol, device)

        # ---------------- Trivial crossing handling (NEXMD cross==2) ----------------
        swap_to = self._trivial_crossing_mask
        skip_hop_mask = torch.zeros((nmol,), dtype=torch.bool, device=device)
        skip_hop_mask |= self.post_hop_holdoff > 0
        active_crossed = False

        # swap_to: (nmol, nstates), with -1 where no perm was computed
        if swap_to is not None:
            swap_to = swap_to.to(device=device, dtype=torch.long)  # (nmol,n)
            has_swap = (swap_to >= 0).any(dim=1)

            if has_swap.any():
                if self._amp_phase is None:
                    raise RuntimeError("Electronic coefficients not initialized (amp_phase is None).")

                # Apply relabeling to electronic coefficients in one shot:
                # new_coeff[p(i)] = old_coeff[i]
                # Build perm = identity then perm[i]=swap_to[i] where defined
                n_states = swap_to.shape[1]
                perm = torch.arange(n_states, device=device).view(1, n_states).repeat(nmol, 1)
                defined = swap_to >= 0
                perm[defined] = swap_to[defined]

                sel = has_swap
                old = self._amp_phase[sel]
                p = perm[sel]
                out = old.clone()
                out.scatter_(dim=1, index=p.unsqueeze(-1).expand_as(old), src=old)
                self._amp_phase[sel] = out

                # Active relabel if active participates in swap
                ar = torch.arange(nmol, device=device)
                a = self._active_states
                a2 = perm[ar, a]

                active_swapped = sel & (a2 != a)
                if active_swapped.any():
                    active_crossed = True
                    # record prev active like ihopprev
                    self.prev_state[active_swapped] = a[active_swapped]
                    self._active_states[active_swapped] = a2[active_swapped]

                    # NEXMD: deterministic relabel => do not attempt stochastic hop this step
                    skip_hop_mask[active_swapped] = True

                    # NEXMD conthop2=1 => block hops for next ~2 steps
                    self.post_relabel_holdoff[active_swapped] = 2
                    self.post_hop_holdoff[active_swapped] = 2

        # ---------------- end trivial crossing handling ----------------

        hop_targets: List[Optional[int]]
        if nac_dot is not None:
            hop_targets = self._attempt_hop(
                state_energies, molecule, step=step if step is not None else self.step_offset
            )
        else:
            hop_targets = [None] * nmol

        # Suppress hop attempts for molecules whose active state had a trivial crossing
        if skip_hop_mask.any():
            for m in torch.where(skip_hop_mask)[0].tolist():
                hop_targets[m] = None

        if self._last_esdriver_args is None:
            # Fallback for analytic models (e.g., Tully) that don't call _compute_electronic_structure
            self._last_esdriver_args = {"learned_parameters": {}, "kwargs": {}, "esdriver_args": ()}

        hop_pairs = []
        for mol, target in enumerate(hop_targets):
            if target is None:
                continue
            hop_pairs.append((int(self._active_states[mol].item()) + 1, target + 1))
        if hop_pairs:
            unique_pairs = []
            seen_pairs = set()
            for pair in hop_pairs:
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                unique_pairs.append(pair)
            _ = self._compute_electronic_structure(
                molecule,
                learned_parameters=self._last_esdriver_args["learned_parameters"],
                compute_nac=True,
                nac_pairs=unique_pairs,
                **self._last_esdriver_args["kwargs"],
                esdriver_args=self._last_esdriver_args["esdriver_args"],
            )
            nac_matrix = self._cache_new.get("nac_vec")

        accepted_mask = torch.zeros((nmol,), dtype=torch.bool, device=device)

        for mol, target in enumerate(hop_targets):
            if target is None:
                continue
            exc_idx = int(self._active_states[mol].item())
            dE = float((state_energies[mol, target] - state_energies[mol, exc_idx]).item())
            success = self._rescale_velocity_along_nac(
                nac_matrix,
                exc_idx,
                target,
                molecule,
                dE,
                step if step is not None else self.step_offset,
                mol_index=mol,
            )
            if self._decohere_on_hop and self._amp_phase is None:
                raise RuntimeError("Electronic coefficients not initialized.")

            if success:
                self._active_states[mol] = target
                accepted_mask[mol] = True
                self.post_hop_holdoff[mol] = 2
                if self._decohere_on_hop:
                    self._amp_phase[mol].zero_()
                    self._amp_phase[mol, target, 0] = 1.0
                self.hop_log.append(
                    HopEvent(
                        step=step + 1 if step is not None else self.step_offset + 1,
                        from_state=exc_idx,
                        to_state=target,
                        accepted=True,
                        mol_index=mol,
                    )
                )
            else:
                # Hop rejected after rescale attempt
                if self._decohere_on_hop:
                    self._amp_phase[mol].zero_()
                    self._amp_phase[mol, exc_idx, 0] = 1.0
                self.hop_log.append(
                    HopEvent(
                        step=step + 1 if step is not None else self.step_offset + 1,
                        from_state=exc_idx,
                        to_state=target,
                        accepted=False,
                        mol_index=mol,
                        reason="Frustrated hop",
                    )
                )

        if (accepted_mask.any() or active_crossed) and self._recompute_on_hop:
            refreshed = self._recompute_active_force(molecule)
            if torch.is_tensor(refreshed):
                state_energies = refreshed
            nac_matrix = self._cache_new.get("nac_vec") if self._cache_new else nac_matrix

            idx = torch.arange(nmol, device=device)
            active_idx = self._active_states.to(device)
            self._current_potential = state_energies[idx, active_idx]
            with torch.no_grad():
                molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE

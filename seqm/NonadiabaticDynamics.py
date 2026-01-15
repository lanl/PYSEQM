import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from seqm.seqm_functions.rcis_batch import get_occ_virt, packone_batch

from .MolecularDynamics import CONSTANTS, Molecular_Dynamics_Basic
from .nac_utils import resolve_nac_config
from .seqm_functions.hcore import overlap_between_geometries

HBAR_EV_FS = 0.6582119514 # Planck's constant (reduced) in eV·fs

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


class NonadiabaticDynamicsBase(Molecular_Dynamics_Basic):
    """
    Base class for non-adiabatic dynamics over an excited-state manifold.

    Only excited states are included in the propagated electronic wavefunction
    (no ground-state component). Subclasses provide force definitions such as
    Ehrenfest mean-field.
    """

    def __init__(
        self,
        seqm_parameters: Dict,
        timestep: float = 0.5,
        Temp: float = 0.0,
        step_offset: int = 0,
        output: Optional[Dict] = None,
        electronic_substeps: int = 10,
        compute_nac: Optional[bool] = None,
        initial_state: int = 0,
        *args,
        **kwargs,
    ):
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
        nac_settings = resolve_nac_config(
            params,
            nroots=n_validate,
            default_enabled=True,
        )
        na_cfg["compute_nac"] = nac_settings.enabled
        if nac_settings.pairs:
            na_cfg["nac_states"] = nac_settings.pairs
        params["nonadiabatic"] = na_cfg
        super().__init__(
            params,
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
        self.initial_state = int(initial_state)
        max_electronic_dt = 0.05  # fs
        self._electronic_substeps = max(int(electronic_substeps), math.ceil(self.timestep / max_electronic_dt))
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
        self._crossing_overlap_thresh = float(params["nonadiabatic"].get("crossing_overlap_threshold", 0.9))
        self._apc_window = int(params["nonadiabatic"].get("apc_window", 2))
        self._use_mid_predict = params["nonadiabatic"].get("use_mid_predict", True)

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
        init_state = int(self.initial_state)
        if init_state >= self._nstates:
            raise ValueError(
                f"Initial state {init_state} >= available states ({self._nstates})."
            )

    def _init_coeffs(self, molecule):
        nmol = molecule.species.shape[0]
        device = molecule.coordinates.device
        if self._active_states is None or self._active_states.shape[0] != nmol:
            init_state = int(self.initial_state)
            self._active_states = torch.full(
                (nmol,),
                init_state,
                dtype=torch.long,
                device=device,
            )
        if self._amp_phase is not None and self._amp_phase.shape[0] == nmol:
            return
        amp_phase = torch.zeros(
            (nmol, self._nstates, 3),
            dtype=molecule.coordinates.dtype,
            device=device,
        )
        idx = torch.arange(nmol, device=device)
        amp_phase[idx, self._active_states, 0] = 1.0
        self._amp_phase = amp_phase

    def _uniform_active_state(self) -> Optional[int]:
        """Return scalar active state if all molecules share it."""
        if self._active_states is None:
            return int(self.initial_state)
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

    def _normalize_coeffs(self):
        if self._amp_phase is None:
            return
        amp_xy = self._amp_phase[..., :2]
        norm = torch.sqrt(torch.sum(amp_xy * amp_xy, dim=(1, 2), keepdim=True))
        norm = torch.clamp(norm, min=1e-12)
        self._amp_phase[..., :2] = amp_xy / norm

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

    def _build_state_energies(
        self,
        molecule,
        active_exc_index: Optional[int] = None,
    ) -> torch.Tensor:
        if self._nstates is None:
            raise RuntimeError("Number of states not initialized.")
        nmol = molecule.species.shape[0]
        dtype = molecule.coordinates.dtype
        device = molecule.coordinates.device
        if molecule.cis_energies is None:
            raise RuntimeError("Excited-state energies not available for nonadiabatic dynamics.")
        n_exc = min(molecule.cis_energies.shape[1], self._nstates)
        state_energies = torch.zeros((nmol, self._nstates), dtype=dtype, device=device)
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

    def _compute_electronic_structure(self, molecule, learned_parameters, compute_nac: bool = False, nac_pairs=None, **kwargs):
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
        _, nac_vec = self._compute_nac_matrix(molecule)
        self._cache_new = {
            "energies": energies.clone(),
            "nac_vec": None if nac_vec is None else nac_vec.clone(),
            "cis_amp": None if molecule.cis_amplitudes is None else molecule.cis_amplitudes.detach().clone(),
        }
        return energies

    def _compute_nac_matrix(self, molecule):
        if self._nstates is None:
            return None, None
        nac_vec = getattr(molecule, "nac", None)
        if nac_vec is None:
            return None, None
        nac_vec = nac_vec[:, : self._nstates, : self._nstates]
        return None, nac_vec

    @staticmethod
    def _permute_cache(cache, perm):
        if cache is None:
            return None
        idx = torch.as_tensor(perm, device=cache["energies"].device)
        new_cache = {
            "energies": cache["energies"][:, idx],
            "nac_vec": None if cache.get("nac_vec") is None else cache["nac_vec"][:, idx][:, :, idx],
            "cis_amp": None if cache.get("cis_amp") is None else cache["cis_amp"][:, idx, ...],
        }
        return new_cache

    @staticmethod
    def _hungarian_perm(cost):
        # cost: 2D torch tensor on CPU
        cost = cost.clone()
        n = cost.shape[0]
        u = torch.zeros(n)
        v = torch.zeros(n)
        p = torch.full((n + 1,), -1, dtype=torch.int64)
        way = torch.full((n + 1,), -1, dtype=torch.int64)
        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = torch.full((n + 1,), float("inf"))
            used = torch.zeros(n + 1, dtype=torch.bool)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float("inf")
                j1 = 0
                for j in range(1, n + 1):
                    if used[j]:
                        continue
                    cur = cost[i0 - 1, j - 1] - u[i0 - 1] - v[j - 1]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                for j in range(n + 1):
                    if used[j]:
                        u[p[j] - 1] += delta
                        v[j - 1] -= delta if j > 0 else 0.0
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == -1:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        perm = [-1] * n
        for j in range(1, n + 1):
            if p[j] != -1:
                perm[p[j] - 1] = j - 1

        if (min(perm) < 0) or (len(set(perm)) != len(perm)):
            # TODO: relax window and retry; for now raise error
            raise RuntimeError("Hungarian algorithm failed to find valid permutation.")
        return perm

    def _compute_perm_from_overlap(self, ref_amp, tgt_amp):
        # ref_amp, tgt_amp: (nmol, nstates, ncoeff)
        _, nstates, _ = ref_amp.shape
        ov = torch.square(torch.einsum("ia,ja->ij", ref_amp[0], tgt_amp[0]))
        ov = ov.to('cpu')
        # maximize overlap within window => minimize negative overlap
        big = 1e5
        cost = torch.full_like(ov, big)

        w = self._apc_window
        for i in range(nstates):
            j0 = max(0, i - w)
            j1 = min(nstates, i + w + 1)
            cost[i, j0:j1] = -ov[i, j0:j1]
        perm = NonadiabaticDynamicsBase._hungarian_perm(cost)
        return perm

    def _reorder_against_ref(self, cache_ref, cache_target):
        if cache_ref is None or cache_target is None:
            return cache_target, None
        ref_amp = cache_ref.get("cis_amp")
        tgt_amp = cache_target.get("cis_amp")
        if ref_amp is None or tgt_amp is None:
            return cache_target, None
        if ref_amp.shape[1] != tgt_amp.shape[1]:
            return cache_target, None
        perm = self._compute_perm_from_overlap(ref_amp, tgt_amp)
        return self._permute_cache(cache_target, perm), perm

    @staticmethod
    def _time_derivative_coupling(
        molecule,
        coords_prev,
        mos_prev,
        cis_prev,   # CIS: (nmol,nstates,nov) or RPA: (2,nmol,nstates,nov)
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

        if molecule.nocc.dim() != 1: # restricted closed-shell
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
            S_ao = overlap_between_geometries(
                molecule,
                molecule.coordinates.detach(),
                coords_prev.detach()
            )
            # S_ao has to be packed (hydrogen blocks with zero-padding have to be removed)  
            S_ao = packone_batch(S_ao, 4*molecule.nHeavy[0], molecule.nHydro[0], norb)

            # MO overlap: S_mo = C(t)^T S_ao(t,t-dt) C(t-dt)
            Cc = molecule.molecular_orbitals  # (nmol, nao, norb)
            Cp = mos_prev                     # (nmol, nao, norb)
            S_mo = Cc.transpose(1, 2) @ (S_ao @ Cp)  # (nmol, norb, norb)

            Soo = S_mo[:, :nocc, :nocc]      # (nmol, nocc, nocc)
            Svv = S_mo[:, nocc:, nocc:]      # (nmol, nvirt, nvirt)

            dSoo = Soo.transpose(1, 2) - Soo
            dSvv = Svv.transpose(1, 2) - Svv

            # helper: MO-derivative term on virtual block
            def mo_term_virtual(C_view):
                # C_view: (nmol, nstates, nocc, nvirt)
                Cd = torch.matmul(C_view, dSvv.transpose(1, 2).unsqueeze(1))      # (nmol, nstates, nocc, nvirt)
                Cf = C_view.reshape(nmol, C_view.shape[1], nov)      # (nmol, nstates, nov)
                Cdf = Cd.reshape(nmol, Cd.shape[1], nov)
                return torch.bmm(Cf, Cdf.transpose(1, 2))            # (nmol, nstates, nstates)

            # helper: MO-derivative term on occupied block
            def mo_term_occ(C_view):
                # apply dSoo^T on the occ index
                Ct = C_view.permute(0, 1, 3, 2)                      # (nmol, nstates, nvirt, nocc)
                Ctd = torch.matmul(Ct, dSoo.transpose(1, 2).unsqueeze(1))         # (nmol, nstates, nvirt, nocc)
                Cd = Ctd.permute(0, 1, 3, 2)                         # (nmol, nstates, nocc, nvirt)
                Cf = C_view.reshape(nmol, C_view.shape[1], nov)
                Cdf = Cd.reshape(nmol, Cd.shape[1], nov)
                return torch.bmm(Cf, Cdf.transpose(1, 2))

            if curr[0] == "cis":
                _, flat_p, view_p = prev
                _, flat_c, view_c = curr

                # CI-derivative term: <p|c> - <c|p>
                ov_pc = torch.bmm(flat_p, flat_c.transpose(1, 2))
                ov_cp = torch.bmm(flat_c, flat_p.transpose(1, 2))
                coup = (ov_pc - ov_cp)

                # MO-derivative terms
                coup = coup + mo_term_virtual(view_c) + mo_term_occ(view_c)

            else:
                _, (Xp_f, Yp_f), (Xp_v, Yp_v) = prev
                _, (Xc_f, Yc_f), (Xc_v, Yc_v) = curr

                # CI-derivative term in Fortran style:
                # (X+Y)^T(X+Y) + (X-Y)^T(X-Y)  antisymmetrized between steps
                Ap_p = (Xp_f + Yp_f)
                Ap_c = (Xc_f + Yc_f)
                Am_p = (Xp_f - Yp_f)
                Am_c = (Xc_f - Yc_f)

                ov_pc = torch.bmm(Ap_p, Ap_c.transpose(1, 2)) + torch.bmm(Am_p, Am_c.transpose(1, 2))
                ov_cp = torch.bmm(Ap_c, Ap_p.transpose(1, 2)) + torch.bmm(Am_c, Am_p.transpose(1, 2))
                coup = (ov_pc - ov_cp)

                # MO-derivative terms: +X part +Y part (note the + sign)
                coup = coup + mo_term_virtual(Xc_v) + mo_term_occ(Xc_v) + mo_term_virtual(Yc_v) + mo_term_occ(Yc_v)

            if enforce_antisym:
                coup = 0.5 * (coup - coup.transpose(1, 2))
                coup = coup - torch.diag_embed(torch.diagonal(coup, dim1=1, dim2=2))

            nac_dt = coup / (2.0 * dt)

        return nac_dt
    
    def _detect_crossings(self, cache_old, cache_new):
        if not self._detect_crossings_flag:
            return []
        if cache_old is None or cache_new is None:
            return []
        ref_amp = cache_old.get("cis_amp")
        tgt_amp = cache_new.get("cis_amp")
        if ref_amp is None or tgt_amp is None or ref_amp.shape != tgt_amp.shape:
            return []
        ov = torch.abs(torch.einsum("nia,nja->nij", ref_amp, tgt_amp))
        crossings = []
        n = ov.shape[1]
        # TODO: this is only comparing overlaps of off-diagonals, but not on-diagonals.
        # 1. Why is it checking if off-diagonal overlaps are lower than a threshold? (it should be higher for crossing right?)
        # 2. After detecting crossings it doesn't seem to affect any control flow. It just seems to be for printing a warning.
        # Check with NEXMD and fix
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if float(ov[0, i, j]) < self._crossing_overlap_thresh:
                    crossings.append((i, j, float(ov[0, i, j])))
        return crossings

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
            elif nd_old is not None:
                nac_dot = nd_old
            elif nd_new is not None:
                nac_dot = nd_new
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

        hop_int = torch.zeros((self._amp_phase.shape[0], self._nstates, self._nstates),
                              dtype=cache_old["energies"].dtype, device=self._amp_phase.device)

        for s in range(nsub):
            base_tau = s * dt_sub / dt_total
            tau_half = base_tau + 0.5 * dt_sub / dt_total
            tau_full = base_tau + dt_sub / dt_total
            # Integrating-factor form removes fast dynamical phase rotation.
            x0 = self._amp_phase[..., 0]
            y0 = self._amp_phase[..., 1]
            th0 = self._amp_phase[..., 2]

            # Stage 1
            e1, nd1 = interp(base_tau)
            dx1, dy1, dth1 = rhs(x0, y0, th0, e1, nd1)
            R1 = hop_numerator(x0, y0, th0, nd1)

            # Stage 2
            x2 = x0 + 0.5 * dt_sub * dx1
            y2 = y0 + 0.5 * dt_sub * dy1
            th2 = th0 + 0.5 * dt_sub * dth1
            e2, nd2 = interp(tau_half)
            dx2, dy2, dth2 = rhs(x2, y2, th2, e2, nd2)
            R2 = hop_numerator(x2, y2, th2, nd2)

            # Stage 3
            x3 = x0 + 0.5 * dt_sub * dx2
            y3 = y0 + 0.5 * dt_sub * dy2
            th3 = th0 + 0.5 * dt_sub * dth2
            e3, nd3 = interp(tau_half)
            dx3, dy3, dth3 = rhs(x3, y3, th3, e3, nd3)
            R3 = hop_numerator(x3, y3, th3, nd3)

            # Stage 4
            x4 = x0 + dt_sub * dx3
            y4 = y0 + dt_sub * dy3
            th4 = th0 + dt_sub * dth3
            e4, nd4 = interp(tau_full)
            dx4, dy4, dth4 = rhs(x4, y4, th4, e4, nd4)
            R4 = hop_numerator(x4, y4, th4, nd4)

            # RK4 coefficient update
            self._amp_phase[..., 0] = x0 + (dt_sub / 6.0) * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
            self._amp_phase[..., 1] = y0 + (dt_sub / 6.0) * (dy1 + 2 * dy2 + 2 * dy3 + dy4)
            self._amp_phase[..., 2] = th0 + (dt_sub / 6.0) * (dth1 + 2 * dth2 + 2 * dth3 + dth4)

            # RK4 hop integral update
            if R4 is not None: 
                hop_int = hop_int + (dt_sub / 6.0) * (R1 + 2.0 * R2 + 2.0 * R3 + R4)

        # self._normalize_coeffs()

        self._hop_integral = hop_int    

    def _thermo_potential(self, molecule):
        if self._current_potential is not None:
            return self._current_potential
        return super()._thermo_potential(molecule)

    def initialize(self, molecule, remove_com=None, learned_parameters=dict(), *args, **kwargs):
        self._setup_states(molecule)
        self._init_coeffs(molecule)
        molecule.active_state = self._active_states + 1  # excited-state index (1-based for downstream grad routines)
        super().initialize(
            molecule,
            remove_com=remove_com,
            learned_parameters=learned_parameters,
            *args,
            **kwargs,
        )
        # Recompute forces with full excited-state information
        state_energies = self._compute_electronic_structure(
            molecule, learned_parameters, compute_nac=True, **kwargs
        )
        # Initialize caches so old == new for the first step
        self._cache_old = {
            "energies": self._cache_new["energies"].clone(),
            "nac_dot": None if self._cache_new.get("nac_dot") is None else self._cache_new["nac_dot"].clone(),
            "nac_vec": None if self._cache_new["nac_vec"] is None else self._cache_new["nac_vec"].clone(),
            "cis_amp": None if self._cache_new.get("cis_amp") is None else self._cache_new["cis_amp"].clone(),
        }
        self._after_electronic_update(molecule, state_energies)
        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE

    def _after_electronic_update(self, molecule, state_energies, nac_matrix=None, nac_dot=None, step: Optional[int] = None):
        raise NotImplementedError

    def _do_integrator_step(self, i, molecule, learned_parameters, **kwargs):
        dt = self.timestep
        cache_old = self._cache_old or self._cache_new
        coords_prev = molecule.coordinates.detach().clone()
        mos_prev = getattr(molecule, "molecular_orbitals", None)
        if torch.is_tensor(mos_prev):
            mos_prev = mos_prev.detach().clone()
        with torch.no_grad():
            start_vel = molecule.velocities.detach().clone()

        # ---- Half kick + drift to t+dt ----
        with torch.no_grad():
            molecule.velocities.add_(0.5 * molecule.acc * dt)
            molecule.coordinates.add_(molecule.velocities * dt)

        _ = self._compute_electronic_structure(
            molecule, learned_parameters, compute_nac=False, **kwargs
        )

        # ---- Half kick to t+dt ----
        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE
            molecule.velocities.add_(0.5 * molecule.acc * dt)

        cache_new = self._cache_new
        nac_dt = self._time_derivative_coupling(
            molecule,
            coords_prev,
            mos_prev,
            cache_old.get("cis_amp") if cache_old is not None else None,
            cache_new.get("cis_amp"),
            dt,
        )
        # # TODO: Detect trivial crossings and reorder states if needed
        # # Not sure code below is correct for that
        # cache_new, perm_new = self._reorder_against_ref(cache_old, self._cache_new)
        # if perm_new:
        #     if molecule.all_forces is not None and molecule.all_forces.shape[1] >= self._nstates + 1:
        #         forces_exc = molecule.all_forces[:, 1 : 1 + self._nstates]
        #         forces_exc = forces_exc[:, perm_new]
        #         molecule.all_forces[:, 1 : 1 + self._nstates] = forces_exc

        # # TODO: Do trivial crossing detection
        # # Disable for now
        # crossings = self._detect_crossings(cache_old, cache_new)
        crossings = None

        def _attach_nac_dot(cache, vel):
            if cache is None or cache.get("nac_vec") is None:
                return cache
            v = vel.unsqueeze(1).unsqueeze(1)  # (nmol,1,1,molsize,3)
            nd = torch.sum(v * cache["nac_vec"], dim=(3, 4))
            cache = dict(cache)
            cache["nac_dot"] = nd
            return cache

        if nac_dt is None:
            cache_old = _attach_nac_dot(cache_old, start_vel) if cache_old is not None else None
            cache_new = _attach_nac_dot(cache_new, molecule.velocities.detach()) if cache_new is not None else None
        else:
            cache_new = dict(cache_new)
            cache_new["nac_dot"] = nac_dt

        self._propagate_electronic(cache_old, cache_new, substeps=self._electronic_substeps)
        self._after_electronic_update(
            molecule,
            state_energies=cache_new["energies"],
            nac_matrix=cache_new.get("nac_vec"),
            nac_dot=cache_new.get("nac_dot"),
            step=i + self.step_offset,
        )

        # shift caches for next step
        self._cache_old = {
            "energies": cache_new["energies"].clone(),
            "nac_dot": None if cache_new.get("nac_dot") is None else cache_new["nac_dot"].clone(),
            "nac_vec": None if cache_new.get("nac_vec") is None else cache_new["nac_vec"].clone(),
            "cis_amp": None if cache_new.get("cis_amp") is None else cache_new["cis_amp"].clone(),
        }
        if crossings:
            msg = "; ".join([f"{a}->{b} ov={ov:.2f}" for (a, b, ov) in crossings])
            print(f"[NonadiabaticDynamics] State overlap below threshold: {msg}")


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

    def _attempt_hop(self, state_energies, molecule, step: int):
        nmol = state_energies.shape[0]
        if self._hop_integral is None:
            return [None] * nmol
        if self._active_states is None or self._active_states.shape[0] != nmol:
            device = molecule.coordinates.device
            self._active_states = torch.full((nmol,), int(self.initial_state), dtype=torch.long, device=device)
        Ek_all = self._kinetic_energy(molecule)
        pop = self.populations
        hop_targets: List[Optional[int]] = [None] * nmol
        for mol in range(nmol):
            i_state = int(self._active_states[mol].item())
            denom = float(torch.clamp(pop[mol, i_state], min=1e-10))
            if denom <= 0.0:
                continue
            r = float(torch.rand(1, device=molecule.coordinates.device))
            cumulative = 0.0
            target: Optional[int] = None
            for j in range(self._nstates):
                if j == i_state:
                    continue
                gij = max(float(self._hop_integral[mol, i_state, j] / denom), 0.0)
                cumulative += gij
                if r < cumulative:
                    target = j
                    break
            if target is None:
                continue
            dE = float((state_energies[mol, target] - state_energies[mol, i_state]).item())  # eV
            Ek = float(Ek_all[mol].item())  # eV
            if dE > Ek + 1e-12:
                self.hop_log.append(
                    HopEvent(
                        step=step + 1,
                        from_state=i_state,
                        to_state=target,
                        accepted=False,
                        mol_index=mol,
                        reason="Insufficient kinetic energy",
                    )
                )
                continue
            hop_targets[mol] = target
        return hop_targets

    def _rescale_velocity_along_nac(self, nac_vec, i_state, j_state, molecule, dE, step, mol_index: int):
        if nac_vec is None:
            return False
        dvec = nac_vec[mol_index, i_state, j_state]
        mass = molecule.mass[mol_index]
        dnorm2 = torch.sum(mass * torch.sum(dvec * dvec, dim=1)).item()
        if dnorm2 < 1e-12:
            return False
        v_dot_d = torch.sum(mass * torch.sum(molecule.velocities[mol_index] * dvec, dim=1)).item()
        rad = v_dot_d * v_dot_d - 2.0 * (dE / CONSTANTS.KINETIC_ENERGY_SCALE) * dnorm2
        if rad < 0.0:
            self.hop_log.append(
                HopEvent(
                    step=step + 1,
                    from_state=i_state,
                    to_state=j_state,
                    accepted=False,
                    mol_index=mol_index,
                    reason="Frustrated hop (no real rescale)",
                )
            )
            return False
        sqrt_rad = math.sqrt(rad)
        alpha1 = (-v_dot_d + sqrt_rad) / dnorm2
        alpha2 = (-v_dot_d - sqrt_rad) / dnorm2
        alpha = alpha1 if abs(alpha1) < abs(alpha2) else alpha2
        with torch.no_grad():
            molecule.velocities[mol_index] = molecule.velocities[mol_index] + alpha * dvec
        return True

    def _recompute_active_force(self, molecule):
        if self._last_esdriver_args is None:
            raise RuntimeError("No cached electronic-structure call available for hop recompute.")
        es = self._last_esdriver_args
        if self._active_states is not None:
            molecule.active_state = self._active_states + 1
        else:
            active_state_scalar = self._uniform_active_state()
            molecule.active_state = (active_state_scalar + 1) if active_state_scalar is not None else 0
        self.esdriver(
            molecule,
            learned_parameters=es["learned_parameters"],
            P0=molecule.dm,
            dm_prop="SCF",
            cis_amp=molecule.cis_amplitudes,
            *es["esdriver_args"],
            **es["kwargs"],
        )
        # self._cache_new need not be updated since energies, nac_vec, etc. don't change. 
        # We only do esdriver to get the forces for the active state.

    def _after_electronic_update(
        self, molecule, state_energies, nac_matrix=None, nac_dot=None, step: Optional[int] = None
    ):
        nmol = state_energies.shape[0]
        device = molecule.coordinates.device
        if self._active_states is None or self._active_states.shape[0] != nmol:
            self._active_states = torch.full((nmol,), int(self.initial_state), dtype=torch.long, device=device)
        else:
            self._active_states = self._active_states.to(device)
        hop_targets: List[Optional[int]] = [None] * nmol
        if nac_dot is not None:
            hop_targets = self._attempt_hop(
                state_energies,
                molecule,
                step=step if step is not None else self.step_offset,
            )
        # TODO: Add decoherence correction (e.g., energy-based decoherence / A-FSSH).
        # Plain FSSH overcoheres when no hops occur; collapse-on-hop alone is not sufficient.

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
                step or self.step_offset,
                mol_index=mol,
            )
            if success:
                self._active_states[mol] = target
                if self._decohere_on_hop:
                    if self._amp_phase is None:
                        raise RuntimeError("Electronic coefficients not initialized.")
                    self._amp_phase[mol, :, :2].zero_()
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
                self.hop_log.append(
                    HopEvent(
                        step=step + 1 if step is not None else self.step_offset + 1,
                        from_state=exc_idx,
                        to_state=target,
                        accepted=False,
                        mol_index=mol,
                        reason="Frustrated hop (no NAC rescale)",
                    )
                )

        idx = torch.arange(nmol, device=device)
        active_idx = self._active_states.to(device)
        if molecule.all_forces is not None and molecule.all_forces.shape[1] >= self._nstates + 1:
            molecule.force = molecule.all_forces[idx, active_idx + 1]
        self._current_potential = state_energies[idx, active_idx]
        molecule.Etot = self._current_potential
        if molecule.force is not None:
            with torch.no_grad():
                molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE
        molecule.active_state = active_idx + 1

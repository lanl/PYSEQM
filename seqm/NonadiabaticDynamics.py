from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from scipy.optimize import linear_sum_assignment

from seqm.seqm_functions.rcis_batch import packone_batch

from .dynamics.nac_utils import resolve_nac_config
from .dynamics.tdc_hamiltonian_fd import compute_tdc_hamiltonian_fd
from .MolecularDynamics import CONSTANTS, Molecular_Dynamics_Langevin
from .seqm_functions.hcore import orthogonalized_overlap_between_geometries
from .seqm_functions.nac import calc_nac
from .seqm_functions.rcis_grad_batch import rcis_grad_batch

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
# Nuclear force is evaluated on the active adiabatic excited surface
# and stochastic hops are chosen via fewest-switches probabilities (FSSH).
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
    (no ground-state component).
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
        # Analytical gradients are required for active-surface FSSH forces.
        params.setdefault("analytical_gradient", [True])
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
        nac_settings = resolve_nac_config(params, nroots=n_validate, default_enabled=False)
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
        self._tdc_method = str(na_cfg.get("tdc_method", "hamiltonian_fd")).strip().lower()
        if self._tdc_method not in ("overlap", "hamiltonian_fd"):
            raise ValueError(
                f"Invalid nonadiabatic.tdc_method '{self._tdc_method}'. "
                "Supported methods: 'overlap', 'hamiltonian_fd'."
            )
        self._dtnact = 5e-5  # small dt for finite-diff, NEXMD uses 0.002 au
        self.initial_state = initial_state
        self._electronic_substeps: Optional[int] = None
        self._nstates: Optional[int] = None
        self._amp_phase: Optional[torch.Tensor] = None  # (nmol, nstates, 3): x, y, theta
        self._current_potential: Optional[torch.Tensor] = None
        self._active_states: Optional[torch.Tensor] = None  # (nmol,)
        self.hop_log: List[HopEvent] = []
        self._cache_old = None
        self._cache_new = None
        self._hop_integral = None
        self._decohere_on_hop = params["nonadiabatic"].get("decohere_on_hop", False)
        self._detect_crossings_flag = params["nonadiabatic"].get("detect_crossings", True)
        self._cache_prev_cis_amp = self._tdc_method == "overlap" or self._detect_crossings_flag
        self._trivial_crossing_mask: Optional[torch.Tensor] = None
        # Reusable per-device caches to avoid reallocations and CPU transfers each step
        self._eye_cache: Dict[tuple, torch.Tensor] = {}
        self._arange_cache: Dict[tuple, torch.Tensor] = {}
        self._trivial_zero_buffers: Dict[tuple, torch.Tensor] = {}
        self._trivial_swap_buffers: Dict[tuple, torch.Tensor] = {}
        self._perm_cost_buffers: Dict[tuple, torch.Tensor] = {}
        self._coords_prev: Optional[torch.Tensor] = None
        self._mos_prev: Optional[torch.Tensor] = None
        self._resume_state = None

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
        exc_cfg = self.seqm_parameters.get("excited_states")
        if not exc_cfg or "n_states" not in exc_cfg:
            raise RuntimeError(
                "Quantum dynamics requires seqm_parameters['excited_states']['n_states'] to be set."
            )
        base_nstates = exc_cfg.get("_nad_nstates", exc_cfg["n_states"])
        self._nstates = int(base_nstates)
        exc_cfg["_nad_nstates"] = self._nstates
        exc_cfg["n_states"] = self._nstates + 2
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
        idx = self._get_arange(nmol, device=device)
        amp_phase[idx, active, 0] = 1.0
        self._amp_phase = amp_phase

    def _ensure_active_states(self, nmol: int, device):
        """Ensure `_active_states` exists with correct shape/device."""
        if self._active_states is None or self._active_states.shape[0] != nmol:
            self._active_states = self._normalize_initial_state(nmol, device)
        else:
            self._active_states = self._active_states.to(device=device)
        return self._active_states

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

    @staticmethod
    def _copy_cache_entry(cache: Dict, key: str, src: torch.Tensor):
        if not torch.is_tensor(src):
            raise RuntimeError(f"Cannot cache missing tensor '{key}'.")
        src = src.detach()
        buf = cache.get(key)
        if buf is None or buf.shape != src.shape or buf.device != src.device or buf.dtype != src.dtype:
            buf = torch.empty_like(src)
        buf.copy_(src)
        cache[key] = buf

    def _coeffs_complex(self) -> torch.Tensor:
        if self._amp_phase is None:
            raise RuntimeError("Electronic coefficients not initialized.")
        x = self._amp_phase[..., 0]
        y = self._amp_phase[..., 1]
        theta = self._amp_phase[..., 2]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        real = x * cos_t - y * sin_t
        imag = x * sin_t + y * cos_t
        return torch.complex(real, imag)

    def _build_state_energies(self, molecule) -> torch.Tensor:
        return molecule.cis_energies[:, : self._nstates]

    def _current_cis_amplitudes(self, molecule) -> torch.Tensor:
        cis_amp = getattr(molecule, "cis_amplitudes", None)
        if not torch.is_tensor(cis_amp):
            raise RuntimeError("molecule.cis_amplitudes is required for nonadiabatic dynamics.")
        if cis_amp.dim() == 3:
            return cis_amp[:, : self._nstates]
        elif cis_amp.dim() == 4 and cis_amp.shape[0] == 2:
            return cis_amp[:, :, : self._nstates]
        else:
            raise RuntimeError(f"Unsupported cis_amplitudes shape {tuple(cis_amp.shape)}.")

    def _compute_electronic_structure(self, molecule, learned_parameters, **kwargs):
        # For FSSH we request gradients on the active excited state.
        old_state = molecule.active_state
        target_state = self._active_states + 1  # 1-based for excited-state gradients
        molecule.active_state = target_state
        esdriver_args = kwargs.pop("esdriver_args", ())
        try:
            self.esdriver(
                molecule,
                learned_parameters=learned_parameters,
                P0=molecule.dm,
                dm_prop="SCF",
                cis_amp=molecule.cis_amplitudes,
                *esdriver_args,
                **kwargs,
            )
        finally:
            molecule.active_state = old_state
        energies = self._build_state_energies(molecule)
        # Keep step-local references; previous-step snapshots are kept in _cache_old.
        cache_new = {"energies": energies, "cis_amp": self._current_cis_amplitudes(molecule)}
        self._cache_new = cache_new
        return energies

    @staticmethod
    def _hungarian_perm(cost) -> torch.Tensor:
        # cost: 2D torch tensor on CPU
        cost_np = cost.numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        perm = torch.full((cost_np.shape[0],), -1, dtype=torch.long)
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            perm[r] = c
        if (perm < 0).any() or torch.unique(perm).numel() != perm.numel():
            raise RuntimeError("Hungarian algorithm failed to find valid permutation.")
        return perm

    def _compute_perm_from_overlap(self, ref_or_ovlp, tgt_amp=None):
        """
        Compute optimal permutation per molecule using APC windowed cost.
        Accepts either:
          - ref_or_ovlp: overlap matrix (nmol,n,n) and tgt_amp=None
          - ref_or_ovlp: ref amplitudes, tgt_amp: target amplitudes (both nmol,nstates, ncoeff)
        """
        if tgt_amp is not None:
            ovlp = torch.square(torch.einsum("nia,nja->nij", ref_or_ovlp, tgt_amp))
        else:
            ovlp = torch.square(ref_or_ovlp)
        if ovlp.dim() != 3:
            raise ValueError("ovlp must have shape (nmol, nstates, nstates)")

        nmol, nstates, _ = ovlp.shape
        big = 1e5
        w = 2  # APC window (same as NEXMD); only allow permutations within ±2 of the diagonal
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
        if coords_prev is None or mos_prev is None:
            raise RuntimeError("Overlap TDC requires previous coordinates and molecular orbitals.")
        if cis_prev is None or cis_curr is None:
            raise RuntimeError("Overlap TDC requires previous and current CIS amplitudes.")

        if molecule.nocc.dim() != 1:  # restricted closed-shell
            raise RuntimeError("Overlap TDC currently supports restricted closed-shell only.")

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

            raise RuntimeError(f"Unsupported CIS amplitude shape {tuple(amp.shape)}.")

        prev = parse_amp(cis_prev)
        curr = parse_amp(cis_curr)

        with torch.no_grad():
            # AO overlap between current geometry (rows) and previous geometry (cols)
            # FIXME: this will fail for d-orbitals, will have to use packd
            def pack_ao(S):
                return packone_batch(S, 4 * molecule.nHeavy[0], molecule.nHydro[0], norb)

            S_ao = orthogonalized_overlap_between_geometries(
                molecule, molecule.coordinates.detach(), coords_prev.detach(), pack_fn=pack_ao
            )

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
                # Ap_p = Xp_f + Yp_f
                # Ap_c = Xc_f + Yc_f
                # Am_p = Xp_f - Yp_f
                # Am_c = Xc_f - Yc_f

                ov_pc = torch.bmm(Xp_f, Xc_f.transpose(1, 2)) + torch.bmm(Yp_f, Yc_f.transpose(1, 2))
                ov_cp = torch.bmm(Xc_f, Xp_f.transpose(1, 2)) + torch.bmm(Yc_f, Yp_f.transpose(1, 2))
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

        if ref_amp is None or tgt_amp is None:
            raise RuntimeError("Trivial-crossing detection requires previous and current CIS amplitudes.")
        if ref_amp.shape != tgt_amp.shape:
            raise RuntimeError(
                f"CIS amplitude shape mismatch for crossing detection: {tuple(ref_amp.shape)} vs {tuple(tgt_amp.shape)}."
            )

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
        w = 2  # APC window (same as NEXMD); only allow permutations within ±2 of the diagonal
        i = diag_idx.view(1, n_states, 1)
        j = diag_idx.view(1, 1, n_states)
        in_win = (j >= (i - w)) & (j <= (i + w))  # (1, n, n)
        ov_win = overlap.masked_fill(~in_win, 0.0)
        # prefilter: if no off-diagonal entry reaches thr, trivial crossing can never occur.
        diag_mask = self._get_eye(n_states, device=device, dtype=torch.bool).unsqueeze(0)
        has_strong_offdiag = (ov_win.masked_fill(diag_mask, 0.0).max(dim=2).values >= thr).any(
            dim=1
        )  # (nmol,)

        # "Holdoff" prevents expensive crossing detection for molecules right after a hop
        holdoff = self.post_hop_holdoff > 0
        active = self._active_states  # (nmol,)
        mol_ar = self._get_arange(nmol, device=device)
        active_row = ov_win[mol_ar, active]  # (nmol, n)
        active_mask = diag_idx.view(1, -1) == active.view(-1, 1)
        active_has_partner = active_row.masked_fill(active_mask, 0.0).max(dim=1).values >= thr  # (nmol,)

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

    def _propagate_electronic(self, cache_old, cache_new, substeps=None):
        if self._amp_phase is None:
            raise RuntimeError("Electronic coefficients not initialized before propagation.")

        energies_old = cache_old.get("energies")
        energies_new = cache_new.get("energies")
        if not torch.is_tensor(energies_old) or not torch.is_tensor(energies_new):
            raise RuntimeError("Both cache_old['energies'] and cache_new['energies'] must be tensors.")

        nd_new = cache_new.get("nac_dot")
        if not torch.is_tensor(nd_new):
            raise RuntimeError("cache_new['nac_dot'] is required for electronic propagation.")

        nd_old = cache_old.get("nac_dot")
        if not torch.is_tensor(nd_old):
            # First nuclear step only. Treat NAC as constant over the step.
            nd_old = nd_new

        amp = self._amp_phase
        device = amp.device
        dtype = amp.dtype

        dt_total = self.timestep

        # excitation energies
        e0 = energies_old
        e1 = energies_new
        de = e1 - e0

        dnd = nd_new - nd_old

        # ------------------------------------------------------------------
        # Cheap adaptive nsub: default baseline, increase only for NAC spikes.
        # ------------------------------------------------------------------
        if substeps is None:
            base = 8
            nmax = 80
            eta_nac = 0.25

            # Dimensionless NAC severity over the nuclear step.
            # chi = dt * max(max |D|, max |Delta D|)
            # With dt = 0.1 fs and D = 100 fs^-1, chi = 10.
            dmax = torch.maximum(torch.abs(nd_old), torch.abs(nd_new)).amax()
            djump = torch.abs(dnd).amax()
            chi = dt_total * torch.maximum(dmax, djump)

            # Soft spike response:
            #
            # nsub = base                                if chi <= trigger
            # nsub = base + ceil((chi-trigger)/eta_nac)  otherwise
            extra = torch.ceil(torch.clamp((chi - 1.0) / eta_nac, min=0.0))
            nsub = base + int(extra.to(torch.int64).item())
            nsub = min(nsub, nmax)
        else:
            nsub = int(substeps)

        inv_nsub = 1.0 / float(nsub)
        dt_sub = dt_total * inv_nsub
        half_dt_sub = 0.5 * dt_sub
        dt_over_hbar = dt_sub / HBAR_EV_FS
        half_dt_over_hbar = 0.5 * dt_over_hbar
        one_sixth_dt = dt_sub / 6.0

        x = amp[..., 0]
        y = amp[..., 1]
        th = amp[..., 2]

        def rhs_amp(xr, yi, theta, nac_proj):
            """
            Interaction-picture RHS, using real arithmetic.

            a_i = x_i + i y_i
            u_i = a_i exp(i theta_i)

            da_i/dt = - exp(-i theta_i) sum_j D_ij u_j
            """

            ct = torch.cos(theta)
            st = torch.sin(theta)

            # u = (x + i y) exp(i theta)
            u_re = xr * ct - yi * st
            u_im = xr * st + yi * ct

            # r = D @ u, D is real.
            r_re = torch.bmm(nac_proj, u_re.unsqueeze(-1)).squeeze(-1)
            r_im = torch.bmm(nac_proj, u_im.unsqueeze(-1)).squeeze(-1)

            # -exp(-i theta) r
            dx = -(ct * r_re + st * r_im)
            dy = st * r_re - ct * r_im

            return dx, dy

        # ------------------------------------------------------------------
        # RK4 loop
        # ------------------------------------------------------------------
        for s in range(nsub):
            tau = s * inv_nsub
            tau_half = tau + 0.5 * inv_nsub
            tau_full = tau + inv_nsub

            e1s = e0 + tau * de
            e2s = e0 + tau_half * de

            nd1 = nd_old + tau * dnd
            nd2 = nd_old + tau_half * dnd
            nd4 = nd_old + tau_full * dnd

            # k1
            dx1, dy1 = rhs_amp(x, y, th, nd1)

            # k2
            x2 = x + half_dt_sub * dx1
            y2 = y + half_dt_sub * dy1
            th2 = th - half_dt_over_hbar * e1s
            dx2, dy2 = rhs_amp(x2, y2, th2, nd2)

            # k3
            x3 = x + half_dt_sub * dx2
            y3 = y + half_dt_sub * dy2
            th3 = th - half_dt_over_hbar * e2s
            dx3, dy3 = rhs_amp(x3, y3, th3, nd2)

            # k4
            x4 = x + dt_sub * dx3
            y4 = y + dt_sub * dy3
            th4 = th - dt_over_hbar * e2s
            dx4, dy4 = rhs_amp(x4, y4, th4, nd4)

            x = x + one_sixth_dt * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
            y = y + one_sixth_dt * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4)

            # Exact for linearly interpolated E over this substep.
            th = th - e2s * dt_over_hbar

        th = torch.remainder(th + torch.pi, 2.0 * torch.pi) - torch.pi

        amp[..., 0] = x
        amp[..., 1] = y
        amp[..., 2] = th

        # ------------------------------------------------------------------
        # Cheap final-time hop integral estimate.
        # ------------------------------------------------------------------
        ct = torch.cos(th)
        st = torch.sin(th)

        u_re = x * ct - y * st
        u_im = x * st + y * ct

        hop_int = u_re.unsqueeze(2) * u_re.unsqueeze(1) + u_im.unsqueeze(2) * u_im.unsqueeze(1)

        hop_int.mul_(nd_new)
        hop_int.mul_(2.0 * dt_total)

        eye = self._get_eye(self._nstates, device=device, dtype=dtype).unsqueeze(0)
        hop_int.mul_(1.0 - eye)

        self._hop_integral = hop_int

    def _thermo_potential(self, molecule):
        if self._current_potential is not None:
            return self._current_potential
        return super()._thermo_potential(molecule)

    def _apply_resume_state(self, molecule):
        state = getattr(self, "_resume_state", None)
        if not state:
            return
        device = molecule.coordinates.device
        amp_phase = state.get("amp_phase")
        if torch.is_tensor(amp_phase):
            self._amp_phase = amp_phase.to(device)
        active_states = state.get("active_states")
        if torch.is_tensor(active_states):
            self._active_states = active_states.to(device)
            molecule.active_state = self._active_states + 1
        for name in ("post_hop_holdoff", "prev_state"):
            val = state.get(name)
            if torch.is_tensor(val):
                setattr(self, name, val.to(device))
        current_potential = state.get("current_potential")
        if torch.is_tensor(current_potential):
            self._current_potential = current_potential.to(device)
        cache_old = state.get("cache_old")
        if isinstance(cache_old, dict):
            restored = dict(self._cache_old) if isinstance(self._cache_old, dict) else {}
            for key, val in cache_old.items():
                restored[key] = val.to(device) if torch.is_tensor(val) else val
            self._cache_old = restored
        self._resume_state = None

    def initialize(
        self, molecule, remove_com=None, learned_parameters=None, steps: Optional[int] = None, *args, **kwargs
    ):
        if learned_parameters is None:
            learned_parameters = {}
        self._setup_states(molecule)
        self._init_coeffs(molecule)
        molecule.active_state = (
            self._active_states + 1
        )  # excited-state index (1-based for downstream grad routines)

        # Initial energies and forces are calculated in parent initialize.
        super().initialize(
            molecule,
            remove_com=remove_com,
            learned_parameters=learned_parameters,
            steps=steps,
            *args,
            **kwargs,
        )
        self.esdriver.conservative_force.energy.namd = True
        excitation_energies = self._build_state_energies(molecule)
        cache_old = self._cache_old or {}
        self._copy_cache_entry(cache_old, "energies", excitation_energies)
        cache_new = self._cache_new if isinstance(self._cache_new, dict) else {}
        init_nac_dot = cache_new.get("nac_dot")
        if not torch.is_tensor(init_nac_dot):
            resume_state = getattr(self, "_resume_state", None) or {}
            resume_cache = resume_state.get("cache_old") if isinstance(resume_state, dict) else None
            if isinstance(resume_cache, dict):
                init_nac_dot = resume_cache.get("nac_dot")
        need_current_cis_amp = self._cache_prev_cis_amp or not torch.is_tensor(init_nac_dot)
        if need_current_cis_amp:
            current_cis_amp = self._current_cis_amplitudes(molecule)
        if self._cache_prev_cis_amp:
            self._copy_cache_entry(cache_old, "cis_amp", current_cis_amp)
        else:
            cache_old.pop("cis_amp", None)
        if torch.is_tensor(init_nac_dot):
            self._copy_cache_entry(cache_old, "nac_dot", init_nac_dot)
        self._cache_old = cache_old

        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE

        if "nac_dot" not in self._cache_old:
            init_cache = {"energies": self._cache_old.get("energies"), "cis_amp": current_cis_amp}
            vel_old = molecule.velocities.detach().clone()
            acc_old = molecule.acc.detach().clone()
            nd = compute_tdc_hamiltonian_fd(self, molecule, init_cache, learned_parameters, vel_old, acc_old)
            self._copy_cache_entry(self._cache_old, "nac_dot", nd)

        nmol = molecule.species.shape[0]
        device = molecule.coordinates.device
        self.post_hop_holdoff = torch.zeros(
            nmol, dtype=torch.int64, device=device
        )  # blocks crossing detection
        self.prev_state = torch.full((nmol,), -1, dtype=torch.long, device=device)  # like ihopprev

        if self.step_offset == 0 and self._h5_writer is not None:
            na_stride = self._h5_writer._write_nonadiabatic
            if na_stride > 0:
                active_states = self._active_states + 1
                amplitudes = self._coeffs_complex()
                self._h5_writer.append_nonadiabatic(
                    0, active_states=active_states, amplitudes=amplitudes, nac_dot=self._cache_old["nac_dot"]
                )

        self._apply_resume_state(molecule)

    def save_checkpoint(self, molecule, steps: int, reuse_P, remove_com, *, step_done: int, path: str):
        """Save checkpoint for restart (nonadiabatic dynamics)."""
        nad_state = {
            "amp_phase": self._tensor_cpu(self._amp_phase),
            "active_states": self._tensor_cpu(self._active_states),
            "post_hop_holdoff": self._tensor_cpu(getattr(self, "post_hop_holdoff", None)),
            "prev_state": self._tensor_cpu(getattr(self, "prev_state", None)),
            "current_potential": self._tensor_cpu(self._current_potential),
        }
        if isinstance(self._cache_old, dict):
            cache_old = {}
            nac_dot = self._cache_old.get("nac_dot")
            if torch.is_tensor(nac_dot):
                cache_old["nac_dot"] = self._tensor_cpu(nac_dot)
            if cache_old:
                nad_state["cache_old"] = cache_old

        ckpt = self._build_checkpoint_base(
            molecule, steps, reuse_P, remove_com, step_done=step_done, include_forces=True
        )
        mol_ckpt = ckpt["molecules"]
        if torch.is_tensor(getattr(molecule, "cis_energies", None)):
            mol_ckpt["cis_energies"] = self._tensor_cpu(molecule.cis_energies)
        ckpt.update(
            {
                "NAD_type": self.__class__.__name__,
                "nad_state": nad_state,
                "nad_nstates": (int(self._nstates) if self._nstates is not None else None),
            }
        )

        self._save_checkpoint_and_report(ckpt, path)

    @staticmethod
    def run_from_checkpoint(path: str, device=None):
        """Load and resume nonadiabatic dynamics from checkpoint."""
        ckpt, molecule, device, reuse_P = Molecular_Dynamics_Langevin._load_checkpoint_base(
            path, device=device
        )

        nad_type = ckpt.get("NAD_type")
        nad_classes = {"SurfaceHoppingDynamics": SurfaceHoppingDynamics}
        if nad_type not in nad_classes:
            raise RuntimeError(f"Unknown nonadiabatic dynamics type '{nad_type}' in checkpoint")

        nad_cls = nad_classes[nad_type]
        kwargs = Molecular_Dynamics_Langevin._checkpoint_init_kwargs(ckpt)
        if "damp" in ckpt:
            kwargs["damp"] = ckpt["damp"]
        saved_nstates = ckpt.get("nad_nstates")
        if saved_nstates is not None:
            seqm_params = kwargs["seqm_parameters"]
            exc_cfg = seqm_params.get("excited_states")
            if not isinstance(exc_cfg, dict):
                exc_cfg = {}
                seqm_params["excited_states"] = exc_cfg
            exc_cfg["_nad_nstates"] = int(saved_nstates)
            exc_cfg["n_states"] = int(saved_nstates)

        nad = nad_cls(**kwargs).to(device)

        resume_state = ckpt.get("nad_state", {})
        active_states = resume_state.get("active_states")
        if torch.is_tensor(active_states):
            nad._active_states = active_states.to(device)
        amp_phase = resume_state.get("amp_phase")
        if torch.is_tensor(amp_phase):
            nad._amp_phase = amp_phase.to(device)
        nad._resume_state = resume_state
        Molecular_Dynamics_Langevin._restore_rng(ckpt)

        nad.run(molecule=molecule, steps=ckpt["steps"], reuse_P=reuse_P, remove_com=ckpt["remove_com"])

    def _after_electronic_update(self, molecule, excitation_energies, step: Optional[int] = None):
        raise NotImplementedError

    def _print_hop_log(self):
        if not self.hop_log:
            print("No hops recorded.")
            return
        by_mol: Dict[int, List[HopEvent]] = {}
        for event in self.hop_log:
            mol = -1 if event.mol_index is None else int(event.mol_index)
            by_mol.setdefault(mol, []).append(event)

        for mol in sorted(by_mol):
            label = f"molecule {mol}" if mol >= 0 else "molecule ?"
            print(f"Hop events for {label}:")
            for event in by_mol[mol]:
                status = "accepted" if event.accepted else "frustrated"
                detail = f", reason={event.reason}" if event.reason else ""
                print(
                    f"  step {event.step:4d}: "
                    f"S{event.from_state + 1} -> S{event.to_state + 1} "
                    f"({status}{detail})"
                )

    def _do_integrator_step(self, i, molecule, learned_parameters, **kwargs):
        dt = self.timestep
        cache_old = self._cache_old or self._cache_new
        if not isinstance(cache_old, dict):
            raise RuntimeError("Electronic cache is not initialized before stepping dynamics.")

        coords_prev = None
        mos_prev = None
        if self._tdc_method == "hamiltonian_fd":
            vel_old = molecule.velocities.detach().clone()
            acc_old = molecule.acc.detach().clone()
        elif self._tdc_method == "overlap":
            coords_prev = self._coords_prev
            if coords_prev is None:
                coords_prev = torch.empty_like(molecule.coordinates)
                self._coords_prev = coords_prev
            coords_prev.copy_(molecule.coordinates.detach())

            mos_curr = getattr(molecule, "molecular_orbitals", None)
            if torch.is_tensor(mos_curr):
                if self._mos_prev is None:
                    self._mos_prev = torch.empty_like(mos_curr)
                self._mos_prev.copy_(mos_curr.detach())
                mos_prev = self._mos_prev

        if self.damp is not None:
            self._apply_langevin_thermostat(molecule)

        # ---- Half kick + drift to t+dt ----
        with torch.no_grad():
            molecule.velocities.add_(0.5 * molecule.acc * dt)
            molecule.coordinates.add_(molecule.velocities * dt)

        _ = self._compute_electronic_structure(molecule, learned_parameters, **kwargs)

        # ---- Half kick to t+dt ----
        with torch.no_grad():
            molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE
            molecule.velocities.add_(0.5 * molecule.acc * dt)

        if self.damp is not None:
            self._apply_langevin_thermostat(molecule)

        cache_new = self._cache_new
        if not isinstance(cache_new, dict):
            raise RuntimeError("Failed to build electronic cache for current step.")
        if not torch.is_tensor(cache_new.get("energies")):
            raise RuntimeError("Missing 'energies' in electronic cache for current step.")

        if torch.is_tensor(cache_new.get("nac_dot")):
            nac_dt = cache_new.get("nac_dot")
        elif self._tdc_method == "hamiltonian_fd":
            nac_dt = compute_tdc_hamiltonian_fd(
                self, molecule, cache_new, learned_parameters, vel_old, acc_old
            )
            if not self._cache_prev_cis_amp:
                cache_new.pop("cis_amp", None)
        elif self._tdc_method == "overlap":
            nac_dt = self._time_derivative_coupling(
                molecule, coords_prev, mos_prev, cache_old.get("cis_amp"), cache_new.get("cis_amp"), dt
            )
        else:
            raise RuntimeError(f"Unsupported TDC method '{self._tdc_method}'.")

        if torch.is_tensor(nac_dt):
            cache_new["nac_dot"] = nac_dt
        else:
            raise RuntimeError("Unable to compute time-derivative coupling")

        # update cooldown timers used for trivial crossing handling
        self.post_hop_holdoff = (self.post_hop_holdoff - 1).clamp(min=0)

        # Detect trivial crossings between previous and current ordering
        self._trivial_crossing_mask = self._detect_crossings(cache_old, cache_new)
        # self._trivial_crossing_mask: (n_mol, nstates) tensor of permutations for crossed mols

        self._propagate_electronic(cache_old, cache_new, substeps=self._electronic_substeps)

        self._after_electronic_update(
            molecule, excitation_energies=cache_new["energies"], step=i + self.step_offset
        )
        molecule.w = None

        if self._h5_writer:
            na_stride = self._h5_writer._write_nonadiabatic
            if na_stride > 0 and ((i + 1) % na_stride == 0):
                amplitudes = self._coeffs_complex()
                self._h5_writer.append_nonadiabatic(
                    i + 1,
                    active_states=self._active_states + 1,
                    amplitudes=amplitudes,
                    nac_dot=cache_new.get("nac_dot"),
                )

        # shift caches for next step
        cache_old = self._cache_old or {}
        self._copy_cache_entry(cache_old, "energies", cache_new["energies"])
        self._copy_cache_entry(cache_old, "nac_dot", cache_new["nac_dot"])
        if self._cache_prev_cis_amp:
            self._copy_cache_entry(cache_old, "cis_amp", cache_new["cis_amp"])
        else:
            cache_old.pop("cis_amp", None)
        self._cache_old = cache_old
        self._cache_new = None


class SurfaceHoppingDynamics(NonadiabaticDynamicsBase):
    """Fewest Switches Surface Hopping (FSSH) with simple energy rescaling."""

    def _attempt_hop(self) -> torch.Tensor:
        if self._active_states is None:
            raise RuntimeError("Active states are not initialized before hop attempts.")
        nmol = self._active_states.shape[0]
        device = self._active_states.device
        hop_targets = torch.full((nmol,), -1, dtype=torch.long, device=device)
        if self._hop_integral is None:
            return hop_targets
        active_states = self._active_states
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

        if has_hop.any():
            cmp_sub = cmp[has_hop]
            tgt = torch.argmax(cmp_sub.to(torch.long), dim=1)
            idx = arange[has_hop]
            hop_targets[idx] = tgt

        return hop_targets

    def _rescale_velocity_along_nac(self, nac_vec, i_state, j_state, molecule, dE, mol_index: int):
        if nac_vec is None:
            raise RuntimeError("NAC vectors are required for velocity rescaling on hops.")
        key = (i_state, j_state) if i_state < j_state else (j_state, i_state)
        pair_vec = nac_vec.get(key)
        if pair_vec is None:
            raise RuntimeError("NAC vectors are required for velocity rescaling on hops.")
        dvec = pair_vec[mol_index] if i_state < j_state else -pair_vec[mol_index]  # (molsize, 3)
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
        # alpha = -(v_dot_d + sqrt_rad) / d2_by_m  # Like NEXMD
        with torch.no_grad():
            molecule.velocities[mol_index] = molecule.velocities[mol_index] + (
                alpha * dvec * m_inv.unsqueeze(1)
            )
        return True

    def _recompute_active_force(self, molecule):
        molecule.active_state = self._active_states + 1

        # Recompute forces on active surfaces; NACs are skipped here for efficiency.
        grad_excited = rcis_grad_batch(
            molecule,
            molecule.w,
            molecule.e_mo,
            None,
            None,
            molecule.dm,
            self.esdriver.conservative_force.energy.excited_states["tolerance"],
            molecule._gam,
            self.esdriver.conservative_force.energy.method,
            molecule._parnuc,
            rpa=self.esdriver.conservative_force.energy.excited_states["method"] == "rpa",
            include_ground_state=True,
            orbital_window=self.esdriver.conservative_force.energy.excited_states.get("orbital_window", None),
            calculate_dipole=False,
        )
        with torch.no_grad():
            molecule.force = -grad_excited

    def _compute_NACR_for_hop(self, molecule, nac_pairs):
        cf = self.esdriver.conservative_force.energy
        pair_list = nac_pairs

        P = molecule.dm
        exc_amps = molecule.cis_amplitudes
        excitation_energies = molecule.cis_energies
        pair_nac = calc_nac(
            molecule,
            exc_amps,
            excitation_energies,
            P,
            None,
            None,
            pair_list,
            rpa=cf.excited_states["method"] == "rpa",
        )
        nac_vec = {}
        for pair_idx, (s1, s2) in enumerate(pair_list):
            nac_vec[(s1 - 1, s2 - 1)] = pair_nac[:, pair_idx]
        return nac_vec

    def _after_electronic_update(self, molecule, excitation_energies, step: Optional[int] = None):
        nmol = excitation_energies.shape[0]
        device = molecule.coordinates.device
        active_idx_ref = self._active_states.clone()
        current_step = step if step is not None else self.step_offset

        # ---------------- Trivial crossing handling (NEXMD cross==2) ----------------
        swap_to = self._trivial_crossing_mask
        skip_hop_mask = self.post_hop_holdoff > 0
        active_crossed = False

        # swap_to: (nmol, nstates), with -1 where no perm was computed
        if swap_to is not None:
            swap_to = swap_to.to(device=device, dtype=torch.long)  # (nmol,n)
            has_swap = (swap_to >= 0).any(dim=1)

            if has_swap.any():
                # Apply relabeling to electronic coefficients in one shot:
                # new_coeff[p(i)] = old_coeff[i]
                # Build perm = identity then perm[i]=swap_to[i] where defined
                n_states = swap_to.shape[1]
                perm = self._get_arange(n_states, device=device).view(1, n_states).repeat(nmol, 1)
                defined = swap_to >= 0
                perm[defined] = swap_to[defined]

                sel = has_swap
                old = self._amp_phase[sel]
                p = perm[sel]
                out = old.clone()
                out.scatter_(dim=1, index=p.unsqueeze(-1).expand_as(old), src=old)
                self._amp_phase[sel] = out

                # Active relabel if active participates in swap
                ar = self._get_arange(nmol, device=device)
                a = self._active_states
                a2 = perm[ar, a]

                active_swapped = sel & (a2 != a)
                if active_swapped.any():
                    active_crossed = True
                    swapped_idx = torch.nonzero(active_swapped, as_tuple=False).squeeze(1)
                    from_states = a[swapped_idx].tolist()
                    to_states = a2[swapped_idx].tolist()
                    # record prev active like ihopprev
                    self.prev_state[active_swapped] = a[active_swapped]
                    self._active_states[active_swapped] = a2[active_swapped]
                    for mol, from_state, to_state in zip(swapped_idx.tolist(), from_states, to_states):
                        self.hop_log.append(
                            HopEvent(
                                step=current_step + 1,
                                from_state=int(from_state),
                                to_state=int(to_state),
                                accepted=True,
                                mol_index=mol,
                                reason="Trivial crossing",
                            )
                        )

                    # NEXMD: deterministic relabel => do not attempt stochastic hop this step
                    skip_hop_mask[active_swapped] = True

                    # NEXMD conthop2=1 => block hops for next ~2 steps
                    self.post_hop_holdoff[active_swapped] = 2

        # ---------------- end trivial crossing handling ----------------

        hop_targets_t = self._attempt_hop()
        # Suppress hop attempts for molecules whose active state had a trivial crossing
        if skip_hop_mask.any():
            hop_targets_t = hop_targets_t.masked_fill(skip_hop_mask, -1)

        hop_idx = torch.nonzero(hop_targets_t >= 0, as_tuple=False).squeeze(1)
        if hop_idx.numel() > 0:
            hop_targets_sel = hop_targets_t[hop_idx]
            a = self._active_states[hop_idx] + 1
            b = hop_targets_sel + 1
            pair_tensor = torch.stack((torch.minimum(a, b), torch.maximum(a, b)), dim=1)
            unique_pairs = [(int(row[0]), int(row[1])) for row in torch.unique(pair_tensor, dim=0)]
            nac_matrix = self._compute_NACR_for_hop(molecule, unique_pairs)

        accepted_mask = torch.zeros((nmol,), dtype=torch.bool, device=device)

        hop_idx_list = hop_idx.tolist()
        for pos, mol in enumerate(hop_idx_list):
            target = int(hop_targets_sel[pos].item())
            exc_idx = int(self._active_states[mol].item())
            dE = float((excitation_energies[mol, target] - excitation_energies[mol, exc_idx]).item())
            success = self._rescale_velocity_along_nac(
                nac_matrix, exc_idx, target, molecule, dE, mol_index=mol
            )

            if success:
                self._active_states[mol] = target
                accepted_mask[mol] = True
                self.post_hop_holdoff[mol] = 2
                if self._decohere_on_hop:
                    self._amp_phase[mol].zero_()
                    self._amp_phase[mol, target, 0] = 1.0
                self.hop_log.append(
                    HopEvent(
                        step=current_step + 1,
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
                        step=current_step + 1,
                        from_state=exc_idx,
                        to_state=target,
                        accepted=False,
                        mol_index=mol,
                        reason="Frustrated hop",
                    )
                )

        if accepted_mask.any() or active_crossed:
            self._recompute_active_force(molecule)
            with torch.no_grad():
                molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE

        idx = self._get_arange(nmol, device=device)
        active_idx = self._active_states
        with torch.no_grad():
            e0 = molecule.Etot.reshape(nmol)
            e0 = e0 - excitation_energies[idx, active_idx_ref]
            self._current_potential = e0 + excitation_energies[idx, active_idx]
            molecule.Etot = self._current_potential

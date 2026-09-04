import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

import torch
from scipy.optimize import linear_sum_assignment

from seqm.seqm_functions.rcis_batch import _uniform_molecule_dimensions, packone_batch
from seqm.utils.torch_compile import optional_compile_function

from .dynamics.tdc_hamiltonian_fd import compute_tdc_hamiltonian_fd
from .MolecularDynamics import CONSTANTS, XL_ESMD, Molecular_Dynamics_Langevin
from .seqm_functions.hcore import (
    orthogonalized_overlap_from_matrices,
    overlap_between_geometries,
    overlap_matrix_current_geometry,
)
from .seqm_functions.nac import calc_nac
from .seqm_functions.rcis_grad_batch import rcis_grad_batch
from .seqm_functions.scf_loop import build_initial_density
from .seqm_functions.XLESMD import transport_mo_transition_amplitudes

HBAR_EV_FS = 0.6582119514  # Planck's constant (reduced) in eV·fs
_electronic_propagation_dispatch = None

_TERM_S0_S1_GAP = 1
_TERM_SCF_FAILED = 2
_TERM_CIS_FAILED = 3
_TERM_REASON_NAMES = {
    _TERM_S0_S1_GAP: "S0/S1 gap below threshold",
    _TERM_SCF_FAILED: "SCF did not converge",
    _TERM_CIS_FAILED: "CIS did not converge",
}


def enable_electronic_propagation_compile(mode=None, **options):
    """Compile only the guarded eight-substep electronic propagator."""
    global _electronic_propagation_dispatch

    compile_options = dict(options)
    if mode is not None:
        compile_options["mode"] = mode
    _electronic_propagation_dispatch = optional_compile_function(
        _electronic_propagation_kernel, compile_options=compile_options, label="namd.electronic_propagation"
    )


def _mo_term_virtual(C_view, dSvv, nmol: int, nov: int):
    Cd = torch.matmul(C_view, dSvv.transpose(1, 2).unsqueeze(1))
    Cf = C_view.reshape(nmol, C_view.shape[1], nov)
    Cdf = Cd.reshape(nmol, Cd.shape[1], nov)
    return torch.bmm(Cf, Cdf.transpose(1, 2))


def _mo_term_occ(C_view, dSoo, nmol: int, nov: int):
    Ct = C_view.permute(0, 1, 3, 2)
    Ctd = torch.matmul(Ct, dSoo.transpose(1, 2).unsqueeze(1))
    Cd = Ctd.permute(0, 1, 3, 2)
    Cf = C_view.reshape(nmol, C_view.shape[1], nov)
    Cdf = Cd.reshape(nmol, Cd.shape[1], nov)
    return torch.bmm(Cf, Cdf.transpose(1, 2))


def _electronic_rhs(xr, yi, theta, nac_proj):
    ct = torch.cos(theta)
    st = torch.sin(theta)

    u_re = xr * ct - yi * st
    u_im = xr * st + yi * ct

    r_re = torch.bmm(nac_proj, u_re.unsqueeze(-1)).squeeze(-1)
    r_im = torch.bmm(nac_proj, u_im.unsqueeze(-1)).squeeze(-1)

    dx = -(ct * r_re + st * r_im)
    dy = st * r_re - ct * r_im
    return dx, dy


def _electronic_propagation_kernel(amp, e0, e1, nd_old, nd_new, active, rows, dt_total: float, nsub: int):
    """Original rotating-frame RK4, with hop flux integrated over substeps."""
    de = e1 - e0
    dnd = nd_new - nd_old

    inv_nsub = 1.0 / nsub
    de_sub = de * inv_nsub
    dt_sub = dt_total * inv_nsub
    half_dt_sub = 0.5 * dt_sub
    dt_over_hbar = dt_sub / HBAR_EV_FS
    half_dt_over_hbar = 0.5 * dt_over_hbar
    one_sixth_dt = dt_sub / 6.0

    x = amp[..., 0]
    y = amp[..., 1]
    th = amp[..., 2]
    hop_int = torch.zeros_like(e0)

    for s in range(nsub):
        tau = s * inv_nsub
        tau_half = tau + 0.5 * inv_nsub
        tau_full = tau + inv_nsub

        e1s = e0 + tau * de
        e2s = e0 + tau_half * de

        nd1 = nd_old + tau * dnd
        nd2 = nd_old + tau_half * dnd
        nd4 = nd_old + tau_full * dnd

        dx1, dy1 = _electronic_rhs(x, y, th, nd1)

        x2 = x + half_dt_sub * dx1
        y2 = y + half_dt_sub * dy1
        th2 = th - half_dt_over_hbar * (e1s + 0.25 * de_sub)
        dx2, dy2 = _electronic_rhs(x2, y2, th2, nd2)

        x3 = x + half_dt_sub * dx2
        y3 = y + half_dt_sub * dy2
        dx3, dy3 = _electronic_rhs(x3, y3, th2, nd2)

        # Midpoint FSSH flux; averaging the two RK midpoint stages is cheap and
        # much better than using only the final coefficients for the full step.
        xm = 0.5 * (x2 + x3)
        ym = 0.5 * (y2 + y3)
        ct, st = torch.cos(th2), torch.sin(th2)
        cr = xm * ct - ym * st
        ci = xm * st + ym * ct
        pair_re = cr[rows, active].unsqueeze(1) * cr + ci[rows, active].unsqueeze(1) * ci
        hop_int.add_(pair_re * nd2[rows, active], alpha=2.0 * dt_sub)

        x4 = x + dt_sub * dx3
        y4 = y + dt_sub * dy3
        th4 = th - dt_over_hbar * e2s
        dx4, dy4 = _electronic_rhs(x4, y4, th4, nd4)

        x = x + one_sixth_dt * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
        y = y + one_sixth_dt * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4)
        th = th - e2s * dt_over_hbar

    th = torch.remainder(th + torch.pi, 2.0 * torch.pi) - torch.pi
    return torch.stack((x, y, th), dim=-1), hop_int


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
        method = str(params["method"]).upper()
        tdc_method = str(na_cfg.get("tdc_method", "hamiltonian_fd")).strip().lower()
        na_cfg["compute_nac"] = tdc_method == "nac_dot_v"
        params["nonadiabatic"] = na_cfg
        if method == "PM6" and tdc_method in ("overlap", "hamiltonian_fd"):
            raise NotImplementedError(f"nonadiabatic.tdc_method='{tdc_method}' is not implemented for PM6.")
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
        self._tdc_method = tdc_method
        if self._tdc_method not in ("overlap", "hamiltonian_fd", "nac_dot_v"):
            raise ValueError(
                f"Invalid nonadiabatic.tdc_method '{self._tdc_method}'. "
                "Supported methods: 'overlap', 'hamiltonian_fd', 'nac_dot_v'."
            )
        self._direct_nac_tdc = self._tdc_method == "nac_dot_v"
        self._dtnact = 5e-5  # small dt for finite-diff, NEXMD uses 0.002 au
        self.initial_state = initial_state

        self._electronic_integrator = str(na_cfg.get("electronic_integrator", "rk4")).strip().lower()
        if self._electronic_integrator not in {"unitary", "rk4"}:
            raise ValueError("nonadiabatic.electronic_integrator must be 'unitary' or 'rk4'.")
        fixed_substeps = na_cfg.get("electronic_substeps")
        self._electronic_substeps = None if fixed_substeps is None else int(fixed_substeps)
        self._electronic_error_target = float(na_cfg.get("electronic_error_target", 1e-4))
        self._electronic_substeps_max = int(na_cfg.get("electronic_substeps_max", 256))
        self._rk4_substeps_base = int(na_cfg.get("rk4_substeps_base", 8))
        self._rk4_substeps_max = int(na_cfg.get("rk4_substeps_max", 400))
        if self._electronic_substeps is not None and self._electronic_substeps < 1:
            raise ValueError("nonadiabatic.electronic_substeps must be positive.")
        if self._electronic_error_target <= 0.0 or self._electronic_substeps_max < 1:
            raise ValueError("Invalid electronic propagation tolerance/substep limit.")
        if self._rk4_substeps_base < 1 or self._rk4_substeps_max < self._rk4_substeps_base:
            raise ValueError("Invalid RK4 electronic substep limits.")

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
        self._electronic_buffers: Dict[tuple, torch.Tensor] = {}
        self._coords_prev: Optional[torch.Tensor] = None
        self._mos_prev: Optional[torch.Tensor] = None
        self._packed_overlap_prev: Optional[torch.Tensor] = None
        self._overlap_pack_spec: Optional[tuple] = None
        self._resume_state = None
        self._full_nac_pairs_1based = None
        self._full_nac_pair_keys = None
        self._full_nac_state_i = None
        self._full_nac_state_j = None
        term_cfg = dict(na_cfg.get("trajectory_termination", {}))
        self._termination_enabled = bool(term_cfg.get("enabled", True))
        self._s0_s1_threshold_ev = float(term_cfg.get("s0_s1_gap_ev", 0.2))
        if self._s0_s1_threshold_ev < 0.0:
            raise ValueError("nonadiabatic.trajectory_termination.s0_s1_gap_ev must be non-negative.")
        self._terminated_mask = None
        self._live_mask_cache = None
        self._termination_reason = None
        self._termination_step = None
        self._has_terminated = False
        self._terminate_run = False
        self._reset_cis_guess = False

    def _enable_torch_compile_if_requested(self, molecule):
        was_applied = self._torch_compile_applied
        super()._enable_torch_compile_if_requested(molecule)
        cfg = self._torch_compile_config
        if (
            was_applied
            or not cfg["enabled"]
            or getattr(self, "k", None) is not None
            or self._electronic_integrator != "rk4"
        ):
            return
        options = dict(cfg["options"])
        kernel_mode = options.pop("mode", None)
        enable_electronic_propagation_compile(mode=kernel_mode, **options)

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
        self._full_nac_pair_keys = [(i, j) for i in range(self._nstates) for j in range(i + 1, self._nstates)]
        self._full_nac_pairs_1based = [(i + 1, j + 1) for i, j in self._full_nac_pair_keys]
        if self._full_nac_pair_keys:
            self._full_nac_state_i, self._full_nac_state_j = zip(*self._full_nac_pair_keys)
        else:
            self._full_nac_state_i = ()
            self._full_nac_state_j = ()
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
        if arr is None:
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
            if fill_value == 0:
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
        if cis_amp.dim() == 4 and cis_amp.shape[0] == 2:
            return cis_amp[:, :, : self._nstates]
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
        if self._direct_nac_tdc:
            cache_new["nac_dot"] = self._nac_dot_from_vectors(molecule, molecule.nac)
        self._cache_new = cache_new
        return energies

    def _nac_pairs(self):
        return self._full_nac_pairs_1based

    def _select_nac_pairs(self, nac_vec, pair_list):
        if nac_vec is None:
            return None
        selected = {}
        for s1, s2 in pair_list:
            key = (s1 - 1, s2 - 1)
            pair = nac_vec.get(key)
            if pair is None:
                return None
            selected[key] = pair
        return selected

    def _nac_dot_from_vectors(self, molecule, nac_vec):
        if nac_vec is None:
            raise RuntimeError("molecule.nac is required for nonadiabatic.tdc_method='nac_dot_v'.")
        vel = molecule.velocities
        nmol = int(vel.shape[0])
        nac_dot = torch.zeros((nmol, self._nstates, self._nstates), dtype=vel.dtype, device=vel.device)
        if not self._full_nac_pair_keys:
            return nac_dot
        try:
            pair_stack = torch.stack([nac_vec[key] for key in self._full_nac_pair_keys], dim=1)
        except KeyError as exc:
            raise RuntimeError(
                "Full excited-state NAC vectors are required for nonadiabatic.tdc_method='nac_dot_v'."
            ) from exc
        proj = torch.sum(pair_stack * vel.unsqueeze(1), dim=(2, 3))
        nac_dot[:, self._full_nac_state_i, self._full_nac_state_j] = proj
        nac_dot[:, self._full_nac_state_j, self._full_nac_state_i] = -proj
        return nac_dot

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

    def _time_derivative_coupling(
        self,
        molecule,
        coords_prev,
        mos_prev,
        cis_prev,  # CIS: (nmol,nstates,nov) or RPA: (2,nmol,nstates,nov)
        cis_curr,
        dt,
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
        _, _, norb, nocc = _uniform_molecule_dimensions(molecule)
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

        def bad_diag_overlap(S, qmin):
            q = torch.diagonal(S.abs(), dim1=1, dim2=2).min(dim=1).values
            return q < qmin

        prev = parse_amp(cis_prev)
        curr = parse_amp(cis_curr)

        with torch.no_grad():
            coords_curr = molecule.coordinates.detach()

            S_curr = packone_batch(overlap_matrix_current_geometry(molecule), *self._overlap_pack_spec)
            S_cross = packone_batch(
                overlap_between_geometries(molecule, coords_curr, coords_prev), *self._overlap_pack_spec
            )
            S_prev = self._packed_overlap_prev

            S_ao = orthogonalized_overlap_from_matrices(S_curr, S_cross, S_prev)
            # S_ao = S_cross

            self._packed_overlap_prev = S_curr  # Save for next step

            # MO overlap: S_mo = C(t)^T S_ao(t,t-dt) C(t-dt)
            Cc = molecule.molecular_orbitals  # (nmol, nao, norb)
            Cp = mos_prev  # (nmol, nao, norb)
            S_mo = Cc.transpose(1, 2) @ (S_ao @ Cp)
            Soo, Svv = S_mo[:, :nocc, :nocc], S_mo[:, nocc:, nocc:]

            bad_mo_overlap = bad_diag_overlap(Soo, 0.75) | bad_diag_overlap(Svv, 0.75)
            if bad_mo_overlap.any():
                return None

            dSoo = Soo.transpose(1, 2) - Soo
            dSvv = Svv.transpose(1, 2) - Svv

            if curr[0] == "cis":
                _, flat_p, _ = prev
                _, flat_c, view_c = curr
                ov_pc = torch.bmm(flat_p, flat_c.transpose(1, 2))
                coup = ov_pc - ov_pc.transpose(-2, -1)
                coup = coup + _mo_term_virtual(view_c, dSvv, nmol, nov)
                coup = coup + _mo_term_occ(view_c, dSoo, nmol, nov)

            else:
                _, (Xp_f, Yp_f), _ = prev
                _, (Xc_f, Yc_f), (Xc_v, Yc_v) = curr

                ov_pc = torch.bmm(Xp_f, Xc_f.transpose(1, 2)) + torch.bmm(Yp_f, Yc_f.transpose(1, 2))
                ov_cp = torch.bmm(Xc_f, Xp_f.transpose(1, 2)) + torch.bmm(Yc_f, Yp_f.transpose(1, 2))
                coup = ov_pc - ov_cp
                coup = coup + _mo_term_virtual(Xc_v, dSvv, nmol, nov)
                coup = coup + _mo_term_occ(Xc_v, dSoo, nmol, nov)
                coup = coup + _mo_term_virtual(Yc_v, dSvv, nmol, nov)
                coup = coup + _mo_term_occ(Yc_v, dSoo, nmol, nov)

            if bad_diag_overlap(ov_pc, 0.75).any():
                return None

        return 0.25 * (coup - coup.transpose(1, 2)) / float(dt)

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
        live = self._live_mask()
        if live is not None:
            has_strong_offdiag &= live
            active_has_partner &= live

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

    @torch.no_grad()
    def _unitary_substeps(self, e0, e1, d0, d1, dt_total: float) -> int:
        """Cheap batch-wide power-of-two estimate for rotating-frame midpoint propagation."""
        b, n = e0.shape
        device, dtype = e0.device, e0.dtype
        key = (b, n, device, dtype)
        buffers = self._electronic_buffers
        a = self._get_tensor(buffers, ("pred_a",) + key, (b, n, n), device, dtype)
        w = self._get_tensor(buffers, ("pred_w",) + key, (b, n, n), device, dtype)
        t = self._get_tensor(buffers, ("pred_t",) + key, (b, n, n), device, dtype)

        # Non-commuting change of the real antisymmetric coupling matrix.
        torch.bmm(d0, d1, out=a)
        torch.bmm(d1, d0, out=t)
        a.sub_(t).abs_()
        eta_comm = a.amax() * (dt_total * dt_total / 12.0)

        # Interaction-picture phase variation, weighted by the coupling itself.
        torch.sub(e0.unsqueeze(2), e0.unsqueeze(1), out=w)
        w.abs_()
        torch.sub(e1.unsqueeze(2), e1.unsqueeze(1), out=t)
        t.abs_()
        torch.maximum(w, t, out=w)
        w.mul_(dt_total / HBAR_EV_FS).square_()

        a.copy_(d0).abs_()
        t.copy_(d1).abs_()
        torch.maximum(a, t, out=a)
        w.mul_(a).mul_(dt_total / 24.0)

        eta = float(torch.maximum(eta_comm, w.amax()).item())
        nreq = max(4, math.ceil(math.sqrt(eta / self._electronic_error_target)))
        nsub = 1 << (nreq - 1).bit_length()
        return min(nsub, self._electronic_substeps_max)

    def _propagate_unitary(self, amp, e0, e1, d0, d1, active, rows, dt_total: float, nsub: int):
        """Rotating-frame midpoint unitary propagation with a real antisymmetric exponential."""
        b, n = e0.shape
        device, dtype = e0.device, e0.dtype
        key = (b, n, device, dtype)
        buffers = self._electronic_buffers

        de = self._get_tensor(buffers, ("de",) + key, (b, n), device, dtype)
        e = self._get_tensor(buffers, ("e",) + key, (b, n), device, dtype)
        dd = self._get_tensor(buffers, ("dd",) + key, (b, n, n), device, dtype)
        dm = self._get_tensor(buffers, ("dm",) + key, (b, n, n), device, dtype)
        z = self._get_tensor(buffers, ("z",) + key, (b, n, n), device, dtype)
        thm = self._get_tensor(buffers, ("thm",) + key, (b, n), device, dtype)
        ct = self._get_tensor(buffers, ("ct",) + key, (b, n), device, dtype)
        st = self._get_tensor(buffers, ("st",) + key, (b, n), device, dtype)
        tmp = self._get_tensor(buffers, ("tmp_ri",) + key, (b, n, 2), device, dtype)
        mid = self._get_tensor(buffers, ("mid_ri",) + key, (b, n, 2), device, dtype)
        pair = self._get_tensor(buffers, ("pair",) + key, (b, n), device, dtype)
        hop = self._get_tensor(buffers, ("hop",) + key, (b, n), device, dtype, fill_value=0)

        x, y, theta = amp.unbind(dim=-1)
        inv = 1.0 / nsub
        h = dt_total * inv
        half_h = 0.5 * h
        inv_hbar = 1.0 / HBAR_EV_FS

        torch.sub(e1, e0, out=de).mul_(inv)
        e.copy_(e0)
        torch.sub(d1, d0, out=dd).mul_(inv)
        dm.copy_(d0).add_(dd, alpha=0.5)

        tr, ti = tmp.unbind(dim=-1)
        mr, mi = mid.unbind(dim=-1)

        for _ in range(nsub):
            # Exact diagonal-energy phase to the substep midpoint for linear E(t).
            thm.copy_(theta).add_(e, alpha=-half_h * inv_hbar)
            thm.add_(de, alpha=-0.25 * half_h * inv_hbar)
            torch.cos(thm, out=ct)
            torch.sin(thm, out=st)

            # P_m a_n, kept as two real channels.
            tr.copy_(x).mul_(ct).addcmul_(y, st, value=-1.0)
            ti.copy_(x).mul_(st).addcmul_(y, ct)

            # exp[-h D_m / 2] is real orthogonal because D_m is real antisymmetric.
            z.copy_(dm).mul_(-half_h)
            rhalf = torch.matrix_exp(z)
            torch.bmm(rhalf, tmp, out=mid)

            # Physical midpoint coefficients are exactly the rotated mid tensor.
            pair.copy_(mr).mul_(mr[rows, active].unsqueeze(1))
            pair.addcmul_(mi, mi[rows, active].unsqueeze(1))
            hop.addcmul_(pair, dm[rows, active], value=2.0 * h)

            # Second identical half-step, then rotate back to interaction-picture amplitudes.
            torch.bmm(rhalf, mid, out=tmp)
            x.copy_(tr).mul_(ct).addcmul_(ti, st)
            y.copy_(ti).mul_(ct).addcmul_(tr, st, value=-1.0)

            theta.add_(e, alpha=-h * inv_hbar).add_(de, alpha=-0.5 * h * inv_hbar)
            e.add_(de)
            dm.add_(dd)

        theta.add_(torch.pi).remainder_(2.0 * torch.pi).sub_(torch.pi)
        return hop

    def _propagate_electronic(self, cache_old, cache_new, substeps=None):
        amp = self._amp_phase
        e0, e1 = cache_old["energies"], cache_new["energies"]
        d1 = cache_new["nac_dot"]
        d0 = cache_old.get("nac_dot", d1)
        dt_total = float(self.timestep)
        live = self._live_mask()

        if live is None:
            amp_work, active = amp, self._active_states
        else:
            amp_work = amp[live]
            e0, e1, d0, d1 = e0[live], e1[live], d0[live], d1[live]
            active = self._active_states[live]
        rows = self._get_arange(active.shape[0], device=amp.device)

        if self._electronic_integrator == "unitary":
            nsub = int(substeps) if substeps is not None else self._unitary_substeps(e0, e1, d0, d1, dt_total)
            hop_work = self._propagate_unitary(amp_work, e0, e1, d0, d1, active, rows, dt_total, nsub)
        else:
            if substeps is None:
                dmax = torch.maximum(d0.abs(), d1.abs()).amax()
                djump = (d1 - d0).abs().amax()
                chi = dt_total * torch.maximum(dmax, djump)
                extra = int(torch.ceil(torch.clamp((chi - 1.0) / 0.25, min=0.0)).item())
                nsub = min(self._rk4_substeps_base + extra, self._rk4_substeps_max)
            else:
                nsub = int(substeps)
            propagate = (
                _electronic_propagation_dispatch
                if nsub == 8 and _electronic_propagation_dispatch is not None
                else _electronic_propagation_kernel
            )
            amp_new, hop_work = propagate(amp_work, e0, e1, d0, d1, active, rows, dt_total, nsub)
            amp_work.copy_(amp_new)

        if live is None:
            self._hop_integral = hop_work
        else:
            amp[live] = amp_work
            hop_shape = tuple(cache_new["nac_dot"].shape[:2])
            hop = self._get_tensor(
                self._electronic_buffers,
                ("hop_full",) + hop_shape + (amp.device, amp.dtype),
                hop_shape,
                amp.device,
                amp.dtype,
                fill_value=0,
            )
            hop[live] = hop_work
            self._hop_integral = hop

    def _init_termination_state(self, molecule):
        if not self._termination_enabled:
            return
        self._terminate_run = False
        nmol = molecule.species.shape[0]
        device = molecule.coordinates.device
        if not torch.is_tensor(getattr(self.esdriver, "notconverged", None)):
            self.esdriver.notconverged = torch.tensor(False, device=device)
        if not torch.is_tensor(getattr(molecule, "cis_converged", None)):
            molecule.cis_converged = torch.tensor(True, device=device)
        if self._terminated_mask is None or self._terminated_mask.shape[0] != nmol:
            self._terminated_mask = torch.zeros(nmol, dtype=torch.bool, device=device)
            self._termination_reason = torch.zeros(nmol, dtype=torch.int8, device=device)
            self._termination_step = torch.full((nmol,), -1, dtype=torch.long, device=device)
        else:
            self._terminated_mask = self._terminated_mask.to(device=device, dtype=torch.bool)
            self._termination_reason = self._termination_reason.to(device=device, dtype=torch.int8)
            if self._termination_step is None:
                self._termination_step = torch.full((nmol,), -1, dtype=torch.long, device=device)
            else:
                self._termination_step = self._termination_step.to(device=device, dtype=torch.long)
        self._has_terminated = bool(self._terminated_mask.any().item())
        self._live_mask_cache = None
        if self._has_terminated:
            self._live_mask_cache = ~self._terminated_mask
            molecule._trajectory_live_mask = self._live_mask_cache

    def _live_mask(self):
        if getattr(self, "_termination_enabled", False) and getattr(self, "_has_terminated", False):
            return self._live_mask_cache
        return None

    def _freeze_terminated(self, molecule):
        with torch.no_grad():
            molecule.velocities[self._terminated_mask] = 0.0
            molecule.acc[self._terminated_mask] = 0.0

    def _update_termination(self, molecule, energies, coords_before, step):
        """Freeze terminal rows; failed electronic-structure rows return to the prior geometry."""
        if not self._termination_enabled:
            return
        scf_bad = self.esdriver.notconverged
        failed = scf_bad | ~molecule.cis_converged
        gap_stop = ~failed & (energies[:, 0] < self._s0_s1_threshold_ev)
        if self._has_terminated:
            failed &= self._live_mask_cache
            gap_stop &= self._live_mask_cache
        stopped = failed | gap_stop
        if not bool(stopped.any().item()):
            return
        scf_failed = failed & scf_bad
        with torch.no_grad():
            molecule.coordinates[failed] = coords_before[failed]
            if bool(scf_failed.any().item()):
                molecule.dm[scf_failed] = build_initial_density(molecule, molecule.dm)[scf_failed]
            if bool(failed.any().item()):
                self._reset_cis_guess = True
            self._terminated_mask[stopped] = True
            self._termination_step[stopped] = step
            self._termination_reason[gap_stop] = _TERM_S0_S1_GAP
            self._termination_reason[scf_failed] = _TERM_SCF_FAILED
            self._termination_reason[failed & ~scf_bad] = _TERM_CIS_FAILED
            self._live_mask_cache = ~self._terminated_mask
            molecule._trajectory_live_mask = self._live_mask_cache
        self._has_terminated = True
        self._terminate_run = not bool(self._live_mask_cache.any().item())
        self._freeze_terminated(molecule)

    def _print_termination_log(self):
        if not self._has_terminated:
            return
        print("Terminated trajectories:")
        for mol in torch.nonzero(self._terminated_mask, as_tuple=False).squeeze(1).tolist():
            reason = _TERM_REASON_NAMES.get(int(self._termination_reason[mol]), "unknown")
            print(f"  molecule {mol}: step {int(self._termination_step[mol])}, {reason}")

    def _apply_langevin_thermostat(self, molecule):
        super()._apply_langevin_thermostat(molecule)
        if self._termination_enabled and self._terminated_mask is not None and self._has_terminated:
            self._freeze_terminated(molecule)

    def _zero_com(self, molecule, **kwargs):
        live = self._live_mask()
        if live is None:
            return super()._zero_com(molecule, **kwargs)
        subset = SimpleNamespace(
            mass=molecule.mass[live],
            coordinates=molecule.coordinates[live],
            velocities=molecule.velocities[live],
        )
        super()._zero_com(subset, **kwargs)
        with torch.no_grad():
            if kwargs.get("translate_to_origin", False):
                molecule.coordinates[live] = subset.coordinates
            molecule.velocities[live] = subset.velocities

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
        self._coords_prev = None
        self._mos_prev = None
        self._packed_overlap_prev = None
        self._overlap_pack_spec = None
        self._setup_states(molecule)
        self._init_coeffs(molecule)
        self._init_termination_state(molecule)
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
        if self._termination_enabled:
            self._coords_prev = torch.empty_like(molecule.coordinates)
        self._mark_torch_compile_step(molecule)
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
        if not torch.is_tensor(init_nac_dot):
            if self._tdc_method == "overlap":
                if molecule.nocc.dim() != 1:
                    raise NotImplementedError("Overlap TDC currently supports restricted closed-shell only.")
                nHeavy, nHydro, norb, _ = _uniform_molecule_dimensions(molecule)
                self._overlap_pack_spec = (4 * nHeavy, nHydro, norb)
                if self._coords_prev is None:
                    self._coords_prev = torch.empty_like(molecule.coordinates)
                self._mos_prev = torch.empty_like(molecule.molecular_orbitals)
                self._packed_overlap_prev = packone_batch(
                    overlap_matrix_current_geometry(molecule), *self._overlap_pack_spec
                )
            elif self._tdc_method == "hamiltonian_fd":
                if molecule.nocc.dim() != 1:
                    raise NotImplementedError(
                        "hamiltonian_fd TD-NAC currently supports restricted closed-shell only."
                    )
                if not torch.is_tensor(current_cis_amp) or current_cis_amp.dim() != 3:
                    raise NotImplementedError("hamiltonian_fd TD-NAC currently supports CIS amplitudes only.")
            elif self._direct_nac_tdc:
                init_nac_dot = self._nac_dot_from_vectors(molecule, molecule.nac)
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
            if self._direct_nac_tdc:
                raise RuntimeError("nac_dot_v TD-NAC should have precomputed nac_dot in cache_old.")
            vel_old = molecule.velocities.detach().clone()
            acc_old = molecule.acc.detach().clone()
            nd = compute_tdc_hamiltonian_fd(
                self, molecule, init_cache, learned_parameters, vel_old, acc_old, validate=False
            )
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
        if self._has_terminated:
            molecule._trajectory_live_mask = self._live_mask_cache
            self._freeze_terminated(molecule)

    def save_checkpoint(self, molecule, steps: int, reuse_P, remove_com, *, step_done: int, path: str):
        """Save checkpoint for restart (nonadiabatic dynamics)."""
        nad_state = {
            "amp_phase": self._tensor_cpu(self._amp_phase),
            "active_states": self._tensor_cpu(self._active_states),
            "post_hop_holdoff": self._tensor_cpu(getattr(self, "post_hop_holdoff", None)),
            "prev_state": self._tensor_cpu(getattr(self, "prev_state", None)),
            "current_potential": self._tensor_cpu(self._current_potential),
        }
        if self._termination_enabled:
            nad_state.update(
                {
                    "terminated_mask": self._tensor_cpu(self._terminated_mask),
                    "termination_reason": self._tensor_cpu(self._termination_reason),
                    "termination_step": self._tensor_cpu(self._termination_step),
                }
            )
        if isinstance(self._cache_old, dict):
            cache_old = {}
            energies = self._cache_old.get("energies")
            if torch.is_tensor(energies):
                cache_old["energies"] = self._tensor_cpu(energies)
            cis_amp = self._cache_old.get("cis_amp")
            if torch.is_tensor(cis_amp):
                cache_old["cis_amp"] = self._tensor_cpu(cis_amp)
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
        for name in ("terminated_mask", "termination_reason", "termination_step"):
            value = resume_state.get(name)
            if torch.is_tensor(value):
                setattr(nad, f"_{name}", value.to(device))
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
        coords_before = self._coords_prev if self._termination_enabled else None
        if coords_before is not None:
            coords_before.copy_(molecule.coordinates.detach())

        coords_prev = None
        mos_prev = None
        if self._tdc_method in ("hamiltonian_fd", "overlap"):
            vel_old = molecule.velocities.detach().clone()
            acc_old = molecule.acc.detach().clone()
        if self._tdc_method == "overlap":
            coords_prev = self._coords_prev
            if not self._termination_enabled:
                coords_prev.copy_(molecule.coordinates.detach())

            self._mos_prev.copy_(molecule.molecular_orbitals.detach())
            mos_prev = self._mos_prev

        if self.damp is not None:
            self._apply_langevin_thermostat(molecule)

        # ---- Half kick + drift to t+dt ----
        with torch.no_grad():
            live = self._live_mask()
            if live is None:
                molecule.velocities.add_(0.5 * molecule.acc * dt)
                molecule.coordinates.add_(molecule.velocities * dt)
            else:
                molecule.velocities[live] += 0.5 * molecule.acc[live] * dt
                molecule.coordinates[live] += molecule.velocities[live] * dt

        _ = self._compute_electronic_structure(molecule, learned_parameters, **kwargs)

        cache_new = self._cache_new
        if not isinstance(cache_new, dict):
            raise RuntimeError("Failed to build electronic cache for current step.")
        if not torch.is_tensor(cache_new.get("energies")):
            raise RuntimeError("Missing 'energies' in electronic cache for current step.")

        self._update_termination(molecule, cache_new["energies"], coords_before, i + self.step_offset + 1)

        # ---- Half kick to t+dt (survivors only) ----
        with torch.no_grad():
            live = self._live_mask()
            if live is None:
                molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE
                molecule.velocities.add_(0.5 * molecule.acc * dt)
            else:
                molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE
                molecule.acc[~live] = 0.0
                molecule.velocities[live] += 0.5 * molecule.acc[live] * dt

        if self.damp is not None:
            self._apply_langevin_thermostat(molecule)

        if self._terminate_run:
            cache_old = self._cache_old or {}
            self._copy_cache_entry(cache_old, "energies", cache_new["energies"])
            self._cache_old = cache_old
            self._cache_new = None
            if self._reset_cis_guess:
                molecule.cis_amplitudes = None
                self._reset_cis_guess = False
            return

        if torch.is_tensor(cache_new.get("nac_dot")):
            nac_dt = cache_new.get("nac_dot")
        elif self._tdc_method == "hamiltonian_fd":
            nac_dt = compute_tdc_hamiltonian_fd(
                self, molecule, cache_new, learned_parameters, vel_old, acc_old, validate=False
            )
            if not self._cache_prev_cis_amp:
                cache_new.pop("cis_amp", None)
        elif self._tdc_method == "overlap":
            nac_dt = self._time_derivative_coupling(
                molecule, coords_prev, mos_prev, cache_old.get("cis_amp"), cache_new.get("cis_amp"), dt
            )
            if nac_dt is None:
                # print("Bad previous overlap")
                nac_dt = compute_tdc_hamiltonian_fd(
                    self,
                    molecule,
                    cache_new,
                    learned_parameters,
                    vel_old,
                    acc_old,
                    include_response_terms=True,
                    validate=True,
                )
                with torch.no_grad():
                    self._packed_overlap_prev = packone_batch(
                        overlap_matrix_current_geometry(molecule), *self._overlap_pack_spec
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
        if self._has_terminated:
            self._freeze_terminated(molecule)
        # molecule.w = None

        if self._h5_writer:
            na_stride = self._h5_writer._write_nonadiabatic
            if na_stride > 0 and ((i + 1) % na_stride == 0):
                amplitudes = self._coeffs_complex()
                self._h5_writer.append_nonadiabatic(
                    i + 1,
                    active_states=self._active_states + 1,
                    amplitudes=amplitudes,
                    nac_dot=cache_new.get("nac_dot"),
                    live_mask=self._live_mask(),
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
        if self._reset_cis_guess:
            molecule.cis_amplitudes = None
            self._reset_cis_guess = False


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
        # FSSH: g_ij = max(0, - Δa_ii / a_ii); _hop_integral already stores the active-state row.
        g_rows = torch.clamp(self._hop_integral / denom.unsqueeze(1), min=0.0)

        # Guard against dt so large that Σ_j g_ij > 1
        g_sum = g_rows.sum(dim=1, keepdim=True)
        g_rows = torch.where(g_sum > 1.0, g_rows / g_sum.clamp(min=1e-12), g_rows)

        # Cumulative draw per molecule
        cumsum = torch.cumsum(g_rows, dim=1)
        r = torch.rand(nmol, device=device)
        # Use a strict comparison so a zero-valued draw does not select a
        # zero-probability first bin and turn into a self-hop.
        cmp = cumsum > r.unsqueeze(1)
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
        if self._direct_nac_tdc:
            cached_nac = self._select_nac_pairs(molecule.nac, pair_list)
            if cached_nac is not None:
                return cached_nac

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
            include_response_terms=cf.nac_config.include_response_terms,
            w=molecule.w,
            e_mo=molecule.e_mo,
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
        live_mask = self._live_mask()

        # ---------------- Trivial crossing handling (NEXMD cross==2) ----------------
        swap_to = self._trivial_crossing_mask
        skip_hop_mask = self.post_hop_holdoff > 0
        active_crossed = False

        # swap_to: (nmol, nstates), with -1 where no perm was computed
        if swap_to is not None:
            swap_to = swap_to.to(device=device, dtype=torch.long)  # (nmol,n)
            has_swap = (swap_to >= 0).any(dim=1)
            if live_mask is not None:
                has_swap &= live_mask

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
                if self._hop_integral is not None:
                    hop_old = self._hop_integral[sel].clone()
                    self._hop_integral[sel].scatter_(1, p, hop_old)

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
        if live_mask is not None:
            hop_targets_t = hop_targets_t.masked_fill(~live_mask, -1)
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


class XLESurfaceHoppingDynamics(SurfaceHoppingDynamics):
    """Simple FSSH driven by ordered multi-state XL-ESMD shadow surfaces.

    The ground-state density and the complete ordered CIS auxiliary block are
    propagated with the same extended-Lagrangian recurrences used by
    :class:`XL_ESMD`.  The existing FSSH electronic propagator, hop selection,
    and NAC machinery then operate on the resulting shadow roots and energies.

    This first implementation deliberately treats the small non-orthogonality
    of the shadow roots as an approximation in both time-derivative and
    coordinate derivative couplings.  Trivial-crossing handling remains
    disabled, while instantaneous energy ordering is optional and enabled by
    default for this nonadiabatic driver.
    """

    _XL_COEFFICIENTS = {
        3: [1.69, 150e-3, -2.0, 3.0, 0.0, -1.0],
        4: [1.75, 57e-3, -3.0, 6.0, -2.0, -2.0, 1.0],
        5: [1.82, 18e-3, -6.0, 14.0, -8.0, -3.0, 4.0, -1.0],
        6: [1.84, 5.5e-3, -14.0, 36.0, -27.0, -2.0, 12.0, -6.0, 1.0],
        7: [1.86, 1.6e-3, -36.0, 99.0, -88.0, 11.0, 32.0, -25.0, 8.0, -1.0],
        8: [1.88, 0.44e-3, -99.0, 286.0, -286.0, 78.0, 78.0, -90.0, 42.0, -10.0, 1.0],
        9: [1.89, 0.12e-3, -286.0, 858.0, -936.0, 364.0, 168.0, -300.0, 184.0, -63.0, 12.0, -1.0],
    }

    def __init__(self, seqm_parameters: Dict, xl_bomd_params: Dict, *args, **kwargs):
        xl_params = dict(xl_bomd_params)
        constraint_mode = str(xl_params.get("constraint_mode", "ordered_linearized")).strip().lower()
        if constraint_mode != "ordered_linearized":
            raise ValueError("XL-FSSH currently requires constraint_mode='ordered_linearized'.")
        xl_params["constraint_mode"] = constraint_mode

        force_mode = str(xl_params.get("force_mode", "autodiff")).strip().lower()
        if force_mode not in {"autodiff", "experimental_analytic"}:
            raise ValueError("XL-FSSH force_mode must be 'autodiff' or 'experimental_analytic'.")

        params = dict(seqm_parameters)
        if isinstance(params.get("excited_states"), dict):
            params["excited_states"] = dict(params["excited_states"])
        na_cfg = dict(params.get("nonadiabatic", {}))
        na_cfg["detect_crossings"] = False
        params["nonadiabatic"] = na_cfg
        if force_mode == "autodiff":
            scf_backward = int(xl_params.get("scf_backward", params.get("scf_backward", 2)))
            if scf_backward not in {1, 2}:
                raise ValueError("Autodiff XL-FSSH requires scf_backward=1 or 2.")
            params["scf_backward"] = scf_backward
            params["analytical_gradient"] = [False]
        else:
            params["analytical_gradient"] = [True]

        super().__init__(seqm_parameters=params, *args, **kwargs)

        self.xl_bomd_params = xl_params
        self.xlesmd_force_mode = force_mode
        self.xlesmd_scf_backward = self.seqm_parameters.get("scf_backward", 0)
        self.xlesmd_energy_order_tracking = bool(xl_params.setdefault("energy_order_tracking", True))
        self.xlesmd_energy_order_events = []
        self.dmprop = "SCF"
        self.xlesmd_orthogonality_log = []

        self.k = int(xl_params["k"])
        if self.k not in self._XL_COEFFICIENTS:
            raise ValueError(
                f"Unsupported XL history order k={self.k}; expected one of {sorted(self._XL_COEFFICIENTS)}."
            )
        self.m = self.k + 1
        coeffs = self._XL_COEFFICIENTS[self.k]
        self.kappa = coeffs[0]
        self.alpha = coeffs[1]
        tmp = torch.as_tensor(coeffs[2:]) * self.alpha
        self.coeff_D = self.kappa
        tmp[0] += 2.0 - self.kappa
        tmp[1] -= 1.0
        self.register_buffer("coeff", tmp.repeat(2))

        self._xlesmd_step_index = 0
        self._xlesmd_coords_before = None
        self._xlesmd_mos_before = None
        self._xlesmd_S_prev = None
        self._xlesmd_learned_parameters = {}
        self._xl_ctx = None

    @staticmethod
    def _packed_current_ao_overlap(molecule):
        return XL_ESMD._packed_current_ao_overlap(molecule)

    def _record_xlesmd_orthogonality(self, molecule, step: int):
        xi = molecule.cis_amplitudes
        eta = self._xl_ctx["es_amp"]
        eye = torch.eye(xi.shape[1], dtype=xi.dtype, device=xi.device).unsqueeze(0)
        xi_error = xi @ xi.transpose(-1, -2) - eye
        eta_error = eta @ eta.transpose(-1, -2) - eye
        diag_mask = torch.eye(xi.shape[1], dtype=torch.bool, device=xi.device).unsqueeze(0)
        diagnostics = {
            "step": int(step),
            "xi_max_gram_error": float(xi_error.abs().amax().item()),
            "xi_max_norm_error": float(torch.diagonal(xi_error, dim1=-2, dim2=-1).abs().amax().item()),
            "xi_max_offdiag_overlap": float(xi_error.masked_fill(diag_mask, 0.0).abs().amax().item()),
            "eta_max_gram_error": float(eta_error.abs().amax().item()),
        }
        solver_diagnostics = getattr(molecule, "xlesmd_diagnostics", {})
        if "krylov_rank" in solver_diagnostics:
            diagnostics["krylov_rank"] = int(solver_diagnostics["krylov_rank"])
            diagnostics["krylov_max_relative_residual"] = float(
                solver_diagnostics["krylov_relative_residual"].amax().item()
            )
        if "krylov_history" in solver_diagnostics:
            diagnostics["krylov_history"] = {
                key: value.detach().cpu().tolist()
                for key, value in solver_diagnostics["krylov_history"].items()
            }
        molecule.xlesmd_nac_diagnostics = diagnostics
        self.xlesmd_orthogonality_log.append(diagnostics)

    def _propagate_xlesmd_auxiliaries(self, molecule):
        P = self._xl_ctx["P"]
        Pt = self._xl_ctx["Pt"]
        es_amp = self._xl_ctx["es_amp"]
        es_amp_t = self._xl_ctx["es_amp_t"]
        cindx = self._xlesmd_step_index % self.m

        with torch.no_grad():
            P = XL_ESMD._propagate_P(self, P, Pt, cindx, molecule)
            Pt[self.m - 1 - cindx] = P
            es_amp = XL_ESMD._propagate_excited_amp(self, es_amp, es_amp_t, cindx, molecule)
            es_amp_t[self.m - 1 - cindx] = es_amp
            P2 = P @ P
            P0 = torch.baddbmm(P2, P2, P, beta=1.5, alpha=-0.5)
        self._xl_ctx.update(P=P, Pt=Pt, es_amp=es_amp, es_amp_t=es_amp_t)
        return P0

    def _transport_xlesmd_history(self, molecule, coords_prev, mos_prev, S_prev):
        if not self.xl_bomd_params.get("transport_mo_auxiliary", True):
            return
        polar = bool(self.xl_bomd_params.get("polar_unitarize_mo_transport", True))
        es_amp = self._xl_ctx["es_amp"]
        es_amp_t = self._xl_ctx["es_amp_t"]
        with torch.no_grad():
            history_shape = es_amp_t.shape
            history_flat = es_amp_t.permute(1, 0, 2, 3).reshape(history_shape[1], -1, history_shape[-1])
            history_flat, S_curr = transport_mo_transition_amplitudes(
                molecule, history_flat, coords_prev, mos_prev, S_prev, polar_unitarize=polar
            )
            es_amp_t = history_flat.reshape(
                history_shape[1], history_shape[0], history_shape[2], history_shape[3]
            ).permute(1, 0, 2, 3)
            es_amp, _ = transport_mo_transition_amplitudes(
                molecule, es_amp, coords_prev, mos_prev, S_prev, polar_unitarize=polar
            )
        self._xl_ctx.update(es_amp=es_amp, es_amp_t=es_amp_t)
        self._xlesmd_S_prev = S_curr

    @staticmethod
    def _permute_state_rows(values, permutation):
        index = permutation
        for _ in range(values.dim() - 2):
            index = index.unsqueeze(-1)
        return values.gather(1, index.expand_as(values))

    @staticmethod
    def _permute_history_state_rows(values, permutation):
        index = permutation.unsqueeze(0)
        for _ in range(values.dim() - 3):
            index = index.unsqueeze(-1)
        return values.gather(2, index.expand_as(values))

    @torch.no_grad()
    def _apply_xlesmd_energy_order(self, molecule, step: int):
        """Relabel the XL state block and its history in ascending-energy order."""
        if not self.xlesmd_energy_order_tracking:
            return

        nroots = self._xl_ctx["es_amp"].shape[1]
        energies = molecule.cis_energies[:, :nroots]
        permutation = torch.argsort(energies, dim=1)
        identity = torch.arange(nroots, device=energies.device).expand_as(permutation)
        if torch.equal(permutation, identity):
            return

        for name in ("cis_energies", "cis_amplitudes", "transition_density_matrices", "dxi2dt2"):
            values = getattr(molecule, name, None)
            if values is not None:
                if not torch.is_tensor(values) or values.shape[1] != nroots:
                    raise RuntimeError(f"Cannot energy-order XL state tensor '{name}'.")
                setattr(molecule, name, self._permute_state_rows(values, permutation))

        multipliers = getattr(molecule, "xlesmd_multipliers", None)
        if torch.is_tensor(multipliers):
            if multipliers.shape[1:] != (nroots, nroots):
                raise RuntimeError("Cannot energy-order XL multipliers.")
            multipliers = self._permute_state_rows(multipliers, permutation)
            molecule.xlesmd_multipliers = multipliers.gather(
                2, permutation.unsqueeze(1).expand_as(multipliers)
            )

        self._xl_ctx["es_amp"] = self._permute_state_rows(self._xl_ctx["es_amp"], permutation)
        self._xl_ctx["es_amp_t"] = self._permute_history_state_rows(self._xl_ctx["es_amp_t"], permutation)
        # FSSH remains in the instantaneous energy-ordered manifold.  Keep its
        # active energy rank while the complete XL root/history block is relabeled.
        molecule.active_state = self._active_states + 1
        for mol in torch.nonzero(permutation.ne(identity).any(dim=1), as_tuple=False).squeeze(1).tolist():
            self.xlesmd_energy_order_events.append(
                {"step": int(step), "molecule": mol, "permutation": permutation[mol].detach().cpu().tolist()}
            )

    def _compute_electronic_structure(self, molecule, learned_parameters, **kwargs):
        if self._xl_ctx is None:
            raise RuntimeError("XL-FSSH auxiliary history is not initialized.")
        self._xlesmd_learned_parameters = learned_parameters
        P0 = self._propagate_xlesmd_auxiliaries(molecule)
        coords_prev = self._xlesmd_coords_before
        mos_prev = self._xlesmd_mos_before
        S_prev = self._xlesmd_S_prev
        polar = bool(self.xl_bomd_params.get("polar_unitarize_mo_transport", True))
        transport = bool(self.xl_bomd_params.get("transport_mo_auxiliary", True))

        old_state = molecule.active_state
        molecule.active_state = self._active_states + 1
        esdriver_args = kwargs.pop("esdriver_args", ())
        try:
            self.esdriver(
                molecule,
                learned_parameters=learned_parameters,
                xl_bomd_params=self.xl_bomd_params,
                P0=P0,
                cis_amp=self._xl_ctx["es_amp"],
                dm_prop=self.dmprop,
                xlesmd_mo_transport=(coords_prev, mos_prev, S_prev, polar) if transport else None,
                *esdriver_args,
                **kwargs,
            )
        finally:
            molecule.active_state = old_state

        self._transport_xlesmd_history(molecule, coords_prev, mos_prev, S_prev)
        self._apply_xlesmd_energy_order(molecule, self._xlesmd_step_index + self.step_offset + 1)
        self._record_xlesmd_orthogonality(molecule, self._xlesmd_step_index + self.step_offset + 1)

        energies = self._build_state_energies(molecule)
        cache_new = {"energies": energies, "cis_amp": self._current_cis_amplitudes(molecule)}
        if self._direct_nac_tdc:
            molecule.nac = None
            molecule.nac = self._compute_NACR_for_hop(molecule, self._nac_pairs())
            cache_new["nac_dot"] = self._nac_dot_from_vectors(molecule, molecule.nac)
        self._cache_new = cache_new
        return energies

    def _compute_NACR_for_hop(self, molecule, nac_pairs):
        if self._direct_nac_tdc:
            cached_nac = self._select_nac_pairs(molecule.nac, nac_pairs)
            if cached_nac is not None:
                return cached_nac
        pair_nac = calc_nac(
            molecule,
            molecule.cis_amplitudes,
            molecule.cis_energies,
            molecule.dm,
            None,
            None,
            nac_pairs,
            rpa=False,
            include_response_terms=self.esdriver.conservative_force.energy.nac_config.include_response_terms,
            w=molecule.w,
            e_mo=molecule.e_mo,
        )
        return {(s1 - 1, s2 - 1): pair_nac[:, pair_idx] for pair_idx, (s1, s2) in enumerate(nac_pairs)}

    def _recompute_active_force(self, molecule):
        """Re-evaluate the current shadow geometry on the newly active root."""
        molecule.active_state = self._active_states + 1
        self.esdriver(
            molecule,
            learned_parameters=self._xlesmd_learned_parameters,
            xl_bomd_params=self.xl_bomd_params,
            P0=molecule.dm,
            cis_amp=self._xl_ctx["es_amp"],
            dm_prop=self.dmprop,
        )

    def _do_integrator_step(self, i, molecule, learned_parameters, **kwargs):
        self._xlesmd_step_index = i
        with torch.no_grad():
            self._xlesmd_coords_before = molecule.coordinates.detach().clone()
            self._xlesmd_mos_before = molecule.molecular_orbitals.detach().clone()
        return super()._do_integrator_step(i, molecule, learned_parameters, **kwargs)

    def initialize(
        self, molecule, remove_com=None, learned_parameters=None, steps: Optional[int] = None, *args, **kwargs
    ):
        learned_parameters = {} if learned_parameters is None else learned_parameters
        self._xlesmd_learned_parameters = learned_parameters
        super().initialize(
            molecule,
            remove_com=remove_com,
            learned_parameters=learned_parameters,
            steps=steps,
            *args,
            **kwargs,
        )

        initial_eta = molecule.cis_amplitudes.detach().clone()
        self.esdriver.conservative_force.energy.excited_states = None
        self.esdriver.conservative_force.energy.xlesmd = True
        molecule.Electronic_entropy = torch.zeros(
            molecule.species.shape[0], dtype=molecule.coordinates.dtype, device=molecule.coordinates.device
        )

        shadow_kwargs = dict(kwargs)
        for key in ("dm_prop", "xl_bomd_params", "P0", "cis_amp", "xlesmd_mo_transport"):
            shadow_kwargs.pop(key, None)
        self.esdriver(
            molecule,
            learned_parameters=learned_parameters,
            xl_bomd_params=self.xl_bomd_params,
            P0=molecule.dm,
            cis_amp=initial_eta,
            dm_prop=self.dmprop,
            *args,
            **shadow_kwargs,
        )

        with torch.no_grad():
            P = molecule.dm.detach().clone()
            self._xl_ctx = {
                "P": P,
                "Pt": P.unsqueeze(0).expand((self.m,) + P.shape).clone(),
                "es_amp": initial_eta.clone(),
                "es_amp_t": initial_eta.unsqueeze(0).expand((self.m,) + initial_eta.shape).clone(),
            }
            self._xlesmd_S_prev = self._packed_current_ao_overlap(molecule)
            molecule.acc = molecule.force * molecule.mass_inverse * CONSTANTS.ACC_SCALE

        self._apply_xlesmd_energy_order(molecule, self.step_offset)
        self._copy_cache_entry(self._cache_old, "energies", self._build_state_energies(molecule))
        if self._cache_prev_cis_amp:
            self._copy_cache_entry(self._cache_old, "cis_amp", self._current_cis_amplitudes(molecule))
        if self._direct_nac_tdc:
            molecule.nac = None
            molecule.nac = self._compute_NACR_for_hop(molecule, self._nac_pairs())
            self._copy_cache_entry(
                self._cache_old, "nac_dot", self._nac_dot_from_vectors(molecule, molecule.nac)
            )
        self._record_xlesmd_orthogonality(molecule, self.step_offset)

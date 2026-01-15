from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch

from seqm.NonadiabaticDynamics import EhrenfestDynamics, SurfaceHoppingDynamics


@dataclass
class TullyModel:
    """Analytic Tully model with energies and NACs in 1D."""

    pot: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    # pot(x) -> E (2,), dE/dx (2,), nac scalar
    name: str = "custom"

    @staticmethod
    def _two_state_from_diabatic(V11, V22, V12, dV11, dV22, dV12):
        """Return adiabatic energies, gradients, and derivative coupling for 2x2 diabatic surfaces."""
        delta = V11 - V22  # energy gap between diabatic states
        sumv = V11 + V22
        # adiabatic splitting
        S = torch.sqrt(delta * delta + 4.0 * V12 * V12)
        E1 = 0.5 * (sumv - S)
        E2 = 0.5 * (sumv + S)

        # gradients of splitting
        dS = (delta * (dV11 - dV22) + 8.0 * V12 * dV12) / S
        dE1 = 0.5 * (dV11 + dV22) - 0.5 * dS
        dE2 = 0.5 * (dV11 + dV22) + 0.5 * dS

        # derivative coupling (1D) d(theta)/dx where tan(2theta)=2V12/delta
        nac = (delta * dV12 - (dV11 - dV22) * V12) / (delta * delta + 4.0 * V12 * V12)
        return torch.stack([E1, E2], dim=-1), torch.stack([dE1, dE2], dim=-1), nac

    @staticmethod
    def single_crossing():
        """Tully Model 1: single avoided crossing."""
        A = 0.27211386246  # 0.01 Eh -> eV
        B = 3.02356179940  # 1.6 1/Bohr -> 1/Å
        C = 0.13605693123  # 0.005 Eh -> eV
        D = 3.57106482609  # 1.0 1/Bohr^2 -> 1/Å^2

        def pot(x: torch.Tensor):
            absx = torch.abs(x)
            exp_abs = torch.exp(-B * absx)
            sign = torch.sign(x)
            V11 = A * (1.0 - exp_abs) * sign
            dV11 = A * B * exp_abs
            V22 = -V11
            dV22 = -dV11

            V12 = C * torch.exp(-D * x * x)
            dV12 = -2.0 * D * x * V12

            return TullyModel._two_state_from_diabatic(V11, V22, V12, dV11, dV22, dV12)

        return TullyModel(pot, name="single_avoided_crossing")

    @staticmethod
    def double_crossing():
        """Tully Model 2: dual avoided crossing."""
        # Parameters from Tully JCP 93, 1061 (1990)
        A = 2.72113862460
        B = 0.99989815131
        C = 0.40817079369
        D = 0.21426388957
        E0 = 1.36056931230

        def pot(x):
            expBx2 = torch.exp(-B * x * x)
            expDx2 = torch.exp(-D * x * x)
            V11 = torch.zeros_like(x)
            V22 = E0 - A * expBx2
            V12 = C * expDx2
            dV11 = torch.zeros_like(x)
            dV22 = A * (2.0 * B * x * expBx2)
            dV12 = -2.0 * D * x * V12
            return TullyModel._two_state_from_diabatic(V11, V22, V12, dV11, dV22, dV12)

        return TullyModel(pot, name="double_avoided_crossing")

    @staticmethod
    def extended_coupling():
        """Tully Model 3: extended coupling with reflection (correct canonical form)."""

        # Original (a.u.): A=0.0006 Eh, B=0.10 Eh, C=0.90 1/Bohr
        A = 0.01632683175  # eV
        B = 2.72113862460  # eV
        bohr_per_ang = 1.8897261246257702
        C = 0.90 * bohr_per_ang  # 1/Angstrom  (NOTE: linear, not squared!)

        def pot(x: torch.Tensor):
            V11 = A * torch.ones_like(x)
            V22 = -A * torch.ones_like(x)
            dV11 = torch.zeros_like(x)
            dV22 = torch.zeros_like(x)

            absx = torch.abs(x)
            exp_abs = torch.exp(-C * absx)
            s = 0.5 * (1.0 + torch.sign(x))
            V12 = B * (s * (2.0 - exp_abs) + (1.0 - s) * exp_abs)
            dV12 = B * C * exp_abs

            return TullyModel._two_state_from_diabatic(V11, V22, V12, dV11, dV22, dV12)

        return TullyModel(pot, name="extended_coupling")


class TullyMolecule:
    """Minimal 1D molecule representation for Tully models."""

    def __init__(self, x0: float, v0: float, mass: float = 2000.0, device=None, dtype=torch.double):
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.coordinates = torch.zeros((1, 1, 3), dtype=dtype, device=self.device)
        self.coordinates[0, 0, 0] = x0
        self.velocities = torch.zeros_like(self.coordinates)
        self.velocities[0, 0, 0] = v0
        self.acc = torch.zeros_like(self.coordinates)
        self.mass = torch.tensor([[mass]], dtype=dtype, device=self.device)
        self.mass_inverse = 1.0 / self.mass
        self.dm = torch.zeros(1, 1, 1, device=self.device)
        self.cis_amplitudes = None
        self.cis_energies = None
        self.transition_density_matrices = None
        self.force = torch.zeros_like(self.coordinates)
        self.all_forces = torch.zeros((1, 3, 1, 3), dtype=dtype, device=self.device)
        self.nac = None
        self.nac_dot = None
        self.Etot = torch.zeros(1, device=self.device)
        from seqm.seqm_functions.constants import Constants

        self.const = Constants().to(self.device)
        self.species = torch.tensor([[1]], device=self.device)
        self.num_atoms = torch.tensor([1.0], device=self.device, dtype=dtype)
        self.tot_charge = torch.zeros(1, device=self.device)
        self.mult = torch.tensor([1.0], device=self.device)
        self.seqm_parameters = {"excited_states": {"n_states": 2}}


_TULLY_NSTATES = 2
_TULLY_OUTPUT = {"h5": {}, "print every": 0, "checkpoint every": 0}


def _tully_seqm_params() -> Dict:
    return {
        "method": "AM1",
        "elements": [1],
        "scf_eps": 1.0e-8,
        "scf_converger": [1],
        "excited_states": {"n_states": _TULLY_NSTATES},
        "nonadiabatic": {"compute_nac": True},
    }


class _TullyDynamicsMixin:
    def _tully_init(
        self, model: TullyModel, *, timestep: float, electronic_substeps: int, nonadiabatic: Dict = None
    ):
        params = _tully_seqm_params()
        if nonadiabatic:
            params["nonadiabatic"].update(nonadiabatic)
        super().__init__(
            params, timestep=timestep, electronic_substeps=electronic_substeps, output=_TULLY_OUTPUT
        )
        self.model = model
        self.initial_state = 0
        self._nstates = _TULLY_NSTATES
        self._active_states = None
        self._active_state = 0  # backward compat for any stray references
        self.rho_history = []

    def _reset_density_history(self):
        self.rho_history = []

    def _record_density_matrix(self):
        coeffs = self._coeffs_complex()
        if coeffs is None:
            return
        if coeffs.shape[1] < 2:
            return
        c0 = coeffs[:, 0]
        c1 = coeffs[:, 1]
        rho00 = torch.real(c0.conj() * c0)
        rho11 = torch.real(c1.conj() * c1)
        rho01 = torch.abs(c0.conj() * c1)
        rho = torch.stack((rho00, rho11, rho01), dim=1)
        self.rho_history.append(rho.detach().cpu())

    def _compute_electronic_structure(self, molecule, learned_parameters, **kwargs):
        x = molecule.coordinates[:, 0, 0]
        E, dE, nac = self.model.pot(x)
        nmol, molsize = molecule.coordinates.shape[:2]
        device = molecule.coordinates.device
        dtype = molecule.coordinates.dtype
        molecule.cis_energies = (E[:, 1] - E[:, 0]).unsqueeze(1)
        molecule.Etot = E[:, 0]
        molecule.all_forces = torch.zeros((nmol, self._nstates + 1, molsize, 3), dtype=dtype, device=device)
        molecule.all_forces[:, 1, 0, 0] = -dE[:, 0]
        molecule.all_forces[:, 2, 0, 0] = -dE[:, 1]
        if self._active_states is None:
            self._active_states = torch.zeros((nmol,), dtype=torch.long, device=device)
        exc_idx = int(self._active_states[0].item())
        self._active_state = exc_idx
        molecule.active_state = self._active_states + 1
        molecule.force = molecule.all_forces[:, exc_idx + 1]
        nac_vec = torch.zeros((nmol, self._nstates, self._nstates, molsize, 3), dtype=dtype, device=device)
        nac_vec[:, 0, 1, 0, 0] = nac
        nac_vec[:, 1, 0, 0, 0] = -nac
        nac_dot = torch.zeros((nmol, self._nstates, self._nstates), dtype=dtype, device=device)
        vel = molecule.velocities[:, 0, 0]
        nac_dot[:, 0, 1] = nac * vel
        nac_dot[:, 1, 0] = -nac * vel
        molecule.nac = nac_vec
        molecule.nac_dot = nac_dot
        self._cache_new = {
            "energies": torch.stack([E[:, 0], E[:, 1]], dim=1),
            "nac_vec": nac_vec,
            "nac_dot": nac_dot,
            "cis_amp": None,
        }
        # Provide a minimal esdriver args cache so FSSH hop recomputation paths have defaults.
        self._last_esdriver_args = {"learned_parameters": {}, "kwargs": {}, "esdriver_args": ()}
        return self._cache_new["energies"]


class TullyDynamics(_TullyDynamicsMixin, EhrenfestDynamics):
    """Ehrenfest dynamics over analytic Tully models."""

    def __init__(self, model: TullyModel, *, timestep=0.05, electronic_substeps=10):
        self._tully_init(model, timestep=timestep, electronic_substeps=electronic_substeps)

    def _after_electronic_update(self, molecule, state_energies, nac_matrix=None, nac_dot=None, step=None):
        pop = self.populations
        # TODO: add non-adiabatic coupling contribution to forces
        force = (
            pop[:, 0].view(-1, 1, 1) * molecule.all_forces[:, 1]
            + pop[:, 1].view(-1, 1, 1) * molecule.all_forces[:, 2]
        )
        molecule.force = force
        molecule.Etot = torch.sum(pop * state_energies, dim=1)
        self._record_density_matrix()


class TullyFSSH(_TullyDynamicsMixin, SurfaceHoppingDynamics):
    def __init__(self, model: TullyModel, *, timestep=0.05, electronic_substeps=10):
        self._tully_init(
            model,
            timestep=timestep,
            electronic_substeps=electronic_substeps,
            nonadiabatic={"recompute_on_hop": True},
        )

    def _recompute_active_force(self, molecule):
        state_energies = self._compute_electronic_structure(molecule, learned_parameters={})
        exc_idx = (
            int(self._active_states[0].item()) if self._active_states is not None else int(self._active_state)
        )
        self._active_state = exc_idx
        molecule.force = molecule.all_forces[:, exc_idx + 1]
        self._current_potential = state_energies[:, exc_idx]
        molecule.Etot = self._current_potential

    def _after_electronic_update(self, molecule, state_energies, nac_matrix=None, nac_dot=None, step=None):
        super()._after_electronic_update(
            molecule, state_energies, nac_matrix=nac_matrix, nac_dot=nac_dot, step=step
        )
        self._record_density_matrix()


def run_tully(
    model: TullyModel,
    method="fssh",
    timestep=0.05,
    steps=200,
    x0=-8.0,
    v0=2.0,
    mass=2000.0,
    seed=0,
    electronic_substeps=10,
):
    torch.manual_seed(seed)
    if isinstance(model, str):
        name = model.lower()
        if name in ("1", "single", "sac", "single_avoided_crossing", "model1"):
            model = TullyModel.single_crossing()
        elif name in ("2", "double", "dac", "double_avoided_crossing", "model2"):
            model = TullyModel.double_crossing()
        elif name in ("3", "extended", "reflection", "model3", "extended_coupling"):
            model = TullyModel.extended_coupling()
        else:
            raise ValueError(f"Unknown Tully model '{model}'")
    dyn_cls = TullyDynamics if method == "ehrenfest" else TullyFSSH
    dyn = dyn_cls(model, timestep=timestep, electronic_substeps=electronic_substeps)
    mol = TullyMolecule(x0=x0, v0=v0, mass=mass, dtype=torch.double)
    # Pre-initialize coefficients so initialization keeps the desired state
    dyn._setup_states(mol)
    dyn._init_coeffs(mol)
    mol.dm = torch.zeros(1, 1, 1, device=mol.coordinates.device)
    return dyn.run(mol, steps=steps, reuse_P=True, remove_com=None)

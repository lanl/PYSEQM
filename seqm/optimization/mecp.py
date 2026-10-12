import os
import tempfile

import numpy as np
import torch

from ..ElectronicStructure import Electronic_Structure as esdriver
from ._geometric import build_geomol, run_geometric, sync_coordinates, write_trajectory

_HARTREE_TO_EV = 27.211386245988


def _mecp_parameters(molecule, states, include_response_terms, compute_nac=True):
    if len(states) != 2 or any(isinstance(state, bool) or int(state) != state for state in states):
        raise ValueError("states must contain two integer CIS state indices.")
    states = tuple(sorted(map(int, states)))
    if states[0] < 1 or states[0] == states[1]:
        raise ValueError("states must identify two distinct excited states (S1 and above).")
    if molecule.seqm_parameters.get("UHF", False):
        raise NotImplementedError("CIS MECP optimization requires restricted closed-shell PYSEQM.")

    parameters = dict(molecule.seqm_parameters)
    excited = dict(parameters.get("excited_states") or {})
    if excited.get("method", "cis").lower() not in {"cis", "tda"}:
        raise ValueError("CIS MECP optimization requires excited_states.method='cis' (or 'tda').")
    excited.update(method="cis", n_states=max(int(excited.get("n_states", 0)), states[1]))
    parameters.update(excited_states=excited, analytical_gradient=[True], do_all_forces=True)

    nonadiabatic = dict(parameters.get("nonadiabatic", {}))
    nonadiabatic["compute_nac"] = compute_nac
    if compute_nac:
        nonadiabatic.update(pairs=[states], include_response_terms=bool(include_response_terms))
    parameters["nonadiabatic"] = nonadiabatic
    return parameters, states


def _projected_objective(energies, gradients, coupling, gap_weight=1.0):
    """Return the gradient-projection MECI objective and search vector."""
    e1, e2 = energies
    difference = (gradients[0] - gradients[1]).reshape(-1)
    coupling = coupling.reshape(-1)
    eps = 100 * torch.finfo(difference.dtype).eps
    if torch.linalg.vector_norm(difference) <= eps or torch.linalg.vector_norm(coupling) <= eps:
        raise RuntimeError("The MECI branching plane is singular.")
    x = difference / torch.linalg.vector_norm(difference)
    y = coupling - x * torch.dot(x, coupling)
    if torch.linalg.vector_norm(y) <= eps:
        raise RuntimeError("The MECI branching plane is rank deficient.")
    basis = torch.stack((x, y / torch.linalg.vector_norm(y)), dim=1)
    average = gradients.mean(dim=0).reshape(-1)
    seam_gradient = average - basis @ (basis.T @ average)
    gap_gradient = gap_weight * (e1 - e2) * x
    return energies.mean(), (seam_gradient + gap_gradient).reshape_as(gradients[0])


def _penalty_objective(energies, gradients, sigma, alpha):
    e1, e2 = energies
    gap = e2 - e1
    objective = energies.mean() + sigma * gap.square() / (gap + alpha)
    scale = sigma * gap * (gap + 2 * alpha) / (gap + alpha).square()
    return objective, gradients.mean(dim=0) + scale * (gradients[1] - gradients[0])


def _scaled_coupling(energies, derivative_coupling):
    return (energies[1] - energies[0]) * derivative_coupling


def _seam_projector(difference, coupling):
    xnorm = np.linalg.norm(difference)
    if xnorm < 1e-12:
        raise RuntimeError("The MECI gradient-difference vector is singular.")
    x = difference / xnorm
    y = coupling - x * (x @ coupling)
    ynorm = np.linalg.norm(y)
    if ynorm < 1e-12:
        raise RuntimeError("The MECI branching plane is rank deficient.")
    basis = np.column_stack((x, y / ynorm))
    return np.eye(x.size) - basis @ basis.T


def _cartesian_seam_gradient(gradients, coupling):
    projector = _seam_projector((gradients[1] - gradients[0]).reshape(-1), coupling.reshape(-1))
    return projector @ gradients.mean(axis=0).reshape(-1)


def _damped_bfgs(hessian, step, gradient_change):
    hs = hessian @ step
    shs = step @ hs
    sy = step @ gradient_change
    scale = np.linalg.norm(step) * np.linalg.norm(gradient_change)
    if shs <= 0 or scale == 0 or sy <= 1e-10 * scale:
        return hessian
    if sy < 0.2 * shs:
        theta = 0.8 * shs / (shs - sy)
        gradient_change = theta * gradient_change + (1 - theta) * hs
        sy = step @ gradient_change
    updated = hessian - np.outer(hs, hs) / shs + np.outer(gradient_change, gradient_change) / sy
    return 0.5 * (updated + updated.T)


def _run_composed(
    molecule, parameters, state_data, states, traj_file, output_file, run_kwargs, gap_tolerance, switch_gap
):
    if molecule.nmol != 1:
        raise ValueError("geomeTRIC optimization supports exactly one molecule.")
    try:
        from geometric.engine import Engine
        from geometric.internal import (
            CartesianCoordinates,
            DelocalizedInternalCoordinates,
            PrimitiveInternalCoordinates,
        )
        from geometric.nifty import ang2bohr, bohr2ang, ev2au, logger
        from geometric.optimize import OPT_STATE, Optimizer, OptParams, calc_drms_dmax
    except ImportError as error:
        raise ImportError("Install the geomeTRIC integration with `pip install '.[geomopt]'`.") from error

    traj_file, output_file = os.path.abspath(traj_file), os.path.abspath(output_file)
    s1, s2 = states

    class PYSEQMComposedEngine(Engine):
        def __init__(self):
            super().__init__(build_geomol(molecule))
            self.driver = esdriver(parameters).to(molecule.coordinates.device)
            self.step = 0

        def sync(self, coords):
            sync_coordinates(molecule, coords)

        def calc(self, coords, dirname, *args, **kwargs):
            self.sync(coords)
            result = super().calc(coords, dirname, *args, **kwargs)
            energies = torch.as_tensor(
                result["state_energies"] / ev2au,
                device=molecule.coordinates.device,
                dtype=molecule.coordinates.dtype,
            )
            molecule.mecp_state_energies = energies
            molecule.mecp_energy_gap = energies[1] - energies[0]
            return result

        def calc_new(self, coords, dirname=None):
            energies, gradients = state_data(self.driver)
            coupling = _scaled_coupling(energies, molecule.nac[(s1 - 1, s2 - 1)][0])
            self.step += 1
            write_trajectory(
                traj_file,
                self.step,
                molecule,
                f"Eavg = {energies.mean().item():.8f} eV, "
                f"E(S{s1}) = {energies[0].item():.8f} eV, "
                f"E(S{s2}) = {energies[1].item():.8f} eV, "
                f"gap = {(energies[1] - energies[0]).item():.8f} eV, method = composed_step",
            )
            scale = ev2au * bohr2ang
            return {
                "energy": float(energies.mean()) * ev2au,
                "gradient": (gradients.mean(dim=0).detach().cpu().numpy() * scale).reshape(-1),
                "state_energies": energies.detach().cpu().numpy() * ev2au,
                "state_gradients": gradients.detach().cpu().numpy().reshape(2, -1) * scale,
                "coupling": coupling.detach().cpu().numpy().reshape(-1) * scale,
            }

    class ComposedStepOptimizer(Optimizer):
        def __init__(self, *args, **kwargs):
            self.gap_tolerance = kwargs.pop("gap_tolerance") * ev2au
            self.switch_gap = kwargs.pop("switch_gap") * ev2au
            super().__init__(*args, **kwargs)

        def calcEnergyForce(self):
            super().calcEnergyForce()
            result = self.engine.calc(self.X, self.dirname)
            self.state_energies = np.asarray(result["state_energies"])
            self.state_gradients = np.asarray(result["state_gradients"])
            self.coupling = np.asarray(result["coupling"])

        def _ci_coordinates(self):
            self.gap = self.state_energies[1] - self.state_energies[0]
            self.Gavg = self.IC.calcGrad(self.X, self.state_gradients.mean(axis=0)).flatten()
            self.Gdiff = self.IC.calcGrad(self.X, self.state_gradients[1] - self.state_gradients[0]).flatten()
            self.Hvec = self.IC.calcGrad(self.X, self.coupling).flatten()
            self.P = _seam_projector(self.Gdiff, self.Hvec)
            self.Q = np.eye(self.P.shape[0]) - self.P
            self.seam_gradient = self.P @ self.Gavg
            self.seam_gradient_cart = _cartesian_seam_gradient(self.state_gradients, self.coupling)

        @staticmethod
        def _solve(hessian, gradient, floor):
            values, vectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
            return -vectors @ ((vectors.T @ gradient) / np.maximum(values, floor))

        def prepareFirstStep(self):
            super().prepareFirstStep()
            self._ci_coordinates()
            self.Havg = self.H.copy()

        def _limit(self, step, radius):
            if radius <= 0 or not np.any(step):
                return np.zeros_like(step)
            norm = self.get_cartesian_norm(step, verbose=0)
            return step if norm <= radius else step * radius / norm

        def step(self):
            self.Iteration += 1
            if self.Iteration % 5 == 0:
                self.engine.clearCalcs()
                self.IC.clearCache()

            # Minimum-norm normal step satisfying the linearized constraint
            # gap + Gdiff @ step = 0.
            branch_step = -self.gap * self.Gdiff / (self.Gdiff @ self.Gdiff)

            values, seam_basis = np.linalg.eigh(self.P)
            seam_basis = seam_basis[:, values > 0.5]
            if seam_basis.shape[1]:
                hessian = seam_basis.T @ self.Havg @ seam_basis
                gradient = seam_basis.T @ self.Gavg
                seam_step = seam_basis @ self._solve(hessian, gradient, self.params.epsilon)
            else:
                seam_step = np.zeros_like(branch_step)

            branch_step = self._limit(self.Q @ branch_step, 0.9 * self.trust)
            branch_norm = self.get_cartesian_norm(branch_step, verbose=0)
            seam_radius = np.sqrt(max(self.trust**2 - branch_norm**2, 0.0))
            if abs(self.gap) > self.switch_gap:
                seam_radius = min(seam_radius, 0.3 * self.trust)
            trial = self._limit(branch_step + self._limit(self.P @ seam_step, seam_radius), self.trust)

            self.Yprev = self.Y.copy()
            self.Xprev = self.X.copy()
            self.Gxprev = self.gradx.copy()
            self.Gprev = self.G.copy()
            self.Eprev = self.E
            self.gap_prev = self.gap
            self.seam_gradient_prev = self.seam_gradient.copy()
            self.seam_gradient_cart_prev = self.seam_gradient_cart.copy()
            self.Havg_prev = self.Havg.copy()
            self.state_energies_prev = self.state_energies.copy()
            self.state_gradients_prev = self.state_gradients.copy()
            self.coupling_prev = self.coupling.copy()
            self.mecp_state_energies_prev = molecule.mecp_state_energies.clone()
            self.mecp_energy_gap_prev = molecule.mecp_energy_gap.clone()

            origin = self.X.copy()
            best_rms, best_x = np.inf, None
            for _ in range(12):
                self.X = origin.copy()
                self.newCartesian(trial)
                rms, _ = calc_drms_dmax(self.X, origin)
                if not self.IC.bork and np.isfinite(rms):
                    if rms < best_rms:
                        best_rms, best_x = rms, self.X.copy()
                    if rms <= 1.2 * self.trust:
                        break
                self.IC.bork = False
                trial *= 0.5
            if best_x is None or best_rms > 1.2 * self.trust:
                logger.info("MECI back-transformation failed; rebuilding coordinates.\n")
                self.X = origin
                self.IC.bork = False
                self.checkCoordinateSystem(recover=True)
                self._ci_coordinates()
                self.Havg = self.H.copy()
                self.Iteration -= 1
                self.state = OPT_STATE.SKIP_EVALUATION
                return

            self.X = best_x
            actual_step = self.IC.calcDiff(self.X, origin)
            self.Y += actual_step
            self.cnorm = self.get_cartesian_norm(actual_step, verbose=0)
            self.branch_step, self.seam_step = self.Q @ actual_step, self.P @ actual_step
            self.predicted_seam_reduction = -(
                self.Gavg @ self.seam_step + 0.5 * self.seam_step @ self.Havg @ self.seam_step
            )
            self.state = OPT_STATE.NEEDS_EVALUATION

        def _reject_step(self, reason):
            logger.info(f"Rejecting MECI step: {reason}.\n")
            self.trust = max(self.params.tmin, min(self.trust, self.cnorm) / 2)
            self.X, self.Y = self.Xprev.copy(), self.Yprev.copy()
            self.gradx, self.G, self.E = self.Gxprev.copy(), self.Gprev.copy(), self.Eprev
            self.Havg = self.Havg_prev
            self.state_energies = self.state_energies_prev
            self.state_gradients, self.coupling = self.state_gradients_prev, self.coupling_prev
            molecule.mecp_state_energies = self.mecp_state_energies_prev
            molecule.mecp_energy_gap = self.mecp_energy_gap_prev
            self._ci_coordinates()
            self.engine.sync(self.X)
            self.IC.clearCache()
            self.progress = self.progress[:-1]
            if self.Iteration >= self.params.maxiter:
                self.state = OPT_STATE.FAILED

        def evaluateStep(self):
            if self.params.xyzout is not None:
                self.progress.write(self.params.xyzout)
            rms_disp, max_disp = calc_drms_dmax(self.X, self.Xprev)
            try:
                self._ci_coordinates()
            except RuntimeError:
                self._reject_step("invalid electronic-structure response")
                return

            seam_atoms = np.linalg.norm(self.seam_gradient_cart.reshape(-1, 3), axis=1)
            old_seam_atoms = np.linalg.norm(self.seam_gradient_cart_prev.reshape(-1, 3), axis=1)
            seam_rms = np.sqrt(np.mean(seam_atoms**2))
            seam_max = np.max(seam_atoms)
            old_seam_rms = np.sqrt(np.mean(old_seam_atoms**2))
            gap_abs, old_gap_abs = abs(self.gap), abs(self.gap_prev)
            converged = (
                gap_abs < self.gap_tolerance
                and seam_rms < self.params.Convergence_grms
                and seam_max < self.params.Convergence_gmax
                and rms_disp < self.params.Convergence_drms
                and max_disp < self.params.Convergence_dmax
            )
            logger.info(
                "MECI step %4i: gap = %.3e Eh, seam grad = %.3e/%.3e, "
                "displace = %.3e/%.3e, trust = %.3e\n"
                % (self.Iteration, gap_abs, seam_rms, seam_max, rms_disp, max_disp, self.trust)
            )
            if converged:
                self.state = OPT_STATE.CONVERGED
                return

            seam_reduction = self.Eprev - self.E
            gap_envelope = max(5 * self.gap_tolerance, self.switch_gap, 1.25 * old_gap_abs)
            feasible = gap_abs < old_gap_abs
            optimal = gap_abs <= gap_envelope and (seam_reduction > 0 or seam_rms < old_seam_rms)
            accepted = feasible or optimal

            if feasible:
                fraction = (old_gap_abs - gap_abs) / max(old_gap_abs, 1e-14)
                quality = 1.0 if fraction > 0.25 else 0.5 if fraction > 0.05 else 0.1
            elif optimal:
                quality = 0.5
            else:
                quality = -1.0
            if gap_abs <= gap_envelope and self.predicted_seam_reduction > 1e-14:
                quality = max(quality, seam_reduction / self.predicted_seam_reduction)

            if not accepted and self.trust > self.params.tmin:
                self._reject_step("neither feasibility nor seam optimality improved")
                return

            self.Havg = _damped_bfgs(self.Havg, self.seam_step, self.seam_gradient - self.seam_gradient_prev)
            self.H, self.G = self.Havg, self.Gavg
            self.X_hist.append(self.X.copy())
            self.Gx_hist.append(self.gradx.copy())
            self.engine.save_guess_files(self.dirname)

            rebuilt = False
            if self.params.check > 0 and self.CoordCounter == self.params.check - 1:
                rebuilt = self.checkCoordinateSystem(recover=False)
                self.CoordCounter = 0
            elif self.params.check > 0:
                self.CoordCounter += 1
            if self.IC.setRegularization(self.X):
                self.rebuild_hessian()
                rebuilt = True
            if rebuilt:
                self._ci_coordinates()
                self.Havg = self.H.copy()

            if quality < 0.25:
                self.trust = max(self.params.tmin, min(self.trust, self.cnorm) / 2)
            elif quality > 0.75:
                self.trust = min(self.params.tmax, np.sqrt(2) * self.trust)
            if self.Iteration >= self.params.maxiter:
                self.state = OPT_STATE.CONVERGED if self.params.Converge_maxiter else OPT_STATE.FAILED

    kwargs = dict(run_kwargs)
    coordsys = kwargs.pop("coordsys", "tric").lower()
    kwargs.pop("constraints", None)
    conmethod = kwargs.pop("conmethod", 0)
    rigid = kwargs.pop("rigid", False)
    kwargs.setdefault("qccnv", False)
    params = OptParams(**kwargs)
    engine = PYSEQMComposedEngine()
    geomol = engine.M
    coords = geomol.xyzs[0].flatten() * ang2bohr
    coordinate_systems = {
        "cart": (CartesianCoordinates, False, False),
        "prim": (PrimitiveInternalCoordinates, True, False),
        "dlc": (DelocalizedInternalCoordinates, True, False),
        "hdlc": (DelocalizedInternalCoordinates, False, True),
        "tric-p": (PrimitiveInternalCoordinates, False, False),
        "tric": (DelocalizedInternalCoordinates, False, False),
    }
    if coordsys not in coordinate_systems:
        raise ValueError(f"Unknown geomeTRIC coordinate system: {coordsys}")
    coordinate_class, connect, addcart = coordinate_systems[coordsys]
    coordinates = coordinate_class(
        geomol,
        build=True,
        connect=connect,
        addcart=addcart,
        constraints=None,
        cvals=None,
        conmethod=conmethod,
        rigid=rigid,
    )
    with tempfile.TemporaryDirectory() as dirname:
        optimizer = ComposedStepOptimizer(
            coords,
            geomol,
            coordinates,
            engine,
            dirname,
            params,
            gap_tolerance=gap_tolerance,
            switch_gap=switch_gap,
        )
        result = optimizer.optimizeGeometry()
    with open(output_file, "w") as out:
        out.write("\n".join(result.write_xyz(selection=[-1])))
    return result


def geomeTRIC_mecp(
    molecule,
    states=(1, 2),
    method: str = "gradient_projection",
    traj_file: str = "mecp_opt_traj.xyz",
    output_file: str = "mecp.xyz",
    gap_weight: float = 1.0,
    include_response_terms: bool = True,
    penalty_sigma: float = 3.5,
    penalty_alpha: float = 0.025,
    gap_tolerance: float = 5e-3,
    seam_switch_gap: float = 0.05,
    **run_kwargs,
):
    """Locate a minimum-energy crossing between two CIS excited states.

    ``gradient_projection`` and ``composed_step`` use the CIS derivative
    coupling. ``penalty`` uses the smooth Levine--Coe--Martinez objective and
    needs only state energies and gradients. State indices are 1-based and do
    not include the ground state.
    """
    method = method.lower().replace("-", "_")
    method = {"direct": "gradient_projection", "gp": "gradient_projection"}.get(method, method)
    if method not in {"gradient_projection", "composed_step", "penalty"}:
        raise ValueError("method must be 'gradient_projection', 'composed_step', or 'penalty'.")
    if method == "gradient_projection" and gap_weight <= 0:
        raise ValueError("gap_weight must be positive.")
    if method == "penalty" and (penalty_sigma <= 0 or penalty_alpha <= 0):
        raise ValueError("penalty_sigma and penalty_alpha must be positive.")
    if method == "composed_step" and (gap_tolerance <= 0 or seam_switch_gap <= 0):
        raise ValueError("gap_tolerance and seam_switch_gap must be positive.")
    if method == "composed_step" and run_kwargs.get("constraints") is not None:
        raise NotImplementedError("Constraints are not supported by composed-step MECI optimization.")
    unsupported = ("transition", "irc", "qdata", "write_cart_hess", "meci")
    if method == "composed_step" and any(run_kwargs.get(option) for option in unsupported):
        raise ValueError(
            "Composed-step MECI does not support transition, IRC, qdata, Hessian output, or meci options."
        )
    if method == "composed_step" and run_kwargs.get("hessian") not in {None, "never"}:
        raise ValueError("Composed-step MECI does not support numerical Hessian calculations.")
    parameters, states = _mecp_parameters(
        molecule, states, include_response_terms, compute_nac=method != "penalty"
    )

    def state_data(driver):
        active_state = molecule.active_state
        try:
            molecule.active_state = 0
            driver(molecule, P0=molecule.dm, dm_prop="SCF")
        finally:
            molecule.active_state = active_state
        energies = molecule.Etot[0] + molecule.cis_energies[0, (states[0] - 1, states[1] - 1)]
        gradients = -molecule.all_forces[0, states]
        molecule.mecp_state_energies = energies.detach()
        molecule.mecp_energy_gap = (energies[1] - energies[0]).detach()
        return energies, gradients

    def evaluate(driver):
        energies, gradients = state_data(driver)
        if method == "penalty":
            energy, gradient = _penalty_objective(
                energies, gradients, penalty_sigma, penalty_alpha * _HARTREE_TO_EV
            )
        else:
            coupling = _scaled_coupling(energies, molecule.nac[(states[0] - 1, states[1] - 1)][0])
            energy, gradient = _projected_objective(energies, gradients, coupling, gap_weight)
        comment = (
            f"objective = {energy.item():.8f} eV, Eavg = {energies.mean().item():.8f} eV, "
            f"E(S{states[0]}) = {energies[0].item():.8f} eV, "
            f"E(S{states[1]}) = {energies[1].item():.8f} eV, "
            f"gap = {molecule.mecp_energy_gap.item():.8f} eV, method = {method}"
        )
        return energy, gradient, comment

    if method == "composed_step":
        return _run_composed(
            molecule,
            parameters,
            state_data,
            states,
            traj_file,
            output_file,
            run_kwargs,
            gap_tolerance,
            seam_switch_gap,
        )
    return run_geometric(
        molecule,
        parameters,
        evaluate,
        traj_file,
        output_file,
        run_kwargs,
        integrate_gradient=method == "gradient_projection",
        meci=True,
    )

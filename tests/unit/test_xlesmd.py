import torch

from seqm.seqm_functions.XLESMD import (
    _make_symmetric_rotation_projector,
    _symmetric_shadow_hamiltonian,
    augmented_cis_adiabatic_readout,
    augmented_cis_residual,
    augmented_smat,
    augmented_svec,
    compute_dxi2dt2_old_jacobian_gmres,
    make_apply_precond_ordered_lowrank,
    make_apply_precond_rank1,
    make_apply_precond_symmetric_lowrank,
    make_augmented_cis_gauge_projector,
    make_augmented_cis_jvp,
    make_augmented_cis_two_sided_operator,
    make_jvp_xi,
    make_jvp_xi_full_normalized,
    make_jvp_xi_ordered,
    make_jvp_xi_symmetric,
    solve_augmented_cis_newton_gmres,
    solve_augmented_cis_newton_minres,
    solve_for_amplitude_omega,
    solve_for_amplitude_omega_newton,
    solve_for_amplitudes_ordered,
    solve_for_amplitudes_symmetric,
    transport_cis_amplitudes,
)
from seqm.seqm_functions.XLESMD_coupled import (
    compute_dxi2dt2_coupled_gmres,
    horizontal_project_state_block,
    make_apply_precond_constraint_lowrank_coupled,
    make_apply_precond_lambda_coupled,
    make_jvp_xi_coupled,
    make_jvp_xi_exact_orthonormal,
    project_state_subspace_tangent,
    solve_for_amplitudes_coupled,
    solve_for_amplitudes_exact_orthonormal,
)


def _problem(n=7, roots=3):
    torch.manual_seed(17)
    dtype = torch.float64
    D = torch.linspace(1.2, 4.8, n, dtype=dtype).reshape(1, 1, n)
    raw = torch.randn(n, n, dtype=dtype)
    B = 0.12 * (raw + raw.T)
    A = torch.diag(D[0, 0]) + B
    energy, columns = torch.linalg.eigh(A)
    eigenvectors = columns.T[:roots].unsqueeze(0)

    def G_flat(x):
        shape = x.shape
        return (x.reshape(-1, n) @ B.T).reshape(shape)

    def G_jvp(x):
        return G_flat(x)

    return D, B, energy[:roots].unsqueeze(0), eigenvectors, G_flat, G_jvp


def _central_difference(mapper, eta, direction, step=2.0e-6):
    return (mapper(eta + step * direction) - mapper(eta - step * direction)) / (2.0 * step)


def test_augmented_cis_residual_reproduces_ordinary_cis_eigenpairs():
    D, _, energy, exact, G, _ = _problem()

    def H(x):
        return D * x + G(x)

    multiplier = torch.diag_embed(energy)
    f_eta, f_s = augmented_cis_residual(exact, multiplier, H)
    torch.testing.assert_close(f_eta, torch.zeros_like(f_eta), rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(f_s, torch.zeros_like(f_s), rtol=3e-12, atol=3e-12)


def test_augmented_cis_analytic_jvp_matches_finite_difference():
    D, _, energy, exact, G, _ = _problem(roots=2)
    eta = exact + 0.02 * torch.randn_like(exact)
    multiplier = torch.diag_embed(energy) + 0.03 * torch.randn(1, 2, 2, dtype=eta.dtype)
    multiplier = 0.5 * (multiplier + multiplier.transpose(-1, -2))

    def H(x):
        return D * x + G(x)

    jvp = make_augmented_cis_jvp(eta, multiplier, H)
    deta = torch.randn_like(eta)
    dmultiplier = torch.randn_like(multiplier)
    dmultiplier = 0.5 * (dmultiplier + dmultiplier.transpose(-1, -2))
    scale = torch.sqrt(torch.sum(deta.square()) + torch.sum(dmultiplier.square()))
    deta, dmultiplier = deta / scale, dmultiplier / scale
    analytic_eta, analytic_s = jvp((deta, dmultiplier))
    step = 2.0e-6
    plus = augmented_cis_residual(eta + step * deta, multiplier + step * dmultiplier, H)
    minus = augmented_cis_residual(eta - step * deta, multiplier - step * dmultiplier, H)
    finite_eta = (plus[0] - minus[0]) / (2.0 * step)
    finite_s = (plus[1] - minus[1]) / (2.0 * step)
    torch.testing.assert_close(analytic_eta, finite_eta, rtol=2e-8, atol=2e-9)
    torch.testing.assert_close(analytic_s, finite_s, rtol=2e-8, atol=2e-9)


def test_augmented_cis_gmres_solves_the_bordered_newton_equations():
    D, _, energy, exact, G, _ = _problem(n=6, roots=2)
    eta = exact + 0.01 * torch.randn_like(exact)
    multiplier = torch.diag_embed(energy) + 0.02 * torch.randn(1, 2, 2, dtype=eta.dtype)
    multiplier = 0.5 * (multiplier + multiplier.transpose(-1, -2))

    def H(x):
        return D * x + G(x)

    deta, dmultiplier, info = solve_augmented_cis_newton_gmres(
        eta,
        multiplier,
        H,
        {"max_rank": eta.numel() + multiplier.numel(), "err_threshold": 1.0e-11},
        ea_ei_flat=D,
    )
    project = make_augmented_cis_gauge_projector(eta, multiplier)
    j_eta, j_s = project(make_augmented_cis_jvp(eta, multiplier, H)((deta, dmultiplier)))
    f_eta, f_s = project(augmented_cis_residual(eta, multiplier, H))
    torch.testing.assert_close(j_eta, -f_eta, rtol=2e-8, atol=2e-9)
    torch.testing.assert_close(j_s, -f_s, rtol=2e-8, atol=2e-9)
    assert torch.isfinite(info["relative_residual"]).all()


def test_augmented_cis_root_rotation_covariance_and_adiabatic_readout():
    D, _, energy, exact, G, _ = _problem(roots=2)
    angle = torch.tensor(0.37, dtype=exact.dtype)
    rotation0 = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]], dtype=exact.dtype
    ).unsqueeze(0)
    eta = rotation0 @ exact
    multiplier = rotation0 @ torch.diag_embed(energy) @ rotation0.transpose(-1, -2)

    def H(x):
        return D * x + G(x)

    f_eta, f_s = augmented_cis_residual(eta, multiplier, H)
    f_exact_eta, f_exact_s = augmented_cis_residual(exact, torch.diag_embed(energy), H)
    torch.testing.assert_close(f_eta, rotation0 @ f_exact_eta, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(
        f_s, rotation0 @ f_exact_s @ rotation0.transpose(-1, -2), rtol=3e-12, atol=3e-12
    )
    energies, x_ad, _ = augmented_cis_adiabatic_readout(eta, H)
    torch.testing.assert_close(energies, energy, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(
        x_ad @ x_ad.transpose(-1, -2), torch.eye(2, dtype=eta.dtype).unsqueeze(0), rtol=3e-12, atol=3e-12
    )


def test_augmented_cis_orthogonality_residual_and_gauge_projector():
    D, _, energy, exact, G, _ = _problem(roots=2)

    def H(x):
        return D * x + G(x)

    eta = exact.clone()
    eta[:, 1] = eta[:, 0]
    _, f_s = augmented_cis_residual(eta, torch.diag_embed(energy), H)
    assert torch.linalg.matrix_norm(f_s).item() > 0.5

    eta = exact + 0.01 * torch.randn_like(exact)
    multiplier = torch.diag_embed(energy)
    raw = torch.randn(2, 2, dtype=eta.dtype)
    generator = (raw - raw.T).unsqueeze(0)
    vertical = (generator @ eta, generator @ multiplier - multiplier @ generator)
    projected = make_augmented_cis_gauge_projector(eta, multiplier)(vertical)
    torch.testing.assert_close(projected[0], torch.zeros_like(eta), rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(projected[1], torch.zeros_like(multiplier), rtol=2e-11, atol=2e-11)


def test_augmented_cis_gauge_fixed_krylov_is_root_rotation_covariant():
    D, _, energy, exact, G, _ = _problem(n=6, roots=2)
    eta = exact + 0.01 * torch.randn_like(exact)
    multiplier = torch.diag_embed(energy) + 0.01 * torch.randn(1, 2, 2, dtype=eta.dtype)
    multiplier = 0.5 * (multiplier + multiplier.transpose(-1, -2))
    angle = torch.tensor(0.23, dtype=eta.dtype)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle))),
            torch.stack((torch.sin(angle), torch.cos(angle))),
        )
    ).unsqueeze(0)

    def H(x):
        return D * x + G(x)

    params = {"max_rank": eta.numel() + multiplier.numel(), "err_threshold": 1.0e-10}
    deta, dlambda, _ = solve_augmented_cis_newton_gmres(eta, multiplier, H, params, ea_ei_flat=D)
    eta_r, lambda_r = rotation @ eta, rotation @ multiplier @ rotation.transpose(-1, -2)
    deta_r, dlambda_r, info_r = solve_augmented_cis_newton_gmres(eta_r, lambda_r, H, params, ea_ei_flat=D)
    torch.testing.assert_close(deta_r, rotation @ deta, rtol=4e-8, atol=4e-9)
    torch.testing.assert_close(
        dlambda_r, rotation @ dlambda @ rotation.transpose(-1, -2), rtol=4e-8, atol=4e-9
    )
    assert torch.isfinite(info_r["relative_residual"]).all()


def _augmented_inner(left, right):
    """Natural amplitude plus symmetric-multiplier Frobenius product."""
    return torch.sum(left[0].conj() * right[0], dim=(-2, -1)) + torch.sum(
        augmented_svec(left[1]).conj() * augmented_svec(right[1]), dim=-1
    )


def _pack_augmented_pair(pair):
    batch, roots, nov = pair[0].shape
    return torch.cat((pair[0].reshape(batch, roots * nov), augmented_svec(pair[1])), dim=-1)


def _unpack_augmented_pair(vector, roots, nov):
    return vector[:, : roots * nov].reshape(vector.shape[0], roots, nov), augmented_smat(
        vector[:, roots * nov :], roots
    )


def _symmetric_random_like(value):
    result = torch.randn_like(value)
    return 0.5 * (result + result.transpose(-1, -2))


def test_augmented_cis_jacobian_is_self_adjoint_in_svec_metric():
    D, _, energy, exact, G, _ = _problem(n=6, roots=3)
    eta = exact + 0.03 * torch.randn_like(exact)
    multiplier = torch.diag_embed(energy) + 0.02 * _symmetric_random_like(torch.diag_embed(energy))

    def H(x):
        return D * x + G(x)

    jvp = make_augmented_cis_jvp(eta, multiplier, H)
    x = (torch.randn_like(eta), _symmetric_random_like(multiplier))
    y = (torch.randn_like(eta), _symmetric_random_like(multiplier))
    torch.testing.assert_close(
        _augmented_inner(x, jvp(y)), _augmented_inner(jvp(x), y), rtol=3e-12, atol=3e-12
    )


def test_augmented_cis_jvp_uses_one_cis_response_per_application():
    D, _, energy, exact, G, _ = _problem(n=6, roots=2)
    calls = 0

    def H(x):
        nonlocal calls
        calls += 1
        return D * x + G(x)

    multiplier = torch.diag_embed(energy)
    jvp = make_augmented_cis_jvp(exact, multiplier, H)
    direction = (torch.randn_like(exact), _symmetric_random_like(multiplier))
    jvp(direction)
    assert calls == 1


def test_augmented_cis_gauge_null_modes_at_converged_cis_solution():
    D, _, energy, exact, G, _ = _problem(n=6, roots=3)

    def H(x):
        return D * x + G(x)

    multiplier = torch.diag_embed(energy)
    generator = torch.zeros((1, 3, 3), dtype=exact.dtype)
    generator[:, 2, 0] = 1.0
    generator[:, 0, 2] = -1.0
    gauge_vector = (generator @ exact, generator @ multiplier - multiplier @ generator)
    j_eta, j_s = make_augmented_cis_jvp(exact, multiplier, H)(gauge_vector)
    torch.testing.assert_close(j_eta, torch.zeros_like(j_eta), rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(j_s, torch.zeros_like(j_s), rtol=3e-12, atol=3e-12)


def test_augmented_cis_gauge_projector_is_orthogonal_and_idempotent():
    _, _, energy, exact, _, _ = _problem(n=6, roots=3)
    eta = exact + 0.02 * torch.randn_like(exact)
    multiplier = torch.diag_embed(energy) + 0.01 * _symmetric_random_like(torch.diag_embed(energy))
    project = make_augmented_cis_gauge_projector(eta, multiplier)
    x = (torch.randn_like(eta), _symmetric_random_like(multiplier))
    y = (torch.randn_like(eta), _symmetric_random_like(multiplier))
    torch.testing.assert_close(
        _pack_augmented_pair(project(project(x))), _pack_augmented_pair(project(x)), rtol=2e-12, atol=2e-12
    )
    torch.testing.assert_close(
        _augmented_inner(project(x), y), _augmented_inner(x, project(y)), rtol=2e-12, atol=2e-12
    )


def test_augmented_cis_two_sided_operator_is_symmetric_and_minres_matches_dense():
    D, _, energy, exact, G, _ = _problem(n=5, roots=2)
    eta = exact + 0.02 * torch.randn_like(exact)
    multiplier = torch.diag_embed(energy) + 0.02 * _symmetric_random_like(torch.diag_embed(energy))

    def H(x):
        return D * x + G(x)

    params = {"max_rank": 32, "err_threshold": 1.0e-12}
    operator, rhs, project, mhalf, _ = make_augmented_cis_two_sided_operator(
        eta, multiplier, H, params, ea_ei_flat=D
    )
    x = (torch.randn_like(eta), _symmetric_random_like(multiplier))
    y = (torch.randn_like(eta), _symmetric_random_like(multiplier))
    torch.testing.assert_close(
        _augmented_inner(x, operator(y)), _augmented_inner(operator(x), y), rtol=4e-11, atol=4e-11
    )

    roots, nov = eta.shape[1:]
    dimension = roots * nov + roots * (roots + 1) // 2
    eye = torch.eye(dimension, dtype=eta.dtype).unsqueeze(0)
    columns = [
        _pack_augmented_pair(operator(_unpack_augmented_pair(eye[:, column], roots, nov)))
        for column in range(dimension)
    ]
    dense_operator = torch.cat(columns, dim=0).transpose(0, 1)
    dense_y = torch.linalg.pinv(dense_operator) @ _pack_augmented_pair(rhs)[0]
    dense_pair = project(mhalf(_unpack_augmented_pair(dense_y.unsqueeze(0), roots, nov)))
    deta, dlambda, info = solve_augmented_cis_newton_minres(eta, multiplier, H, params, ea_ei_flat=D)
    torch.testing.assert_close(deta, dense_pair[0], rtol=2e-8, atol=2e-9)
    torch.testing.assert_close(dlambda, dense_pair[1], rtol=2e-8, atol=2e-9)
    assert info["solver"] == "minres"


def test_augmented_cis_schur_preconditioner_iteration_count_comparison():
    D, _, energy, exact, G, _ = _problem(n=8, roots=3)
    eta = exact + 0.04 * torch.randn_like(exact)
    multiplier = torch.diag_embed(energy) + 0.03 * _symmetric_random_like(torch.diag_embed(energy))

    def H(x):
        return D * x + G(x)

    base = {"max_rank": 64, "err_threshold": 1.0e-10}
    _, _, with_schur = solve_augmented_cis_newton_minres(
        eta, multiplier, H, {**base, "augmented_schur_preconditioner": True}, ea_ei_flat=D
    )
    _, _, without_schur = solve_augmented_cis_newton_minres(
        eta, multiplier, H, {**base, "augmented_schur_preconditioner": False}, ea_ei_flat=D
    )
    assert with_schur["lambda_schur_enabled"].item()
    assert not without_schur["lambda_schur_enabled"].item()
    assert with_schur["rank"].item() > 0
    assert without_schur["rank"].item() > 0


def test_all_five_formulations_reduce_to_ordinary_cis_at_self_consistency():
    D, _, energy, exact, G, _ = _problem()

    independent_linearized, omega_1a = solve_for_amplitude_omega(exact, D, G(exact))
    independent_exact, omega_1b = solve_for_amplitude_omega_newton(
        exact, D, G(exact), omega_init=energy, tol=1.0e-12
    )
    ordered, omega_ordered = solve_for_amplitudes_ordered(exact, D, G_eta=G(exact))
    coupled_linearized, lambda_2a = solve_for_amplitudes_coupled(exact, D, G(exact))
    coupled_exact, lambda_2b, info_2b = solve_for_amplitudes_exact_orthonormal(
        exact, D, G(exact), initial_xi=exact, tolerance=1.0e-12
    )

    for amplitudes in (independent_linearized, independent_exact, ordered, coupled_linearized, coupled_exact):
        torch.testing.assert_close(amplitudes, exact, rtol=4e-12, atol=4e-12)

    torch.testing.assert_close(omega_1a, energy, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(omega_1b, energy, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(torch.diagonal(omega_ordered, dim1=-2, dim2=-1), energy)
    torch.testing.assert_close(lambda_2a, torch.diag_embed(energy), rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(lambda_2b, torch.diag_embed(energy), rtol=3e-12, atol=3e-12)
    assert info_2b["converged"]


def test_independent_linearized_solver_fixed_point_constraint_and_shadow_identity():
    D, B, energy, exact, G, _ = _problem()
    xi, multiplier = solve_for_amplitude_omega(exact, D, G(exact))
    torch.testing.assert_close(xi, exact, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(multiplier, energy, rtol=2e-12, atol=2e-12)

    eta = exact + 0.025 * torch.randn_like(exact)
    xi, _ = solve_for_amplitude_omega(eta, D, G(eta))
    constraint = torch.sum((2.0 * xi - eta) * eta, dim=-1)
    torch.testing.assert_close(constraint, torch.ones_like(constraint), rtol=2e-12, atol=2e-12)

    ordinary = torch.sum(xi * D * xi, dim=-1) + torch.sum(xi * G(xi), dim=-1)
    shadow = torch.sum(xi * D * xi, dim=-1) + torch.sum((2.0 * xi - eta) * G(eta), dim=-1)
    quadratic_error = torch.sum((xi - eta) * G(xi - eta), dim=-1)
    torch.testing.assert_close(ordinary - shadow, quadratic_error, rtol=2e-11, atol=2e-11)


def test_independent_linearized_jvp_matches_finite_difference():
    D, _, _, exact, G, G_jvp = _problem()
    eta = exact + 0.02 * torch.randn_like(exact)
    xi, multiplier = solve_for_amplitude_omega(eta, D, G(eta))
    jvp = make_jvp_xi(D, eta, xi, multiplier, G_jvp, 1, D.shape[-1])
    direction = torch.randn_like(eta)
    direction /= torch.linalg.vector_norm(direction)

    analytic = jvp(direction)
    finite_difference = _central_difference(
        lambda trial: solve_for_amplitude_omega(trial, D, G(trial))[0], eta, direction
    )
    torch.testing.assert_close(analytic, finite_difference, rtol=2e-7, atol=2e-9)


def test_independent_exact_solver_and_jvp_follow_selected_branch():
    D, _, energy, exact, G, G_jvp = _problem(roots=2)
    eta = exact + 5.0e-4 * torch.randn_like(exact)
    xi, multiplier = solve_for_amplitude_omega_newton(eta, D, G(eta), omega_init=energy, pole_eps=1e-9)
    torch.testing.assert_close(
        torch.sum(xi.square(), dim=-1), torch.ones_like(multiplier), rtol=2e-10, atol=2e-10
    )
    stationarity = (D - multiplier.unsqueeze(-1)) * xi + G(eta)
    torch.testing.assert_close(stationarity, torch.zeros_like(stationarity), rtol=1e-9, atol=1e-10)

    jvp = make_jvp_xi_full_normalized(D, eta, xi, multiplier, G_jvp, 1, D.shape[-1])
    direction = torch.randn_like(eta)
    direction /= torch.linalg.vector_norm(direction)
    analytic = jvp(direction)
    finite_difference = _central_difference(
        lambda trial: solve_for_amplitude_omega_newton(
            trial, D, G(trial), omega_init=multiplier, pole_eps=1e-9
        )[0],
        eta,
        direction,
        step=5.0e-7,
    )
    torch.testing.assert_close(analytic, finite_difference, rtol=2e-5, atol=2e-7)


def test_coupled_linearized_solver_constraint_identity_and_jvp():
    D, _, _, exact, G, G_jvp = _problem()
    eta = exact + 0.01 * torch.randn_like(exact)
    xi, multiplier = solve_for_amplitudes_coupled(eta, D, G(eta))
    constraint = torch.einsum("bjn,bkn->bjk", eta, xi)
    expected = 0.5 * (
        torch.eye(eta.shape[1], dtype=eta.dtype).unsqueeze(0) + torch.einsum("bjn,bkn->bjk", eta, eta)
    )
    torch.testing.assert_close(constraint, expected, rtol=3e-12, atol=3e-12)
    # States are rows: Case 2a is D xi + G(eta) = Lambda eta.
    electronic_residual = D * xi + G(eta) - multiplier @ eta
    torch.testing.assert_close(
        electronic_residual, torch.zeros_like(electronic_residual), rtol=3e-12, atol=3e-12
    )

    weak_constraint = eta @ xi.transpose(-1, -2) + xi @ eta.transpose(-1, -2)
    weak_expected = eta @ eta.transpose(-1, -2) + torch.eye(eta.shape[1], dtype=eta.dtype).unsqueeze(0)
    torch.testing.assert_close(weak_constraint, weak_expected, rtol=3e-12, atol=3e-12)

    delta = xi - eta
    orthogonality_error = torch.einsum("bjn,bkn->bjk", xi, xi) - torch.eye(
        eta.shape[1], dtype=eta.dtype
    ).unsqueeze(0)
    torch.testing.assert_close(
        orthogonality_error, torch.einsum("bjn,bkn->bjk", delta, delta), rtol=2e-10, atol=2e-10
    )

    jvp = make_jvp_xi_coupled(D, eta, xi, multiplier, G_jvp, 1, D.shape[-1])
    direction = torch.randn_like(eta)
    direction /= torch.linalg.vector_norm(direction)
    analytic = jvp(direction)
    finite_difference = _central_difference(
        lambda trial: solve_for_amplitudes_coupled(trial, D, G(trial))[0], eta, direction
    )
    torch.testing.assert_close(analytic, finite_difference, rtol=4e-7, atol=4e-9)


def test_coupled_linearized_has_rotational_null_mode_at_fixed_point():
    D, _, energy, exact, G, G_jvp = _problem()
    xi, multiplier = solve_for_amplitudes_coupled(exact, D, G(exact))
    torch.testing.assert_close(xi, exact, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(multiplier, torch.diag_embed(energy), rtol=3e-12, atol=3e-12)
    jvp = make_jvp_xi_coupled(D, exact, xi, multiplier, G_jvp, 1, D.shape[-1])
    raw = torch.randn(exact.shape[1], exact.shape[1], dtype=exact.dtype)
    skew = raw - raw.T
    vertical = torch.einsum("jk,bkn->bjn", skew, exact)
    residual_jvp = jvp(vertical) - vertical
    torch.testing.assert_close(residual_jvp, torch.zeros_like(residual_jvp), rtol=1e-9, atol=1e-10)


def test_coupled_horizontal_projection_removes_only_state_rotations():
    _, _, _, exact, _, _ = _problem()
    raw = torch.randn(exact.shape[1], exact.shape[1], dtype=exact.dtype)
    skew = raw - raw.T
    vertical = torch.einsum("jk,bkn->bjn", skew, exact)
    torch.testing.assert_close(
        horizontal_project_state_block(exact, vertical), torch.zeros_like(vertical), rtol=2e-12, atol=2e-12
    )

    direction = torch.randn_like(exact)
    horizontal = horizontal_project_state_block(exact, direction)
    overlap = horizontal @ exact.transpose(-1, -2)
    torch.testing.assert_close(overlap, overlap.transpose(-1, -2), rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(
        horizontal_project_state_block(exact, horizontal), horizontal, rtol=2e-12, atol=2e-12
    )


def test_coupled_subspace_projection_removes_all_internal_coordinates():
    _, _, _, exact, _, _ = _problem()
    internal = torch.randn(exact.shape[1], exact.shape[1], dtype=exact.dtype) @ exact[0]
    projected = project_state_subspace_tangent(exact, internal.unsqueeze(0))
    torch.testing.assert_close(projected, torch.zeros_like(projected), rtol=2e-12, atol=2e-12)

    direction = torch.randn_like(exact)
    tangent = project_state_subspace_tangent(exact, direction)
    torch.testing.assert_close(
        tangent @ exact.transpose(-1, -2),
        torch.zeros((1, exact.shape[1], exact.shape[1]), dtype=exact.dtype),
        rtol=2e-12,
        atol=2e-12,
    )


def _dense_projected_correction(eta, xi, jvp):
    dimension = eta.shape[1] * eta.shape[2]
    columns = []
    for column in range(dimension):
        basis = torch.zeros_like(eta).reshape(eta.shape[0], dimension)
        basis[:, column] = 1.0
        basis = basis.reshape_as(eta)
        basis = project_state_subspace_tangent(eta, basis)
        columns.append(project_state_subspace_tangent(eta, basis - jvp(basis)).reshape(eta.shape[0], -1))
    operator = torch.stack(columns, dim=-1)
    rhs = project_state_subspace_tangent(eta, xi - eta).reshape(eta.shape[0], -1)
    dense = torch.bmm(torch.linalg.pinv(operator), rhs.unsqueeze(-1)).squeeze(-1).reshape_as(eta)
    return project_state_subspace_tangent(eta, dense), operator, rhs


def test_coupled_linearized_projected_gmres_matches_dense_pseudoinverse():
    D, _, _, exact, G, G_jvp = _problem(n=6, roots=2)
    eta = exact + 0.015 * torch.randn_like(exact)
    xi, multiplier = solve_for_amplitudes_coupled(eta, D, G(eta))
    jvp = make_jvp_xi_coupled(D, eta, xi, multiplier, G_jvp, 1, D.shape[-1])
    correction, info = compute_dxi2dt2_coupled_gmres(
        eta, xi, jvp, {"max_rank": eta.numel(), "err_threshold": 1.0e-11}, return_info=True
    )
    dense, operator, rhs = _dense_projected_correction(eta, xi, jvp)
    torch.testing.assert_close(correction, dense, rtol=3e-8, atol=3e-9)
    residual = torch.bmm(operator, correction.reshape(1, -1, 1)).squeeze(-1) - rhs
    assert torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(rhs) < 2.0e-9
    assert info["converged"].all()


def test_coupled_constraint_preconditioned_gmres_matches_dense_pseudoinverse():
    D, _, _, exact, G, G_jvp = _problem(n=6, roots=2)
    eta = exact + 0.015 * torch.randn_like(exact)
    eta = eta / torch.linalg.vector_norm(eta, dim=-1, keepdim=True)
    xi, multiplier = solve_for_amplitudes_coupled(eta, D, G(eta))
    jvp = make_jvp_xi_coupled(D, eta, xi, multiplier, G_jvp, 1, D.shape[-1])
    kernel_inverse = make_apply_precond_constraint_lowrank_coupled(D, eta, xi, multiplier)

    def right_preconditioner(v):
        return -project_state_subspace_tangent(eta, kernel_inverse(v))

    correction, info = compute_dxi2dt2_coupled_gmres(
        eta,
        xi,
        jvp,
        {"max_rank": eta.numel(), "err_threshold": 1.0e-11},
        preconditioner=right_preconditioner,
        return_info=True,
    )
    dense, operator, rhs = _dense_projected_correction(eta, xi, jvp)
    torch.testing.assert_close(correction, dense, rtol=3e-8, atol=3e-9)
    residual = torch.bmm(operator, correction.reshape(1, -1, 1)).squeeze(-1) - rhs
    assert torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(rhs) < 2.0e-9
    assert info["converged"].all()


def test_coupled_lambda_preconditioner_targets_i_minus_j_with_the_correct_sign():
    D, _, _, exact, G, _ = _problem(n=6, roots=2)
    eta = exact + 0.015 * torch.randn_like(exact)
    _, multiplier = solve_for_amplitudes_coupled(eta, D, G(eta))
    inverse_j_minus_i = make_apply_precond_lambda_coupled(D, multiplier, tau=0.0)
    direction = torch.randn_like(eta)
    j_minus_i_direction = (multiplier @ direction) / D - direction

    torch.testing.assert_close(inverse_j_minus_i(j_minus_i_direction), direction, rtol=3e-11, atol=3e-12)
    # The production caller negates this inverse before right-preconditioning
    # GMRES, so that it approximates (I-J)^-1 rather than (J-I)^-1.
    torch.testing.assert_close(-inverse_j_minus_i(-j_minus_i_direction), direction, rtol=3e-11, atol=3e-12)


def test_coupled_linearized_map_and_block_energy_are_rotation_covariant():
    D, _, _, exact, G, _ = _problem(n=7, roots=3)
    eta = exact + 0.01 * torch.randn_like(exact)
    xi, multiplier = solve_for_amplitudes_coupled(eta, D, G(eta))
    rotation, _ = torch.linalg.qr(torch.randn(eta.shape[1], eta.shape[1], dtype=eta.dtype))
    rotated_eta = rotation.unsqueeze(0) @ eta
    rotated_xi, rotated_multiplier = solve_for_amplitudes_coupled(rotated_eta, D, G(rotated_eta))
    torch.testing.assert_close(rotated_xi, rotation.unsqueeze(0) @ xi, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(
        rotated_multiplier,
        rotation.unsqueeze(0) @ multiplier @ rotation.T.unsqueeze(0),
        rtol=2e-11,
        atol=2e-11,
    )

    energy = torch.sum(xi * D * xi) + torch.sum((2.0 * xi - eta) * G(eta))
    rotated_energy = torch.sum(rotated_xi * D * rotated_xi) + torch.sum(
        (2.0 * rotated_xi - rotated_eta) * G(rotated_eta)
    )
    torch.testing.assert_close(rotated_energy, energy, rtol=2e-12, atol=2e-12)


def test_exact_coupled_solver_and_bordered_jvp():
    D, _, _, exact, G, G_jvp = _problem(n=6, roots=2)
    xi, multiplier, info = solve_for_amplitudes_exact_orthonormal(exact, D, G(exact), tolerance=1e-12)
    assert info["converged"]
    torch.testing.assert_close(xi, exact, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(
        xi @ xi.transpose(-1, -2), torch.eye(xi.shape[1], dtype=xi.dtype).unsqueeze(0), rtol=3e-12, atol=3e-12
    )
    torch.testing.assert_close(multiplier, multiplier.transpose(-1, -2), rtol=0.0, atol=0.0)
    assert torch.max(info["electronic_residual_norm"]) <= 1.0e-12
    assert torch.max(info["constraint_residual_norm"]) <= 1.0e-12
    torch.testing.assert_close(
        info["residual_norm"].square(),
        info["electronic_residual_norm"].square() + info["constraint_residual_norm"].square(),
    )

    jvp = make_jvp_xi_exact_orthonormal(D, xi, multiplier, G_jvp, 1, D.shape[-1])
    direction = torch.randn_like(exact)
    direction /= torch.linalg.vector_norm(direction)
    analytic = jvp(direction)

    def solve_map(trial):
        solved, _, solve_info = solve_for_amplitudes_exact_orthonormal(
            trial, D, G(trial), initial_xi=xi, tolerance=2e-12, max_iter=4000
        )
        assert solve_info["converged"], solve_info
        return solved

    finite_difference = _central_difference(solve_map, exact, direction, step=2e-5)
    torch.testing.assert_close(analytic, finite_difference, rtol=3e-4, atol=3e-6)


def test_exact_coupled_projected_gmres_matches_dense_pseudoinverse():
    D, _, _, exact, G, G_jvp = _problem(n=5, roots=2)
    eta = exact + 2.0e-3 * torch.randn_like(exact)
    xi, multiplier, solve_info = solve_for_amplitudes_exact_orthonormal(
        eta, D, G(eta), initial_xi=eta, tolerance=1.0e-12, max_iter=2000
    )
    assert solve_info["converged"], solve_info
    jvp = make_jvp_xi_exact_orthonormal(D, xi, multiplier, G_jvp, 1, D.shape[-1])
    correction, info = compute_dxi2dt2_coupled_gmres(
        eta, xi, jvp, {"max_rank": eta.numel(), "err_threshold": 1.0e-10}, return_info=True
    )
    dense, operator, rhs = _dense_projected_correction(eta, xi, jvp)
    torch.testing.assert_close(correction, dense, rtol=2e-7, atol=2e-8)
    residual = torch.bmm(operator, correction.reshape(1, -1, 1)).squeeze(-1) - rhs
    assert torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(rhs) < 2.0e-8
    assert info["converged"].all()


def test_ordered_linearized_solver_fixed_point_constraints_and_jvp():
    D, _, _, exact, G, G_jvp = _problem()
    xi_exact, _ = solve_for_amplitudes_ordered(exact, D, G)
    torch.testing.assert_close(xi_exact, exact, rtol=3e-12, atol=3e-12)

    eta = exact + 0.01 * torch.randn_like(exact)
    xi, multiplier = solve_for_amplitudes_ordered(eta, D, G)
    lhs = torch.einsum("bjn,bkn->bjk", eta, 2.0 * xi - eta)
    for state in range(eta.shape[1]):
        torch.testing.assert_close(
            lhs[:, : state + 1, state],
            torch.eye(eta.shape[1], dtype=eta.dtype)[None, : state + 1, state],
            rtol=3e-12,
            atol=3e-12,
        )

    jvp = make_jvp_xi_ordered(D, eta, multiplier, G_jvp, 1, D.shape[-1])
    direction = torch.randn_like(eta)
    direction /= torch.linalg.vector_norm(direction)
    analytic = jvp(direction)
    finite_difference = _central_difference(
        lambda trial: solve_for_amplitudes_ordered(trial, D, G)[0], eta, direction
    )
    torch.testing.assert_close(analytic, finite_difference, rtol=5e-7, atol=5e-9)


def test_symmetric_linearized_solver_full_constraints_and_jvp():
    D, _, energy, exact, G, G_jvp = _problem()
    xi_exact, multiplier_exact = solve_for_amplitudes_symmetric(exact, D, G)
    torch.testing.assert_close(xi_exact, exact, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(multiplier_exact, torch.diag_embed(energy), rtol=3e-12, atol=3e-12)

    eta = exact + 0.01 * torch.randn_like(exact)
    xi, multipliers = solve_for_amplitudes_symmetric(eta, D, G)
    lhs = 2.0 * eta @ xi.transpose(-1, -2)
    rhs = eta @ eta.transpose(-1, -2) + torch.eye(eta.shape[1], dtype=eta.dtype)[None]
    torch.testing.assert_close(lhs, rhs, rtol=3e-12, atol=3e-12)
    assert multipliers.shape == (eta.shape[0], eta.shape[1], eta.shape[1])

    jvp = make_jvp_xi_symmetric(D, eta, multipliers, G_jvp, 1, D.shape[-1])
    for _ in range(3):
        direction = torch.randn_like(eta)
        direction /= torch.linalg.vector_norm(direction)
        analytic = jvp(direction)
        finite_difference = _central_difference(
            lambda trial: solve_for_amplitudes_symmetric(trial, D, G)[0], eta, direction
        )
        torch.testing.assert_close(analytic, finite_difference, rtol=5e-7, atol=5e-9)


def test_symmetric_shadow_states_are_gauge_invariant_and_recover_cis():
    D, _, energy, exact, G, _ = _problem(n=7, roots=3)
    xi, _ = solve_for_amplitudes_symmetric(exact, D, G_eta=G(exact))
    H = _symmetric_shadow_hamiltonian(xi, exact, D, G(exact))
    values, vectors = torch.linalg.eigh(H)
    physical = vectors.transpose(-1, -2) @ xi
    eta_physical = vectors.transpose(-1, -2) @ exact
    row_energy = torch.sum(physical * physical * D, dim=-1) + torch.sum(
        (2.0 * physical - eta_physical) * G(eta_physical), dim=-1
    )
    torch.testing.assert_close(values, energy, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(row_energy, values, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(
        (physical @ exact.transpose(-1, -2)).abs(),
        torch.eye(exact.shape[1], dtype=exact.dtype)[None],
        rtol=3e-12,
        atol=3e-12,
    )

    eta = exact + 0.01 * torch.randn_like(exact)
    xi, _ = solve_for_amplitudes_symmetric(eta, D, G_eta=G(eta))
    H = _symmetric_shadow_hamiltonian(xi, eta, D, G(eta))
    values, vectors = torch.linalg.eigh(H)
    physical = vectors.transpose(-1, -2) @ xi
    eta_physical = vectors.transpose(-1, -2) @ eta
    row_energy = torch.sum(physical * physical * D, dim=-1) + torch.sum(
        (2.0 * physical - eta_physical) * G(eta_physical), dim=-1
    )
    torch.testing.assert_close(row_energy, values, rtol=3e-12, atol=3e-12)
    rotation, _ = torch.linalg.qr(torch.randn(1, eta.shape[1], eta.shape[1], dtype=eta.dtype))
    eta_rotated = rotation @ eta
    xi_rotated, _ = solve_for_amplitudes_symmetric(eta_rotated, D, G_eta=G(eta_rotated))
    H_rotated = _symmetric_shadow_hamiltonian(xi_rotated, eta_rotated, D, G(eta_rotated))
    values_rotated, vectors_rotated = torch.linalg.eigh(H_rotated)
    physical_rotated = vectors_rotated.transpose(-1, -2) @ xi_rotated
    torch.testing.assert_close(values_rotated, values, rtol=2e-12, atol=2e-12)
    signs = torch.sign(torch.sum(physical_rotated * physical, dim=-1, keepdim=True))
    torch.testing.assert_close(signs * physical_rotated, physical, rtol=2e-10, atol=2e-10)


def test_symmetric_gmres_projects_all_rotational_gauge_modes():
    D, _, _, exact, _, _ = _problem(n=5, roots=3)
    rhs = torch.randn_like(exact)
    xi = exact + rhs
    project = _make_symmetric_rotation_projector(exact)

    def jvp(v):
        return v - project(v)

    correction, info = compute_dxi2dt2_old_jacobian_gmres(
        exact,
        xi,
        torch.zeros((1, exact.shape[1], exact.shape[1]), dtype=exact.dtype),
        D,
        lambda v: v,
        1,
        D.shape[-1],
        {
            "max_rank": exact.numel(),
            "err_threshold": 1.0e-12,
            "krylov_root_mode": "coupled",
            "constraint_mode": "symmetric_linearized",
        },
        jvp_xi=jvp,
        return_info=True,
        constraint_mode="symmetric_linearized",
    )
    torch.testing.assert_close(correction, project(rhs), rtol=3e-10, atol=3e-11)
    torch.testing.assert_close(project(correction - jvp(correction)) - project(rhs), torch.zeros_like(rhs))
    torch.testing.assert_close(
        info["quotient_gauge_overlap"], torch.zeros_like(info["quotient_gauge_overlap"])
    )
    assert not info["rotational_regularized"].any()

    trial = torch.randn_like(exact)
    projected = project(trial)
    torch.testing.assert_close(project(projected), projected, rtol=3e-12, atol=3e-12)
    gauge_overlap = (projected @ exact.transpose(-1, -2)).transpose(-1, -2)
    torch.testing.assert_close(
        gauge_overlap - gauge_overlap.transpose(-1, -2), torch.zeros_like(gauge_overlap)
    )


def test_ordered_lowrank_preconditioner_inverts_ordered_jacobian_without_kernel_response():
    D, _, _, exact, G, _ = _problem(n=6, roots=3)
    eta = exact + 0.01 * torch.randn_like(exact)
    xi, multiplier = solve_for_amplitudes_ordered(eta, D, G_eta=G(eta))

    def zero_kernel(v):
        return torch.zeros_like(v)

    jvp = make_jvp_xi_ordered(D, eta, multiplier, zero_kernel, 1, D.shape[-1], G_eta=G(eta))
    n = D.shape[-1]
    columns = []
    for column in range(eta.shape[1] * n):
        basis = torch.zeros_like(eta).reshape(1, -1)
        basis[:, column] = 1.0
        basis = basis.reshape_as(eta)
        columns.append((jvp(basis) - basis).reshape(1, -1))
    operator = torch.stack(columns, dim=-1)[0]
    rhs = torch.randn_like(eta).reshape(1, -1)
    for tau in (0.0, 0.07):
        preconditioner = make_apply_precond_ordered_lowrank(D, eta, multiplier, G(eta), tau=tau)
        expected = (
            torch.linalg.solve(
                operator - tau * torch.eye(operator.shape[-1], dtype=eta.dtype), rhs.unsqueeze(-1)
            )
            .squeeze(-1)
            .reshape_as(eta)
        )
        torch.testing.assert_close(preconditioner(rhs.reshape_as(eta)), expected, rtol=3e-10, atol=3e-11)


def test_symmetric_lowrank_preconditioner_inverts_symmetric_jacobian_without_kernel_response():
    D, _, _, exact, G, _ = _problem(n=6, roots=3)
    eta = exact + 0.01 * torch.randn_like(exact)
    _, multipliers = solve_for_amplitudes_symmetric(eta, D, G_eta=G(eta))

    def zero_kernel(v):
        return torch.zeros_like(v)

    jvp = make_jvp_xi_symmetric(D, eta, multipliers, zero_kernel, 1, D.shape[-1], G_eta=G(eta))
    dimension = eta.shape[1] * eta.shape[2]
    columns = []
    for column in range(dimension):
        basis = torch.zeros_like(eta).reshape(1, dimension)
        basis[:, column] = 1.0
        columns.append((jvp(basis.reshape_as(eta)) - basis.reshape_as(eta)).reshape(1, -1))
    operator = torch.stack(columns, dim=-1)[0]
    rhs = torch.randn_like(eta).reshape(1, -1)
    for tau in (0.0, 0.07):
        preconditioner = make_apply_precond_symmetric_lowrank(D, eta, multipliers, G(eta), tau=tau)
        expected = (
            torch.linalg.solve(
                operator - tau * torch.eye(operator.shape[-1], dtype=eta.dtype), rhs.unsqueeze(-1)
            )
            .squeeze(-1)
            .reshape_as(eta)
        )
        torch.testing.assert_close(preconditioner(rhs.reshape_as(eta)), expected, rtol=3e-10, atol=3e-11)


def test_full_rank_gmres_matches_dense_inverse_action():
    D, _, _, exact, G, G_jvp = _problem(n=6, roots=2)
    eta = exact + 0.015 * torch.randn_like(exact)
    xi, multiplier = solve_for_amplitude_omega(eta, D, G(eta))
    jvp = make_jvp_xi(D, eta, xi, multiplier, G_jvp, 1, D.shape[-1])
    correction = compute_dxi2dt2_old_jacobian_gmres(
        eta,
        xi,
        multiplier,
        D,
        G_jvp,
        1,
        D.shape[-1],
        {"max_rank": D.shape[-1], "err_threshold": 1e-12},
        jvp_xi=jvp,
    )

    columns = []
    for col in range(D.shape[-1]):
        basis = torch.zeros_like(eta)
        basis[..., col] = 1.0
        columns.append(basis - jvp(basis))
    operator = torch.stack(columns, dim=-1)
    dense = torch.linalg.solve(operator, (xi - eta).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(correction, dense, rtol=2e-8, atol=2e-9)


def test_right_preconditioned_full_rank_gmres_matches_dense_inverse_action():
    D, _, _, exact, G, G_jvp = _problem(n=6, roots=2)
    eta = exact + 0.015 * torch.randn_like(exact)
    xi, multiplier = solve_for_amplitude_omega(eta, D, G(eta))
    jvp = make_jvp_xi(D, eta, xi, multiplier, G_jvp, 1, D.shape[-1])
    kernel_inverse = make_apply_precond_rank1(D, eta, xi, multiplier)

    def right_preconditioner(v):
        return -kernel_inverse(v.reshape(-1, v.shape[-1])).reshape_as(v)

    correction = compute_dxi2dt2_old_jacobian_gmres(
        eta,
        xi,
        multiplier,
        D,
        G_jvp,
        1,
        D.shape[-1],
        {"max_rank": D.shape[-1], "err_threshold": 1e-12},
        jvp_xi=jvp,
        preconditioner=right_preconditioner,
    )
    columns = []
    for col in range(D.shape[-1]):
        basis = torch.zeros_like(eta)
        basis[..., col] = 1.0
        columns.append(basis - jvp(basis))
    operator = torch.stack(columns, dim=-1)
    dense = torch.linalg.solve(operator, (xi - eta).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(correction, dense, rtol=2e-8, atol=2e-9)


def test_coupled_root_gmres_matches_ordered_dense_inverse_action():
    D, _, _, exact, G, G_jvp = _problem(n=5, roots=3)
    eta = exact + 0.01 * torch.randn_like(exact)
    xi, multiplier = solve_for_amplitudes_ordered(eta, D, G_eta=G(eta))
    jvp = make_jvp_xi_ordered(D, eta, multiplier, G_jvp, 1, D.shape[-1], G_eta=G(eta))
    preconditioner_shapes = []

    def identity_preconditioner(v):
        preconditioner_shapes.append(v.shape)
        return v

    size = eta.shape[1] * eta.shape[2]
    correction, info = compute_dxi2dt2_old_jacobian_gmres(
        eta,
        xi,
        multiplier,
        D,
        G_jvp,
        1,
        D.shape[-1],
        {"max_rank": size, "err_threshold": 1e-12, "krylov_root_mode": "coupled"},
        jvp_xi=jvp,
        preconditioner=identity_preconditioner,
        return_info=True,
    )

    columns = []
    for col in range(size):
        basis = torch.zeros_like(eta).reshape(1, size)
        basis[:, col] = 1.0
        basis = basis.reshape_as(eta)
        columns.append((basis - jvp(basis)).reshape(1, size))
    operator = torch.stack(columns, dim=-1)
    dense = torch.linalg.solve(operator, (xi - eta).reshape(1, size, 1)).reshape_as(eta)

    assert all(shape == eta.shape for shape in preconditioner_shapes)
    assert info["root_mode"] == "coupled"
    assert info["relative_residual"].shape == (eta.shape[0],)
    torch.testing.assert_close(correction, dense, rtol=3e-8, atol=3e-9)


def test_ordered_gmres_regularizes_a_soft_root_rotation_to_the_requested_max_gain():
    """A near-null labelled-root rotation is capped without shifting other directions."""
    eta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float64)
    rotation = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float64)
    xi = eta + rotation
    multiplier = torch.zeros((1, 2, 2), dtype=torch.float64)
    q = rotation / torch.linalg.vector_norm(rotation)
    softness = 1.0e-3

    # K = I - J has eigenvalue ``softness`` only along q and is identity on
    # its orthogonal complement.
    def jvp(v):
        K_v = v - (1.0 - softness) * q * torch.sum(q * v, dim=(1, 2), keepdim=True)
        return v - K_v

    correction, info = compute_dxi2dt2_old_jacobian_gmres(
        eta,
        xi,
        multiplier,
        torch.ones_like(eta),
        lambda v: v,
        1,
        3,
        {
            "max_rank": 1,
            "err_threshold": 1.0e-12,
            "krylov_root_mode": "coupled",
            "rotational_max_gain": 20.0,
            "constraint_mode": "ordered_linearized",
        },
        jvp_xi=jvp,
        return_info=True,
    )

    torch.testing.assert_close(correction, 20.0 * rotation, rtol=2e-12, atol=2e-12)
    assert info["rotational_regularized"].item()
    torch.testing.assert_close(info["rotational_pair_shift"], torch.tensor([[0.049]], dtype=torch.float64))


def test_jacobian_regularization_shifts_the_gmres_response_operator():
    eta = torch.zeros((1, 1, 2), dtype=torch.float64)
    xi = torch.ones_like(eta)
    gaps = torch.ones((1, 1, 2), dtype=torch.float64)
    multiplier = torch.zeros((1, 1), dtype=torch.float64)
    jvp = lambda v: 0.5 * v

    correction = compute_dxi2dt2_old_jacobian_gmres(
        eta,
        xi,
        multiplier,
        gaps,
        lambda v: v,
        1,
        2,
        {"max_rank": 1, "err_threshold": 1.0e-12, "jacobian_regularization": 0.25},
        jvp_xi=jvp,
    )

    torch.testing.assert_close(correction, torch.full_like(correction, 1.0 / 0.75), rtol=1e-12, atol=1e-12)


def test_amplitude_transport_preserves_ao_transition_density_under_gauge_rotation():
    torch.manual_seed(4)
    dtype = torch.float64
    nocc, nvirt, nao = 3, 4, 9
    Cocc, _ = torch.linalg.qr(torch.randn(nao, nocc, dtype=dtype))
    raw_v = torch.randn(nao, nvirt, dtype=dtype)
    raw_v = raw_v - Cocc @ (Cocc.T @ raw_v)
    Cvirt, _ = torch.linalg.qr(raw_v)
    Qocc, _ = torch.linalg.qr(torch.randn(nocc, nocc, dtype=dtype))
    Qvirt, _ = torch.linalg.qr(torch.randn(nvirt, nvirt, dtype=dtype))
    eta = torch.randn(2, nocc, nvirt, dtype=dtype)

    Cocc_new = Cocc @ Qocc
    Cvirt_new = Cvirt @ Qvirt
    transported = transport_cis_amplitudes(eta, Qocc.T, Qvirt.T, polar_unitarize=True)
    old_ao = torch.einsum("mi,ria,na->rmn", Cocc, eta, Cvirt)
    new_ao = torch.einsum("mi,ria,na->rmn", Cocc_new, transported, Cvirt_new)
    torch.testing.assert_close(new_ao, old_ao, rtol=2e-12, atol=2e-12)

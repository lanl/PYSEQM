import torch

from seqm.seqm_functions.XLESMD import (
    compute_dxi2dt2_old_jacobian_gmres,
    make_apply_precond_rank1,
    make_jvp_xi,
    make_jvp_xi_full_normalized,
    make_jvp_xi_ordered,
    solve_for_amplitude_omega,
    solve_for_amplitude_omega_newton,
    solve_for_amplitudes_ordered,
    transport_cis_amplitudes,
)
from seqm.seqm_functions.XLESMD_coupled import (
    compute_dxi2dt2_coupled_gmres,
    horizontal_project_state_block,
    make_apply_precond_constraint_lowrank_coupled,
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
    D, _, _, exact, G, G_jvp = _problem()
    xi, multiplier = solve_for_amplitudes_coupled(exact, D, G(exact))
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


def test_coupled_linearized_map_and_block_energy_are_rotation_covariant():
    D, _, _, exact, G, _ = _problem(n=7, roots=3)
    eta = exact + 0.01 * torch.randn_like(exact)
    xi, _ = solve_for_amplitudes_coupled(eta, D, G(eta))
    rotation, _ = torch.linalg.qr(torch.randn(eta.shape[1], eta.shape[1], dtype=eta.dtype))
    rotated_eta = rotation.unsqueeze(0) @ eta
    rotated_xi, _ = solve_for_amplitudes_coupled(rotated_eta, D, G(rotated_eta))
    torch.testing.assert_close(rotated_xi, rotation.unsqueeze(0) @ xi, rtol=2e-11, atol=2e-11)

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

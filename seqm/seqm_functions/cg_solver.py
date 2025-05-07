import torch

def conjugate_gradient(
    A, 
    B, 
    X0=None, 
    tol=1e-6, 
    max_iter=100, 
    M=None
):
    """
    Solves AX = B using the Conjugate Gradient (CG) method.
    
    Parameters:
        A (torch.Tensor or callable): 
            - If torch.Tensor: SPD matrix of shape (n, n).
            - If callable: Function that performs matrix-vector product, i.e., A(v) -> Av.
        B (torch.Tensor): 
            - Right-hand side matrix of shape (k,n), where k is the number of vectors.
        X0 (torch.Tensor, optional): 
            - Initial guess for X of shape (k,n). Defaults to a zero matrix.
        tol (float, optional): 
            - Tolerance for the residual norm. Defaults to 1e-6.
        max_iter (int, optional): 
            - Maximum number of iterations. Defaults to 1000.
        M (torch.Tensor or callable, optional): 
            - Preconditioner. If torch.Tensor, should be the diagonal of M (for Jacobi).
            - If callable, should perform M^{-1} v.
    
    Returns:
        X (torch.Tensor): Solution matrix of shape (k,n).
    """
    device = B.device
    k, n = B.shape

    with torch.no_grad():
        # Initialize solution X
        if X0 is None:
            X = torch.zeros_like(B, device=device)
            R = B.clone()
        else:
            X = X0.clone().to(device)
            # Compute initial residual R = B - A(X)
            if callable(A):
                R = B - A(X)
            else:
                R = B - A @ X

        # Apply preconditioner: Z = M^{-1} R
        if M is not None:
            if callable(M):
                Z = M(R)
            else:
                # Assuming M is the diagonal preconditioner
                Z = R / M.unsqueeze(0)
        else:
            Z = R.clone()

        # Initialize P = Z
        P = Z.clone()

        # Compute initial residual norms squared
        R_norm_sq = torch.sum(R * Z, dim=1)  # Shape: (k,)

        # Initialize convergence mask (True means not yet converged)
        converged = torch.zeros(k, dtype=torch.bool, device=device)
        not_converged = torch.sqrt(R_norm_sq) >= tol
        converged = converged | (~not_converged)
        for it in range(max_iter):
            if converged.all():
                print(f"CG converged in {it} iterations.")
                break

            # Compute A @ P
            if callable(A):
                AP = A(P)
            else:
                AP = A @ P  # Shape: (k, n)

            # Compute alpha = R_norm_sq / (P^T AP)
            PAP = torch.sum(P * AP, dim=1)  # Shape: (k,)
            alpha = R_norm_sq / PAP  # Shape: (k,)

            # Update X: X += P * alpha
            X += P * alpha.unsqueeze(1)  # Broadcasting alpha over rows

            # Update R: R -= AP * alpha
            R -= AP * alpha.unsqueeze(1)

            # Apply preconditioner: Z = M^{-1} R
            if M is not None:
                if callable(M):
                    Z_new = M(R)
                else:
                    Z_new = R / M.unsqueeze(0)
            else:
                Z_new = R.clone()

            # Compute new residual norms squared: R_new_norm_sq = R^T Z_new
            R_new_norm_sq = torch.sum(R * Z_new, dim=1)  # Shape: (k,)

            # Compute beta = R_new_norm_sq / R_norm_sq
            beta = R_new_norm_sq / R_norm_sq  # Shape: (k,)

            # Update P: P = Z_new + P * beta
            P = Z_new + P * beta.unsqueeze(1)

            # Update residual norms squared for next iteration
            R_norm_sq = R_new_norm_sq

            # Check convergence: ||R|| < tol
            residual_norm = torch.sqrt(R_norm_sq)  # Shape: (k,)
            not_converged = residual_norm >= tol
            newly_converged = (~not_converged) & (~converged)
            converged = converged | (~not_converged)

            if it % 1 == 0 or it == max_iter - 1:
                print(f"Iteration {it:3}: Residual norms = {residual_norm.sum().cpu().numpy():.7f}")

        else:
            print(f"CG did not converge within {max_iter} iterations.")

    return X

# Example Usage
if __name__ == "__main__":
    # Parameters
    n = 1000  # Dimension of the matrix
    k = 1    # Number of right-hand sides
    tol = 1e-6
    max_iter = 1000

    # Device configuration: use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate a random SPD matrix A
    torch.manual_seed(0)
    A_dense = torch.randn(n, n, device=device)
    A_dense = A_dense @ A_dense.t() + n * torch.eye(n, device=device)  # Ensures SPD

    # Alternatively, define A as a sparse matrix or a linear operator
    # For this example, we'll use the dense matrix

    # Define right-hand side B as multiple vectors
    B = torch.randn(n, k, device=device)

    # Optionally, define a preconditioner (Jacobi: diagonal of A)
    M = torch.diag(A_dense).to(device)  # Shape: (n,)

    # Solve AX = B using CG
    X = conjugate_gradient(A_dense, B, tol=tol, max_iter=max_iter, M=M)

    # Verify the solution
    with torch.no_grad():
        residual = torch.matmul(A_dense, X) - B
        residual_norm = torch.norm(residual, dim=0)
        print(f"Residual norms after CG: {residual_norm.cpu().numpy()}")

        # Optionally, compute the relative residual
        relative_residual = residual_norm / torch.norm(B, dim=0)
        print(f"Relative residual norms: {relative_residual.cpu().numpy()}")

def conjugate_gradient_batch(A,b,M_diag,max_iter=50,tol=1e-6):
    """
    Fully-batched Preconditioned Conjugate Gradient (PCG).
    For systems that converge early, we zero out their updates
    (stop changing x, r, etc.), but we still do a full batch
    matrix-vector product (A_mv) each iteration.

    Args:
        A (callable):
            A function that takes a [batch_size, n] tensor and returns A@v
            in shape [batch_size, n]. Must handle the entire batch at once.
        b (torch.Tensor):
            RHS tensor of shape [batch_size, n].
        M_diag (torch.Tensor):
            The diagonal of the preconditioner, shape [batch_size, n].
        max_iter (int):
            Maximum number of iterations.
        tol (float):
            Residual tolerance for declaring convergence.

    Returns:
        x (torch.Tensor):
            The solution batch of shape [batch_size, n].
    """

    batch_size, n = b.shape
    device = b.device

    # 1) Initialization
    x = torch.zeros_like(b)           # [batch_size, n]
    r = b.clone()                     # [batch_size, n]  (residual)
    # Preconditioned residual z = M^{-1} r, but M is diagonal => elementwise division
    z = r / M_diag                    # [batch_size, n]
    p = z.clone()                     # [batch_size, n]  (search direction)
    rs_old = torch.sum(r * z, dim=-1) # [batch_size]     (rᵀ z for each system)

    # Track which systems are still active (haven't converged yet)
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    active_mask = active_mask & (torch.sqrt(rs_old) >= tol)

    for it in range(max_iter):
        if not active_mask.any():
            # All systems have converged
            break

        # 2) Compute A@p for the entire batch
        Ap = A(p)  # shape [batch_size, n]

        # 3) Compute the step size alpha = (rᵀ z) / (pᵀ Ap) for each system
        denom = torch.sum(p * Ap, dim=-1)   # [batch_size]
        denom_clamped = torch.clamp(denom, min=1e-30)
        alpha = rs_old / denom_clamped      # [batch_size]
        # Zero-out alpha for converged systems
        alpha = alpha * active_mask.float()
        alpha_ = alpha.unsqueeze(-1)        # [batch_size, 1]

        # 4) Update x, r (for all, but no change if not active)
        x = x + alpha_ * p
        r = r - alpha_ * Ap

        # 5) Check convergence
        r_norm = r.norm(dim=-1)  # [batch_size]
        newly_converged = (r_norm < tol)
        active_mask = active_mask & (~newly_converged)

        # 6) Precondition the new residual: z = r / M_diag
        z = r / M_diag

        # 7) Update the search direction p
        rs_new = torch.sum(r * z, dim=-1)    # [batch_size]
        beta = rs_new / torch.clamp(rs_old, min=1e-30)
        # Zero-out beta for converged systems
        beta = beta * active_mask.float()

        p = z + beta.unsqueeze(-1) * p
        rs_old = rs_new

        # if it % 1 == 0 or it == max_iter - 1:
        #     print(f"Iteration {it:3}: Residual norms = {r_norm}")

    return x

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

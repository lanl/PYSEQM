import torch
from typing import Callable, Optional

def conjugate_gradient_batch(
    A: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    M_diag: Optional[torch.Tensor] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Fully‐batched (Preconditioned) Conjugate Gradient over arbitrary residual dimensions.

    Solves A(x) = b for each example in the batch.  If M_diag is given,
    uses it as diagonal preconditioner; otherwise runs plain CG.

    Args:
        A (callable): maps x of shape [B, …] → A(x) (same shape).
        b (Tensor): RHS of shape [B, D1, D2, …, Dk].
        M_diag (Optional[Tensor]): diag of preconditioner, same shape as b.
        max_iter (int): max CG iterations.
        tol (float): stop tolerance on residual norm.

    Returns:
        x (Tensor): solution, same shape as b.
    """
    batch_size = b.shape[0]
    # dims to sum over when computing inner‐products / norms:
    sum_dims = tuple(range(1, b.dim()))

    # 1) init
    x = torch.zeros_like(b)
    r = b.clone()                      # residual
    if M_diag is not None:
        z = r / M_diag                 # preconditioned residual
    else:
        z = r                          # no preconditioner
    p = z.clone()                      # search direction
    rs_old = torch.sum(r * z, dim=sum_dims)  # [B]

    # mask of active (not yet converged) systems
    active = torch.norm(r,dim=sum_dims) >= tol
    alpha_dtype = rs_old.dtype

    for i in range(max_iter):
        if not active.any():
            return x

        Ap = A(p)  # [B, D1…Dk]
        denom = torch.sum(p * Ap, dim=sum_dims)         # [B]
        alpha = rs_old / denom.clamp(min=1e-20)         # [B]
        # zero‐out for converged
        alpha = alpha * active.to(alpha_dtype)

        # reshape alpha to broadcast over [D1…Dk]:
        alpha_exp = alpha.view(batch_size, *([1] * (b.dim() - 1)))

        # 4) update solution & residuals
        x = x + alpha_exp * p
        r = r - alpha_exp * Ap

        # check convergence
        r_norm = torch.norm(r, dim=sum_dims)
        newly_conv = r_norm < tol
        active = active & (~newly_conv)

        # precondition
        if M_diag is not None:
            z = r / M_diag
        else:
            z = r

        # update direction
        rs_new = torch.sum(r * z, dim=sum_dims)
        beta = rs_new / rs_old.clamp(min=1e-20)
        beta = beta * active.to(beta.dtype)
        beta_exp = beta.view(batch_size, *([1] * (b.dim() - 1)))

        p = z + beta_exp * p
        rs_old = rs_new

        # if i % 1 == 0 or i == max_iter - 1:
        #     print(f"Iteration {i:3}: Residual norms = {r_norm}")
    if torch.any(active): raise RuntimeError(f"Conjugate gradient did not converge in {max_iter} steps (resid={r_norm})")

# Example Usage
if __name__ == "__main__":
    # Parameters
    batch = 1
    n = 5  # Dimension of the matrix
    k = 1    # Number of right-hand sides
    tol = 1e-6
    max_iter = 100

    # Device configuration: use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate a random SPD matrix A
    torch.manual_seed(0)
    A_dense = torch.rand(batch,n, n, device=device)
    A_dense = A_dense + A_dense.mT + 100*n * torch.eye(n, device=device).unsqueeze(0)  # Ensures SPD
    print(f"A shape is {A_dense.shape}")
    applyA = lambda b: torch.einsum('bij,bj->bi',A_dense,b)

    # Alternatively, define A as a sparse matrix or a linear operator
    # For this example, we'll use the dense matrix

    # Define right-hand side B as multiple vectors
    B = torch.randn(batch, n, device=device)

    # Optionally, define a preconditioner (Jacobi: diagonal of A)
    M = torch.diagonal(A_dense,dim1=1,dim2=2).to(device)  # Shape: (n,)

    # Solve AX = B using CG
    X = conjugate_gradient_batch(applyA, B, tol=tol, max_iter=max_iter, M_diag=M)

    # Verify the solution
    with torch.no_grad():
        residual = applyA(X) - B
        residual_norm = torch.norm(residual, dim=1)
        print(f"Residual norms after CG: {residual_norm.cpu().numpy()}")

        # Optionally, compute the relative residual
        relative_residual = residual_norm / torch.norm(B, dim=1)
        print(f"Relative residual norms: {relative_residual.cpu().numpy()}")


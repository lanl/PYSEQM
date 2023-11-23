def orthogonalize_matrix(U, eps=1e-15):
    """
    Orthogonalizes the matrix U (d x n) using Gram-Schmidt Orthogonalization.
    If the columns of U are linearly dependent with rank(U) = r, the last n-r columns 
    will be 0.
    
    Args:
        U (numpy.array): A d x n matrix with columns that need to be orthogonalized.
        eps (float): Threshold value below which numbers are regarded as 0 (default=1e-15).
    
    Returns:
        (numpy.array): A d x n orthogonal matrix. If the input matrix U's cols were
            not linearly independent, then the last n-r cols are zeros.
    
    Examples:
    ```python
    >>> import numpy as np
    >>> import gram_schmidt as gs
    >>> gs.orthogonalize(np.array([[10., 3.], [7., 8.]]))
    array([[ 0.81923192, -0.57346234],
       [ 0.57346234,  0.81923192]])
    >>> gs.orthogonalize(np.array([[10., 3., 4., 8.], [7., 8., 6., 1.]]))
    array([[ 0.81923192 -0.57346234  0.          0.        ]
       [ 0.57346234  0.81923192  0.          0.        ]])
    ```
    """
    
    n = len(U[0])
    # numpy can readily reference rows using indices, but referencing full rows is a little
    # dirty. So, work with transpose(U)
    V = U.T
    for i in range(n):
        prev_basis = V[0:i]     # orthonormal basis before V[i]
        coeff_vec = prev_basis @ V[i].T  # each entry is np.dot(V[j], V[i]) for all j < i
        # subtract projections of V[i] onto already determined basis V[0:i]
        V[i] -= (coeff_vec @ prev_basis).T
        if torch.norm(V[i]) < eps:
            V[i][V[i] < eps] = 0.   # set the small entries to 0
        else:
            V[i] /= torch.norm(V[i])
    return V.T
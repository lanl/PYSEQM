import torch


def orthogonalize_to_current_subspace(V, newsubspace, vend, tol):
    """Orthogonalize newsubspace vectors against the first vend rows of V.

    New accepted orthonormal vectors are written in-place into V starting at vend.
    Returns the updated subspace size.
    """
    n = newsubspace.shape[0]
    for i in range(n):
        vec = newsubspace[i]
        vec -= (vec @ V[:vend].T) @ V[:vend]
        vec -= (vec @ V[:vend].T) @ V[:vend]
        vecnorm = torch.norm(vec)

        if vecnorm > tol:
            V[vend] = vec / vecnorm
            vend = vend + 1

    return vend

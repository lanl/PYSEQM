import torch
from .rcis_batch import get_occ_virt, makeA_pi_batched

def elec_energy_excited_xl(mol,R,w,e_mo):
    # e_mo = mol.e_mo
    # R = mol.transition_density_matrices
    nocc, nvirt, Cocc, Cvirt, ea_ei = get_occ_virt(mol, orbital_window=None, e_mo=e_mo)
    # R = torch.einsum('bmi,bia,bna->bmn', Cocc,Xbar.view(-1,nocc,nvirt), Cvirt)
    Xbar = torch.einsum('bmi,brmn,bna->bria', Cocc,R, Cvirt).reshape(-1,nocc*nvirt)
    G_ao = makeA_pi_batched(mol,R,w)
    G = torch.einsum('bmi,brmn,bna->bria', Cocc, G_ao, Cvirt).view_as(Xbar)*2.0
    
    # Get X, omega from Xbar
    ea_ei = ea_ei.view_as(Xbar)
    X, omega = solve_for_amplitude_omega(Xbar,ea_ei,G)
    # X = Xbar
    # omega = 0.0

    E1 = (X*X*ea_ei).sum(dim=1)
    E2 = ((2.0*X-Xbar)*G).sum(dim=1)

    E = E1 + E2

    # print(f"CIS Energy is {E.item()}, omega is {omega}, norm of X difference is {torch.linalg.vector_norm(X-Xbar)}")
    # print(mol.cis_energies[0,0].item())
    # exit()
    X_AO = torch.einsum('bmi,bia,bna->bmn', Cocc, X.view(-1,nocc,nvirt), Cvirt).unsqueeze(1)  # convert it back to the shape of brmn
    return E, X_AO

def solve_for_amplitude_omega(Xbar,ea_ei,G):
    """
    Computes (M^{-1}) @ E = [X; omega] for
        M = [[A, B],
             [C, D]]
    with A = diag(ea_ei), B = neg_Xbar (i.e., -Xbar),
         C = two_Xbar (i.e., 2*Xbar), D = 0,
    and E = [G; L].
    with G = -G[Xbar] and L = 1 + Xbar^T @ Xbar

    Shapes: 
        ea_ei:    (b,n,)   diagonal entries of A
        neg_Xbar: (b,n,)   equals -Xbar  (this is B)
        two_Xbar: (b,n,)   equals 2*Xbar (this is C)
        G:        (b,n,)   first n entries of E, this will be -G
        L:        (b,1,)  last entry of E (scalar)
        where n = nocc*nvirt
    Returns:
        y: (b,n+1) == M^{-1} @ E
    """
    # Flatten to 1-D and align dtype/device
    L = 1.0 + torch.sum(Xbar*Xbar,dim=1)

    # Invert A (diagonal) elementwise
    invA_diag = 1.0 / ea_ei
    if not torch.isfinite(invA_diag).all():
        raise RuntimeError("HOMO-LUMO gap is zero for some molecules")

    # Helper: A^{-1} * vector == elementwise multiply by invA_diag
    Ainv_G = -invA_diag * G

    # Schur complement S = D - C A^{-1} B; here D = 0 (scalar)
    # With B = -Xbar and C = 2*Xbar this is S = - two_Xbar · Ainv_B = two_Xbar · Ainv_Xbar
    # Here we calculate inverse of S = 0.5 / (Xbar · Ainv_Xbar)
    S_inv = 0.5 / (invA_diag * (Xbar * Xbar)).sum(dim=1)

    if not torch.isfinite(S_inv).all():
        raise RuntimeError("Invalid Schur complement S")
     
    # Right-hand side for the bottom equation: L - C A^{-1} G
    rhs2 = L - 2.0 * (Xbar * Ainv_G).sum(dim=1)

    # Solve for x2 (scalar), then x1 (n-vector)
    omega = rhs2 * S_inv

    # x1 = A^{-1} (G - B x2) = A^{-1} (G - (-X) x2) = A^{-1} (G + X * x2)
    X = (-G + Xbar * omega.unsqueeze(-1)) * invA_diag  # (b, n)

    return X, omega

def solve_exactly_X_omega(eaei):
    # exactly solve instead of solve_for_amplitude_omega
    pass


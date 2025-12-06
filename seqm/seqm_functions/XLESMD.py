import torch
import numpy as np
import matplotlib.pyplot as plt
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

def sample_noisy_R_energy(mol, R, w, e_mo, n_steps=50, noise_scale=1e-5, cumulative=False, seed=None, return_data=False, plot=True, fit_order=2):
	"""
	Generate noisy copies Rbar of R, compute total energy for each, and plot |R-Rbar| vs E.

	Args:
		mol: molecule object expected by elec_energy_excited_xl
		R: torch.Tensor, original transition-density tensor (same shape used by elec_energy_excited_xl)
		w, e_mo: arguments forwarded to elec_energy_excited_xl
		n_steps: number of noisy samples
		noise_scale: standard deviation of Gaussian noise added to R (absolute scale)
		cumulative: if True, noise is added cumulatively (Rbar <- Rbar + noise). If False, noise is added to original R each sample.
		seed: optional int seed for reproducibility
		return_data: if True, returns (deltas, energies)
		plot: if True, plots |R-Rbar| vs E using matplotlib
		fit_order: polynomial order to fit to (deltas, energies) for plotting (default 2)

	Returns:
		None or (deltas, energies) if return_data=True
	"""
	if seed is not None:
		torch.manual_seed(seed)
		np.random.seed(seed)

	# Work with detached copies to avoid grads / side-effects
	R = R.clone().detach()
	Rbar = R.clone().detach()
	deltas = []
	energies = []
	Rbase = torch.randn_like(R)
	for i in range(n_steps):
		noise = i*noise_scale * Rbase
		if cumulative:
			Rbar = Rbar + noise
		else:
			Rbar = R + noise

		with torch.no_grad():
			E, _ = elec_energy_excited_xl(mol, Rbar, w, e_mo)

		# Reduce E to a scalar for plotting: mean across batch if batched
		if torch.is_tensor(E):
			energy_scalar = float(E.mean().item())
		else:
			energy_scalar = float(np.asarray(E).mean())

		delta = float(torch.linalg.norm(R - Rbar).item())

		deltas.append(delta)
		energies.append(energy_scalar)
		# print(f"Step {i+1}/{n_steps}: |R-Rbar| = {delta:.6e}, E = {energy_scalar:.6e}")

	if plot:
		plt.figure()
		plt.plot(deltas, energies, marker='o', linestyle='-', label='data')
		plt.xlabel('|R - Rbar| (transition density error)')
		plt.ylabel('E')
		plt.title('Energy vs Error in transition density')

		# Fit an n-th order polynomial if there are enough points
		try:
			if len(deltas) > fit_order:
				x = np.array(deltas)
				y = np.array(energies)
				# compute polynomial coefficients (highest degree first)
				coeffs = np.polyfit(x, y, fit_order)
				print("Fitted polynomial coefficients (highest degree first):", coeffs)
				p = np.poly1d(coeffs)
				# sort for a smooth curve
				idx = np.argsort(x)
				xs = x[idx]
				ys = p(xs)
				plt.plot(xs, ys, color='red', linestyle='--', label=f'polynomial fit (order={fit_order})')
				plt.legend()
		except Exception:
			# don't fail plotting if fit fails; still show raw data
			pass

		plt.grid(True)
		plt.show()

	exit()

	if return_data:
		return deltas, energies
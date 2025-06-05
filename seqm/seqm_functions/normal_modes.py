import torch

def normal_modes(molecule, energy):
    """
    Calculates the mass‐weighted Hessian and normal‐mode frequencies (in cm⁻¹) using PyTorch autograd.

    Parameters
    ----------
    molecule : molecule object 
    energy : torch.Tensor
        A tensor representing the potential energy

    Returns
    -------
    frequencies_cm1 : torch.Tensor
        A 1D tensor of vibrational frequencies (in cm⁻¹), sorted in ascending order, 
        excluding the six near‐zero modes (3 translations + 3 rotations).
    modes : torch.Tensor
        A tensor of shape (3*N_atoms, 3*N_atoms–6). Each column is a mass‐weighted normal‐mode 
        eigenvector in Cartesian coordinates (after removing the first six non‐vibrational modes).

    Notes
    -----
    1. The Hessian matrix H is defined by H_{ij} = ∂²E/(∂x_i ∂x_j).
       We compute it by first taking ∂E/∂x (grad1) and then differentiating each component of grad1.
    2. Mass‐weighting: H_w = M^(−1/2) · H · M^(−1/2), where M is diagonal with each atom’s mass 
       repeated for x, y, z. After diagonalizing H_w, the eigenvalues λ give normal-mode frequencies ω in cm⁻¹
       from the equation ω = sqrt(λ)/(2*pi*c)
    """
    if molecule.nmol != 1:
        raise ValueError("Hessian and Normal mode calculation currently not available for a batch of more than one molecule. Please input only one molecule")
        # TODO: extend this function to handle multiple molecules at once

    coords = molecule.coordinates

    # Sum energy to ensure it’s a scalar
    total_energy = energy.sum()

    # === 1) First derivative: ∂E/∂coords ===
    # create_graph=True is needed in order to take a second derivative (for Hessian)
    grad1 = torch.autograd.grad(total_energy, coords, create_graph=True)[0]
    # Flatten the gradient so that we can index each component: shape (3*N_atoms,)
    grad1_flat = grad1.view(-1)

    num_coords = grad1_flat.numel()  # should equal 3*N_atoms
    device = coords.device
    dtype = coords.dtype

    # Preallocate the Hessian (3N × 3N) on the same device/dtype
    H = torch.zeros((num_coords, num_coords), dtype=dtype, device=device)

    # === 2) Build Hessian by differentiating each component of grad1_flat ===
    # Note: create_graph=False here since we don’t need third derivatives.
    # retain_graph=True is required until the last iteration; 
    for i in range(num_coords):
        # ∂(grad1_flat[i]) / ∂ coords  ⇒ shape (N_atoms, 3)
        grad2_i = torch.autograd.grad(grad1_flat[i], coords, retain_graph=True)[0]
        # Flatten and store in row i of H
        H[i, :] = grad2_i.view(-1)

    # === 3) Retrieve atomic masses ===
    masses = molecule.const.mass[molecule.species].squeeze().to(device=device, dtype=dtype)

    # === 4) Build 3N mass vector (mass repeated for x, y, z) ===
    # e.g. [m₁, m₁, m₁, m₂, m₂, m₂, ...]
    mass_vector = masses.repeat_interleave(3)  # shape: (3*N_atoms,)

    # === 5) Construct mass‐weighted Hessian: H_w = M^(−1/2) H M^(−1/2) ===
    sqrt_mass = torch.sqrt(mass_vector)  # shape: (3N,)
    # Use broadcasting to divide each element: H_w[i,j] = H[i,j] / (√m_i * √m_j)
    H_mass = H / (sqrt_mass[:, None] * sqrt_mass[None, :])

    # === 6) Diagonalize the symmetric mass‐weighted Hessian ===
    # eigvals: shape (3N,), modes: shape (3N, 3N)
    eigvals, modes = torch.linalg.eigh(H_mass)

    # === 7) Convert eigenvalues → frequencies (cm⁻¹) ===
    # eigval is in the units of eV/Å²/amu. 
    # Conversion to cm-1: sqrt(eV/Å²/amu) → cm⁻¹ = 521.470898 (approx).
    conv_factor = 521.470898

    # Eliminate small/negative numerical noise before taking sqrt
    eigvals_clipped = torch.clamp(eigvals, min=0.0)

    # We expect 6 near‐zero modes (3 translations + 3 rotations). Check that we have >6 total modes:
    if eigvals_clipped.numel() <= 6:
        raise ValueError( f"Only {eigvals_clipped.numel()} eigenvalues found; need at least 7 to extract vibrational modes.")

    # Discard first six modes (indices 0..5), keep indices 6..end
    # Compute frequencies in cm⁻¹: ω = conv_factor * sqrt(λ)
    frequencies_cm1 = conv_factor * torch.sqrt(eigvals_clipped[6:])

    # Sort frequencies in ascending order for nicer output
    frequencies_cm1, sort_idx = torch.sort(frequencies_cm1)

    # Extract the corresponding eigenvectors (columns) for vibrational modes
    # modes has shape (3N, 3N); we skip the first 6 columns, then reorder by sort_idx
    vib_modes = modes[:, 6:][:, sort_idx]  # shape: (3N, 3N−6)

    # === 9) Nicely format and print frequencies ===
    freq_list = frequencies_cm1.cpu().numpy().tolist()
    print("Vibrational Frequencies (cm⁻¹):")
    for i, freq in enumerate(freq_list, start=1):
        print(f"  Mode {i:3d}: {freq:12.6f}")

    return frequencies_cm1, vib_modes

    if molecule.nmol > 1:
        raise Exception("Hessian and Normal mode calculation currently not available for a batch of more than one molecule. Please input only one molecule")

    y = energy.sum()               # To get scalar energy
    grad_y = torch.autograd.grad(y, molecule.coordinates, create_graph=True)[0]
    n = molecule.coordinates.numel() # 3N
    H = torch.zeros(n, n, dtype=molecule.coordinates.dtype, device=molecule.coordinates.device)

    flat_grad = grad_y.reshape(-1)   # shape (n,)

    for i in range(n):
        # ∂(grad_y[i]) / ∂ coords  ⇒ a vector of length n
        grad2_i = torch.autograd.grad(flat_grad[i], molecule.coordinates, retain_graph=True)[0]
        H[i, :] = grad2_i.reshape(-1)

    masses = molecule.const.mass[molecule.species].squeeze()
    N = masses.shape[0]
    assert H.shape == (3*N, 3*N), "Hessian shape mismatch"

    # Build 3N mass vector (mass for each Cartesian coordinate)
    mass_vector = masses.repeat_interleave(3)  # shape: (3N,)

    # Construct mass-weighted Hessian
    mass_sqrt = torch.sqrt(mass_vector)
    H_mass_weighted = H / (mass_sqrt[:, None] * mass_sqrt[None, :])

    # Diagonalize
    eigvals, modes = torch.linalg.eigh(H_mass_weighted)
    
    factor = 521.470898  # sqrt(eV/Å²/amu) to cm⁻¹

    # Take only positive eigenvalues (filter out small/negative numerical artifacts)
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    frequencies_cm1 = factor * torch.sqrt(eigvals_clipped[6:]) # eliminate 6 lowest modes, of which 3 are translations and 3 are rotations

    print(f"Frequencies are:\n{frequencies_cm1}")

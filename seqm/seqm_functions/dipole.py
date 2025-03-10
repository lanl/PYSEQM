from numpy import dtype
import torch
from .cal_par import dd_qq
from .constants import a0

def calc_dipole_matrix(mol):
    """
    Build the block-diagonal dipole tensor for atoms.
    
    Each atom has 4 orbitals (sp), so the full dipole matrix is represented by a tensor
    of shape (natom*natom, 4, 4, 3) where the last dimension corresponds to the x, y, and z components.
    but here we build only the diagonal blocks of the full dipole matrix
    
    For non-hydrogen atoms
      - All diagonal entries (positions (i,i) for i=0,...,3) are set to -coord.
      - Additionally, for each Cartesian direction i (i = 0 for x, 1 for y, 2 for z),
        the off-diagonal element (0, i+1) is set to multip_2c_elec_params.
    
    For hydrogen atoms
      - Only the (0,0) element is set to -coord, with the other diagonal elements remaining zero.
    
    Parameters:
      mol: contains all the info on the molecules
      
    Returns:
      diagonal_dipole: Tensor of shape (natom, 4, 4, 3) with the dipole blocks.
    """

    dtype = mol.rij.dtype
    device = mol.rij.device
    zetas=mol.parameters['zeta_s']
    zetap=mol.parameters['zeta_p']
    qn = mol.const.qn
    Z = mol.Z
    qn0=qn[Z]
    # dd=torch.zeros_like(qn0)
    isX = Z>2   # Heavy atom
    isH = Z==1
    dd, _ = dd_qq(qn0[isX],zetas[isX], zetap[isX])
    dd *= a0

    coord = mol.coordinates.view(mol.nmol*mol.molsize,3)
    diagonal_dipole = torch.zeros((3, mol.nmol*mol.molsize, 4, 4), dtype=dtype, device=device)
    
    I = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # shape (1,1,4,4)
    # Get -coord for non-H atoms and rearrange from (n_nonH, 3) to (3, n_nonH, 1, 1).
    nonH_coord = -coord[isX].T.unsqueeze(-1).unsqueeze(-1)  # shape (3, n_nonH, 1, 1)
    # Multiply by the identity so that only the diagonal entries are nonzero.
    diag_block_nonH = nonH_coord * I  # shape (3, n_nonH, 4, 4)
    diagonal_dipole[:, isX, :, :] = diag_block_nonH
    
    # Set the off-diagonal s-p interaction elements:
    # For each Cartesian direction i (0: x, 1: y, 2: z), set
    for i in range(3):
      diagonal_dipole[i, isX, 0, i+1] = -dd
      diagonal_dipole[i, isX, i+1, 0] = -dd
    # cart_idx = torch.arange(3, device=device)
    # diagonal_dipole[cart_idx, isX, 0, cart_idx + 1] = -dd
    # diagonal_dipole[cart_idx, isX, cart_idx + 1, 0] = -dd
    
    # --- Process hydrogen atoms ---
    # For hydrogen atoms, only the (0,0) element is set.
    diagonal_dipole[:, isH, 0, 0] = -coord[isH].T


    dipole_mat = torch.zeros(3,mol.nmol*mol.molsize*mol.molsize,4,4,dtype=dtype,device=device)
    dipole_mat[:,mol.maskd] = diagonal_dipole
    dipole_mat = dipole_mat.reshape(3,mol.nmol,mol.molsize,mol.molsize,4,4) \
             .permute(1,0,2,4,3,5) \
             .reshape(mol.nmol, 3, 4*mol.molsize, 4*mol.molsize)
    
    return dipole_mat

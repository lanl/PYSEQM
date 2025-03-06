from numpy import dtype
import torch
from .cal_par import dd_qq
from .constants import a0

def dipole_matrix(mol):
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

    molsize = mol.molsize
    coord = mol.coordinates.view(mol.nmol*molsize,3)
    diagonal_dipole = torch.zeros((molsize, 4, 4, 3), dtype=dtype, device=device)
    
    # For non-hydrogen atoms: all diagonal elements are -coord.
    diag_idx = torch.arange(4, device=device)
    # For each non-H atom, assign -coord to all diagonal positions.
    diagonal_dipole[isX, diag_idx, diag_idx, :] = -coord[isX].unsqueeze(1)
    # Set off-diagonal s-p interaction elements:
    # For each Cartesian component i (0: x, 1: y, 2: z), assign at position (0, i+1).
    cart_idx = torch.arange(3, device=device)
    diagonal_dipole[isX, 0, cart_idx+1, cart_idx] = -dd.unsqueeze(1)
    diagonal_dipole[isX, cart_idx+1, 0, cart_idx] = -dd.unsqueeze(1)
    
    # For hydrogen atoms: only the (0,0) element is set to -coord.
    diagonal_dipole[isH, 0, 0, :] = -coord[isH]
    
    return diagonal_dipole

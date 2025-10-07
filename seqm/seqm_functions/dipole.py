from numpy import dtype
import torch
from .cal_par import dd_qq
from .constants import a0

def calc_dipole_matrix(mol,return_diag_dipole=False):
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

    Z = mol.Z
    isX = Z>2   # Heavy atom
    isH = Z==1

    zetas=mol.parameters['zeta_s'][isX]
    zetap=mol.parameters['zeta_p'][isX]

    b, n = mol.species.shape

    qn0 = mol.const.qn[Z[isX]]
    dd, _ = dd_qq(qn0, zetas, zetap)
    dd *= a0
    dd_map = torch.zeros(b, n, dtype=dtype, device=device)
    dd_map.view(-1)[isX] = dd


    diagonal_dipole = torch.zeros((b, 3, n, 4, 4), dtype=dtype, device=device)
    with torch.no_grad():
        neg = -(mol.coordinates).permute(0,2,1).unsqueeze(-1).unsqueeze(-1)  # (b, 3, n, 1, 1)

    I4  = torch.eye(4, dtype=dtype, device=device).view(1,1,1,4,4)
    S00 = torch.zeros(4,4, dtype=dtype, device=device); S00[0,0] = 1
    S00 = S00.view(1,1,1,4,4)

    diagonal_dipole += neg * I4  * isX.view(b,1,n,1,1)  # heavy: all 4 diag
    diagonal_dipole += neg * S00  * isH.view(b,1,n,1,1)  # heavy: all 4 diag
    
    dd_vals = (-dd_map).unsqueeze(1)  # (b,1,n)
    for d in range(3):
        diagonal_dipole[:, d, :, 0, d+1] = dd_vals[:, 0, :]
        diagonal_dipole[:, d, :, d+1, 0] = dd_vals[:, 0, :]

    if return_diag_dipole:
        return diagonal_dipole  # (b,3,n,4,4)

    eye = torch.eye(n, device=device, dtype=dtype).view(1, 1, n, 1, n, 1)
    dipole_mat = (diagonal_dipole.unsqueeze(4) * eye).reshape(b, 3, 4*n, 4*n)  # (b,3,4n,4n)
    
    return dipole_mat

from .constants import debye_to_AU, to_debye
def calc_ground_dipole(molecule, P):
    with torch.no_grad():
        b,n = molecule.coordinates.shape[:2]
        dipole_diag_blocks = calc_dipole_matrix(molecule,return_diag_dipole=True) # (b, 3, n, 4, 4)
        
        # Extract 4x4 block diagonals
        P_blocks = P.view(b, n, 4, n, 4).diagonal(0, 1, 3)          # (b, 4, 4, n)

        # Electronic dipole
        electronic_dipole = torch.einsum('bxyn,bdnxy->bd', P_blocks, dipole_diag_blocks)        # (b, 3)

        # Nuclear dipole
        nuclear_dipole = (molecule.const.tore[molecule.species].unsqueeze(-1)
                   * molecule.coordinates).sum(dim=1)                            # (b, 3)

        molecule.dipole = (electronic_dipole + nuclear_dipole)*to_debye*debye_to_AU
        return

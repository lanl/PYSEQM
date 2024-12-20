import torch


def mo2ao(device, N_cis, V_mo, mol):
    """
   Transforms the density matrix from MO to AO basis.
   Vectorized adaptation of subroutine site2ao from Lioville in NEXMD
   Full multiplication as C @ V @ C.T did not work
   
    Parameters:
    - device (torch.device): cpu or gpu
    - N_cis (int): size of the CIS matrix
    - V_mo (torch.Tensor): Guess density in MO basis
    - mol (Molecule): PYSEQM molecule object 

    Returns:
    - V_ao (torch.Tensor): Guess in AO basis of shape (nmol, norb, norb).
    """
    
    #TODO: vectorize  to avoid L_xi[:,i] = form_cis(device, V[:,i], mol, N_cis, N_rpa) 
    #TODO: add batch mode for nmol > 1

    V_mo = V_mo.contiguous().view(-1, mol.nvirt) # test
    V_mo_tmp = torch.zeros((mol.norb, mol.norb), device=device) # temp storage, presumably faster than padding
    V_ao = V_mo @ mol.C_mo[0][:, mol.nocc:mol.norb].transpose(0,1) # operations on |X|, |Y| from RPA is ignored for now
    V_mo_tmp[:mol.nocc] = V_ao
    V_ao = mol.C_mo[0] @ V_mo_tmp

    return V_ao


def ao2mo(device, N_cis, G_ao, mol):
    """
    Transforms the density from MO to AO basis
    Vectorized adaptation of subroutine 2aosite from Lioville in NEXMD
    # TODO: introduce ERI (electron repulsion integrals notation)
    Args:
    - device: cpu or gpu
    - N_cis: size of the CIS matrix
    - G_ao: The matrix containing the two-electron integrals (sym and antisym) in the AO basis.
    - mol: PYSEQM molecule object 

    Returns:
    - G_mo: ERI matrix in MO basis
    """

    #TODO: vectorize  to avoid L_xi[:,i] = form_cis(device, V[:,i], mol, N_cis, N_rpa) 
    #TODO: add batch mode for nmol > 1
    #TODO: rename dgemms, ugly remnants of Fortran, to be consistent with ao2mo and other new routines
    
    
    dgemm1 = G_ao.transpose(1,2) @ mol.C_mo
    dgemm2 =  mol.C_mo[:, :, mol.nocc:mol.norb].transpose(1,2) @ dgemm1[:, :, :mol.nocc]
    G_mo = dgemm2.transpose(1,2).flatten()

    return G_mo

def decompose_to_sym_antisym(A):
    """
    Decomposes a matrix into symmetric and antisymmetric parts.

    Args:
        A (tensor): The input matrix.

    Returns:
        Tuple[tensor, tensor]: A tuple containing the symmetric and antisymmetric parts of the input matrix.
    """   
    #vectorized be default
    #TODO: batch mode for nmol > 1
    A_sym = 0.5 * (A + A.transpose(0, 1))
    A_antisym = 0.5 * (A - A.transpose(0, 1))
    
    return A_sym, A_antisym

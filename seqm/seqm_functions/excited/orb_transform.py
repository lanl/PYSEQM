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
    # mol.C_mo = mol.C_mo[0]
    V_mo = V_mo.view(-1, mol.nvirt) # reshape V[:, :, i], column, to 2d in batch of mols
    print('V_mo', V_mo.shape)
    V_mo_tmp = torch.zeros((mol.norb, mol.norb), device=device) # temp storage, presumably faster than padding
    print('V_mo_tmp', V_mo_tmp.shape)
    V_ao = V_mo @ mol.C_mo[0][:, mol.nocc:mol.norb].transpose(0,1) # operations on |X|, |Y| from RPA is ignored for now
    print("V_ao", V_ao.shape)
    V_mo_tmp[h :mol.nocc] = V_ao
    
    V_ao = mol.C_mo[0] @ V_mo_tmp

    return V_ao


def ao2mo(device, N_cis, G_ao, mol):
    """
    Transforms the density from MO to AO basis
    Vectorized adaptation of subroutine 2aosite from Lioville in NEXMD
    # TODO: intrduce ERI (electron repulsion integrals notation)
    Args:
    - device: cpu or gpu
    - N_cis: size of the CIS matrix
    - G_ao: The matrix containing the two-electron integrals (sym and antisym) in the AO basis.
    - mol: PYSEQM molecule object 

    Returns:
    - G_mo: ERI matrix in MO basis
    """

    dgemm1 = G_ao.transpose(0,1) @ mol.C_mo
    dgemm2 =  mol.C_mo[:, mol.nocc:mol.norb].transpose(0,1) @ dgemm1[:, :mol.nocc]
    G_mo = dgemm2.transpose(0,1).flatten()

    return G_mo

def decompose_to_sym_antisym(A):
    """
    Decomposes a matrix into symmetric and antisymmetric parts.

    Args:
        A (tensor): The input matrix.

    Returns:
        Tuple[tensor, tensor]: A tuple containing the symmetric and antisymmetric parts of the input matrix.
    """   
    A_sym = 0.5 * (A + A.transpose(1, 2))
    A_antisym = 0.5 * (A - A.transpose(1, 2))
    
    return A_sym, A_antisym

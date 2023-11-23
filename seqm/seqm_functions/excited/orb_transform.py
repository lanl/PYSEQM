import torch


def mo2ao_nexmd(device, N_cis, V_mo, mol):
    """
    transform matrix from AO to MO basis

    Parameters
    ----------
    M_AO : torch tensor # TODO add size
        matrix in AO basis
    C : torch tensor # TODO add size
        matrix of MO coefficients # TODO row or columns, structure?
        
    Returns
    -------
    M_MO : torch tensor # TODO add size 
        matrix in MO basisnb
    """    
    V_mo = V_mo.view(1, -1, mol.nvirt)
    # print('V_mo.shape', V_mo.shape)
    # print(V_mo)
    V_mo_tmp = torch.zeros((mol.nmol, mol.norb, mol.norb), device=device)
    # print('V_mo tmp.shape', V_mo_tmp.shape)
    # print(V_mo_tmp)
    # print('mol.C_mo[:, :, mol.nocc:mol.norb].transpose(1,2)', mol.C_mo[:, :, mol.nocc:mol.norb].transpose(1,2).shape)\
    V_ao = V_mo @ mol.C_mo[:, :, mol.nocc:mol.norb].transpose(1,2) # operations on |X| 
    # print('*** V_ao', V_ao.shape)
    # print('V_ao\n', V_ao)
    V_mo_tmp[:, :mol.nocc] = V_ao
    V_ao = mol.C_mo @ V_mo_tmp 

    # print('*** V_ao', V_ao.shape)
    # print('V_ao\n', V_ao)
    return V_ao


def ao2mo_nexmd(device, N_cis, G_ao, mol):
    """
    transform matrix from AO to MO basis

    Parameters
    ----------
    M_AO : torch tensor # TODO add size
        matrix in AO basis
    C : torch tensor # TODO add size
        matrix of MO coefficients # TODO row or columns, structure?
        
    Returns
    -------
    M_MO : torch tensor # TODO add size
        matrix in MO basis
    """    
    
    # COPY of subroutine site2mo from Lioville in NEXMD
    print('G_ao.shape', G_ao.shape)
    dgemm1 = G_ao.transpose(1,2) @ mol.C_mo
    print('** dgemm1.shape', dgemm1.shape)
    print(dgemm1)
    dgemm2 =  mol.C_mo[:, :, mol.nocc:mol.norb].transpose(1,2) @ dgemm1[:, :, :mol.nocc]
    dgemm2 = dgemm2.transpose(1,2).flatten()
    G_mo = dgemm2 # keep dgemm2 so far for nexmd comparison
    print('** G_mo.shape', G_mo.shape)
    print(G_mo)
    return G_mo




def ao2mo(device, N_cis, V, mol, full=True):
    """
    transform matrix from AO to MO basis

    Parameters
    ----------
    M_AO : torch tensor # TODO add size
        matrix in AO basis
    C : torch tensor # TODO add size
        matrix of MO coefficients # TODO row or columns, structure?
        
    Returns
    -------
    M_MO : torch tensor # TODO add size
        matrix in MO basis
    """     
    if full == True:
        V_rightsize = torch.zeros((mol.nmol, N_cis, mol.norb), device=device)
        M_mo = mol.C_mo.transpose(1,2) @ V.transpose(1,2) @ mol.C_mo
        M_mo = M_mo.transpose(1,2)
        V_rightsize[:M_mo.shape[0], :M_mo.shape[1]] = M_mo
        return V_rightsize
        
    else:
         # COPY of subroutine site2mo from Lioville
         
        G_ao = V # TODO rename
        
        # eta1 = eta1.view(-1, m.nvirt[0]) # 1d -> 2d
        # print(eta1.shape)
        # print('eta1', eta1)
        # print('==============')
        
        eta_mo = torch.zeros((N_rpa))
       # eta_mo = torch.zeros((m.norb, m.norb), device=device)

        dgemm1 = G_ao.T @ m.C_mo[0]

        # print('dgemm1.shape', dgemm1.shape)
        # print(dgemm1)
        
        dgemm2 =  m.C_mo[0][:, m.nocc:m.norb].T @ dgemm1[:,:m.nocc]
        

        dgemm2 = dgemm2.T.flatten()
        eta_mo[:dgemm2.size(0)] = dgemm2 
        # print('eta_mo', eta_mo.shape)
        # print(eta_mo)
        
        dgemm3 =  dgemm1[:, m.nocc:].T @ m.C_mo[0][:, :m.nocc]
        
        # print('dgemm3.T.shape', dgemm3.T.shape)
        # print(dgemm3.T)
        
        eta_mo[N_cis:] = dgemm3.T.flatten() 
        # print('eta_mo', eta_mo.shape)
        # print(eta_mo)

        M_mo = eta_mo
    
    return M_mo


def mo2ao(device, N_cis, V, mol, full=True):
    """
    transform matrix from AO to MO basis

    Parameters
    ----------
    M_AO : torch tensor # TODO add size
        matrix in AO basis
    C : torch tensor # TODO add size
        matrix of MO coefficients # TODO row or columns, structure?
        
    Returns
    -------
    M_MO : torch tensor # TODO add size
        matrix in MO basis
    """    
    # print('mol.c mo shape', mol.C_mo.shape)
    # print('mol.c_mo',  mol.C_mo)
    # print('M mo shape', M_mo.shape)
    print('V ENTER mo2ao')#
    print('*** V.shape', V.shape)
   # print('mol C_mo', mol.C_mo)
    print('*** mol C_mo shape', mol.C_mo.shape)
    # print('mol.C_mo[:, mol.nocc:mol.norb].transpose(1,2) shape', mol.C_mo[:, mol.nocc:mol.norb].transpose(1,2).shape)
    
   # print('V.shape', V.shape)
   # print('V device', V.device)
  #  print('V', V)
    # V = V.view(-1, mol.nvirt)
    # print('*** V.shape', V.shape)
  #  V_ao = mol.C_mo @ V @ mol.C_mo[:, mol.nocc:mol.norb].transpose(1,2)
    #V_ao = V @ mol.C_mo[:, :, mol.nocc:mol.norb].transpose(1,2)
    # TODO: fix left multiplication by C_mo
    print('*** V[:, :mol.norb, :].shape', V[:, :mol.norb, :].shape)
    print('*** V.shape', V.shape)
    # V_ao = torch.zeros_like(mol.C_mo[0])
    # V_ao = mol.C_mo @ V[:, :mol.norb, :] @ mol.C_mo.transpose(1,2)
    if full == True:
        
        V_ao = mol.C_mo @ V[:, :mol.norb, :mol.norb] @ mol.C_mo.transpose(1,2)
        #V_ao = mol.C_mo @ V_ao
        print('HHHHHHHHHHHHHHHHHH')
        #mol.C_mo @ V.transpose(1,2) 
        # V_ao = mol.C_mo @ V.transpose(1,2) 
        # print('*** V_ao.shape', V_ao.shape)
        # print('*** V_ao.transpose(1,2).shape', V_ao.transpose(1,2).shape)
        # print('*** mol.C_mo.shape', mol.C_mo.shape)
        
        # V_ao = V_ao.transpose(1,2) @ mol.C_mo # !!!ALERT COULD BE WRONG !!!!!!!!!!!!!!!!!!!
        # V_ao = V @ mol.C_mo.transpose(1,2)

    # else:

    #     eta = V # TODO rename
        
    #     eta1 = eta[:N_cis]
    #     eta1 = eta1.view(-1, mol.nvirt) # 1d -> 2d
    #     print(eta1.shape)
    #     print('eta1', eta1)
    #     print('==============')
        
    #     eta_mo = torch.zeros((m.norb, m.norb), device=device)

    #     dgemm1 = eta1 @ m.C_mo[0][:, m.nocc:m.norb].T # operations on |X| ?

    #     # print('dgemm1.shape', dgemm1.shape)
    #     # print(dgemm1)
        
    #     eta_mo[:m.nocc] = dgemm1
    #     # print('eta_mo', eta_mo.shape)
    #     # print(eta_mo)
        
        
    #     eta2 = eta[N_cis:]                            # operations on |Y| ?
    #     eta2 = eta2.view(-1, m.nvirt[0]) # 1d -> 2d
    
    #     dgemm2 = eta2.T @ m.C_mo[0][:, :m.nocc].T
    # print('*** V_ao.shape', V_ao.shape)

    return V_ao 






















    # if full == True:
        
    #     eta = M_mo # TODO rename
    #     print(eta.shape)
    #     print('eta', eta)
    #     eta1 = eta[:N_cis]
    #     eta1 = eta1.view(-1, mol.nvirt) # 1d -> 2d
        
        
        
    #     print(eta1.shape)
    #     print('eta1', eta1)
    #     print('==============')
    #     M_ao = mol.C_mo.transpose(1,2) @ M_mo @ mol.C_mo #! does not currently work
        
        

    #     return M_ao
    
    # else:
        
    #     eta = M_mo # TODO rename
        
    #     eta1 = eta[:N_cis]
    #     eta1 = eta1.view(-1, mol.nvirt) # 1d -> 2d
    #     print(eta1.shape)
    #     print('eta1', eta1)
    #     print('==============')
        
    #     eta_mo = torch.zeros((mol.norb, mol.norb), device=device)

    #     dgemm1 = eta1 @ mol.C_mo[:, mol.nocc:mol.norb].T # operations on |X| ?

    #     # print('dgemm1.shape', dgemm1.shape)
    #     # print(dgemm1)
        
    #     eta_mo[:mol.nocc] = dgemm1
    #     # print('eta_mo', eta_mo.shape)
    #     # print(eta_mo)
        
        
    #     eta2 = eta[N_cis:]                            # operations on |Y| ?
    #     eta2 = eta2.view(-1, m.nvirt[0]) # 1d -> 2d
    
    #     dgemm2 = eta2.T @ mol.C_mo[0][:, :mol.nocc].T
        
    #     # print('dgemm2.shape', dgemm2.shape)
    #     # print(dgemm2)
        
    #     eta_mo[m.nocc:] = dgemm2
    #     # print('eta_mo', eta_mo.shape)
    #     # print(eta_mo)
        
    #     dgemm3 = mol.C_mo[0] @ eta_mo
    #     eta_ao = dgemm3 
    
    
    
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

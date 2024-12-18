"""

  module contains functiins to build excited state Hamiltonian
  currently only CIS is implemented
  
  FUNCTIONS:
    form_cis - build A matrix for CIS
    mult_by_gap - multiply A element-wise by MO energy (ϵa−ϵi) like in Aia,jb=δijδab(ϵa−ϵi)+⟨aj||ib⟩
    build_G_sym - build 2e integrals in AO basis; notation: G(symmetric part of guess density) in F = H_core + G
                                                            note: H_core is not used in excited calculations
    build_G_antisym - build 2e integrals in AO basis; notation: G(antisymmetric part of guess density)
    gen_V - generate orthohonal guess vector based on MO 

"""
import torch
from seqm.seqm_functions.excited.orb_transform import mo2ao, ao2mo, decompose_to_sym_antisym
from seqm.seqm_functions.pack import pack, unpack

dtype = torch.float64 # TODO feed as param

def form_cis(device, V, mol, N_cis, CIS = True):
    """
    build A matrix for CIS
    splits guess density into symmetric and antisymmetric parts
    unclear why returns A @ b (guess vector)
    
    analogue of NEXMD Lxi_testing routine
    everything is/will be renamed for consistency with chemical papers, not Liouvuille formalism
    #! RPA is not implemented yet
    # TODO: implement TDHF/RPA
    Parameters:
    - V (tensor): guess vector based on MO (0 and 1 only)
    - molecule (PYSEQM object): batch of molecules
    - N_cis (int): dimension of CIS space, nocch*nvirt
    - CIS (bool, optional): CIS or TDHF (RPA) Defaults to True
    
    Returns:
      - G_tot_mo (torch.Tensor): column of CIS Hamiltonian in MO basis 
    """        
    gss = mol.parameters['g_ss']
    gsp = mol.parameters['g_sp']
    gpp = mol.parameters['g_pp']
    gp2 = mol.parameters['g_p2']
    hsp = mol.parameters['h_sp']  
    
    mask  = mol.mask
    maskd = mol.maskd
    idxi  = mol.idxi
    idxj  = mol.idxj
    nmol  = mol.nmol
    molsize = mol.molsize
    w       = mol.w
    nHeavy = mol.nHeavy
    nHydro = mol.nHydro
    
    V_orig = torch.clone(V)
    V_ao =  mo2ao(device, N_cis, V, mol)     # mo to ao basis (mo2site)
    print('V_ao.shape', V_ao.shape)
    print(V_ao)

    V_ao_sym, V_ao_asym = decompose_to_sym_antisym(V_ao) # decompose guess in ao to sym and asym
    # Vxi - build 2e integrals in AO basis: G(guess density) in F = H_core + G
    # note density is split into sym and anisym matrices
    # sym is processed as padded 3d array in PYSEQM, see fock module 
    # antisym: 2c-2e works with modified PYSEQM routine; should be antisimmterized afterwards
    # antisym: 1c-2e (diagonal)

    #------------------symmetric (1c-2e + 2c-2e)------------------------------
    G_sym   =  build_G_sym(device, V_ao_sym,
                           gss, gsp, gpp, gp2, hsp,
                           mask, maskd, idxi, idxj, nmol, molsize,
                           w,
                           nHeavy,
                           nHydro)
    # G sym is 1c-2e and 2c-2e of symmetric part of guess density
    G_sym_ao = pack(G_sym, nHeavy, nHydro) # pack 2c-2e part to standard shape of (norb x norb)
    
    
    G_antisym_ao = build_G_antisym(device, V_ao, V_ao_asym, G_sym,
                            gss, gsp, gpp, gp2, hsp,
                            mask, maskd, idxi, idxj, nmol, molsize,
                            w, 
                            mol,
                            nHeavy,
                            nHydro)
    
    G_tot_ao = G_sym_ao + G_antisym_ao

    G_tot_mo = ao2mo(device, N_cis, G_tot_ao, mol) # G_tot back to MO basis 
                                                         # G_tot_mo is a column of Hamiltonian 
                                                         # similar to L_xi in NEXMD
    
    G_tot_mo = torch.squeeze(G_tot_mo, 0) #TODO: remove, for MRS23 only
    G_tot_mo = mult_by_gap(device, G_tot_mo, N_cis, mol, V_orig)
    
    return G_tot_mo
  
def build_G_sym(device, M_ao,
                gss, gsp, gpp, gp2, hsp,
                mask, maskd, idxi, idxj, nmol, molsize,
                w,
                nHydro,
                nHeavy):
      """
      Builds the 1c-2e and 2c-2e part of Hamiltonian based on symmetric density 

      Parameters:
      - device (torch.device): The device on which to perform the calculations.
      - M_ao (torch.Tensor): symmetric density - should be in ao basis
      - gss (torch.Tensor): semi-empirical table param
      - gsp (torch.Tensor): -//-
      - gpp (torch.Tensor): -//-
      - gp2 (torch.Tensor): -//-.
      - hsp (torch.Tensor): -//-
      - mask (torch.Tensor): see original implementation of PYSEQM, was not documented
      - maskd (torch.Tensor): -//-
      - idxi (torch.Tensor):  -//-
      - idxj (torch.Tensor):  -//-
      - nmol (int): The number of molecules in batch
      - molsize (int): The size of largest moelecule - to unify tensor shapes
      - w (torch.Tensor): The 1c-2e and 2c-2e integrals (ERI) #TODO: rename consistently
      - nHydro (int): The number of hydrogen (H) atoms.
      - nHeavy (int): The number of heavy (non-H) atoms.

      Returns:
      - F0 (torch.Tensor): Fock matrix based on symmetric guess
      """
    
      F = torch.zeros((nmol*molsize**2,4,4), device=device) # 0 Fock matrix to fill
      P0 = unpack(M_ao, nHydro, nHeavy, (nHeavy+nHydro)*4) # 

      #---------------fill diagonal 1c-2e -------------------
      P = P0.reshape((nmol,molsize,4,molsize,4)) \
          .transpose(2,3).reshape(nmol*molsize*molsize,4,4)
          
      
      Pptot = P[...,1,1]+P[...,2,2]+P[...,3,3]
      ## http://openmopac.net/manual/1c2e.html
    #  (s,s)
      TMP = torch.zeros_like(F)
      TMP[maskd,0,0] = 0.5*P[maskd,0,0]*gss + Pptot[maskd]*(gsp-0.5*hsp)
      for i in range(1,4):
          #(p,p)
          TMP[maskd,i,i] = P[maskd,0,0]*(gsp-0.5*hsp) + 0.5*P[maskd,i,i]*gpp \
                          + (Pptot[maskd] - P[maskd,i,i]) * (1.25*gp2-0.25*gpp)
          #(s,p) = (p,s) upper triangle
          TMP[maskd,0,i] = P[maskd,0,i]*(1.5*hsp - 0.5*gsp)
      #(p,p*)
      for i,j in [(1,2),(1,3),(2,3)]:
          TMP[maskd,i,j] = P[maskd,i,j]* (0.75*gpp - 1.25*gp2)

      F.add_(TMP)
      
      #-----------------fill 2c-2e integrals----------------
      weight = torch.tensor([1.0,
                        2.0, 1.0,
                        2.0, 2.0, 1.0,
                        2.0, 2.0, 2.0, 1.0],dtype=dtype, device=device).reshape((-1,10))
      
      PA = (P[maskd[idxi]][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).reshape((-1,10,1))
      PB = (P[maskd[idxj]][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).reshape((-1,1,10))
      suma = torch.sum(PA*w,dim=1)
      sumb = torch.sum(PB*w,dim=2)
      sumA = torch.zeros(w.shape[0],4,4,dtype=dtype, device=device)
      sumB = torch.zeros_like(sumA)
      
      sumA[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = suma
      sumB[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = sumb
      F.index_add_(0,maskd[idxi],sumB)
      F.index_add_(0,maskd[idxj],sumA)
      
      sum = torch.zeros(w.shape[0],4,4,dtype=dtype, device=device)
      # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
      #   0,     1         2       3       4         5       6      7         8        9
      ind = torch.tensor([[0,1,3,6],
                          [1,2,4,7],
                          [3,4,5,8],
                          [6,7,8,9]],dtype=torch.int64, device=device)
      
      Pp = -0.5*P[mask]
      for i in range(4):
        for j in range(4):
            #\sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            sum[...,i,j] = torch.sum(Pp*w[...,ind[i],:][...,:,ind[j]],dim=(1,2))
      F.index_add_(0,mask,sum)

      F0 = F.reshape(nmol,molsize,molsize,4,4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
      F0.add_(F0.triu(1).transpose(1,2))     
      F0 = 2 * F0 
      return F0
  
  
def build_G_antisym(device, eta_ao, eta_ao_asym, G_sym,
                    gss, gsp, gpp, gp2, hsp,
                    mask, maskd, idxi, idxj, nmol, molsize,
                    w, 
                    mol,
                    nHydro,
                    nHeavy):

      # #--------------- diagonal 1c-2e ( baded on original PYSEQM code by G. Zhou)
      # vectorized to match nexmd 
      F = torch.zeros((nmol*molsize**2,4,4), device=device) # 0 Fock matrix to fill
      P0 = unpack(eta_ao_asym, nHydro, nHeavy, (nHeavy+nHydro)*4) # 

      #---------------fill diagonal 1c-2e -------------------
      P = P0.reshape((nmol,molsize,4,molsize,4)) \
          .transpose(2,3).reshape(nmol*molsize*molsize,4,4)
          
      Pptot = P[...,1,1]+P[...,2,2]+P[...,3,3]
      
      TMP = torch.zeros_like(F)
      
      # modified by FNS
      for i in range(1,4):
          TMP[maskd,0,i] = P[maskd,0,i]*(hsp - gsp)
      #(p,p*)
      for i,j in [(1,2),(1,3),(2,3)]:
          TMP[maskd,i,j] = 2*P[maskd,i,j]* (0.25*gpp - 0.6*1.25*gp2)
          
      TMP = TMP.reshape(nmol,molsize,molsize,4,4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
             
      TMP = pack(TMP, mol.nHeavy, mol.nHydro)
      TMP =  ( TMP - TMP.transpose(1,2))
      #--------------- diagonal 1c-2e -------
      
      #---------------- 2c-2e integrals----------------
      weight = torch.tensor([1.0,
                        2.0, 1.0,
                        2.0, 2.0, 1.0,
                        2.0, 2.0, 2.0, 1.0],dtype=dtype, device=device).reshape((-1,10))
      
      PA = (P[maskd[idxi]][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).reshape((-1,10,1))
      PB = (P[maskd[idxj]][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).reshape((-1,1,10))
      suma = torch.sum(PA*w,dim=1)
      sumb = torch.sum(PB*w,dim=2)
      sumA = torch.zeros(w.shape[0],4,4,dtype=dtype, device=device)
      sumB = torch.zeros_like(sumA)
      
      sumA[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = suma
      sumB[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = sumb
      F.index_add_(0,maskd[idxi],sumB)
      #\sum_A
      F.index_add_(0,maskd[idxj],sumA)
      
      sum = torch.zeros(w.shape[0],4,4,dtype=dtype, device=device)
      # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
      #   0,     1         2       3       4         5       6      7         8        9

      ind = torch.tensor([[0,1,3,6],
                          [1,2,4,7],
                          [3,4,5,8],
                          [6,7,8,9]],dtype=torch.int64, device=device)
      
      Pp = -0.5*P[mask]
      for i in range(4):
        for j in range(4):
            #\sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            sum[...,i,j] = torch.sum(Pp*w[...,ind[i],:][...,:,ind[j]],dim=(1,2))
     # print('mask', mask)    #! DIFFERS FROM PYSEQM, PROBABLY packing
      F.index_add_(0,mask,sum)

      F0 = F.reshape(nmol,molsize,molsize,4,4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)

      F0.add_(F0.triu(1).transpose(1,2))     
      
      F0 = pack(F0, mol.nHeavy, mol.nHydro)
      rows, cols = torch.tril_indices(F0.shape[1], F0.shape[2])
      
      F0[:, rows, cols] *= -1
      F0[:, torch.eye(F0.shape[1]).bool()] *= -1
      F0[0].diagonal().fill_(0)
      F0 = F0*2
      # TMP = 1c-2e antisym Fock 
      # F0  = 2c-2e antisym Fock
      G_antisym =  F0 + TMP
      
      return G_antisym 
  
  
def gen_V(device, mol, N_cis, n_V_start):
    
    # returns V (formerly vexp1) - guess vector for L-xi routine
    V = torch.zeros((mol.nmol, N_cis, N_cis), device=device)


    for m in range(mol.nmol):

      rrwork = torch.zeros(N_cis * 4, device=device) # TODO: VECTORIZE
      i = 0
      for ip in range(mol.nocc[m]):
          for ih in range(mol.nvirt[m]):
            rrwork[i] = mol.e_mo[m][mol.nocc[m] + ih] - mol.e_mo[m][ip] # 

     
      rrwork_sorted, indices = torch.sort(rrwork[:N_cis], descending=False, stable=True) # stable to preserve order of degenerate
      row_idx = torch.arange(0, int(N_cis), device=device)
      col_idx = indices[:N_cis]
      V[m, row_idx, col_idx] = 1.0    
                              
    V = V[:, :,  :n_V_start] # TODO: fix to initially generate no more than n_V_start
    print(" == GEN V ==")
    print('V shape', V.shape)
    print('V\n', V)                               
    return V

# TODO: check whether L_xi should be regenerated during expansion each time or just part of it


def mult_by_gap(device, G_mo, N_cis, mol, V_orig):
    
    nmol = mol.nmol # TODO move as ragument
    
    occ_idx = torch.arange(int(mol.nocc))
    virt_idx = torch.arange(int(mol.nocc), int(mol.norb))

    print('occ_idx\n', occ_idx)
    print('virt_idx\n', virt_idx)
    
    combined_idx = torch.cartesian_prod(occ_idx, virt_idx)  # combinations similar to itertools
    print('combined_idx\n', combined_idx)
    print('combined_idx shape', combined_idx.shape)
    
    mo_diff = mol.e_mo[0][combined_idx[:, 1]] - mol.e_mo[0][combined_idx[:, 0]]  # difference between virtual and occupied
    # print('mo diff', mo_diff)                                                                         # see how elements of A matrix are defined in any CIS paper
                                                                             # Aia,jb=δijδab(ϵa−ϵi)+⟨aj||ib⟩
    mo_kronecker = torch.zeros((N_cis), dtype=torch.float64)
    mo_kronecker[:N_cis] = (mo_diff * V_orig) 
    

    G_mo += mo_kronecker 
    # print('mo_kronecker\n', mo_kronecker)
    # print('G_mo AFTER FAST', G_mo)
    return G_mo
    # nmol = mol.nmol
    # max_norb = max(mol.norb)
    # max_nocc = max(mol.nocc)
    # occ_idx_2d = [torch.arange(max_norb + 1, dtype=torch.float32) for mol.norb in mol.norb.tolist()]
    # print('occ_idx_2d\n', occ_idx_2d)
    # mol.nocc = torch.tensor([[3], [5]])
    # occ_idx_2d = torch.zeros(nmol, int(torch.max(mol.nocc)))
    # occ_idx_2d[:, :int(mol.nocc)] = torch.arange((int(mol.nocc)))
    

    
    # torch.arange(nmol, int(torch.max(mol.nocc)))
    # virt_idx_2d = torch.arange(nmol, int(torch.max(mol.nocc)), int(torch.max(mol.norb)))
    
    # occ_idx = torch.arange(int(torch.max(mol.nocc)))
    
    # virt_idx = torch.arange(int(torch.max(mol.nocc)), int(torch.max(mol.norb)))
    
    # print('occ_idx\n', occ_idx)
    # print('occ index shape', occ_idx.shape)
    # print('virt_idx\n', virt_idx)
    # print('virt index shape', virt_idx.shape)
    # print()

    # occ_idx = torch.arange(int(torch.max(mol.nocc))) #.unsqueeze(0)
    # virt_idx = torch.arange(int(torch.max(mol.nocc)), int(torch.max(mol.norb))) #.unsqueeze(0)

    # combined_idx = torch.cartesian_prod(occ_idx, virt_idx)

    # mo_diff = mol.e_mo[combined_idx[:, 1]] - mol.e_mo[0][combined_idx[:, 0]]

    # mo_kronecker = torch.zeros((N_cis * 2, nmol), dtype=torch.float64)
    # mo_kronecker[:N_cis] = (mo_diff * V_orig.unsqueeze(1).T)

    G_mo += mo_kronecker

    return G_mo

#===================================================================





def mult_by_gap_copy(device, G_mo, N_cis, mol, V_orig):
    
        print('eta ORIG FASR', eta_orig)
        print('G_mo\n', G_mo)

        nmol = mol.nmol # TODO move as ragument
        
        occ_idx = torch.arange(int(mol.nocc))
        virt_idx = torch.arange(int(mol.nocc), int(mol.norb))

        print('occ_idx\n', occ_idx)
        print('virt_idx\n', virt_idx)
        
        combined_idx = torch.cartesian_prod(occ_idx, virt_idx)  # combinations similar to itertools
        print('combined_idx\n', combined_idx)
        print('combined_idx shape', combined_idx.shape)
        
        mo_diff = mol.e_mo[combined_idx[:, 1]] - mol.e_mo[0][combined_idx[:, 0]]  # difference between virtual and occupied
        # print('mo diff', mo_diff)                                                                         # see how elements of A matrix are defined in any CIS paper
                                                                                 # Aia,jb=δijδab(ϵa−ϵi)+⟨aj||ib⟩
        mo_kronecker = torch.zeros((N_cis * 2), dtype=torch.float64)
        mo_kronecker[:N_cis] = (mo_diff * V_orig) 
        

        G_mo += mo_kronecker 
        print('mo_kronecker\n', mo_kronecker)
        print('G_mo AFTER FAST', G_mo)
        return G_mo
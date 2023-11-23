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
from seqm.seqm_functions.excited.orb_transform import mo2ao_nexmd, ao2mo_nexmd
from seqm.seqm_functions.pack import pack, unpack

dtype = torch.float64 # TODO feed as param

def form_cis_nexmd(device, V, mol, N_cis, CIS = True):
    #MIMICS NEXMD for DEBUGGING
    """
    build A matrix for CIS
    splits guess density into symmetric and antisymmetric parts
    unclear why returns A @ b (guess vector)
    
    analogue of NEXMD Lxi_testing routine
    everything is renamed for consistency with chemical papers, not Liouvuille formalism
    #! RPA is not implemented yet
    # TODO: implement TDHF/RPA
    Args:
        vexp1 (tensor): guess vector
        molecule (PYSEQM object): batch of molecules
        N_cis (int): dimension of CIS space, nocch*nvirt
        N_rpa (int): N_cis *2
        CIS (bool, optional): CIS or TDHF (RPA) Defaults to True.

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
    V_ao =  mo2ao_nexmd(device, N_cis, V, mol)     # mo to ao basis (mo2site)
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
    G_sym = pack(G_sym, nHeavy, nHydro) # pack 2c-2e part to standard shape of (norb x norb)
    
    
    G_tot = build_G_antisym(device, V_ao, V_ao_asym, G_sym,
                            gss, gsp, gpp, gp2, hsp,
                            mask, maskd, idxi, idxj, nmol, molsize,
                            w, 
                            mol,
                            nHeavy,
                            nHydro)
    
   # print('G_tot.shape', G_tot.shape)
   # print('G_tot\n', G_tot)                  
    # build_G_antisym returns both sym and antisym!
    # TODO: refactor into: 2c-2e antisym, 1c-2e antisym
    # TODO: vectorize 1c-2e antisym, avoid ugly loops
    #! remember about making 2c-2e diagonal 0

    # print('G total \n', G_tot)
    
    # print('============================================')
    # print('Converting Gao full back into MO basis')
    G_mo = ao2mo_nexmd(device, N_cis, G_tot, mol) # G in MO basis #! [0] not batched yet
    
    # print('!!! G_mo.shape AFTER AO to MO', G_mo.shape)
    # print('!!! G_mo AFTER AO to MO\n', G_mo)
    
    # multiply by MO differencies
    # G_mo = mult_by_gap(G_mo, N_cis, mol, V_orig)

    return G_mo
  
  
  

def form_cis_nexmd_working_copy(device, V, mol, N_cis, CIS = True):
    #MIMICS NEXMD for DEBUGGING
    """
    build A matrix for CIS
    splits guess density into symmetric and antisymmetric parts
    unclear why returns A @ b (guess vector)
    
    analogue of NEXMD Lxi_testing routine
    everything is renamed for consistency with chemical papers, not Liouvuille formalism
    #! RPA is not implemented yet
    # TODO: implement TDHF/RPA
    Args:
        vexp1 (tensor): guess vector
        molecule (PYSEQM object): batch of molecules
        N_cis (int): dimension of CIS space, nocch*nvirt
        N_rpa (int): N_cis *2
        CIS (bool, optional): CIS or TDHF (RPA) Defaults to True.

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
    V_ao =  mo2ao_nexmd(device, N_cis, V, mol)     # mo to ao basis (mo2site)
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
    G_sym = pack(G_sym, nHeavy, nHydro) # pack 2c-2e part to standard shape of (norb x norb)
    
    
    G_tot = build_G_antisym(device, V_ao, V_ao_asym, G_sym,
                            gss, gsp, gpp, gp2, hsp,
                            mask, maskd, idxi, idxj, nmol, molsize,
                            w, 
                            mol,
                            nHeavy,
                            nHydro)
    
   # print('G_tot.shape', G_tot.shape)
   # print('G_tot\n', G_tot)                  
    # build_G_antisym returns both sym and antisym!
    # TODO: refactor into: 2c-2e antisym, 1c-2e antisym
    # TODO: vectorize 1c-2e antisym, avoid ugly loops
    #! remember about making 2c-2e diagonal 0

    # print('G total \n', G_tot)
    
    # print('============================================')
    # print('Converting Gao full back into MO basis')
    G_mo = ao2mo_nexmd(device, N_cis, G_tot, mol) # G in MO basis #! [0] not batched yet
    
    # print('!!! G_mo.shape AFTER AO to MO', G_mo.shape)
    # print('!!! G_mo AFTER AO to MO\n', G_mo)
    
    # multiply by MO differencies
    # G_mo = mult_by_gap(G_mo, N_cis, mol, V_orig)



    return G_mo

def form_cis(device, V, mol, N_cis, N_rpa, CIS = True):
    """
    build A matrix for CIS
    splits guess density into symmetric and antisymmetric parts
    unclear why returns A @ b (guess vector)
    
    analogue of NEXMD Lxi_testing routine
    everything is renamed for consistency with chemical papers, not Liouvuille formalism
    #! RPA is not implemented yet
    # TODO: implement TDHF/RPA
    Args:
        
        V (tensor): guess vector
        mol (PYSEQM object): batch of molecules
        N_cis (int): dimension of CIS space, nocch*nvirt
        N_rpa (int): N_cis *2
        CIS (bool, optional): CIS or TDHF (RPA) Defaults to True.

    """        

    #m = molecule # DEBUG 
    #logger 
    gss = mol.parameters['g_ss']
    gsp = mol.parameters['g_sp']
    gpp = mol.parameters['g_pp']
    gp2 = mol.parameters['g_p2']
    hsp = mol.parameters['h_sp']  #logger 
    
    mask  = mol.mask
    maskd = mol.maskd
    idxi  = mol.idxi
    idxj  = mol.idxj
    nmol  = mol.nmol
    molsize = mol.molsize
    w       = mol.w
    nHeavy = mol.nHeavy
    nHydro = mol.nHydro

    V_ao =  mo2ao(device, N_cis, V, mol, full=True)   # mo to ao basis (mo2site)
    V_ao_sym, V_ao_asym = decompose_to_sym_antisym(V_ao) # decompose guess in ao basis to sym and asym

    # Vxi - build 2e integrals in AO basis: G(guess density) in F = H_core + G
    # note density is split into sym and anisym matrices
    # sym is processed as padded 3d array in PYSEQM, see fock module 
    # antisym: 2c-2e works with modified PYSEQM routine; should be antisimmterized afterwards
    # antisym: 1c-2e (diagonal) are taken from NEXMD for now - ugly code with loops
    
    #------------------symmetric (1c-2e + 2c-2e)------------------------------
    G_sym   =  build_G_sym(device, V_ao_sym,
                           gss, gsp, gpp, gp2, hsp,
                           mask, maskd, idxi, idxj, nmol, molsize,
                           w,
                           nHeavy,
                           nHydro)
    
    # G sym is 1c-2e and 2c-2e of symmetric part of guess density
    # pack 2c-2e part to standard shape of (norb, norb)
    G_sym = pack(G_sym, nHeavy, nHydro)
    # G_tot = build_G_antisym(device, V_ao, V_ao_asym, G_sym,
    #                         gss, gsp, gpp, gp2, hsp,
    #                         mask, maskd, idxi, idxj, nmol, molsize,
    #                         w, 
    #                         mol,
    #                         nHeavy,
    #                         nHydro)
    
   # print('G_tot.shape', G_tot.shape)
   # print('G_tot\n', G_tot)                  
    # build_G_antisym returns both sym and antisym!
    # TODO: refactor into: 2c-2e antisym, 1c-2e antisym
    # TODO: vectorize 1c-2e antisym, avoid ugly loops
    #! remember about making 2c-2e diagonal 0

    # print('G total \n', G_tot)
    
    # print('============================================')
    # print('Converting Gao full back into MO basis')
    G_mo = ao2mo(device, N_cis, G_sym, mol, full=True) # G in MO basis #! [0] not batched yet

    # print('G_mo.shape', G_mo.shape)
    # print('G_mo\n', G_mo)
    
    # multiply by MO differencies
    # G_mo = mult_by_gap(G_mo, N_cis, mol, V_orig)


    return G_mo


def mult_by_gap(G_mo, N_cis, mol, V_orig):
    
    # print('eta ORIG FASR', eta_orig)
    # print('G_mo\n', G_mo)

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
    mo_kronecker[:N_cis] = (mo_diff * eta_orig) 
    

    G_mo += mo_kronecker 
    # print('mo_kronecker\n', mo_kronecker)
    # print('G_mo AFTER FAST', G_mo)
    return G_mo

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

      # TODO; figure how/why constants are defined in fock_skew
      #!
      #! CHECK
      #!
      # TODO: for batch likely need to figure max norb value and pad by 0s
      # print('G SYM\n', G_sym)
      # for (mol.norb*(mol.norb+1)//2) see nth triangular number formula 
      eta_anti = torch.zeros(mol.nmol, (mol.norb*(mol.norb+1)//2), device=device)
      print('** eta_anti.shape', eta_anti.shape)
      indices = torch.zeros((nmol, 2, mol.norb*(mol.norb+1)//2)) 
      print('** indices', indices.shape)
      # indices_3d = torch.tril_indices(int(mol.norb), int(mol.norb), offset = 0)
      print('** eta_ao.shape', eta_ao.shape)
      for i in range(nmol): # torch.tril for 2d matrices only # TODO:  vectorize
        indices[i] = torch.tril_indices(int(mol.norb[i]), int(mol.norb[i]), offset = 0) #TODO: likely to use molsize here
      indices = indices.long()
      print('** indices', indices.shape)
      print('** indices[:,0]', indices[:,0])
      # print('** eta_ao', eta_ao[:, indices[0], indices[1]])
      eta_anti = 0.5 * (eta_ao[:, indices[:, 0], indices[:, 1]] - eta_ao[:, indices[:, 1], indices[:, 0]])
      print('** eta_anti 0.5', eta_anti.shape)
      print('** eta_anti 0.5', eta_anti)
      
      eta_anti_2d = torch.zeros((nmol, mol.norb, mol.norb), device='cpu') 
      eta_anti_2d[:, indices[:, 1], indices[:, 0]] = -eta_anti 
      eta_anti_2d = eta_anti_2d - eta_anti_2d.transpose(1,2)
      
      print('** eta_anti_2d', eta_anti_2d.shape)
      print('** eta_anti_2d', eta_anti_2d)
      
      # ===== VECTORIZED ============
      F = torch.zeros((nmol*molsize**2,4,4), device=device) # 0 Fock matrix to fill
      # # TODO: feed params programmatically
      
      P0 = unpack(eta_anti_2d, nHydro, nHeavy, (nHeavy+nHydro)*4) # 
    #   print('P0.shape', P0.shape)
    #   print('P0\n', P0)
      
     # P0 = torch.unsqueeze(P0, 0) # add dimension
      
      # print('P0.shape', P0.shape)
      # print('P0\n', P0)
      #---------------fill diagonal 1c-2e -------------------
      P = P0.reshape((nmol,molsize,4,molsize,4)) \
          .transpose(2,3).reshape(nmol*molsize*molsize,4,4)
          
      Pptot = P[...,1,1]+P[...,2,2]+P[...,3,3]
      
      TMP = torch.zeros_like(F)
      
      #! MODIFIED BY FNS
      for i in range(1,4):
          #(p,p)
          # ! TWO LINES BELOW COULD BE NEEDED, NOT SURE
          # TMP[maskd,i,i] = P[maskd,0,0]*(gsp-hsp) + \
          #                 + (Pptot[maskd] - P[maskd,i,i]) * (0.6*1.25*gp2-0.25*gpp)
         # (s,p) = (p,s) upper triangle
          # TMP[maskd,i,i] = 100
          TMP[maskd,0,i] = P[maskd,0,i]*(hsp - gsp)
      #(p,p*)
      for i,j in [(1,2),(1,3),(2,3)]:
          TMP[maskd,i,j] = 2*P[maskd,i,j]* (0.25*gpp - 0.6*1.25*gp2)

    #   print('*** TMP *** \n', TMP)
     # ! MAYBE SHOULD be ANTISYMMETRIZD TMP = 
      # TMP = 0.5* ( TMP - TMP.T)
      # F.add_(TMP)  
          
      TMP = TMP.reshape(nmol,molsize,molsize,4,4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
             
      TMP = pack(TMP, mol.nHeavy, mol.nHydro)
      TMP =  ( TMP - TMP.transpose(1,2))
      print('*$$$ G_anti_1c2e\n', TMP)
      # print('*** TMP *** \n', TMP.shape)
      
      # G_anti_1c2e = torch.zeros(nmol, mol.norb, mol.norb)
      # print('** G_anti_1c2e.shape', G_anti_1c2e.shape)
      # print('** G_anti_1c2e\n', G_anti_1c2e)

      
      # G_anti_1c2e[:, indices[:, 1], indices[:, 0]] = -TMP
      # G_anti_1c2e = G_anti_1c2e - G_anti_1c2e.transpose(1,2)
      # # print('TMP.shape', TMP.shape)
      # print('TMP\n', TMP)

      # build 2c-2e part of antisymmetric G
      # copied from FOCK
      # TODO is eta_ao_asym equal to transfomrations above?
      #P = P[...,1,1]+P[...,2,2]+P[...,3,3] #! MODIFIED
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
    #


      F0.add_(F0.triu(1).transpose(1,2))     
      
      F0 = pack(F0, mol.nHeavy, mol.nHydro)
      rows, cols = torch.tril_indices(F0.shape[1], F0.shape[2])
      
      F0[:, rows, cols] *= -1
      F0[:, torch.eye(F0.shape[1]).bool()] *= -1
      
      # F0 is still symmetric, probably symmetrized above
      # here we make it antisymmetric back
    #   F0 = 2 * F0 
      

       #! BE WARNED, THIS iS TAKEN FROM OLD NEXMD, PYSEQM produces non-zero diagonal
    #   F0 = F0
      # print('G ANTISYM shape', F0.shape)
      # print('G ANTISYM\n', F0*2)
    
    # BEFORE FINAL ASSEMBLY
     
      
      F0[0].diagonal().fill_(0)

      print('G ANTISYM shape', F0.shape)
      print('G ANTISYM\n', F0*2+TMP) 
      # print('G ANTISYM\n', F0*2+TMP)
     # G_sym = torch.unsqueeze(G_sym, 0)
      print(' %%% G ANTISYM 2C = F0\n', F0*2)
      G_full = G_sym +  F0*2 + TMP# summ of G_sym(sym 1c2e + 2c2e) + F0 (antisym 1c2e + 2c2e) 
      print('!!! G_full shape', G_full.shape)

      print('!!! G_full\n', G_full)

      return G_full # ! WORKS FOR ONE ONLY
  
  
def gen_V(device, mol, N_cis, N_rpa, n_V_start):
    
    # returns V (formerly vexp1) - guess vector for L-xi routine
    V = torch.zeros((mol.nmol, N_cis, N_cis), device=device)

    # print(' NOCC', mol.nocc)
    # print(' NVIRT', mol.nvirt)
    for m in range(mol.nmol):
      # print('m in gen V', m)
      # print()
      
      rrwork = torch.zeros(N_rpa * 4, device=device)
      i = 0
      for ip in range(mol.nocc[m]):
          for ih in range(mol.nvirt[m]):
            rrwork[i] = mol.e_mo[m][mol.nocc[m] + ih] - mol.e_mo[m][ip] # !Lancos vectors(i) ???
            i += 1                                                #  TODO: [0] should be replaced by m batch index
     
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
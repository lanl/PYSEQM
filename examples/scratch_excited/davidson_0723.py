# %%
import argparse

# === IMPORTS ===

import logging, sys
import torch
import seqm
from ase.io import read as ase_read
from seqm.seqm_functions.constants import Constants
from seqm.Molecule import Molecule
from seqm.ElectronicStructure import Electronic_Structure
from termcolor import colored


from seqm.seqm_functions.fock import fock
from seqm.seqm_functions.pack import unpack
import seqm.seqm_functions.pack as pack
import torch.nn.functional as F

#=== TORCH OPTIONS ===

torch.set_default_dtype(torch.float64)
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
device = torch.device('cpu')
dtype = torch.float64
torch.set_printoptions(precision=5, linewidth=200, sci_mode=False)

# %%
# colored logging with custom level QM for deeper routines

logging.basicConfig(level=logging.DEBUG,
                    format='%(funcName)s : %(lineno)d : %(levelname)s : %(message)s')

QM1 = evel=logging.DEBUG - 3 # informal level of depth; QM1 - almost always, usually outside of loops
QM2 = evel=logging.DEBUG - 4 #                          QM2 - sometimes, in the loops
QM3 = evel=logging.DEBUG - 5

logging.addLevelName(QM1, "QM1")
def qm1(self, message, *args, **kwargs):
    if self.isEnabledFor(QM1 ):
        self._log(QM1, message, args, **kwargs) 
        
logging.addLevelName(QM2, "QM2")
def qm2(self, message, *args, **kwargs):
    if self.isEnabledFor(QM2):
        self._log(QM2, message, args, **kwargs) 
 
logging.addLevelName(QM3, "QM3")
def qm3(self, message, *args, **kwargs):
    if self.isEnabledFor(QM3 ):
        self._log(QM3, message, args, **kwargs) 
           
        
logging.Logger.qm1 = qm1   
logging.Logger.qm2 = qm2
logging.Logger.qm3 = qm3
  
logger = logging.getLogger()

                              
colors = {'qm'        : ('cyan',     None, None),
          'matrix'    : ('blue',     None, ['bold']),
          'vector'    : ('yellow',   None, ['bold']),
          'evals'     : ('green',    None, ['bold']),
          'warn'     : ('red',    None, ['bold'])
          }

def fmt_log(data, message, fmt):
    """
    fmt_log : formats log message with color and style using termcolor module

    Args:
        data (any): data to print
        message (str or None): message to print, pass None if no message is needed
        fmt (str): style from colors dict

    Returns:
        str: formatted string with color and style
    """    

    if type(data) is list or type(data) is tuple or type(data) is torch.Tensor:
        
        mes = f'{colored(message, colors[fmt][0], colors[fmt][1], attrs=colors[fmt][2])}\n' # add new line to align array
    else:
        mes = f'{colored(message, colors[fmt][0], colors[fmt][1], attrs=colors[fmt][2])} : '
        
    if data == None:
        return mes
    else:
        return mes + str(colored(data, colors[fmt][0], colors[fmt][1], attrs=colors[fmt][2]))

# %% [markdown]
# ### log
# 
# 07/13/23 - QM part seems to be wortking fine
# full diagonalization agrees with NEXMD
# small guess space misses relevant vectors, but large guess includes them
# 
# 
# PASCAL 1 COULD BE INCORRECT

# %% [markdown]
# ### QM routines

# %%
def run_seqm_1mol(xyz):
    """
    run_seqm_1mol : run PYSEQM for a single molecule

    Args:
        xyz (str): path to xyz file

    Returns:
        Molecule object: PYSEQM object with molecule data
    """    
    
    atoms = ase_read(xyz)
    species = torch.tensor([atoms.get_atomic_numbers()], dtype=torch.long, device=device)
    coordinates = torch.tensor([atoms.get_positions()], dtype=dtype, device=device)
    
    const = Constants().to(device)

    elements = [0]+sorted(set(species.reshape(-1).tolist()))

    seqm_parameters = {
                    'method' : 'PM3',  # AM1, MNDO, PM#
                    'scf_eps' : 1.0e-6,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                    'scf_converger' : [2,0.0], # converger used for scf loop
                                            # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                            # [1], adaptive mixing
                                            # [2], adaptive mixing, then pulay
                    'sp2' : [False, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                                #[True, eps] or [False], eps for SP2 conve criteria
                    'elements' : elements, #[0,1,6,8],
                    'learned' : [], # learned parameters name list, e.g ['U_ss']
                    #'parameter_file_dir' : '../seqm/params/', # file directory for other required parameters
                    'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                    'eig' : True,
                    'excited' : True,
                    }

    mol = seqm.Molecule.Molecule(const, seqm_parameters, coordinates, species).to(device)

    ### Create electronic structure driver:
    esdriver = Electronic_Structure(seqm_parameters).to(device)

    ### Run esdriver on m:
    esdriver(mol)
    
    return mol

# %%
def form_L_xi_no_split(vexp1, molecule, N_cis, N_rpa, CIS = True):

    # WRONG
    
    m = molecule
    gss = m.parameters['g_ss']
    gsp = m.parameters['g_sp']
    gpp = m.parameters['g_pp']
    gp2 = m.parameters['g_p2']
    hsp = m.parameters['h_sp']
    
    mask  = m.mask
    maskd = m.maskd
    idxi  = m.idxi
    idxj  = m.idxj
    nmol  = m.nmol
    molsize = m.molsize
    w       = m.w
    nHeavy = m.nHeavy
    nHydro = m.nHydro
    
    eta = torch.zeros((N_rpa), device=device) 
    
    #print('vexp1.shape', vexp1.shape)
   # print('eta.shape', eta.shape)
    eta[:vexp1.size(0)] = vexp1 # eta is stored as |X|; dcopy?
    
    eta_orig = torch.clone(eta)
    # print('eta_orig.shape', eta_orig.shape)
    # print('eta_orig\n', eta_orig)
    
   # print('m.C MO', m.C_mo[0])
    # print('eta.shape', eta.shape)
    # print(eta)
    eta_ao =  mo2ao(N_cis, eta, m, full=False)     # mo to ao basis (mo2site)
    # print('eta_ao.shape', eta_ao.shape)
    # print(eta_ao)

    # eta_ao_sym, eta_ao_asym = decompose_to_sym_antisym(eta_ao) # decompose to sym and asym

    # Vxi - build 2e integrals in AO basis: G(guess density) in F = H_core + G
    # note density is split into sym and anisym matrices
    # sym is processed as padded 3d array in PYSEQM, see fock module 
    # antisym: 2c-2e works with modified PYSEQM routine; should be antisimmterized afterwards
    # antisym: 1c-2e (diagonal) are taken from NEXMD for now - ugly code with loops
    # TODO: vectorize 1c-2e part
    
    #------------------symmetric------------------------------
    G_sym   =  build_G_sym(eta_ao,
                        gss, gsp, gpp, gp2, hsp,
                        mask, maskd, idxi, idxj, nmol, molsize,
                        w,
                         nHeavy,
                         nHydro)
    # G sym is 1c-2e and 2c-2e of symmetric part of guess density
    
    
    
    # pack 2c-2e part to standard shape
    G_sym = pack.pack(G_sym, nHeavy, nHydro)
    # print('G_sym \n', G_sym)
    
    # G_tot = build_G_antisym(eta_ao, eta_ao_asym, G_sym,
    #                         gss, gsp, gpp, gp2, hsp,
    #                         mask, maskd, idxi, idxj, nmol, molsize,
    #                         w, 
    #                         m,
    #                         nHeavy,
    #                         nHydro)
                        
    # build_G_antisym returns both sym and antisym!
    # TODO: refactor into: 2c-2e antisym, 1c-2e antisym
    # TODO: vectorize 1c-2e antisym, avoid ugly loops
    #! remember about making 2c-2e diagonal 0

    # print('G total \n', G_tot)
    4
    # print('============================================')
    # print('Converting Gao full back into MO basis')
    G_mo = ao2mo(N_cis, N_rpa, m, G_sym[0], m.C_mo, full=False) # G in MO basis #! [0] not batched yet
    
    # multiply by MO differencies
    ii=0
    for p in range(m.nocc):
    # print('p', p)
        for h in range(m.nocc, m.norb):
            # print('h', h)
            # print('i', i)
            f = m.e_mo[0][h] - m.e_mo[0][p]
            G_mo[ii] = G_mo[ii] + f * eta_orig[ii]
            G_mo[ii+N_cis] = -G_mo[ii + N_cis] + f * eta_orig[ii+N_cis]
            ii += 1
        
    # print('G_mo.shape', G_mo.shape)
    # print('G_mo\n', G_mo)
    return G_mo

# %%
def form_L_xi(vexp1, molecule, N_cis, N_rpa, CIS = True):
    """
    form_L_xi: build A matrix for CIS
               splits guess density into symmetric and antisymmetric parts
               unclear why returns A @ b (guess vector)
               see NEXMD code for QM details
               #! RPA is not implemented yet
               
    Args:
        vexp1 (tensor): guess vector
        molecule (PYSEQM object): _description_
        N_cis (int): dimension of CIS space, nocch*nvirt
        N_rpa (int): N_cis *2
        CIS (bool, optional): CIS or TDHF (RPA) Defaults to True.

    """        

    m = molecule
    gss = m.parameters['g_ss']
    gsp = m.parameters['g_sp']
    gpp = m.parameters['g_pp']
    gp2 = m.parameters['g_p2']
    hsp = m.parameters['h_sp']
    
    mask  = m.mask
    maskd = m.maskd
    idxi  = m.idxi
    idxj  = m.idxj
    nmol  = m.nmol
    molsize = m.molsize
    w       = m.w
    nHeavy = m.nHeavy
    nHydro = m.nHydro
    
    eta = torch.zeros((N_rpa), device=device) 
    
    #print('vexp1.shape', vexp1.shape)
   # print('eta.shape', eta.shape)
    eta[:vexp1.size(0)] = vexp1 # eta is stored as |X|; dcopy?
    
    eta_orig = torch.clone(eta)
    # print('eta_orig.shape', eta_orig.shape)
    # print('eta_orig\n', eta_orig)
    
   # print('m.C MO', m.C_mo[0])
    # print('eta.shape', eta.shape)
    # print(eta)
    eta_ao =  mo2ao(N_cis, eta, m, full=False)     # mo to ao basis (mo2site)
    # print('eta_ao.shape', eta_ao.shape)
    # print(eta_ao)

    eta_ao_sym, eta_ao_asym = decompose_to_sym_antisym(eta_ao) # decompose to sym and asym

    # Vxi - build 2e integrals in AO basis: G(guess density) in F = H_core + G
    # note density is split into sym and anisym matrices
    # sym is processed as padded 3d array in PYSEQM, see fock module 
    # antisym: 2c-2e works with modified PYSEQM routine; should be antisimmterized afterwards
    # antisym: 1c-2e (diagonal) are taken from NEXMD for now - ugly code with loops
    # TODO: vectorize 1c-2e part
    
    #------------------symmetric------------------------------
    G_sym   =  build_G_sym(eta_ao_sym,
                        gss, gsp, gpp, gp2, hsp,
                        mask, maskd, idxi, idxj, nmol, molsize,
                        w,
                         nHeavy,
                         nHydro)
    
   # print('G_SYM\n', G_sym)
    # G sym is 1c-2e and 2c-2e of symmetric part of guess density
    
    
    
    # pack 2c-2e part to standard shape
    G_sym = pack.pack(G_sym, nHeavy, nHydro)
    # print('G_sym \n', G_sym)
    
    G_tot = build_G_antisym(eta_ao, eta_ao_asym, G_sym,
                            gss, gsp, gpp, gp2, hsp,
                            mask, maskd, idxi, idxj, nmol, molsize,
                            w, 
                            m,
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
    G_mo = ao2mo(N_cis, N_rpa, m, G_tot[0], m.C_mo, full=False) # G in MO basis #! [0] not batched yet
    
    # print('G_mo.shape', G_mo.shape)
    # print('G_mo\n', G_mo)
    # multiply by MO differencies
    ii=0
    for p in range(m.nocc):
    # print('p', p)
        for h in range(m.nocc, m.norb):
            # print('h', h)
            # print('i', i)
            f = m.e_mo[0][h] - m.e_mo[0][p]
            G_mo[ii] = G_mo[ii] + f * eta_orig[ii]
            G_mo[ii+N_cis] = -G_mo[ii + N_cis] + f * eta_orig[ii+N_cis]
            ii += 1
        
    # print('G_mo.shape', G_mo.shape)
    # print('G_mo\n', G_mo)
    return G_mo

# %%
def decompose_to_sym_antisym(A):
    """
    decomposes matrix into symmetric and antisymmetric parts

    Args:
        A (tensor): some matrix

    Returns:
        tuple of tensors: sym and antisym parts
    """   

    A_sym = 0.5 * (A + A.T)
    A_antisym = 0.5 * (A - A.T)
    
    return A_sym, A_antisym

# %%
def build_G_sym(M_ao,
                gss, gsp, gpp, gp2, hsp,
                mask, maskd, idxi, idxj, nmol, molsize,
                w,
                nHydro,
                nHeavy):
    
    
      F = torch.zeros((nmol*molsize**2,4,4), device=device) # 0 Fock matrix to fill
      # # TODO: feed params programmatically
      
      P0 = unpack(M_ao, nHydro, nHeavy, (nHeavy+nHydro)*4) # 
      P0 = torch.unsqueeze(P0, 0) # add dimension
      
      # print('P0.shape', P0.shape)
      # print('P0\n', P0)
      #---------------fill diagonal 1c-2e -------------------
      P = P0.reshape((nmol,molsize,4,molsize,4)) \
          .transpose(2,3).reshape(nmol*molsize*molsize,4,4)
          
      # print('P.shape', P.shape)
      # print('P\n', P)
      
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
      #print('mask', mask)    #! DIFFERS FROM PYSEQM, PROBABLY packing
      F.index_add_(0,mask,sum)

      F0 = F.reshape(nmol,molsize,molsize,4,4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
    #
      F0.add_(F0.triu(1).transpose(1,2))     
      
      F0 = 2 * F0 #! BE CAREFUL
      # print('F0.shape', F0.shape)
      # print(F0) 
      
      return F0

# %%
def build_G_antisym(eta_ao, eta_ao_asym, G_sym,
                gss, gsp, gpp, gp2, hsp,
                mask, maskd, idxi, idxj, nmol, molsize,
                w, 
                m,
                nHydro,
                nHeavy):

      # TODO; figure how/why constants are defined in fock_skew
      #!
      #! CHECK
      #!
      # gss = torch.tensor([15.7558, 14.7942, 14.7942])  # GSSII   7.87788  7.39710   7.39710   1/2 # not used
      # gpp = torch.tensor([13.6540,  0.0000,  0.0000])  # GPPII   6.82700   1/2
      # gsp = torch.tensor([10.6212,  0.0000,  0.0000])  # GSPII   10.6211    
      # gp2 = torch.tensor([12.4061,  0.0000,  0.0000])  # GP2II   15.5076
      # hsp = torch.tensor([0.5939, 0.0000, 0.0000])     # HSPII   0.29694   1/2
  
      # see fock_skew in qm_fock in NEXMD
      # create 1d array 
      eta_anti = torch.zeros((m.norb*(m.norb+1)//2), device=device)
      
      # l=0
      # for i in range(m.norb):
      #       # print('i', i)
      #       for j in range(i+1):
      #             print('i j', i, j)

      #             eta_anti[l] = 0.5 * (eta_ao[i,j] - eta_ao[j,i])
      #             l += 1   
      #             # print('l', l) 
      # print('eta_anti NEXMD\n')
      # for e in eta_anti: print(f"{e:.4f}")
      
      eta_anti = torch.zeros((m.norb*(m.norb+1)//2), device=device)
      indices = torch.tril_indices(int(m.norb), int(m.norb), offset = 0)  # Generate the upper triangular indices
      
      # print('indices\n', indices)
      # print('indices[0]\n', indices[0].size())
      eta_anti = 0.5 * (eta_ao[indices[0], indices[1]] - eta_ao[indices[1], indices[0]])

      # print('eta_anti vectorized\n')
      # for e in eta_anti: print(f"{e:.4f}")
      
      
      # print tensor, one element per line
      
      eta_anti_2d = torch.zeros((m.norb, m.norb), device='cpu') 

      #restore to 2d form to build G 2c-2e part
      # l = 0
      # for i in range(0, m.norb): # TODO" vectorize
      #     for j in range(0,i):
      #         l += 1
      #         eta_anti_2d[i,j] += eta_anti[l-1]
      #         eta_anti_2d[j,i] -= eta_anti[l-1]
      #     l += 1 
      # print('eta_anti_2d\n', eta_anti_2d)
      
      # eta_anti_2d_vec = torch.zeros((m.norb, m.norb))     
      
      
      eta_anti_2d[indices[1], indices[0]] = -eta_anti 
      eta_anti_2d = eta_anti_2d - eta_anti_2d.T # antisymmetrize
      
      # print('eta_anti_2d VEC\n', eta_anti_2d)
      #eta_anti_2d_vec.fill_diagonal_(0)

      # print('maskd\n', maskd)

      # TODO: should be vectorized as in build G
      # below explicit working example for H2O taken from NEXMD
      # pascal2 = [1, 3, 6, 10, 15, 21]  # -1 pascal triangle
      # pascal2 = [x -1 for x in pascal2]
      # pascal1 = [0, 1, 3, 6, 10, 15] # -1 for python indexing
      # pascal1 = [x -1 for x in pascal1]
      # orb_loc1 = [0,4,5] # O orbs
      # orb_loc2 = [3,4,5]
      pascal2 = torch.cumsum(torch.arange(1, eta_anti_2d.shape[0]+1), dim=0)  # -1 pascal triangle
      # pascal2 = [-1, 0, 2, 5, 9, 14]
      pascal2 = pascal2 -1 
      # print('pascal2\n', pascal2)
      
      pascal1 = torch.cumsum(torch.arange(0, eta_anti_2d.shape[0]), dim=0) 
      pascal1 = pascal1 - 1
      # pascal1 = [0,  2,  5,  9, 14, 20, 27]
      # pascal1 = [-1, 0, 2, 5, 9, 14]
      # print('pascal1\n', pascal1)
      # print(m.Z)
      
      # ! TODO: write programamtically
      
      # orb_loc1 = [0] # [start of orbitals index of x]
      # orb_loc2 = [1 if m.Z[0] == 1 else 3] # [end of orbitals index of x]
      orb_loc1 = []
      orb_loc2 = []
      # orb_loc1 = [orb_loc1[i - 1] + 4 if m.Z[i] != 1 else orb_loc1[i - 1] + 1 for i, x in enumerate(m.Z)]
      Z = m.Z
      for i,z in enumerate(Z):

        if i == 0:
          orb_loc1.append(0)
          if Z[i] != 1:
            orb_loc2.append(3)
          else:
            orb_loc2.append(0)
          continue
        
        if Z[i-1] != 1:
            orb_loc1.append(orb_loc1[i-1]+4)
        elif Z[i-1] == 1:
            orb_loc1.append(orb_loc1[i-1]+1)
          
        if Z[i] != 1:
            orb_loc2.append(orb_loc2[i-1]+4)
        elif Z[i] == 1:    
            orb_loc2.append(orb_loc2[i-1]+1)
          
             
      # print('orb_loc1\n', orb_loc1)
      # print('orb_loc2\n', orb_loc2)
            
      # orb_loc1 = [0,4,5]
      G_1c2e = torch.zeros((m.norb*(m.norb+1)//2))
    
      for ii in range(molsize): # n_atoms?
      # print('ii', ii)
        if m.Z[ii] == 1:
          pass
        
        else:
          gsp_ii = gsp[ii]
          gpp_ii = gpp[ii]
          gp2_ii = gp2[ii]
          hsp_ii = hsp[ii]
          
          ia = orb_loc1[ii]
          # print('ia', ia)
          ib = orb_loc2[ii]
          
          iplus = ia+1
          ka = pascal2[ia]
          l = ka

          for j in range(iplus, ib+1):
          # print('j', j)
          # print('ia', ia)
          # print('ib', ib)
          # print('l', l)

            mm = l+ia+1
            l = l+j+1
            
            # print(type(mm)) 
            G_1c2e[mm] = G_1c2e[mm] + 0.5*eta_anti[mm] * (hsp_ii - gsp_ii)
            # print('(hsp - gsp)', (hsp - gsp))
            # print('F(M) FIRST', G_1c2e[mm])
           # print('G_1c2e[mm]', G_1c2e[mm])
          #  print('===================')
            
          iminus = ib-1
        
          for j in range(iplus, iminus+1):
            icc = j
            # print('icc', icc)
            for l in range(icc, ib):
              # print('l', l)
              mm = pascal1[l+1] + j+1
              # print('mm', mm)
              # print('(0.25*gpp_ii - 0.6*gp2_ii)', (0.25*gpp_ii - 0.6*gp2_ii) )
              G_1c2e[mm] = G_1c2e[mm]+ eta_anti[mm] * (0.25*gpp_ii - 1.25*0.6*gp2_ii) 
              # print('F(M) SECOND', G_1c2e[mm])
            #  print('G_1c2e[mm]', G_1c2e[mm])
            
      G_1c2e = G_1c2e*2 # antisym 1c2e part
      # print('G_1c2e\n', G_1c2e)
      

      # buils antisymmetric part as Vxi_packA, requires G_sym
      
   #   print('G_sym shape', G_sym.shape)
   #   print('G_sym\n', G_sym)
      
     # print('molsize', molsize)
     
      G_sym = G_sym[0] #! works for one mol only?
      G_sym_orig = G_sym.clone()
      # print('G_sym shape', G_sym.shape)
      # print('G_sym\n', G_sym)
      
      # l = 0
      # for i in range(0, m.norb): # TODO vectorize
      #     for j in range(0,i):
      #         l += 1

      #         G_sym[i,j] += G_1c2e[l-1]
      #         G_sym[j,i] -= G_1c2e[l-1]             
      #     l += 1 # skip diagonal
      # print('G sym after ADDING ANTSYM PART')
      # print('G_sym\n', G_sym)
      
      # pack from 1d to 2d ANTISYM 1c2e part of G
      G_anti_1c2e = torch.zeros(m.norb, m.norb)
      G_anti_1c2e[indices[1], indices[0]] = -G_1c2e
      G_anti_1c2e = G_anti_1c2e - G_anti_1c2e.T
      
      # print('G_anti_1c2e\n', G_anti_1c2e)
      # print(G_anti_1c2e.shape)
      
      # print('SUM')
      # print(G_sym_orig + G_anti_1c2e)
    
      # print('G_anti_1c shape', G_anti_1c.shape)
      # print('G_anti_1c\n', G_anti_1c)
      
    #  print('eta_ao_asym shape', eta_ao_asym.shape)
    #  print('eta_ao_asym\n', eta_ao_asym)
      
      # G_tmp = 2* G_anti_1c + G_sym
      G_sym = torch.unsqueeze(G_sym, 0) #  add dimension

      # build 2c-2e part of antisymmetric G
      # copied from FOCK
      
      F = torch.zeros((nmol*molsize**2,4,4), device=device) # 0 Fock matrix to fill
      P0 = unpack(eta_ao_asym, nHydro, nHeavy, (nHeavy+nHydro)*4) # 
      P0 = torch.unsqueeze(P0, 0) # add dimension
      
      P = P0.reshape((nmol,molsize,4,molsize,4)) \
          .transpose(2,3).reshape(nmol*molsize*molsize,4,4)
          
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
      
      # F0 is still symmetric, probably symmetrized above
      # here we make it antisymmetric back
      #F0 = 2 * F0 
      F0 = pack.pack(F0, m.nHeavy, m.nHydro)
      
      rows, cols = torch.tril_indices(F0.shape[1], F0.shape[2])
      F0[0][rows, cols] *= -1
      F0[0][torch.eye(F0.shape[1]).bool()] *= -1
      
      

      F0[0].diagonal().fill_(0) #! BE WARNED, THIS iS TAKEN FROM OLD NEXMD, PYSEQM produces non-zero diagonal
      F0 = F0*2
    #  print('G ANTISYM shape', F0.shape)
    #  print('G ANTISYM\n', F0*2)
      G_full = G_sym + G_anti_1c2e + F0 # summ of G_sym(sym 1c2e + 2c2e) + antisym 1c2e and 2c2e (F0)
      # print('G_full shape', G_full.shape)
      # print('G_full\n', G_full)
      
      return G_full

# %%
def ao2mo(N_cis, N_rpa, molecule, M_ao, C, full=False):
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
    m = molecule
    
    if full == True:
        M_mo = C.T @ M_ao @ C
        return M_mo
        
    else:
         # COPY of subroutine site2mo from Lioville
         
        G_ao = M_ao # TODO rename
        
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

# %%
def mo2ao(N_cis, M_mo, molecule, full=False):
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
    m = molecule
    
    # print('m C_mo', m.C_mo)
    if full == True:
        M_ao = C.T @ M_mo @ C #! does not currently work
        
        return M_ao
    else:
        
        eta = M_mo # TODO rename
        
        eta1 = eta[:N_cis]
        eta1 = eta1.view(-1, m.nvirt[0]) # 1d -> 2d
        # print(eta1.shape)
        # print('eta1', eta1)
        # print('==============')
        
        eta_mo = torch.zeros((m.norb, m.norb), device=device)

        dgemm1 = eta1 @ m.C_mo[0][:, m.nocc:m.norb].T # operations on |X| ?

        # print('dgemm1.shape', dgemm1.shape)
        # print(dgemm1)
        
        eta_mo[:m.nocc] = dgemm1
        # print('eta_mo', eta_mo.shape)
        # print(eta_mo)
        
        
        eta2 = eta[N_cis:]                            # operations on |Y| ?
        eta2 = eta2.view(-1, m.nvirt[0]) # 1d -> 2d
    
        dgemm2 = eta2.T @ m.C_mo[0][:, :m.nocc].T
        
        # print('dgemm2.shape', dgemm2.shape)
        # print(dgemm2)
        
        eta_mo[m.nocc:] = dgemm2
        # print('eta_mo', eta_mo.shape)
        # print(eta_mo)
        
        dgemm3 = m.C_mo[0] @ eta_mo
        eta_ao = dgemm3 
    
    
        return eta_ao

# %% [markdown]
# ### AUX routines

# %%
def orthogonalize_torch(U, eps=1e-15):
    """
    Orthogonalizes the matrix U (d x n) using Gram-Schmidt Orthogonalization.
    If the columns of U are linearly dependent with rank(U) = r, the last n-r columns 
    will be 0.
    
    Args:
        U (numpy.array): A d x n matrix with columns that need to be orthogonalized.
        eps (float): Threshold value below which numbers are regarded as 0 (default=1e-15).
    
    Returns:
        (numpy.array): A d x n orthogonal matrix. If the input matrix U's cols were
            not linearly independent, then the last n-r cols are zeros.
    
    Examples:
    ```python
    >>> import numpy as np
    >>> import gram_schmidt as gs
    >>> gs.orthogonalize(np.array([[10., 3.], [7., 8.]]))
    array([[ 0.81923192, -0.57346234],
       [ 0.57346234,  0.81923192]])
    >>> gs.orthogonalize(np.array([[10., 3., 4., 8.], [7., 8., 6., 1.]]))
    array([[ 0.81923192 -0.57346234  0.          0.        ]
       [ 0.57346234  0.81923192  0.          0.        ]])
    ```
    """
    
    n = len(U[0])
    # numpy can readily reference rows using indices, but referencing full rows is a little
    # dirty. So, work with transpose(U)
    V = U.T
    for i in range(n):
        prev_basis = V[0:i]     # orthonormal basis before V[i]
        coeff_vec = prev_basis @ V[i].T  # each entry is np.dot(V[j], V[i]) for all j < i
        # subtract projections of V[i] onto already determined basis V[0:i]
        V[i] -= (coeff_vec @ prev_basis).T
        if torch.norm(V[i]) < eps:
            V[i][V[i] < eps] = 0.   # set the small entries to 0
        else:
            V[i] /= torch.norm(V[i])
    return V.T

# %%
def gen_V(N_cis, N_rpa, n_V_start):
    
    # returns vexp1 - guess vector for L-xi routine
    logger.qm2(fmt_log(n_V_start, 'n_V_start', 'qm'))
    rrwork = torch.zeros(N_rpa * 4, device=device)
    i = 0
    for ip in range(mol.nocc):
        for ih in range(mol.nvirt):
            rrwork[i] = mol.e_mo[0][mol.nocc + ih] - mol.e_mo[0][ip]  # !Lancos vectors(i) ???
            i += 1                                                                      #  TODO: [0] should be replaced by m batch index
                                                                                    
    rrwork_sorted, indices = torch.sort(rrwork[:N_cis], descending=False, stable=True) # preserve order of degenerate
    logger.qm2(fmt_log(rrwork_sorted, 'rrwork_sorted', 'qm'))


    # vexp1 = torch.zeros((N_cis, N_cis), device=device)
    vexp1 = torch.zeros((N_cis, N_cis), device=device)
    
    row_idx = torch.arange(0, int(N_cis), device=device)
    col_idx = indices[:N_cis]

    vexp1[row_idx, col_idx] = 1.0 
    logger.qm2(fmt_log(vexp1, 'V  BEFORE SELECTING PART', 'qm'))
    logger.qm2(fmt_log(vexp1.shape, 'V shape', 'qm'))
   #! THIS IS NEW, TAKE only part 

    V = vexp1[:,  :n_V_start]
    logger.qm2(fmt_log(V, 'V = vexp1', 'qm'))
    logger.qm2(fmt_log(V.shape, 'V shape', 'qm'))

    return V

# TODO: check whether L_xi should be regenerated during expansion each time or just part of it

# %% [markdown]
# ### DAVIDSON routines

# %%
logger.setLevel(logging.DEBUG)  # custom logging level; lower than DEBUG
                               # printed above QM (QM, DEBUG, INFO, etc)

# %%
# for i in range(4):
    
#     logger.debug('i = %d', i)

# %%
def davidson(mol, N_exc, keep_n, n_V_max,  max_iter, tol):
    """
    Davidson algorithm for solving eigenvalue problem of large sparse diagonally dominant matrices
    Hamiltonian is not generated or stored explicitly, only matrix-vector products are used on-the fly:
    guess space V should be orthogonalized at each iteration
    M (projection of smaller size) is V.T @ H @ V 
    #! RPA (TDHF) is not implemented yet, non-Hermitian (non-symmetric), requires also left eigenvectors 
    note that notation differes between implementations: V.T x A x V is bAb
    # TODO: 1) check if convergence of e_vals is needed
    # TODO: 2) vectorize and optimize orthogonalization
    # TODO: 3) check if some vectors should be dropped 
    # TODO: 4) eliminate loops 
    # TODO: 5) check if whole M should be regenerated, or only sub-blocks corresponding to new guess vectors

    Args:
        mol (PYSEQM object): object to hold all qm data from PYSEQM
        N_exc (int)        : number of excited states to calculate
        keep_n (int)       : number of e_vals, e_vecs to keep at each iteration
        n_V_max (int)      : maximum size of Krylov subspace, 
                             projected matrix will be no more than M(n_V_max x n_V_max)
        max_iter (int)     : maximum number of iterations in Davidson
        tol (float)        : treshold for residual
        
    Returns:
        tuple of tensors: eigenvalues (excitation energies in default units, eV) and eigenvectors 
    """    
    
    n_V_start = N_exc * 2 # dimension of Krylov subspace, analogue of nd1  
    N_cis = mol.nocc * mol.nvirt
    N_rpa = 2 * N_cis
    
    term = False  # terminate algorithm
    iter = 0
    L_xi = torch.zeros((N_rpa, n_V_start), device=device)

    V = gen_V(N_cis, N_rpa, n_V_start) # generate initial guess, V here #! should be renamed
    diag = None # create diagonal of M only once
    
    while iter < max_iter and not term: # Davidson loop
        
        if iter > 0: # skip first step, as initial V is orthogonal
            V = orthogonalize_torch(V)
            
        print('=================================', flush=True)
        print(colored(f' ITERATION : {iter} ', 'red', 'on_white', attrs=['bold']), flush=True)
        print('SUBSPACE SIZE V: ', V.shape, flush=True)
        print('=================================')
       
        # ---------- form A x b product --------------------
        L_xi = torch.zeros((N_rpa, V.shape[1] ), device=device) #! NOT iter here
        logger.qm1(fmt_log(V, 'V BEFORE L_xi after ORTO', 'qm'))
        for i in range(V.shape[1]): 
            logger.qm3('Lxi iterations=%s', i)
            L_xi[:,i] = form_L_xi(V[:,i], mol, N_cis, N_rpa)
            logger.qm3(fmt_log(L_xi[:,i], 'L_xi[:,i]', 'qm'))
        
        L_xi[N_cis:, :] = L_xi[:N_cis] #! TODO: make sure that this A+B, A-B, not just copy for RPA
    
        right_V = L_xi[N_cis:] # (A)b 
        
        logger.qm1(fmt_log(right_V.shape, 'right_V shape', 'matrix'))
        logger.qm1(fmt_log(right_V, 'right_V', 'matrix'))       
        # ---------- form b.T x Ab product --------------------
        
        M =  V.T @ right_V
        
        # logger.debug(fmt_log(M.shape, 'M shape', 'qm'))
        # logger.debug(fmt_log(M, 'M', 'qm'))
        if iter == 0:
            diag = torch.diag(M) # create diagonal only once
            
        iter += 1
        
        logger.qm1(fmt_log(diag, 'diag', 'qm'))
    
        # ---------- diagonalize projection M --------------------
        r_eval, r_evec = torch.linalg.eig(M) # find eigenvalues and eigenvectors
       
        r_eval = r_eval.real
        r_evec = r_evec.real
        r_eval, r_idx = torch.sort(r_eval, descending=False) # sort eigenvalues in ascending order
        logger.debug(fmt_log(r_eval, 'RIGHT EVALS', 'evals'))
        r_evec = r_evec[:, r_idx] # sort eigenvectors accordingly
    
        e_val_n = r_eval[:keep_n] # keep only the lowest keep_n eigenvalues; full are still stored as e_val
        e_vec_n = r_evec[:, :keep_n]
        resids = torch.zeros(V.shape[0], len(e_val_n)) # account for left and right evecs

        # ---------- calculate residual vectors --------------------
        for j in range(len(e_val_n)): # calc residuals 
            resids[:,j] = right_V @ e_vec_n[:,j] - e_val_n[j] * (V @ e_vec_n[:,j])
            
       # logger.debug(fmt_log(resids, 'resids', 'matrix'))     
        resids_norms_r = torch.tensor([resids[:,x].norm() for x in range(resids.shape[1])])

        # ---------- expand guess space V buy not-converged resids --------------------
        # !!! PROBABLY HIGHLY INEFFICIENT !!! 
        if torch.any(resids_norms_r > tol):
            mask_r = resids_norms_r >= tol
            large_res_r = resids[:,mask_r] # residuals larger than tol
           # logger.debug(fmt_log(large_res_r, 'LARGE RESIDUALS', 'vector'))           
            large_res_r.to(device)
            cor_e_val_r = e_val_n[mask_r] # corresponding eigenvalues !!! check if matches
            
            # ------keep adding new resids --------------------
            if V.shape[1] <= n_V_max:     

                    for j in range(large_res_r.shape[1]):
                        if V.shape[1] <= n_V_max:
                            s = large_res_r[:,j] # conditioned residuals > tol

                            if s.norm() >= tol:
                                logger.debug(fmt_log((s.norm().item()), 'NORM of RESIDUAL', 'warn'))
                                denom = (diag[j] - cor_e_val_r[j])
                                denom.to(device) 
                                s = s/denom # conditioned residuals
                                s.to(device)
                                # logger.debug(fmt_log(s.norm(), 'NORM OF NEW RESIDUAAL', 'vector'))
                                V = torch.column_stack((V, s/s.norm()))
                            else:
                                pass
            # ------ collapse (restart) if space V is too large; mix eigenvectors with V------------
            else:
                logger.debug(fmt_log(None, '!!!! MAX subspace reached !!!!', 'warn'))
                #logger.debug(fmt_log(V, 'V before collapse', 'qm'))

                V =  V @ r_evec[:, :n_V_start]
                logger.debug(fmt_log(V.shape, 'V shape after restart', 'qm'))
                #logger.debug(fmt_log(V, 'V AFTER collapse', 'qm'))

                continue

        else:
            term = True
            print('============================', flush=True)
            print('all residuals are below tolerance')
            print('DAVIDSON ALGORITHM CONVERGED', flush=True)
            print('============================', flush=True)

            return r_eval, r_evec

    # runs after big loop if did not converge
    print('============================', flush=True)
    print('!!! DAVIDSON ALGORITHM DID NOT CONVERGE !!!', flush=True)
    print('============================', flush=True)
    
    return r_eval, r_evec

# %%
# mol = run_seqm_1mol('c6h6.xyz')
# eval, _ = davidson(mol = mol, 
#                    N_exc = 8,
#                    keep_n = 4,
#                    n_V_max = 50, 
#                    max_iter = 50, 
#                    tol = 1e-6)

# logger.debug(fmt_log(eval, 'FINAL eval ', 'evals'))



mol = run_seqm_1mol('coronene.xyz')
eval, _ = davidson(mol = mol, 
                   N_exc = 10,
                   keep_n = 6,
                   n_V_max = 70, 
                   max_iter = 100, 
                   tol = 1e-6)

logger.debug(fmt_log(eval, 'FINAL eval ', 'evals'))

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to parse parameters for the given functions.")

    parser.add_argument("--xyz", type=str, help="Input XYZ file (e.g., 'coronene.xyz')")
    parser.add_argument("--nexc", type=int, default=10, help="Number of excitations (default: 10)")
    parser.add_argument("--keepn", type=int, default=6, help="Number of states to keep (default: 6)")
    parser.add_argument("--vmax", type=int, default=70, help="Maximum number of vectors (default: 70)")
    parser.add_argument("--maxiter", type=int, default=100, help="Maximum iterations (default: 100)")
    parser.add_argument("--tol", type=float, default=6, help="Tolerance (default: 1e-6)")

    args = parser.parse_args()

    mol = run_seqm_1mol(args.mol_file)
    eval, _ = davidson(
        mol=mol,
        N_exc=args.nexc,
        keep_n=args.keepn,
        n_V_max=args.vmax,
        max_iter=args.maxiter,
        tol=10 ** (-args.tol)
    )

# %%


# %%


# %%


# %%


# %%




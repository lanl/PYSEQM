import torch
from .seqm_functions.scf_loop_d import scf_loop
from .seqm_functions.energy import *
from .seqm_functions.parameters import params, PWCCT
from torch.autograd import grad
from .seqm_functions.constants import ev
import os
import time

"""
Semi-Emperical Quantum Mechanics: AM1/MNDO/PM3/PM6
"""

parameterlist={'AM1':['U_ss', 'U_pp', 'zeta_s', 'zeta_p','beta_s', 'beta_p',
                      'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp',
                      'alpha',
                      'Gaussian1_K', 'Gaussian2_K', 'Gaussian3_K','Gaussian4_K',
                      'Gaussian1_L', 'Gaussian2_L', 'Gaussian3_L','Gaussian4_L',
                      'Gaussian1_M', 'Gaussian2_M', 'Gaussian3_M','Gaussian4_M'
                     ],
                'MNDO':['U_ss', 'U_pp', 'zeta_s', 'zeta_p','beta_s', 'beta_p',
                        'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'alpha'],
                'PM3':['U_ss', 'U_pp', 'zeta_s', 'zeta_p','beta_s', 'beta_p',
                       'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp',
                       'alpha',
                       'Gaussian1_K', 'Gaussian2_K',
                       'Gaussian1_L', 'Gaussian2_L',
                       'Gaussian1_M', 'Gaussian2_M'
                      ],

                'PM6':['U_ss', 'U_pp', 'U_dd', 'zeta_s', 'zeta_p', 'zeta_d',  'beta_s', 'beta_p',
                       'beta_d', 's_orb_exp_tail', 'p_orb_exp_tail', 'd_orb_exp_tail',
                       'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'F0SD', 'G2SD','rho_core',
                       'alpha', 'EISOL',
                       'Gaussian1_K', 'Gaussian2_K', 'Gaussian3_K','Gaussian4_K',
                       'Gaussian1_L', 'Gaussian2_L', 'Gaussian3_L','Gaussian4_L',
                       'Gaussian1_M', 'Gaussian2_M', 'Gaussian3_M','Gaussian4_M'
                      ]}


class Parser(torch.nn.Module):
    """
    parsing inputs from coordinates and types
    """
    def __init__(self, seqm_parameters):
        """
        Constructor
        """
        super().__init__()
        self.outercutoff = seqm_parameters['pair_outer_cutoff']
        self.elements = seqm_parameters['elements']
        if 'UHF' in seqm_parameters:
            self.uhf = seqm_parameters['UHF']
        else:
            self.uhf = False

    def forward(self, molecule, themethod, return_mask_l=False, *args, **kwargs):
        """
        constants : instance of Class Constants
        species : atom types for atom in each molecules,
                  shape (nmol, molsize),  dtype: torch.int64
        coordinates : atom position, shape (nmol, molsize, 3)
        charges: total charge for each molecule, shape (nmol,), 0 if None
        """
        device = molecule.coordinates.device
        dtype = molecule.coordinates.dtype
        nmol, molsize = molecule.species.shape
        nonblank = molecule.species>0
        n_real_atoms = torch.sum(nonblank)

        
        
        atom_index = torch.arange(nmol*molsize, device=device,dtype=torch.int64)
        real_atoms = atom_index[nonblank.reshape(-1)>0]
        inv_real_atoms = torch.zeros((nmol*molsize,), device=device,dtype=torch.int64)
        inv_real_atoms[real_atoms] = torch.arange(n_real_atoms, device=device,dtype=torch.int64)

        
        Z = molecule.species.reshape(-1)[real_atoms]
        
        if(themethod == 'PM6'):
            nHeavy = torch.sum(((molecule.species>1) & ((molecule.species <= 12) | ((molecule.species >= 18) & (molecule.species <=20)) | ((molecule.species >= 30) & (molecule.species <= 32)) | ((molecule.species >= 36) & (molecule.species <= 38)) | ((molecule.species >= 48) & (molecule.species <= 50)) | ((molecule.species >= 54) & (molecule.species <= 56)) | ((molecule.species >= 80) & (molecule.species <= 83)))),dim=1)
        else:
            nHeavy = torch.sum(molecule.species>1,dim=1)

        nSuperHeavy = torch.sum((((molecule.species > 12) & (molecule.species <18)) | ((molecule.species > 20) & (molecule.species <30)) | ((molecule.species > 32) & (molecule.species <36)) | ((molecule.species > 38) & (molecule.species <48)) | ((molecule.species > 50) & (molecule.species <54)) | ((molecule.species > 70) & (molecule.species <80)) | (molecule.species ==57)),dim=1)
        
        
        nHydro = torch.sum(molecule.species==1,dim=1)
        tore=molecule.const.tore
        n_charge = torch.sum(tore[molecule.species],dim=1).reshape(-1).type(torch.int64)
        if torch.is_tensor(molecule.tot_charge):
            n_charge -= molecule.tot_charge.reshape(-1).type(torch.int64)
        
        if self.uhf:
            nocc_alpha = n_charge/2 + (molecule.mult-1)/2
            nocc_beta = n_charge/2 - (molecule.mult-1)/2
            if (nocc_alpha%1 != 0).any() or (nocc_beta%1 != 0).any():
                raise ValueError("Invalid charge/multiplicity combination!")
            nocc = torch.stack((nocc_alpha,nocc_beta), dim=1)
            nocc = nocc.type(torch.int64)
        else:
            nocc = n_charge//2
            if ((n_charge%2)==1).any():
                raise ValueError("Only closed shell system (with even number of electrons) are supported")
        

        
        t1 = (torch.arange(molsize,dtype=torch.int64,device=device)*(molsize+1)).reshape((1,-1))
        t2 = (torch.arange(nmol,dtype=torch.int64,device=device)*molsize**2).reshape((-1,1))
        maskd = (t1+t2).reshape(-1)[real_atoms]
        atom_molid = torch.arange(nmol, device=device,dtype=torch.int64).unsqueeze(1).expand(-1,molsize).reshape(-1)[nonblank.reshape(-1)>0]

        nonblank_pairs = (nonblank.unsqueeze(1)*nonblank.unsqueeze(2)).reshape(-1)
        pair_first = atom_index.reshape(nmol, molsize) \
                               .unsqueeze(2) \
                               .expand(nmol,molsize,molsize) \
                               .reshape(-1)
        #
        pair_second = atom_index.reshape(nmol, molsize) \
                                .unsqueeze(1) \
                                .expand(nmol,molsize,molsize) \
                                .reshape(-1)
        #
        paircoord_raw = (molecule.coordinates.unsqueeze(1)-molecule.coordinates.unsqueeze(2)).reshape(-1,3)
        pairdist_raw = torch.norm(paircoord_raw,dim=1)
        close_pairs = pairdist_raw < self.outercutoff

        pairs = (pair_first<pair_second) * nonblank_pairs * close_pairs

        paircoord = paircoord_raw[pairs]
        pairdist = pairdist_raw[pairs]
        rij = pairdist*molecule.const.length_conversion_factor
        #print( 'bohrs_to_a, rij', molecule.const.length_conversion_factor, pairdist)

        idxi = inv_real_atoms[pair_first[pairs]]
        idxj = inv_real_atoms[pair_second[pairs]]
        ni = Z[idxi]
        nj = Z[idxj]
        xij = paircoord/pairdist.unsqueeze(1)
        mask = real_atoms[idxi]*molsize+real_atoms[idxj]%molsize
        mask_l = real_atoms[idxj]*molsize+real_atoms[idxi]%molsize
        pair_molid = atom_molid[idxi] # doesn't matter atom_molid[idxj]
        # nmol, molsize : scalar
        # nHeavy, nHydro, nocc : (nmol,)
        # Z, maskd, atom_molid: (natoms, )
        # mask, pair_molid, ni, nj, idxi, idxj, xij, rij ; (npairs, )
        if not return_mask_l:
            return nmol, molsize, \
                nSuperHeavy, nHeavy, nHydro, nocc, \
                Z, maskd, atom_molid, \
                mask, pair_molid, ni, nj, idxi, idxj, xij, rij
        else:
            return nmol, molsize, \
                nSuperHeavy, nHeavy, nHydro, nocc, \
                Z, maskd, atom_molid, \
                mask, mask_l, pair_molid, ni, nj, idxi, idxj, xij, rij

class Parser_For_Ovr(torch.nn.Module):
    """
    parsing inputs from coordinates and types
    """
    def __init__(self, seqm_parameters):
        """
        Constructor
        """
        super().__init__()
        self.outercutoff = seqm_parameters['pair_outer_cutoff']
        self.elements = seqm_parameters['elements']

    def forward(self, constansts, species, coordinates, *args, **kwargs):
        """
        constants : instance of Class Constants
        species : atom types for atom in each molecules,
                  shape (nmol, molsize),  dtype: torch.int64
        coordinates : atom position, shape (nmol, molsize, 3)
        charges: total charge for each molecule, shape (nmol,), 0 if None
        """
        device = coordinates.device
        dtype = coordinates.dtype

        nmol, molsize = species.shape
        nonblank = species>0
        n_real_atoms = torch.sum(nonblank)

        atom_index = torch.arange(nmol*molsize, device=device,dtype=torch.int64)
        real_atoms = atom_index[nonblank.reshape(-1)>0]

        inv_real_atoms = torch.zeros((nmol*molsize,), device=device,dtype=torch.int64)
        inv_real_atoms[real_atoms] = torch.arange(n_real_atoms, device=device,dtype=torch.int64)

        Z = species.reshape(-1)[real_atoms]
        nHeavy = torch.sum(species>1,dim=1)
        nHydro = torch.sum(species==1,dim=1)
        tore=constansts.tore
        n_charge = torch.sum(tore[species],dim=1).reshape(-1).type(torch.int64)
        if 'charges' in kwargs and torch.is_tensor(kwargs['charges']):
            n_charge += kwargs['charges'].reshape(-1).type(torch.int64)
        nocc = n_charge//2

        if ((n_charge%2)==1).any():
            raise ValueError("Only closed shell system (with even number of electrons) are supported")
        t1 = (torch.arange(molsize,dtype=torch.int64,device=device)*(molsize+1)).reshape((1,-1))
        t2 = (torch.arange(nmol,dtype=torch.int64,device=device)*molsize**2).reshape((-1,1))
        maskd = (t1+t2).reshape(-1)[real_atoms]
        atom_molid = torch.arange(nmol, device=device,dtype=torch.int64).unsqueeze(1).expand(-1,molsize).reshape(-1)[nonblank.reshape(-1)>0]

        nonblank_pairs = (nonblank.unsqueeze(1)*nonblank.unsqueeze(2)).reshape(-1)
        pair_first = atom_index.reshape(nmol, molsize) \
                               .unsqueeze(2) \
                               .expand(nmol,molsize,molsize) \
                               .reshape(-1)
        #
        pair_second = atom_index.reshape(nmol, molsize) \
                                .unsqueeze(1) \
                                .expand(nmol,molsize,molsize) \
                                .reshape(-1)
        #
        paircoord_raw = (coordinates.unsqueeze(1)-coordinates.unsqueeze(2)).reshape(-1,3)
        pairdist_raw = torch.norm(paircoord_raw,dim=1)
        close_pairs = pairdist_raw < self.outercutoff

        pairs = (pair_first<=pair_second) * nonblank_pairs * close_pairs

        paircoord = paircoord_raw[pairs]
        pairdist = pairdist_raw[pairs]
        rij = pairdist*constansts.length_conversion_factor
        

        idxi = inv_real_atoms[pair_first[pairs]]
        idxj = inv_real_atoms[pair_second[pairs]]
        ni = Z[idxi]
        nj = Z[idxj]
        xij = paircoord/pairdist.unsqueeze(1)
        mask = real_atoms[idxi]*molsize+real_atoms[idxj]%molsize
        pair_molid = atom_molid[idxi] # doesn't matter atom_molid[idxj]
        # nmol, molsize : scalar
        # nHeavy, nHydro, nocc : (nmol,)
        # Z, maskd, atom_molid: (natoms, )
        # mask, pair_molid, ni, nj, idxi, idxj, xij, rij ; (npairs, )
        return nmol, molsize, \
               nHeavy, nHydro, nocc, \
               Z, maskd, atom_molid, \
               mask, pair_molid, ni, nj, idxi, idxj, xij, rij

class Pack_Parameters(torch.nn.Module):
    """
    pack the parameters, combine the learned parameters and the ones from mopac
    """
    def __init__(self, seqm_parameters):
        """
        elements : elements will be used
        device : cpu, cuda etc
        method : seqm method
        learned : list for parameters will be provided and require grad, e.g. learned = ['U_ss']
        filedir : mopac parameter files directory
        """
        super().__init__()
        self.elements = seqm_parameters['elements']
        self.learned_list = seqm_parameters['learned']
        self.method = seqm_parameters['method']
        self.filedir = seqm_parameters['parameter_file_dir'] \
            if 'parameter_file_dir' in seqm_parameters \
            else os.path.abspath(os.path.dirname(__file__))+'/params/'
        self.parameters = parameterlist[self.method]
        self.required_list = []
        for i in self.parameters:
            if i not in self.learned_list:
                self.required_list.append(i)
        self.nrp = len(self.required_list)
        self.p = params(method=self.method, elements=self.elements,root_dir=self.filedir,
                 parameters=self.required_list)
        
        self.alpha,self.chi = PWCCT(method=self.method, elements=self.elements,root_dir=self.filedir,
                 parameters=self.required_list)

    def forward(self, Z, learned_params=dict()):
        """
        combine the learned_parames with other required parameters
        """
        for i in range(self.nrp):
            learned_params[self.required_list[i]] = self.p[Z,i] #.contiguous()
        return learned_params, self.alpha, self.chi

class Hamiltonian(torch.nn.Module):
    """
    build the Hamiltonian
    """
    def __init__(self, seqm_parameters):
        """
        Constructor
        """
        super().__init__()
        #put eps and scf_backward_eps as torch.nn.Parameter such that it is saved with model and can
        #be used to restart jobs
        self.eps = torch.nn.Parameter(torch.as_tensor(seqm_parameters['scf_eps']), requires_grad=False)
        self.sp2 = seqm_parameters['sp2']
        self.scf_converger = seqm_parameters['scf_converger']
        # whether return eigenvalues, eigenvectors, otherwise they are None
        if 'eig' in seqm_parameters:
            self.eig = seqm_parameters['eig']
        else:
            self.eig = False
        if 'scf_backward' in seqm_parameters:
            self.scf_backward = seqm_parameters['scf_backward']
        else:
            self.scf_backward = 0
        if 'scf_backward_eps' not in seqm_parameters:
            seqm_parameters['scf_backward_eps'] = 1.0e-2
        self.scf_backward_eps = torch.nn.Parameter(torch.as_tensor(seqm_parameters['scf_backward_eps']), requires_grad=False)
        # 0: ignore gradient on density matrix from Hellmann Feymann Theorem,
        # 1: use recursive formula go back through scf loop

    def forward(self, themethod, const, molsize, nSuperHeavy, nHeavy, nHydro, nocc, Z, maskd, mask, atom_molid, pair_molid, idxi, idxj, ni,nj,xij,rij, parameters, alpha, chi, P0=None):
        """
        SCF loop
        const : Constants instance
        molsize : maximal number of atoms in each molecule
        nHeavy : number of heavy atoms in each molecule, shape (nmol,) nmol: number of molecules in this batch
        nHydro : number of hydrogen in each molecule, shape (nmol,)
        nocc : number of occupied molecular orbitals, shape (nmol,)
        maskd : diagonal block postions, shape (n_atoms,)
        mask: off diagonal block positions, shape (n_pairs,)
        idxi/idxj : atom indexes for first/second atom in each pair, shape (n_pairs,)
        ni/nj : atom number for first/second atom in each pair, shape (n_pairs,)
        xij : unit vector for each pair, from i to j, (Rj-Ri)/|Rj-Ri|
        rij : distance for each pair, in atomic unit, shape (n_pairs,)
        Z: atom number, shape (n_atoms,)
        zetas/zetap : Zeta for s/p orbital for each atom, shape (n_atoms, )
        uss, upp, gss, gsp, gpp, gp2, hsp: parameters for AM1/PM3/MNDO, shape (n_atoms,)
        #
        return F, e, P, Hcore
        F : fock matrix, i.e. the Hamiltonian for the system, shape (nmol, molsize*4, molsize*4)
        e : orbital energies, shape (nmol, molsize*4), 0 padding is used
        P : Density matrix for closed shell system, shape (nmol, molsize*4, molsize*4)
        Hcore : Hcore matrix, same shape as F
        w : two electron two center integrals
        v : eigenvectors of F
        """
        if(themethod == 'PM6'):
            beta = torch.cat((parameters['beta_s'].unsqueeze(1), parameters['beta_p'].unsqueeze(1), parameters['beta_d'].unsqueeze(1)),dim=1)
            udd=parameters['U_dd']
        else:
            beta = torch.cat((parameters['beta_s'].unsqueeze(1), parameters['beta_p'].unsqueeze(1)),dim=1)
        if "Kbeta" in parameters:
            Kbeta = parameters["Kbeta"]
        else:
            Kbeta = None

        if(themethod == 'PM6'):
            F, e, P, Hcore, w, charge,rho0xi,rho0xj, notconverged = scf_loop(const=const,
                                  molsize=molsize,
                                  nSuperHeavy = nSuperHeavy,
                                  nHeavy=nHeavy,
                                  nHydro=nHydro,
                                  nOccMO=nocc,
                                  maskd=maskd,
                                  mask=mask,
                                  atom_molid=atom_molid,
                                  pair_molid=pair_molid,
                                  idxi=idxi,
                                  idxj=idxj,
                                  ni=ni,
                                  nj=nj,
                                  xij=xij,
                                  rij=rij,
                                  Z=Z,
                                  zetas=parameters['zeta_s'],
                                  zetap=parameters['zeta_p'],
                                  zetad=parameters['zeta_d'],
                                  zs=parameters['s_orb_exp_tail'],
                                  zp=parameters['p_orb_exp_tail'],
                                  zd=parameters['d_orb_exp_tail'],
                                  uss=parameters['U_ss'],
                                  upp=parameters['U_pp'],
                                  udd=parameters['U_dd'],
                                  gss=parameters['g_ss'],
                                  gsp=parameters['g_sp'],
                                  gpp=parameters['g_pp'],
                                  gp2=parameters['g_p2'],
                                  hsp=parameters['h_sp'],
                                  F0SD = parameters['F0SD'], 
                                  G2SD = parameters['G2SD'],
                                  rho_core = parameters['rho_core'],
                                  alpha = alpha,
                                  chi = chi,
                                  themethod = themethod,
                                  beta=beta,
                                  Kbeta=Kbeta,
                                  eps = self.eps,
                                  P=P0,
                                  sp2=self.sp2,
                                  scf_converger=self.scf_converger,
                                  eig=self.eig,
                                  scf_backward=self.scf_backward,
                                  scf_backward_eps=self.scf_backward_eps)

        else:
            F, e, P, Hcore, w, charge, rho0xi,rho0xj, notconverged = scf_loop(const=const,
                              molsize=molsize,
                              nSuperHeavy = nSuperHeavy,
                              nHeavy=nHeavy,
                              nHydro=nHydro,
                              nOccMO=nocc,
                              maskd=maskd,
                              mask=mask,
                              atom_molid=atom_molid,
                              pair_molid=pair_molid,
                              idxi=idxi,
                              idxj=idxj,
                              ni=ni,
                              nj=nj,
                              xij=xij,
                              rij=rij,
                              Z=Z,
                              zetas=parameters['zeta_s'],
                              zetap=parameters['zeta_p'],
                              zetad=torch.zeros_like(parameters['zeta_s']),
                              zs=torch.zeros_like(parameters['zeta_s']),
                              zp=torch.zeros_like(parameters['zeta_s']),
                              zd=torch.zeros_like(parameters['zeta_s']),
                              uss=parameters['U_ss'],
                              upp=parameters['U_pp'],
                              udd=torch.zeros_like(parameters['U_ss']),
                              gss=parameters['g_ss'],
                              gsp=parameters['g_sp'],
                              gpp=parameters['g_pp'],
                              gp2=parameters['g_p2'],
                              hsp=parameters['h_sp'],
                              F0SD = torch.zeros_like(parameters['U_ss']),
                              G2SD = torch.zeros_like(parameters['U_ss']),
                              rho_core = torch.zeros_like(parameters['U_ss']),
                              alpha = alpha,
                              chi = chi,
                              themethod = themethod,
                              beta=beta,
                              Kbeta=Kbeta,
                              eps = self.eps,
                              P=P0,
                              sp2=self.sp2,
                              scf_converger=self.scf_converger,
                              eig=self.eig,
                              scf_backward=self.scf_backward,
                              scf_backward_eps=self.scf_backward_eps)
        #
        return F, e, P, Hcore, w, charge,rho0xi,rho0xj, notconverged

class Energy(torch.nn.Module):
    def __init__(self, seqm_parameters):
        """
        Constructor
        """
        super().__init__()
        self.seqm_parameters =seqm_parameters
        self.method = seqm_parameters['method']
        self.parser = Parser(seqm_parameters)
        self.packpar = Pack_Parameters(seqm_parameters)
        self.hamiltonian = Hamiltonian(seqm_parameters)
        self.Hf_flag = True
        if "Hf_flag" in seqm_parameters:
            self.Hf_flag = seqm_parameters["Hf_flag"] # Hf_flag: true return Hf, false return Etot-Eiso
        
        if 'UHF' in seqm_parameters:
            self.uhf = seqm_parameters['UHF']
        else:
            self.uhf = False

    def forward(self, molecule, learned_parameters=dict(), all_terms=False, P0=None, *args, **kwargs):
        """
        get the energy terms
        """
        nmol, molsize, \
        nSuperHeavy, nHeavy, nHydro, nocc, \
        Z, maskd, atom_molid, \
        mask, pair_molid, ni, nj, idxi, idxj, xij, rij = self.parser(molecule, self.method, *args, **kwargs)
                
        if callable(learned_parameters):
            adict = learned_parameters(molecule.species, molecule.coordinates)
            parameters, alp, chi = self.packpar(Z, learned_params = adict)    
        else:
            parameters, alp, chi = self.packpar(Z, learned_params = learned_parameters)
        F, e, P, Hcore, w, charge, rho0xi,rho0xj, notconverged =  self.hamiltonian(self.method, molecule.const, molsize, \
                                                 nSuperHeavy, nHeavy, nHydro, nocc, \
                                                 Z, maskd, \
                                                 mask, atom_molid, pair_molid, idxi, idxj, ni,nj,xij,rij, \
                                                 parameters, alp, chi, P0=P0)

        if torch.is_tensor(e):
            if self.uhf == False:
                e_gap = e.gather(1, nocc.unsqueeze(0).T) - e.gather(1, nocc.unsqueeze(0).T-1)
            else:
                if 0 in nocc:
                    print('Zero occupied alpha or beta orbitals found (e.g. triplet H2). HOMO-LUMO gaps are not available.')
                    e_gap = torch.tensor([])
                else:
                    e_gap_a = e[:,0].gather(1, nocc[:,0].unsqueeze(0).T) - e[:,0].gather(1, nocc[:,0].unsqueeze(0).T-1)
                    e_gap_b = e[:,1].gather(1, nocc[:,1].unsqueeze(0).T) - e[:,1].gather(1, nocc[:,1].unsqueeze(0).T-1)
                    e_gap = torch.stack((e_gap_a, e_gap_b), dim=1)
        else:
            e_gap = None

            
        #nuclear energy
        alpha = parameters['alpha']
        if self.method=='MNDO':
            parnuc = (alpha,)
        elif self.method=='AM1' or self.method=='PM6':
            K = torch.stack((parameters['Gaussian1_K'],
                             parameters['Gaussian2_K'],
                             parameters['Gaussian3_K'],
                             parameters['Gaussian4_K']),dim=1)
            #
            L = torch.stack((parameters['Gaussian1_L'],
                             parameters['Gaussian2_L'],
                             parameters['Gaussian3_L'],
                             parameters['Gaussian4_L']),dim=1)
            #
            M = torch.stack((parameters['Gaussian1_M'],
                             parameters['Gaussian2_M'],
                             parameters['Gaussian3_M'],
                             parameters['Gaussian4_M']),dim=1)
            #
            parnuc = (alpha, K, L, M)
        elif self.method=='PM3':
            K = torch.stack((parameters['Gaussian1_K'],
                             parameters['Gaussian2_K']),dim=1)
            #
            L = torch.stack((parameters['Gaussian1_L'],
                             parameters['Gaussian2_L']),dim=1)
            #
            M = torch.stack((parameters['Gaussian1_M'],
                             parameters['Gaussian2_M']),dim=1)
            #
            parnuc = (alpha, K, L, M)

        if 'g_ss_nuc' in parameters:
            g = parameters['g_ss_nuc']
            rho0a = 0.5*ev/g[idxi]
            rho0b = 0.5*ev/g[idxj]
            gam = ev/torch.sqrt(rij**2 + (rho0a+rho0b)**2)
        else:
            gam = w[...,0,0]

        EnucAB = pair_nuclear_energy(Z, molecule.const, nmol, ni, nj, idxi, idxj, rij, rho0xi,rho0xj,alp, chi, gam=gam, method=self.method, parameters=parnuc)
        Eelec = elec_energy(P, F, Hcore)
        ### perhaps, need to update elec_energy_isolated_atom for d-elements
        if all_terms:
            Etot, Enuc = total_energy(nmol, pair_molid,EnucAB, Eelec)
            Eiso = elec_energy_isolated_atom(molecule.const, Z,
                                         uss=parameters['U_ss'],
                                         upp=parameters['U_pp'],
                                         gss=parameters['g_ss'],
                                         gpp=parameters['g_pp'],
                                         gsp=parameters['g_sp'],
                                         gp2=parameters['g_p2'],
                                         hsp=parameters['h_sp'])
            Hf, Eiso_sum = heat_formation(molecule.const, nmol,atom_molid, Z, Etot, Eiso, flag = self.Hf_flag)
            return Hf, Etot, Eelec, Enuc, Eiso_sum, EnucAB, e_gap, e, P, charge, notconverged
        else:
            #for computing force, Eelec.sum()+EnucAB.sum() and backward is enough
            #index_add is used in total_energy and heat_formation function
            # P can be used as the initialization
            return Eelec, EnucAB, P, notconverged

class Force(torch.nn.Module):
    """
    get force
    """
    def __init__(self, seqm_parameters):
        super().__init__()
        self.energy = Energy(seqm_parameters)
        if "2nd_grad" in seqm_parameters:
            self.create_graph = seqm_parameters["2nd_grad"]
        else:
            self.create_graph = False
        self.seqm_parameters = seqm_parameters
        if 'UHF' in seqm_parameters:
            self.uhf = seqm_parameters['UHF']
        else:
            self.uhf = False


    def forward(self, molecule, learned_parameters=dict(), P0=None, *args, **kwargs):
        t0=time.time()
        molecule.coordinates.requires_grad_(True)
        Hf, Etot, Eelec, Enuc, Eiso, EnucAB, e_gap, e, D, charge, notconverged = \
            self.energy(molecule, learned_parameters=learned_parameters, all_terms=True, P0=P0, *args, **kwargs)
        
        if torch.is_tensor(e):
            e = e.detach()
            e_gap = e_gap.detach()
        #L = Etot.sum()
        L = Hf.sum()
        if molecule.const.do_timing:
            t0 = time.time()

        #gv = [coordinates]
        #gradients  = grad(L, gv,create_graph=self.create_graph)
        L.backward(create_graph=self.create_graph)
        if molecule.const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            molecule.const.timing["Force"].append(t1-t0)
        #force = -gradients[0] 
        if self.create_graph:
            force = -molecule.coordinates.grad.clone()
            with torch.no_grad():
                molecule.coordinates.grad.zero_()
        else:
            force = -molecule.coordinates.grad.detach()
            molecule.coordinates.grad.zero_()

        #print('full force module ', time.time()-t0)
        return force.detach(), D.detach(), Hf.detach(), Etot.detach(), Eelec.detach(), Enuc.detach(), Eiso.detach(), e, e_gap, charge, notconverged

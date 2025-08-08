import torch
from .seqm_functions.scf_loop import scf_loop
from .seqm_functions.energy import *
from .seqm_functions.parameters import params, PWCCT
from torch.autograd import grad
from .seqm_functions.constants import ev
from .seqm_functions.pack import pack
from .seqm_functions.anal_grad import scf_analytic_grad, scf_grad
from .seqm_functions.rcis_batch import rcis_batch, calc_cis_energy
from .seqm_functions.rcis_grad_batch import rcis_grad_batch
from .seqm_functions.nac import calc_nac
from .seqm_functions.rpa import rpa
from .seqm_functions.normal_modes import normal_modes

import os
import time
import copy

"""
Semi-Emperical Quantum Mechanics: AM1/MNDO/PM3/PM6/PM6_SP
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
                      ],
               
                'PM6_SP':['U_ss', 'U_pp', 'zeta_s', 'zeta_p',  'beta_s', 'beta_p',
                          's_orb_exp_tail', 'p_orb_exp_tail',
                       'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'F0SD', 'G2SD','rho_core',
                       'alpha', 'EISOL',
                       'Gaussian1_K', 'Gaussian2_K', 'Gaussian3_K','Gaussian4_K',
                       'Gaussian1_L', 'Gaussian2_L', 'Gaussian3_L','Gaussian4_L',
                       'Gaussian1_M', 'Gaussian2_M', 'Gaussian3_M','Gaussian4_M'
                      ],
                      
                'PM6_SP_STAR':['U_ss', 'U_pp', 'zeta_s', 'zeta_p',  'beta_s', 'beta_p',
                          's_orb_exp_tail', 'p_orb_exp_tail',
                       'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'F0SD', 'G2SD','rho_core',
                       'alpha', 'EISOL',
                       'Gaussian1_K', 'Gaussian2_K', 'Gaussian3_K','Gaussian4_K',
                       'Gaussian1_L', 'Gaussian2_L', 'Gaussian3_L','Gaussian4_L',
                       'Gaussian1_M', 'Gaussian2_M', 'Gaussian3_M','Gaussian4_M'
                      ],
              }

class Parser(torch.nn.Module):
    """
    parsing inputs from coordinates and types
    """
    def __init__(self, seqm_parameters):
        """
        Constructor
        """
        super().__init__()
        self.outercutoff = seqm_parameters.get('pair_outer_cutoff',1e10)
        if seqm_parameters.get('elements') is None:
            raise RuntimeError("Please instantiate a Molecule object before instantiating an object of Electronic_structure or Molecular_Dynamics class")
        self.elements = seqm_parameters['elements']
        self.uhf = seqm_parameters.get('UHF', False)
        self.hipnn_automatic_doublet = seqm_parameters.get('HIPNN_automatic_doublet', False)

    def forward(self, molecule, themethod, return_mask_l=False, do_large_tensors=True, *args, **kwargs):
        """
        constants : instance of Class Constants
        species : atom types for atom in each molecules,
                  shape (nmol, molsize),  dtype: torch.int64
        coordinates : atom position, shape (nmol, molsize, 3)
        charges: total charge for each molecule, shape (nmol,), 0 if None
        do_large_tensors: option to skip computation of idxi, idxj, ni, nj, rij, xij. For SEDACS only.
        """
        device = molecule.coordinates.device
        dtype = molecule.coordinates.dtype

        nmol, molsize = molecule.species.shape
        nonblank = molecule.species>0
        n_real_atoms = torch.sum(nonblank)

        atom_index = torch.arange(nmol*molsize, device=device,dtype=torch.int64)
        real_atoms = atom_index[nonblank.reshape(-1)>0]

        Z = molecule.species.reshape(-1)[real_atoms]

        if(themethod == 'PM6'): # PM6 is not implemented yet
            nHeavy = torch.sum(((molecule.species>1) & ((molecule.species <= 12) | ((molecule.species >= 18) & (molecule.species <=20)) | ((molecule.species >= 30) & (molecule.species <= 32)) | ((molecule.species >= 36) & (molecule.species <= 38)) | ((molecule.species >= 48) & (molecule.species <= 50)) | ((molecule.species >= 54) & (molecule.species <= 56)) | ((molecule.species >= 80) & (molecule.species <= 83)))),dim=1)
        else:
            nHeavy = torch.sum(molecule.species>1,dim=1)

        nSuperHeavy = torch.sum((((molecule.species > 12) & (molecule.species <18)) | ((molecule.species > 20) & (molecule.species <30)) | ((molecule.species > 32) & (molecule.species <36)) | ((molecule.species > 38) & (molecule.species <48)) | ((molecule.species > 50) & (molecule.species <54)) | ((molecule.species > 70) & (molecule.species <80)) | (molecule.species ==57)),dim=1)

        nHydro = torch.sum(molecule.species==1,dim=1)
        tore = molecule.const.tore
        n_charge = torch.sum(tore[molecule.species],dim=1).reshape(-1).type(torch.int64) # number of valence electrons, N*|e|
        if torch.is_tensor(molecule.tot_charge):
            n_charge -= molecule.tot_charge.reshape(-1).type(torch.int64)
        
        if self.uhf:
            nocc_alpha = n_charge/2. + (molecule.mult-1)/2.
            nocc_beta = n_charge/2. - (molecule.mult-1)/2.
            if ((nocc_alpha%1 != 0).any() or (nocc_beta%1 != 0).any()):
                ### if self.hipnn_automatic_doublet = True, the block below assumes no multiplicity was provided (i.e. all provided as default singlets)
                ### and converts molecules with odd number of electrons to doublets.
                if not self.hipnn_automatic_doublet:
                    raise ValueError("Invalid charge/multiplicity combination!")
                else:
                    #print('alpha beta',nocc_alpha, nocc_beta)
                    nocc_alpha[nocc_alpha%1 != 0] += 0.5
                    nocc_beta[nocc_beta%1 != 0] -= 0.5
                    #print('hipnn_automatic_doublet flag is True. Molecules with odd number of electrons are treated as doublets.\n')
                    
                    #print('alpha beta',nocc_alpha, '\n', nocc_beta, '\n')
            nocc = torch.stack((nocc_alpha,nocc_beta), dim=1)
            nocc = nocc.type(torch.int64)
        else:
            nocc = n_charge//2
            if ((n_charge%2)==1).any():
                raise ValueError("RHF setting requires closed shell systems (even number of electrons)")
        
        
        
        t1 = (torch.arange(molsize,dtype=torch.int64,device=device)*(molsize+1)).reshape((1,-1))
        t2 = (torch.arange(nmol,dtype=torch.int64,device=device)*molsize**2).reshape((-1,1))
        maskd = (t1+t2).reshape(-1)[real_atoms]

        

        if do_large_tensors:
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
            pairdist_sq = torch.square(paircoord_raw).sum(dim=1)
            close_pairs = pairdist_sq < self.outercutoff**2
            
            pairs = (pair_first < pair_second) * nonblank_pairs * close_pairs
            
            paircoord = paircoord_raw[pairs]
            pairdist = torch.sqrt(pairdist_sq[pairs])
            rij = pairdist * molecule.const.length_conversion_factor

            inv_real_atoms = torch.zeros((nmol*molsize,), device=device,dtype=torch.int64)
            inv_real_atoms[real_atoms] = torch.arange(n_real_atoms, device=device,dtype=torch.int64)

            idxi = inv_real_atoms[pair_first[pairs]]
            idxj = inv_real_atoms[pair_second[pairs]]
            ni = Z[idxi]
            nj = Z[idxj]
            xij = paircoord / pairdist.unsqueeze(1)
            mask = real_atoms[idxi] * molsize + real_atoms[idxj]%molsize
            mask_l = real_atoms[idxj] * molsize + real_atoms[idxi]%molsize
            #mask_l = torch.sort(mask_l)[0]
            pair_molid = atom_molid[idxi] # doesn't matter atom_molid[idxj]

        else:
            atom_molid = None
            idxi = None
            idxj = None
            ni = None
            nj = None
            rij = None
            xij = None
            mask = None
            mask_l = None
            pair_molid = None

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
        self.learned_list = seqm_parameters.get('learned',[])
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
        
        # If we are calculating excited states with CIS, then SCF convergence should be at least 1e-2 smaller than
        # CIS tolerance. Here I check for that
        if seqm_parameters.get('excited_states') is not None: # If excited_states are requested in the input
            # Get the cis_tolerance. If cis_tolerance was not in the seqm_parameters, then set it 
            # to the default value here
            excited_options = seqm_parameters.get('excited_states')
            if not isinstance(excited_options,dict):
                raise Exception("Invalid format for excited_states. Expected input like  'excited_states': {'method': 'rpa', 'n_states': 3, 'tolerance' : 1e-6}")
            excited_options['tolerance'] = excited_options.get('tolerance',1e-6)
            excited_options['method'] = excited_options.get('method','cis').lower()
            cis_tol = excited_options['tolerance']
            if seqm_parameters['scf_eps'] > 1e-1*cis_tol:
                seqm_parameters['scf_eps'] = 1e-1*cis_tol

        #put eps and scf_backward_eps as torch.nn.Parameter such that it is saved with model and can
        #be used to restart jobs
        self.eps = torch.nn.Parameter(torch.as_tensor(seqm_parameters['scf_eps']), requires_grad=False)
        self.sp2 = seqm_parameters.get('sp2', [False])
        self.scf_converger = seqm_parameters['scf_converger']
        # whether return eigenvalues, eigenvectors, and gap. Otherwise they are None
        self.eig = seqm_parameters.get('eig', True)
        self.scf_backward = seqm_parameters.get('scf_backward', 0)
        scf_back_eps = seqm_parameters.get('scf_backward_eps', 1e-2)
        self.scf_backward_eps = torch.nn.Parameter(torch.as_tensor(scf_back_eps), requires_grad=False)
        # 0: ignore gradient on density matrix from Hellmann Feymann Theorem,
        # 1: use recursive formula go back through SCF loop
        # 2: direct backprop through SCF loop

    def forward(self, molecule, themethod, P0=None):
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
        
        
        if(themethod == 'PM6'): # not implemented yet
            F, e, P, Hcore, w, charge,rho0xi,rho0xj, riXH, ri, notconverged, molecular_orbitals = scf_loop(molecule,
                                  eps = self.eps,
                                  P=P0,
                                  sp2=self.sp2,
                                  scf_converger=self.scf_converger,
                                  eig=self.eig,
                                  scf_backward=self.scf_backward,
                                  scf_backward_eps=self.scf_backward_eps)

        else:
            F, e, P, Hcore, w, charge, rho0xi,rho0xj, riXH, ri, notconverged, molecular_orbitals = scf_loop(molecule,
                              eps = self.eps,
                              P=P0,
                              sp2=self.sp2,
                              scf_converger=self.scf_converger,
                              eig=self.eig,
                              scf_backward=self.scf_backward,
                              scf_backward_eps=self.scf_backward_eps)
        #
        return F, e, P, Hcore, w, charge,rho0xi,rho0xj, riXH, ri, notconverged, molecular_orbitals

class Energy(torch.nn.Module):
    def __init__(self, seqm_parameters):
        """
        Constructor
        """
        super().__init__()
        self.seqm_parameters = seqm_parameters
        self.method = seqm_parameters['method']
        self.parser = Parser(seqm_parameters)
        self.packpar = Pack_Parameters(seqm_parameters)
        self.hamiltonian = Hamiltonian(seqm_parameters)
        self.Hf_flag = seqm_parameters.get('Hf_flag', True)
        self.uhf = seqm_parameters.get('UHF', False)
        self.eig = seqm_parameters.get('eig', True)
        self.excited_states = seqm_parameters.get('excited_states')
        # log_memory("Energy Initialization")

    def forward(self, molecule, learned_parameters=dict(), all_terms=False, P0=None, cis_amp=None, *args, **kwargs):
        """
        get the energy terms
        """

        # log_memory("Energy forward start")
        molecule.nmol, molecule.molsize, \
        molecule.nSuperHeavy, molecule.nHeavy, molecule.nHydro, molecule.nocc, \
        molecule.Z, molecule.maskd, molecule.atom_molid, \
        molecule.mask, molecule.pair_molid, molecule.ni, molecule.nj, molecule.idxi, molecule.idxj, molecule.xij, molecule.rij = self.parser(molecule, self.method, *args, **kwargs)
         
        if callable(learned_parameters):
            adict = learned_parameters(molecule.species, molecule.coordinates)
            molecule.parameters, molecule.alp, molecule.chi = copy.deepcopy(self.packpar(molecule.Z, learned_params = adict)   )
        else:
            molecule.parameters, molecule.alp, molecule.chi = copy.deepcopy(self.packpar(molecule.Z, learned_params = learned_parameters))


        if(molecule.method == 'PM6'):
            molecule.parameters['beta'] = torch.cat((molecule.parameters['beta_s'].unsqueeze(1), molecule.parameters['beta_p'].unsqueeze(1), molecule.parameters['beta_d'].unsqueeze(1)),dim=1)
        else:
            molecule.parameters['beta'] = torch.cat((molecule.parameters['beta_s'].unsqueeze(1), molecule.parameters['beta_p'].unsqueeze(1)),dim=1)        
            molecule.parameters['zeta_d'] = torch.zeros_like(molecule.parameters['zeta_s'])
            molecule.parameters['s_orb_exp_tail'] = torch.zeros_like(molecule.parameters['zeta_s'])
            molecule.parameters['p_orb_exp_tail'] = torch.zeros_like(molecule.parameters['zeta_s'])
            molecule.parameters['d_orb_exp_tail'] = torch.zeros_like(molecule.parameters['zeta_s'])

            molecule.parameters['U_dd'] = torch.zeros_like(molecule.parameters['U_ss'])
            molecule.parameters['F0SD'] = torch.zeros_like(molecule.parameters['U_ss'])
            molecule.parameters['G2SD'] = torch.zeros_like(molecule.parameters['U_ss'])
            molecule.parameters['rho_core'] = torch.zeros_like(molecule.parameters['U_ss'])

        
        molecule.parameters['Kbeta'] = molecule.parameters.get('Kbeta', None)
        # log_memory("Energy forward scf start")
        
        F, e, P, Hcore, w, charge, rho0xi,rho0xj, riXH, ri, notconverged, molecular_orbitals =  self.hamiltonian(molecule, self.method, \
                                                 P0=P0)
        
        # log_memory("Energy forward scf done")
        
        if self.eig:
            if self.uhf:
                if 0 in molecule.nocc:
                    print('Zero occupied alpha or beta orbitals found (e.g. triplet H2). HOMO-LUMO gaps are not available.')
                    e_gap = torch.tensor([])
                else:
                    lumo_a, lumo_b = molecule.nocc[:,0].unsqueeze(0).T, molecule.nocc[:,1].unsqueeze(0).T
                    e_gap_a = e[:,0].gather(1, lumo_a) - e[:,0].gather(1, lumo_a-1)
                    e_gap_b = e[:,1].gather(1, lumo_b) - e[:,1].gather(1, lumo_b-1)
                    e_gap = torch.stack((e_gap_a.reshape(-1), e_gap_b.reshape(-1)), dim=1)
            else:
                lumo = molecule.nocc.unsqueeze(0).T
                e_gap = (e.gather(1, lumo) - e.gather(1, lumo-1)).reshape(-1)
        else:
            e_gap = None
        
        molecule.molecular_orbitals = molecular_orbitals
        
        #nuclear energy
        alpha = molecule.parameters['alpha']
        if self.method=='MNDO':
            parnuc = (alpha,)
        elif self.method=='AM1' or self.method=='PM6' or self.method=='PM6_SP' or self.method=='PM6_SP_STAR':
            K = torch.stack((molecule.parameters['Gaussian1_K'],
                             molecule.parameters['Gaussian2_K'],
                             molecule.parameters['Gaussian3_K'],
                             molecule.parameters['Gaussian4_K']),dim=1)
            #
            L = torch.stack((molecule.parameters['Gaussian1_L'],
                             molecule.parameters['Gaussian2_L'],
                             molecule.parameters['Gaussian3_L'],
                             molecule.parameters['Gaussian4_L']),dim=1)
            #molecule.
            M = torch.stack((molecule.parameters['Gaussian1_M'],
                             molecule.parameters['Gaussian2_M'],
                             molecule.parameters['Gaussian3_M'],
                             molecule.parameters['Gaussian4_M']),dim=1)
            #
            parnuc = (alpha, K, L, M)
        elif self.method=='PM3':
            K = torch.stack((molecule.parameters['Gaussian1_K'],
                             molecule.parameters['Gaussian2_K']),dim=1)
            #
            L = torch.stack((molecule.parameters['Gaussian1_L'],
                             molecule.parameters['Gaussian2_L']),dim=1)
            #
            M = torch.stack((molecule.parameters['Gaussian1_M'],
                             molecule.parameters['Gaussian2_M']),dim=1)
            #
            parnuc = (alpha, K, L, M)

        if 'g_ss_nuc' in molecule.parameters:
            g = molecule.parameters['g_ss_nuc']
            rho0a = 0.5 * ev / g[molecule.idxi]
            rho0b = 0.5 * ev / g[molecule.idxj]
            gam = ev / torch.sqrt(molecule.rij**2 + (rho0a + rho0b)**2)
        else:
            gam = w[...,0,0]
        # log_memory("Energy forward energy param stack done")
        
        EnucAB = pair_nuclear_energy(molecule.Z, molecule.const, molecule.nmol, molecule.ni, molecule.nj, molecule.idxi, molecule.idxj, molecule.rij, \
                                     rho0xi,rho0xj,molecule.alp, molecule.chi, gam=gam, method=self.method, parameters=parnuc)
        # log_memory("Energy forward EnucAB done")
        Eelec = elec_energy(P, F, Hcore)
        # log_memory("Energy forward Eelec done")
        
        do_analytical_gradient = self.seqm_parameters.get('analytical_gradient',[False])
        if do_analytical_gradient[0] and molecule.active_state==0:
            log_memory("Energy forward ground analytical start")
            # None of the tensors will need gradients with backpropogation (unless I wnat to do second derivatives), so 
            # we can save on memory since the compuational graph doesn't have to be stored.
            beta = molecule.parameters['beta']
            if molecule.const.do_timing: t0 = time.time()
            with torch.no_grad():
                if len(do_analytical_gradient) > 1 and do_analytical_gradient[1].lower() == 'numerical':
                    molecule.analytical_gradient =  scf_grad( P0=P, molecule = molecule, const=molecule.const, method = self.method,
                                                  molsize=molecule.molsize, maskd=molecule.maskd, mask=molecule.mask, idxi=molecule.idxi,
                                                  idxj=molecule.idxj, ni=molecule.ni, nj=molecule.nj, xij=molecule.xij, gam=gam, rij=molecule.rij,
                                                  Z=molecule.Z, parnuc = parnuc, zetas=molecule.parameters['zeta_s'], zetap=molecule.parameters['zeta_p'],
                                                  beta=beta,)
                # elif analytical_gradient[1].lower()=='analytical':
                else:
                    molecule.analytical_gradient =  scf_analytic_grad( P0=P, molecule=molecule, const=molecule.const, method = self.method,
                              molsize=molecule.molsize, maskd=molecule.maskd, mask=molecule.mask, idxi=molecule.idxi, idxj=molecule.idxj,
                              ni=molecule.ni, nj=molecule.nj, xij=molecule.xij, rij=molecule.rij, Z=molecule.Z, gam=gam, parnuc = parnuc,
                              zetas=molecule.parameters['zeta_s'], zetap=molecule.parameters['zeta_p'], gss=molecule.parameters['g_ss'],
                              gpp=molecule.parameters['g_pp'], gp2=molecule.parameters['g_p2'], hsp=molecule.parameters['h_sp'],
                              beta=beta, ri=ri, riXH=riXH,)
                    
            log_memory("Energy forward ground analytical done")
            if molecule.const.do_timing:
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t1 = time.time()
                molecule.const.timing["Force"].append(t1 - t0)

        if molecule.active_state > 0 and self.excited_states is None:
            raise Exception("You have requested for excited state dynamics but have not given input parameters for excited states (like n_states) in seqm_parameters")

        Eexcited = 0.0
        if self.excited_states is not None:
            # log_memory("Energy forward excited states start")
            cis_tol = self.excited_states['tolerance']
            method = self.excited_states['method'].lower()
            with torch.no_grad():
                if molecule.const.do_timing: t0 = time.time()
                if method == 'cis':
                    excitation_energies, exc_amps = rcis_batch(molecule,w,e,self.excited_states['n_states'],cis_tol,init_amplitude_guess=cis_amp)
                elif method == 'rpa':
                    excitation_energies, exc_amps = rpa(molecule,w,e,self.excited_states['n_states'],cis_tol,init_amplitude_guess=cis_amp)
                else:
                    raise Exception("Excited state method has to be CIS or RPA")
                torch.cuda.empty_cache()

                molecule.cis_amplitudes = exc_amps

                if molecule.const.do_timing:
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    t1 = time.time()
                    molecule.const.timing["CIS/RPA"].append(t1 - t0)


                cis_nac = kwargs.get('cis_nac',[False])
                if cis_nac[0]:
                    calc_nac(molecule,exc_amps, excitation_energies, P, ri, riXH,cis_nac[1],cis_nac[2],rpa=method=='rpa')

            # log_memory("Energy forward excited states done")
            if molecule.active_state>0:
                molecule.cis_energies = excitation_energies
                # Eelec += excitation_energies[:,molecule.active_state-1]
                log_memory("Energy forward excited states energy start")
                Eexcited = calc_cis_energy(molecule,w,e,exc_amps[...,self.seqm_parameters['active_state']-1,:],rpa=method=='rpa')
                log_memory("Energy forward excited states energy done")

                if do_analytical_gradient[0]:
                    log_memory("Energy forward excited states analytical grad start")
                    if molecule.const.do_timing: t0 = time.time()
                    molecule.analytical_gradient = rcis_grad_batch(molecule,w,e,riXH,ri,P,cis_tol,gam,self.method,parnuc,rpa=method=='rpa',include_ground_state=True)
                    t1 = time.time()
                    molecule.const.timing["Force"].append(t1 - t0)
                    log_memory("Energy forward excited states analytical grad done")


        if self.eig and not self.uhf:
            molecule.old_mos = molecule.molecular_orbitals.clone()

        # log_memory("Energy forward clone mo's")
        if all_terms:
            Etot, Enuc = total_energy(molecule.nmol, molecule.pair_molid,EnucAB, Eelec)
            Eiso = elec_energy_isolated_atom(molecule.const, molecule.Z,
                                         uss=molecule.parameters['U_ss'],
                                         upp=molecule.parameters['U_pp'],
                                         gss=molecule.parameters['g_ss'],
                                         gpp=molecule.parameters['g_pp'],
                                         gsp=molecule.parameters['g_sp'],
                                         gp2=molecule.parameters['g_p2'],
                                         hsp=molecule.parameters['h_sp'])
            Etot += Eexcited
            Hf, Eiso_sum = heat_formation(molecule.const, molecule.nmol, molecule.atom_molid, molecule.Z, Etot, Eiso, flag = self.Hf_flag)
            # log_memory("Energy forward total energy done")
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
        self.create_graph = seqm_parameters.get('2nd_grad', False)
        self.uhf = seqm_parameters.get('UHF', False)
        self.eig = seqm_parameters.get('eig', True)
        self.seqm_parameters = seqm_parameters
        # log_memory("Force Initialization")

    def forward(self, molecule, learned_parameters=dict(), P0=None, cis_amp=None, do_force=True, *args, **kwargs):
        
        # log_memory("Force forward start")
        # We have two options to calculate force: 1. Analytical gradients (including semi-numerical gradients) and 2. From back-propogagation
        do_analytical_gradient = self.seqm_parameters.get('analytical_gradient', [False])

        # For excited states, back-prop forces work only if we have scf_backward == 1 or 2. We will have to fall back on analytical_gradient otherwise
        if molecule.active_state > 0 and do_force:
            if self.seqm_parameters.get('scf_backward', 0) == 0:
                do_analytical_gradient = [True]
                self.seqm_parameters['analytical_gradient'] = do_analytical_gradient

        molecule.coordinates.requires_grad_(do_force and not do_analytical_gradient[0])

        Hf, Etot, Eelec, Enuc, Eiso, _, e_gap, e, D, charge, notconverged = \
            self.energy(molecule, learned_parameters=learned_parameters, all_terms=True, P0=P0, cis_amp=cis_amp, *args, **kwargs)
        # log_memory("Force forward energy done")

        if self.seqm_parameters.get('normal modes', False):
            if self.seqm_parameters.get('scf_backward', 0) != 2:
                raise Exception("You have requested for normal mode calculation but scf_backward has not been set to 2 in your input seqm_parameters")
            normal_modes(molecule,Hf)
        
        if self.eig:
            e = e.detach()
            e_gap = e_gap.detach()

        if do_analytical_gradient[0]:
            # log_memory("Force forward analytical gradient start")
            force = -molecule.analytical_gradient
            # log_memory("Force forward analytical gradient done")
            return force.detach(), D.detach(), Hf.detach(), Etot.detach(), Eelec.detach(), Enuc.detach(), Eiso.detach(), e, e_gap, charge, notconverged
        #L = Etot.sum()
        if do_force:
            log_memory("Force forward backprop start")
            L = Hf.sum()
            if molecule.const.do_timing: t0 = time.time()

            #gv = [coordinates]
            #gradients  = grad(L, gv,create_graph=self.create_graph)
            #torch.save(D.detach(), 'gs_1_P_py.pt')
            L.backward(create_graph=self.create_graph)
            if molecule.const.do_timing:
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t1 = time.time()
                molecule.const.timing["Force"].append(t1 - t0)
            #force = -gradients[0] 
            if self.create_graph:
                force = -molecule.coordinates.grad.clone()
                with torch.no_grad(): molecule.coordinates.grad.zero_()
            else:
                force = -molecule.coordinates.grad.detach()
                molecule.coordinates.grad.zero_()
            log_memory("Force forward backprop done")
        else:
            force = torch.tensor([])
            # log_memory("Force forward noforce done")
            return force.detach(), D.detach(), Hf.detach(), Etot.detach(), Eelec, Enuc, Eiso.detach(), e, e_gap, charge, notconverged


        # log_memory("Force forward done")
        return force.detach(), D.detach(), Hf.detach(), Etot.detach(), Eelec.detach(), Enuc.detach(), Eiso.detach(), e, e_gap, charge, notconverged


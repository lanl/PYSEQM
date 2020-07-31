import torch
from .seqm_functions.scf_loop import scf_loop
from .seqm_functions.energy import *
from .seqm_functions.parameters import params
from torch.autograd import grad
import time
#chemnn.indexers.OpenPairIndexer,
#     cutoff should be large enough to include all atoms at this moment
"""
Semi-Emperical Quantum Mechanics: AM1/MNDO/PM3
"""



#
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
                      ]}
#


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

    def forward(self, constansts, species, coordinates):
        """
        constants : instance of Class Constants
        species : atom types for atom in each molecules,
                  shape (nmol, molsize),  dtype: torch.int64
        coordinates : atom position, shape (nmol, molsize, 3)
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
        nocc = (torch.sum(tore[species],dim=1)/2.0).reshape(-1).type(torch.int64)
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

        pairs = (pair_first<pair_second) * nonblank_pairs * close_pairs

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
        self.filedir = seqm_parameters['parameter_file_dir']
        self.parameters = parameterlist[self.method]
        self.required_list = []
        for i in self.parameters:
            if i not in self.learned_list:
                self.required_list.append(i)
        self.nrp = len(self.required_list)
        self.p = params(method=self.method, elements=self.elements,root_dir=self.filedir,
                 parameters=self.required_list)

    def forward(self, Z, learned_params=dict()):
        """
        combine the learned_parames with other required parameters
        """
        for i in range(self.nrp):
            learned_params[self.required_list[i]] = self.p[Z,i] #.contiguous()
        return learned_params

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

    def forward(self, const, molsize, nHeavy, nHydro, nocc, Z, maskd, mask, atom_molid, pair_molid, idxi, idxj, ni,nj,xij,rij, parameters, P0=None):
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
        beta = torch.cat((parameters['beta_s'].unsqueeze(1), parameters['beta_p'].unsqueeze(1)),dim=1)
        if "Kbeta" in parameters:
            Kbeta = parameters["Kbeta"]
        else:
            Kbeta = None
        F, e, P, Hcore, w, charge, notconverged = scf_loop(const=const,
                              molsize=molsize,
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
                              uss=parameters['U_ss'],
                              upp=parameters['U_pp'],
                              gss=parameters['g_ss'],
                              gsp=parameters['g_sp'],
                              gpp=parameters['g_pp'],
                              gp2=parameters['g_p2'],
                              hsp=parameters['h_sp'],
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
        return F, e, P, Hcore, w, charge, notconverged

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
            self.Hf_flag = seqm_parameters["Hf_flag"]
        # Hf_flag: true return Hf, false return Etot-Eiso


    def forward(self, const, coordinates, species, learned_parameters=dict(), all_terms=False, P0=None):
        """
        get the energy terms
        """
        nmol, molsize, \
        nHeavy, nHydro, nocc, \
        Z, maskd, atom_molid, \
        mask, pair_molid, ni, nj, idxi, idxj, xij, rij = self.parser(const, species, coordinates)
        if callable(learned_parameters):
            adict = learned_parameters(species, coordinates)
            parameters = self.packpar(Z, learned_params = adict)    
        else:
            parameters = self.packpar(Z, learned_params = learned_parameters)
        F, e, P, Hcore, w, charge, notconverged =  self.hamiltonian(const, molsize, \
                                                 nHeavy, nHydro, nocc, \
                                                 Z, maskd, \
                                                 mask, atom_molid, pair_molid, idxi, idxj, ni,nj,xij,rij, \
                                                 parameters, P0=P0)
        #nuclear energy
        alpha = parameters['alpha']
        if self.method=='MNDO':
            parnuc = (alpha,)
        elif self.method=='AM1':
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
        EnucAB = pair_nuclear_energy(const, nmol, ni, nj, idxi, idxj, rij, gam=w[...,0,0], method=self.method, parameters=parnuc)
        Eelec = elec_energy(P, F, Hcore)
        if all_terms:
            Etot, Enuc = total_energy(nmol, pair_molid,EnucAB, Eelec)
            Eiso = elec_energy_isolated_atom(const, Z,
                                         uss=parameters['U_ss'],
                                         upp=parameters['U_pp'],
                                         gss=parameters['g_ss'],
                                         gpp=parameters['g_pp'],
                                         gsp=parameters['g_sp'],
                                         gp2=parameters['g_p2'],
                                         hsp=parameters['h_sp'])
            Hf, Eiso_sum = heat_formation(const, nmol,atom_molid, Z, Etot, Eiso, flag = self.Hf_flag)
            return Hf, Etot, Eelec, Enuc, Eiso_sum, EnucAB, e, P, charge, notconverged
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


    def forward(self, const, coordinates, species, learned_parameters=dict(), P0=None):

        coordinates.requires_grad_(True)
        #print(learned_parameters)
        #learned_parameters['U_ss'].register_hook(print)
        #"""
        Hf, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, notconverged = self.energy(const, coordinates, species, learned_parameters=learned_parameters, all_terms=True, P0=P0)
        L = Etot.sum()
        if const.do_timing:
            t0 = time.time()
        #"""
        """
        Eelec, EnucAB, P = self.energy(const, coordinates, species, learned_parameters=learned_parameters, all_terms=False, P0=P0)
        L = Eelec.sum() + EnucAB.sum()
        #"""
        #L.backward()
        #"""
        gv = [coordinates]

        gradients  = grad(L, gv,create_graph=self.create_graph)
        #coordinates.grad = gradients[0]
        #"""
        if const.do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            const.timing["Force"].append(t1-t0)


        #L.backward(retain_graph=True)
        #print(learned_parameters['U_ss'].grad)
        force = -gradients[0]
        """
        with torch.no_grad():
            force = -coordinates.grad.clone()
            coordinates.grad.zero_()
        #"""

        return force, P, Etot, Hf, Eelec, Enuc, Eiso, EnucAB, e, charge, notconverged

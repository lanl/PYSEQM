import torch
from seqm.seqm_functions import diat_overlap
from seqm.seqm_functions.constants import overlap_cutoff
from seqm.seqm_functions.parameters import params
from seqm.basics import Parser
from seqm.seqm_functions.pack import unpack, pack
from seqm.seqm_functions.diag import sym_eig_trunc1

def huckel(const,nmol, molsize, maskd, mask, idxi, idxj, ni,nj,xij,rij, Z, zetas, zetap, beta, Kstar=None):
    dtype = xij.dtype
    device = xij.device
    qn_int = const.qn_int

    zeta = torch.cat((zetas.unsqueeze(1), zetap.unsqueeze(1)),dim=1)
    overlap_pairs = rij<=overlap_cutoff
    #di=th.zeros((npairs,4,4),dtype=dtype, device=device)
    di = torch.zeros((xij.shape[0], 4, 4),dtype=dtype, device=device)
    di[overlap_pairs] = diatom_overlap_matrix(ni[overlap_pairs],
                               nj[overlap_pairs],
                               xij[overlap_pairs],
                               rij[overlap_pairs],
                               zeta[idxi][overlap_pairs],
                               zeta[idxj][overlap_pairs],
                               qn_int)
    #
    ntotatoms = nmol * molsize
    M = torch.zeros(nmol*molsize*molsize,4,4,dtype=dtype,device=device)
    M[maskd,0,0] = beta[:,0]
    M[maskd,1,1] = beta[:,1]
    M[maskd,2,2] = beta[:,1]
    M[maskd,3,3] = beta[:,1]

    if torch.is_tensor(Kstar):
        M[mask,0,0]   = di[...,0,0]*(beta[idxi,0]+beta[idxj,0])/2.0 * Kstar
        M[mask,0,1:]  = di[...,0,1:]*(beta[idxi,0:1]+beta[idxj,1:2])/2.0 * Kstar[:,None]
        M[mask,1:,0]  = di[...,1:,0]*(beta[idxi,1:2]+beta[idxj,0:1])/2.0 * Kstar[:,None]
        M[mask,1:,1:] = di[...,1:,1:]*(beta[idxi,1:2,None]+beta[idxj,1:2,None])/2.0 * Kstar[:,None,None]
    else:
        Kstar = 1.75
        #beta is for each atom in the molecules, shape (ntotatoms,2)
        M[mask,0,0]   = di[...,0,0]*(beta[idxi,0]+beta[idxj,0])/2.0*Kstar
        M[mask,0,1:]  = di[...,0,1:]*(beta[idxi,0:1]+beta[idxj,1:2])/2.0*Kstar
        M[mask,1:,0]  = di[...,1:,0]*(beta[idxi,1:2]+beta[idxj,0:1])/2.0*Kstar
        M[mask,1:,1:] = di[...,1:,1:]*(beta[idxi,1:2,None]+beta[idxj,1:2,None])/2.0*Kstar

    H = M.reshape(nmol,molsize,molsize,4,4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
    #

    return H.triu()+H.triu(1).transpose(1,2)

def Huckel(torch.nn.Module):

    def __init__(self, parameters):
        super().__init__()
        self.parser = Parser(parameters)
        self.method = parameters['method']
        self.elements = parameters['elements']
        self.learned_list = parameters['learned']
        self.dtype = parameters['dtype']
        self.device = parameters['device']
        self.filedir = parameters['parameter_file_dir']
        self.parameters = ['U_ss', 'U_pp', 'zeta_s', 'zeta_p']
        self.required_list = []
        for i in self.parameters:
            if i not in self.learned_list:
                self.required_list.append(i)
        self.nrp = len(self.required_list)
        self.p = params(method=self.method, elements=self.elements,root_dir=self.filedir,
                 parameters=self.required_list,
                 dtype=self.dtype,
                 device=self.device).transpose(0,1)

    def forward(self, const, coordinates, species, learned_parameters=dict()):
        nmol, molsize, \
        nHeavy, nHydro, nocc, \
        Z, maskd, atom_molid, \
        mask, pair_molid, ni, nj, idxi, idxj, xij, rij = self.parser(const, species, coordinates)
        parameters = learned_parameters
        for i in range(self.nrp):
            parameters[self.required_list[i]] = self.p[i,Z]
        beta = torch.cat((parameters['U_ss'].unsqueeze(1), parameters['U_pp'].unsqueeze(1)),dim=1)
        if "Kstar" in parameters:
            Kstar = parameters["Kstar"]
        else:
            Kstar = None
        H = huckel(const=const,
                   nmol=nmol,
                   molsize=molsize,
                   maskd=maskd,
                   mask=mask,
                   idxi=idxi,
                   idxj=idxj,
                   ni=ni,
                   nj=nj,
                   xij=xij,
                   rij=rij,
                   Z=Z,
                   zetas=parameters['zeta_s'],
                   zetap=parameters['zeta_p'],
                   beta=beta,
                   Kstar=Kstar,
                   )
        #
        e, v = sym_eig_trunc1(H, nHeavy, nHydro, nOccMO,eig_only=True)
        return H, e, v






if __name__ == "__main__":
    dtype=torch.float64
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    parameters = {
        'method' : 'AM1', #methods for geting zeta_s and zeta_p for overlap calculation
        'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
        'elements' : [0,1,6,7,8],
        'learned' : [],
        'dtype' : dtype,
        'device' : device,
    }

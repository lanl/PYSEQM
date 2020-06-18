import torch
import numpy as np
from torch.utils.data import Dataset

################################################################################
# important point for torch.s data_loader
# atoms inside each molecule is sorted in descending order based on atom type
# only pairs witorch.ni>nj are generated, O-C ==> OK, C-O ==> not
# atom indexing is used for real and virtual padding atoms
# which is consistent witorch.maskd: mask for diagonal block in test.py
# ==> ntotatoms = nmol*molsize
# Z : shape (nmol*molsize,), maskd : shape (nmol*molsize,)
# when shifting to use torch. index from hippynn, maskd should also be updated and
#      define witorch.consistency
# in hippynn, real_atom shape (total_num_real_atoms,)
#      only real atom, not padding in batch
################################################################################

class ALDataset(Dataset):
    """
    AL dataset
    load data from AL dataset
    return nmol, molsize, Z, nHeavy, nHydro, nocc, mask, ni, nj, idxi, idxj, xij, rij
    """

    def __init__(self, root_dir="/auto/nest/nest/u/gzhou/softwares/mopac/mopac7/pytorch/datasets/data-AL-complete-4-4-0/",
                       prefix="data-AL-complete-4-4-0",
                       innercutoff=0.005,
                       outercutoff=1.0e10,
                       device=torch.device('cpu')):
        """
        prefix (string): prefix of torch. npy file name
        root_dir (string): directory of files
        exact path will be root_dir+prefix+'-'+suffix+'.npy'
        suffix (string) will be used is R, Z, T at torch.s moment
        Z: atom type
        R: atom position
        T: total energy
        ODMOtrimenergy: energy for HOMO-3, ..., HOMO,LUMO, ..., LUMO+3

        inndercutoff,outercutoff: for pair of atoms, if distance < innercutoff
        or > outercutoff, then ignore
        unit angstrom
         the cutoff for the two electron two center integrals in mopac
         is 1.0e10 angstrom, but 10 angstrom in the overlap integral parts

        each batch may have different number of pairs
        """
        self.Z = torch.from_numpy(np.load(root_dir+prefix+"-Z.npy")).type(torch.int64).to(device)
        self.R = torch.from_numpy(np.load(root_dir+prefix+"-R.npy")).to(device)
        self.prefix = prefix
        self.root_dir = root_dir
        self.innercutoff = innercutoff
        self.outercutoff = outercutoff
        self.natoms = self.Z.shape[1] #number of atoms in each molecules
        self.nparis = ((self.natoms+1)*self.natoms)// 2
        self.elements = set(self.Z.reshape(-1).tolist())
        if 0 in self.elements:
            self.elements.remove(0)
        self.elements=sorted(self.elements,reverse=False)
        self.tore=torch.tensor([0.0,
                           1.0,0.0,
                           1.0,2.0,3.0,4.0,5.0,6.0,7.0,0.0,
                           1.0,2.0,3.0,4.0,5.0,6.0,7.0,0.0,], device=device)
        self.a0=0.529167  #used in mopac7
        self.device = device
        #a0=0.5291772109


    def __len__(self):
        return self.Z.shape[0]

    def __getitem__(self, idx):



        #atom type for each molecule is sorted in decending order,
        #pair_type[0]>=pair_type[1]
        #int64 is required for index_select in pytorch
        #
        # Z : atom type
        a0=self.a0
        Z = self.Z[idx,:]
        # nheavy : number of heavy atoms in each molecule
        nHeavy = torch.tensor([torch.sum(Z>2)], dtype=torch.int64, device=self.device)
        # nHydro: number of Hydrogen atoms in each molecule
        nHydro = torch.tensor([torch.sum(Z==1)], dtype=torch.int64, device=self.device)
        # nOccupiedMO : number of occupied moleculer orbitals
        #               only for closed shell, sum(tore[Z]) must be even
        # tore : valence shell charge of each atom type
        nOccupiedMO = (torch.sum(self.tore[Z])/2.0).reshape(1).type(torch.int64).to(self.device)

        # mask: mask to tell torch. position of each pair block in torch. Hamiltonian
        #       only torch. upper triangle blocks are required,
        #       diagonal block not needed
        mask = torch.zeros((self.nparis,), dtype=torch.int64, device=self.device)

        #ni, nj will be used to access torch. charge, so has to be longtensor
        # ni, nj: atom type of torch. first and second atom in each pair
        ni = torch.zeros((self.nparis,),dtype=torch.int64, device=self.device)
        nj = torch.zeros((self.nparis,),dtype=torch.int64, device=self.device)
        # idxi, idxj: atom index of torch. first and second atom in each pair
        #             torch. index is across each batch
        #             torch. padding zeros will also take index, even not used
        #             as idxi, idxj will be used in maskd (diagonal mask) in
        #             hcore and fock matrix construction
        idxi = torch.zeros((self.nparis,),dtype=torch.int64, device=self.device)
        idxj = torch.zeros((self.nparis,),dtype=torch.int64, device=self.device)
        # xij : unit vector pointing from atom i to atom j, rj-ri/|rj-ri|
        xij = torch.zeros((self.nparis,3), device=self.device)
        # rij : distance between atom i and j in atomic unit (bohr)
        rij = torch.zeros((self.nparis,), device=self.device)

        k=0
        #print(idx)
        for i in range(self.natoms):
            if self.Z[idx, i]<=0:
                continue
            for j in range(i+1,self.natoms):
                if self.Z[idx, j]<=0:
                    continue
                rv = self.R[idx, j]-self.R[idx, i]
                r = torch.norm(rv)
                if r>=self.innercutoff and r<=self.outercutoff:
                    ni[k] = self.Z[idx, i]
                    nj[k] = self.Z[idx, j]
                    ##doesn't support random sampling, as torch. value of idx is used in idxi and idxj
                    # ==> fixed, shuffle is supported
                    #idxi[k] = idx*self.natoms+i
                    #idxj[k] = idx*self.natoms+j
                    #mask[i*self.natoms+j]=1
                    #mask[k] = i*self.natoms+j+idx*self.natoms**2
                    #local index, will transfer to global index in torch. collate function below
                    idxi[k] = i
                    idxj[k] = j
                    mask[k] = i*self.natoms+j

                    xij[k] = rv/r
                    rij[k] = r/a0

                    k+=1
        return Z, nHeavy, nHydro, nOccupiedMO, \
                mask[:k], ni[:k], nj[:k], idxi[:k], idxj[:k], xij[:k,:], rij[:k]

    #@staticmetorch.d
    def collate(self, batch):
        #used for DataLoader
        ##### idxi, idxj are torch. atom index across torch. whole dataset
        #can use %(batch_size*self.natoms) to bring back to torch. index for each batch
        #doesn't support random sampling  ==> fixed, shuffle is supported
        #print(batch[0])
        nmol = len(batch)
        #print(nmol)
        Z      = torch.cat([item[0] for item in batch])
        nHeavy = torch.cat([item[1] for item in batch])
        nHydro = torch.cat([item[2] for item in batch])
        nocc   = torch.cat([item[3] for item in batch])
        #mask   = torch.cat([item[4] for item in batch])
        mask   = torch.cat([batch[i][4]+i*self.natoms**2 for i in range(nmol)])
        ni     = torch.cat([item[5] for item in batch])
        nj     = torch.cat([item[6] for item in batch])
        #idxi   = torch.cat([item[7] for item in batch])
        idxi   = torch.cat([batch[i][7]+i*self.natoms for i in range(nmol)])
        #idxj   = torch.cat([item[8] for item in batch])
        idxj   = torch.cat([batch[i][8]+i*self.natoms for i in range(nmol)])
        xij    = torch.cat([item[9] for item in batch])
        rij    = torch.cat([item[10] for item in batch])

        return nmol, self.natoms, Z, nHeavy, nHydro, nocc, mask, ni, nj, idxi, idxj, xij, rij

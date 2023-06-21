import torch
from seqm.basics import *
import time
from .basics import Parser

class Molecule(torch.nn.Module):
    def __init__(self, const, seqm_parameters, coordinates, species, charges=0, mult=1, *args, **kwargs):
        """
        unit for timestep is femtosecond
        output: [molecule id list, frequency N, prefix]
            molecule id in the list are output, staring from 0 to nmol-1
            geometry is writted every dump step to the file with name prefix + molid + .xyz
            step, temp, and total energy is print to screens for select molecules every thermo
        """
        super().__init__(*args, **kwargs)
        self.const = const
        self.seqm_parameters = seqm_parameters
        self.coordinates = coordinates
        self.species = species
        if not torch.is_tensor(charges):
            charges = charges * torch.ones(coordinates.size()[0], device=coordinates.device)
        self.tot_charge = charges
        if not torch.is_tensor(mult):
            mult = mult * torch.ones(coordinates.size()[0], device=coordinates.device)
        self.mult = mult
        
        self.parser = Parser(self.seqm_parameters)
        
        self.nmol, self.molsize, \
        self.nHeavy, self.nHydro, \
        self.norb, self.nocc, self.nvirt, \
        self.Z, self.maskd, self.atom_molid, \
        self.mask, self.mask_l, self.pair_molid, \
        self.ni, self.nj, self.idxi, self.idxj, self.xij, self.rij = self.parser(self, return_mask_l=True, *args, **kwargs)

        MASS = torch.as_tensor(self.const.mass)
        # put the padding virtual atom mass finite as for accelaration, F/m evaluation.
        MASS[0] = 1.0
        self.mass = MASS[self.species].unsqueeze(2)
        
        self.force = None
        self.velocities = None
        self.acc = None
        
        self.dm = None
        self.q = None
        self.d = None
        
        self.Hf = None
        self.Etot = None
        self.Eelec = None
        self.Enuc = None
        self.Eiso = None
        self.e_mo = None #
        self.e_gap = None #
        
        self.charge = None
        self.dipole = None
        
        self.Electronic_entropy = None
        self.Fermi_occ = None
        self.dP2dt2 = None
        self.Krylov_Error = None
        
                
        def get_coordinates(self):
            return self.coordinates

        def get_species(self):
            return self.species



import torch
from seqm.basics import *
import time
from .basics import Parser, Pack_Parameters
import copy

class Molecule(torch.nn.Module):
    def __init__(self, const, seqm_parameters, coordinates, species, charges=0, mult=1, learned_parameters=dict(), do_large_tensors=True, *args, **kwargs):
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
        #self.coordinates = coordinates
        self.coordinates = torch.nn.Parameter(coordinates)
        self.coordinates.requires_grad_(do_large_tensors)
        self.species = species
        if not torch.is_tensor(charges):
            charges = charges * torch.ones(coordinates.size()[0], device=coordinates.device)
        self.tot_charge = charges
        if not torch.is_tensor(mult):
            mult = mult * torch.ones(coordinates.size()[0], device=coordinates.device)
        self.mult = mult
        
        self.seqm_parameters = seqm_parameters
        self.method = seqm_parameters['method']
        
        self.parser = Parser(self.seqm_parameters)
        self.packpar = Pack_Parameters(self.seqm_parameters).to(coordinates.device)
        
        self.nmol, self.molsize, \
        self.nSuperHeavy, self.nHeavy, self.nHydro, self.nocc, \
        self.Z, self.maskd, self.atom_molid, \
        self.mask, self.mask_l, self.pair_molid, \
        self.ni, self.nj, self.idxi, self.idxj, self.xij, self.rij = self.parser(self, self.method, return_mask_l=True, do_large_tensors=do_large_tensors, *args, **kwargs)

        # C = 30
        # y1 = self.rij.clone()
        # y2 = 2*self.rij - C/3 - 3*(self.rij**2)/(4*C)
        # y3 = torch.ones(self.rij.shape)*C
        # mask_y2 =( self.rij >= 2/3*C) * (self.rij <= 4/3*C)
        # y1[mask_y2] = y2[mask_y2]
        # mask_y3 = (self.rij > 4/3*C)
        # y1[mask_y3] = y3[mask_y3]
        # self.rij = y1





        if callable(learned_parameters):
            adict = learned_parameters(self.species, self.coordinates)
            self.parameters, self.alp, self.chi = copy.deepcopy(self.packpar(self.Z, learned_params = adict)  )
        else:
            self.parameters, self.alp, self.chi = copy.deepcopy(self.packpar(self.Z, learned_params = learned_parameters))



        
        if(self.method == 'PM6'): # PM6 not implemented yet. Only PM6_SP
            self.parameters['beta'] = torch.cat((self.parameters['beta_s'].unsqueeze(1), self.parameters['beta_p'].unsqueeze(1), self.parameters['beta_d'].unsqueeze(1)),dim=1)
        else:
            self.parameters['beta'] = torch.cat((self.parameters['beta_s'].unsqueeze(1), self.parameters['beta_p'].unsqueeze(1)),dim=1)        
            self.parameters['zeta_d'] = torch.zeros_like(self.parameters['zeta_s'])
            self.parameters['s_orb_exp_tail'] = torch.zeros_like(self.parameters['zeta_s'])
            self.parameters['p_orb_exp_tail'] = torch.zeros_like(self.parameters['zeta_s'])
            self.parameters['d_orb_exp_tail'] = torch.zeros_like(self.parameters['zeta_s'])

            self.parameters['U_dd'] = torch.zeros_like(self.parameters['U_ss'])
            self.parameters['F0SD'] = torch.zeros_like(self.parameters['U_ss'])
            self.parameters['G2SD'] = torch.zeros_like(self.parameters['U_ss'])
            self.parameters['rho_core'] = torch.zeros_like(self.parameters['U_ss'])

        
        self.parameters['Kbeta'] = self.parameters.get('Kbeta', None)

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



import torch
from .basics import *
import time
from seqm.XLBOMD import ForceXL
from seqm.XLBOMD_LR import ForceXL as ForceXL_lr

class Electronic_Structure(torch.nn.Module):
    def __init__(self, seqm_parameters, *args, **kwargs):
        """
        unit for timestep is femtosecond
        output: [molecule id list, frequency N, prefix]
            molecule id in the list are output, staring from 0 to nmol-1
            geometry is writted every dump step to the file with name prefix + molid + .xyz
            step, temp, and total energy is print to screens for select molecules every thermo
        """
        super().__init__(*args, **kwargs)
        #self.molecule = molecule
        self.seqm_parameters = seqm_parameters
        self.conservative_force = Force(self.seqm_parameters)
        self.conservative_force_xl = ForceXL(self.seqm_parameters)
        self.conservative_force_xl_lr = ForceXL_lr(self.seqm_parameters)


        #self.acc_scale = 0.009648532800137615
        #self.output = output

    def forward(self, molecule, learned_parameters=dict(), P0=None, err_threshold = None, rank = None, T_el = None, dm_prop='SCF', *args, **kwargs):
        """
        return force in unit of eV/Angstrom
        return force, density matrix, total energy of this batch
        """
        if dm_prop=='SCF':
            molecule.force, P, molecule.Hf, molecule.Etot, molecule.Eelec, molecule.Enuc, molecule.Eiso, molecule.e_mo, molecule.e_gap, self.charge, self.notconverged = \
                        self.conservative_force(molecule, P0=P0, learned_parameters=learned_parameters, *args, **kwargs)
            molecule.dm = P.detach()
            
            # self.force, P, self.Etot, self.Hf, self.Eelec, self.Enuc, self.Eiso, self.EnucAB, self.e_gap, self.e_mo, self.charge, self.notconverged = \
            #             self.conservative_force(molecule, P0=P0, learned_parameters=learned_parameters, *args, **kwargs)
            # self.P = P.detach()
            
        elif dm_prop=='XL-BOMD':
            molecule.force, molecule.dm, molecule.Hf, molecule.Etot, molecule.Eelec, molecule.Enuc, molecule.Eiso, molecule.e_mo, molecule.e_gap =\
                        self.conservative_force_xl(molecule, P=P0, learned_parameters=learned_parameters, *args, **kwargs)
            
            # self.force, self.Hf, P, self.e_gap, self.e_mo = self.conservative_force_xl(molecule, P=P0, learned_parameters=learned_parameters, *args, **kwargs)
            # self.P = P.detach()
        
        elif dm_prop=='XL-BOMD-LR':
            molecule.force, molecule.dm, molecule.Hf, molecule.Etot, molecule.Eelec, molecule.Enuc, molecule.Eiso, molecule.e_mo, molecule.e_gap, \
            molecule.Electronic_entropy, molecule.dP2dt2, molecule.Krylov_Error,  molecule.Fermi_occ = \
                        self.conservative_force_xl_lr(molecule, P0, err_threshold, rank, T_el, learned_parameters, *args, **kwargs)
            
            # self.force, self.Hf, self.El_Ent, P, self.dP2dt2, self.Error, self.e_gap, self.e_mo, self.Fermi_occ = self.conservative_force_xl_lr(molecule, P0, err_threshold, rank, T_el, learned_parameters, *args, **kwargs)
            # self.P = P.detach()


        #return F, P, L

    def get_force(self):
        return self.force

    def get_dm(self):
        return self.P

    def get_Hf(self):
        return self.Hf
    
    def get_Electronic_entropy(self):
        return self.El_Ent
    
    def get_dP2dt2(self):
        return self.dP2dt2

    def get_Krylov_Error(self):
        return self.Error

    def get_e_gap(self):
        return self.e_gap
    
    def get_e_mo(self):
        return self.e_mo

    def get_Fermi_occ(self):
        return self.Fermi_occ

    # def get_charger(self, xx):

    #     return charge


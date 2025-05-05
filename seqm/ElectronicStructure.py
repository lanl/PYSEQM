import torch
from .basics import *
import time
from seqm.XLBOMD import ForceXL
#from seqm.XLBOMD_LR import ForceXL as ForceXL_lr

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
        #self.conservative_force_xl_lr = ForceXL_lr(self.seqm_parameters)


        #self.acc_scale = 0.009648532800137615
        #self.output = output
    
    @staticmethod
    def atomic_charges(P, n_orbital=4):
        """
        get atomic charge based on single-particle density matrix P
        n_orbital : number of orbitals for each atom, default is 4
        """
        n_molecule = P.shape[0]
        n_atom = P.shape[1]//n_orbital
        q = P.diagonal(dim1=1,dim2=2).reshape(n_molecule, n_atom, n_orbital).sum(axis=2)
        return q
    
    @staticmethod
    def dipole(q, coordinates):
        return torch.sum(q.unsqueeze(2)*coordinates, axis=1)

    def forward(self, molecule, learned_parameters=dict(), xl_bomd_params=dict(), P0=None, err_threshold = None, max_rank = None, T_el = None, dm_prop='SCF', *args, **kwargs):
        """
        return force in unit of eV/Angstrom
        return force, density matrix, total energy of this batch
        """
        if dm_prop=='SCF':
            molecule.force, P, molecule.Hf, molecule.Etot, molecule.Eelec, molecule.Enuc, molecule.Eiso, molecule.e_mo, molecule.e_gap, self.charge, self.notconverged = \
                        self.conservative_force(molecule, P0=P0, learned_parameters=learned_parameters, *args, **kwargs)
            molecule.dm = P.detach()
            
        elif dm_prop=='XL-BOMD':
            molecule.force, molecule.dm, molecule.Hf, molecule.Etot, molecule.Eelec, molecule.Enuc, molecule.Eiso, molecule.e_mo, molecule.e_gap, \
            molecule.Electronic_entropy, molecule.dP2dt2, molecule.Krylov_Error,  molecule.Fermi_occ = \
                        self.conservative_force_xl(molecule, P0, learned_parameters=learned_parameters, xl_bomd_params=xl_bomd_params, *args, **kwargs)

        with torch.no_grad():
            # $$$
            if molecule.dm.dim() ==4: # open shell
                if molecule.method == 'PM6':
                    molecule.q = molecule.const.tore[molecule.species] - self.atomic_charges(molecule.dm[:,0], n_orbital=9)
                    molecule.q -= self.atomic_charges(molecule.dm[:,1], n_orbital=9) # unit +e, i.e. electron: -1.0
                else:
                    molecule.q = molecule.const.tore[molecule.species] - self.atomic_charges(molecule.dm[:,0])
                    molecule.q -= self.atomic_charges(molecule.dm[:,1]) # unit +e, i.e. electron: -1.0
            else:  # closed shell
                if molecule.method == 'PM6':
                    molecule.q = molecule.const.tore[molecule.species] - self.atomic_charges(molecule.dm, n_orbital=9) # unit +e, i.e. electron: -1.0
                else:
                    molecule.q = molecule.const.tore[molecule.species] - self.atomic_charges(molecule.dm) # unit +e, i.e. electron: -1.0

                molecule.d = self.dipole(molecule.q, molecule.coordinates)
 
            



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


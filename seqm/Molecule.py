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

        check_input(species)
        self.species = species

        #self.coordinates = coordinates
        self.coordinates = torch.nn.Parameter(coordinates)
        self.coordinates.requires_grad_(do_large_tensors)
        if not torch.is_tensor(charges):
            charges = charges * torch.ones(coordinates.size()[0], device=coordinates.device)
        self.tot_charge = charges
        if not torch.is_tensor(mult):
            mult = mult * torch.ones(coordinates.size()[0], device=coordinates.device)
        self.mult = mult
        
        # Previously 'elements' was a user input list of unique elements in the input molecules.
        # However, it didnt make sense for the user to input this list since this list can be easily calculated,
        # thus reducing the number of things a user has to input
        if seqm_parameters.get('elements') is None:
            seqm_parameters['elements'] = [0]+sorted(set(species.reshape(-1).tolist()))
        self.seqm_parameters = seqm_parameters
        self.method = seqm_parameters['method']
        
        self.parser = Parser(self.seqm_parameters)
        self.packpar = Pack_Parameters(self.seqm_parameters).to(coordinates.device)
        
        self.nmol, self.molsize, \
        self.nSuperHeavy, self.nHeavy, self.nHydro, self.nocc, \
        self.Z, self.maskd, self.atom_molid, \
        self.mask, self.mask_l, self.pair_molid, \
        self.ni, self.nj, self.idxi, self.idxj, self.xij, self.rij = self.parser(self, self.method, return_mask_l=True, do_large_tensors=do_large_tensors, *args, **kwargs)

        if callable(learned_parameters):
            adict = learned_parameters(self.species, self.coordinates)
            self.parameters, self.alp, self.chi = copy.deepcopy(self.packpar(self.Z, learned_params = adict)  )
        else:
            self.parameters, self.alp, self.chi = copy.deepcopy(self.packpar(self.Z, learned_params = learned_parameters))

        self.norb = self.nHydro + 4 * self.nHeavy # number of orbitals

        
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

        # MASS = torch.as_tensor(self.const.mass)
        # # put the padding virtual atom mass finite as for accelaration, F/m evaluation.
        # MASS[0] = 1.0
        non_zero_species = self.species != 0
        self.num_atoms = (torch.sum(non_zero_species, dim=1)).to(coordinates.dtype).to(coordinates.device)

        self.mass = self.const.mass[self.species].unsqueeze(2)
        self.mass_inverse = torch.zeros_like(self.mass)
        self.mass_inverse[non_zero_species] = 1.0/self.mass[non_zero_species] 
        
        self.force = None
        self.velocities = None
        self.acc = None
        
        self.dm = None
        self.q = None
        
        self.Hf = None
        self.Etot = None
        self.Eelec = None
        self.Enuc = None
        self.Eiso = None
        self.e_mo = None #
        self.e_gap = None #
        
        self.charge = None
        self.dipole = None

        self.verbose = True
        
        self.Electronic_entropy = None
        self.Fermi_occ = None
        self.dP2dt2 = None
        self.Krylov_Error = None

        self.analytical_gradient = None
        self.active_state = seqm_parameters.get('active_state',0)
        self.cis_amplitudes = None
        self.cis_energies = None
        self.molecular_orbitals = None

        self.all_forces = None
        self.all_nac = None
        self.all_cis_relaxed_diploles = None
        self.all_cis_unrelaxed_diploles = None

        self.old_mos = None

        self.transition_dipole = None
        self.oscillator_strength = None
        self.cis_state_unrelaxed_dipole = None
        self.cis_state_relaxed_dipole = None
        
                
        def get_coordinates(self):
            return self.coordinates

        def get_species(self):
            return self.species


def check_input(species):
    """
    Does input checking
    The species (atomic numbers of molecules) tensor should have non-increasing rows so that elements are sorted from biggest to smallest
    """
        
    ok = species[:, :-1] >= species[:, 1:]       # compares each element in a row to the one to its right
    # now see which rows are *all* True
    row_ok = ok.all(dim=1)

    if not row_ok.all():
        # gather the indices of the offending rows
        bad = (~row_ok).nonzero(as_tuple=False).squeeze(1).tolist()
        rows = ", ".join(map(str, bad))
        # pick correct plural forms
        row_word = "row" if len(bad) == 1 else "rows"
        verb     = "is"  if len(bad) == 1 else "are"
        msg = (
            f"species must be non-increasing along each row, "
            f"but {row_word} {rows} {verb} not sorted."
        )
        raise ValueError(msg)

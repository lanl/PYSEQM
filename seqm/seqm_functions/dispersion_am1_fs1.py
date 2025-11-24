import os
import math
import torch
from .constants import a0

# AM1-FS1 constants
S_R = 1.1058892
d = 1000.0

EV_PER_ATOM_PER_J_PER_MOL = 1.036426966e-5
D_TOL = 12.0 * math.log(10.0)    # tolerance for the exponential argument for the damping function

def dispersion_am1_fs1(mol, P):
    # Dispersion corrected AM1 method called AM1-FS1
    # Foster, Michael E., and Karl Sohlberg. "A new empirical correction to the AM1 method for macromolecular complexes." Journal of chemical theory and computation 6.7 (2010): 2153-2166.
    # https://doi.org/10.1021/ct100177u
    # There is a small discrpency in how the vdw radii are calculated as claimed by the authors in their paper and in their implementation. We follow their implementation
    # in order to match the results given in their paper.
    # The Fortran implementation of AM1-FS1 can be found in Appendix F of the PhD thesis of Michael E. Foster:
    # Foster, M. E. (2011). The development of an empirically corrected semi-empirical method and its
    # application to macromolecular complexes [Drexel University]. https://doi.org/10.17918/etd-3517
    # Downloaded On 2025/02/24 14:20:30 -0500

    f_damp, C6ij = dispersion_damping(mol)

    # E_disp = \sum_ij (i<j) -C_6^ij/r^6*f_damp(r_ij)
    E_disp_pair = -C6ij * torch.pow(a0 * mol.rij, -6.0) * f_damp  # J/mol nm^6/Ang^6

    E_disp = torch.zeros((mol.nmol, ), dtype=mol.rij.dtype, device=mol.rij.device)
    E_disp.index_add_(0, mol.pair_molid, E_disp_pair)
    E_disp = E_disp * EV_PER_ATOM_PER_J_PER_MOL * 1e6  # nm^6/Ang^6 = 1e6; now the energy is in eV/atom

    # # H-bonding correction depends on density matrix P, so it should be included in the SCF cycle. Not doing this for now
    # E_hbonding = hbond_correction_am1fs1(mol, P, R_van)
    # E_disp = E_disp + E_hbonding

    # print(f'Dispersion correction + H-Bonding correction to the total energy is {E_disp}')

    return E_disp

def dispersion_damping(mol, get_grad_factor=False):
    # E_disp = \sum_ij (i<j) -C_6^ij/r^6*f_damp(r_ij)
    # where f_damp(r_ij) = 1/(1+exp(-d(r_ij/(S_R*R_vdw)-1)))
    # C_6^ij = sqrt(C_6^i C_6^j) and R_vdw = (R_i + R_j)

    # get C6 parameters
    # C6 is in J nm^6/mol, R_van is in Ang
    C6ij, R_vdw = get_c6_r0_params(mol, mol.seqm_parameters['elements'])

    exp_arg = d * (a0 * mol.rij / (S_R * R_vdw) - 1.0)
    # f_damp = 1.0 / (1.0 + torch.exp(-exp_arg))
    f_damp = torch.sigmoid(exp_arg)
    # The original AM1-FS1 code corrects the f_damp function for numerical extreme values
    # clip in saturated regions 
    f_damp = torch.where(exp_arg >  D_TOL, torch.ones_like(f_damp),  f_damp)
    f_damp = torch.where(exp_arg < -D_TOL, torch.zeros_like(f_damp), f_damp)

    # f_damp[exp_arg > d_tol] = 1.0
    # f_damp[exp_arg < -d_tol] = 0.0
    if not get_grad_factor:
        return f_damp, C6ij

    alpha = d * a0 / (S_R * R_vdw)
    # (masking alpha is not needed since the derivative f*(1-f)=0 where saturated)
    return f_damp, C6ij, alpha


def get_c6_r0_params(mol, elements):
    # Parameters taken from Grimme, S. Semiempirical GGA-Type Density Functional Constructed with a Long-Range Dispersion Correction. J. Com- put. Chem. 2006, 27, 1787–1799.
    file_path = os.path.join(os.path.dirname(__file__), "../params/grimme_2006_b97-d.csv")

    m = max(elements)
    C_6 = torch.zeros(m + 1, device=mol.rij.device,dtype=mol.rij.dtype)  # m+1 because indexing starts from 1 for atomic number
    R_0 = torch.zeros(m + 1, device=mol.rij.device,dtype=mol.rij.dtype)

    # Open file and read line by line
    with open(file_path, "r") as f:
        _ = f.readline()  # Read the header line

        for line in f:
            values = line.strip().replace(' ', '').split(",")  # Split CSV row
            at_no = int(values[0])  # Convert at_no to int

            if at_no in elements:  # Check if at_no is in the target set
                C_6[at_no] = float(values[2])  # Store C6 directly
                R_0[at_no] = float(values[3])  # Store R0 directly

    C6ij = torch.sqrt(C_6[mol.ni] * C_6[mol.nj])     # J nm^6 / mol

    # get van der Walls radii R_van
    # Angstrom
    R_vdw = (
        R_0[mol.ni] + R_0[mol.nj]
    )  # The original paper claims that R_vdw = 0.5*(Ri + Rj), but in their actual implementation R_vdw = Ri + Rj
    return C6ij, R_vdw

def dEdisp_dr(mol):
    """
    dE/dr = 6*C*f/r^7 - (C/r^6) * (d/(S*Rvdw)) * f*(1-f)
    Uses the same clipped f(r) for numerical stability.
    """
    f_damp, C6ij, alpha = dispersion_damping(mol,get_grad_factor=True)

    r = mol.rij
    inv_r7 = torch.pow(a0*r,-7.0)

    bracket = 6.0 - alpha * r * (1.0 - f_damp)  # well-behaved: (1-f)=0 in tails
    dE_pair = C6ij * f_damp * inv_r7 * bracket * 1e6 * EV_PER_ATOM_PER_J_PER_MOL  # convert to eV
    return -dE_pair.unsqueeze(1)*mol.xij # negative because xij is xj-xi


def hbond_correction_am1fs1(mol, P, R_van):
    # Works only for a single molecule.
    # Need to vectorize for a batch of molecules instead of looping
    E_hbonding = torch.empty(mol.nmol,device=mol.coordinates.device,dtype=mol.coordinates.dtype)
    for i in range(mol.nmol):
        species = mol.species[i]
        coords = mol.coordinates[i]

        real_atoms = species > 0
        species = species[real_atoms]
        coords = coords[real_atoms]

        # Step 1. Identify hydrogen and heavy atoms (O, N, F)
        hydrogen_mask = (species == 1)
        heavy_mask = (species == 7) | (species == 8) | (species == 9)
        hydrogen_indices = torch.where(hydrogen_mask)[0]
        heavy_indices = torch.where(heavy_mask)[0]

        # Step 2. For each hydrogen, find its nearest neighbor among all atoms.
        hydrogen_coords = coords[hydrogen_indices]  # shape: (n_H, 3)
        distances = torch.cdist(hydrogen_coords, coords)  # shape: (n_H, N)

        # Exclude self-distance:
        rows = torch.arange(hydrogen_coords.shape[0])
        distances[rows, hydrogen_indices] = float('inf')

        # For each hydrogen, get the index of its nearest neighbor
        _, nn_indices = distances.min(dim=1)
        nn_species = species[nn_indices]
        # Select hydrogens whose nearest neighbor is O, N, or F.
        sel_mask = (nn_species == 7) | (nn_species == 8) | (nn_species == 9)
        selected_hydrogen_indices = hydrogen_indices[sel_mask]
        selected_nn_indices = nn_indices[sel_mask]

        # Step 3. Compute connecting vectors: from hydrogen to its nearest heavy neighbor.
        vector_H_to_nn = coords[selected_nn_indices] - coords[selected_hydrogen_indices]

        # Step 4. Compute vectors from each selected hydrogen to all heavy atoms.
        sel_h_coords = coords[selected_hydrogen_indices]  # shape: (n_sel, 3)
        heavy_coords = coords[heavy_indices]  # shape: (n_heavy, 3)
        all_vectors = heavy_coords.unsqueeze(0) - sel_h_coords.unsqueeze(1)  # shape: (n_sel, n_heavy, 3)

        # Exclude the nearest neighbor from each list.
        # Create a mask comparing heavy_indices (broadcasted) with selected_nn_indices.
        mask_exclude_nn = heavy_indices.unsqueeze(0) != selected_nn_indices.unsqueeze(1)
        all_vectors = all_vectors[mask_exclude_nn].view(sel_h_coords.shape[0], -1, 3)
        # Record the corresponding heavy atom indices.
        heavy_excl_nn = heavy_indices.unsqueeze(0).expand(sel_h_coords.shape[0], -1)
        heavy_excl_nn = heavy_excl_nn[mask_exclude_nn].view(sel_h_coords.shape[0], -1)

        # Step 5. Compute cosine similarities between each hydrogen's nearest neighbor vector and
        # its vectors to the other heavy atoms.
        nn_norm = vector_H_to_nn / vector_H_to_nn.norm(dim=-1, keepdim=True)
        other_norm = all_vectors / all_vectors.norm(dim=-1, keepdim=True)
        cos_angles = (nn_norm.unsqueeze(1) * other_norm).sum(dim=-1)

        # Filter for angles greater than 90° (cosine < 0)
        angle_mask = cos_angles < 0

        # Flatten filtered heavy indices and vectors.
        filtered_heavy_indices = heavy_excl_nn[angle_mask]

        # Also retrieve the corresponding hydrogen indices.
        h_expanded = selected_hydrogen_indices.unsqueeze(1).expand_as(heavy_excl_nn)
        filtered_hydrogen_indices = h_expanded[angle_mask]

        # Step 6. Compute distances between the corresponding hydrogen and heavy atoms.
        h_coords_filtered = coords[filtered_hydrogen_indices]
        heavy_coords_filtered = coords[filtered_heavy_indices]
        pair_vectors = heavy_coords_filtered - h_coords_filtered
        rij = pair_vectors.norm(dim=1)

        # Get atomic charges
        natoms = real_atoms.sum()
        atomic_charge = mol.const.tore[species] - P[i,:4*natoms,:4*natoms].diagonal().reshape(natoms, -1).sum(dim=1)

        bohr_to_ang = 0.52917724924
        ha_to_eV = 27.2114079527
        alpha1, alpha2, alpha3, alpha4 = 0.4882, 0.6211, 0.3344, 1.5451

        # Determine the species for the heavy atoms in the filtered list.
        heavy_species = species[filtered_heavy_indices]

        # Compute van der Waals radius (R_vdw) for heavy atoms and hydrogen pairs. This is multiplied by 2 because each of the R_van have to be multiplied by 2 (idk why!)
        R_vdw = 2.0 * (R_van[heavy_species]**3 + R_van[1].unsqueeze(0)**3) / (R_van[heavy_species]**2 +
                                                                              R_van[1].unsqueeze(0)**2)

        # Compute the damping function.
        f_damp = torch.exp(-((rij - alpha2 * R_vdw)**2) / (alpha3 * (bohr_to_ang + alpha4 * (rij - alpha2 * R_vdw)))**2)

        # Calculate the hydrogen bonding correction energy.
        E_hbonding[i] = ha_to_eV * alpha1 * torch.sum(
            atomic_charge[filtered_heavy_indices] * atomic_charge[filtered_hydrogen_indices] / rij * bohr_to_ang *
            (cos_angles[angle_mask]**2) * f_damp)

    return E_hbonding


def pair_disp(mol):
    f_damp, C6ij = dispersion_damping(mol)

    # E_disp = \sum_ij (i<j) -C_6^ij/r^6*f_damp(r_ij)
    E_disp_pair = -C6ij * torch.pow(a0 * mol.rij, -6.0) * f_damp * EV_PER_ATOM_PER_J_PER_MOL * 1e6  # nm^6/Ang^6 = 1e6; now the energy is in eV/atom  # J/mol nm^6/Ang^6
    E_disp = torch.zeros((mol.nmol, ), dtype=mol.rij.dtype, device=mol.rij.device)
    E_disp.index_add_(0, mol.pair_molid, E_disp_pair)
    print(f'Grad: Dispersion correction + H-Bonding correction to the total energy is {E_disp}')
    return E_disp_pair

delta = 1e-5 # delta for finite difference calcs
def numericaldE(mol):
    dtype = mol.rij.dtype
    device = mol.rij.device
    xij= mol.xij
    rij=mol.rij
    Xij = xij * rij.unsqueeze(1) * a0
    npairs = Xij.shape[0]

    pair_grad = torch.zeros((npairs,3),dtype=dtype, device=device)

    for coord in range(3):
        for t in ('+','-'):
            # since Xij = Xj-Xi, when I want to do Xi+delta, I have to subtract delta from from Xij
            if t=="-":
                Xij[:, coord] -= delta
                rij_= torch.norm(Xij, dim=1)
                rij_= rij_/ a0
            else:
                Xij[:, coord] += delta
                rij_= torch.norm(Xij, dim=1)
                rij_= rij_/ a0

            old_rij = mol.rij
            mol.rij = rij_
            diff = pair_disp(mol)
            mol.rij = old_rij
            if t=="-":
                pair_grad[:, coord] -= diff 
            else:
                pair_grad[:, coord] += diff 
            Xij = xij * rij.unsqueeze(1) * a0

    return pair_grad/ (2.0 * delta)

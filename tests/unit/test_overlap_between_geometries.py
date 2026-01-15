import torch

from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants, overlap_cutoff
from seqm.seqm_functions.diat_overlap_PM6_SP import diatom_overlap_matrix_PM6_SP
from seqm.seqm_functions.diat_overlapD import diatom_overlap_matrixD
from seqm.seqm_functions.hcore import overlap_between_geometries


def _build_full_overlap(mol):
    """Build the full AO overlap matrix (both triangles + diagonal) for a Molecule."""
    is_pm6 = mol.method == "PM6"
    orb_dim = 9 if is_pm6 else 4
    if is_pm6:
        overlap_fn = diatom_overlap_matrixD
        overlap_args = (mol.const.qn_int, mol.const.qnD_int)
        zeta_fields = ["zeta_s", "zeta_p", "zeta_d"]
    else:
        overlap_fn = diatom_overlap_matrix_PM6_SP
        overlap_args = (mol.const.qn_int,)
        zeta_fields = ["zeta_s", "zeta_p"]

    zeta = torch.stack([mol.parameters[f] for f in zeta_fields], dim=1)

    di = torch.zeros((mol.xij.size(0), orb_dim, orb_dim), dtype=mol.xij.dtype, device=mol.xij.device)
    mask_ov = mol.rij <= overlap_cutoff
    if mask_ov.any():
        ni = mol.ni[mask_ov]
        nj = mol.nj[mask_ov]
        x_flat = mol.xij[mask_ov]
        r_flat = mol.rij[mask_ov]
        zeta_i = zeta[mol.idxi[mask_ov]]
        zeta_j = zeta[mol.idxj[mask_ov]]
        di[mask_ov] = overlap_fn(ni, nj, x_flat, r_flat, zeta_i, zeta_j, *overlap_args)

    di_full = torch.zeros(
        (mol.nmol * mol.molsize * mol.molsize, orb_dim, orb_dim), dtype=mol.xij.dtype, device=mol.xij.device
    )
    mask_H = mol.Z == 1
    mask_heavy = mol.Z > 1
    if mask_H.any():
        H_self = torch.zeros((orb_dim, orb_dim), dtype=mol.xij.dtype, device=mol.xij.device)
        H_self[0, 0] = 1.0
        di_full[mol.maskd[mask_H]] = H_self
    if mask_heavy.any():
        di_full[mol.maskd[mask_heavy]] = torch.eye(orb_dim, dtype=mol.xij.dtype, device=mol.xij.device)

    di_full[mol.mask] = di
    di_full[mol.mask_l] = di.transpose(1, 2)

    return (
        di_full.reshape(mol.nmol, mol.molsize, mol.molsize, orb_dim, orb_dim)
        .transpose(2, 3)
        .reshape(mol.nmol, orb_dim * mol.molsize, orb_dim * mol.molsize)
    )


def test_overlap_between_geometries_matches_block(device):
    torch.set_default_dtype(torch.float64)
    const = Constants().to(device)

    # Base H2 geometry and a slightly translated copy
    species = torch.tensor([[1, 1]], dtype=torch.int64, device=device)
    coords1 = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]], device=device)
    shift = torch.tensor([0.1, -0.05, 0.02], device=device)
    coords2 = coords1 + shift

    seqm_parameters = {"method": "PM6_SP", "scf_eps": 1.0e-6, "scf_converger": [1]}

    base_mol = Molecule(const, seqm_parameters, coords1.clone(), species.clone()).to(device)
    s_cross = overlap_between_geometries(base_mol, coords1, coords2)

    # Double the molecule to contain both geometries in one system
    doubled_species = torch.tensor([[1, 1, 1, 1]], dtype=torch.int64, device=device)
    doubled_coords = torch.cat([coords1, coords2], dim=1)
    doubled_mol = Molecule(const, seqm_parameters, doubled_coords.clone(), doubled_species.clone()).to(device)

    full_overlap = _build_full_overlap(doubled_mol)[0]
    orb_dim = 4  # PM6_SP
    block = orb_dim * species.shape[1]
    s_block = full_overlap[:block, block : 2 * block]

    torch.testing.assert_close(s_cross[0], s_block, rtol=1e-8, atol=1e-8)


def test_overlap_between_geometries_batch_mixed_sizes(device):
    torch.set_default_dtype(torch.float64)
    const = Constants().to(device)

    # Two molecules: water (3 atoms) and methanal (4 atoms), padded to molsize=4
    species = torch.tensor([[8, 1, 1, 0], [8, 6, 1, 1]], dtype=torch.int64, device=device)
    coords1 = torch.tensor(
        [
            [
                [0.0000, 0.0000, 0.0000],  # O
                [0.9584, 0.0000, 0.0000],  # H
                [-0.2390, 0.9270, 0.0000],  # H
                [0.0000, 0.0000, 0.0000],  # padding
            ],
            [
                [-0.00104, -0.00028, 0.00000],  # O
                [1.20966, -0.00003, -0.00000],  # C
                [1.63293, 0.95572, 0.00000],  # H
                [1.82758, -0.85100, -0.00000],  # H
            ],
        ],
        device=device,
    )
    shifts = torch.tensor([[0.05, -0.02, 0.01], [-0.10, 0.05, -0.03]], device=device)
    coords2 = coords1.clone()
    coords2[0, :3] += shifts[0]
    coords2[1, :4] += shifts[1]

    seqm_parameters = {"method": "PM6_SP", "scf_eps": 1.0e-6, "scf_converger": [1]}

    base_mol = Molecule(const, seqm_parameters, coords1.clone(), species.clone()).to(device)
    s_cross = overlap_between_geometries(base_mol, coords1, coords2)

    doubled_species = torch.cat([species, species], dim=1)
    doubled_coords = torch.cat([coords1, coords2], dim=1)
    # sort atoms to descending species to satisfy Molecule, keep track of which geometry each atom came from
    perms = torch.argsort(doubled_species, dim=1, descending=True)
    species_sorted = torch.gather(doubled_species, 1, perms)
    coords_sorted = torch.gather(doubled_coords, 1, perms.unsqueeze(-1).expand(-1, -1, 3))

    doubled_mol = Molecule(const, seqm_parameters, coords_sorted.clone(), species_sorted.clone()).to(device)

    full_overlap = _build_full_overlap(doubled_mol)
    orb_dim = 4  # PM6_SP

    total_atoms = doubled_species.size(1)
    inv_perm = torch.empty_like(perms)
    inv_perm.scatter_(1, perms, torch.arange(total_atoms, device=device).unsqueeze(0).expand_as(perms))
    orb_perm = (
        inv_perm.unsqueeze(2) * orb_dim + torch.arange(orb_dim, device=device).view(1, 1, -1)
    ).reshape(species.shape[0], -1)

    full_unperm = torch.stack(
        [full_overlap[i][orb_perm[i]][:, orb_perm[i]] for i in range(species.shape[0])], dim=0
    )
    block = orb_dim * species.shape[1]
    s_block = full_unperm[:, :block, block : 2 * block]

    torch.testing.assert_close(s_cross, s_block, rtol=1e-8, atol=1e-8)

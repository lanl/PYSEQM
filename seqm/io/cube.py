"""Cube-file export for MNDO-type SP CIS transition densities.

The NDDO Hamiltonian is expressed in an orthonormal valence-AO basis, whereas
the orbitals used for visualization are non-orthogonal Slater-type orbitals
(STOs).  :func:`cis_transition_density_ao` applies the symmetric Löwdin
transformation needed to express a CIS density in that STO basis before it is
evaluated on a cube grid.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from seqm.seqm_functions.constants import a0
from seqm.seqm_functions.hcore import orthogonalize_operator_from_overlap, overlap_matrix_current_geometry
from seqm.seqm_functions.rcis_batch import packone_batch


def cis_transition_density_ao(molecule, n_states: int | None = None) -> torch.Tensor:
    """Build singlet CIS transition densities in the non-orthogonal STO AO basis.

    The returned tensor has shape ``(n_states, nao, nao)`` and is suitable for
    real-space evaluation as ``sqrt(2) * phi.T @ R @ phi``.  ``molecule`` must
    hold a single restricted MNDO-type SP calculation with CIS amplitudes.
    """

    if molecule.method not in {"MNDO", "AM1", "PM3"}:
        raise NotImplementedError("STO cube export currently supports MNDO-type SP methods only.")
    if int(molecule.nmol) != 1:
        raise ValueError("Cube export accepts exactly one molecule at a time.")
    if molecule.cis_amplitudes is None or molecule.molecular_orbitals is None:
        raise ValueError("Run a CIS calculation before requesting transition densities.")

    nocc = int(molecule.nocc[0].item())
    norb = int(molecule.norb[0].item())
    amplitudes = molecule.cis_amplitudes[0]
    if n_states is not None:
        amplitudes = amplitudes[:n_states]
    nroots = amplitudes.shape[0]
    c = molecule.molecular_orbitals[0, :norb, :norb]
    cocc, cvirt = c[:, :nocc], c[:, nocc:]
    cis_amplitudes = amplitudes.reshape(nroots, nocc, norb - nocc)

    # R is in PYSEQM's orthonormal NDDO AO representation.  The STO functions
    # have metric S; chi = S^(-1/2) phi is the corresponding orthonormal AO
    # basis, so the density in the physical STO basis is X R X, X=S^(-1/2).
    density_nddo = torch.einsum("mi,ria,na->rmn", cocc, cis_amplitudes, cvirt)
    overlap = packone_batch(
        overlap_matrix_current_geometry(molecule),
        4 * int(molecule.nHeavy[0].item()),
        int(molecule.nHydro[0].item()),
        norb,
    )
    # Reuse the shared Löwdin operator transformation used by the dynamics
    # overlap machinery: it applies S^(-1/2) R S^(-1/2).
    return orthogonalize_operator_from_overlap(overlap, density_nddo)


def _sto_layout(molecule) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return centers, principal n, zeta, and Cartesian p-component for packed MNDO-type AOs."""

    species = molecule.species[0]
    coordinates_bohr = molecule.coordinates[0] / a0
    zeta_s = molecule.parameters["zeta_s"]
    zeta_p = molecule.parameters["zeta_p"]
    qn = molecule.const.qn_int

    centers, principal_n, zetas, components = [], [], [], []
    parameter_atom = 0
    for atom_index, z in enumerate(species.tolist()):
        if z == 0:
            continue
        center = coordinates_bohr[atom_index]
        n = int(qn[z].item())
        if z == 1:
            centers.append(center)
            principal_n.append(n)
            zetas.append(zeta_s[parameter_atom])
            components.append(-1)  # s
        else:
            centers.extend((center, center, center, center))
            principal_n.extend((n, n, n, n))
            zetas.extend(
                (
                    zeta_s[parameter_atom],
                    zeta_p[parameter_atom],
                    zeta_p[parameter_atom],
                    zeta_p[parameter_atom],
                )
            )
            components.extend((-1, 0, 1, 2))  # s, px, py, pz
        parameter_atom += 1

    return (
        torch.stack(centers),
        torch.tensor(principal_n, dtype=torch.long, device=coordinates_bohr.device),
        torch.stack(zetas),
        torch.tensor(components, dtype=torch.long, device=coordinates_bohr.device),
    )


def evaluate_mndo_sto_basis(molecule, points_bohr: torch.Tensor) -> torch.Tensor:
    """Evaluate the packed MNDO-type Slater basis on ``points_bohr``.

    The normalized functions are ``r**(n-1) exp(-zeta r) Y_00`` for s
    functions and their real Cartesian p counterparts.  AO ordering is the
    same packed ordering as the SCF/CIS coefficients: all heavy-atom
    ``[s, px, py, pz]`` blocks followed by hydrogen ``[s]`` functions.
    """

    centers, principal_n, zetas, components = _sto_layout(molecule)
    displacement = points_bohr[:, None, :] - centers[None, :, :]
    radius = torch.linalg.vector_norm(displacement, dim=-1)
    n = principal_n.to(dtype=points_bohr.dtype)
    zeta = zetas.to(dtype=points_bohr.dtype)

    # Normalized STO radial function times Y_00.  This form also avoids a
    # division by r at the center for p functions.
    s_normalization = torch.pow(2.0 * zeta, n + 0.5) / torch.sqrt(
        4.0 * math.pi * torch.exp(torch.lgamma(2.0 * n + 1.0))
    )
    radial = torch.pow(radius, (n - 1.0).unsqueeze(0)) * torch.exp(-radius * zeta.unsqueeze(0))
    values = radial * s_normalization.unsqueeze(0)

    is_p = components >= 0
    if torch.any(is_p):
        # sqrt(3) * x/r converts Y_00 into a real normalized p angular part;
        # the r**(n-1) radial factor then becomes x*r**(n-2).
        cartesian = displacement[:, is_p, :]
        values[:, is_p] *= (
            math.sqrt(3.0)
            * cartesian.gather(
                2, components[is_p].view(1, -1, 1).expand(points_bohr.shape[0], -1, 1)
            ).squeeze(-1)
            / radius[:, is_p].clamp_min(torch.finfo(points_bohr.dtype).tiny)
        )
    return values


def cube_grid(molecule, spacing_angstrom: float = 0.20, padding_angstrom: float = 3.0):
    """Construct an axis-aligned cube grid in bohr around a molecule."""

    if spacing_angstrom <= 0 or padding_angstrom < 0:
        raise ValueError("Cube spacing must be positive and padding cannot be negative.")
    coordinates_bohr = molecule.coordinates[0, molecule.species[0] > 0] / a0
    spacing = spacing_angstrom / a0
    padding = padding_angstrom / a0
    origin = coordinates_bohr.amin(dim=0) - padding
    upper = coordinates_bohr.amax(dim=0) + padding
    shape = torch.ceil((upper - origin) / spacing).to(dtype=torch.long) + 1
    return origin, (int(shape[0]), int(shape[1]), int(shape[2])), spacing


def write_cis_transition_density_cubes(
    molecule,
    output_directory: str | Path,
    n_states: int = 3,
    spacing_angstrom: float = 0.20,
    padding_angstrom: float = 3.0,
    chunk_size: int = 20_000,
) -> list[Path]:
    """Write signed, singlet CIS transition-density cube files for leading roots."""

    if n_states < 1 or chunk_size < 1:
        raise ValueError("n_states and chunk_size must be positive.")
    densities = cis_transition_density_ao(molecule, n_states=n_states).detach()
    origin, shape, spacing = cube_grid(molecule, spacing_angstrom, padding_angstrom)
    nx, ny, nz = shape
    x = torch.arange(nx, dtype=densities.dtype, device=densities.device)
    y = torch.arange(ny, dtype=densities.dtype, device=densities.device)
    z = torch.arange(nz, dtype=densities.dtype, device=densities.device)
    grid = torch.meshgrid(x, y, z, indexing="ij")
    points = origin.to(dtype=densities.dtype, device=densities.device) + spacing * torch.stack(
        grid, dim=-1
    ).reshape(-1, 3)

    values = torch.empty(
        (densities.shape[0], points.shape[0]), dtype=densities.dtype, device=densities.device
    )
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        basis = evaluate_mndo_sto_basis(molecule, points[start:stop])
        # The CIS amplitudes are spin-free spatial-orbital amplitudes.  The
        # physical singlet transition density carries sqrt(2).
        values[:, start:stop] = math.sqrt(2.0) * torch.einsum("pm,rmn,pn->rp", basis, densities, basis)

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    atomic_numbers = molecule.species[0, molecule.species[0] > 0].detach().cpu().tolist()
    coordinates = (molecule.coordinates[0, molecule.species[0] > 0] / a0).detach().cpu()
    files = []
    for state, cube_values in enumerate(values.detach().cpu(), start=1):
        path = output_directory / f"state_{state:02d}_transition_density.cube"
        _write_cube(
            path, atomic_numbers, coordinates, origin.detach().cpu(), shape, float(spacing), cube_values
        )
        files.append(path)
    return files


def _write_cube(
    path: Path, atomic_numbers, coordinates_bohr, origin_bohr, shape, spacing_bohr, values
) -> None:
    """Write one scalar field using the conventional Gaussian cube layout."""

    with path.open("w") as handle:
        handle.write("PYSEQM MNDO-type SP singlet CIS transition density (Löwdin-corrected STO basis)\n")
        handle.write("Units: electron / bohr^3; signed transition density\n")
        handle.write(
            f"{len(atomic_numbers):5d}{origin_bohr[0]:13.6f}{origin_bohr[1]:13.6f}{origin_bohr[2]:13.6f}\n"
        )
        for npoint, axis in zip(shape, range(3)):
            vector = [0.0, 0.0, 0.0]
            vector[axis] = spacing_bohr
            handle.write(f"{npoint:5d}{vector[0]:13.6f}{vector[1]:13.6f}{vector[2]:13.6f}\n")
        for atomic_number, coord in zip(atomic_numbers, coordinates_bohr):
            handle.write(
                f"{atomic_number:5d}{float(atomic_number):13.6f}{coord[0]:13.6f}{coord[1]:13.6f}{coord[2]:13.6f}\n"
            )
        for start in range(0, values.numel(), 6):
            handle.write(" ".join(f"{value:13.5e}" for value in values[start : start + 6].tolist()) + "\n")

"""Shared bookkeeping and utility functions for excited-state solvers."""

import psutil
import torch


def gather_new_subspace(V, vstart, vend, done=None):
    """Gather newly added rows, padding only when batch subspaces differ."""
    delta = vend - vstart
    if done is not None:
        delta[done] = 0
    max_v = int(delta.max().item())
    if max_v == 0:
        return None, None, None, None

    nmol = V.shape[0]
    rel_idx = torch.arange(max_v, device=V.device).unsqueeze(0)
    abs_idx = rel_idx + vstart.unsqueeze(1)
    batch_idx = torch.arange(nmol, device=V.device).unsqueeze(1).expand(-1, max_v)
    mask = rel_idx < delta.unsqueeze(1)
    if nmol == 1 or bool(torch.all(mask).item()):
        return V[batch_idx, abs_idx, :], batch_idx, abs_idx, None

    gathered = torch.zeros(nmol, max_v, V.shape[-1], dtype=V.dtype, device=V.device)
    gathered[mask] = V[batch_idx[mask], abs_idx[mask], :]
    return gathered, batch_idx, abs_idx, mask


def scatter_new_subspace(destination, values, batch_idx, abs_idx, mask):
    """Store a matrix action returned for ``gather_new_subspace`` rows."""
    if mask is None:
        destination[batch_idx, abs_idx, :] = values
    else:
        destination[batch_idx[mask], abs_idx[mask], :] = values[mask]


def update_subspace_status(
    done, roots_not_converged, vend, max_subspace_size, full_space_size, iteration, n_iters
):
    """Record convergence and identify subspaces that need collapse."""
    molecule_converged = roots_not_converged.sum(dim=1) == 0
    done_this_iteration = (~done) & molecule_converged
    done[done_this_iteration] = True
    n_iters[done_this_iteration] = iteration
    collapse_condition = (roots_not_converged.sum(dim=1) + vend > max_subspace_size) & (
        max_subspace_size != full_space_size
    )
    collapse_mask = (~done) & (~molecule_converged) & collapse_condition
    return molecule_converged, done_this_iteration, collapse_mask


def raise_max_iterations(
    done, roots_not_converged, residual_norm, vend, max_subspace_size, n_iters, n_collapses, nroots
):
    """Raise a consistent failure report for scalar or per-molecule roots."""
    info = []
    for molecule in range(done.shape[0]):
        target = int(nroots[molecule].item()) if isinstance(nroots, torch.Tensor) else nroots
        missing = roots_not_converged[molecule] & ~done[molecule]
        converged = target if bool(done[molecule].item()) else target - int(missing.sum().item())
        max_error = residual_norm[molecule, missing].max().item() if bool(missing.any().item()) else 0.0
        errors = ", ".join(f"{error:.2e}" for error in residual_norm[molecule].detach().cpu().tolist())
        info.append(
            f"mol {molecule}: {converged}/{target} roots converged, max remaining error {max_error:.3e}, "
            f"subspace {vend[molecule].item()}/{max_subspace_size}, errors [{errors}], "
            f"iters {n_iters[molecule].item()}, collapses {n_collapses[molecule].item()}"
        )
    raise RuntimeError("Maximum iterations reached but roots have not converged; " + "; ".join(info))


def _uniform_molecule_dimensions(mol):
    """Return immutable uniform-molecule dimensions without repeated CUDA syncs."""
    cached = getattr(mol, "_rcis_uniform_dimensions", None)
    if cached is not None:
        return cached

    dimensions = torch.stack((mol.nHeavy, mol.nHydro, mol.norb, mol.nocc), dim=1).cpu()
    if dimensions.shape[0] > 1 and not torch.equal(dimensions, dimensions[:1].expand_as(dimensions)):
        raise ValueError("All molecules in the batch must have the same number of orbitals and electrons")
    cached = tuple(map(int, dimensions[0]))
    mol._rcis_uniform_dimensions = cached
    return cached


def orthogonalize_to_current_subspace(V, newsubspace, vend, tol):
    """Append an orthonormal basis for corrections surviving projection."""

    V_old = V[:vend]
    W = newsubspace - (newsubspace @ V_old.T) @ V_old
    W = W - (W @ V_old.T) @ V_old
    W = W[torch.linalg.vector_norm(W, dim=1) > tol]
    if W.shape[0] == 0:
        return vend

    _, singular_values, right_vectors = torch.linalg.svd(W, full_matrices=False)
    Q_new = right_vectors[singular_values > tol]
    if Q_new.shape[0] == 0:
        return vend

    orth_atol = 1e-13 if Q_new.dtype == torch.float64 else 1e-6
    if (Q_new @ V_old.T).abs().max() > orth_atol:
        Q_new -= (Q_new @ V_old.T) @ V_old
        Q_new -= (Q_new @ V_old.T) @ V_old
        _, singular_values, right_vectors = torch.linalg.svd(Q_new, full_matrices=False)
        Q_new = right_vectors[singular_values > tol]

    n_new = Q_new.shape[0]
    if vend + n_new > V.shape[0]:
        raise ValueError(
            f"Not enough space in V: need {vend + n_new} rows, but V only has {V.shape[0]} rows."
        )
    V[vend : vend + n_new] = Q_new
    return vend + n_new


def getMaxSubspacesize(
    dtype, device, nov, nroots, nmol=1, num_big_matrices=2, memory_fraction=0.4, retry_memory_fraction=0.65
):
    """Return the largest memory-safe Davidson subspace dimension."""
    device = device.type
    bytes_per_element = torch.finfo(dtype).bits // 8

    def _candidate(fraction):
        if device == "cpu":
            available_memory = psutil.virtual_memory().available
        elif device == "cuda":
            available_memory, _ = torch.cuda.mem_get_info(device)
        else:
            raise ValueError("Unsupported device. Use 'cpu' or 'cuda'.")
        n_calculated = int(available_memory * fraction // (nov * nmol * bytes_per_element * num_big_matrices))
        return max(1, min(n_calculated, nov))

    max_subspace_size = _candidate(memory_fraction)
    min_required_subspace = min(nov, 3 * nroots)
    if max_subspace_size >= min_required_subspace:
        return max_subspace_size
    if device == "cuda":
        torch.cuda.empty_cache()
        max_subspace_size = _candidate(memory_fraction)
    if max_subspace_size >= min_required_subspace:
        return max_subspace_size

    retry_size = _candidate(retry_memory_fraction)
    if retry_size >= min_required_subspace:
        return retry_size
    raise RuntimeError(
        "Unable to allocate a Davidson subspace large enough for the requested roots. "
        f"Requested at least 3*nroots={3 * nroots}, got {max_subspace_size} at memory_fraction={memory_fraction:.2f} "
        f"and {retry_size} at memory_fraction={retry_memory_fraction:.2f} "
        f"with nov={nov}, nmol={nmol}, num_big_matrices={num_big_matrices}."
    )


def getMemUse(dtype, device, mol, nroots=1):
    """Estimate whether the two-electron matrix action must be root-chunked."""
    if device.type == "cpu":
        available_memory = psutil.virtual_memory().available
    elif device.type == "cuda":
        available_memory, _ = torch.cuda.mem_get_info(device)
    else:
        raise ValueError("Unsupported device type. Use 'cpu' or 'cuda'.")

    bytes_per_element = torch.finfo(dtype).bits // 8
    mem_per_root = 200.0 * mol.nmol * (mol.molsize**2) * bytes_per_element
    need_to_chunk = mem_per_root * nroots > available_memory
    chunk_size = max(1, min(nroots, int(available_memory // mem_per_root))) if need_to_chunk else nroots
    return need_to_chunk, chunk_size


def make_guess(ea_ei, nroots, maxSubspacesize, V, nmol, nov):
    """Populate a uniform-batch Davidson guess from the lowest orbital gaps."""
    sorted_ediff, sortedidx = torch.sort(ea_ei, stable=True, descending=False)
    nroots_expand = nroots
    while nroots_expand < len(sorted_ediff[0]) and torch.all(
        (sorted_ediff[:, nroots_expand] - sorted_ediff[:, nroots_expand - 1]) < 1e-4
    ):
        nroots_expand += 1
    if nroots_expand > nroots:
        print(
            f"Increasing the number of states calculated from {nroots} to {nroots_expand} because of orbital degeneracies"
        )
        nroots = nroots_expand

    extra_subspace = min(7, nov - nroots)
    extra_subspace = (
        extra_subspace
        if 2 * nroots + extra_subspace < maxSubspacesize
        else max(0, maxSubspacesize - 2 * nroots)
    )
    nstart = nroots + extra_subspace
    V[
        torch.arange(nmol, device=V.device).unsqueeze(1),
        torch.arange(nstart, device=V.device),
        sortedidx[:, :nstart],
    ] = 1.0
    return nstart, nroots


def get_occ_virt(mol, orbital_window=None, e_mo=None, molecular_orbitals=None):
    """Return occupied/virtual MO blocks for uniform or mixed molecule batches."""
    C = mol.molecular_orbitals if molecular_orbitals is None else molecular_orbitals
    nocc_b, norb_b = mol.nocc, mol.norb
    nvirt_b = norb_b - nocc_b
    nmol, nbasis = C.shape[:2]
    device, dtype = C.device, C.dtype

    try:
        _, _, norb, nocc = _uniform_molecule_dimensions(mol)
    except ValueError:
        uniform = False
    else:
        uniform = True

    if uniform:
        if orbital_window is not None:
            n_below, m_above = map(int, orbital_window)
            if not (0 <= n_below <= nocc and 0 <= m_above <= norb - nocc):
                raise ValueError("orbital_window out of bounds.")
            occ_start, occ_stop = nocc - n_below, nocc
            virt_start, virt_stop = nocc, nocc + m_above
        else:
            occ_start, occ_stop, virt_start, virt_stop = 0, nocc, nocc, norb
        Cocc = C[:, :, occ_start:occ_stop]
        Cvirt = C[:, :, virt_start:virt_stop]
        nocc, nvirt = occ_stop - occ_start, virt_stop - virt_start
        if e_mo is not None:
            ea_ei = e_mo[:, virt_start:virt_stop].unsqueeze(1) - e_mo[:, occ_start:occ_stop].unsqueeze(2)
            return nocc, nvirt, Cocc, Cvirt, ea_ei
        return nocc, nvirt, Cocc, Cvirt

    if orbital_window is not None:
        raise ValueError("orbital_window requires uniform nocc/norb across the batch.")
    nocc_max, nvirt_max = int(nocc_b.max().item()), int(nvirt_b.max().item())
    Cocc = torch.zeros((nmol, nbasis, nocc_max), device=device, dtype=dtype)
    Cvirt = torch.zeros((nmol, nbasis, nvirt_max), device=device, dtype=dtype)
    for i in range(nmol):
        nocc_i, norb_i = int(nocc_b[i].item()), int(norb_b[i].item())
        Cocc[i, :, :nocc_i] = C[i, :, :nocc_i]
        Cvirt[i, :, : nvirt_b[i]] = C[i, :, nocc_i:norb_i]

    if e_mo is not None:
        ea_ei = torch.zeros(nmol, nocc_max, nvirt_max, device=device, dtype=dtype)
        for i in range(nmol):
            nocc_i, norb_i = int(nocc_b[i].item()), int(norb_b[i].item())
            ea_ei[i, :nocc_i, : nvirt_b[i]] = e_mo[i, nocc_i:norb_i].unsqueeze(0) - e_mo[
                i, :nocc_i
            ].unsqueeze(1)
        return nocc_max, nvirt_max, Cocc, Cvirt, ea_ei
    return nocc_max, nvirt_max, Cocc, Cvirt


def print_rcis_analysis(excitation_energies, transition_dipole, oscillator_strength):
    """Print excitation energies, transition dipoles, and oscillator strengths."""
    print(f"Number of excited states: {excitation_energies.shape[1]}\n")
    print("Excitation energies E (eV), Transition dipoles d (au), and Oscillator strengths f (au)")
    row_format = "{:<10}   {:>10}   {:>10}   {:>10}      {:<10}"
    print(row_format.format("E", "d x", "d y", "d z", "f"))
    print("-" * 65)
    values = (
        excitation_energies.detach().cpu().tolist(),
        transition_dipole.detach().cpu().tolist(),
        oscillator_strength.detach().cpu().tolist(),
    )
    for mol_idx, (mol_energy, mol_dipole, mol_strength) in enumerate(zip(*values), start=1):
        if excitation_energies.shape[0] > 1:
            print(f"Molecule {mol_idx}:")
        for energy, (dx, dy, dz), strength in zip(mol_energy, mol_dipole, mol_strength):
            print(
                row_format.format(f"{energy:.6f}", f"{dx:.6f}", f"{dy:.6f}", f"{dz:.6f}", f"{strength:.6f}")
            )
        print("")

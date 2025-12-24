import math
from pathlib import Path

import h5py
import numpy as np
import torch

from seqm.seqm_functions.constants import Constants
from seqm.Molecule import Molecule
from seqm.MolecularDynamics import (
    Molecular_Dynamics_Basic,
    Molecular_Dynamics_Langevin,
    XL_BOMD,
    KSA_XL_BOMD,
)
from tests.reference_data import assert_allclose, load_or_update_reference, reference_path


def _output_config(prefix, molid):
    return {
        "molid": molid,
        "prefix": prefix,
        "print every": 0,
        "checkpoint every": 0,
        "xyz": 1,
        "h5": {
            "data": 1,
            "coordinates": 1,
            "velocities": 1,
            "forces": 1,
        },
    }


def _build_molecule(device, species, coordinates, seqm_parameters):
    const = Constants().to(device)
    return Molecule(const, seqm_parameters, coordinates, species).to(device)


def _run_md(md, molecule, steps=30):
    md.run(molecule, steps=steps, reuse_P=True, remove_com=None, seed=0)


def _h5_paths(prefix, molid):
    return [Path(f"{prefix}.{mol}.h5") for mol in molid]


def _xyz_paths(prefix, molid):
    return [Path(f"{prefix}.{mol}.xyz") for mol in molid]


def _read_xyz_frames(path):
    lines = path.read_text().splitlines()
    if not lines:
        return 0
    n_atoms = int(lines[0].strip())
    frame_lines = n_atoms + 2
    return len(lines) // frame_lines


def _read_xyz_first_coords(path):
    lines = path.read_text().splitlines()
    if not lines:
        return np.zeros((0, 3), dtype=float)
    n_atoms = int(lines[0].strip())
    coords = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coords, dtype=float)


def _read_h5_thermo(path):
    with h5py.File(path, "r") as h5:
        steps = h5["data/steps"][...]
        T = h5["data/thermo/T"][...]
        Ek = h5["data/thermo/Ek"][...]
        Ep = h5["data/thermo/Ep"][...]
    return steps, T, Ek, Ep


def _assert_output_files(prefix, molid, steps, expect_excited=False, n_states=None):
    for h5_path in _h5_paths(prefix, molid):
        assert h5_path.exists()
        with h5py.File(h5_path, "r") as h5:
            assert "data" in h5
            assert h5["data/steps"].shape[0] == steps
            assert h5["coordinates/values"].shape[0] == steps
            assert h5["velocities/values"].shape[0] == steps
            assert h5["forces/values"].shape[0] == steps
            if expect_excited:
                assert "excitation" in h5["data"]
                assert "excitation_energy" in h5["data/excitation"]
                assert "oscillator_strength" in h5["data/excitation"]
                if n_states is not None:
                    assert h5["data/excitation/excitation_energy"].shape[1] == n_states
                    assert h5["data/excitation/oscillator_strength"].shape[1] == n_states
    for xyz_path in _xyz_paths(prefix, molid):
        assert xyz_path.exists()
        assert _read_xyz_frames(xyz_path) == steps
        mol = int(xyz_path.stem.split(".")[-1])
        h5_path = Path(f"{prefix}.{mol}.h5")
        with h5py.File(h5_path, "r") as h5:
            h5_coords = h5["coordinates/values"][0]
        xyz_coords = _read_xyz_first_coords(xyz_path)
        assert_allclose(xyz_coords, h5_coords, rtol=5e-5, atol=5e-5)


def _metrics_from_thermo(T, Ek, Ep):
    Etot = Ek + Ep
    # drift = float(Etot[-1] - Etot[0])
    drift = np.max(np.abs(Etot-Etot[0]))
    if len(Etot) > 1:
        slope = float(np.polyfit(np.arange(len(Etot)), Etot, 1)[0])
    else:
        slope = 0.0
    return {
        "drift": drift,
        "slope": slope,
        "Etot": Etot.tolist(),
        "T": T.tolist(),
    }


def _assert_md_metrics(metrics, ref, tol_drift=1e-3, tol_slope=1e-3):
    assert_allclose(metrics["Etot"], ref["Etot"], rtol=1e-2, atol=1e-2)
    assert_allclose(metrics["T"], ref["T"], rtol=1e-1, atol=1e-1)
    assert math.isfinite(metrics["drift"])
    assert math.isfinite(metrics["slope"])
    assert abs(metrics["drift"]) <= tol_drift
    assert abs(metrics["slope"]) <= tol_slope


def _assert_md_bounds(metrics, tol_drift=1e-3, tol_slope=1e-3):
    assert math.isfinite(metrics["drift"])
    assert math.isfinite(metrics["slope"])
    assert abs(metrics["drift"]) <= tol_drift
    assert abs(metrics["slope"]) <= tol_slope


def _assert_temperature_target(metrics, target, tail_fraction=0.1, tol=50.0):
    T = np.array(metrics["T"], dtype=float)
    tail = T[-max(1, int(len(T) * tail_fraction)) :]
    tail_mean = float(tail.mean())
    tail_std = float(tail.std())
    assert abs(tail_mean - target) < tol
    assert tail_std > 1.0
    assert tail_std < target


def _run_and_check(md, molecule, prefix, molid, steps, expect_excited=False, n_states=None):
    _run_md(md, molecule, steps=steps)
    _assert_output_files(prefix, molid, steps, expect_excited=expect_excited, n_states=n_states)
    h5_path = Path(f"{prefix}.{molid[0]}.h5")
    _, T, Ek, Ep = _read_h5_thermo(h5_path)
    return _metrics_from_thermo(T, Ek, Ep)


def test_md_basic_single(tmp_path, device, methane_molecule_data):
    species, coordinates = methane_molecule_data
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
    }

    molid = [0]
    prefix = str(tmp_path / "md_basic_single")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = Molecular_Dynamics_Basic(
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid),
    ).to(device)

    metrics = _run_and_check(md, molecule, prefix, molid, steps=30)
    ref_path = reference_path("md_basic_methane")
    ref = load_or_update_reference(ref_path, metrics)
    _assert_md_metrics(metrics, ref, tol_drift=1e-2, tol_slope=1e-3)


def test_md_basic_batch_mixed(tmp_path, device, batch_molecule_data):
    species, coordinates = batch_molecule_data
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
    }

    molid = list(range(species.shape[0]))
    prefix = str(tmp_path / "md_basic_batch_mixed")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = Molecular_Dynamics_Basic(
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid),
    ).to(device)

    metrics = _run_and_check(md, molecule, prefix, molid, steps=30)
    ref_path = reference_path("md_basic_batch_mixed")
    ref = load_or_update_reference(ref_path, metrics)
    _assert_md_metrics(metrics, ref, tol_drift=1e-2, tol_slope=1e-3)


def test_md_langevin_single(tmp_path, device, methane_molecule_data):
    species, coordinates = methane_molecule_data
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
    }

    molid = [0]
    prefix = str(tmp_path / "md_langevin_single")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = Molecular_Dynamics_Langevin(
        damp=10.0,
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid),
    ).to(device)

    metrics = _run_and_check(md, molecule, prefix, molid, steps=100)
    _assert_temperature_target(metrics, 300.0, tail_fraction=0.1, tol=50.0)


def test_md_langevin_batch_mixed(tmp_path, device, batch_molecule_data):
    species, coordinates = batch_molecule_data
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
    }

    molid = list(range(species.shape[0]))
    prefix = str(tmp_path / "md_langevin_batch_mixed")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = Molecular_Dynamics_Langevin(
        damp=10.0,
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid),
    ).to(device)

    metrics = _run_and_check(md, molecule, prefix, molid, steps=100)
    _assert_temperature_target(metrics, 300.0, tail_fraction=0.1, tol=50.0)


def test_md_xl_bomd_single(tmp_path, device, methane_molecule_data):
    species, coordinates = methane_molecule_data
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
    }

    molid = [0]
    prefix = str(tmp_path / "md_xl_bomd_single")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = XL_BOMD(
        xl_bomd_params={"k": 6},
        damp=None,
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid),
    ).to(device)

    metrics = _run_and_check(md, molecule, prefix, molid, steps=30)
    ref_path = reference_path("md_xl_bomd_methane")
    ref = load_or_update_reference(ref_path, metrics)
    _assert_md_metrics(metrics, ref, tol_drift=5e-2, tol_slope=5e-3)


def test_md_xl_bomd_single_k4(tmp_path, device, methane_molecule_data):
    species, coordinates = methane_molecule_data
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
    }

    molid = [0]
    prefix = str(tmp_path / "md_xl_bomd_single_k4")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = XL_BOMD(
        xl_bomd_params={"k": 4},
        damp=None,
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid),
    ).to(device)

    metrics = _run_and_check(md, molecule, prefix, molid, steps=30)
    ref_path = reference_path("md_xl_bomd_methane_k4")
    ref = load_or_update_reference(ref_path, metrics)
    _assert_md_metrics(metrics, ref, tol_drift=5e-2, tol_slope=5e-3)


def test_md_ksa_xl_bomd_single(tmp_path, device, methane_molecule_data):
    species, coordinates = methane_molecule_data
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
    }

    molid = [0]
    prefix = str(tmp_path / "md_ksa_xl_bomd_single")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = KSA_XL_BOMD(
        xl_bomd_params={
            "k": 6,
            "max_rank": 3,
            "err_threshold": 0.0,
            "T_el": 1500,
        },
        damp=None,
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid),
    ).to(device)

    metrics = _run_and_check(md, molecule, prefix, molid, steps=30)
    ref_path = reference_path("md_ksa_xl_bomd_methane")
    ref = load_or_update_reference(ref_path, metrics)
    _assert_md_metrics(metrics, ref, tol_drift=5e-2, tol_slope=5e-3)


def test_md_ksa_xl_bomd_single_k4(tmp_path, device, methane_molecule_data):
    species, coordinates = methane_molecule_data
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
    }

    molid = [0]
    prefix = str(tmp_path / "md_ksa_xl_bomd_single_k4")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = KSA_XL_BOMD(
        xl_bomd_params={
            "k": 4,
            "max_rank": 3,
            "err_threshold": 0.0,
            "T_el": 1500,
        },
        damp=None,
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid),
    ).to(device)

    metrics = _run_and_check(md, molecule, prefix, molid, steps=30)
    ref_path = reference_path("md_ksa_xl_bomd_methane_k4")
    ref = load_or_update_reference(ref_path, metrics)
    _assert_md_metrics(metrics, ref, tol_drift=5e-2, tol_slope=5e-3)


def test_md_ksa_xl_bomd_batch_mixed(tmp_path, device, batch_molecule_data):
    species, coordinates = batch_molecule_data
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
    }

    molid = list(range(species.shape[0]))
    prefix = str(tmp_path / "md_ksa_xl_bomd_batch_mixed")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = KSA_XL_BOMD(
        xl_bomd_params={
            "k": 6,
            "max_rank": 3,
            "err_threshold": 0.0,
            "T_el": 1500,
        },
        damp=None,
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid),
    ).to(device)

    metrics = _run_and_check(md, molecule, prefix, molid, steps=30)
    ref_path = reference_path("md_ksa_xl_bomd_batch_mixed")
    ref = load_or_update_reference(ref_path, metrics)
    _assert_md_metrics(metrics, ref, tol_drift=5e-2, tol_slope=5e-3)


def test_md_excited_basic_batch(tmp_path, device, methanal_batch_data):
    species, coordinates = methanal_batch_data
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "excited_states": {"n_states": 4, "method": "cis"},
        "active_state": 1,
    }

    molid = list(range(species.shape[0]))
    prefix = str(tmp_path / "md_excited_basic_batch")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = Molecular_Dynamics_Basic(
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid),
    ).to(device)

    metrics = _run_and_check(md, molecule, prefix, molid, steps=30, expect_excited=True, n_states=4)
    ref_path = reference_path("md_excited_basic_batch_methanal")
    ref = load_or_update_reference(ref_path, metrics)
    _assert_md_metrics(metrics, ref, tol_drift=5e-2, tol_slope=5e-2)

def test_md_excited_langevin_batch(tmp_path, device, methanal_batch_data):
    species, coordinates = methanal_batch_data
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "excited_states": {"n_states": 4, "method": "cis"},
        "active_state": 1,
    }

    molid = list(range(species.shape[0]))
    prefix = str(tmp_path / "md_excited_langevin_batch")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = Molecular_Dynamics_Langevin(
        damp=10.0,
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid),
    ).to(device)

    metrics = _run_and_check(md, molecule, prefix, molid, steps=150, expect_excited=True, n_states=4)
    _assert_temperature_target(metrics, 300.0, tail_fraction=0.1, tol=50.0)


def test_md_excited_xl_bomd_batch(tmp_path, device, methanal_batch_data):
    species, coordinates = methanal_batch_data
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "excited_states": {"n_states": 4, "method": "cis"},
        "active_state": 1,
    }

    molid = list(range(species.shape[0]))
    prefix = str(tmp_path / "md_excited_xl_bomd_batch")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = XL_BOMD(
        xl_bomd_params={"k": 6},
        damp=None,
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid),
    ).to(device)

    metrics = _run_and_check(md, molecule, prefix, molid, steps=30, expect_excited=True, n_states=4)
    ref_path = reference_path("md_excited_xl_bomd_batch_methanal")
    ref = load_or_update_reference(ref_path, metrics)
    _assert_md_metrics(metrics, ref, tol_drift=5e-2, tol_slope=5e-3)

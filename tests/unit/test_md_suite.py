import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from seqm.MolecularDynamics import KSA_XL_BOMD, XL_BOMD, Molecular_Dynamics_Basic, Molecular_Dynamics_Langevin
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

from ..reference_data import (
    AM1_AND_OM2_METHODS,
    LANGEVIN_METHODS,
    assert_allclose,
    load_or_update_reference,
    reference_path_for_method,
)


def _output_config(prefix, molid):
    return {
        "molid": molid,
        "prefix": prefix,
        "print every": 0,
        "checkpoint every": 0,
        "xyz": 1,
        "h5": {"data": 1, "coordinates": 1, "velocities": 1, "forces": 1},
    }


def _build_molecule(device, species, coordinates, seqm_parameters):
    const = Constants().to(device)
    return Molecule(const, seqm_parameters, coordinates, species).to(device)


def _seqm_parameters(method, excited=False):
    params = {"method": method, "scf_eps": 1.0e-7, "scf_converger": [1], "torch_compile": False}
    if excited:
        params.update({"excited_states": {"n_states": 4, "method": "cis"}, "active_state": 1})
    return params


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
    expected_steps = steps + 1
    for h5_path in _h5_paths(prefix, molid):
        assert h5_path.exists()
        with h5py.File(h5_path, "r") as h5:
            assert "data" in h5
            assert h5["data/steps"].shape[0] == expected_steps
            assert h5["coordinates/values"].shape[0] == expected_steps
            assert h5["velocities/values"].shape[0] == expected_steps
            assert h5["forces/values"].shape[0] == expected_steps
            if expect_excited:
                assert "excitation" in h5["data"]
                assert "excitation_energy" not in h5["data/excitation"]
                assert "state_energies" in h5["data/excitation"]
                assert "transition_dipole" not in h5["data/excitation"]
                assert "oscillator_strength" not in h5["data/excitation"]
                if n_states is not None:
                    state = h5["data/excitation/state_energies"][...]
                    assert state.shape[1] == (n_states + 1)
                    assert np.isfinite(state[:, 0]).all()
                    rel = state[:, 1 : 1 + n_states] - state[:, [0]]
                    assert np.isfinite(rel).all()
    for xyz_path in _xyz_paths(prefix, molid):
        assert xyz_path.exists()
        assert _read_xyz_frames(xyz_path) == expected_steps
        mol = int(xyz_path.stem.split(".")[-1])
        h5_path = Path(f"{prefix}.{mol}.h5")
        with h5py.File(h5_path, "r") as h5:
            h5_coords = h5["coordinates/values"][0]
        xyz_coords = _read_xyz_first_coords(xyz_path)
        assert_allclose(xyz_coords, h5_coords, rtol=5e-5, atol=5e-5)


def _metrics_from_thermo(T, Ek, Ep):
    Etot = Ek + Ep
    drift = np.max(np.abs(Etot - Etot[0]))
    if len(Etot) > 1:
        slope = float(np.polyfit(np.arange(len(Etot)), Etot, 1)[0])
    else:
        slope = 0.0
    return {"drift": drift, "slope": slope, "Etot": Etot.tolist(), "Ek": Ek.tolist(), "T": T.tolist()}


def _assert_md_metrics(metrics, ref, tol_drift=1e-3, tol_slope=1e-3):
    assert_allclose(metrics["Etot"], ref["Etot"], rtol=1e-2, atol=1e-2)
    Ek = np.array(metrics["Ek"], dtype=float)
    assert np.isfinite(Ek).all()
    assert Ek.min() >= 0.0
    assert math.isfinite(metrics["drift"])
    assert math.isfinite(metrics["slope"])
    assert abs(metrics["drift"]) <= tol_drift
    assert abs(metrics["slope"]) <= tol_slope


def _assert_temperature_envelope(metrics, Tmin=10.0, Tmax=1500.0):
    T = np.array(metrics["T"], dtype=float)
    assert np.isfinite(T).all()
    assert T.min() >= Tmin
    assert T.max() <= Tmax


def _run_and_check(md, molecule, prefix, molid, steps, expect_excited=False, n_states=None):
    _run_md(md, molecule, steps=steps)
    _assert_output_files(prefix, molid, steps, expect_excited=expect_excited, n_states=n_states)
    metrics_by_mol = []
    for mol in molid:
        h5_path = Path(f"{prefix}.{mol}.h5")
        _, T, Ek, Ep = _read_h5_thermo(h5_path)
        metrics_by_mol.append(_metrics_from_thermo(T, Ek, Ep))
    return metrics_by_mol


@pytest.mark.parametrize("method", AM1_AND_OM2_METHODS)
def test_md_basic_batch_mixed(tmp_path, device, batch_molecule_data, method):
    species, coordinates = batch_molecule_data
    seqm_parameters = _seqm_parameters(method)

    molid = list(range(species.shape[0]))
    prefix = str(tmp_path / "md_basic_batch_mixed")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = Molecular_Dynamics_Basic(
        seqm_parameters=seqm_parameters, timestep=0.5, Temp=300.0, output=_output_config(prefix, molid)
    ).to(device)

    metrics = _run_and_check(md, molecule, prefix, molid, steps=30)
    ref_path = reference_path_for_method("md_basic_batch_mixed", method, am1_name="md_basic_batch_mixed")
    ref = load_or_update_reference(ref_path, metrics)
    for current, expected in zip(metrics, ref):
        _assert_md_metrics(current, expected, tol_drift=1e-2, tol_slope=1e-3)


@pytest.mark.parametrize("method", LANGEVIN_METHODS)
def test_md_langevin_single(tmp_path, device, methane_molecule_data, method):
    species, coordinates = methane_molecule_data
    seqm_parameters = _seqm_parameters(method)

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

    metrics = _run_and_check(md, molecule, prefix, molid, steps=20)
    _assert_temperature_envelope(metrics[0])


def test_torch_compile_config_rejects_targets(monkeypatch):
    from seqm.utils.torch_compile import normalize_torch_compile_config

    assert not normalize_torch_compile_config({})["enabled"]
    assert not normalize_torch_compile_config({"torch_compile": False})["enabled"]
    assert not normalize_torch_compile_config({"torch_compile": {"enabled": False}})["enabled"]
    assert normalize_torch_compile_config({"torch_compile": True})["enabled"]
    assert normalize_torch_compile_config({"torch_compile": {"enabled": True}})["enabled"]
    assert not normalize_torch_compile_config({}, False)["enabled"]

    cfg = normalize_torch_compile_config({}, {"enabled": True, "mode": "reduce-overhead"})
    assert cfg["enabled"]
    assert cfg["options"] == {"mode": "reduce-overhead"}

    with pytest.raises(ValueError, match="target"):
        normalize_torch_compile_config({}, {"enabled": True, "target": "force"})


def test_md_torch_compile_requires_explicit_opt_in(monkeypatch):
    import seqm.MolecularDynamics as md_module
    from seqm.utils.torch_compile import normalize_torch_compile_config

    calls = []
    monkeypatch.setattr(
        md_module, "enable_two_center_compile", lambda **kwargs: calls.append(("integrals", kwargs))
    )
    monkeypatch.setattr(md_module, "enable_fock_compile", lambda **kwargs: calls.append(("fock", kwargs)))

    seqm_parameters = {"method": "AM1"}

    class DummyMolecule:
        coordinates = type("Coordinates", (), {"is_cuda": True})()

    md = type("DummyMD", (), {})()
    md._torch_compile_config = normalize_torch_compile_config(seqm_parameters)
    md._torch_compile_applied = False
    md.seqm_parameters = seqm_parameters
    md_module.Molecular_Dynamics_Basic._enable_torch_compile_if_requested(md, DummyMolecule())

    assert calls == []
    assert not md._torch_compile_applied

    md._torch_compile_config = normalize_torch_compile_config({"method": "AM1", "torch_compile": True})
    md_module.Molecular_Dynamics_Basic._enable_torch_compile_if_requested(md, DummyMolecule())

    assert calls == [("integrals", {"mode": None}), ("fock", {"mode": None})]
    assert md._torch_compile_applied


def test_md_torch_compile_family_toggles(monkeypatch):
    import seqm.MolecularDynamics as md_module
    from seqm.utils.torch_compile import normalize_torch_compile_config

    calls = []
    monkeypatch.setattr(md_module, "enable_two_center_compile", lambda **kwargs: calls.append("integrals"))
    monkeypatch.setattr(md_module, "enable_omx_compile", lambda **kwargs: calls.append("omx"))
    monkeypatch.setattr(md_module, "enable_fock_compile", lambda **kwargs: calls.append("fock"))
    monkeypatch.setattr(md_module, "enable_rcis_compile", lambda **kwargs: calls.append("cis"))
    monkeypatch.setattr(md_module, "enable_rcis_grad_compile", lambda **kwargs: calls.append("cis_grad"))
    monkeypatch.setattr(md_module, "enable_nac_compile", lambda **kwargs: calls.append("nac"))
    monkeypatch.setattr(md_module, "enable_tdc_hamiltonian_fd_compile", lambda **kwargs: calls.append("tdc"))

    seqm_parameters = _seqm_parameters("OM1", excited=True)
    seqm_parameters["nonadiabatic"] = {"compute_nac": False}
    cfg = normalize_torch_compile_config(
        seqm_parameters,
        {
            "enabled": True,
            "compile_omx": False,
            "compile_fock": False,
            "compile_cis": False,
            "compile_nac": False,
        },
    )

    class DummyMolecule:
        coordinates = torch.zeros(1)

    md = type("DummyMD", (), {})()
    md._torch_compile_config = cfg
    md._torch_compile_applied = False
    md.seqm_parameters = seqm_parameters
    md_module.Molecular_Dynamics_Basic._enable_torch_compile_if_requested(md, DummyMolecule())

    assert calls == []
    assert md._torch_compile_applied


def test_md_torch_compile_registers_repeated_kernels(monkeypatch, tmp_path, device, methane_molecule_data):
    from seqm.dynamics import tdc_hamiltonian_fd
    from seqm.seqm_functions import fock, nac, rcis_batch, rcis_grad_batch, two_elec_two_center_int

    calls = []

    def fake_compile(fn, **kwargs):
        calls.append((getattr(fn, "__name__", ""), kwargs))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(two_elec_two_center_int, "_rotate_sp_integrals_dispatch", None)
    monkeypatch.setattr(fock, "_fock_sp_dispatch", None)
    monkeypatch.setattr(rcis_batch, "_makeA_pi_batched_dispatch", None)
    monkeypatch.setattr(rcis_batch, "_makeA_pi_symm_batch_dispatch", None)
    monkeypatch.setattr(rcis_batch, "_ao_transition_density_dispatch", None)
    monkeypatch.setattr(rcis_batch, "_mo_fock_action_dispatch", None)
    monkeypatch.setattr(rcis_batch, "_relaxed_rhs_dispatch", None)
    monkeypatch.setattr(rcis_batch, "_relaxed_finish_dispatch", None)
    monkeypatch.setattr(rcis_grad_batch, "_rcis_grad_contract_dispatch", None)
    monkeypatch.setattr(nac, "_contract_nac_density_dispatch", None)
    monkeypatch.setattr(nac, "_contract_mixed_transition_terms_dispatch", None)
    monkeypatch.setattr(nac, "_pair_response_rhs_dispatch", None)
    monkeypatch.setattr(tdc_hamiltonian_fd, "_prepare_directional_pair_ops_dispatch", None)
    monkeypatch.setattr(tdc_hamiltonian_fd, "_contract_pair_density_directional_dispatch", None)
    monkeypatch.setattr(tdc_hamiltonian_fd, "_contract_mixed_transition_directional_dispatch", None)

    species, coordinates = methane_molecule_data
    seqm_parameters = _seqm_parameters("AM1", excited=True)
    seqm_parameters.pop("torch_compile", None)
    seqm_parameters["nonadiabatic"] = {"compute_nac": False}

    molid = [0]
    prefix = str(tmp_path / "md_compile")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = Molecular_Dynamics_Basic(
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid),
        torch_compile={"enabled": True, "mode": "reduce-overhead"},
    ).to(device)

    _run_md(md, molecule, steps=1)

    compiled_names = {name for name, _ in calls}
    assert "_rotate_sp_integrals_kernel" in compiled_names
    assert "_fock_sp_kernel" in compiled_names
    assert "_makeA_pi_batched_kernel" in compiled_names
    assert "_ao_transition_density_kernel" in compiled_names
    assert "_mo_fock_action_kernel" in compiled_names
    assert "_rcis_grad_contract_kernel" in compiled_names
    assert all(kwargs["mode"] == "reduce-overhead" for _, kwargs in calls)
    assert getattr(fock._fock_sp_dispatch, "is_torch_compile_wrapper", False)
    assert getattr(two_elec_two_center_int._rotate_sp_integrals_dispatch, "is_torch_compile_wrapper", False)
    assert getattr(rcis_batch._makeA_pi_symm_batch_dispatch, "is_torch_compile_wrapper", False)
    assert getattr(rcis_grad_batch._rcis_grad_contract_dispatch, "is_torch_compile_wrapper", False)
    assert getattr(nac._contract_nac_density_dispatch, "is_torch_compile_wrapper", False)
    assert getattr(nac._contract_mixed_transition_terms_dispatch, "is_torch_compile_wrapper", False)
    assert getattr(nac._pair_response_rhs_dispatch, "is_torch_compile_wrapper", False)
    assert getattr(
        tdc_hamiltonian_fd._prepare_directional_pair_ops_dispatch, "is_torch_compile_wrapper", False
    )
    assert getattr(
        tdc_hamiltonian_fd._contract_pair_density_directional_dispatch, "is_torch_compile_wrapper", False
    )
    assert getattr(
        tdc_hamiltonian_fd._contract_mixed_transition_directional_dispatch, "is_torch_compile_wrapper", False
    )


@pytest.mark.parametrize("method", LANGEVIN_METHODS)
def test_md_langevin_batch_mixed(tmp_path, device, batch_molecule_data, method):
    species, coordinates = batch_molecule_data
    seqm_parameters = _seqm_parameters(method)

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

    metrics = _run_and_check(md, molecule, prefix, molid, steps=20)
    for current in metrics:
        _assert_temperature_envelope(current)


@pytest.mark.parametrize("method", AM1_AND_OM2_METHODS)
def test_md_xl_bomd_single(tmp_path, device, methane_molecule_data, method):
    species, coordinates = methane_molecule_data
    seqm_parameters = _seqm_parameters(method)

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
    ref_path = reference_path_for_method("md_xl_bomd_methane", method, am1_name="md_xl_bomd_methane")
    ref = load_or_update_reference(ref_path, metrics)
    _assert_md_metrics(metrics[0], ref[0], tol_drift=5e-2, tol_slope=5e-3)


@pytest.mark.parametrize("method", AM1_AND_OM2_METHODS)
def test_md_ksa_xl_bomd_single(tmp_path, device, methane_molecule_data, method):
    species, coordinates = methane_molecule_data
    seqm_parameters = _seqm_parameters(method)

    molid = [0]
    prefix = str(tmp_path / "md_ksa_xl_bomd_single")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = KSA_XL_BOMD(
        xl_bomd_params={"k": 6, "max_rank": 3, "err_threshold": 0.0, "T_el": 1500},
        damp=None,
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid),
    ).to(device)

    metrics = _run_and_check(md, molecule, prefix, molid, steps=30)
    ref_path = reference_path_for_method("md_ksa_xl_bomd_methane", method, am1_name="md_ksa_xl_bomd_methane")
    ref = load_or_update_reference(ref_path, metrics)
    _assert_md_metrics(metrics[0], ref[0], tol_drift=5e-2, tol_slope=5e-3)


@pytest.mark.parametrize("method", AM1_AND_OM2_METHODS)
def test_md_ksa_xl_bomd_batch_mixed(tmp_path, device, batch_molecule_data, method):
    species, coordinates = batch_molecule_data
    seqm_parameters = _seqm_parameters(method)

    molid = list(range(species.shape[0]))
    prefix = str(tmp_path / "md_ksa_xl_bomd_batch_mixed")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = KSA_XL_BOMD(
        xl_bomd_params={"k": 6, "max_rank": 3, "err_threshold": 0.0, "T_el": 1500},
        damp=None,
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid),
    ).to(device)

    metrics = _run_and_check(md, molecule, prefix, molid, steps=30)
    ref_path = reference_path_for_method(
        "md_ksa_xl_bomd_batch_mixed", method, am1_name="md_ksa_xl_bomd_batch_mixed"
    )
    ref = load_or_update_reference(ref_path, metrics)
    for current, expected in zip(metrics, ref):
        _assert_md_metrics(current, expected, tol_drift=5e-2, tol_slope=5e-3)


@pytest.mark.parametrize("method", AM1_AND_OM2_METHODS)
def test_md_excited_basic_batch(tmp_path, device, methanal_batch_data, method):
    species, coordinates = methanal_batch_data
    seqm_parameters = _seqm_parameters(method, excited=True)

    molid = list(range(species.shape[0]))
    prefix = str(tmp_path / "md_excited_basic_batch")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = Molecular_Dynamics_Basic(
        seqm_parameters=seqm_parameters, timestep=0.5, Temp=300.0, output=_output_config(prefix, molid)
    ).to(device)

    metrics = _run_and_check(md, molecule, prefix, molid, steps=30, expect_excited=True, n_states=4)
    ref_path = reference_path_for_method(
        "md_excited_basic_batch_methanal", method, am1_name="md_excited_basic_batch_methanal"
    )
    ref = load_or_update_reference(ref_path, metrics)
    for current, expected in zip(metrics, ref):
        _assert_md_metrics(current, expected, tol_drift=5e-2, tol_slope=5e-2)


@pytest.mark.parametrize("method", AM1_AND_OM2_METHODS)
def test_md_excited_transition_properties_opt_in(tmp_path, device, methanal_batch_data, method):
    species, coordinates = methanal_batch_data
    seqm_parameters = _seqm_parameters(method, excited=True)

    molid = list(range(species.shape[0]))
    prefix = str(tmp_path / "md_excited_transition_properties")
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    output = _output_config(prefix, molid)
    output["h5"]["transition_properties"] = True
    md = Molecular_Dynamics_Basic(
        seqm_parameters=seqm_parameters, timestep=0.5, Temp=300.0, output=output
    ).to(device)

    _run_md(md, molecule, steps=2)
    with h5py.File(f"{prefix}.0.h5", "r") as h5:
        assert h5["data/excitation/transition_dipole"].shape == (3, 4, 3)
        assert h5["data/excitation/oscillator_strength"].shape == (3, 4)


@pytest.mark.parametrize("method", LANGEVIN_METHODS)
def test_md_excited_langevin_batch(tmp_path, device, methanal_batch_data, method):
    species, coordinates = methanal_batch_data
    seqm_parameters = _seqm_parameters(method, excited=True)

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

    metrics = _run_and_check(md, molecule, prefix, molid, steps=20, expect_excited=True, n_states=4)
    for current in metrics:
        _assert_temperature_envelope(current)


@pytest.mark.parametrize("method", AM1_AND_OM2_METHODS)
def test_md_excited_xl_bomd_batch(tmp_path, device, methanal_batch_data, method):
    species, coordinates = methanal_batch_data
    seqm_parameters = _seqm_parameters(method, excited=True)

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
    ref_path = reference_path_for_method(
        "md_excited_xl_bomd_batch_methanal", method, am1_name="md_excited_xl_bomd_batch_methanal"
    )
    ref = load_or_update_reference(ref_path, metrics)
    for current, expected in zip(metrics, ref):
        _assert_md_metrics(current, expected, tol_drift=5e-2, tol_slope=5e-3)

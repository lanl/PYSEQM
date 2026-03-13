from datetime import datetime

import h5py
import torch

from seqm.MolecularDynamics import Molecular_Dynamics_Langevin
from seqm.Molecule import Molecule
from seqm.NonadiabaticDynamics import SurfaceHoppingDynamics
from seqm.seqm_functions.constants import Constants

from ..reference_data import assert_allclose


class _SimulatedCrash(RuntimeError):
    pass


def _output_config(prefix, molid, checkpoint_every):
    return {
        "molid": molid,
        "prefix": prefix,
        "print every": 0,
        "checkpoint every": checkpoint_every,
        "xyz": 0,
        "h5": {"data": 1, "coordinates": 1, "velocities": 1, "forces": 1, "nonadiabatic": 1},
    }


def _build_molecule(device, species, coordinates, seqm_parameters):
    const = Constants().to(device)
    return Molecule(const, seqm_parameters, coordinates, species).to(device)


def _read_last_frame(prefix, molid):
    results = []
    for mol in molid:
        with h5py.File(f"{prefix}.{mol}.h5", "r") as h5:
            coords = h5["coordinates/values"][-1]
            vels = h5["velocities/values"][-1]
            steps = h5["data/steps"].shape[0]
            active = h5["data/nonadiabatic/active_surface"][-1]
            amps = h5["data/nonadiabatic/electronic_amplitudes"][-1]
        results.append((coords, vels, steps, active, amps))
    return results


def _restore_initialized_from_checkpoint(path, device):
    ckpt, molecule, device, reuse_P = Molecular_Dynamics_Langevin._load_checkpoint_base(path, device=device)
    kwargs = Molecular_Dynamics_Langevin._checkpoint_init_kwargs(ckpt)
    if "damp" in ckpt:
        kwargs["damp"] = ckpt["damp"]
    saved_nstates = ckpt.get("nad_nstates")
    if saved_nstates is not None:
        seqm_params = kwargs["seqm_parameters"]
        exc_cfg = seqm_params.get("excited_states")
        if not isinstance(exc_cfg, dict):
            exc_cfg = {}
            seqm_params["excited_states"] = exc_cfg
        exc_cfg["_nad_nstates"] = int(saved_nstates)
        exc_cfg["n_states"] = int(saved_nstates)

    dyn = SurfaceHoppingDynamics(**kwargs).to(device)
    resume_state = ckpt.get("nad_state", {})
    active_states = resume_state.get("active_states")
    if torch.is_tensor(active_states):
        dyn._active_states = active_states.to(device)
    amp_phase = resume_state.get("amp_phase")
    if torch.is_tensor(amp_phase):
        dyn._amp_phase = amp_phase.to(device)
    dyn._resume_state = resume_state
    dyn.initialize(molecule, remove_com=ckpt["remove_com"], learned_parameters={}, steps=ckpt["steps"])
    return dyn, molecule


def _run_continuous(seqm_parameters, device, species, coordinates, prefix, molid, steps):
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    dyn = SurfaceHoppingDynamics(
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid, checkpoint_every=0),
        initial_state=1,
    ).to(device)
    dyn.run(molecule, steps=steps, reuse_P=True, remove_com=None, seed=0)
    return _read_last_frame(prefix, molid)


def _run_checkpoint_resume(seqm_parameters, device, species, coordinates, prefix, molid, steps, split):
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    dyn = SurfaceHoppingDynamics(
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid, checkpoint_every=split),
        initial_state=1,
    ).to(device)
    orig_save = dyn.save_checkpoint

    def _save_and_crash(*args, **kwargs):
        orig_save(*args, **kwargs)
        raise _SimulatedCrash("Simulated crash after checkpoint.")

    dyn.save_checkpoint = _save_and_crash
    try:
        dyn.run(molecule, steps=steps, reuse_P=True, remove_com=None, seed=0)
    except _SimulatedCrash:
        pass

    ckpt_path = f"{prefix}.restart.pt"
    SurfaceHoppingDynamics.run_from_checkpoint(ckpt_path, device=device)
    return _read_last_frame(prefix, molid)


def _compare_runs(continuous, resumed, steps):
    expected_steps = steps + 1
    for (coords_a, vels_a, steps_a, active_a, amps_a), (coords_b, vels_b, steps_b, active_b, amps_b) in zip(
        continuous, resumed
    ):
        assert steps_a == expected_steps
        assert steps_b == expected_steps
        assert_allclose(coords_a, coords_b, rtol=1e-5, atol=1e-5)
        assert_allclose(vels_a, vels_b, rtol=1e-5, atol=1e-5)
        assert active_a == active_b
        assert_allclose(amps_a, amps_b, rtol=1e-5, atol=1e-5)


def test_nonadiabatic_checkpoint_resume_surface_hopping(methanal_batch_data, device, tmp_path):
    species, coordinates = methanal_batch_data
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "excited_states": {"n_states": 2, "method": "cis"},
    }

    steps = 4
    split = 2
    molid = [0, 1]

    continuous = _run_continuous(
        seqm_parameters,
        device,
        species.clone(),
        coordinates.clone(),
        str(tmp_path / "nad_surface_cont"),
        molid,
        steps,
    )
    resumed = _run_checkpoint_resume(
        seqm_parameters,
        device,
        species.clone(),
        coordinates.clone(),
        str(tmp_path / "nad_surface_ckpt"),
        molid,
        steps,
        split,
    )
    _compare_runs(continuous, resumed, steps)


def test_nonadiabatic_checkpoint_restores_internal_state(methanal_batch_data, device, tmp_path):
    species, coordinates = methanal_batch_data
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "excited_states": {"n_states": 2, "method": "cis"},
    }

    molecule = _build_molecule(device, species.clone(), coordinates.clone(), seqm_parameters)
    dyn = SurfaceHoppingDynamics(
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output={
            "molid": [0, 1],
            "prefix": str(tmp_path / "nad_restore_state"),
            "print every": 0,
            "checkpoint every": 0,
            "xyz": 0,
            "h5": {},
        },
        initial_state=1,
    ).to(device)
    dyn.start_time = datetime.now()
    dyn.initialize(molecule, remove_com=None, learned_parameters={}, steps=4)

    dtype = molecule.coordinates.dtype
    custom_active = torch.tensor([1, 0], dtype=torch.long, device=device)
    custom_amp = torch.zeros_like(dyn._amp_phase)
    custom_amp[0, 1] = torch.tensor([0.9, 0.0, 0.2], dtype=dtype, device=device)
    custom_amp[1, 0] = torch.tensor([0.8, 0.1, -0.3], dtype=dtype, device=device)
    custom_post_hop = torch.tensor([2, 1], dtype=torch.long, device=device)
    custom_prev = torch.tensor([0, 1], dtype=torch.long, device=device)
    custom_potential = torch.tensor([1.35, 1.18], dtype=dtype, device=device)
    custom_nac_dot = torch.tensor(
        [[[0.0, 0.12], [-0.12, 0.0]], [[0.0, 0.07], [-0.07, 0.0]]], dtype=dtype, device=device
    )

    dyn._active_states = custom_active.clone()
    dyn._amp_phase = custom_amp.clone()
    dyn.post_hop_holdoff = custom_post_hop.clone()
    dyn.prev_state = custom_prev.clone()
    dyn._current_potential = custom_potential.clone()
    dyn._cache_old["nac_dot"] = custom_nac_dot.clone()

    ckpt_path = str(tmp_path / "nad_manual_restore.restart.pt")
    dyn.save_checkpoint(molecule, steps=4, reuse_P=True, remove_com=None, step_done=2, path=ckpt_path)

    restored_dyn, restored_molecule = _restore_initialized_from_checkpoint(ckpt_path, device=device)

    assert torch.equal(restored_dyn._active_states, custom_active)
    assert_allclose(restored_dyn._amp_phase, custom_amp, rtol=0.0, atol=0.0)
    assert torch.equal(restored_dyn.post_hop_holdoff, custom_post_hop)
    assert torch.equal(restored_dyn.prev_state, custom_prev)
    assert_allclose(restored_dyn._current_potential, custom_potential, rtol=0.0, atol=0.0)
    assert_allclose(restored_dyn._cache_old["nac_dot"], custom_nac_dot, rtol=0.0, atol=0.0)
    assert torch.equal(restored_molecule.active_state, custom_active + 1)

import h5py

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

import h5py

from seqm.MolecularDynamics import KSA_XL_BOMD, XL_BOMD, Molecular_Dynamics_Basic, Molecular_Dynamics_Langevin
from seqm.Molecule import Molecule
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
        "h5": {"data": 1, "coordinates": 1, "velocities": 1, "forces": 1},
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
        results.append((coords, vels, steps))
    return results


def _run_continuous(md_cls, seqm_parameters, device, species, coordinates, prefix, molid, steps, **md_kwargs):
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = md_cls(
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid, checkpoint_every=0),
        **md_kwargs,
    ).to(device)
    md.run(molecule, steps=steps, reuse_P=True, remove_com=None, seed=0)
    return _read_last_frame(prefix, molid)


def _run_checkpoint_resume(
    md_cls, seqm_parameters, device, species, coordinates, prefix, molid, steps, split, **md_kwargs
):
    molecule = _build_molecule(device, species, coordinates, seqm_parameters)
    md = md_cls(
        seqm_parameters=seqm_parameters,
        timestep=0.5,
        Temp=300.0,
        output=_output_config(prefix, molid, checkpoint_every=split),
        **md_kwargs,
    ).to(device)
    orig_save = md.save_checkpoint

    def _save_and_crash(*args, **kwargs):
        orig_save(*args, **kwargs)
        raise _SimulatedCrash("Simulated crash after checkpoint.")

    md.save_checkpoint = _save_and_crash
    try:
        md.run(molecule, steps=steps, reuse_P=True, remove_com=None, seed=0)
    except _SimulatedCrash:
        pass

    ckpt_path = f"{prefix}.restart.pt"

    Molecular_Dynamics_Basic.run_from_checkpoint(ckpt_path, device=device)
    return _read_last_frame(prefix, molid)


def _compare_runs(continuous, resumed, steps):
    for (coords_a, vels_a, steps_a), (coords_b, vels_b, steps_b) in zip(continuous, resumed):
        assert steps_a == steps
        assert steps_b == steps
        assert_allclose(coords_a, coords_b, rtol=1e-5, atol=1e-5)
        assert_allclose(vels_a, vels_b, rtol=1e-5, atol=1e-5)


def test_md_checkpoint_resume_basic(batch_molecule_data, device, tmp_path):
    species, coordinates = batch_molecule_data
    seqm_parameters = {"method": "AM1", "scf_eps": 1.0e-7, "scf_converger": [1]}

    steps = 6
    split = 3
    molid = [0, 1]

    continuous = _run_continuous(
        Molecular_Dynamics_Basic,
        seqm_parameters,
        device,
        species.clone(),
        coordinates.clone(),
        str(tmp_path / "md_basic_cont"),
        molid,
        steps,
    )
    resumed = _run_checkpoint_resume(
        Molecular_Dynamics_Basic,
        seqm_parameters,
        device,
        species.clone(),
        coordinates.clone(),
        str(tmp_path / "md_basic_ckpt"),
        molid,
        steps,
        split,
    )
    _compare_runs(continuous, resumed, steps)


def test_md_checkpoint_resume_langevin(batch_molecule_data, device, tmp_path):
    species, coordinates = batch_molecule_data
    seqm_parameters = {"method": "AM1", "scf_eps": 1.0e-7, "scf_converger": [1]}

    steps = 6
    split = 3
    molid = [0, 1]

    continuous = _run_continuous(
        Molecular_Dynamics_Langevin,
        seqm_parameters,
        device,
        species.clone(),
        coordinates.clone(),
        str(tmp_path / "md_langevin_cont"),
        molid,
        steps,
        damp=10.0,
    )
    resumed = _run_checkpoint_resume(
        Molecular_Dynamics_Langevin,
        seqm_parameters,
        device,
        species.clone(),
        coordinates.clone(),
        str(tmp_path / "md_langevin_ckpt"),
        molid,
        steps,
        split,
        damp=10.0,
    )
    _compare_runs(continuous, resumed, steps)


def test_md_checkpoint_resume_xl_bomd(batch_molecule_data, device, tmp_path):
    species, coordinates = batch_molecule_data
    seqm_parameters = {"method": "AM1", "scf_eps": 1.0e-7, "scf_converger": [1]}

    steps = 6
    split = 3
    molid = [0, 1]

    continuous = _run_continuous(
        XL_BOMD,
        seqm_parameters,
        device,
        species.clone(),
        coordinates.clone(),
        str(tmp_path / "md_xl_bomd_cont"),
        molid,
        steps,
        damp=None,
        xl_bomd_params={"k": 6},
    )
    resumed = _run_checkpoint_resume(
        XL_BOMD,
        seqm_parameters,
        device,
        species.clone(),
        coordinates.clone(),
        str(tmp_path / "md_xl_bomd_ckpt"),
        molid,
        steps,
        split,
        damp=None,
        xl_bomd_params={"k": 6},
    )
    _compare_runs(continuous, resumed, steps)


def test_md_checkpoint_resume_ksa_xl_bomd(batch_molecule_data, device, tmp_path):
    species, coordinates = batch_molecule_data
    seqm_parameters = {"method": "AM1", "scf_eps": 1.0e-7, "scf_converger": [1]}

    steps = 6
    split = 3
    molid = [0, 1]

    xl_params = {"k": 6, "max_rank": 3, "err_threshold": 0.0, "T_el": 1500}

    continuous = _run_continuous(
        KSA_XL_BOMD,
        seqm_parameters,
        device,
        species.clone(),
        coordinates.clone(),
        str(tmp_path / "md_ksa_xl_bomd_cont"),
        molid,
        steps,
        damp=None,
        xl_bomd_params=xl_params,
    )
    resumed = _run_checkpoint_resume(
        KSA_XL_BOMD,
        seqm_parameters,
        device,
        species.clone(),
        coordinates.clone(),
        str(tmp_path / "md_ksa_xl_bomd_ckpt"),
        molid,
        steps,
        split,
        damp=None,
        xl_bomd_params=xl_params,
    )
    _compare_runs(continuous, resumed, steps)


def test_md_checkpoint_resume_excited_basic(methanal_batch_data, device, tmp_path):
    species, coordinates = methanal_batch_data
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "excited_states": {"n_states": 4, "method": "cis"},
        "active_state": 1,
    }

    steps = 6
    split = 3
    molid = [0, 1, 2]

    continuous = _run_continuous(
        Molecular_Dynamics_Basic,
        seqm_parameters,
        device,
        species.clone(),
        coordinates.clone(),
        str(tmp_path / "md_excited_basic_cont"),
        molid,
        steps,
    )
    resumed = _run_checkpoint_resume(
        Molecular_Dynamics_Basic,
        seqm_parameters,
        device,
        species.clone(),
        coordinates.clone(),
        str(tmp_path / "md_excited_basic_ckpt"),
        molid,
        steps,
        split,
    )
    _compare_runs(continuous, resumed, steps)


def test_md_checkpoint_resume_excited_xl_bomd(methanal_batch_data, device, tmp_path):
    species, coordinates = methanal_batch_data
    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "excited_states": {"n_states": 4, "method": "cis"},
        "active_state": 1,
    }

    steps = 6
    split = 3
    molid = [0, 1, 2]

    continuous = _run_continuous(
        XL_BOMD,
        seqm_parameters,
        device,
        species.clone(),
        coordinates.clone(),
        str(tmp_path / "md_excited_xl_bomd_cont"),
        molid,
        steps,
        damp=None,
        xl_bomd_params={"k": 6},
    )
    resumed = _run_checkpoint_resume(
        XL_BOMD,
        seqm_parameters,
        device,
        species.clone(),
        coordinates.clone(),
        str(tmp_path / "md_excited_xl_bomd_ckpt"),
        molid,
        steps,
        split,
        damp=None,
        xl_bomd_params={"k": 6},
    )
    _compare_runs(continuous, resumed, steps)

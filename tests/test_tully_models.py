import torch

from scripts.tully_surface_hopping.TullyModels import TullyDynamics, TullyFSSH, TullyModel, TullyMolecule


def test_tully_cache_shapes_and_dtypes():
    model = TullyModel.single_crossing()
    dyn = TullyDynamics(model)
    mol = TullyMolecule(x0=-8.0, v0=2.0, mass=2000.0)
    dyn._setup_states(mol)
    dyn._init_coeffs(mol)
    energies = dyn._compute_electronic_structure(mol, {})
    assert energies.shape == (1, 2)
    cache = dyn._cache_new
    assert not cache["nac_dot"].dtype.is_complex
    assert cache["nac_vec"].shape == (1, 2, 2, 1, 3)


def test_tully_ehrenfest_population_norm():
    model = TullyModel.single_crossing()
    dyn = TullyDynamics(model, timestep=0.05, electronic_substeps=5)
    mol = TullyMolecule(x0=-8.0, v0=2.0, mass=2000.0)
    dyn.run(mol, steps=10, reuse_P=True, remove_com=None)
    pop = dyn.populations
    assert torch.allclose(pop.sum(), torch.tensor(1.0, dtype=pop.dtype), atol=1e-6)


def test_tully_fssh_runs_and_logs():
    model = TullyModel.single_crossing()
    dyn = TullyFSSH(model, timestep=0.05, electronic_substeps=5)
    mol = TullyMolecule(x0=-8.0, v0=2.0, mass=2000.0)
    dyn.run(mol, steps=10, reuse_P=True, remove_com=None)
    # hop log should never exceed number of steps
    assert len(dyn.hop_log) <= 10
    assert dyn._amp_phase.shape[1] == 2


def test_other_tully_models_produce_finite_values():
    for builder in (TullyModel.double_crossing, TullyModel.extended_coupling):
        model = builder()
        x = torch.tensor([-5.0, 0.0, 5.0])
        E, dE, nac = model.pot(x)
        assert torch.isfinite(E).all()
        assert torch.isfinite(dE).all()
        assert torch.isfinite(nac).all()

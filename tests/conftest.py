import os
from pathlib import Path

import pytest
import torch

from seqm.seqm_functions.read_xyz import read_xyz


@pytest.fixture(scope="session")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def repo_root():
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def methane_xyz_path(repo_root):
    return repo_root / "tests" / "data" / "methane.xyz"


@pytest.fixture(scope="function")
def methane_molecule_data(methane_xyz_path, device):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    species, coordinates = read_xyz([os.fspath(methane_xyz_path)])
    species = torch.as_tensor(species, dtype=torch.int64, device=device)
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64, device=device)
    return species.clone(), coordinates.clone()


@pytest.fixture(scope="function")
def batch_molecule_data(repo_root, device):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    paths = [repo_root / "tests" / "data" / "methane.xyz", repo_root / "tests" / "data" / "benzene.xyz"]
    species, coordinates = read_xyz([os.fspath(p) for p in paths])
    species = torch.as_tensor(species, dtype=torch.int64, device=device)
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64, device=device)
    return species.clone(), coordinates.clone()


@pytest.fixture(scope="function")
def methanal_batch_data(repo_root, device):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    paths = [
        repo_root / "tests" / "data" / "methanal.1.xyz",
        repo_root / "tests" / "data" / "methanal.2.xyz",
        repo_root / "tests" / "data" / "methanal.3.xyz",
    ]
    species, coordinates = read_xyz([os.fspath(p) for p in paths])
    species = torch.as_tensor(species, dtype=torch.int64, device=device)
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64, device=device)
    return species.clone(), coordinates.clone()


@pytest.fixture(scope="function")
def excited_mixed_batch_data(repo_root, device):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    paths = [
        repo_root / "tests" / "data" / "methanal.1.xyz",
        repo_root / "tests" / "data" / "methane.xyz",
        repo_root / "tests" / "data" / "benzene.xyz",
    ]
    species, coordinates = read_xyz([os.fspath(p) for p in paths])
    species = torch.as_tensor(species, dtype=torch.int64, device=device)
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64, device=device)
    return species.clone(), coordinates.clone()

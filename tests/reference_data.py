import json
import os
from pathlib import Path

import pytest
import torch

UPDATE_REFERENCES = os.environ.get("PYSEQM_UPDATE_REFERENCES") == "1"
REFERENCE_DIR = Path(__file__).resolve().parent / "reference"


def reference_path(name):
    filename = f"{name}.json"
    return REFERENCE_DIR / filename


def load_or_update_reference(path, data):
    if UPDATE_REFERENCES:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        return data

    if not path.exists():
        pytest.skip(
            f"Missing reference data at {path}. "
            "Run tests with PYSEQM_UPDATE_REFERENCES=1 to generate it."
        )

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_reference(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_allclose(actual, expected, rtol=1e-6, atol=1e-6):
    actual_t = torch.as_tensor(actual)
    expected_t = torch.as_tensor(expected)
    assert torch.allclose(actual_t, expected_t, rtol=rtol, atol=atol)

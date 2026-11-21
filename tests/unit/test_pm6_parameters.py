"""Consistency of the PM6 parameter files."""

import csv
from pathlib import Path

import pytest

from seqm.seqm_functions.parameters import params

PARAMS = Path(__file__).resolve().parents[2] / "seqm" / "params"
MAIN_FILES = ["parameters_PM6_MOPAC.csv", "parameters_PM6_SP_MOPAC.csv"]


def read_params(name):
    rows = list(csv.reader(open(PARAMS / name, newline="")))
    header = [h.strip() for h in rows[0]]
    return rows, {int(r[0]): dict(zip(header, [x.strip() for x in r])) for r in rows[1:]}


@pytest.mark.parametrize("name", MAIN_FILES)
def test_every_row_has_every_field(name):
    rows, _ = read_params(name)
    assert {len(r) for r in rows} == {len(rows[0])}
    assert [int(r[0]) for r in rows[1:]] == list(range(1, 108))
    for r in rows[1:]:
        for field in r[2:]:  # a missing comma leaves two numbers in one field
            float(field)


def test_sp_file_only_drops_d_orbitals_of_main_group():
    _, full = read_params("parameters_PM6_MOPAC.csv")
    _, sp = read_params("parameters_PM6_SP_MOPAC.csv")
    differing = {(z, c) for z in full for c in full[z] if full[z][c] != sp[z][c]}
    assert differing == {(z, c) for z in range(13, 18) for c in ("U_dd", "zeta_d", "beta_d")}


def test_warns_for_unparameterized_elements():
    with pytest.warns(UserWarning, match=r"no parameters for atomic numbers \[58\]"):
        params(method="PM6", elements=[1, 6, 58], root_dir=str(PARAMS) + "/")

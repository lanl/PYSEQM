"""The per-element tables in Constants are indexed by atomic number."""

import csv
from pathlib import Path

import pytest
import torch

from seqm.seqm_functions.constants import D_SHELL_ELEMENTS, Constants, has_d_shell

TABLES = ["atomic_num", "tore", "iso", "qn", "qn_int", "qnD_int", "qnS_int", "qnP_int", "ussc", "uppc", "gssc",
          "gspc", "hspc", "gp2c", "gppc", "eheat", "mass"]  # fmt: skip
PM6_CSV = Path(__file__).resolve().parents[2] / "seqm" / "params" / "parameters_PM6_MOPAC.csv"


@pytest.fixture(scope="module")
def const():
    return Constants()


def test_tables_are_indexed_by_atomic_number(const):
    assert len(const.label) == 87
    assert all(len(getattr(const, name)) == 87 for name in TABLES)
    symbols = {1: "H", 55: "Cs", 57: "La", 71: "Lu", 72: "Hf", 80: "Hg", 82: "Pb", 86: "Rn"}
    assert {z: const.label[z].strip() for z in symbols} == symbols


def test_d_shell_elements_agree_with_qnD_and_the_pm6_parameters(const):
    # has_d_shell decides the 9-orbital basis and the d-orbital integrals, qnD the d overlaps; a mismatch gives
    # d integrals for elements without a d exponent (math domain error) or skips them
    assert [z for z in range(87) if int(const.qnD_int[z]) > 0] == list(D_SHELL_ELEMENTS)
    assert has_d_shell(torch.arange(87)).nonzero().flatten().tolist() == list(D_SHELL_ELEMENTS)
    rows = list(csv.reader(open(PM6_CSV, newline="")))
    zeta_d = [h.strip() for h in rows[0]].index("zeta_d")
    assert [int(r[0]) for r in rows[1:87] if float(r[zeta_d]) > 0] == list(D_SHELL_ELEMENTS)

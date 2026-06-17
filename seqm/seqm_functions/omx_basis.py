import os

import torch

_OM1_BASIS_RAW = {
    1: {
        "shell_type": 0,
        "exponents": (2.227660584, 0.4057711562, 0.1098175104),
        "cs": (0.1543289673, 0.5353281423, 0.4446345422),
        "cp": (0.0, 0.0, 0.0),
    },
    6: {
        "shell_type": 1,
        "exponents": (2.64486, 0.54215, 0.14466),
        "cs": (-0.19188, 0.61628, 0.54896),
        "cp": (0.20259, 0.55830, 0.45514),
    },
    7: {
        "shell_type": 1,
        "exponents": (3.68849, 0.77534, 0.20498),
        "cs": (-0.19269, 0.61888, 0.54926),
        "cp": (0.22281, 0.56032, 0.43859),
    },
    8: {
        "shell_type": 1,
        "exponents": (4.78499, 0.99860, 0.25687),
        "cs": (-0.19248, 0.66952, 0.50270),
        "cp": (0.24158, 0.55890, 0.43160),
    },
    9: {
        "shell_type": 1,
        "exponents": (6.01783, 1.25315, 0.31760),
        "cs": (-0.18850, 0.69800, 0.47427),
        "cp": (0.25667, 0.56013, 0.42139),
    },
}

from .om1_ppecp_tables import (
    DAWERF_C,
    DAWERF_H,
    DAWERF_IFIRST,
    DAWERF_ILAST,
    DAWF_C,
    DAWF_H,
    DAWF_IFIRST,
    DAWF_ILAST,
    OM1_FIRST_ROW_ECP,
)


def _make_horner_table(coeffs, ifirst, ilast):
    """
    Build [n_interval, max_poly_len] table in Horner order.

    Original scalar evaluation:
        val = c[end]
        for idx = end - 1 ... start:
            val = c[idx] + x * val

    Stored row:
        [0, ..., c[end], c[end-1], ..., c[start]]
    """
    n_interval = ifirst.numel()
    lengths = ilast - ifirst + 1
    max_len = int(lengths.max().item())

    table = coeffs.new_zeros((n_interval, max_len))

    for i in range(n_interval):
        start = int(ifirst[i].item()) - 1
        end = int(ilast[i].item()) - 1

        vals = coeffs[start : end + 1].flip(0)  # c[end], ..., c[start]
        table[i, -vals.numel() :] = vals

    return table


def _build_om1_ppecp_tensor_tables(*, dtype, device):
    """
    Build static tensor tables used by vectorized OM1 PPECP.

    These are intentionally built once with the basis tables, not inside
    the hot PPECP function.
    """
    max_ecp_z = max(OM1_FIRST_ROW_ECP)

    ecp_supported = torch.zeros((max_ecp_z + 1,), dtype=torch.bool, device=device)
    ecp_zlp = torch.zeros((max_ecp_z + 1, 3), dtype=dtype, device=device)
    ecp_clp = torch.zeros((max_ecp_z + 1, 3), dtype=dtype, device=device)

    for z, ecp in OM1_FIRST_ROW_ECP.items():
        ecp_supported[z] = True
        ecp_zlp[z] = torch.tensor(ecp["zlp"], dtype=dtype, device=device)
        ecp_clp[z] = torch.tensor(ecp["clp"], dtype=dtype, device=device)

    tri_i, tri_j = torch.tril_indices(3, 3, device=device)
    tri_same = tri_i == tri_j

    tri_sym = torch.ones((1, tri_i.numel()), dtype=dtype, device=device)
    tri_sym[:, ~tri_same] = 2.0

    tri_offdiag = torch.ones((1, tri_i.numel()), dtype=dtype, device=device)
    tri_offdiag[:, tri_same] = 0.0

    dawf_c = torch.tensor(DAWF_C, dtype=dtype, device=device)
    dawf_ifirst = torch.tensor(DAWF_IFIRST, dtype=torch.long, device=device)
    dawf_ilast = torch.tensor(DAWF_ILAST, dtype=torch.long, device=device)

    dawerf_c = torch.tensor(DAWERF_C, dtype=dtype, device=device)
    dawerf_ifirst = torch.tensor(DAWERF_IFIRST, dtype=torch.long, device=device)
    dawerf_ilast = torch.tensor(DAWERF_ILAST, dtype=torch.long, device=device)

    dawf_table = _make_horner_table(dawf_c, dawf_ifirst, dawf_ilast)
    dawerf_table = _make_horner_table(dawerf_c, dawerf_ifirst, dawerf_ilast)

    return {
        "dawf_c": dawf_c,
        "dawf_ifirst": dawf_ifirst,
        "dawf_ilast": dawf_ilast,
        "dawf_h": DAWF_H,
        "dawf_table": dawf_table,
        "dawerf_c": dawerf_c,
        "dawerf_ifirst": dawerf_ifirst,
        "dawerf_ilast": dawerf_ilast,
        "dawerf_h": DAWERF_H,
        "dawerf_table": dawerf_table,
        "ecp_supported": ecp_supported,
        "ecp_zlp": ecp_zlp,
        "ecp_clp": ecp_clp,
        "tri_i": tri_i,
        "tri_j": tri_j,
        "tri_sym": tri_sym,
        "tri_offdiag": tri_offdiag,
        # Useful tiny constants to avoid rebuilding Python lists in hot funcs.
        "fctrl": torch.tensor([1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0], dtype=dtype, device=device),
        "dfctrl": torch.tensor(
            [1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0], dtype=dtype, device=device
        ),
        "fjps_dfctrl": torch.tensor([1.0, 3.0, 15.0, 105.0], dtype=dtype, device=device),
    }


def build_omx_basis_tables(atomic_numbers, method, dtype, device):
    """
    Build tensor-indexed OMx basis tables keyed by atomic number.
    """
    if atomic_numbers.numel() == 0:
        return {}

    real_atomic_numbers = [int(z) for z in torch.unique(atomic_numbers).tolist() if int(z) > 0]
    unsupported = sorted(z for z in real_atomic_numbers if z not in _OM1_BASIS_RAW)
    if unsupported:
        raise ValueError(f"OMx basis only supports H/C/N/O/F; got atomic numbers {unsupported}")

    max_z = max(max(real_atomic_numbers), max(_OM1_BASIS_RAW))
    shell_type = torch.full((max_z + 1,), -1, dtype=torch.int64, device=device)
    exponents = torch.zeros((max_z + 1, 3), dtype=dtype, device=device)
    coeff_s = torch.zeros((max_z + 1, 3), dtype=dtype, device=device)
    coeff_p = torch.zeros((max_z + 1, 3), dtype=dtype, device=device)

    for z, basis in _OM1_BASIS_RAW.items():
        shell_type[z] = basis["shell_type"]
        exponents[z] = torch.tensor(basis["exponents"], dtype=dtype, device=device)
        coeff_s[z] = torch.tensor(basis["cs"], dtype=dtype, device=device)
        coeff_p[z] = torch.tensor(basis["cp"], dtype=dtype, device=device)

    return {
        "shell_type": shell_type,
        "exponents": exponents,
        "coeff_s": coeff_s,
        "coeff_p": coeff_p,
        "boys_integrals": BoysInterpolationTable().to(device),
        "ppecp": _build_om1_ppecp_tensor_tables(dtype=dtype, device=device) if method == "OM1" else None,
    }


def gather_om1_basis(atomic_numbers, zeta, basis_tables):
    """
    Gather OM1 basis primitives for the requested atoms and zetas.
    """
    shell_type = basis_tables["shell_type"][atomic_numbers]
    if (shell_type < 0).any():
        unsupported = torch.unique(atomic_numbers[shell_type < 0]).detach().cpu().tolist()
        raise ValueError(f"OMx basis only supports H/C/N/O/F; got atomic numbers {unsupported}")

    exponents = basis_tables["exponents"][atomic_numbers]
    coeff_s = basis_tables["coeff_s"][atomic_numbers]
    coeff_p = basis_tables["coeff_p"][atomic_numbers]

    scaled_exponents = exponents * zeta.unsqueeze(1) ** 2
    norm_s = ((2.0 / torch.pi) ** 0.75) * scaled_exponents.pow(0.75)
    norm_p = 2.0 * ((2.0 / torch.pi) ** 0.75) * scaled_exponents.pow(1.25)
    coeff_s = coeff_s * norm_s
    coeff_p = coeff_p * norm_p
    return shell_type, scaled_exponents, coeff_s, coeff_p


def build_om1_basis_payload(atomic_numbers, zeta, basis_tables):
    shell_type, exponents, coeff_s, coeff_p = gather_om1_basis(atomic_numbers, zeta, basis_tables)
    return {"shell_type": shell_type, "exponents": exponents, "coeff_s": coeff_s, "coeff_p": coeff_p}


def select_om1_basis_payload(basis_payload, idx):
    return {name: values[idx] for name, values in basis_payload.items()}


class BoysInterpolationTable(torch.nn.Module):
    def __init__(self, table_path="fortran_boys_table.pt"):
        super().__init__()
        current_dir = os.path.dirname(__file__)
        table_path = os.path.join(current_dir, "..", "params", "boys_integral_table.pt")
        tab = torch.load(table_path, map_location="cpu", weights_only=True)

        self.register_buffer("A", tab["A"])
        self.register_buffer("B", tab["B"])
        self.register_buffer("C", tab["C"])

        self.maxfmt = int(tab["MAXFMT"])
        self.xmax = float(tab["XMAX"])
        self.xlim = float(tab["XLIM"])

    def forward(self, x, m_count=5):
        """
        Table interpolation for X < XMAX.
        Returns tuple F0...F_{m_count-1}.
        """
        qq = x * 20.0
        n = torch.floor(qq).to(torch.long).clamp(0, self.maxfmt - 2)
        th = qq - n.to(x.dtype)

        th2 = th * (th - 1.0)
        th3 = th2 * (th - 2.0)
        th4 = th2 * (th + 1.0)

        vals = (
            self.A[n, :m_count]
            + th[..., None] * self.B[n, :m_count]
            - th3[..., None] * self.C[n, :m_count]
            + th4[..., None] * self.C[n + 1, :m_count]
        )

        return tuple(vals[..., m] for m in range(m_count))

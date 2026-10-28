import torch

from .omx_basis import build_om1_basis_payload, build_omx_basis_tables

OMX_METHODS = {"OM1", "OM2", "OM3"}


def get_orbital_zeta_tensor(parameters, method, include_d=False):
    if method in OMX_METHODS:
        return parameters["zeta"]
    fields = ["zeta_s", "zeta_p"]
    if include_d:
        fields.append("zeta_d")
    return torch.stack([parameters[field] for field in fields], dim=1)


def get_orbital_zetas(parameters, method):
    if method in OMX_METHODS:
        zeta = parameters["zeta"]
        return zeta, zeta
    return parameters["zeta_s"], parameters["zeta_p"]


def build_beta_tensor(parameters, method):
    fields = ["beta_s", "beta_p"]
    if method == "PM6":
        fields.append("beta_d")
    return torch.stack([parameters[field] for field in fields], dim=1)


def build_omx_parameter_tables(packpar, parameters, atomic_numbers=None):
    """
    Build element-indexed OMx parameter tables from the packed parameter module.

    The resulting dictionary is typically cached on molecule.parameters under
    "_omx_tables" so OMx hcore code can reuse it without rebuilding it.
    """
    if not hasattr(packpar, "p") or not hasattr(packpar, "required_list"):
        return {}

    tables = {}
    for idx, name in enumerate(packpar.required_list):
        tables[name] = packpar.p[:, idx]

    nelem = packpar.p.shape[0]
    for name, values in parameters.items():
        if name in tables or not torch.is_tensor(values) or values.dim() != 1:
            continue
        if (
            name in packpar.learned_list
            and atomic_numbers is not None
            and values.shape == atomic_numbers.shape
        ):
            # Learned parameters arrive per atom, but the OMx code indexes element tables with ni/nj.
            # Scatter them into an element-indexed table (mean over atoms of the same element, which is
            # exact whenever the value depends only on the element, as in a static parameter fit).
            idx = atomic_numbers.long()
            total = torch.zeros(nelem, dtype=values.dtype, device=values.device).index_add(0, idx, values)
            count = torch.zeros(nelem, dtype=values.dtype, device=values.device).index_add(
                0, idx, torch.ones_like(values)
            )
            values = total / count.clamp(min=1)
        tables[name] = values
    return tables


def prepare_parameters(parameters, packpar, method, atomic_numbers, *, dtype, device):
    """
    Apply the OMx-specific derived-parameter setup in one place.
    """
    if method in OMX_METHODS:
        zeros_zeta = torch.zeros_like(parameters["zeta"])
        parameters["_omx_basis"] = build_omx_basis_tables(atomic_numbers, method, dtype=dtype, device=device)
        parameters["_omx_tables"] = build_omx_parameter_tables(packpar, parameters, atomic_numbers)
        parameters["_omx_basis_data"] = build_om1_basis_payload(
            atomic_numbers, parameters["zeta"], parameters["_omx_basis"]
        )
    else:
        zeros_zeta = torch.zeros_like(parameters["zeta_s"])
        parameters.pop("_omx_basis", None)
        parameters.pop("_omx_tables", None)
        parameters.pop("_omx_basis_data", None)
    zeros_uss = torch.zeros_like(parameters["U_ss"])
    parameters["zeta_d"] = zeros_zeta
    parameters["s_orb_exp_tail"] = zeros_zeta.clone()
    parameters["p_orb_exp_tail"] = zeros_zeta.clone()
    parameters["d_orb_exp_tail"] = zeros_zeta.clone()
    parameters["U_dd"] = zeros_uss
    parameters["F0SD"] = zeros_uss.clone()
    parameters["G2SD"] = zeros_uss.clone()
    parameters["rho_core"] = zeros_uss.clone()

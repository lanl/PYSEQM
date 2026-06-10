import torch

from .omx_basis import build_omx_basis_tables

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


def build_omx_parameter_tables(packpar, parameters):
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

    for name, values in parameters.items():
        if name in tables or not torch.is_tensor(values) or values.dim() != 1:
            continue
        tables[name] = values
    return tables


def prepare_parameters(parameters, packpar, method, atomic_numbers, *, dtype, device):
    """
    Apply the OMx-specific derived-parameter setup in one place.
    """
    if method in OMX_METHODS:
        zeros_zeta = torch.zeros_like(parameters["zeta"])
        parameters["_omx_basis"] = build_omx_basis_tables(atomic_numbers, method, dtype=dtype, device=device)
        parameters["_omx_tables"] = build_omx_parameter_tables(packpar, parameters)
    else:
        zeros_zeta = torch.zeros_like(parameters["zeta_s"])
        parameters.pop("_omx_basis", None)
        parameters.pop("_omx_tables", None)
    zeros_uss = torch.zeros_like(parameters["U_ss"])
    parameters["zeta_d"] = zeros_zeta
    parameters["s_orb_exp_tail"] = zeros_zeta.clone()
    parameters["p_orb_exp_tail"] = zeros_zeta.clone()
    parameters["d_orb_exp_tail"] = zeros_zeta.clone()
    parameters["U_dd"] = zeros_uss
    parameters["F0SD"] = zeros_uss.clone()
    parameters["G2SD"] = zeros_uss.clone()
    parameters["rho_core"] = zeros_uss.clone()

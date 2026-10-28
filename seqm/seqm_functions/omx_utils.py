import torch

from .omx_basis import build_om1_basis_payload, build_omx_basis_tables

OMX_METHODS = {"OM1", "OM2", "OM3"}
OMX_PAIR_PARAMS = (
    "g_ss", "U_ss", "U_pp", "fval1", "fval2", "gval1", "gval2", "zscor", "fscor", "bscor", "ascor",
    "beta_s", "beta_p", "beta_pi", "beta_sh", "beta_ph", "alpha_s", "alpha_p", "alpha_pi", "alpha_s_h", "alpha_p_h",
)  # fmt: skip


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


def omx_pair_params(parameters, idxi, idxj):
    names = [name for name in OMX_PAIR_PARAMS if name in parameters]
    return {name: parameters[name][idxi] for name in names}, {name: parameters[name][idxj] for name in names}


def prepare_parameters(parameters, method, atomic_numbers, *, dtype, device):
    """
    Apply the OMx-specific derived-parameter setup in one place.
    """
    if method in OMX_METHODS:
        zeros_zeta = torch.zeros_like(parameters["zeta"])
        parameters["_omx_basis"] = build_omx_basis_tables(atomic_numbers, method, dtype=dtype, device=device)
        parameters["_omx_basis_data"] = build_om1_basis_payload(
            atomic_numbers, parameters["zeta"], parameters["_omx_basis"]
        )
    else:
        zeros_zeta = torch.zeros_like(parameters["zeta_s"])
        parameters.pop("_omx_basis", None)
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

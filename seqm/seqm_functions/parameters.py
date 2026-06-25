import torch


def params(
    method="MNDO",
    elements=[1, 6, 7, 8],
    parameters=[
        "U_ss",
        "U_pp",
        "zeta_s",
        "zeta_p",
        "beta_s",
        "beta_p",
        "g_ss",
        "g_sp",
        "g_pp",
        "g_p2",
        "h_sp",
        "alpha",
    ],
    root_dir="./params/MOPAC/",
):
    """
    load method parameters from CSV files
    """
    # method=MNDO, AM1, PM3, PM6, OM1, OM2, OM3
    # load the parameters taken from MOPAC or OMx Fortran tables
    # elements: elements needed, not checking on the type, but > 0 and <= 107
    # parameters: parameter lists
    # root_dir : directory for these parameter files
    if method in {"OM1", "OM2", "OM3"}:
        fn = root_dir + "parameters_" + method + ".csv"
    else:
        fn = root_dir + "parameters_" + method + "_MOPAC.csv"
    # will directly use atomic number as array index
    # elements.sort()
    m = max(elements)
    n = len(parameters)
    p = torch.zeros((m + 1, n))
    requested_elements = {int(x) for x in elements if int(x) > 0}
    found_elements = set()
    with open(fn) as f:
        header = f.readline().strip().replace(" ", "").split(",")
        idx = [header.index(item) for item in parameters]
        for line in f:
            t = line.strip().replace(" ", "").split(",")
            element_id = int(t[0])
            if element_id in requested_elements:
                found_elements.add(element_id)
                p[element_id, :] = torch.tensor([float(t[x]) for x in idx])
    if method in {"OM1", "OM2", "OM3"}:
        missing = sorted(requested_elements - found_elements)
        if missing:
            raise ValueError(
                f"{method} parameters are only available for elements with CSV entries; missing atomic numbers: {missing}"
            )
    return torch.nn.Parameter(p, requires_grad=False)


def PWCCT(
    method="MNDO",
    elements=[1, 6, 7, 8],
    parameters=[
        "U_ss",
        "U_pp",
        "zeta_s",
        "zeta_p",
        "beta_s",
        "beta_p",
        "g_ss",
        "g_sp",
        "g_pp",
        "g_p2",
        "h_sp",
        "alpha",
    ],
    root_dir="./params/MOPAC/",
):
    """
    loads the diatomic core-core paramters for PM6
    returns them in q
    """
    # will directly use atomic number as array index just as in the params method
    m = max(elements)
    q = torch.zeros((m + 1, m + 1))
    p = torch.zeros((m + 1, m + 1))

    if method in {"PM6", "PM6_SP"}:
        fo = root_dir + "PWCCT_" + method + "_MOPAC.csv"
        requested_elements = set(elements)
        with open(fo) as f:
            for line in f:
                t = line.strip().replace(" ", "").split(",")
                element_i = int(t[0])
                element_j = int(t[1])
                if element_i in requested_elements and element_j in requested_elements:
                    q[element_i, element_j] = float(t[2])
                    p[element_i, element_j] = float(t[3])
    return torch.nn.Parameter(q, requires_grad=False), torch.nn.Parameter(p, requires_grad=False)

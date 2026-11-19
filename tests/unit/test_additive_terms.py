import torch

from seqm.seqm_functions.cal_par import HPP_MIN, additive_term_rho1, additive_term_rho2, sp_additive_terms
from seqm.seqm_functions.constants import Constants


def test_additive_term_rho1_gradients():
    hsp = torch.tensor([0.7178, 0.4, 2.0], dtype=torch.float64, requires_grad=True)
    D1 = torch.tensor([0.80, 0.55, 0.9], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(additive_term_rho1.apply, (hsp, D1), eps=1.0e-6, atol=1.0e-5, rtol=1.0e-4)


def test_additive_term_rho2_gradients():
    hpp = torch.tensor([1.2, 0.9, 3.0], dtype=torch.float64, requires_grad=True)
    D2 = torch.tensor([0.70, 0.60, 0.8], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(additive_term_rho2.apply, (hpp, D2), eps=1.0e-6, atol=1.0e-5, rtol=1.0e-4)


def test_negative_hpp_is_clamped():
    """PM6 gives F g_pp < g_p2, i.e. hpp < 0, for which the quadrupole additive term does not exist (as in MOPAC)."""
    torch.set_default_dtype(torch.float64)  # Constants() builds its tables in the default dtype
    f64 = dict(dtype=torch.float64)
    Z = torch.tensor([9, 9])
    args = dict(
        zetas=torch.full((2,), 3.0, **f64), zetap=torch.full((2,), 3.0, **f64),
        gss=torch.full((2,), 12.4, **f64), gp2=torch.full((2,), 12.18, **f64), hsp=torch.full((2,), 2.6, **f64),
    )  # fmt: skip
    # hpp = -1.88 for the first atom and exactly the clamp value for the second
    gpp = torch.tensor([8.417, 12.18 + 2 * HPP_MIN], **f64)
    rho_2 = sp_additive_terms(Constants(), Z, gpp=gpp, **args)[4]

    assert torch.isfinite(rho_2).all() and (rho_2 > 0).all()
    assert torch.equal(rho_2[0], rho_2[1])

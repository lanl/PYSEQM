import torch

from seqm.seqm_functions.cal_par import additive_term_rho1, additive_term_rho2


def test_additive_term_rho1_gradients():
    hsp = torch.tensor([0.7178, 0.4, 2.0], dtype=torch.float64, requires_grad=True)
    D1 = torch.tensor([0.80, 0.55, 0.9], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(additive_term_rho1.apply, (hsp, D1), eps=1.0e-6, atol=1.0e-5, rtol=1.0e-4)


def test_additive_term_rho2_gradients():
    hpp = torch.tensor([1.2, 0.9, 3.0], dtype=torch.float64, requires_grad=True)
    D2 = torch.tensor([0.70, 0.60, 0.8], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(additive_term_rho2.apply, (hpp, D2), eps=1.0e-6, atol=1.0e-5, rtol=1.0e-4)

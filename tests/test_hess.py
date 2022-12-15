import torch
from seqm.seqm_functions.constants import Constants, ev_kcalpmol
from seqm.basics import  Energy
from seqm.seqm_functions.parameters import params
import seqm
seqm.seqm_functions.scf_loop.debug = False
seqm.seqm_functions.scf_loop.SCF_BACKWARD_MAX_ITER = 50
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
species = torch.as_tensor([[8,6,1,1]],dtype=torch.int64, device=device)
coordinates = torch.tensor([
                  [
                   [0.0000,    0.0000,    0.0000],
                   [1.22732374,    0.0000,    0.0000],
                   [1.8194841064614802,    0.93941263319067747,    0.0000],
                   [1.8193342232738994,    -0.93951967178254525,    3.0565334533430606e-006]
                  ]
                 ], device=device)
# coordinates.requires_grad_(True)
elements = [0]+sorted(set(species.reshape(-1).tolist()))
seqm_parameters = {
                   'method' : 'AM1',  # AM1, MNDO, PM#
                   'scf_eps' : 1.0e-6,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                   'scf_converger' : [2,0.5], # converger used for scf loop
                                         # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                         # [1], adaptive mixing
                                         # [2], adaptive mixing, then pulay
                   'sp2' : [False, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                            #[True, eps] or [False], eps for SP2 conve criteria
                   'elements' : elements, #[0,1,6,8],
                   'learned' : [], # learned parameters name list, e.g ['U_ss']
                   'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                   'scf_backward': 1,
                   'scf_backward_eps': 1.0e-6,
                   }
const = Constants().to(device)
with torch.autograd.set_detect_anomaly(False):
    eng = Energy(seqm_parameters).to(device)
    x0 = coordinates.reshape(-1)
    x0.requires_grad_(True)
    def func(x):
        Hf, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, notconverged = eng(const, x.reshape(1,-1,3), species, all_terms=True)
        return Hf
    func(x0)
    #grad = torch.autograd.grad(Etot.sum(), coordinates, create_graph=True, retain_graph=True)[0]
    hess = torch.autograd.functional.hessian(func, x0)
    print((hess-hess.transpose(0,1)).abs().max().item())
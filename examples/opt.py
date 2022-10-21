import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import ase.io
import numpy as np
import torch
import seqm
seqm.seqm_functions.scf_loop.debug=False
seqm.seqm_functions.scf_loop.MAX_ITER=5000
import hippynn
import scipy.optimize


torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

xyz = "mol1.xyz" #sys.argv[1]
mol = ase.io.read(xyz)

Z = mol.get_atomic_numbers()
n_atom = Z.shape[0]
# make sure the atomic number is sorted in descending order
indx = len(Z) - 1 - np.argsort(Z[::-1], kind='stable')[::-1]
Z = Z[indx]
R = mol.get_positions()[indx]
mol.set_positions(R)
mol.set_atomic_numbers(Z)
mol.write("mol_orig.xyz")

Z = torch.tensor(Z, dtype=torch.int64, device=device).unsqueeze(0)

R = torch.tensor(R, device=device).unsqueeze(0)

model = torch.load("./model/model.pt", map_location=device)
model.eval()


# adjust the convergency criteria for the SCF loop inside PYSEQM
model.node_from_name("SEQM_Energy").torch_module.energy.hamiltonian.eps.data = torch.tensor(1.0e-10, dtype=torch.float64, device=device)

# build the predictor
#input_nodes = [model.node_from_name('Z_long'), model.node_from_name('R')]
#print(model.input_nodes)
input_nodes = [model.input_nodes[1], model.input_nodes[2]]
output_nodes = [model.node_from_name('SEQM_Energy.mol_energy'), model.node_from_name("gradients")]
p  = hippynn.graphs.Predictor(input_nodes, output_nodes)


step = np.array(0,dtype=np.int64)

def predict(Z, R):
    node_values = {input_nodes[0]: Z, input_nodes[1]: R}
    y = p.predict_all(node_values)
    e = y[output_nodes[0]].detach().cpu().numpy().reshape(-1)
    g = y[output_nodes[1]].detach().cpu().numpy().reshape(-1)
    return e, g

def func(x, step):
    r = torch.tensor(x, dtype=torch.double, device=device).reshape(1,n_atom,3)
    e = predict(Z, r)[0]
    print(step, 'E: ', e[0])
    step += 1
    return e

def jac(x, step):
    r = torch.tensor(x, dtype=torch.double, device=device).reshape(1,n_atom,3)
    dx = predict(Z, r)[1]
    print(step, " Max Force: ", np.max(np.abs(dx)))
    return dx


x0 = R.detach().cpu().numpy().reshape(-1)

res = scipy.optimize.minimize(func, x0, args=(step, ), method='L-BFGS-B', jac=jac, options={'disp':True,'maxiter':50000, 'gtol':1.0e-7, 'ftol':1.0e-14})

print(res.success)

mol.set_positions(res.x.reshape(-1,3))
mol.write("opt.xyz")


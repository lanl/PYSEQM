import ase.io
import torch
import hippynn
import numpy as np

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

mol = ase.io.read("mol1.xyz")

Z = mol.get_atomic_numbers()
# make sure the atomic number is sorted in descending order
indx = len(Z) - 1 - np.argsort(Z[::-1], kind='stable')[::-1]
Z = Z[indx]
R = mol.get_positions()[indx]
mol.set_positions(R)
mol.set_atomic_numbers(Z)
mol.write("mol_orig.xyz")

Z = torch.tensor(Z, dtype=torch.int64, device=device).unsqueeze(0)
R = torch.tensor(R, dtype=torch.float64, device=device).unsqueeze(0)

model = torch.load("./model/model.pt", map_location=device)
model.eval()

#input_nodes = [model.node_from_name('Z_long'), model.node_from_name('R')]
#print(model.input_nodes)
input_nodes = [model.input_nodes[1], model.input_nodes[2]]
output_nodes = [model.node_from_name('SEQM_Energy.mol_energy'), model.node_from_name("gradients")]
model.node_from_name("SEQM_Energy").torch_module.energy.hamiltonian.eps.data = torch.tensor(1.0e-10, dtype=torch.float64, device=device)

p  = hippynn.graphs.Predictor(input_nodes, output_nodes)
node_values = {input_nodes[0]: Z, input_nodes[1]: R}
y = p.predict_all(node_values)

energy = y[output_nodes[0]]
grad = y[output_nodes[1]]
print(energy, grad)
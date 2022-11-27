import torch

from seqm.seqm_functions.constants import Constants, ev_kcalpmol
from seqm.basics import  Parser, Hamiltonian, Pack_Parameters, Energy
from seqm.seqm_functions.parameters import params
from os import path

here = path.abspath(path.dirname(__file__))

#check code to produce energy terms for each molecule
# with a 'learned' given parameters


torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


species = torch.as_tensor([[8,6,1,1],[8,6,1,1],[8,8,6,0]],dtype=torch.int64, device=device)

coordinates = torch.tensor([
                  [
                   [0.0000,    0.0000,    0.0000],
                   [1.22732374,    0.0000,    0.0000],
                   [1.8194841064614802,    0.93941263319067747,    0.0000],
                   [1.8193342232738994,    -0.93951967178254525,    3.0565334533430606e-006]
                  ],
                  [
                   [0.0000,    0.0000,    0.0000],
                   [1.22732374,    0.0000,    0.0000],
                   [1.8194841064614802,    0.93941263319067747,    0.0000],
                   [1.8193342232738994,    -0.93951967178254525,    3.0565334533430606e-006]
                  ],
                  [
                   [0.0000,    0.0000,    0.0000],
                   [1.22732374,    0.0000,    0.0000],
                   [1.8194841064614802,    0.93941263319067747,    0.0000],
                   [0.0,0.0,0.0]
                  ]
                 ], device=device)

coordinates.requires_grad_(True)

elements = [0]+sorted(set(species.reshape(-1).tolist()))
seqm_parameters = {
                   'method' : 'AM1',  # AM1, MNDO, PM#
                   'scf_eps' : 1.0e-6,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                   'scf_converger' : [2,0.0], # converger used for scf loop
                                         # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                         # [1], adaptive mixing
                                         # [2], adaptive mixing, then pulay
                   'sp2' : [True, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                            #[True, eps] or [False], eps for SP2 conve criteria
                   'elements' : elements, #[0,1,6,8],
                   'learned' : ['U_ss'], # learned parameters name list, e.g ['U_ss']
                   'parameter_file_dir' : here+'/../seqm/params/', # file directory for other required parameters
                   'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                   }


const = Constants().to(device)



#parser is not needed here, just use it to get Z and create "learned" parameters
#prepare a fake learned parameters: learnedpar
parser = Parser(seqm_parameters).to(device)
nmol, molsize, \
nHeavy, nHydro, nocc, \
Z, maskd, atom_molid, \
mask, pair_molid, ni, nj, idxi, idxj, xij, rij = parser(const, species, coordinates)
#add learned parameters
#here use the data from mopac as example

p=params(method=seqm_parameters['method'],
         elements=seqm_parameters['elements'],
         root_dir=seqm_parameters['parameter_file_dir'],
         parameters=seqm_parameters['learned']).to(device)
p, =p[Z].transpose(0,1).contiguous()
p.requires_grad_(True)
learnedpar = {'U_ss':p}


#"""
#this part is same as in module Energy().forward()
packpar = Pack_Parameters(seqm_parameters).to(device)
parameters = packpar(Z,learned_params=learnedpar)
hamiltonian = Hamiltonian(seqm_parameters).to(device)
F, e, P, Hcore, w, v, notconverged = hamiltonian(const, molsize, nHeavy, nHydro, nocc, Z, maskd, mask, atom_molid, pair_molid, idxi, idxj, ni,nj,xij,rij, parameters)

'''
d = Hcore[0,(0,1,2,3,4,5,6,7,8,12)][:,(0,1,2,3,4,5,6,7,8,12)].transpose(0,1).tolist()
for i in range(len(d)):
    for j in range(i+1):
        print("%11.6f" % d[i][j], end="")
    print()
#'''
"""
with open('tmp1.txt', 'w') as f:
    for i in range(6):
        f.write("%d %d\n" % (ni[i],nj[i]))
        for j in range(10):
            for k in range(10):
                f.write("%f " % w[i,j,k])
            f.write('\n')
#"""
seqm_parameters['eig']=True
with torch.autograd.set_detect_anomaly(True):
    eng = Energy(seqm_parameters).to(device)
    Hf, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, notconverged = eng(const, coordinates, species, learned_parameters=learnedpar, all_terms=True)
#

    L=Etot.sum()
    #L.backward()
    #"""
    coordinates.grad=torch.autograd.grad(L,coordinates,
                                         #grad_outputs=torch.tensor([1.0]),
                                         create_graph=True,
                                         retain_graph=True)[0]
    #"""

print("Orbital Energy (eV): ", e.tolist())
print("Electronic Energy (eV): ", Eelec.tolist())
print("Nuclear Energy (eV): ", Enuc.tolist())
print("Total Energy (eV): ", Etot.tolist())
print("Heat of Formation (kcal/mol): ", (Hf*ev_kcalpmol).tolist())

print(coordinates.grad)
#print(p.grad)
#"""
if const.do_timing:
    print(const.timing)

import torch

from seqm.seqm_functions.constants import Constants
from seqm.seqm_functions.parameters import params
from seqm.basics import Parser, Pack_Parameters, Energy



torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

N = 100 #1000
N += 1
#percentage of the original vaslue
dxmin = -0.1
dxmax = 0.1

dx = torch.arange(N+0.0,device=device)*(dxmax-dxmin)/(N-1.0)+dxmin

const = Constants().to(device)

species = torch.as_tensor([[8,6,1,1]],dtype=torch.int64, device=device) \
               .expand(N,4)

coordinates = torch.tensor([
             [
              [0.014497983896917479, 3.208059775069048e-05, -1.0697192017402962e-07],
              [1.3364260303072648, -3.2628339194439124e-05, 8.51016890853131e-07],
              [1.757659914731728, 1.03950803854101, -5.348699815983099e-07],
              [1.7575581407994696, -1.039614529391432, 2.84735846426227e-06]
             ],
             ], device=device).expand(N,4,3)
#


elements = [0]+sorted(set(species.reshape(-1).tolist()))
seqm_parameters = {
                   'method' : 'AM1',  # AM1, MNDO, PM#
                   'scf_eps' : 1.0e-6,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                   'scf_converger' : [2,0.0], # converger used for scf loop
                                         # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                         # [1], adaptive mixing
                                         # [2], adaptive mixing, then pulay
                   'sp2' : [False, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                            #[True, eps] or [False], eps for SP2 conve criteria
                   'elements' : elements, #[0,1,6,8],
                   'learned' : ['U_ss'], # learned parameters name list, e.g ['U_ss']
                   'parameter_file_dir' : '../../params/MOPAC/', # file directory for other required parameters
                   'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                   }


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
         parameters=seqm_parameters['learned'],).to(device)
p, =p[Z].transpose(0,1).contiguous()
t=p[0:-1:4]*(dx+1.0)
p[0:-1:4]=t

p.requires_grad_(True)
learnedpar = {seqm_parameters['learned'][0]:p}


with torch.autograd.set_detect_anomaly(True):
    eng = Energy(seqm_parameters).to(device)
    Hf, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, notconverged = eng(const, coordinates, species, learned_parameters=learnedpar, all_terms=True)
    L=Etot.sum()
    L.backward()

tg=p.grad[0:-1:4]


f=open('log.dat', 'w')
f.write("#index, par, energy (eV), grad\n")
for i in range(N):
    f.write("%d %12.8e %12.8e %12.8e\n" % (i,t[i],Etot[i],tg[i] ))
f.close()

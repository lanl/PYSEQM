import torch

from seqm.seqm_functions.constants import Constants
from seqm.MolecularDynamics import *
from seqm.XLBOMD import XL_BOMD

#check MD
#velocity verlet algorithm
#start from no thermostats
C=[]
H=[]
with open("nanostar.xyz") as f:
    next(f)
    next(f)
    for l in f:
        t=l.strip().split()
        if t[0]=='C':
            C.append([float(x) for x in t[1:]])
        else:
            H.append([float(x) for x in t[1:]])

nC = len(C)
nat = len(C) + len(H)


torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

species = torch.ones((1,nat),dtype=torch.int64, device=device)
species[0,:nC] = 6

coordinates = torch.zeros((1,nat,3), device=device)
coordinates[0,:nC] = torch.tensor(C, device=device)
coordinates[0,nC:] = torch.tensor(H, device=device)

const = Constants().to(device)
#may need to add scaling factor for length and energy on const, check constants.py

elements = [0]+sorted(set(species.reshape(-1).tolist()))
seqm_parameters = {
                   'method' : 'AM1',  # AM1, MNDO, PM#
                   'scf_eps' :  27.211386e-5,  # unit eV, change of electronic energy, as nuclear energy doesnt' change during SCF
                   'scf_converger' : [1,0.0], # converger used for scf loop
                                         # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                         # [1], adaptive mixing
                                         # [2], adaptive mixing, then pulay
                   'sp2' : [False, 1.0e-2],  # whether to use sp2 algorithm in scf loop,
                                            #[True, eps] or [False], eps for SP2 conve criteria
                   'elements' : elements, #[0,1,6,8],
                   'learned' : [], # learned parameters name list, e.g ['U_ss']
                   #'parameter_file_dir' : '../../seqm/params/', # file directory for other required parameters
                   'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                   }
#


#"""
md =  Molecular_Dynamics_Basic(seqm_parameters, timestep=1.0).to(device)
velocities = md.initialize_velocity(const, coordinates, species, Temp=300.0)
#md =  Molecular_Dynamics_Langevin(seqm_parameters, timestep=1.0, damp=100.0, T=300.0)
with torch.autograd.set_detect_anomaly(True):
    coordinates, velocities, accelaration=  md.run(const, 10, coordinates, velocities, species)
#"""
"""
md =  XL_BOMD(seqm_parameters, timestep=0.5, k=5)
velocities = md.initialize_velocity(const, coordinates, species, Temp=300.0)
#remove center of mass velocity
with torch.autograd.set_detect_anomaly(True):
    coordinates, velocities, accelaration, P, Pt =  md.run(const, 10, coordinates, velocities, species)
#"""


print(const.timing)

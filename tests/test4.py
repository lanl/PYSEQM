import torch

from seqm.seqm_functions.constants import Constants
from seqm.MolecularDynamics import Molecular_Dynamics_Basic, Molecular_Dynamics_Langevin
from seqm.XLBOMD import XL_BOMD

#check MD
#velocity verlet algorithm
#start from no thermostats


torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

species = torch.as_tensor([[8,6,1,1],[8,6,1,1]],dtype=torch.int64, device=device)[:1]

#optimized structure from test3_hippynn.py
coordinates = torch.tensor([
                  [
                   [0.014497970281389074, 3.208059797520201e-05, -1.0697192468126102e-07],
                   [1.3364260161590171, -3.26283382508022e-05, 8.510168803526663e-07],
                   [1.7576599286132542, 1.0395080227523756, -5.348699492766755e-07],
                   [1.757558154681721, -1.039614513603968, 2.8473584469483316e-06]
                  ],
                  [
                   [0.014497970281389074, 3.208059797520201e-05, -1.0697192468126102e-07],
                   [1.3364260161590171, -3.26283382508022e-05, 8.510168803526663e-07],
                   [1.7576599286132542, 1.0395080227523756, -5.348699492766755e-07],
                   [1.757558154681721, -1.039614513603968, 2.8473584469483316e-06]
                  ]
                 ], device=device)[:1]


const = Constants().to(device)
#may need to add scaling factor for length and energy on const, check constants.py

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
                   'learned' : [], # learned parameters name list, e.g ['U_ss']
                   #'parameter_file_dir' : '../seqm/params/', # file directory for other required parameters
                   'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                   }
#

output={'molid':[0], 'thermo':1, 'dump':1, 'prefix':'md'}

md0 =  Molecular_Dynamics_Basic(seqm_parameters, timestep=1.0,output=output).to(device)
md1 =  Molecular_Dynamics_Langevin(seqm_parameters, timestep=1.0, damp=100.0, Temp=300.0, output=output).to(device)
md2 = XL_BOMD(seqm_parameters, timestep=1.0, k=9,output=output).to(device)
velocities = md0.initialize_velocity(const, coordinates, species, Temp=300.0)
#remove center of mass velocity
md=md2
#coordinates, velocities, accelaration =  md1.run(const, 10, coordinates, velocities, species)
#coordinates, velocities, accelaration =  md1.run(const, 20, coordinates, velocities, species)
#coordinates, velocities, accelaration, P, Pt =  md2.run(const, 100, coordinates, velocities, species)
_ = md.run(const, 10, coordinates, velocities, species)
_ = md.run(const, 10, coordinates, velocities, species, control_energy_shift=True)
_ = md.run(const, 10, coordinates, velocities, species, scale_vel=[1,300])


if const.do_timing:
    print(const.timing)
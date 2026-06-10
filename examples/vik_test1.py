import warnings

import torch

from seqm.api import Constants, Electronic_Structure, Molecule, read_xyz

warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

### create molecule object:
species = torch.as_tensor(
    [
        # [8,1,1],
        # [6, 1, 1, 1, 1]
        # [1,1,],
        # [1,1,0,0],
        # [8,6,],
        # [8,6,1,1],
        [6,6,1,1,1,1]
        # [8,8,6,0]
    ],  # zero-padding for batching
    dtype=torch.int64,
    device=device,
)

coordinates = torch.tensor(
    [
        #  [
        # [  0.000000000 ,    0.000000000  ,   0.000000],
        # [  0.957200    ,    0.000000000  ,   0.000000],
        # [  1.197187    ,    0.926627     ,   0.000000],
        # ]
        # [
        #     [-2.7878725, 1.3475389, 0.0000000],
        #     [-2.2590725, 1.5085389, 0.9359000],
        #     [-2.5827725, 2.1715389, -0.6786000],
        #     [-2.4533725, 0.4161389, -0.4496000],
        #     [-3.8563725, 1.2938389, 0.1921000],
        # ]
        # [
        #     [0.0,0.0,0.0],
        #     [1.1282,0.0,0.0]
        #     ]
        # [
        #  [0.82,    0.00,    0.00],
        #  [1.82,    0.94,    0.00],
        #  [1.81,   -0.94,    0.10],
        # # [0.0,0.0,0.0],            # zero-padding for batching
        # # [0.0,0.0,0.0],            # zero-padding for batching
        # ],
        # [
        #  [0.00,    0.00,    0.00],
        #  [1.23,    0.00,    0.00],
        #  [1.82,    0.94,    0.00],
        #  [1.81,    -0.94,    0.00],
        #  # [0.0,0.0,0.0],            # zero-padding for batching
        # ]
        
# [[0.000000 ,           0.000000 ,           0.000000 ],
# [1.216287 ,           0.000000 ,           0.000000 ],
# [1.827198 ,           0.922174 ,           0.000000 ],
# [1.827198 ,          -0.922174 ,           0.000000 ],],
[[ 0.00000 ,   0.66819 ,   0.00000],
[ 0.00000 ,  -0.66819 ,   0.00000],
[ 0.66414 ,   1.23830 ,  -0.64135],
[ 0.66414 ,  -1.23830 ,   0.64135],
[-0.66414 ,   1.23830 ,   0.64135],
[-0.66414 ,  -1.23830 ,  -0.64135],],

    ],
    device=device,
)
# species, coordinates = read_xyz(["/Users/vishikh/onedrive/calculations/CNT_10.xyz"])
# species, coordinates = read_xyz(['./examples/methane.xyz'])
# species, coordinates = read_xyz(['./methane.xyz'])
species = torch.as_tensor(species, dtype=torch.int64, device=device)
coordinates = torch.as_tensor(coordinates, device=device)

const = Constants().to(device)
active_state = 1

seqm_parameters = {
    "method": "OM2",  # AM1, MNDO, PM#
    # "method": "AM1",  # AM1, MNDO, PM#
    "scf_eps": 1.0e-8,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
    "scf_converger": [2],  # converger used for scf loop
    # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
    # [1], adaptive mixing
    # [2], adaptive mixing, then pulay
    # 'uhf' : True,
    # "excited_states": {"n_states": 9, "method": "cis"},
    # "active_state": active_state,
    # 'scf_backward': 1,
    # "analytical_gradient": [True],
}

molecules = Molecule(const, seqm_parameters, coordinates, species).to(device)

### Create electronic structure driver:
esdriver = Electronic_Structure(seqm_parameters).to(device)

esdriver(molecules,do_force=False)

print(' Total Energy (eV):\n', molecules.Etot)
# print(f"Dipoles\n{molecules.dipole}")
# print(f"Charges:\n{molecules.q}")
print('\n Electronic Energy (eV): ', molecules.Eelec)
print('\n Nuclear Energy (eV):\n', molecules.Enuc)
print('\n Heat of Formation (ev):\n', molecules.Hf)
print('\n Orbital energies (eV):\n', molecules.e_mo)

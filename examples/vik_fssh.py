"""
Minimal single-trajectory surface-hopping (FSSH) example, mirroring vik_esmd.py
but using nonadiabatic dynamics.
"""

import torch

from seqm.api import Constants, Molecule, SurfaceHoppingDynamics

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Formaldehyde-like geometry (O-C-H-H)
species = torch.as_tensor(
    [
        [8, 6, 1, 1],
        [8, 6, 1, 1],
        # [8,8,6,0],
    ],
    dtype=torch.long,
    device=device,
)

# fmt: off
coordinates = torch.tensor(
    [
        [[0.00, 0.00, 0.00],
         [1.22, 0.00, 0.00],
         [1.82, 0.94, 0.00],
         [1.82, -0.94, 0.00]],

        [[0.00, 0.00, 0.00],
         [1.21, 0.00, 0.00],
         [1.62, 0.94, 0.00],
         [1.82, -0.84, 0.00]],

    ],
    device=device,
)

const = Constants().to(device)

seqm_parameters = {
    "method": "AM1",
    "scf_eps": 1.0e-8,
    "scf_converger": [1],
    "excited_states": {
        "n_states": 5,
        "method": "cis",
        "tolerance": 1e-6,
    },
    "analytical_gradient": [True],
    # "nonadiabatic": {
    #     "compute_nac": True,  # enable NAC vectors for hopping
    #     "force_mode": "active",  # forces on active surface only
    #     "recompute_on_hop": True,  # refresh forces after accepted hops
    #     "skip_first_step_prop": True,  # skip electronic propagation on step 0
    # },
}

timestep = 0.1  # fs
output = {
    "molid": [0],
    "prefix": f"./examples/Outputs/vik_fssh.step_{timestep:.1f}",
    "print every": 1,
    "xyz": 1,
    "h5": {"data": 1, "velocities": 1, "coordinates": 1, "forces": 1, "nonadiabatic": 1},
    "checkpoint every": 0,
}

torch.manual_seed(42)
molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
# Set the same initial velocity as the NEXMD input file
with torch.no_grad():
    molecule.velocities = torch.tensor(
    [[-20.9603, -0.5334,  0.4491  ],  
    [-21.2032, -0.2912, -1.8145  ],
    [20.0957 ,-2.0413 , 7.2473   ],
    [ 20.4869, 13.9784,  7.2473  ]],
    device=device
    )*1e-3
    molecule.velocities = molecule.velocities.unsqueeze(0).repeat(molecule.coordinates.shape[0],1,1)


initial_excited_state = 3  # 1-based (1 -> S1, 2 -> S2, ...)

dyn = SurfaceHoppingDynamics(
    seqm_parameters,
    timestep=timestep,
    Temp=300.0,  # initialize velocities at 300 K
    initial_state=initial_excited_state,
    output=output,
).to(device)

steps = 10
dyn.run(molecule, steps, reuse_P=True, remove_com=None)

if dyn.hop_log:
    print("Hop events (step, from->to, accepted):")
    for event in dyn.hop_log:
        status = "accepted" if event.accepted else "frustrated"
        print(f"  step {event.step:4d}: S{event.from_state + 1} -> S{event.to_state + 1} ({status})")
else:
    print("No hops recorded.")

import torch

from seqm.api import XL_BOMD, XL_ESMD, Constants, Molecular_Dynamics_Basic, Molecule, read_xyz

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# torch.set_num_threads(1)
# Molecular_Dynamics_Basic.run_from_checkpoint(
#     "./examples/Outputs/vik_ckpt_esmd.step_0.1.restart.pt", device=device
# )
# exit()


species = torch.as_tensor(
    [
        [8, 6, 1, 1],
        [8, 6, 1, 1],
        # [8,8,6,0],
    ],
    dtype=torch.int64,
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

one = True
# one = True
if one:
    species = torch.as_tensor([[8, 6, 1, 1]], dtype=torch.int64, device=device)

    coordinates = torch.tensor(
        [[[0.00, 0.00, 0.00],
          [1.21, 0.00, 0.00],
          [1.62, 0.94, 0.00],
          [1.82, -0.84, 0.00]]],
         device=device
    )

# species, coordinates = read_xyz(["/Users/vishikh/projects/PYSEQM/examples/benzene.xyz"])
species, coordinates = read_xyz(["/Users/vishikh/onedrive/PYSEQM/xyz_outputs/cyclopropene.xyz"])
species = torch.as_tensor(species, dtype=torch.int64, device=device)
coordinates = torch.as_tensor(coordinates, device=device)

const = Constants().to(device)

seqm_parameters = {
    "method": "AM1",
    "scf_eps": 1.0e-8,
    "scf_converger": [2],
    "excited_states": {"n_states": 3, "cis_tol": 1e-9},
    "active_state": 1,
    "scf_backward": 1,
    # "analytical_gradient": [True],
}

# timestep = 3.0
timestep = 0.5
md_type = "ksaxlesmd"  # "esmd", "xlbomd", "xlesmd", "ksaxlesmd"
# md_type = "esmd"  # "esmd", "xlbomd", "xlesmd", "ksaxlesmd"

output = {
    # 'molid': [0,1],
    "molid": [0],
    "prefix": f"./examples/Outputs/vik_{md_type}.step_{timestep:.1f}",
    "print every": 1,
    "xyz": 1,
    "h5": {
        "data": 1,  # write T/Ek/Ep, excitations, MO, etc.; 0 disables
        "velocities": 1,  # write vel/forces/coords; 0 disables
        # "write_mo": True,
        "coordinates": 1,
    },
    "checkpoint every": 0,
}

torch.manual_seed(42)
molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)

# md = Molecular_Dynamics_Langevin( damp=50.0, seqm_parameters=seqm_parameters,
#                                            Temp=300.0, timestep=timestep,
#                                            output=output).to(device)
xl_bomd_params = {"k": 6}

temp = 0.0
if md_type == "esmd":
    md = Molecular_Dynamics_Basic(
        seqm_parameters=seqm_parameters, Temp=temp, timestep=timestep, output=output
    ).to(device)
elif md_type == "xlbomd":
    md = XL_BOMD(
        xl_bomd_params=xl_bomd_params, Temp=temp, seqm_parameters=seqm_parameters, timestep=timestep, output=output
    ).to(device)
elif md_type == "xlesmd" or md_type=="ksaxlesmd":
    if md_type=="ksaxlesmd":
        xl_bomd_params={'k':6, 'max_rank':4, 'err_threshold':1e-14, 'T_el':1500}
    md = XL_ESMD(
        xl_bomd_params=xl_bomd_params, Temp=temp, seqm_parameters=seqm_parameters, timestep=timestep, output=output
    ).to(device)
else:
    raise ValueError(f"Unknown md_type '{md_type}'. Use 'basic', 'xl_bomd', or 'xl_esmd'.")

_ = md.run(molecule, 500, remove_com=None, reuse_P=False,dmprop="SCF")

# fmt: on

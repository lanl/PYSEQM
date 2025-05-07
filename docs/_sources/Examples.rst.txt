Examples 
============


Excited States RIS
------------------------------

.. code-block:: python


    import torch
    from seqm.seqm_functions.constants import Constants
    from seqm.Molecule import Molecule
    from seqm.ElectronicStructure import Electronic_Structure

    import warnings
    warnings.filterwarnings("ignore")

    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define species and coordinates
    species = torch.as_tensor([[6, 1, 1, 1, 1]], dtype=torch.int64, device=device)
    coordinates = torch.tensor([[
        [ 0.00000,  0.00000,  0.00000], 
        [ 0.00000,  0.00000,  1.08900],
        [ 1.02672,  0.00000, -0.36300],
        [-0.51336, -0.88916, -0.36300],
        [-0.51336,  0.88916, -0.36300],
    ]], device=device)

    # Set up constants and parameters
    const = Constants().to(device)
    elements = [0] + sorted(set(species.reshape(-1).tolist()))

    seqm_parameters = {
        'method': 'AM1',
        'scf_eps': 1.0e-6,
        'scf_converger': [2, 0.0],
        'sp2': [False, 1.0e-5],
        'elements': elements,
        'learned': [],
        'pair_outer_cutoff': 1.0e10,
        'eig': True,
        'excited_states': {'n_states': 10, 'method': 'rpa'},
    }

    # Create molecule and electronic structure driver
    molecules = Molecule(const, seqm_parameters, coordinates, species).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecules)




Excited States CIS
------------------------------

.. code-block:: python


    import torch
    from seqm.seqm_functions.constants import Constants
    from seqm.Molecule import Molecule
    from seqm.ElectronicStructure import Electronic_Structure

    import warnings
    warnings.filterwarnings("ignore")

    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define species and coordinates
    species = torch.as_tensor([[6, 1, 1, 1, 1]], dtype=torch.int64, device=device)
    coordinates = torch.tensor([[
        [ 0.00000,  0.00000,  0.00000], 
        [ 0.00000,  0.00000,  1.08900],
        [ 1.02672,  0.00000, -0.36300],
        [-0.51336, -0.88916, -0.36300],
        [-0.51336,  0.88916, -0.36300],
    ]], device=device)

    # Set up constants and parameters
    const = Constants().to(device)
    elements = [0] + sorted(set(species.reshape(-1).tolist()))

    seqm_parameters = {
        'method': 'AM1',
        'scf_eps': 1.0e-6,
        'scf_converger': [2, 0.0],
        'sp2': [False, 1.0e-5],
        'elements': elements,
        'learned': [],
        'pair_outer_cutoff': 1.0e10,
        'eig': True,
        'excited_states': {'n_states': 10, 'method': 'cis'},
    }

    # Create molecule and electronic structure driver
    molecules = Molecule(const, seqm_parameters, coordinates, species).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecules)






BOMD
------------------------------

.. code-block:: python



    import torch
    from seqm.seqm_functions.constants import Constants
    from seqm.Molecule import Molecule
    from seqm.MolecularDynamics import Molecular_Dynamics_Basic
    from seqm.MolecularDynamics import Molecular_Dynamics_Langevin


    # Use double precision
    torch.set_default_dtype(torch.float64)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define atomic species and coordinates
    species = torch.tensor([[8, 6, 1, 1], [5, 1, 1, 1]], dtype=torch.int64, device=device)

    coordinates = torch.tensor(
        [
            [
                [0.00, 0.0, 0.0],
                [1.22, 0.0, 0.0],
                [1.82, 0.94, 0.0],
                [1.82, -0.94, 0.0],
            ],
            [
                [0.00, 0.00, 0.00],
                [1.20, 0.00, 0.00],
                [-0.60, 1.03, 0.00],
                [-0.60, -1.03, 0.00],
            ],
        ],
        device=device,
    )

    # Constants and SEQM parameters
    const = Constants().to(device)
    elements = [0] + sorted(set(species.reshape(-1).tolist()))

    seqm_parameters = {
        "method": "AM1",
        "scf_eps": 1.0e-6,
        "scf_converger": [2, 0.0],
        "sp2": [False, 1.0e-5],
        "elements": elements,
        "learned": [],
        "pair_outer_cutoff": 1.0e10,
        "eig": True,
    }

    # Output settings
    output = {
        "molid": [0, 1],
        "thermo": 1,
        "dump": 1,
        "prefix": "Outputs/MD_BOMD",
    }

    # Create molecule object
    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)

    # Example 1: Basic NVE dynamics
    md_nve = Molecular_Dynamics_Basic(
        seqm_parameters=seqm_parameters, Temp=400.0, timestep=0.4, output=output
    ).to(device)
    md_nve.initialize_velocity(molecule)
    md_nve.run(molecule, steps=10, remove_com=[True, 1], Info_log=True)

    # Example 2: NVE with energy shift compensation
    output["prefix"] = "Outputs/MD_BOMD_Energy_Control"
    md_energy_control = Molecular_Dynamics_Basic(
        seqm_parameters=seqm_parameters, Temp=400.0, timestep=0.4, output=output
    ).to(device)
    md_energy_control.initialize_velocity(molecule)
    md_energy_control.run(
        molecule, steps=10, control_energy_shift=True, remove_com=[True, 1], Info_log=True
    )

    # Example 3: NVT with temperature control
    output["prefix"] = "Outputs/MD_BOMD_Temp_Control"
    md_temp_control = Molecular_Dynamics_Basic(
        seqm_parameters=seqm_parameters, Temp=400.0, timestep=0.4, output=output
    ).to(device)
    md_temp_control.initialize_velocity(molecule)
    md_temp_control.run(
        molecule, steps=10, scale_vel=[1, 400], remove_com=[True, 1], Info_log=True
    )

    # Example 4: Langevin dynamics
    output["prefix"] = "Outputs/MD_BOMD_Langevin"
    md_langevin = Molecular_Dynamics_Langevin(
        damp=100.0, seqm_parameters=seqm_parameters, Temp=400.0, timestep=0.4, output=output
    ).to(device)
    md_langevin.initialize_velocity(molecule)
    md_langevin.run(molecule, steps=10, remove_com=[True, 1], Info_log=True)





XL-BOMD
------------------------------

.. code-block:: python


    import torch
    from seqm.seqm_functions.constants import Constants
    from seqm.Molecule import Molecule
    from seqm.MolecularDynamics import KSA_XL_BOMD

    # Set default tensor precision
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define species and coordinates
    species = torch.tensor([[8, 6, 1, 1], [5, 1, 1, 1]], dtype=torch.int64, device=device)

    coordinates = torch.tensor([
        [
            [0.00, 0.0, 0.0],
            [1.22, 0.0, 0.0],
            [1.82, 0.94, 0.0],
            [1.82, -0.94, 0.0],
        ],
        [
            [0.00, 0.00, 0.00],
            [1.20, 0.00, 0.00],
            [-0.60, 1.03, 0.00],
            [-0.60, -1.03, 0.00],
        ]
    ], device=device)

    # Load constants and configure parameters
    const = Constants().to(device)
    elements = [0] + sorted(set(species.reshape(-1).tolist()))

    seqm_parameters = {
        'method': 'AM1',
        'scf_eps': 1.0e-6,
        'scf_converger': [2, 0.0],
        'sp2': [False, 1.0e-5],
        'elements': elements,
        'learned': [],
        'pair_outer_cutoff': 1.0e10,
        'eig': True
    }

    # Output and KSA-XL-BOMD specific parameters
    output = {
        'molid': [0, 1],
        'thermo': 1,
        'dump': 1,
        'prefix': 'Outputs/KSA_XL_BOMD'
    }

    xl_bomd_params = {
        'k': 6,
        'max_rank': 3,
        'err_threshold': 0.0,
        'T_el': 1500
    }

    # Initialize molecule and dynamics engine
    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)

    md = KSA_XL_BOMD(
        xl_bomd_params=xl_bomd_params,
        seqm_parameters=seqm_parameters,
        Temp=400.0,
        timestep=0.4,
        output=output
    ).to(device)

    # Initialize velocity and run dynamics
    md.initialize_velocity(molecule)
    md.run(molecule, steps=10, remove_com=[True, 1], Info_log=True)
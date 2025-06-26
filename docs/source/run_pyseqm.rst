Run PYSEQM
==========

Introduction
------------
To perform any calculation with PYSEQM, you write a self-contained Python script, which is your **input file**, that:

1. Imports core modules and utilities  
2. Sets numerical precision and selects CPU or GPU  
3. Defines your molecular system (species and coordinates)  
4. Specifies SEQM parameters (parameters for your semiempirical quantum chemistry calculation)  
5. Chooses and runs a calculation driver (SCF, excited states, or MD)

Workflow Overview
-----------------
1. **Create** a new file (e.g. `run_pyseqm.py`)  
2. **Copy** the template below and **paste** into your file  
3. **Customize** species, coordinates, and parameters  
4. **Run** with:

.. code-block:: bash

   python run_pyseqm.py

Minimal Input File Template
---------------------------

.. code-block:: python

   # 1. Imports modules and utilities  
   import torch
   from seqm.seqm_functions.constants import Constants
   from seqm.Molecule import Molecule
   # Choose one driver:
   from seqm.ElectronicStructure import Electronic_Structure
   # from seqm.MolecularDynamics import Molecular_Dynamics_Basic

   # 2. Precision & device
   torch.set_default_dtype(torch.float64)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # 3. Molecular system
   #    - species: (n_mols × n_atoms) integer tensor of atomic numbers
   #    - coordinates: (n_mols × n_atoms × 3) float tensor in Å
   species = torch.as_tensor([[6, 6, 1, 1]], dtype=torch.int64, device=device)
   coordinates = torch.tensor(
       [[[0.0, 0.0, 0.0],
         [1.2, 0.0, 0.0],
         [0.0, 1.2, 0.0],
         [0.0, 0.0, 1.2]]],
       dtype=torch.float64, device=device
   )

   # 4a. SEQM parameters
   seqm_parameters = {
       'method': 'AM1',            # MNDO, PM3, PM6, etc.
       'scf_eps': 1e-8,            # SCF convergence threshold (eV)
       'scf_converger': [0, 0.1],  # SCF algorithm
   }

   # 4b. Constants & Molecule
   const = Constants().to(device)
   molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)


   # 5a. Run single-point SCF
   driver = Electronic_Structure(seqm_parameters).to(device)
   driver(molecule)
   print("Total Energy (eV):", molecule.Etot)
   print("Forces (eV/Å):",   molecule.force)

   # # 5b. (Alternative) Run a BOMD simulation
   # md = Molecular_Dynamics_Basic(
   #  seqm_parameters=seqm_parameters,
   #  Temp=300.0,
   #  timestep=0.5,
   #  output={'molid':[0],'thermo':1,'dump':1,'prefix':'./out'}
   # ).to(device)
   # md.initialize_velocity(molecule)
   # _ = md.run(molecule, steps=10, remove_com=[True,1], Info_log=True)

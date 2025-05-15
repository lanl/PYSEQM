Quickstart
==========




<<<<<<< HEAD
This quickstart is a high-level introduction on how to get started with using PySEQM. 
=======
This quickstart is a high level introduction on how to get started using PySEQM. 

For full examples, please check out our `GitHub repository <https://github.com/lanl/pyseqm>`_ and the `examples directory <https://github.com/lanl/PYSEQM/tree/master/examples>`_.






Single Point SCF
------------------------------


Computes the electronic structure of a molecule by iteratively solving for the electron density or wavefunction until convergence, enabling calculation of total energy and related properties.

.. code-block:: python


   import torch
   from seqm.seqm_functions.constants import Constants
   from seqm.Molecule import Molecule
   from seqm.ElectronicStructure import Electronic_Structure

   torch.set_default_dtype(torch.float64)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   species = torch.as_tensor([[8,6,1,1],
                              [8,6,1,1],
                              [8,8,6,0]],
                           dtype=torch.int64, device=device)

   coordinates = torch.tensor([
                                 [
                                 [0.00,    0.00,    0.00],
                                 [1.22,    0.00,    0.00],
                                 [1.82,    0.94,    0.00],
                                 [1.82,   -0.94,    0.00]
                                 ],
                                 [
                                 [0.00,    0.00,    0.00],
                                 [1.22,    0.00,    0.00],
                                 [1.82,    0.94,    0.00],
                                 [1.82,   -0.94,    0.00]
                                 ],
                                 [
                                 [0.00,    0.00,    0.00],
                                 [1.23,    0.00,    0.00],
                                 [1.82,    0.94,    0.00],
                                 [0.0,     0.0,     0.0]
                                 ]
                              ], device=device)

                              
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

   molecules = Molecule(const, seqm_parameters, coordinates, species).to(device)
   esdriver = Electronic_Structure(seqm_parameters).to(device)
   esdriver(molecules)
>>>>>>> b817174 (update)






SCF and Excited State Calculations
------------------------------

<<<<<<< HEAD
Once initialized, an SCF calculation can be run directly to compute total energy:
=======
Estimates the energies of electronically excited states by solving for higher energy eigenvalues of the electronic Hamiltonian, extending SCF results beyond the ground state.
>>>>>>> b817174 (update)

.. code-block:: python



   import torch
   from seqm.seqm_functions.constants import Constants
   from seqm.Molecule import Molecule
   from seqm.ElectronicStructure import Electronic_Structure
   from seqm.seqm_functions.read_xyz import read_xyz


   torch.set_default_dtype(torch.float64)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



   species = torch.as_tensor([[8,6,1,1],
                              [8,6,1,1],
                              [8,8,6,0]],
                           dtype=torch.int64, device=device)

   coordinates = torch.tensor([
                                 [
                                 [0.00,    0.00,    0.00],
                                 [1.22,    0.00,    0.00],
                                 [1.82,    0.94,    0.00],
                                 [1.82,   -0.94,    0.00]
                                 ],
                                 [
                                 [0.00,    0.00,    0.00],
                                 [1.22,    0.00,    0.00],
                                 [1.82,    0.94,    0.00],
                                 [1.82,   -0.94,    0.00]
                                 ],
                                 [
                                 [0.00,    0.00,    0.00],
                                 [1.23,    0.00,    0.00],
                                 [1.82,    0.94,    0.00],
                                 [0.0,     0.0,     0.0]
                                 ]
                              ], device=device)

<<<<<<< HEAD
.. code-block:: python




   species, coordinates = read_xyz(['./data.xyz'])
=======
>>>>>>> b817174 (update)

   species = torch.as_tensor(species, dtype=torch.int64, device=device)[:]
   coordinates = torch.tensor(coordinates, device=device)[:]
   const = Constants().to(device)

   elements = [0] + sorted(set(species.reshape(-1).tolist()))

   seqm_parameters = {
      'method': 'AM1',
      'scf_eps': 1.0e-8,
      'scf_converger': [2, 0.0],
      'sp2': [False, 1.0e-5],
      'elements': elements,
      'learned': [],
      'pair_outer_cutoff': 1.0e8,
      'eig': True,
      'excited_states': {'n_states': 10},
   }

<<<<<<< HEAD
User-defined parameters for calculations are set using the seqm_parameters dictionary.


**method:** Austin Model 1

**scf_eps:** If the energy change between two SCF steps is smaller than this value, then SCF is converged.

**scf_converger:** Converger used for SCF.

**sp2:** 

**elements:** Atomic numbers in the data from species.

**learned:** Learned parameters name list.

**pair_outer_cutoff:** The value for how far apart two atoms can be before their interaction is ignored.

**eig:** Whether or not to diagonalize the Fock matrix.

**excited_states:** The number of excited states to calcuate.





.. code-block:: python



=======
>>>>>>> b817174 (update)
   molecules = Molecule(const, seqm_parameters, coordinates, species).to(device)
   esdriver = Electronic_Structure(seqm_parameters).to(device)
   esdriver(molecules)



Molecular Dynamics(NVE)
----------------------

<<<<<<< HEAD
You can run molecular dynamics using Born-Oppenheimer Molecular Dynamics (BOMD) or Extended-Lagrangian BOMD:
=======
Tracks the natural evolution of a system of atoms under Newton’s laws in an isolated environment—no energy exchange with surroundings. Energy is conserved, and atomic motion arises solely from interatomic forces.

>>>>>>> b817174 (update)

.. code-block:: python

   import torch
   from seqm.seqm_functions.constants import Constants
   from seqm.Molecule import Molecule
   from seqm.MolecularDynamics import Molecular_Dynamics_Basic
   from seqm.seqm_functions.read_xyz import read_xyz

   torch.set_default_dtype(torch.float64)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


<<<<<<< HEAD
Add import statements for PyTorch, and the relevant seqm modules.
Set the datatype to float64 for calculations in double precision and set device to GPU.

.. code-block:: python

   species, coordinates = read_xyz(['./data.xyz'])
=======
   species = torch.as_tensor([[8,6,1,1],
                              [8,6,1,1],
                              [8,8,6,0]],
                           dtype=torch.int64, device=device)

   coordinates = torch.tensor([
                                 [
                                 [0.00,    0.00,    0.00],
                                 [1.22,    0.00,    0.00],
                                 [1.82,    0.94,    0.00],
                                 [1.82,   -0.94,    0.00]
                                 ],
                                 [
                                 [0.00,    0.00,    0.00],
                                 [1.22,    0.00,    0.00],
                                 [1.82,    0.94,    0.00],
                                 [1.82,   -0.94,    0.00]
                                 ],
                                 [
                                 [0.00,    0.00,    0.00],
                                 [1.23,    0.00,    0.00],
                                 [1.82,    0.94,    0.00],
                                 [0.0,     0.0,     0.0]
                                 ]
                              ], device=device)
>>>>>>> b817174 (update)

   species = torch.as_tensor(species, dtype=torch.int64, device=device)[:]
   coordinates = torch.tensor(coordinates, device=device)[:]
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
   }


<<<<<<< HEAD
User-defined parameters for calculations are set using the seqm_parameters dictionary.

.. code-block:: python



=======
>>>>>>> b817174 (update)
   output = {
   'molid': [0], 
   'thermo': 1, 
   'dump': 1, 
   'prefix': '../../Outputs_location'
   }

<<<<<<< HEAD
The 'molid' key takes a list as the value. This list should contain the indices of the molecules on which MD has to be run, if multiple molecules have been given as input.

xxx

xxx

Set the output file path.

.. code-block:: python

=======
>>>>>>> b817174 (update)
   molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
   md = Molecular_Dynamics_Basic(seqm_parameters=seqm_parameters, Temp=300.0, timestep=0.4, output=output).to(device)
   md.initialize_velocity(molecule)
   _ = md.run(molecule, 10, remove_com=[True, 1], Info_log=True)


Molecular Dynamics(Langevin Thermostat)
----------------------

Simulates atomic trajectories under the influence of both deterministic interatomic forces and stochastic collisions with an implicit thermal bath. Temperature is controlled by damping and random noise, mimicking a system in thermal equilibrium.

.. code-block:: python

   import torch
   from seqm.seqm_functions.constants import Constants
   from seqm.Molecule import Molecule
   from seqm.MolecularDynamics import Molecular_Dynamics_Basic, Molecular_Dynamics_Langevin
   from seqm.seqm_functions.read_xyz import read_xyz
   import warnings

   torch.set_default_dtype(torch.float64)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


   species = torch.as_tensor([[8,6,1,1],
                              [8,6,1,1],
                              [8,8,6,0]],
                           dtype=torch.int64, device=device)

   coordinates = torch.tensor([
                                 [
                                 [0.00,    0.00,    0.00],
                                 [1.22,    0.00,    0.00],
                                 [1.82,    0.94,    0.00],
                                 [1.82,   -0.94,    0.00]
                                 ],
                                 [
                                 [0.00,    0.00,    0.00],
                                 [1.22,    0.00,    0.00],
                                 [1.82,    0.94,    0.00],
                                 [1.82,   -0.94,    0.00]
                                 ],
                                 [
                                 [0.00,    0.00,    0.00],
                                 [1.23,    0.00,    0.00],
                                 [1.82,    0.94,    0.00],
                                 [0.0,     0.0,     0.0]
                                 ]
                              ], device=device)

   species = torch.as_tensor(species, dtype=torch.int64, device=device)[:]
   coordinates = torch.tensor(coordinates, device=device)[:]
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


   output = {
   'molid': [0,1], 
   'thermo': 1, 
   'dump': 1, 
   'prefix': 
   '../../Outputs_location'
   }

   molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
   md = Molecular_Dynamics_Langevin(damp=100.0, seqm_parameters=seqm_parameters, Temp=400.0, timestep=0.4, output=output).to(device)
   md.initialize_velocity(molecule)
   _ = md.run(molecule, 10, remove_com=[True, 1], Info_log=True)



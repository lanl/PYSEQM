Quickstart
==========




This quickstart is a high-level introduction on how to get started with using PySEQM. 

For full examples, please check out our `GitHub repository <https://github.com/lanl/pyseqm>`_ and the `examples directory <https://github.com/lanl/PYSEQM/tree/master/examples>`_.






Single Point SCF
------------------------------

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


Molecular Dynamics(NVE)
----------------------

.. code-block:: python

   import torch
   from seqm.seqm_functions.constants import Constants
   from seqm.Molecule import Molecule
   from seqm.MolecularDynamics import Molecular_Dynamics_Basic
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

   const = Constants().to(device)

   seqm_parameters = {
      'method': 'AM1',
      'scf_eps': 1.0e-6,
      'scf_converger': [2, 0.0],
      'sp2': [False, 1.0e-5],
      'learned': [],
      'pair_outer_cutoff': 1.0e10,
   }

   output = {
   'molid': [0], 
   'thermo': 1, 
   'dump': 1, 
   'prefix': '../../Outputs_location'
   }

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


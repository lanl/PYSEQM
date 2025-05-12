Quickstart
==========

.. This quickstart provides a high-level introduction to using **PYSEQM** for semi-empirical quantum chemical simulations. It includes examples using the `seqm` Python module, supporting geometry optimization, molecular dynamics, and excited-state simulations — all GPU-accelerated and integrated with PyTorch 🔥.

.. For full example scripts, see the `examples/` directory in the `PYSEQM` GitHub repo.



This quickstart is a high-level introduction on how to get started with using PySEQM. 

For full examples, please check out our `GitHub repository <https://github.com/lanl/pyseqm>`_ and the `examples directory <https://github.com/lanl/pyseqm/tree/main/examples>`_.





SCF and Excited State Calculations
------------------------------

Once initialized, an SCF calculation can be run directly to compute total energy:

.. code-block:: python



   import torch
   from seqm.seqm_functions.constants import Constants
   from seqm.Molecule import Molecule
   from seqm.ElectronicStructure import Electronic_Structure
   from seqm.seqm_functions.read_xyz import read_xyz


   torch.set_default_dtype(torch.float64)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




Add import statements for PyTorch, PySEQM and its seqm modules.
Set the floating point precision to 64 bit and set device to GPU.

.. code-block:: python




   species, coordinates = read_xyz(['./data.xyz'])

   species = torch.as_tensor(species, dtype=torch.int64, device=device)[:]
   coordinates = torch.tensor(coordinates, device=device)[:]
   const = Constants().to(device)

   elements = [0] + sorted(set(species.reshape(-1).tolist()))

Load data in xyz form collecting species and coordinates and move both of them to the GPU.


.. code-block:: python

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



   molecules = Molecule(const, seqm_parameters, coordinates, species).to(device)
   esdriver = Electronic_Structure(seqm_parameters).to(device)
   esdriver(molecules)

Sends all information collected so far to the GPU and runs the calculation.



Molecular Dynamics
----------------------

You can run molecular dynamics using Born-Oppenheimer Molecular Dynamics (BOMD) or Extended-Lagrangian BOMD:

.. code-block:: python

   import torch
   from seqm.seqm_functions.constants import Constants
   from seqm.Molecule import Molecule
   from seqm.MolecularDynamics import Molecular_Dynamics_Basic
   from seqm.seqm_functions.read_xyz import read_xyz

   torch.set_default_dtype(torch.float64)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


Add import statements for PyTorch, and the relevant seqm modules.
Set the datatype to float64 for calculations in double precision and set device to GPU.

.. code-block:: python

   species, coordinates = read_xyz(['./data.xyz'])

   species = torch.as_tensor(species, dtype=torch.int64, device=device)[:]
   coordinates = torch.tensor(coordinates, device=device)[:]
   const = Constants().to(device)

   elements = [0] + sorted(set(species.reshape(-1).tolist()))

Load data in xyz form collecting species and coordinates and move both of them to the GPU.

.. code-block:: python
   
   seqm_parameters = {
      'method': 'AM1',
      'scf_eps': 1.0e-6,
      'scf_converger': [2, 0.0],
      'sp2': [False, 1.0e-5],
      'elements': elements,
      'learned': [],
      'pair_outer_cutoff': 1.0e10,
   }


User-defined parameters for calculations are set using the seqm_parameters dictionary.

.. code-block:: python



   output = {
   'molid': [0], 
   'thermo': 1, 
   'dump': 1, 
   'prefix': '../../Outputs_location'
   }

The 'molid' key takes a list as the value. This list should contain the indices of the molecules on which MD has to be run, if multiple molecules have been given as input.

xxx

xxx

Set the output file path.

.. code-block:: python

   molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
   md = Molecular_Dynamics_Basic(seqm_parameters=seqm_parameters, Temp=300.0, timestep=0.4, output=output).to(device)
   md.initialize_velocity(molecule)
   _ = md.run(molecule, 10, remove_com=[True, 1], Info_log=True)

Temp specifies the simulation temperature in Kelvin.

Timestep specifies the time step in femtoseconds.

Sends all collected information to the GPU.

.. code-block:: python


   _ = md.run(molecule, 10, remove_com=[True, 1], Info_log=True)

The number of MD steps.

Removes center of mass velocity every set number of step to prevent drifting.

Info_log determines whether to save additional information.

Then runs the calculation.


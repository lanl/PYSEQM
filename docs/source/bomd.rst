.. _molecular_dynamics:

Born–Oppenheimer Molecular Dynamics (BOMD)
==========================================

Simulates nuclear motion on the electronic ground-state potential energy surface by recalculating the electronic structure at each time step. Forces are derived from fully converged self-consistent field (SCF) calculations, ensuring accurate energy conservation. Suitable for studying thermal processes, structural fluctuations, and vibrational dynamics in molecules. Computationally intensive due to repeated SCF optimizations at every step.

Driver Classes & Initialization
-------------------------------

PYSEQM had MD engines for:

- **Born–Oppenheimer MD** via `Molecular_Dynamics_Basic`  
- **Langevin-thermostatted dynamics** via `Molecular_Dynamics_Langevin`

BOMD driver
~~~~~~~~~~~~~~~~~
Call the BOMD engine using

.. code-block:: python

   from seqm.MolecularDynamics import (
       Molecular_Dynamics_Basic,
       Molecular_Dynamics_Langevin
   )

   # Basic BOMD
   md = Molecular_Dynamics_Basic( seqm_parameters=seqm_parameters, Temp=300.0,
                                  timestep=0.5, output=output).to(device)

The parameters of `Molecular_Dynamics_Basic` are:

- `seqm_parameters` (`dict`)
  
- `Temp` (`float`) Specify the inital temperature (K) for BOMD. Given the initial temperature in BOMD, the initial nuclear velocities are set by drawing from a Maxwell–Boltzmann distribution so that each degree of freedom has an average kinetic energy of ½ kT

- `timestep` (`float`) Integration step size (fs)

- `output`: (`dict`) controlling prints & file dumps, see below

Langevin dynamics
~~~~~~~~~~~~~~~~~
In `Molecular_Dynamics_Langevin`, the total force on each atom is  

.. math::

   \mathbf{F} = \mathbf{F}_c + \mathbf{F}_f + \mathbf{F}_r

where:

- **F_c**: conservative force (–∇E_tot)  
- **F_f**: frictional force  

  .. math::

     \mathbf{F}_f = -\frac{m}{\tau}\,\mathbf{v}

  with `τ` being the damping constant (in units of time)

- **F_r**: random force  

  .. math::

     \mathbf{F}_r = \sqrt{\frac{2\,k_B\,T\,m}{\Delta t\,\tau}}\;\mathbf{R}(t)

  Each component of **R(t)** is sampled from **N(0,1)** with:

  .. math::

     \langle R_{ij}(t)\rangle = 0,\quad
     \langle R_{ij}(t)\,R_{ik}(t)\rangle = \delta_{jk}

Call the Langevin dynamics engine using

.. code-block:: python

   md_langevin = Molecular_Dynamics_Langevin( damp=100.0, seqm_parameters=seqm_parameters,
                                              Temp=300.0, timestep=0.5, 
                                              output=output).to(device)

In addition to the same parameters as `Molecular_Dynamics_Basic`, `Molecular_Dynamics_Langevin` has the following parameter(s):

- `damp` (`float`) Damping constant in fs

Configuring Outputs from BOMD
-----------------------------
In addition to specifying :ref:`seqm-parameters`, the MD engines also require an `output` dictionary to specfiy the output settings for MD runs:

.. code-block:: python

    output = {
        'molid': [0,1],
        'thermo': 1,
        'dump': 1,
        'prefix': 'output'
    }

- **molid** (`list[int]`)  
  Indices of molecules to output (e.g. ``[0,1]`` if you have at least two molecules and want MD outputs for molecule 0 and molecule 1).

- **thermo** (`int`)  
  How often (in timesteps) to print updated information (Step, Temperature, Energy) to the console.  

  - `1` = every timestep  
  - `10` = every 10 timesteps

- **dump** (`int`)  
  How often (in timesteps) to write an XYZ-format trajectory file.

- **prefix** (`str`)  
  Path prefix for all output files.  
  If you set prefix to `'my_output'`, files will be named like ``./my_output.{molid}.xyz``, etc.

Running the MD Simulation
--------------------------
Once your MD driver is initialized, call its `.run()` method:

.. code-block:: python

   _  = md.run(
       molecule,
       n_steps=1000,
       remove_com=[True, 1],
       Info_log=True
   )

Parameters: 

- **molecule**  
  Your `Molecule` object (with constants, coordinates, species) moved to the correct device.

- **n_steps** (`int`)  
  Number of MD integration steps to perform.

- **remove_com** (`[bool, int]`)  
  Control removal of center‐of‐mass motion:  
  - First element (`bool`): whether to remove COM drift.  
  - Second element (`int`): how often to apply it (every N steps).

- **Info_log** (`bool`)  
  If `True`, dump extra information (orbital energies, dipoles, etc.) to ``{prefix}.{molid}.Info.txt`` file.

Running BOMD
--------------

.. code-block:: python

   import torch
   from seqm.seqm_functions.constants import Constants
   from seqm.Molecule import Molecule
   from seqm.MolecularDynamics import Molecular_Dynamics_Basic

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
      'scf_converger': [2],
   }

   output = {
   'molid': [0,1,2], 
   'thermo': 1, 
   'dump': 1, 
   'prefix': '../../Outputs_location'
   }

   molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
   md = Molecular_Dynamics_Basic(seqm_parameters=seqm_parameters, Temp=300.0, timestep=0.5, output=output).to(device)
   md.initialize_velocity(molecule)
   _ = md.run(molecule, 10, remove_com=[True, 1], Info_log=True)

.. warning::
   Remember to initialize velocity before running the MD simulation and after making the Molecular_Dynamics object

In addition to the output printed to console, the MD trajectory will be saved in the ``{prefix}.{molid}.xyz`` file for each molecule. The file ``{prefix}.{molid}.Info.txt`` will also be created with extra information if requested.

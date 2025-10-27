.. _molecular_dynamics:

Born–Oppenheimer Molecular Dynamics (BOMD)
==========================================

This page walks you through running molecular dynamics in PYSEQM from scratch.
It explains the **MD engines**, the **output files you will get**, how to **configure
outputs**, and how to **restart** a run that ended early.

BOMD
-------------

**Born–Oppenheimer Molecular Dynamics (BOMD)** simulates nuclear motion on the
electronic ground‐state potential energy surface. At **every time step**, the
electronic structure is recomputed and the forces are taken from a converged
SCF calculation. 

Engines available
-----------------

- ``Molecular_Dynamics_Basic`` — standard BOMD using the velocity–Verlet
  integrator in an NVE ensemble.

- ``Molecular_Dynamics_Langevin`` — BOMD with a Langevin thermostat (adds
  friction and random forces to control temperature).

Both engines share the **same output system** (console, XYZ trajectory, and an
HDF5 file **per molecule**) and the **same checkpoint/restart mechanism**.

Units & conventions
-------------------

- Time step (``timestep``): **fs**
- Temperature (``Temp``): **K**
- Coordinates: **Å**
- Energies: eV 
- Langevin damping time constant: **fs**
- Velocities: **Å/fs**
- Forces: **eV/Å**

Files written by a run
----------------------

Every run can produce up to four kinds of outputs. You control which ones are
written and how often using the ``output`` dictionary (see below).

1) Console (screen) log
~~~~~~~~~~~~~~~~~~~~~~~

- Printed every ``output['print every']`` steps.
- Shows the following columns (per molecule): **Step, Temp, E(kinetic), E(potential), E(total)**.

2) XYZ trajectory (one file per molecule)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Path: ``{prefix}.{molid}.xyz`` (e.g., ``./runs/water.0.xyz``).
- Cadence/Writing frequency: ``output['xyz']`` (``0`` disables).
- **Appends** across restarts (the file is not overwritten when resuming).
- Human-readable; good for quick visualization in VMD, etc.

3) HDF5 data (one file per molecule)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Path: ``{prefix}.{molid}.h5`` (e.g., ``./runs/water.0.h5``).
- Machine-friendly, structured, and **appendable** across restarts.
- You choose **independent cadences** (steps) for writing data, coordinates, velocitites, forces: 

**HDF5 layout** (a group exists **only** if its cadence > 0):

- ``/atoms`` : (Natoms,) atomic numbers for the atoms in the molecule
- ``/coordinates``:

  - ``/coordinates/steps``  : (Tcoord,) **absolute** MD step indices
  - ``/coordinates/values`` : (Tcoord, Natoms, 3) positions (Å)
- ``/velocities``:
  
  - ``/velocities/steps``  : (Tvel,) 
  - ``/velocities/values`` : (Tvel, Natoms, 3) velocities
- ``/forces``:
  
  - ``/forces/steps``  : (Tforce,) 
  - ``/forces/values`` : (Tforce, Natoms, 3) forces
- ``/data`` (thermo and optional electronic info):
  
  - ``/data/steps``              : (Tdata,) 
  - ``/data/thermo/T``           : (Tdata,) Temperature (K)
  - ``/data/thermo/Ek``          : (Tdata,) Kinetic Energy (eV)
  - ``/data/thermo/Ep``          : (Tdata,) Potential Energy (eV)
  - ``/data/properties/ground_dipole`` : (Tdata, 3) Dipole moment of the ground state in atomic units (a.u.)
  - **Optional excited-state and MO outputs.**
    
    If you are doing excited state dynamics, the following are created under ``/data/excitation``:

    - ``/data/excitation/excitation_energy``:
      (Tdata, Nstates) — vertical excitation energies (eV)

    - ``/data/excitation/transition_dipole``:
      (Tdata, Nstates, 3) — transition dipole moments (a.u.)

    - ``/data/excitation/oscillator_strength``:
      (Tdata, Nstates) — oscillator strengths for each state

    - ``/data/excitation/unrelaxed_dipole``:
      (Tdata, 3) — dipole of the current excited state (no orbital relaxation) in a.u. . 
      Add to ground_dipole to get total dipole.

    - ``/data/excitation/relaxed_dipole``:
      (Tdata, 3) — dipole of the current excited state, with orbital relaxation, in a.u. . 
      Add to ground_dipole to get total dipole.

    - ``/data/excitation/active_state``:
      Scalar data (int) giving the index of the active electronic excited state used during the MD run.

    If ``output['h5']['write_mo'] = True``, molecular orbital information is stored under
    ``/data/mo`` at the same cadence:

    - ``/data/mo/homo_lumo_gap``  
      (Tdata, 1) for restricted SCF, or (Tdata, 2) for unrestricted runs.
      Contains the instantaneous HOMO–LUMO energy gap (eV).

**Each group maintains its own ``steps`` array as a commit log**, with **absolute**
step numbers that continue across restarts.

4) Checkpoint (restart) file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Long MD runs can be interrupted (queue limits, maintenance, etc.). PYSEQM can
checkpoint and later resume to **finish** the run, writing into the *same* HDF5/XYZ
files at the correct positions.

- Path: ``{prefix}.restart.pt`` (e.g., ``./runs/water.restart.pt``).
- Written every ``output['checkpoint every']`` steps (``0`` disables).

The checkpoint is written **atomically** (temp file + rename) after flushing
HDF5/XYZ, so it always reflects everything that was persisted to disk up to
that step.

Configuring outputs (details)
-----------------------------

The ``output`` dictionary controls all output behavior. It should contain the following keys/values:

``'molid'`` (list of int)
  Indices of molecule for which output has to be written. If you simulate multiple molecules in a batch,
  list them here (e.g., ``[0, 1]`` for 2 molecules).
  **Default:** ``[0]``

``'prefix'`` (str)
  Path prefix for all files. Output files will be named
  ``{prefix}.{molid}.xyz``, ``{prefix}.{molid}.h5``, and ``{prefix}.restart.pt``.
  **Default:** ``'md'``

``'print every'`` (int)
  Screen log cadence (frequency of writing) in steps. Use ``0`` to disable.
  **Default:** ``1``

``'xyz'`` (int)
  XYZ trajectory file cadence in steps. Use ``0`` to disable. Appends across restarts.
  **Default:** ``0``

``'checkpoint every'`` (int)
  Checkpoint cadence in steps. Use ``0`` to disable chekpointing. 
  **Default:** ``100``

``'h5'`` (dict) cadences for writing to HDF5 file per molecule:
  - ``'data'``: cadence for the ``/data`` group (thermo/properties/MO). **Default:** ``0``
  - ``'coordinates'``: cadence for the ``/coordinates`` group. **Default:** ``0``
  - ``'velocities'``: cadence for the ``/velocities`` group. **Default:** ``0``
  - ``'forces'``: cadence for the ``/forces`` group. **Default:** ``0``
  - ``'write_mo'`` (bool): if ``True`` include molecular orbital info in ``/data``. **Default:** ``False``

Initializing velocities
-----------------------

By default, the ``Temp`` parameter of the MD driver is used to generate initial
atomic velocities from a **Maxwell–Boltzmann distribution** corresponding to the
specified temperature. This ensures that, on average, each degree of freedom
carries a kinetic energy of :math:`\frac{1}{2} k_B T`.

If you want to start from a specific set of velocities instead of random
Maxwell–Boltzmann ones, simply assign them to the molecule before calling
``md.run()``:

.. code-block:: python

   molecule.velocities = torch.tensor([...], dtype=molecule.coordinates.dtype, device=device)
   # velocities must have shape (Nmolecules, Natoms, 3)
   # and be expressed in Å/fs

   _ = md.run(molecule, steps=1000)

When ``molecule.velocities`` is already defined, the MD driver **will not**
reinitialize them based on ``Temp``. It will directly use the values you
provide as the starting velocities.

Getting started: a minimal BOMD run
-----------------------------------

.. code-block:: python

   import torch
   from seqm.seqm_functions.constants import Constants
   from seqm.Molecule import Molecule
   from seqm.MolecularDynamics import Molecular_Dynamics_Basic

   torch.set_default_dtype(torch.float64)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # --- System definition (one molecule)
   species = torch.as_tensor([[8,6,1,1]], dtype=torch.int64, device=device)
   coordinates = torch.tensor([[[0.00, 0.00, 0.00],
                                [1.22, 0.00, 0.00],
                                [1.82, 0.94, 0.00],
                                [1.82,-0.94, 0.00]]], device=device)

   const = Constants().to(device)
   seqm_parameters = {
       'method': 'AM1',
       'scf_eps': 1.0e-7,
       'scf_converger': [1],
   }

   # --- Output configuration
   output = {
       'molid': [0],               # write only molecule 0
       'prefix': './runs/demo',    # prefix for output file names
       'print every': 2,          # screen log every 10 steps
       'xyz': 10,                  # XYZ cadence (0 = off)
       'h5': {
           'data': 10,             # /data cadence (thermo, properties)
           'coordinates': 10,      # /coordinates cadence
           'velocities':  10,      # /velocities cadence
           'forces':      10,      # /forces cadence
           'write_mo': False,      # include MO info in /data (optional)
       },
       'checkpoint every': 100,    # write restart file every 100 steps
   }

   molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)

   md = Molecular_Dynamics_Basic(seqm_parameters=seqm_parameters,
                                 Temp=300.0,             # K
                                 timestep=0.5,           # fs
                                 output=output).to(device)

   # No need to call initialize_velocity(); md.run() handles this for fresh runs.
   md.run(molecule,
          steps=200,
          remove_com=False,
          reuse_P=False)


.. _bomd_driver:

API reference (BOMD)
--------------------

.. code-block:: python

   md = Molecular_Dynamics_Basic(
       seqm_parameters: dict,
       timestep: float = 0.5,          # fs
       Temp: float = 0.0,              # K (sets initial velocity distribution)
       output: dict | None = None,
   )

Parameters:

- ``seqm_parameters`` (``dict``): electronic structure options (method, SCF, excited states, etc.)
- ``timestep`` (``float``): integration step size in femtoseconds
- ``Temp`` (``float``): initial temperature (K); Initializes velocities drawn from a Maxwell–Boltzmann at this temperature
- ``output`` (``dict``): controls console/XYZ/HDF5 cadence and checkpointing (see below)


.. code-block:: python

   md.run(
       molecule,
       steps,
       reuse_P=True,
       remove_com=None,
       seed=None,
   )

Runs a molecular dynamics trajectory for the given molecules.

Parameters

- ``molecule`` (:class:`Molecule`)  
  The molecule object containing atomic species, coordinates, and (optionally) velocities.  
  Must be allocated on the same device as the MD driver.

- ``steps`` (``int``)  
  Number of integration steps to perform.  

- ``reuse_P`` (``bool``, default ``True``)  
  Whether to reuse the density matrix (and CIS/RPA amplitudes, if applicable) between steps.  
  Set to ``False`` to recompute electronic states independently each step.

- ``remove_com`` (``tuple`` or ``None``, default ``None``)  
  Controls removal of center-of-mass motion.  

  Use ``('linear', N)`` to remove translation every ``N`` steps,  
  or ``('angular', N)`` to remove both translation and rotation.  
  ``None`` disables COM removal.

- ``seed`` (``int`` or ``None``, default ``None``)  
  Random number generator seed used for initializing velocities from the
  Maxwell–Boltzmann distribution.  
  If ``None``, the current global RNG state is used.  
  Setting a fixed seed ensures reproducible initial velocity assignments
  and reproducible stochastic dynamics/Langevin dynamics (useful for benchmarking or debugging).  
  Has no effect when the molecule already defines its own initial velocities.

Restarting a run that ended early
---------------------------------

If a job stops before completing (walltime limit, preemption, etc.), you can
resume and **finish** it using the built-in loader by giving it the path to the checkpoint file:

.. code-block:: python

   from seqm.MolecularDynamics import Molecular_Dynamics_Basic

   Molecular_Dynamics_Basic.run_from_checkpoint(
       path='./runs/demo.restart.pt', device=device
   )
   # The loader sets step offsets and continues until the planned 'steps' are done.

Use `Molecular_Dynamics_Basic.run_from_checkpoint()` function to restart any MD simulation type (Basic MD, Langevin, XL-BOMD, etc.).
The function automatically restores all necessary information from the checkpoint file.
The checkpoint file contains all run parameters and internal state needed to continue the simulation exactly where it stopped; manual reconstruction is unnecessary.

Thermostat: Langevin dynamics
---------------------------------

``Molecular_Dynamics_Langevin`` adds friction and random forces:

.. math::

   \mathbf{F} = \mathbf{F}_c
                - \frac{m}{\tau}\,\mathbf{v}
                + \sqrt{\frac{2\,k_B\,T\,m}{\Delta t\,\tau}}\;\mathbf{R}(t)

where each component of :math:`\mathbf{R}(t)` is sampled from :math:`\mathcal{N}(0,1)`.

Usage mirrors the BOMD engine. The temperature for the thermostat is specified by the `Temp` parameter (which is also used to initialize velocities):

.. code-block:: python

   from seqm.MolecularDynamics import Molecular_Dynamics_Langevin

   md_langevin = Molecular_Dynamics_Langevin(
       damp=50.0,                   # damping time (fs)
       seqm_parameters=seqm_parameters,
       Temp=300.0, timestep=0.5,
       output=output
   ).to(device)

   _ = md_langevin.run(molecule, steps=10000, remove_com=('linear', 100))

Reading HDF5 outputs in Python
------------------------------

A small example using ``h5py`` to read positions and temperature:

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File('./runs/demo.0.h5', 'r') as f:
       steps_coord = f['/coordinates/steps'][...]             # shape (Tcoord,)
       coords      = f['/coordinates/values'][...]            # shape (Tcoord, Nat, 3)

       steps_data  = f['/data/steps'][...]
       T_series    = f['/data/thermo/T'][...]

       print("Coordinate frames:", len(steps_coord))
       print("Thermo samples:", len(steps_data))
       print("First temperature sample:", T_series[0])

Best practices
--------------

- **Choose cadences** that balance file size and analysis needs:
  e.g., write ``/data`` every 10–20 steps; write vectors every 10–100 steps.
- **Checkpoint cadence**: pick a value that fits your queue walltime (e.g., every
  500–2000 steps).
- **COM removal**: use ``remove_com=('linear', N)`` to remove translation (3 DoF);
  ``('angular', N)`` removes both translation and rotation (6 DoF). linear molecules (5 DoF) are not auto-detected.

Removed features
----------------

- ``Info_log`` files are no longer produced. All analysis data lives in the
  per-molecule HDF5 files under ``/data`` and the vector groups.

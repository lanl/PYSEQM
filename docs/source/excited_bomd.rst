.. _excited_state_dynamics:

Excited-State Molecular Dynamics (Adiabatic)
============================================

Overview
--------

PYSEQM supports **adiabatic excited-state molecular dynamics** on a chosen
electronic excited state.  
This means the nuclei move on the potential energy surface of **one** selected
excited state (CIS or RPA/TDHF), while the electronic wavefunction for that
state is recalculated self-consistently at each step.

.. important::

   This is **not** nonadiabatic dynamics.  
   Transitions between states are *not* included. The trajectory stays on one
   chosen excited surface throughout the simulation.

Supported engines
-----------------

All existing MD drivers support adiabatic excited-state dynamics:

- ``Molecular_Dynamics_Basic`` — standard BOMD 
- ``Molecular_Dynamics_Langevin`` — BOMD with thermostat 
- ``XL_BOMD`` — extended-Lagrangian excited-state MD

The **input structure, outputs, and checkpointing** remain exactly the same as
for ground-state BOMD.
You only need to add a few extra keywords to your input dictionaries to choose
the active excited state.

Setting up excited-state BOMD
-----------------------------

Excited-state BOMD uses the same ``Molecular_Dynamics_Basic`` class as
ground-state BOMD.  
The only difference is the **extra keys** you add to ``seqm_parameters``.

.. code-block:: python

   from seqm.MolecularDynamics import Molecular_Dynamics_Basic

   seqm_parameters = {
       'method': 'AM1',
       'scf_eps': 1.0e-7,          # SCF convergence
       'scf_converger': [1],
       'excited_states': {'n_states': 5},   # number of excited states to compute
       'active_state': 2,                   # index of the excited state to follow
   }

   output = {
       'prefix': './runs/h2co_es',
       'print every': 1,
       'xyz': 1,
       'checkpoint every': 100,
       'h5': {'data': 10, 'coordinates': 10, 'velocities': 10, 'forces': 10},
   }

   md = Molecular_Dynamics_Basic(
       seqm_parameters=seqm_parameters,
       Temp=300.0, timestep=0.5, output=output
   ).to(device)

   _ = md.run(molecule, steps=1000)

Required keys in ``seqm_parameters``:

- **``'excited_states'``**  
  Dictionary specifying excited states to compute:
  ``'n_states'`` must be **at least two greater than** the chosen ``'active_state'``.
  This ensures accurate evaluation of the excited wavefunction and gradients.

- **``'active_state'``**  
  Integer (≥ 1) selecting which excited state to run on.  
  ``0`` always corresponds to the ground state.

- All other SCF keys (``method``, ``scf_eps``, etc.) work as in ground-state BOMD.

Langevin thermostats
~~~~~~~~~~~~~~~~~~~~

Excited-state dynamics also support the **Langevin** thermostat:
use the same ``Molecular_Dynamics_Langevin`` driver with the
same ``seqm_parameters`` (including ``excited_states`` and ``active_state``)
and specify a damping constant ``damp`` (fs).

.. code-block:: python

   md_langevin = Molecular_Dynamics_Langevin(
       damp=100.0,
       seqm_parameters=seqm_parameters,
       Temp=300.0,
       timestep=0.5,
       output=output
   ).to(device)

   _ = md_langevin.run(molecule, steps=2000)

Setting up excited-state XL-BOMD
--------------------------------

Extended-Lagrangian excited-state MD runs exactly like the ground-state
XL-BOMD, but you can additionally provide relaxed convergence settings for both
the SCF and excited-state solvers.

.. code-block:: python

   from seqm.MolecularDynamics import XL_BOMD, KSA_XL_BOMD

   xl_bomd_params = {
       'k': 6,              # dissipative force coefficient
       'scf_eps': 5e-4,     # relaxed SCF convergence (default)
       'es_eps':  5e-3,     # relaxed excited-state convergence (default)
   }

   seqm_parameters = {
       'method': 'AM1',
       'excited_states': {'n_states': 5},
       'active_state': 2,
   }

   md_xl = XL_BOMD(
       xl_bomd_params=xl_bomd_params,
       seqm_parameters=seqm_parameters,
       Temp=400.0,
       timestep=0.4,
       output=output
   ).to(device)

   _ = md_xl.run(molecule, steps=5000)

Notes for XL-BOMD:

- ``scf_eps`` and ``es_eps`` in ``xl_bomd_params`` control the convergence
  thresholds for SCF and excited-state solvers during each time step.  
  By default:
  - ``scf_eps = 5 × 10⁻⁴``
  - ``es_eps = 5 × 10⁻³``
- These relaxed thresholds typically maintain energy conservation
  while greatly reducing cost.
- You can override them if tighter convergence is desired.

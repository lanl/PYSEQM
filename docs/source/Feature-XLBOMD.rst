Extended Lagrangian Born Oppenheimer Molecular Dynamics (XL-BOMD)
=================================================================

Extended Lagrangian Born–Oppenheimer Molecular Dynamics (XL-BOMD) improves upon traditional Born–Oppenheimer Molecular Dynamics (BOMD) by introducing auxiliary dynamical variables that track the electronic density matrix in a harmonic potential. This eliminates the need for fully converged self-consistent field (SCF) calculations at each time step.

The method enables energy-conserving dynamics even with relaxed SCF convergence, significantly lowering computational cost while maintaining stability. XL-BOMD is particularly effective for long-time simulations and large molecular systems.

PYSEQM provides two variants of XL-BOMD:

- **XL_BOMD**  
  Base extended‐Lagrangian Born–Oppenheimer MD.

- **KSA_XL_BOMD**  
  An improved Krylov subspace approximation (KSA) scheme for the integration of the electronic equations of motion within XL-BOMD
  (more accurate dynamics without sacrificing speed).


Driver Classes & Initialization
-------------------------------
Import the drivers and instantiate with common arguments:

.. code-block:: python

   from seqm.MolecularDynamics import XL_BOMD, KSA_XL_BOMD

   # Common arguments:
   #  - seqm_parameters: dict for SCF at each MD step
   #  - Temp:           temperature (K) to initialize velocities
   #  - timestep:       integration step (fs)
   #  - output:         dict controlling logs & trajectory files

   # 1. XL-BOMD 
   xl_bomd_params = {'k': 6}   # dissipative force coefficient
   md_xl = XL_BOMD(
       xl_bomd_params=xl_bomd_params,
       damp=100.0,                # optional, if needed, damping constant (fs) for Langevin dynamics 
       seqm_parameters=seqm_parameters,
       Temp=400.0,
       timestep=0.4,
       output=output
   ).to(device)

   # 2. KSA-XL-BOMD (low-rank Krylov approximation)
   xl_bomd_params = {
       'k':             6,        # dissipative force coefficient
       'max_rank':      3,        # Krylov subspace approximation kernel rank
       'err_threshold': 0.0,      # approximation error tolerance
       'T_el':         1500       # electronic temperature (K)
   }
   md_ksa = KSA_XL_BOMD(
       xl_bomd_params=xl_bomd_params,
       damp=100.0,                # optional, if needed, damping constant (fs) for Langevin dynamics 
       seqm_parameters=seqm_parameters,
       Temp=400.0,
       timestep=0.4,
       output=output
   ).to(device)

Key Parameters of XL-BOMD
~~~~~~~~~~~~~~~~~~~~~~~~~~

The XL-BOMD drivers need a dictionary that specify the parameters for XL-BOMD dynamics. In the above code, this dictionary is `xl_bomd_params`.
The key/value pairs for this dictionary are:

- **'k'** (`int`)  
  Controls `K` of the dissipative electronic force. See `Niklasson, Anders, et al., The Journal of Chemical Physics 130.21 (2009) <https://aip.scitation.org/doi/full/10.1063/1.3148075>`_.

  Values from ``3`` to ``9`` are accepted.

- **'max_rank'** (`int`, KSA only)  
  Maximum rank of the kernel for the update in the Krylov subspace approximation.

- **'err_threshold'** (`float`, KSA only)  
  Error tolerance for the low‐rank approximation (set to 0.0 to fix at `max_rank`).

- **'T_el'** (`float`, KSA only)  
  Electronic “temperature” for thermalized Hartree–Fock.

The parameters to initialize the MD drivers `XL_BOMD()` and `KSA_XL_BOMD()` are the same as those for `Molecular_Dynamics_Basic()` as in :ref:`bomd_driver`.
The extra parameters you can specify for XL-BOMD drivers are:

- **xl_bomd_params** (`dict`)
  The dictionary that specifes the parameters for XL-BOMD dynamics, as described above.

- **damp** (`float`)  
  If you want to run Langevin dynamics for thermostatting, then specify the damping constant `damp` in fs. Langevin dynamics is turned off by default.

Running an XL-BOMD Simulation
------------------------------
After creating your MD driver run:

.. code-block:: python

   # Run N steps
   _ = md_xl.run(
       molecule,
       n_steps=1000,
   )
   # or
   _ = md_ksa.run(
       molecule,
       n_steps=1000,
   )


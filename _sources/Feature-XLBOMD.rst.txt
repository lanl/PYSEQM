Extended Lagrangian Born Oppenheimer Molecular Dynamics (XL-BOMD)
=================================================================

Extended Lagrangian Born–Oppenheimer Molecular Dynamics (XL-BOMD) improves upon traditional Born–Oppenheimer Molecular Dynamics (BOMD) by introducing auxiliary dynamical variables that track the electronic density matrix in a harmonic potential. This reduces the need for fully converged self-consistent field (SCF) calculations at each time step.

The method enables energy-conserving dynamics even with relaxed SCF convergence, significantly lowering computational cost while maintaining stability. XL-BOMD is particularly effective for long-time simulations and large molecular systems, and is well-suited for semiempirical, Hartree–Fock, and DFT-based electronic structure methods.

This implementation supports efficient simulations by reducing the number of SCF iterations required per time step, enabling practical quantum-based molecular dynamics for systems where conventional BOMD would be prohibitively expensive.

PYSEQM provides two variants:

- **XL_BOMD**  
  Base extended‐Lagrangian Born–Oppenheimer MD.

- **KSA_XL_BOMD**  
  Krylov‐subspace‐approximated XL‐BOMD with low‐rank electronic updates (more accurate dynamics without sacrificing speed).


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
   ksa_params = {
       'k':             6,        # dissipative force coefficient
       'max_rank':      3,        # Krylov subspace rank
       'err_threshold': 0.0,      # approximation error tolerance
       'T_el':         1500       # electronic temperature (K)
   }
   md_ksa = KSA_XL_BOMD(
       xl_bomd_params=ksa_params,
       damp=100.0,                # optional, if needed, damping constant (fs) for Langevin dynamics 
       seqm_parameters=seqm_parameters,
       Temp=400.0,
       timestep=0.4,
       output=output
   ).to(device)

Key Parameters
~~~~~~~~~~~~~~
- **xl_bomd_params['k']** (`float`)  
  Controls the strength of the dissipative electronic force `Niklasson, Anders, et al., The Journal of Chemical Physics 130.21 (2009) <https://aip.scitation.org/doi/full/10.1063/1.3148075>`_.

- **max_rank** (`int`, KSA only)  
  Maximum rank of the Krylov subspace used in the low‐rank update.

- **err_threshold** (`float`, KSA only)  
  Error tolerance for the low‐rank approximation (set to 0.0 to fix at `max_rank`).

- **T_el** (`float`, KSA only)  
  Electronic “temperature” for thermalized Hartree–Fock.

- **damp** (`float`)  
  If you want to run Langevin dynamics, then specify the damping constant `damp` in fs. Langevin dynamics is turned off by default.

Running an XL-BOMD Simulation
------------------------------
After creating your MD driver, initialize velocities and run:

.. code-block:: python

   # Initialize Maxwell–Boltzmann velocities
   md_xl.initialize_velocity(molecule)
   # or for KSA:
   md_ksa.initialize_velocity(molecule)

   # Run N steps
   _ = md_xl.run(
       molecule,
       n_steps=1000,
       remove_com=[True, 1],
       Info_log=True
   )
   # or
   _ = md_ksa.run(
       molecule,
       n_steps=1000,
       remove_com=[True, 1],
       Info_log=True
   )


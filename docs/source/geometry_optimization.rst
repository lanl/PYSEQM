Geometry Optimization
=====================

Overview
--------
PYSEQM provides two approaches for finding minimum-energy structures:

- **geomeTRIC** integration 
  
  Robust, advanced algorithms using an external package called `geomeTRIC <https://geometric.readthedocs.io/en/latest/>`_; single-molecule only.
- **Steepest-Descent (SD)** via `Geometry_Optimization_SD` driver  

  Batch-capable but may converge extremely slowly or fail on complex surfaces.  

geomeTRIC Integration
---------------------
PYSEQM can run robust optimization on a single molecule using `geomeTRIC <https://geometric.readthedocs.io/en/latest/>`_ as the optimization driver.
First install the geomeTRIC interface:

.. code-block:: bash

   cd PYSEQM/
   pip install '.[geomopt]'

Then in your script:

.. code-block:: python

   from seqm.geometryOptimization import geomeTRIC_optimization

   # `molecules` must be a single-entry batch:
   geomeTRIC_optimization(molecules)


Example optimization with geomeTRIC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from seqm.seqm_functions.constants import Constants
    from seqm.Molecule import Molecule
    from seqm.geometryOptimization import geomeTRIC_optimization
    
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    species = torch.as_tensor([[6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1]] ,                           
                              dtype=torch.int64, device=device)
    
    coordinates = torch.tensor([
                                 [
                                   [ -35.894866 ,   9.288626 ,  10.054420], 
                                   [ -36.369609 ,   8.136872 ,   9.487425],
                                   [ -34.536331 ,   9.569570 ,   9.955502],
                                   [ -35.492308 ,   7.159723 ,   8.983842],
                                   [ -33.712643 ,   8.697458 ,   9.304412],
                                   [ -34.183340 ,   7.494548 ,   8.762036],
                                   [ -36.585465 ,   9.939934 ,  10.537961],
                                   [ -37.450824 ,   8.064607 ,   9.436016],                            
                                   [ -34.070057 ,  10.502494 ,  10.500065],                            
                                   [ -35.950748 ,   6.201235 ,   8.765212],                            
                                   [ -33.438373 ,   6.826374 ,   8.246956],                            
                                   [ -32.654911 ,   8.872607 ,   9.123019]
                                  ],
                               ], device=device)
    
    const = Constants().to(device)
    
    seqm_parameters = {
                       'method' : 'AM1', 
                       'scf_eps' : 1.0e-8,
                       'scf_converger' : [0,0.1],
                       }
    
    molecules = Molecule(const, seqm_parameters, coordinates, species).to(device)

    geomeTRIC_optimization(molecules)


CIS minimum-energy crossing points
----------------------------------
``geomeTRIC_mecp`` locates a minimum-energy conical intersection (MECI)
between two CIS excited states.  At a MECI the state-energy gap is zero and the
mean-energy gradient has no component in the seam space.  The two directions
removed from the seam gradient are the energy-difference gradient

.. math::

   \mathbf{g} = \nabla(E_2-E_1)

and the interstate-gradient vector

.. math::

   \mathbf{h} = (E_2-E_1)\mathbf{d}_{12},

where :math:`\mathbf{d}_{12}` is the derivative coupling returned by PYSEQM.

Example
~~~~~~~

.. code-block:: python

   from seqm.api import geomeTRIC_mecp

   seqm_parameters["excited_states"] = {
       "method": "cis",
       "n_states": 2,
       "tolerance": 1e-7,
   }
   molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
   geomeTRIC_mecp(
       molecule,
       states=(1, 2),
       method="composed_step", # or "penalty" / "gradient_projection"
       gap_tolerance=5e-3,      # eV
       seam_switch_gap=0.05,    # eV
       maxiter=100,
   )

``states`` is 1-based and excludes S0, so ``(1, 2)`` means S1/S2. The final
geometry is written to ``mecp.xyz`` and the trajectory to
``mecp_opt_traj.xyz``. The trajectory contains every electronic-structure
evaluation, including rejected trial steps, and records both state energies
and their gap. ``molecule.mecp_state_energies`` and
``molecule.mecp_energy_gap`` contain the values at the returned geometry.

Methods
~~~~~~~

``method="composed_step"`` directly treats the MECI as a constrained
optimization.  In geomeTRIC's independent internal-coordinate space it adds:

* a minimum-norm normal step

  .. math::

     \Delta\mathbf{q}_{\mathrm{BP}}
     = -\frac{\Delta E}{\mathbf{g}^{T}\mathbf{g}}\mathbf{g},

  which solves the linearized degeneracy condition; and
* a seam-space quasi-Newton step that minimizes the mean energy after projecting
  out the orthonormalized :math:`\mathbf{g}` and :math:`\mathbf{h}` directions.

The two steps are trust-limited separately and then combined.  This method does
not invent a scalar objective, and convergence requires the requested energy
gap, projected-gradient, and displacement thresholds.  ``seam_switch_gap``
reserves most of the trust radius for reaching degeneracy while the gap is
large.

``method="gradient_projection"`` supplies the standard Bearpark--Robb--Schlegel
search vector to geomeTRIC:

.. math::

   \mathbf{G}_{\mathrm{GP}} =
   \mathbf{P}\nabla\bar E
   + w(E_1-E_2)\frac{\nabla(E_1-E_2)}{\lVert\nabla(E_1-E_2)\rVert}.

It usually takes fewer calculations, but this vector is not the derivative of
the reported mean energy.  PYSEQM therefore integrates it only to provide
geomeTRIC with consistent trust-radius bookkeeping.  ``gap_weight`` is the
dimensionless :math:`w` above.

``method="penalty"`` uses the conservative Levine--Coe--Martinez objective

.. math::

   L = \bar E + \sigma\frac{\Delta E^2}{\Delta E + \alpha}.

It does not require derivative couplings and can use geomeTRIC's ordinary
minimizer.  ``penalty_sigma`` defaults to 3.5 and ``penalty_alpha`` defaults to
0.025 Hartree.  A finite :math:`\alpha` makes the objective smooth but generally
leaves a finite optimized gap; restart with a smaller ``penalty_alpha`` if a
tighter crossing is required.

Requirements and practical choices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only one restricted, closed-shell molecule and crossings between two excited
CIS states are supported.  ``gradient_projection`` and ``composed_step`` need
the CIS derivative coupling;
``penalty`` avoids that
cost.  Start reasonably close to the crossing: these are local optimizations.

The implementations follow the direct gradient of `Bearpark, Robb, and
Schlegel <https://doi.org/10.1016/0009-2614(94)00433-1>`_, the composed-step
seam optimization of `Sicilia et al. <https://doi.org/10.1021/ct7002435>`_,
and the penalty objective of `Levine, Coe, and Martínez
<https://doi.org/10.1021/jp0761618>`_.



Built-in Steepest-Descent
-------------------------
Use `Geometry_Optimization_SD` driver for simple, batched optimizations:

.. code-block:: python

   from seqm.MolecularDynamics import Geometry_Optimization_SD

   opt = Geometry_Optimization_SD(
       seqm_parameters,
       alpha=0.008,          # scaling factor for step size
       force_tol=1e-2,       # converge when max force < 0.01 eV/Å
       max_evl=40            # max steps (energy+force evaluations)
   ).to(device)

   max_force, dE = opt.run(molecule)

- **alpha** (`float`): scaling factor for scaling step size of coordinate updates  
- **force_tol** (`float`): convergence threshold on maximum force  
- **max_evl** (`int`): maximum number of evaluations  

.. warning::
   Steepest-descent can be very inefficient and may fail for rough potential-energy surfaces.

XL-ESMD
========

``XL_ESMD`` propagates CIS auxiliary amplitudes in the occupied--virtual MO
basis and evaluates a variational shadow excitation energy.  MO amplitudes and
their complete Verlet history are transported between geometries with
occupied/virtual maximum-overlap (polar) transformations.

Recommended configuration
-------------------------

For dynamics on one isolated CIS surface, use the default linearized
independent-state formulation (case 1a):

.. code-block:: python

   xl_params = {
       "k": 6,
       "constraint_mode": "independent_linearized",
       "max_rank": 3,
       "err_threshold": 1.0e-8,
   }

   md = XL_ESMD(
       xl_bomd_params=xl_params,
       seqm_parameters=seqm_parameters,
       timestep=0.1,
       Temp=0.0,
       output=output,
   )

The kernel action is solved with nonsymmetric Arnoldi/GMRES.  Rank 3 is a
useful starting point; convergence is controlled by ``err_threshold``, and
increasing the maximum rank is useful only when the reported residual remains
above that tolerance.

For several individually labelled adiabatic states, select
``"ordered_linearized"``.  State ``i`` is constrained against itself and all
lower states.  This retains the cheap linearized solve but removes the
internal state-rotation null modes of an unconstrained block formulation.

For a deliberately propagated *state subspace*, Cases 2a/2b are selectable as
``"coupled_linearized"`` (alias ``"2a"``) and ``"coupled_exact"`` (alias
``"2b"``).  These modes ignore ``active_state`` when constructing the excited
potential: the trace (sum) of all requested shadow-state energies is used.
Only that total block energy is rotation invariant and variational.  Do not
use these modes as substitutes for dynamics on one labelled adiabatic surface.

Available formulations
----------------------

``independent_linearized``
   Closed-form case 1a.  This is the production default and the preferred
   method for an isolated surface.

``independent_exact``
   Pole-safe stationary-branch case 1b.  It is retained for comparison.  The
   diagnostic ``minimum_pole_distance`` must be monitored because both the
   solve and its Jacobian become ill-conditioned near an orbital-gap pole.

``ordered_linearized``
   Lower-state/triangular variant of case 2a.  This is preferred for multiple
   labelled states.

``coupled_linearized`` / ``2a``
   Raw linearized block orthogonality.  The closed-form solve needs one small
   state-block matrix solve.  The Krylov action is restricted to the
   Grassmann subspace tangent: internal rotations are gauge, while radial
   state-block changes violate the row-orthonormal auxiliary constraint.  A
   polar retraction keeps the propagated auxiliary block on that manifold.
   This is the recommended coupled-subspace method.

``coupled_exact`` / ``2b``
   Exact row-orthonormal block solved on the Stiefel manifold.  Every JVP needs
   a dense bordered response solve.  It is implemented and verified as the
   exact-constraint comparison, but is not recommended for large production
   calculations.

Shadow forces
-------------

The default ``force_mode`` is ``"autodiff"``.  It differentiates the
variational shadow energy through SCF/orbital response and has been checked
against fixed-auxiliary-amplitude nuclear finite differences off shell.
``XL_ESMD`` defaults to ``scf_backward=2`` and disables the ordinary
analytical-gradient path in its private copy of ``seqm_parameters``.  Set
``xl_bomd_params["scf_backward"] = 1`` to use the implicit response solver;
both 1 and 2 pass the force finite-difference tests.  In the methanal stability
sweep, mode 2 was faster and avoided an implicit-response convergence warning,
so it remains the default.  Direct SCF backpropagation still requires a well
converged forward SCF calculation.

For Cases 2a/2b, autodiff is applied to the total block shadow energy while
the stationary shadow amplitudes are detached.  This is intentional: SCF
backpropagation supplies orbital response, and block stationarity removes the
need to differentiate the coupled shadow solve.  The experimental analytical
single-state XL gradient is rejected for coupled modes.

The hand-derived shadow relaxed-density/Z-vector implementation remains
available for the independent 1a/1b and labelled ordered formulations through
the explicit ``force_mode="experimental_analytic"`` opt-in.  ``XL_ESMD`` then
enables ``analytical_gradient=[True]`` in its private parameter copy.  This
force is exact at the CIS fixed point, but it is not exact for general
off-shell auxiliary amplitudes; use autodiff for energy-conservation studies.
Raw coupled 2a/2b deliberately reject this selected-root gradient.
The cyclopropene runner exposes it as
``--force-mode experimental_analytic`` for a direct comparison.


Diagnostics
-----------

After every electronic solve, ``molecule.xlesmd_diagnostics`` contains:

* the fixed-point residual for every state;
* the subspace-tangent fixed-point residual and the discarded internal
  (rotation plus radial) component;
* auxiliary and shadow-amplitude orthogonality errors;
* coupled electronic/KKT residuals and iteration count;
* Krylov rank, relative residual and convergence flags;
* the minimum secular pole distance for exact normalization.

Set ``verbose_xlesmd=True`` or ``verbose_krylov=True`` only when console
diagnostics are wanted.  Normal production runs are quiet.

Validation coverage
-------------------

``tests/unit/test_xlesmd.py`` checks all four static formulations, the
linearized ordered variant, constraints, fixed points, the quadratic shadow
energy identity, analytic JVPs against central differences, the coupled gauge
null mode, the horizontal projector, projected coupled GMRES against dense
pseudoinverses for both 2a and 2b, block rotation covariance, and MO-gauge
transport invariance.  ``tests/unit/test_xlesmd_integration.py`` verifies
off-shell independent and coupled block forces against nuclear finite
differences for both ``scf_backward=1`` and 2, and runs short coupled
trajectories from a consistently initialized shadow force.

Stability comparison
--------------------

For a two-state methanal block at 300 K with a 0.1 fs timestep and rank-3
projected GMRES, both methods remained finite and electronically bounded for
200 steps (20 fs).  The maximum total-energy deviations were 5.25 meV for 2a
and 5.11 meV for 2b.  Over 40 steps, rank 3 to rank 12 reduced the maximum
GMRES residual from about 0.39 to 2.7e-4 for 2a (and 0.24 to 2.6e-5 for 2b),
while changing the energy drift by less than 5 micro-eV.  Exact 2b
orthogonality therefore gave no material dynamical advantage in this test.
Use 2a for coupled-subspace production work and retain 2b as a validation
comparison; use ``ordered_linearized`` instead when the states are individually
labelled surfaces.

Cyclopropene three-state example
-------------------------------

``examples/run_xlesmd_cyclopropene.py`` is a reproducible 500-step example
using ``xyz_outputs/cyclopropene.xyz``, 0.1 fs, AM1, three CIS roots, and the
verified ``scf_backward=2`` shadow force.  For three **labelled** roots, use
the default ordered formulation, rank-1 right preconditioner, and rank 4:

.. code-block:: console

   conda activate base
   python examples/run_xlesmd_cyclopropene.py --steps 500

To compare the retained analytical gradient on the same ordered trajectory:

.. code-block:: console

   python examples/run_xlesmd_cyclopropene.py --steps 500 --force-mode experimental_analytic

The script writes ``xlesmd_runs/cyclopropene_3state_ordered.0.h5`` and an XYZ
trajectory.  It also supports an interruption-safe split run:

.. code-block:: console

   python examples/run_xlesmd_cyclopropene.py --steps 500 --stop-at 200
   python examples/run_xlesmd_cyclopropene.py --resume

With seed 197 at 300 K, the 500-step validation had a maximum NVE drift of
1.22 meV and a minimum interatomic distance of 1.022 Å.  The right
preconditioner uses the diagonal orbital-gap response plus the rank-one
normalization term, approximating ``(J_xi-I)^-1``; GMRES applies its negative
on the right because it solves ``(I-J_xi)d=F``.  It reduced the required
labelled-root rank from 6 to 4 for cyclopropene without degrading the
trajectory.  Fixed-geometry tests on formaldehyde, cyclopropene, and benzene
showed the same qualitative reduction.  For raw 2a, the ``lambda`` block
preconditioner is selected automatically; it retains interstate multiplier
coupling and lowered the rank needed for a comparable residual in the same
three-molecule sweep.

The rank-12 raw-2a solve did reduce its linear residual over short runs, but
its soft inverse directions amplified the correction by up to about ``10^3``
and do not define dynamics on a labelled adiabatic surface.  Raw ``2a``/``2b``
instead propagate the trace of a state subspace; they are available through
``--constraint-mode 2a`` or ``2b`` when that is the intended physical model.
The NVE trajectory converts downhill excited-state potential energy into
kinetic energy (its final instantaneous temperature is about 2220 K); use
``--damp 50`` when a 300 K Langevin ensemble is required instead.

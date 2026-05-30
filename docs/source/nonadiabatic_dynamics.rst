.. _nonadiabatic_dynamics:

Nonadiabatic Dynamics and Surface Hopping
=========================================

PYSEQM supports mixed quantum-classical nonadiabatic dynamics in an
adiabatic excited-state basis. The currently useful production driver is
``SurfaceHoppingDynamics``, which implements fewest-switches surface hopping
(FSSH) over a set of CIS excited states.

The propagated electronic wavefunction contains excited states only:

.. math::

   |\Psi(t)\rangle = \sum_i c_i(t) |\phi_i(R(t))\rangle

where ``i = 1`` means ``S1``, ``i = 2`` means ``S2``, and so on. The ground
state ``S0`` is not part of the propagated coefficient vector.

Which driver should I use?
--------------------------

- ``SurfaceHoppingDynamics``: FSSH. Nuclei move on one active excited-state
  surface at a time. Electronic amplitudes are propagated, stochastic hops are
  attempted from the active surface, and accepted hops rescale nuclear
  velocities along the nonadiabatic coupling vector.

- ``EhrenfestDynamics`` (Not tested): mean-field dynamics. Nuclei move on the
  population-weighted mean excited-state force. This requires all per-state
  excited-state forces.

For surface hopping, use CIS excited states. 

Minimal FSSH run
----------------

This example runs two trajectories for a formaldehyde-like molecule, starting
on ``S2``. All state numbers in user input are 1-based excited-state indices.

.. code-block:: python

   import os
   import torch

   from seqm.api import Constants, Molecule, SurfaceHoppingDynamics

   torch.set_default_dtype(torch.float64)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   os.makedirs("./runs", exist_ok=True)

   species = torch.as_tensor(
       [[8, 6, 1, 1],
        [8, 6, 1, 1]],
       dtype=torch.int64,
       device=device,
   )

   coordinates = torch.tensor(
       [
           [[0.00, 0.00, 0.00],
            [1.22, 0.00, 0.00],
            [1.82, 0.94, 0.00],
            [1.82, -0.94, 0.00]],

           [[0.00, 0.00, 0.00],
            [1.21, 0.00, 0.00],
            [1.62, 0.94, 0.00],
            [1.82, -0.84, 0.00]],
       ],
       dtype=torch.float64,
       device=device,
   )

   seqm_parameters = {
       "method": "AM1",
       "scf_eps": 1.0e-8,
       "scf_converger": [1],
       "excited_states": {
           "n_states": 4,        # propagate S1-S4; two extra roots are added internally
           "method": "cis",
           "tolerance": 1.0e-6,
       },
       "analytical_gradient": [True],
       "nonadiabatic": {
           "detect_crossings": True,
           "decohere_on_hop": False,
        },
    }

   output = {
       "molid": [0, 1],
       "prefix": "./runs/h2co_fssh",
       "print every": 1,
       "xyz": 10,
       "checkpoint every": 100,
       "h5": {
           "data": 10,
           "coordinates": 10,
           "velocities": 10,
           "forces": 10,
           "nonadiabatic": 1,
       },
   }

   const = Constants().to(device)
   molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)

   dyn = SurfaceHoppingDynamics(
       seqm_parameters=seqm_parameters,
       timestep=0.1,       # fs
       Temp=300.0,         # used only if intial molecule.velocities is not set
       initial_state=2,    # start on S2
       output=output,
   ).to(device)

   dyn.run(molecule, steps=1000, reuse_P=True, remove_com=("angular", 1), seed=7)


If ``molecule.velocities`` is already defined before ``dyn.run(...)``, PYSEQM
uses those velocities directly. They must have shape ``(nmol, natoms, 3)`` and
units of Angstrom/fs. If velocities are not set, they are sampled from a
Maxwell-Boltzmann distribution at ``Temp``. The ``seed`` argument controls both
initial velocity sampling and stochastic hop draws.

Important FSSH inputs
---------------------

``excited_states['n_states']``
  Number of excited states included in the nonadiabatic electronic manifold.
  If you set ``4``, FSSH propagates amplitudes on ``S1`` through ``S4``.
  The driver requests two additional CIS roots internally to make the target
  states more stable. The propagated manifold and HDF5 nonadiabatic output
  still contain only the requested ``n_states``.

``initial_state``
  Initial active excited state. This is 1-based: ``1`` means ``S1``.
  You may also pass a 1D tensor of length ``nmol`` to start different
  trajectories on different surfaces. For nonadiabatic dynamics,
  ``initial_state`` overrides ``seqm_parameters['active_state']`` during
  initialization. The ``active_state`` key is used for adiabatic excited-state
  BOMD, but FSSH and Ehrenfest reset ``molecule.active_state`` from
  ``initial_state`` before the first force evaluation.

  The electronic amplitudes are initialized as a pure adiabatic state. If
  ``initial_state=2``, the propagated coefficient vector starts as
  ``c(S2) = 1 + 0j`` and all other excited-state coefficients start at zero.
  Internally this means each trajectory has population 1 on its requested
  initial state and population 0 on every other propagated excited state.
  For a batched tensor such as ``initial_state=torch.tensor([1, 3, 2])``,
  molecule 0 starts on ``S1``, molecule 1 starts on ``S3``, and molecule 2
  starts on ``S2``.

  On a fresh run, this pure-state initialization is applied before the first
  electronic-structure call. On a restart, the saved electronic amplitudes and
  active surfaces from the checkpoint are restored instead; ``initial_state``
  is not reapplied.

``timestep``
  Nuclear time step in fs. Surface hopping usually needs a smaller time step
  than ground-state BOMD; check convergence for your molecule and energy gaps.

``nonadiabatic['tdc_method']``
  Method for calculating the time-derivative nonadiabatic coupling
  :math:`\tau_{ij} = \dot{R}\cdot d_{ij}` used in electronic propagation.
  ``'hamiltonian_fd'`` is the default and recommended CIS option. It computes
  the nonadiabatic coupling from finite differences of the Hamiltonian.
  ``'overlap'`` uses finite differences of state
  overlaps and should be treated as a diagnostic option since the 
  implementation is not verified.

``nonadiabatic['decohere_on_hop']``
  If ``True``, collapse the electronic amplitudes after stochastic hop
  decisions: accepted hops collapse to the target state, and frustrated hops
  collapse back to the current active state. The default is ``False``. This is
  a simple collapse rule, not a continuous decoherence-time correction.

``nonadiabatic['detect_crossings']``
  Enables trivial crossing detection. The default is ``True``.

Output and HDF5 layout
----------------------

Set ``output['h5']['nonadiabatic']`` to a positive cadence to write the
nonadiabatic state history. The data are written under
``/data/nonadiabatic`` in each per-molecule HDF5 file:

- ``/data/nonadiabatic/steps``: absolute MD step indices.
- ``/data/nonadiabatic/active_surface``: active excited-state surface,
  1-based.
- ``/data/nonadiabatic/electronic_amplitudes``: shape ``(T, R, 2)`` where
  ``R`` is the number of propagated excited states and the last dimension is
  ``(real, imag)``.
- ``/data/nonadiabatic/NACT``: shape ``(T, R, R)`` time-derivative coupling
  matrix in the propagated excited-state manifold.

The ordinary MD output groups described in :doc:`bomd` still work. In
particular, ``/data/excitation/state_energies`` stores absolute energies for
``S0`` and the excited states when ``output['h5']['data']`` is enabled.

For excited-state trajectories, output requests in ``output['h5']`` also drive
what excited-state properties are computed:

- If ``output['h5']['data'] > 0``, transition dipoles and oscillator strengths
  are computed so they can be written under ``/data/excitation``.
- If ``output['h5']['transition_density_matrices'] > 0``, transition density
  matrices are computed and written under
  ``/data/excitation/transition_density_matrices``.

You do not need to set separate compute flags in ``seqm_parameters`` for these
HDF5 outputs.

Reading populations from HDF5
-----------------------------

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File("./runs/h2co_fssh.0.h5", "r") as h5:
       g = h5["data/nonadiabatic"]
       steps = g["steps"][...]
       active = g["active_surface"][...]
       amps = g["electronic_amplitudes"][...]

   coeff = amps[..., 0] + 1j * amps[..., 1]
   populations = np.abs(coeff) ** 2

   print("steps:", steps)
   print("active surfaces:", active)
   print("S1 population:", populations[:, 0])

Restarts
--------

Nonadiabatic runs use the same checkpoint system as BOMD. Set
``output['checkpoint every']`` to a positive step interval. PYSEQM writes
``{prefix}.restart.pt`` atomically after flushing HDF5 and XYZ output.

To resume a stopped FSSH calculation:

.. code-block:: python

   from seqm.api import SurfaceHoppingDynamics

   SurfaceHoppingDynamics.run_from_checkpoint(
       "./runs/h2co_fssh.restart.pt",
       device=device,
   )

Do not recreate the molecule or driver manually for a normal restart. The
checkpoint stores the molecule, density matrix and CIS amplitudes when
``reuse_P=True``, velocities, forces, active surfaces, electronic amplitudes,
trivial-crossing holdoff state, cached time-derivative couplings, and random
number generator state. The restarted job continues from the saved step until
the original total ``steps`` value is reached, appending to the same HDF5 and
XYZ files with absolute step numbers.

Getting nonadiabatic coupling vectors
-------------------------------------

You can request CIS nonadiabatic coupling vectors in a single-point excited
state calculation by setting ``seqm_parameters['nonadiabatic']``. State labels
in the input are 1-based.

.. code-block:: python

   from seqm.api import Electronic_Structure, Molecule

   seqm_parameters = {
       "method": "AM1",
       "scf_eps": 1.0e-8,
       "scf_converger": [1],
       "excited_states": {"n_states": 4, "method": "cis", "tolerance": 1.0e-6},
       "nonadiabatic": {
           "compute_nac": True,
           "pairs": [(1, 2), (2, 3)],
           # Or use "states": [1, 2, 3] to request all unique pairs among them.
       },
   }

   molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
   esdriver = Electronic_Structure(seqm_parameters).to(device)
   esdriver(molecule)

   # molecule.nac is a sparse dict keyed by 0-based (i, j) with i < j.
   # Each value has shape (nmol, natoms, 3).
   nac_s1_s2 = molecule.nac[(0, 1)]

For reverse ordering, use antisymmetry explicitly:
``d_ji = -molecule.nac[(i, j)]`` for ``i < j``.
NAC vectors have units of inverse Angstrom and are currently implemented for
CIS, not RPA.

Trivial crossings and trivial hops
----------------------------------

CIS state ordering can swap near very small energy gaps. When
``detect_crossings`` is enabled, PYSEQM compares the old and new CIS amplitude
overlap matrix. If two states have effectively exchanged identity, the driver
treats this as a trivial crossing instead of a stochastic hop:

- electronic amplitudes are relabeled to follow the state identities,
- the active surface is relabeled if it participates in the crossing,
- the corresponding time-derivative coupling is zeroed for that step so a
  stochastic hop is not attempted for the same swap,
- a short post-hop holdoff prevents immediate repeated hop attempts,
- the event is recorded in ``dyn.hop_log`` with reason ``"Trivial crossing"``.

In HDF5 output this appears as a change in
``/data/nonadiabatic/active_surface``. In the printed hop log it is reported
like an accepted event, but it is deterministic relabeling rather than a
velocity-rescaled stochastic hop.

.. Ehrenfest dynamics
.. ------------------
..
.. ``EhrenfestDynamics`` uses the same electronic propagation machinery but
.. replaces active-surface forces with population-weighted mean forces. It needs
.. all excited-state gradients at each step:
..
.. .. code-block:: python
..
..    from seqm.api import EhrenfestDynamics
..
..    seqm_parameters["do_all_forces"] = True
..
..    dyn = EhrenfestDynamics(
..        seqm_parameters=seqm_parameters,
..        timestep=0.1,
..        Temp=300.0,
..        initial_state=2,
..        output=output,
..    ).to(device)
..
..    dyn.run(molecule, steps=1000, reuse_P=True, remove_com=None)
..
.. Because all per-state forces are evaluated, Ehrenfest dynamics is usually much
.. more expensive than FSSH with active-surface forces.

Current limitations and checks
------------------------------

- Use CIS for FSSH. NAC vectors needed for hop velocity rescaling are not
  implemented for RPA.
- The propagated electronic amplitudes cover excited states only; there is no
  ``S0`` amplitude.
- Excited-state calculations require a homogeneous batch: all trajectories in
  the batch must have the same atom ordering and composition.
- Hops are stochastic. Use ``seed=...`` for reproducible fresh runs; restarts
  restore the RNG state stored in the checkpoint.

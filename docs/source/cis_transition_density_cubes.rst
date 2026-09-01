.. _cis-transition-density-cubes:

CIS Transition-Density Cube Files
=================================

PYSEQM can export real-space CIS transition densities as Gaussian
cube files for MNDO-type methods: ``MNDO``, ``AM1``, and ``PM3``. These
files help visualize the spatial character of a transition from the ground
state to an excited state.

Generating cube files
---------------------

Use ``scripts/cis_transition_density_cubes.py`` to run a CIS calculation for
one XYZ geometry and write cubes for its leading excited states. For example,
this calculates ten AM1/CIS states and writes transition-density cubes for the
first three:

.. code-block:: bash

   python scripts/cis_transition_density_cubes.py molecule.xyz \
       --method AM1 \
       --n-states 10 \
       --cube-states 3 \
       --out-dir cis_transition_density

Supported ``--method`` values are ``MNDO``, ``AM1``, and ``PM3``. The default
is ``AM1``.

The output directory contains:

- ``cis_energies.csv``: excitation energies and oscillator strengths for all
  requested CIS states.
- ``state_01_transition_density.cube``, etc.: signed transition-density
  volumes for the requested leading states.

The grid spacing defaults to 0.20 Angstrom and the grid extends 3.0 Angstrom
past the outermost atoms. Adjust these values when needed:

.. code-block:: bash

   python scripts/cis_transition_density_cubes.py molecule.xyz \
       --method PM3 \
       --spacing 0.15 \
       --padding 4.0

``--chunk-size`` controls how many grid points are evaluated at once. Reduce
it if cube generation uses too much memory.


Basis and orthogonalization
---------------------------

The exporter evaluates PYSEQM's normalized Slater-type valence basis: one
``1s`` function per hydrogen and ``[ns, npx, npy, npz]`` functions per heavy
atom. The CIS amplitudes are first transformed from PYSEQM's orthonormal NDDO
AO representation into this non-orthogonal Slater basis using symmetric
Lowdin orthogonalization:

.. math::

   R_{\mathrm{STO}} = S^{-1/2} R_{\mathrm{NDDO}} S^{-1/2}.

This transformation is necessary before evaluating the real-space density.

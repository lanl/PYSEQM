Features
========



PySEQM supports input data from `.xyz` files or directly from PyTorch tensors.

Reading from .xyz files:

.. code-block:: python

    species, coordinates = read_xyz([
        '../../data_one.xyz',
        '../../data_two.xyz',
        '../../data_three.xyz'
    ])



Manual tensor input:

.. code-block:: python

    species = torch.as_tensor([
        [8, 6, 1, 1],
        [8, 6, 1, 1]
    ], dtype=torch.int64, device=device)

    coordinates = torch.tensor([
        [
            [0.00, 0.00, 0.00],
            [1.22, 0.00, 0.00],
            [1.82, 0.94, 0.00],
            [1.82, -0.94, 0.00]
        ],
        [
            [0.00, 0.00, 0.00],
            [1.22, 0.00, 0.20],
            [1.82, 0.94, 0.00],
            [1.81, -0.93, -0.20]
        ]
    ], device=device)

Padding for Variable-Length Molecules
-------------------------------------

When batching molecules with different numbers of atoms, **0-padding** must be applied so all inputs have the same shape.

- Pad species with `0` to represent a non-existent atom.
- Pad coordinates with `[0.0, 0.0, 0.0]` for those atoms.

Example: Padding a molecule with 3 atoms to match a batch of size 4:

.. code-block:: python

    species = torch.as_tensor([8, 6, 1, 0], dtype=torch.int64, device=device)
    coordinates = torch.tensor([
        [0.00, 0.00, 0.00],
        [1.22, 0.00, 0.00],
        [1.82, 0.94, 0.00],
        [0.00, 0.00, 0.00]  # padded atom
    ], device=device)

Ensure the batch tensor dimensions are consistent across all molecules:
- `species`: shape `[batch_size, max_atoms]`
- `coordinates`: shape `[batch_size, max_atoms, 3]`

Device and Precision
--------------------

Set the computation device automatically using:

.. code-block:: python

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

This ensures your code will use a GPU if available, and fall back to CPU otherwise.

All tensors in PySEQM should be explicitly placed on the correct device to avoid runtime errors:

.. code-block:: python

    tensor.to(device)

Precision
~~~~~~~~~

PySEQM requires **double precision** floats (`torch.float64` for floats) to maintain numerical accuracy in electronic structure and molecular dynamics calculations.

Specify the dtype when creating new tensors. For example:

.. code-block:: python
    dtype=torch.float64  # for coordinates, etc.

Using lower precision (e.g., `torch.float32`) may result in:

- Inaccurate energy or force evaluations
- Unstable SCF convergence
- Incorrect gradient behavior in autograd


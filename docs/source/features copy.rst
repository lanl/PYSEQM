Features
========

PySEQM can take in data in either a .xyz file or in vector format.


.. code-block:: python

        species, coordinates = read_xyz(['../../data_one.xyz', '../../data_two.xyz', '../../data_three.xyz'])


        species, coordinates = read_xyz([['../../data_one.xyz'], ['../../data_two.xyz'], ['../../data_three.xyz']])



.. code-block:: python

        species = torch.as_tensor([
                        [8,6,1,1],
                        [8,6,1,1],
                        ],dtype=torch.int64, device=device)

        coordinates = torch.tensor([
                        [
                        [0.00,    0.00,    0.00],
                        [1.22,    0.00,    0.00],
                        [1.82,    0.94,    0.00],
                        [1.82,   -0.94,    0.00]
                        ],
                        [
                        [0.00,    0.00,    0.00],
                        [1.22,    0.00,    0.20],
                        [1.82,    0.94,    0.00],
                        [1.81,   -0.93,    -0.20]
                        ],
                        ],device=device)


In PySEQM, 0 padding must be used when working with batched molecules that have a different number of atoms.


.. code-block:: python

    [0.0,0.0,0.0]           

PySEQM requires 64-bit precision for accurate and stable calculations.
.. code-block:: python

    dtype=torch.int64


PySEQM uses PyTorch, which allows you to run calculations on the GPU.


We set the device to either GPU or CPU.

.. code-block:: python


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Then we can set device=device so that calculations are run on the correct hardware.

.. code-block:: python


    ([],device=device)
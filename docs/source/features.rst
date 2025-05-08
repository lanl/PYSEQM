Features
========

PySEQM can do take in data in eithere a .xyz file or in vector formate.


.. code-block:: python

        species, coordinates = read_xyz(['../../data.xyz'])


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


In PySEQM 0 padding can be used when working with batched molecules that have a differnet number of atoms 

.. code-block:: python

    [0.0,0.0,0.0]           

You can get set the torch data type with either bit 64 and 32 based on the levle of accuracy you want for your values you are working with PySEQM is set to float 64 but can be change by
 
.. code-block:: python

    dtype=torch.int32

    dtype=torch.int64


PySEQM uses PyTorch which gives you to run callcation on GPU.

We set device to either GPU or CPU.

Then we can 

.. code-block:: python


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Then we can set device=device so that calculations are ran on the correct hardware.

.. code-block:: python


    ([],device=device)
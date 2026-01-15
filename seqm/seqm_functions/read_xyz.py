import numpy as np

# fmt: off
_element_dict = {
            'H':  1,                                                                                                                                 'He':2,
            'Li': 3, 'Be': 4,                                                                                'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne':10,
            'Na': 11,'Mg':12,                                                                                'Al':13,'Si':14,'P': 15,'S': 16,'Cl':17,'Ar':18,
            'K':  19,'Ca':20,'Sc':21,'Ti':22,'V': 23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,
            'Rb': 37,'Sr':38,'Y': 39,'Zr':40,'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,'Sb':51,'Te':52,'I': 53,'Xe':54,
            }
# fmt: on


def read_xyz(files, sort=True):
    """
    Read .xyz files, pad all to same atom-count, return NumPy arrays.

    Returns:
        species: shape (M, K) int array of atomic numbers
        coords:  shape (M, K, 3) float array of XYZ coords
    where M = len(files), K = max atoms in any file.
    """
    mols = []
    for fn in files:
        with open(fn) as f:
            lines = f.readlines()
        n = int(lines[0])
        # detect if first field is numeric
        numeric = lines[2].split()[0].isdigit()
        data = np.zeros((n, 4), float)
        for i, L in enumerate(lines[2 : 2 + n]):
            a, *xyz = L.split()
            data[i, 0] = int(a) if numeric else _element_dict[a]
            data[i, 1:4] = list(map(float, xyz))
        if sort:  # sort such than atoms are ordered in decreasing atomic numbers
            data = data[data[:, 0].argsort(kind="stable")[::-1]]
        mols.append(data)
    # pad out
    M = len(mols)
    K = max(m.shape[0] for m in mols)
    species = np.zeros((M, K), int)
    coords = np.zeros((M, K, 3), float)
    for i, m in enumerate(mols):
        k = m.shape[0]
        species[i, :k] = m[:, 0].astype(int)
        coords[i, :k, :] = m[:, 1:]
    return species, coords

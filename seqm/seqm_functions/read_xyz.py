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


def read_xyz_trajectory(file, start=0, stop=None, step=1, num=None, sort=True):
    """
    Read concatenated XYZ frames from one file.

    Use start/stop/step slicing, or set num to sample evenly between
    start and stop (inclusive).
    """

    def _parse_atoms(atom_lines):
        if not atom_lines:
            return np.zeros((0, 4), float)
        numeric = atom_lines[0].split()[0].isdigit()
        data = np.zeros((len(atom_lines), 4), float)
        for i, L in enumerate(atom_lines):
            a, *xyz = L.split()
            data[i, 0] = int(a) if numeric else _element_dict[a]
            data[i, 1:4] = list(map(float, xyz[:3]))
        if sort:  # sort such than atoms are ordered in decreasing atomic numbers
            data = data[data[:, 0].argsort(kind="stable")[::-1]]
        return data

    with open(file) as f:
        lines = f.readlines()

    frames = []
    i = 0
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        try:
            n = int(lines[i].split()[0])
        except ValueError as exc:
            raise ValueError(f"Expected atom count on line {i + 1}: {lines[i].strip()}") from exc
        if i + 1 >= len(lines):
            raise ValueError(f"Missing comment line after atom count on line {i + 1}")
        start_atoms = i + 2
        end_atoms = start_atoms + n
        if end_atoms > len(lines):
            raise ValueError("Truncated XYZ frame.")
        frames.append(_parse_atoms(lines[start_atoms:end_atoms]))
        i = end_atoms

    if not frames:
        return np.zeros((0, 0), int), np.zeros((0, 0, 3), float)

    total = len(frames)
    if num is None:
        selected = frames[start:stop:step]
    else:
        if step != 1:
            raise ValueError("Provide either step or num, not both.")
        start_idx = 0 if start is None else (start + total if start < 0 else start)
        stop_idx = (total - 1) if stop is None else (stop + total if stop < 0 else stop)
        start_idx = max(0, min(start_idx, total - 1))
        stop_idx = max(0, min(stop_idx, total - 1))
        if start_idx > stop_idx:
            raise ValueError("start must be <= stop when num is provided.")
        count = stop_idx - start_idx + 1
        if num >= count:
            selected = frames[start_idx : stop_idx + 1]
        else:
            indices = np.rint(np.linspace(start_idx, stop_idx, num=num)).astype(int)
            selected = [frames[i] for i in indices]

    if not selected:
        return np.zeros((0, 0), int), np.zeros((0, 0, 3), float)

    M = len(selected)
    K = max(m.shape[0] for m in selected)
    species = np.zeros((M, K), int)
    coords = np.zeros((M, K, 3), float)
    for i, m in enumerate(selected):
        k = m.shape[0]
        species[i, :k] = m[:, 0].astype(int)
        coords[i, :k, :] = m[:, 1:]
    return species, coords

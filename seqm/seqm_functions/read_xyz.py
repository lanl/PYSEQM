import os

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


def read_xyz_trajectory(
    file, start=0, stop=None, step=1, num=None, sort=True, vel_file=None, return_vel=False
):
    """
    Read concatenated XYZ frames from one file.

    Use start/stop/step slicing, or set num to sample evenly between
    start and stop (inclusive).

    Set return_vel=True to also read velocity.out in the same folder
    (or pass vel_file).
    """

    def _count_frames(path):
        count = 0
        with open(path) as fh:
            while True:
                line = fh.readline()
                if not line:
                    break
                while line and not line.strip():
                    line = fh.readline()
                if not line:
                    break
                try:
                    n = int(line.split()[0])
                except ValueError as exc:
                    raise ValueError(f"Expected atom count: {line.strip()}") from exc
                if not fh.readline():
                    raise ValueError("Truncated XYZ frame.")
                for _ in range(n):
                    if not fh.readline():
                        raise ValueError("Truncated XYZ frame.")
                count += 1
        return count

    def _read_atoms(fh, n):
        data = np.zeros((n, 4), float)
        line = fh.readline()
        if not line:
            raise ValueError("Truncated XYZ frame.")
        toks = line.split()
        numeric = toks[0].isdigit()
        data[0, 0] = int(toks[0]) if numeric else _element_dict[toks[0]]
        data[0, 1:4] = list(map(float, toks[1:4]))
        for i in range(1, n):
            line = fh.readline()
            if not line:
                raise ValueError("Truncated XYZ frame.")
            toks = line.split()
            data[i, 0] = int(toks[0]) if numeric else _element_dict[toks[0]]
            data[i, 1:4] = list(map(float, toks[1:4]))
        if sort:
            order = data[:, 0].argsort(kind="stable")[::-1]
            data = data[order]
            return data, order
        return data, None

    def _skip_atoms(fh, n):
        for _ in range(n):
            if not fh.readline():
                raise ValueError("Truncated XYZ frame.")

    def _parse_vel_line(line):
        toks = line.split()
        if len(toks) < 3:
            return None
        if len(toks) >= 4 and (toks[0].isdigit() or toks[0] in _element_dict):
            xyz = toks[1:4]
        else:
            xyz = toks[:3]
        try:
            return list(map(float, xyz))
        except ValueError:
            return None

    def _read_vel_frame(vh, n):
        line = vh.readline()
        while line and "$VELOC" not in line.upper():
            line = vh.readline()
        if not line:
            raise ValueError("velocity.out missing $VELOC block.")
        vals = np.zeros((n, 3), float)
        i = 0
        while i < n:
            line = vh.readline()
            if not line:
                raise ValueError("velocity.out truncated.")
            s = line.strip()
            if not s or "$END" in s.upper():
                continue
            vec = _parse_vel_line(s)
            if vec is None:
                continue
            vals[i, :] = vec
            i += 1
        return vals

    need_total = num is not None or (start is not None and start < 0) or (stop is not None and stop < 0)
    total = _count_frames(file) if need_total else None
    if total == 0:
        return np.zeros((0, 0), int), np.zeros((0, 0, 3), float)

    if num is None:
        start_idx = 0 if start is None else (start + total if total is not None and start < 0 else start)
        stop_idx = (
            total
            if stop is None and total is not None
            else (stop + total if total is not None and stop < 0 else stop)
        )
        start_idx = max(0, start_idx)
        if total is not None:
            start_idx = min(start_idx, total)
        if stop_idx is not None:
            stop_idx = max(0, stop_idx)
            if total is not None:
                stop_idx = min(stop_idx, total)
            if start_idx >= stop_idx:
                return np.zeros((0, 0), int), np.zeros((0, 0, 3), float)
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
        use_all = num >= count
        if not use_all:
            idx_list = np.rint(np.linspace(start_idx, stop_idx, num=num)).astype(int).tolist()
            last_idx = idx_list[-1] if idx_list else -1

    if return_vel:
        if vel_file is None:
            guess = os.path.join(os.path.dirname(file), "velocity.out")
            if os.path.exists(guess):
                vel_file = guess
        if vel_file is None or not os.path.exists(vel_file):
            raise FileNotFoundError("velocity.out not found (or vel_file not provided).")

    selected = []
    vselected = [] if return_vel else None

    with open(file) as f, open(vel_file) if return_vel else open(os.devnull) as vf:
        idx = 0
        pos = 0
        while True:
            line = f.readline()
            if not line:
                break
            while line and not line.strip():
                line = f.readline()
            if not line:
                break
            try:
                n = int(line.split()[0])
            except ValueError as exc:
                raise ValueError(f"Expected atom count: {line.strip()}") from exc
            if not f.readline():
                raise ValueError("Truncated XYZ frame.")

            if num is None:
                if stop_idx is not None and idx >= stop_idx:
                    break
                take = idx >= start_idx and ((idx - start_idx) % step == 0)
                dup = 1
            else:
                if use_all:
                    if idx > stop_idx:
                        break
                    take = idx >= start_idx
                    dup = 1
                else:
                    if idx > last_idx:
                        break
                    dup = 0
                    while pos < len(idx_list) and idx_list[pos] == idx:
                        dup += 1
                        pos += 1
                    take = dup > 0

            if take:
                data, order = _read_atoms(f, n)
                if return_vel:
                    v = _read_vel_frame(vf, n)
                    if order is not None:
                        v = v[order]
                for _ in range(dup):
                    selected.append(data)
                    if return_vel:
                        vselected.append(v)
            else:
                _skip_atoms(f, n)
                if return_vel:
                    _read_vel_frame(vf, n)

            idx += 1

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
    if return_vel:
        vels = np.zeros((M, K, 3), float)
        for i, v in enumerate(vselected):
            k = v.shape[0]
            vels[i, :k, :] = v[:, :3]
        return species, coords, vels
    return species, coords

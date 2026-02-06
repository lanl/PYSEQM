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
    Read concatenated XYZ frames from coords.xyz file produced by NEXMD

    Use start/stop/step slicing, or set num to sample evenly between
    start and stop (inclusive).

    Set return_vel=True to also read velocity.out (from NEXMD) in the same folder
    (or pass vel_file).
    
    assumptions:

    XYZ per frame (no blank lines):
        n
        comment
        n lines: <id_or_elem> x y z

    velocity.out per frame (fixed, no blank lines):
        comment
        $VELOC
        n lines: vx vy vz
        $ENDVELOC

    Also:
      - n constant across frames
      - first column (id_or_elem) constant across frames (so we compute species + order once)
    """

    if num is not None and step != 1:
        raise ValueError("Provide either step or num, not both.")

    # Count frames only if needed (num or negative indices)
    need_total = (num is not None) or (start < 0) or (stop is not None and stop < 0)

    def count_frames():
        c = 0
        with open(file, "r", buffering=8<<20) as f:
            while True:
                nline = f.readline()
                if not nline: break
                n = int(nline)
                f.readline()                       # comment
                for _ in range(n): f.readline()    # atoms
                c += 1
        return c

    total = count_frames() if need_total else None
    if total == 0:
        zS = np.zeros((0, 0), int); zC = np.zeros((0, 0, 3), float)
        return (zS, zC, zC.copy()) if return_vel else (zS, zC)

    if start < 0: start += total
    if stop is not None and stop < 0: stop += total

    if num is None:
        if stop is None and total is not None: stop = total
        idxs = np.arange(start, stop, step, dtype=int)
    else:
        if total is None: total = count_frames()
        if stop is None: stop = total - 1
        idxs = np.rint(np.linspace(start, stop, num=num)).astype(int)
        idxs = np.unique(idxs)  # drop duplicates for speed/clarity

    if idxs.size == 0:
        zS = np.zeros((0, 0), int); zC = np.zeros((0, 0, 3), float)
        return (zS, zC, zC.copy()) if return_vel else (zS, zC)

    want = set(idxs.tolist())
    M = idxs.size

    # ---- velocity file ----
    if return_vel:
        if vel_file is None:
            vel_file = os.path.join(os.path.dirname(file), "velocity.out")
        if not os.path.exists(vel_file):
            raise FileNotFoundError("velocity.out not found (or vel_file not provided).")

    def read_n_lines(fh, n):
        lines = [fh.readline() for _ in range(n)]
        if any(l == "" for l in lines):
            raise ValueError("Truncated frame.")
        return lines

    def read_vel(vh, n):
        vh.readline()  # comment
        if vh.readline().strip().upper() != "$VELOC": raise ValueError("Bad velocity.out ($VELOC).")
        v = np.fromstring("".join(read_n_lines(vh, n)), sep=" ")
        if v.size != n * 3: raise ValueError("Bad velocity.out (vector count).")
        if vh.readline().strip().upper() != "$ENDVELOC": raise ValueError("Bad velocity.out ($ENDVELOC).")
        return v.reshape(n, 3)

    # ---- main pass ----
    species_fixed = order = None
    numeric = None
    K = None

    with open(file, "r", buffering=64<<20) as f, (open(vel_file, "r", buffering=64<<20) if return_vel else open(os.devnull, "r")) as vf:
        # first pass: find K from first frame header
        nline = f.readline()
        if not nline: raise ValueError("Empty XYZ.")
        K = int(nline)
        f.seek(0)

        species = np.empty((M, K), dtype=int)
        coords  = np.empty((M, K, 3), dtype=float)
        vels    = np.empty((M, K, 3), dtype=float) if return_vel else None

        out_i = 0
        frame_i = 0
        while True:
            nline = f.readline()
            if not nline: break
            n = int(nline)
            if n != K: raise ValueError("Variable atom count not supported in fast path.")
            f.readline()  # comment

            atom_lines = read_n_lines(f, K)

            if frame_i in want:
                if species_fixed is None:
                    t0 = atom_lines[0].split(None, 1)[0]
                    numeric = t0.isdigit()
                    if numeric:
                        a = np.fromstring("".join(atom_lines), sep=" ").reshape(K, 4)
                        sp = a[:, 0].astype(int, copy=False)
                        co = a[:, 1:4]
                    else:
                        sp = np.array([_element_dict[l.split(None, 1)[0]] for l in atom_lines], dtype=int)
                        co = np.fromstring("".join(l.split(None, 1)[1] for l in atom_lines), sep=" ").reshape(K, 3)

                    if sort:
                        order = sp.argsort(kind="stable")[::-1]
                        species_fixed = sp[order]
                        co = co[order]
                    else:
                        order = None
                        species_fixed = sp

                else:
                    if numeric:
                        co = np.fromstring("".join(atom_lines), sep=" ").reshape(K, 4)[:, 1:4]
                    else:
                        co = np.fromstring("".join(l.split(None, 1)[1] for l in atom_lines), sep=" ").reshape(K, 3)
                    if order is not None:
                        co = co[order]

                species[out_i, :] = species_fixed
                coords[out_i, :, :] = co

                if return_vel:
                    v = read_vel(vf, K)
                    if order is not None:
                        v = v[order]
                    vels[out_i, :, :] = v
                out_i += 1
            else:
                if return_vel:
                    # skip velocity frame: fixed layout
                    vf.readline(); vf.readline()
                    for _ in range(K): vf.readline()
                    vf.readline()

            frame_i += 1

        species = species[:out_i]
        coords = coords[:out_i]
        if return_vel:
            vels = vels[:out_i]
            return species, coords, vels
        return species, coords

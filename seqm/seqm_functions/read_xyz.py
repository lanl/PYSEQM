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
            order = np.argsort(-data[:, 0], kind="stable")
            data = data[order]
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
    file, start=0, stop=None, step=1, num=None, sort=True, vel_file=None, return_vel=False, indices=None
):
    """
    Read concatenated XYZ frames from coords.xyz file produced by NEXMD

    If `indices` is provided: reads exactly those frames (in that order; duplicates allowed).
    Else:
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

    if indices is not None and num is not None:
        raise ValueError("Use either indices or num, not both.")
    if indices is not None and step != 1:
        raise ValueError("When using indices, step must be 1.")
    if num is not None and step != 1:
        raise ValueError("Provide either step or num, not both.")

    if return_vel:
        if vel_file is None:
            vel_file = os.path.join(os.path.dirname(file), "velocity.out")
        if not os.path.exists(vel_file):
            raise FileNotFoundError("velocity.out not found (or vel_file not provided).")

    def count_frames():
        c = 0
        with open(file, "r", buffering=8 << 20) as f:
            while True:
                nline = f.readline()
                if not nline:
                    break
                n = int(nline)
                f.readline()
                for _ in range(n):
                    f.readline()
                c += 1
        return c

    def read_n_lines(fh, n):
        lines = [fh.readline() for _ in range(n)]
        if any(l == "" for l in lines):
            raise ValueError("Truncated frame.")
        return lines

    def read_vel(vh, n):
        vh.readline()
        if vh.readline().strip().upper() != "$VELOC":
            raise ValueError("Bad velocity.out ($VELOC).")
        v = np.fromstring("".join(read_n_lines(vh, n)), sep=" ")
        if v.size != n * 3:
            raise ValueError("Bad velocity.out (vector count).")
        if vh.readline().strip().upper() != "$ENDVELOC":
            raise ValueError("Bad velocity.out ($ENDVELOC).")
        return v.reshape(n, 3)

    # ---- choose frames: build order_out (output order, with duplicates), and out_map (frame->output slots) ----
    if indices is not None:
        idxs = np.asarray(indices, dtype=int)
        if idxs.size == 0:
            zS = np.zeros((0, 0), int)
            zC = np.zeros((0, 0, 3), float)
            return (zS, zC, zC.copy()) if return_vel else (zS, zC)
        total = count_frames() if np.any(idxs < 0) else None
        if total is not None:
            idxs = np.where(idxs < 0, idxs + total, idxs)
        order_out = idxs.tolist()  # keep order + duplicates
    else:
        need_total = (num is not None) or (start < 0) or (stop is not None and stop < 0)
        total = count_frames() if need_total else None
        if total == 0:
            zS = np.zeros((0, 0), int)
            zC = np.zeros((0, 0, 3), float)
            return (zS, zC, zC.copy()) if return_vel else (zS, zC)

        if start < 0:
            start += total
        if stop is not None and stop < 0:
            stop += total

        if num is None:
            if stop is None and total is not None:
                stop = total
            idxs = np.arange(start, stop, step, dtype=int)
        else:
            if total is None:
                total = count_frames()
            if stop is None:
                stop = total - 1
            idxs = np.rint(np.linspace(start, stop, num=num)).astype(int)
            idxs = np.unique(idxs)  # your current behavior
        order_out = idxs.tolist()

    if len(order_out) == 0:
        zS = np.zeros((0, 0), int)
        zC = np.zeros((0, 0, 3), float)
        return (zS, zC, zC.copy()) if return_vel else (zS, zC)

    out_map = {}
    for j, fr in enumerate(order_out):
        out_map.setdefault(int(fr), []).append(j)
    want = set(out_map.keys())
    M = len(order_out)

    # ---- main pass (one scan; cache species/order; fill all requested output slots) ----
    species_fixed = order = None
    numeric = None

    with open(file, "r", buffering=64 << 20) as f, (
        open(vel_file, "r", buffering=64 << 20) if return_vel else open(os.devnull, "r")
    ) as vf:
        nline = f.readline()
        if not nline:
            raise ValueError("Empty XYZ.")
        K = int(nline)
        f.seek(0)

        species = np.empty((M, K), dtype=int)
        coords = np.empty((M, K, 3), dtype=float)
        vels = np.empty((M, K, 3), dtype=float) if return_vel else None

        frame_i = 0
        while True:
            nline = f.readline()
            if not nline:
                break
            n = int(nline)
            if n != K:
                raise ValueError("Variable atom count not supported in fast path.")
            f.readline()
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
                        co = np.fromstring("".join(l.split(None, 1)[1] for l in atom_lines), sep=" ").reshape(
                            K, 3
                        )

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
                        co = np.fromstring("".join(l.split(None, 1)[1] for l in atom_lines), sep=" ").reshape(
                            K, 3
                        )
                    if order is not None:
                        co = co[order]

                v = None
                if return_vel:
                    v = read_vel(vf, K)
                    if order is not None:
                        v = v[order]

                for j in out_map[frame_i]:
                    species[j, :] = species_fixed
                    coords[j, :, :] = co
                    if return_vel:
                        vels[j, :, :] = v
            else:
                if return_vel:
                    vf.readline()
                    vf.readline()
                    for _ in range(K):
                        vf.readline()
                    vf.readline()

            frame_i += 1

    return (species, coords, vels) if return_vel else (species, coords)

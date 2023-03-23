import numpy as np

def read_xyz(files):
    '''
    reads xyz structure from a list (files) of files names
    '''
    COORDINATES = []
    for file in files:
        f = open(file)
        lines = f.readlines()
        f.close()
        Natoms = int(lines[0])
        coords = []
        for i in range(2, 2+Natoms):
            #species.append(int(lines[i].split()[0]))
            coords.append([int(lines[i].split()[0]), float(lines[i].split()[1]), float(lines[i].split()[2]), float(lines[i].split()[3])])
        COORDINATES.append(coords)
    COORDINATES = np.array(COORDINATES)
    COORDINATES = np.array([x[(-1*x[ :, 0]).argsort()] for x in COORDINATES])

    SPECIES =COORDINATES[:,:,0].astype(int)
    COORDINATES = COORDINATES[:,:,1:4]

    return SPECIES, COORDINATES
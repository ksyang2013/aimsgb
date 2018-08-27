import numpy as np
try:
    from . import coord_utils_cython as cuc
except ImportError:
    pass


def lattice_points_in_supercell(supercell_matrix):
    diagonals = np.array(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
         [1, 1, 0], [1, 1, 1]])
    d_points = np.dot(diagonals, supercell_matrix)

    mins = np.min(d_points, axis=0)
    maxes = np.max(d_points, axis=0) + 1

    ar = np.arange(mins[0], maxes[0])[:, None] * np.array([1, 0, 0])
    br = np.arange(mins[1], maxes[1])[:, None] * np.array([0, 1, 0])
    cr = np.arange(mins[2], maxes[2])[:, None] * np.array([0, 0, 1])

    all_points = ar[:, None, None] + br[None, :, None] + cr[None, None, :]
    all_points = all_points.reshape((-1, 3))

    frac_points = np.dot(all_points, np.linalg.inv(supercell_matrix))

    tvects = frac_points[np.all(frac_points < 1 - 1e-10, axis=1)
                         & np.all(frac_points >= -1e-10, axis=1)]
    assert len(tvects) == round(abs(np.linalg.det(supercell_matrix)))
    return tvects


def all_distances(coords1, coords2):
    c1 = np.array(coords1)
    c2 = np.array(coords2)
    z = (c1[:, None, :] - c2[None, :, :]) ** 2
    return np.sum(z, axis=-1) ** 0.5


def pbc_diff(fcoords1, fcoords2):
    fdist = np.subtract(fcoords1, fcoords2)
    return fdist - np.round(fdist)


def pbc_shortest_vectors(lattice, fcoords1, fcoords2, mask=None, return_d2=False):
    return cuc.pbc_shortest_vectors(lattice, fcoords1, fcoords2, mask, return_d2)

from __future__ import division

import re
import warnings
import collections
from tabulate import tabulate
from fractions import Fraction
from math import sqrt, atan, degrees, pi
import numpy as np
from numpy import sin, cos, ceil, radians, inner, identity
from numpy.linalg import inv, det, norm, solve
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from aimsgb import Grain
from aimsgb.utils import reduce_vector, co_prime, plus_minus_gen, \
    is_integer, get_smallest_multiplier, reduce_integer, transpose_matrix

__author__ = "Jianli Cheng, Kesong Yang"
__copyright__ = "Copyright 2018, Yanggroup"
__maintainer__ = "Jianli Cheng"
__email__ = "jic198@ucsd.edu"
__status__ = "Production"
__date__ = "January 26, 2018"

# SIGMA_SYMBOL = u'\u03A3'
UNIMODULAR_MATRIX = np.array([identity(3),
                              [[1, 0, 1],
                               [0, 1, 0],
                               [0, 1, 1]],
                              [[1, 0, 1],
                               [0, 1, 0],
                               [0, 1, -1]],
                              [[1, 0, 1],
                               [0, 1, 0],
                               [-1, 1, 0]],
                              [[1, 0, 1],
                               [1, 1, 0],
                               [1, 1, 1]]])
STRUCTURE_MATRIX = np.array([identity(3),
                             [[0.5, -0.5, 0],
                              [0.5, 0.5, 0],
                              [0.5, 0.5, 1]],
                             [[0.5, 0.5, 0],
                              [0, 0.5, 0.5],
                              [0.5, 0, 0.5]]])


@transpose_matrix
def reduce_csl(csl):
    """
    Reduce CSL matrix
    Args:
        csl: 3x3 matrix

    Returns:
        3x3 CSL matrix
    """
    csl = csl.round().astype(int)
    return np.array([reduce_vector(i) for i in csl])


@transpose_matrix
def o_lattice_to_csl(o_lattice, n):
    """
    The algorithm was borrowed from gosam project with slight changes.
    Link to the project: https://github.com/wojdyr/gosam

    There are two major steps: (1) Manipulate the columns of O-lattice to get
    an integral basis matrix for CSL: make two columns become integers and
    the remaining column can be multiplied by n whereby the determinant
    becomes sigma. (2) Simplify CSL so its vectors acquire the shortest length:
    decrease the integers in the matrix while keeping the determinant the same
    by adding other column vectors (row vectors in the following example) to a
    column vector. If after the addition or subtraction, the maximum value or
    absolute summation of added or subtracted vector is smaller than the
    original, then proceed the addition or subtraction.
    0 0 -1      0 0 -1      0 0 -1      0 0 -1      0 0 -1
    1 2 -1  ->  1 2 0   ->  1 2 0   ->  1 2 0   ->  1 2 0
    1 -3 2      1 -3 2      1 -3 1      1 -3 0      2 -1 0

    Args:
     o_lattice (3x3 array): O-lattice in crystal coordinates
     n (int): Number of O-lattice units per CSL unit

    Returns:
     CSL matrix (3x3 array) in crystal coordinates
    """
    csl = o_lattice.copy()
    if n < 0:
        csl[0] *= -1
        n *= -1
    while True:
        m = [get_smallest_multiplier(i) for i in csl]
        m_prod = np.prod(m)
        if m_prod <= n:
            for i in range(3):
                csl[i] *= m[i]
            if m_prod < n:
                assert n % m_prod == 0
                csl[0] *= n / m_prod
            break
        else:
            changed = False
            for i in range(3):
                for j in range(3):
                    if changed or i == j or m[i] == 1 or m[j] == 1:
                        continue
                    a, b = (i, j) if m[i] <= m[j] else (j, i)
                    for k in plus_minus_gen(1, m[b]):
                        handle = csl[a] + k * csl[b]
                        if get_smallest_multiplier(handle) < m[a]:
                            csl[a] += k * csl[b]
                            changed = True
                            break
            if not changed:
                # This situation rarely happens. Not sure if this solution is
                # legit, as det not equals to sigma. E.g. Sigma 115[113]
                for i in range(3):
                    csl[i] *= m[i]
                break
    csl = csl.round().astype(int)

    # Reshape CSL
    def simplify(l1, l2):
        x = abs(l1 + l2)
        y = abs(l1)
        changed = False
        while max(x) < max(y) or (max(x) == max(y) and sum(x) < sum(y)):
            l1 += l2
            changed = True
            x = abs(l1 + l2)
            y = abs(l1)
        return changed

    while True:
        changed = False
        for i in range(3):
            for j in range(3):
                if i != j and not changed:
                    changed = simplify(csl[i], csl[j]) or simplify(csl[i], -csl[j])
                    if changed:
                        break
        if not changed:
            break
    return csl


@transpose_matrix
def orthogonalize_csl(csl, axis):
    """
    (1) Set the 3rd column of csl same as the rotation axis. The algorithm was
    borrowed from gosam project with slight changes. Link to the project:
    https://github.com/wojdyr/gosam
    (2) Orthogonalize CSL, which is essentially a Gram-Schmidt process. At the
    same time, we want to make sure the column vectors of orthogonalized csl
    has the smallest value possible. That's why we compared two different ways.
    csl = [v1, v2, v3], vi is the column vector
    u1 = v3/||v3||, y2 = v1 - (v1 . u1)u1
    u2 = y2/||y2||, y3 = v3 - [(v3 . u1)u1 + (v3 . u2)u2]
    u3 = y3/||y3||
    """
    axis = np.array(axis)
    c = solve(csl.transpose(), axis)
    if not is_integer(c):
        mult = get_smallest_multiplier(c)
        c *= mult
    c = c.round().astype(int)
    ind = min([(i, v) for i, v in enumerate(c) if not np.allclose(v, 0)],
              key=lambda x: abs(x[1]))[0]
    if ind != 2:
        csl[ind], csl[2] = csl[2].copy(), -csl[ind]
        c[ind], c[2] = c[2], -c[ind]

    csl[2] = np.dot(c, csl)
    if c[2] < 0:
        csl[1] *= -1

    def get_integer(vec):
        # Used vec = np.array(vec, dtype=float) before, but does not work for
        # [5.00000000e-01, -5.00000000e-01,  2.22044605e-16] in Sigma3[112]
        vec = np.round(vec, 12)
        vec_sign = np.array([1 if abs(i) == i else -1 for i in vec])
        vec = list(abs(vec))
        new_vec = []
        if 0.0 in vec:
            zero_ind = vec.index(0)
            vec.pop(zero_ind)
            if 0.0 in vec:
                new_vec = [get_smallest_multiplier(vec) * i for i in vec]
            else:
                frac = Fraction(vec[0] / vec[1]).limit_denominator()
                new_vec = [frac.numerator, frac.denominator]
            new_vec.insert(zero_ind, 0)
        else:
            for i in range(len(vec) - 1):
                frac = Fraction(vec[i] / vec[i + 1]).limit_denominator()
                new_vec.extend([frac.numerator, frac.denominator])
            if new_vec[1] == new_vec[2]:
                new_vec = [new_vec[0], new_vec[1], new_vec[3]]
            else:
                new_vec = reduce_vector([new_vec[0] * new_vec[2],
                                         new_vec[1] * new_vec[2],
                                         new_vec[3] * new_vec[1]])
        assert is_integer(new_vec)
        return new_vec * vec_sign

    u1 = csl[2] / norm(csl[2])
    y2_1 = csl[1] - np.dot(csl[1], u1) * u1
    c0_1 = get_integer(y2_1)
    y2_2 = csl[0] - np.dot(csl[0], u1) * u1
    c0_2 = get_integer(y2_2)
    if sum(abs(c0_1)) > sum(abs(c0_2)):
        u2 = y2_2 / norm(y2_2)
        y3 = csl[1] - np.dot(csl[1], u1) * u1 - np.dot(csl[1], u2) * u2
        csl[1] = get_integer(y3)
        csl[0] = c0_2
    else:
        u2 = y2_1 / norm(y2_1)
        y3 = csl[0] - np.dot(csl[0], u1) * u1 - np.dot(csl[0], u2) * u2
        csl[1] = c0_1
        csl[0] = get_integer(y3)
    for i in range(3):
        for j in range(i + 1, 3):
            if not np.allclose(np.dot(csl[i], csl[j]), 0):
                raise ValueError("Non-orthogonal basis: %s" % csl)
    return csl.round().astype(int)


class GBInformation(dict):
    """
    GBInformation object essentially consists of a dictionary with information
    including sigma, CSL matrix, GB plane, rotation angle and rotation matrix
    """
    def __init__(self, axis, max_sigma, specific=False):
        """
        Creates a GBInformation object.
        Args:
            axis ([u, v, w]): Rotation axis.
            max_sigma (int): The largest sigma value
            specific (bool): Whether collecting information for a specific sigma
        """
        super(GBInformation, self).__init__()
        if max_sigma < 3:
            raise ValueError("Sigma should be larger than 2. '1' or '2' "
                             "means a layer by layer epitaxial film.")

        axis = np.array(reduce_vector(axis))
        self.axis = axis
        self.max_sigma = max_sigma
        self.specific = specific
        self.update(self.get_gb_info())

    def __str__(self):
        axis_str = "".join(map(str, self.axis))
        outs = ["Grain boundary information for rotation axis: %s" % axis_str,
                "Show the sigma values up to %s (Note: * means twist GB)"
                % (self.max_sigma)]
        data = []
        to_s = lambda x: "%.2f" % x
        for key, item in sorted(self.items()):
            for i, value in enumerate(item["plane"]):
                count = -1
                for v in value:
                    count += 1
                    plane_str = " ".join(map(str, v))
                    if v == list(self.axis):
                        plane_str += "*"
                    if count == 0:
                        row = [key, to_s(self[key]["theta"][i])]
                    else:
                        row = [None, None]
                    csl = [" ".join('%2s' % k for k in j)
                           for j in self[key]["csl"][i]]
                    row.extend(["(%s)" % plane_str, csl[count]])
                    data.append(row)
        outs.append(tabulate(data, numalign="center", tablefmt='orgtbl',
                             headers=["Sigma", "Theta", "GB Plane", "CSL"]))
        return "\n".join(outs)

    def get_gb_info(self):
        """
        Calculate sigma, rotation angle, GB plane, rotation matrix and CSL matrix
        The algorithm for getting sigma, m, n, theta is from H. Grimmer:
        https://doi.org/10.1107/S0108767384000246

        Returns:
            gb_info(dict)
        """
        max_m = int(ceil(sqrt(4 * self.max_sigma)))
        gb_info = collections.defaultdict(dict)
        sigma_theta = collections.defaultdict(list)

        for m in range(max_m):
            for n in range(max_m):
                if not co_prime(m, n):
                    continue
                sigma = self.get_sigma(m, n)
                if self.specific and sigma != self.max_sigma:
                    continue
                if not sigma or sigma > self.max_sigma:
                    continue
                theta = self.get_theta(m, n)
                sigma_theta[sigma].append([theta, m, n])

        if not sigma_theta:
            raise ValueError("Cannot find any matching GB. Most likely there "
                             "is no sigma %s%s GB." % (self.max_sigma, self.axis))
        for sigma in sigma_theta:
            sigma_theta[sigma] = sorted(sigma_theta[sigma], key=lambda t: t[0])
            min_theta = sigma_theta[sigma][0][0]
            rot_matrix = self.get_rotate_matrix(min_theta)
            csl_matrix = self.get_csl_matrix(sigma, rot_matrix)
            csl = orthogonalize_csl(csl_matrix, self.axis)
            # Sometime when getting CSL from O-lattice, det not equals to sigma.
            # That's why it needs to be reduced. E.g. Sigma 115[113]
            csl = reduce_csl(csl)
            all_csl = [csl]
            if sorted(self.axis) == [0, 0, 1]:
                ind = 0
                while True:
                    m, n = sigma_theta[sigma][ind][1:]
                    ext_csl = csl.copy()
                    for i, v in enumerate(ext_csl):
                        if sorted(v) != [0, 0, 1]:
                            if abs(v[0]) > abs(v[1]):
                                ext_csl[i] = [v[0] / abs(v[0]) * m, v[1] / abs(v[1]) * n, 0]
                            else:
                                ext_csl[i] = [v[0] / abs(v[0]) * n, v[1] / abs(v[1]) * m, 0]
                    if (csl == ext_csl).all():
                        ind += 1
                    else:
                        break
                all_csl = [csl, ext_csl]
                if ind:
                    gb_info[sigma] = {"theta": [min_theta, 90 - min_theta]}
                else:
                    gb_info[sigma] = {"theta": [90 - min_theta, min_theta]}
            else:
                gb_info[sigma] = {"theta": [min_theta]}

            gb_info[sigma].update({"plane": [[list(j) for j in i.transpose()]
                                             for i in all_csl],
                                   "rot_matrix": rot_matrix, "csl": all_csl})
        return gb_info

    def get_theta(self, m, n):
        """
        Calculate rotation angle from m, n

        Args:
            m (int)
            n (int)

        Returns:
            theta
        """
        try:
            return degrees(2 * atan(n * sqrt(inner(self.axis, self.axis)) / m))
        except ZeroDivisionError:
            return degrees(pi)

    def get_sigma(self, m, n):
        """
        Calculate sigma from m, n

        Args:
            m (int)
            n (int)

        Returns:
            sigma
        """
        sigma = m ** 2 + n ** 2 * inner(self.axis, self.axis)
        alph = 1
        while sigma != 0 and sigma % 2 == 0:
            alph *= 2
            sigma //= 2
        return sigma if sigma > 1 else None

    def get_rotate_matrix(self, angle):
        """
        use Rodrigues' rotation formula to get rotation matrix
        https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        """
        rotate_axis = np.array(self.axis / sqrt(inner(self.axis, self.axis)))
        angle = radians(angle)
        k_matrix = np.array([[0, -rotate_axis[2], rotate_axis[1]],
                            [rotate_axis[2], 0, -rotate_axis[0]],
                            [-rotate_axis[1], rotate_axis[0], 0]])
        return identity(3) + k_matrix * sin(angle) + \
               np.dot(k_matrix, k_matrix) * (1 - cos(angle))

    def get_csl_matrix(self, sigma, rotate_matrix):
        """
        Calculate CSL matrix from sigma and rotation matrix. The algorithm is
        from H. Grimmer et al. https://doi.org/10.1107/S056773947400043X
        For the structure matrix, we will use the identity matrix. Since for
        the initial structure that is not primitive cubic, we will transform it
        to conventional standard cell.
        Args:
            sigma (int): Degree of fit
            rotate_matrix (3x3 matrix): Rotation matrix

        Returns:
            3x3 CSL matrix
        """
        s = STRUCTURE_MATRIX[0]
        for u in UNIMODULAR_MATRIX:
            t = np.eye(3) - np.dot(np.dot(np.dot(u, inv(s)), inv(rotate_matrix)), s)
            if abs(det(t)) > 1e-6:
                break
        o_lattice = np.round(inv(t), 12)
        n = np.round(sigma / det(o_lattice), 6)
        csl_matrix = o_lattice_to_csl(o_lattice, n)
        return csl_matrix


class GrainBoundary(object):
    """
    A grain boundary (GB) object. The initial structure can be cubic or non-cubic
    crystal. If non-cubic, the crystal will be transferred to conventional cell.
    The generated GB could be either tilted or twisted based on the given GB
    plane. If the GB plane is parallel to the rotation axis, the generated GB
    will be a twisted one. Otherwise, tilted.
    """
    def __init__(self, axis, sigma, plane, initial_struct, uc_a=1, uc_b=1):
        """
        Build grain boundary based on rotation axis, sigma, GB plane, grain size,
        if_model and vacuum thickness.

        Args:
            axis ([u, v, w]): Rotation axis.
            sigma (int): The area ratio between the unit cell of CSL and the
                given crystal lattice.
            plane ([h, k, l]): Miller index of GB plane. If the GB plane is parallel
                to the rotation axis, the generated GB will be a twist GB. If they
                are perpendicular, the generated GB will be a tilt GB.
            initial_struct (Grain): Initial input structure. Must be an
                object of Grain
            uc_a (int): Number of unit cell of grain A. Default to 1.
            uc_b (int): Number of unit cell of grain B. Default to 1.
        """
        if not isinstance(initial_struct, Grain):
            raise ValueError("The input 'initial_struct' must be an object "
                             "of Grain.")
        self.axis = axis
        self.plane = list(plane)
        self.plane_str = " ".join(map(str, self.plane))

        if sigma % 2 == 0:
            reduce_sigma = reduce_integer(sigma)
            warnings.warn(
                "{} is an even number. However sigma must be an odd number. "
                "We will choose sigma={}.".format(sigma, reduce_sigma),
                RuntimeWarning)
            sigma = reduce_sigma
        self.sigma = sigma
        self.gb_info = GBInformation(self.axis, self.sigma, specific=True)
        self.gb_direction = None
        for i, v in enumerate(self.csl.transpose()):
            if self.plane == list(v):
                self.gb_direction = i
        sg = SpacegroupAnalyzer(initial_struct)
        new_s = sg.get_conventional_standard_structure()
        initial_struct = Grain.from_sites(new_s[:])
        self._grain_a, self._grain_b = initial_struct.build_grains(
            self.csl, self.gb_direction, uc_a, uc_b)

    @property
    def rot_matrix(self):
        """
        Rotation matrix for calculating CSL matrix
        """
        return self.gb_info[self.sigma]["rot_matrix"]

    @property
    def theta(self):
        """
        Rotation angle for calculating rotation matrix and to bring two grains
        into a perfect match
        """
        return self.gb_info[self.sigma]["theta"]

    @property
    def csl(self):
        """
        CSL matrix
        """
        for i, v in enumerate(self.gb_info[self.sigma]["plane"]):
            if self.plane in v:
                return self.gb_info[self.sigma]["csl"][i]
        avail_plane = ", ".join([", ".join([" ".join(map(str, j)) for j in i])
                                 for i in self.gb_info[self.sigma]["plane"]])
        raise ValueError("The given GB plane '%s' cannot be realized. Choose "
                         "the plane in [%s]" % (self.plane_str, avail_plane))

    @property
    def grain_a(self):
        """
        Grain class instance for grain A
        """
        return self._grain_a

    @property
    def grain_b(self):
        """
        Grain class instance for grain B
        """
        return self._grain_b

    def build_gb(self, vacuum=0.0, add_if_dist=0.0, to_primitive=True,
                 delete_layer="0b0t0b0t", tol=0.25):
        """
        Build the GB based on the given crystal, uc of grain A and B, if_model,
        vacuum thickness, distance between two grains and tolerance factor.
        Args:
            vacuum (float), Angstrom: Vacuum thickness for GB.
                Default to 0.0
            add_if_dist (float), Angstrom: Add extra distance at the interface
                between two grains.
                Default to 0.0
            to_primitive (bool): Whether to get primitive structure of GB.
                Default to true.
            delete_layer (str): Delete interface layers on both sides of each grain.
                8 characters in total. The first 4 characters is for grain A and
                the other 4 is for grain B. "b" means bottom layer and "t" means
                top layer. Integer represents the number of layers to be deleted.
                Default to "0b0t0b0t", which means no deletion of layers. The
                direction of top and bottom layers is based on gb_direction.
            tol (float), Angstrom: Tolerance factor to determine whether two
                atoms are at the same plane.
                Default to 0.25
        Returns:
             GB structure (Grain)
        """
        ind = self.gb_direction
        delete_layer = delete_layer.lower()
        delete = re.findall('(\d+)(\w)', delete_layer)
        if len(delete) != 4:
            raise ValueError("'%s' is not supported. Please make sure the format "
                             "is 0b0t0b0t.")
        for i, v in enumerate(delete):
            for j in range(int(v[0])):
                if i <= 1:
                    self.grain_a.delete_bt_layer(v[1], tol, ind)
                else:
                    self.grain_b.delete_bt_layer(v[1], tol, ind)
        abc_a = list(self.grain_a.lattice.abc)
        abc_b, angles = np.reshape(self.grain_b.lattice.parameters, (2, 3))
        if ind == 1:
            l = (abc_a[ind] + add_if_dist) * sin(radians(angles[2]))
        else:
            l = abc_a[ind] + add_if_dist
        abc_a[ind] += abc_b[ind] + 2 * add_if_dist + vacuum
        new_lat = Lattice.from_parameters(*abc_a, *angles)
        a_fcoords = new_lat.get_fractional_coords(self.grain_a.cart_coords)

        grain_a = Grain(new_lat, self.grain_a.species, a_fcoords)
        l_vector = [0, 0]
        l_vector.insert(ind, l)
        b_fcoords = new_lat.get_fractional_coords(
            self.grain_b.cart_coords + l_vector)
        grain_b = Grain(new_lat, self.grain_b.species, b_fcoords)

        gb = Grain.from_sites(grain_a[:] + grain_b[:])
        gb = gb.get_sorted_structure()
        if to_primitive:
            gb = gb.get_primitive_structure()
        return gb

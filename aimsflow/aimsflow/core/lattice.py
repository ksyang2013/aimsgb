import math
import numpy as np
from numpy.linalg import inv
from numpy import pi, dot, radians, transpose

from aimsflow.util.num_utils import abs_cap
from aimsflow.util.coord_utils import pbc_shortest_vectors


class Lattice(object):

    def __init__(self, matrix):
        m = np.array(matrix, dtype=np.float64).reshape((3, 3))
        lengths = np.sqrt(np.sum(m ** 2, axis=1))
        cos_angles = np.zeros(3)
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            cos_angles[i] = abs_cap(dot(m[j], m[k]) / (lengths[j] * lengths[k]))

        self._angles = np.arccos(cos_angles) * 180 / pi
        self._lengths = lengths
        self._matrix = m
        self._inv_matrix = None
        self._metric_tensor = None
        self._diags = None
        self._lll_matrix_mappings = {}
        self._lll_inverse = None
        self.is_orthogonal = all([abs(a - 90) < 1e-5 for a in self._angles])

    def __repr__(self):
        outs = ["Lattice",
                "    abc : %s" % " ".join("%.6f" % i for i in self._lengths),
                " angles : %s" % " ".join("%.1f" % i for i in self._angles),
                " volume : %.6f" % self.volume,
                "      A : %s" % " ".join("%.6f" % i for i in self._matrix[0]),
                "      B : %s" % " ".join("%.6f" % i for i in self._matrix[1]),
                "      C : %s" % " ".join("%.6f" % i for i in self._matrix[2])]
        return "\n".join(outs)

    def __str__(self):
        return "\n".join([" ".join("%.6f" % i for i in row)
                          for row in self._matrix])

    @property
    def matrix(self):
        return np.copy(self._matrix)

    @property
    def abc(self):
        return tuple(self._lengths)

    @property
    def inv_matrix(self):
        if self._inv_matrix is None:
            self._inv_matrix = inv(self._matrix)
        return self._inv_matrix

    @property
    def angles(self):
        return tuple(self._angles)

    @property
    def lengths_and_angles(self):
        return tuple(self._lengths), tuple(self._angles)

    @property
    def volume(self):
        m = self._matrix
        return abs(np.dot(np.cross(m[0], m[1]), m[2]))

    @property
    def reciprocal_lattice(self):
        try:
            return self._reciprocal_lattice
        except AttributeError:
            v = np.linalg.inv(self._matrix).T
            self._reciprocal_lattice = Lattice(v * 2 * np.pi)
            return self._reciprocal_lattice

    @property
    def reciprocal_lattice_crystallographic(self):
        return Lattice(self.reciprocal_lattice.matrix / (2 * np.pi))

    @property
    def lll_matrix(self):
        if 0.75 not in self._lll_matrix_mappings:
            self._lll_matrix_mappings[0.75] = self._calculate_lll()
        return self._lll_matrix_mappings[0.75][0]

    @property
    def lll_mapping(self):
        if 0.75 not in self._lll_matrix_mappings:
            self._lll_matrix_mappings[0.75] = self._calculate_lll()
        return self._lll_matrix_mappings[0.75][1]

    @property
    def lll_inverse(self):
        if self._lll_inverse is not None:
            return self._lll_inverse
        else:
            self._lll_inverse = np.linalg.inv(self.lll_mapping)
            return self._lll_inverse

    @staticmethod
    def from_parameters(a, b, c, alpha, beta, gamma):
        alpha_r = radians(alpha)
        beta_r = radians(beta)
        gamma_r = radians(gamma)
        val = (np.cos(alpha_r) * np.cos(beta_r) - np.cos(gamma_r))\
              / (np.sin(alpha_r) * np.sin(beta_r))
        val = abs_cap(val)
        gamma_star = np.arccos(val)
        vector_a = [a * np.sin(beta_r), 0.0, a * np.cos(beta_r)]
        vector_b = [-b * np.sin(alpha_r) * np.cos(gamma_star),
                    b * np.sin(alpha_r) * np.sin(gamma_star),
                    b * np.cos(alpha_r)]
        vector_c = [0.0, 0.0, float(c)]
        return Lattice([vector_a, vector_b, vector_c])

    @staticmethod
    def from_lengths_angles(abc, ang):
        return Lattice.from_parameters(abc[0], abc[1], abc[2],
                                       ang[0], ang[1], ang[2])

    @staticmethod
    def tetragonal(a, c):
        return Lattice.from_parameters(a, a, c, 90, 90, 90)

    @staticmethod
    def orthorhombic(a, b, c):
        return Lattice.from_parameters(a, b, c, 90, 90, 90)


    def is_hex(self, angle_tol=5, d_tol=0.01):
        lengths, angles = self.lengths_and_angles
        orth = [i for i, v in enumerate(angles) if abs(v - 90) < angle_tol]
        hex = [i for i, v in enumerate(angles)
               if abs(v - 60) < angle_tol or abs(v - 120) < angle_tol]
        return (len(orth) == 2 and len(hex) == 1
                and abs(lengths[orth[0]] - lengths[orth[1]]) < d_tol)

    def _calculate_lll(self, delta=0.75):
        """
        Performs a Lenstra-Lenstra-Lovasz lattice basis reduction to obtain a
        c-reduced basis. This method returns a basis which is as "good" as
        possible, with "good" defined by orthongonality of the lattice vectors.

        This basis is used for all the periodic boundary condition calculations.

        Args:
            delta (float): Reduction parameter. Default of 0.75 is usually
                fine.

        Returns:
            Reduced lattice matrix, mapping to get to that lattice.
        """
        # Transpose the lattice matrix first so that basis vectors are columns.
        # Makes life easier.
        a = self.matrix.copy().T

        b = np.zeros((3, 3))  # Vectors after the Gram-Schmidt process
        u = np.zeros((3, 3))  # Gram-Schmidt coeffieicnts
        m = np.zeros(3)  # These are the norm squared of each vec.

        b[:, 0] = a[:, 0]
        m[0] = dot(b[:, 0], b[:, 0])
        for i in range(1, 3):
            u[i, 0:i] = dot(a[:, i].T, b[:, 0:i]) / m[0:i]
            b[:, i] = a[:, i] - dot(b[:, 0:i], u[i, 0:i].T)
            m[i] = dot(b[:, i], b[:, i])

        k = 2

        mapping = np.identity(3, dtype=np.double)
        while k <= 3:
            # Size reduction.
            for i in range(k - 1, 0, -1):
                q = round(u[k - 1, i - 1])
                if q != 0:
                    # Reduce the k-th basis vector.
                    a[:, k - 1] = a[:, k - 1] - q * a[:, i - 1]
                    mapping[:, k - 1] = mapping[:, k - 1] - q * \
                        mapping[:, i - 1]
                    uu = list(u[i - 1, 0:(i - 1)])
                    uu.append(1)
                    # Update the GS coefficients.
                    u[k - 1, 0:i] = u[k - 1, 0:i] - q * np.array(uu)

            # Check the Lovasz condition.
            if dot(b[:, k - 1], b[:, k - 1]) >=\
                    (delta - abs(u[k - 1, k - 2]) ** 2) *\
                    dot(b[:, (k - 2)], b[:, (k - 2)]):
                # Increment k if the Lovasz condition holds.
                k += 1
            else:
                # If the Lovasz condition fails,
                # swap the k-th and (k-1)-th basis vector
                v = a[:, k - 1].copy()
                a[:, k - 1] = a[:, k - 2].copy()
                a[:, k - 2] = v

                v_m = mapping[:, k - 1].copy()
                mapping[:, k - 1] = mapping[:, k - 2].copy()
                mapping[:, k - 2] = v_m

                # Update the Gram-Schmidt coefficients
                for s in range(k - 1, k + 1):
                    u[s - 1, 0:(s - 1)] = dot(a[:, s - 1].T,
                                              b[:, 0:(s - 1)]) / m[0:(s - 1)]
                    b[:, s - 1] = a[:, s - 1] - dot(b[:, 0:(s - 1)],
                                                    u[s - 1, 0:(s - 1)].T)
                    m[s - 1] = dot(b[:, s - 1], b[:, s - 1])

                if k > 2:
                    k -= 1
                else:
                    # We have to do p/q, so do lstsq(q.T, p.T).T instead.
                    p = dot(a[:, k:3].T, b[:, (k - 2):k])
                    q = np.diag(m[(k - 2):k])
                    result = np.linalg.lstsq(q.T, p.T)[0].T
                    u[k:3, (k - 2):k] = result

        return a.T, mapping.T

    def get_niggli_reduced_lattice(self, tol=1e-5):
        """
        Get the Niggli reduced lattice using the numerically stable algo
        proposed by R. W. Grosse-Kunstleve, N. K. Sauter, & P. D. Adams,
        Acta Crystallographica Section A Foundations of Crystallography, 2003,
        60(1), 1-6. doi:10.1107/S010876730302186X

        Args:
            tol (float): The numerical tolerance. The default of 1e-5 should
                result in stable behavior for most cases.

        Returns:
            Niggli-reduced lattice.
        """
        # lll reduction is more stable for skewed cells
        matrix = self.lll_matrix
        a = matrix[0]
        b = matrix[1]
        c = matrix[2]
        e = tol * self.volume ** (1 / 3)

        # Define metric tensor
        G = [[dot(a, a), dot(a, b), dot(a, c)],
             [dot(a, b), dot(b, b), dot(b, c)],
             [dot(a, c), dot(b, c), dot(c, c)]]
        G = np.array(G)

        # This sets an upper limit on the number of iterations.
        for count in range(100):
            # The steps are labelled as Ax as per the labelling scheme in the
            # paper.
            (A, B, C, E, N, Y) = (G[0, 0], G[1, 1], G[2, 2],
                                  2 * G[1, 2], 2 * G[0, 2], 2 * G[0, 1])

            if A > B + e or (abs(A - B) < e and abs(E) > abs(N) + e):
                # A1
                M = [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
                G = dot(transpose(M), dot(G, M))
            if (B > C + e) or (abs(B - C) < e and abs(N) > abs(Y) + e):
                # A2
                M = [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]
                G = dot(transpose(M), dot(G, M))
                continue

            l = 0 if abs(E) < e else E / abs(E)
            m = 0 if abs(N) < e else N / abs(N)
            n = 0 if abs(Y) < e else Y / abs(Y)
            if l * m * n == 1:
                # A3
                i = -1 if l == -1 else 1
                j = -1 if m == -1 else 1
                k = -1 if n == -1 else 1
                M = [[i, 0, 0], [0, j, 0], [0, 0, k]]
                G = dot(transpose(M), dot(G, M))
            elif l * m * n == 0 or l * m * n == -1:
                # A4
                i = -1 if l == 1 else 1
                j = -1 if m == 1 else 1
                k = -1 if n == 1 else 1

                if i * j * k == -1:
                    if n == 0:
                        k = -1
                    elif m == 0:
                        j = -1
                    elif l == 0:
                        i = -1
                M = [[i, 0, 0], [0, j, 0], [0, 0, k]]
                G = dot(transpose(M), dot(G, M))

            (A, B, C, E, N, Y) = (G[0, 0], G[1, 1], G[2, 2],
                                  2 * G[1, 2], 2 * G[0, 2], 2 * G[0, 1])

            # A5
            if abs(E) > B + e or (abs(E - B) < e and 2 * N < Y - e) or\
                    (abs(E + B) < e and Y < -e):
                M = [[1, 0, 0], [0, 1, -E / abs(E)], [0, 0, 1]]
                G = dot(transpose(M), dot(G, M))
                continue

            # A6
            if abs(N) > A + e or (abs(A - N) < e and 2 * E < Y - e) or\
                    (abs(A + N) < e and Y < -e):
                M = [[1, 0, -N / abs(N)], [0, 1, 0], [0, 0, 1]]
                G = dot(transpose(M), dot(G, M))
                continue

            # A7
            if abs(Y) > A + e or (abs(A - Y) < e and 2 * E < N - e) or\
                    (abs(A + Y) < e and N < -e):
                M = [[1, -Y / abs(Y), 0], [0, 1, 0], [0, 0, 1]]
                G = dot(transpose(M), dot(G, M))
                continue

            # A8
            if E + N + Y + A + B < -e or\
                    (abs(E + N + Y + A + B) < e < Y + (A + N) * 2):
                M = [[1, 0, 1], [0, 1, 1], [0, 0, 1]]
                G = dot(transpose(M), dot(G, M))
                continue

            break

        A = G[0, 0]
        B = G[1, 1]
        C = G[2, 2]
        E = 2 * G[1, 2]
        N = 2 * G[0, 2]
        Y = 2 * G[0, 1]
        a = math.sqrt(A)
        b = math.sqrt(B)
        c = math.sqrt(C)
        alpha = math.acos(E / 2 / b / c) / math.pi * 180
        beta = math.acos(N / 2 / a / c) / math.pi * 180
        gamma = math.acos(Y / 2 / a / b) / math.pi * 180

        latt = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

        mapped = self.find_mapping(latt, e, skip_rotation_matrix=True)
        if mapped is not None:
            if np.linalg.det(mapped[0].matrix) > 0:
                return mapped[0]
            else:
                return Lattice(-mapped[0].matrix)

        raise ValueError("can't find niggli")

    def get_lll_reduced_lattice(self, delta=0.75):
        if delta not in self._lll_matrix_mappings:
            self._lll_matrix_mappings[delta] = self._calculate_lll()
        return Lattice(self._lll_matrix_mappings[delta][0])

    def find_all_mappings(self, other_lattice, ltol=1e-5, atol=1,
                          skip_rotation_matrix=False):
        """
        Finds all mappings between current lattice and another lattice.

        Args:
            other_lattice (Lattice): Another lattice that is equivalent to
                this one.
            ltol (float): Tolerance for matching lengths. Defaults to 1e-5.
            atol (float): Tolerance for matching angles. Defaults to 1.
            skip_rotation_matrix (bool): Whether to skip calculation of the
                rotation matrix

        Yields:
            (aligned_lattice, rotation_matrix, scale_matrix) if a mapping is
            found. aligned_lattice is a rotated version of other_lattice that
            has the same lattice parameters, but which is aligned in the
            coordinate system of this lattice so that translational points
            match up in 3D. rotation_matrix is the rotation that has to be
            applied to other_lattice to obtain aligned_lattice, i.e.,
            aligned_matrix = np.inner(other_lattice, rotation_matrix) and
            op = SymmOp.from_rotation_and_translation(rotation_matrix)
            aligned_matrix = op.operate_multi(latt.matrix)
            Finally, scale_matrix is the integer matrix that expresses
            aligned_matrix as a linear combination of this
            lattice, i.e., aligned_matrix = np.dot(scale_matrix, self.matrix)

            None is returned if no matches are found.
        """
        (lengths, angles) = other_lattice.lengths_and_angles
        (alpha, beta, gamma) = angles

        frac, dist, _ = self.get_points_in_sphere([[0, 0, 0]], [0, 0, 0],
                                                  max(lengths) * (1 + ltol),
                                                  zip_results=False)
        cart = self.get_cart_coords(frac)
        # this can't be broadcast because they're different lengths
        inds = [np.logical_and(dist / l < 1 + ltol,
                               dist / l > 1 / (1 + ltol)) for l in lengths]
        c_a, c_b, c_c = (cart[i] for i in inds)
        f_a, f_b, f_c = (frac[i] for i in inds)
        l_a, l_b, l_c = (np.sum(c ** 2, axis=-1) ** 0.5 for c in (c_a, c_b, c_c))

        def get_angles(v1, v2, l1, l2):
            x = np.inner(v1, v2) / l1[:, None] / l2
            x[x > 1] = 1
            x[x < -1] = -1
            angles = np.arccos(x) * 180. / pi
            return angles

        alphab = np.abs(get_angles(c_b, c_c, l_b, l_c) - alpha) < atol
        betab = np.abs(get_angles(c_a, c_c, l_a, l_c) - beta) < atol
        gammab = np.abs(get_angles(c_a, c_b, l_a, l_b) - gamma) < atol

        for i, all_j in enumerate(gammab):
            inds = np.logical_and(all_j[:, None],
                                  np.logical_and(alphab,
                                                 betab[i][None, :]))
            for j, k in np.argwhere(inds):
                scale_m = np.array((f_a[i], f_b[j], f_c[k]), dtype=np.int)
                if abs(np.linalg.det(scale_m)) < 1e-8:
                    continue

                aligned_m = np.array((c_a[i], c_b[j], c_c[k]))

                if skip_rotation_matrix:
                    rotation_m = None
                else:
                    rotation_m = np.linalg.solve(aligned_m,
                                                 other_lattice.matrix)

                yield Lattice(aligned_m), rotation_m, scale_m

    def find_mapping(self, other_lattice, ltol=1e-5, atol=1,
                     skip_rotation_matrix=False):
        """
        Finds a mapping between current lattice and another lattice. There
        are an infinite number of choices of basis vectors for two entirely
        equivalent lattices. This method returns a mapping that maps
        other_lattice to this lattice.

        Args:
            other_lattice (Lattice): Another lattice that is equivalent to
                this one.
            ltol (float): Tolerance for matching lengths. Defaults to 1e-5.
            atol (float): Tolerance for matching angles. Defaults to 1.

        Returns:
            (aligned_lattice, rotation_matrix, scale_matrix) if a mapping is
            found. aligned_lattice is a rotated version of other_lattice that
            has the same lattice parameters, but which is aligned in the
            coordinate system of this lattice so that translational points
            match up in 3D. rotation_matrix is the rotation that has to be
            applied to other_lattice to obtain aligned_lattice, i.e.,
            aligned_matrix = np.inner(other_lattice, rotation_matrix) and
            op = SymmOp.from_rotation_and_translation(rotation_matrix)
            aligned_matrix = op.operate_multi(latt.matrix)
            Finally, scale_matrix is the integer matrix that expresses
            aligned_matrix as a linear combination of this
            lattice, i.e., aligned_matrix = np.dot(scale_matrix, self.matrix)

            None is returned if no matches are found.
        """
        for x in self.find_all_mappings(
                other_lattice, ltol, atol,
                skip_rotation_matrix=skip_rotation_matrix):
            return x

    def get_points_in_sphere(self, frac_points, center, r, zip_results=True):
        """
        Find all points within a sphere from the point taking into account
        periodic boundary conditions. This includes sites in other periodic
        images.

        Algorithm:

        1. place sphere of radius r in crystal and determine minimum supercell
           (parallelpiped) which would contain a sphere of radius r. for this
           we need the projection of a_1 on a unit vector perpendicular
           to a_2 & a_3 (i.e. the unit vector in the direction b_1) to
           determine how many a_1"s it will take to contain the sphere.

           Nxmax = r * length_of_b_1 / (2 Pi)

        2. keep points falling within r.

        Args:
            frac_points: All points in the lattice in fractional coordinates.
            center: Cartesian coordinates of center of sphere.
            r: radius of sphere.
            zip_results (bool): Whether to zip the results together to group by
                 point, or return the raw fcoord, dist, index arrays

        Returns:
            if zip_results:
                [(fcoord, dist, index) ...] since most of the time, subsequent
                processing requires the distance.
            else:
                fcoords, dists, inds
        """
        recp_len = np.array(self.reciprocal_lattice.abc) / (2 * pi)
        nmax = float(r) * recp_len + 0.01

        pcoords = self.get_frac_coords(center)
        center = np.array(center)

        n = len(frac_points)
        fcoords = np.array(frac_points) % 1
        indices = np.arange(n)

        mins = np.floor(pcoords - nmax)
        maxes = np.ceil(pcoords + nmax)
        arange = np.arange(start=mins[0], stop=maxes[0])
        brange = np.arange(start=mins[1], stop=maxes[1])
        crange = np.arange(start=mins[2], stop=maxes[2])
        arange = arange[:, None] * np.array([1, 0, 0])[None, :]
        brange = brange[:, None] * np.array([0, 1, 0])[None, :]
        crange = crange[:, None] * np.array([0, 0, 1])[None, :]
        images = arange[:, None, None] + brange[None, :, None] + \
                 crange[None, None, :]

        shifted_coords = fcoords[:, None, None, None, :] + \
                         images[None, :, :, :, :]

        cart_coords = self.get_cart_coords(fcoords)
        cart_images = self.get_cart_coords(images)
        coords = cart_coords[:, None, None, None, :] + \
                 cart_images[None, :, :, :, :]
        coords -= center[None, None, None, None, :]
        coords **= 2
        d_2 = np.sum(coords, axis=4)

        within_r = np.where(d_2 <= r ** 2)
        if zip_results:
            return list(zip(shifted_coords[within_r], np.sqrt(d_2[within_r]),
                            indices[within_r[0]]))
        else:
            return shifted_coords[within_r], np.sqrt(d_2[within_r]), \
                   indices[within_r[0]]

    def get_frac_coords(self, cart_coords):
        return dot(cart_coords, self.inv_matrix)

    def get_cart_coords(self, fract_coords):
        return dot(fract_coords, self._matrix)

    def get_distance_and_image(self, frac_coords1, frac_coords2, jimage=None):
        if jimage is None:
            v, d2 = pbc_shortest_vectors(self, frac_coords1, frac_coords2,
                                         return_d2=True)
            fc = self.get_frac_coords(v[0][0]) + frac_coords1 - \
                 frac_coords2
            fc = np.array(np.round(fc), dtype=np.int)
            return np.sqrt(d2[0, 0]), fc

        mapped_vec = self.get_cart_coords(jimage + frac_coords2
                                          - frac_coords1)
        return np.linalg.norm(mapped_vec), jimage

    def get_lll_frac_coords(self, frac_coords):
        """
        Given fractional coordinates in the lattice basis, returns corresponding
        fractional coordinates in the lll basis.
        """
        return np.dot(frac_coords, self.lll_inverse)
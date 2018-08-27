import spglib
import numpy as np
from fractions import Fraction
from itertools import groupby, permutations
from math import cos, sin, pi, sqrt

from aimsflow import Lattice, PeriodicSite, Structure
from aimsflow.core.operations import SymmOp


class SpacegroupAnalyzer(object):
    def __init__(self, structure, symprec=1e-3, angle_tolerance=5):
        self._symprec = symprec
        self._angle_tol = angle_tolerance
        self._structure = structure
        latt = structure.lattice.matrix
        positions = structure.frac_coords
        unique_species = []
        zs = []

        for species, g in groupby(structure, key=lambda s: s.species_and_occu):
            if species in unique_species:
                ind = unique_species.index(species)
                zs.extend([ind + 1] * len(tuple(g)))
            else:
                unique_species.append(species)
                zs.extend([len(unique_species)] * len(tuple(g)))

        self._unique_species = unique_species
        self._numbers = zs
        self._cell = latt, positions, zs

        self._space_group_data = spglib.get_symmetry_dataset(
            self._cell, symprec=self._symprec, angle_tolerance=angle_tolerance)


    def _get_symmetry(self):
        d = spglib.get_symmetry(self._cell, symprec=self._symprec,
                                angle_tolerance=self._angle_tol)
        # Sometimes spglib returns small translation vectors, e.g.
        # [1e-4, 2e-4, 1e-4]
        # (these are in fractional coordinates, so should be small denominator
        # fractions)
        trans = []
        for t in d["translations"]:
            trans.append([float(Fraction.from_float(c).limit_denominator(1000))
                          for c in t])
        trans = np.array(trans)

        # fractional translations of 1 are more simply 0
        trans[np.abs(trans) == 1] = 0
        return d["rotations"], trans

    def get_lattice_type(self):
        n = self._space_group_data["number"]
        system = self.get_crystal_system()
        if n in [146, 148, 155, 160, 161, 166, 167]:
            return "rhombohedral"
        elif system == "trigonal":
            return "hexagonal"
        else:
            return system


    def get_crystal_system(self):
        n = self._space_group_data["number"]

        f = lambda i, j: i <= n <= j
        cs = {"triclinic": (1, 2), "monoclinic": (3, 15),
              "orthorhombic": (16, 74), "tetragonal": (75, 142),
              "trigonal": (143, 167), "hexagonal": (168, 194),
              "cubic": (195, 230)}

        crystal_sytem = None

        for k, v in cs.items():
            if f(*v):
                crystal_sytem = k
                break
        return crystal_sytem

    def get_space_group_operations(self):
        """
        Get the SpacegroupOperations for the Structure.

        Returns:
            SpacgroupOperations object.
        """
        return SpacegroupOperations(self.get_space_group_symbol(),
                                    self.get_space_group_number(),
                                    self.get_symmetry_operations())

    def get_space_group_symbol(self):
        return self._space_group_data["international"]

    def get_symmetry_operations(self, cartesian=False):
        rotation, translation = self._get_symmetry()
        symmops = []
        mat = self._structure.lattice.matrix.T
        invmat = np.linalg.inv(mat)
        for rot, trans in zip(rotation, translation):
            if cartesian:
                rot = np.dot(mat, np.dot(rot, invmat))
                trans = np.dot(trans, self._structure.lattice.matrix)
            op = SymmOp.from_rotation_and_translation(rot, trans)
            symmops.append(op)
        return symmops

    def get_space_group_number(self):
        return int(self._space_group_data["number"])

    def get_refined_structure(self):
        lattice, scaled_positions, numbers \
            = spglib.refine_cell(self._cell, self._symprec, self._angle_tol)

        species = [self._unique_species[i - 1] for i in numbers]
        s = Structure(lattice, species, scaled_positions)
        return s.get_sorted_structure()

    def get_primitive_standard_structure(self, international_monoclinic=True):
        conv = self.get_conventional_standard_structure(
            international_monoclinic=international_monoclinic)
        lattice = self.get_lattice_type()

        if "P" in self.get_space_group_symbol() or lattice == "hexagonal":
            return conv

        if lattice == "rhombohedral":
            # check if the conventional representation is hexagonal or
            # rhombohedral
            lengths, angles = conv.lattice.lengths_and_angles
            if abs(lengths[0]-lengths[2]) < 0.0001:
                transf = np.eye
            else:
                transf = np.array([[-1, 1, 1], [2, 1, 1], [-1, -2, 1]],
                                  dtype=np.float) / 3

        elif "I" in self.get_space_group_symbol():
            transf = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
                              dtype=np.float) / 2
        elif "F" in self.get_space_group_symbol():
            transf = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]],
                              dtype=np.float) / 2
        elif "C" in self.get_space_group_symbol():
            if self.get_crystal_system() == "monoclinic":
                transf = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]],
                                  dtype=np.float) / 2
            else:
                transf = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]],
                                  dtype=np.float) / 2
        else:
            transf = np.eye(3)

        new_sites = []
        latt = Lattice(np.dot(transf, conv.lattice.matrix))
        for s in conv:
            new_s = PeriodicSite(
                s.specie, s.coords, latt,
                to_unit_cell=True, coords_cartesian=True,
                properties=s.properties)
            if not any(map(new_s.is_periodic_image, new_sites)):
                new_sites.append(new_s)

        if lattice == "rhombohedral":
            prim = Structure.from_sites(new_sites)
            lengths, angles = prim.lattice.lengths_and_angles
            a = lengths[0]
            alpha = pi * angles[0] / 180
            new_matrix = [
                [a * cos(alpha / 2), -a * sin(alpha / 2), 0],
                [a * cos(alpha / 2), a * sin(alpha / 2), 0],
                [a * cos(alpha) / cos(alpha / 2), 0,
                 a * sqrt(1 - (cos(alpha) ** 2 / (cos(alpha / 2) ** 2)))]]
            new_sites = []
            latt = Lattice(new_matrix)
            for s in prim:
                new_s = PeriodicSite(
                    s.specie, s.frac_coords, latt,
                    to_unit_cell=True, properties=s.site_properties)
                if not any(map(new_s.is_periodic_image, new_sites)):
                    new_sites.append(new_s)
            return Structure.from_sites(new_sites)

        return Structure.from_sites(new_sites)

    def get_conventional_standard_structure(
            self, international_monoclinic=True):
        tol = 1e-5
        struct = self.get_refined_structure()
        latt = struct.lattice
        latt_type = self.get_lattice_type()
        sorted_lengths = sorted(latt.abc)
        sorted_dic = sorted([{'vec': latt.matrix[i],
                              'length': latt.abc[i],
                              'orig_index': i} for i in [0, 1, 2]],
                            key=lambda k: k['length'])

        if latt_type in ("orthorhombic", "cubic"):
            # you want to keep the c axis where it is
            # to keep the C- settings
            transf = np.zeros(shape=(3, 3))
            if self.get_space_group_symbol().startswith("C"):
                transf[2] = [0, 0, 1]
                a, b = sorted(latt.abc[:2])
                sorted_dic = sorted([{'vec': latt.matrix[i],
                                      'length': latt.abc[i],
                                      'orig_index': i} for i in [0, 1]],
                                    key=lambda k: k['length'])
                for i in range(2):
                    transf[i][sorted_dic[i]['orig_index']] = 1
                c = latt.abc[2]
            else:
                for i in range(len(sorted_dic)):
                    transf[i][sorted_dic[i]['orig_index']] = 1
                a, b, c = sorted_lengths
            latt = Lattice.orthorhombic(a, b, c)

        elif latt_type == "tetragonal":
            # find the "a" vectors
            # it is basically the vector repeated two times
            transf = np.zeros(shape=(3, 3))
            a, b, c = sorted_lengths
            for d in range(len(sorted_dic)):
                transf[d][sorted_dic[d]['orig_index']] = 1

            if abs(b - c) < tol:
                a, c = c, a
                transf = np.dot([[0, 0, 1], [0, 1, 0], [1, 0, 0]], transf)
            latt = Lattice.tetragonal(a, c)
        elif latt_type in ("hexagonal", "rhombohedral"):
            # for the conventional cell representation,
            # we allways show the rhombohedral lattices as hexagonal

            # check first if we have the refined structure shows a rhombohedral
            # cell
            # if so, make a supercell
            a, b, c = latt.abc
            if np.all(np.abs([a - b, c - b, a - c]) < 0.001):
                struct.make_supercell(((1, -1, 0), (0, 1, -1), (1, 1, 1)))
                a, b, c = sorted(struct.lattice.abc)

            if abs(b - c) < 0.001:
                a, c = c, a
            new_matrix = [[a / 2, -a * sqrt(3) / 2, 0],
                          [a / 2, a * sqrt(3) / 2, 0],
                          [0, 0, c]]
            latt = Lattice(new_matrix)
            transf = np.eye(3, 3)

        elif latt_type == "monoclinic":
            # You want to keep the c axis where it is to keep the C- settings

            if self.get_space_group_operations().int_symbol.startswith("C"):
                transf = np.zeros(shape=(3, 3))
                transf[2] = [0, 0, 1]
                sorted_dic = sorted([{'vec': latt.matrix[i],
                                      'length': latt.abc[i],
                                      'orig_index': i} for i in [0, 1]],
                                    key=lambda k: k['length'])
                a = sorted_dic[0]['length']
                b = sorted_dic[1]['length']
                c = latt.abc[2]
                new_matrix = None
                for t in permutations(list(range(2)), 2):
                    m = latt.matrix
                    landang = Lattice(
                        [m[t[0]], m[t[1]], m[2]]).lengths_and_angles
                    if landang[1][0] > 90:
                        # if the angle is > 90 we invert a and b to get
                        # an angle < 90
                        landang = Lattice(
                            [-m[t[0]], -m[t[1]], m[2]]).lengths_and_angles
                        transf = np.zeros(shape=(3, 3))
                        transf[0][t[0]] = -1
                        transf[1][t[1]] = -1
                        transf[2][2] = 1
                        a, b, c = landang[0]
                        alpha = pi * landang[1][0] / 180
                        new_matrix = [[a, 0, 0],
                                      [0, b, 0],
                                      [0, c * cos(alpha), c * sin(alpha)]]
                        continue

                    elif landang[1][0] < 90:
                        transf = np.zeros(shape=(3, 3))
                        transf[0][t[0]] = 1
                        transf[1][t[1]] = 1
                        transf[2][2] = 1
                        a, b, c = landang[0]
                        alpha = pi * landang[1][0] / 180
                        new_matrix = [[a, 0, 0],
                                      [0, b, 0],
                                      [0, c * cos(alpha), c * sin(alpha)]]

                if new_matrix is None:
                    # this if is to treat the case
                    # where alpha==90 (but we still have a monoclinic sg
                    new_matrix = [[a, 0, 0],
                                  [0, b, 0],
                                  [0, 0, c]]
                    transf = np.zeros(shape=(3, 3))
                    for c in range(len(sorted_dic)):
                        transf[c][sorted_dic[c]['orig_index']] = 1
            #if not C-setting
            else:
                # try all permutations of the axis
                # keep the ones with the non-90 angle=alpha
                # and b<c
                new_matrix = None
                for t in permutations(list(range(3)), 3):
                    m = latt.matrix
                    landang = Lattice(
                        [m[t[0]], m[t[1]], m[t[2]]]).lengths_and_angles
                    if landang[1][0] > 90 and landang[0][1] < landang[0][2]:
                        landang = Lattice(
                            [-m[t[0]], -m[t[1]], m[t[2]]]).lengths_and_angles
                        transf = np.zeros(shape=(3, 3))
                        transf[0][t[0]] = -1
                        transf[1][t[1]] = -1
                        transf[2][t[2]] = 1
                        a, b, c = landang[0]
                        alpha = pi * landang[1][0] / 180
                        new_matrix = [[a, 0, 0],
                                      [0, b, 0],
                                      [0, c * cos(alpha), c * sin(alpha)]]
                        continue
                    elif landang[1][0] < 90 and landang[0][1] < landang[0][2]:
                        transf = np.zeros(shape=(3, 3))
                        transf[0][t[0]] = 1
                        transf[1][t[1]] = 1
                        transf[2][t[2]] = 1
                        a, b, c = landang[0]
                        alpha = pi * landang[1][0] / 180
                        new_matrix = [[a, 0, 0],
                                      [0, b, 0],
                                      [0, c * cos(alpha), c * sin(alpha)]]
                if new_matrix is None:
                    # this if is to treat the case
                    # where alpha==90 (but we still have a monoclinic sg
                    new_matrix = [[sorted_lengths[0], 0, 0],
                                  [0, sorted_lengths[1], 0],
                                  [0, 0, sorted_lengths[2]]]
                    transf = np.zeros(shape=(3, 3))
                    for c in range(len(sorted_dic)):
                        transf[c][sorted_dic[c]['orig_index']] = 1

            if international_monoclinic:
                # The above code makes alpha the non-right angle.
                # The following will convert to proper international convention
                # that beta is the non-right angle.
                op = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
                transf = np.dot(op, transf)
                new_matrix = np.dot(op, new_matrix)
                beta = Lattice(new_matrix).beta
                if beta < 90:
                    op = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
                    transf = np.dot(op, transf)
                    new_matrix = np.dot(op, new_matrix)

            latt = Lattice(new_matrix)

        elif latt_type == "triclinic":
            #we use a LLL Minkowski-like reduction for the triclinic cells
            struct = struct.get_reduced_structure("LLL")

            a, b, c = latt.lengths_and_angles[0]
            alpha, beta, gamma = [pi * i / 180
                                  for i in latt.lengths_and_angles[1]]
            new_matrix = None
            test_matrix = [[a, 0, 0],
                          [b * cos(gamma), b * sin(gamma), 0.0],
                          [c * cos(beta),
                           c * (cos(alpha) - cos(beta) * cos(gamma)) /
                           sin(gamma),
                           c * sqrt(sin(gamma) ** 2 - cos(alpha) ** 2
                                         - cos(beta) ** 2
                                         + 2 * cos(alpha) * cos(beta)
                                         * cos(gamma)) / sin(gamma)]]

            def is_all_acute_or_obtuse(m):
                recp_angles = np.array(Lattice(m).reciprocal_lattice.angles)
                return np.all(recp_angles <= 90) or np.all(recp_angles > 90)

            if is_all_acute_or_obtuse(test_matrix):
                transf = np.eye(3)
                new_matrix = test_matrix

            test_matrix = [[-a, 0, 0],
                           [b * cos(gamma), b * sin(gamma), 0.0],
                           [-c * cos(beta),
                            -c * (cos(alpha) - cos(beta) * cos(gamma)) /
                            sin(gamma),
                            -c * sqrt(sin(gamma) ** 2 - cos(alpha) ** 2
                                           - cos(beta) ** 2
                                           + 2 * cos(alpha) * cos(beta)
                                           * cos(gamma)) / sin(gamma)]]

            if is_all_acute_or_obtuse(test_matrix):
                transf = [[-1, 0, 0],
                          [0, 1, 0],
                          [0, 0, -1]]
                new_matrix = test_matrix

            test_matrix = [[-a, 0, 0],
                           [-b * cos(gamma), -b * sin(gamma), 0.0],
                           [c * cos(beta),
                            c * (cos(alpha) - cos(beta) * cos(gamma)) /
                            sin(gamma),
                            c * sqrt(sin(gamma) ** 2 - cos(alpha) ** 2
                                          - cos(beta) ** 2
                                          + 2 * cos(alpha) * cos(beta)
                                          * cos(gamma)) / sin(gamma)]]

            if is_all_acute_or_obtuse(test_matrix):
                transf = [[-1, 0, 0],
                          [0, -1, 0],
                          [0, 0, 1]]
                new_matrix = test_matrix

            test_matrix = [[a, 0, 0],
                           [-b * cos(gamma), -b * sin(gamma), 0.0],
                           [-c * cos(beta),
                            -c * (cos(alpha) - cos(beta) * cos(gamma)) /
                            sin(gamma),
                            -c * sqrt(sin(gamma) ** 2 - cos(alpha) ** 2
                                           - cos(beta) ** 2
                                           + 2 * cos(alpha) * cos(beta)
                                           * cos(gamma)) / sin(gamma)]]
            if is_all_acute_or_obtuse(test_matrix):
                transf = [[1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1]]
                new_matrix = test_matrix

            latt = Lattice(new_matrix)

        new_coords = np.dot(transf, np.transpose(struct.frac_coords)).T
        new_struct = Structure(latt, struct.species_and_occu, new_coords,
                               site_properties=struct.site_properties,
                               to_unit_cell=True)
        return new_struct.get_sorted_structure()




class SpacegroupOperations(list):
    def __init__(self, int_symbol, int_number, symmops):
        self.int_symbol = int_symbol
        self.int_number = int_number
        super(SpacegroupOperations, self).__init__(symmops)
import warnings
import numpy as np
from functools import reduce
from itertools import groupby
from aimsgb.utils import reduce_vector
from pymatgen.core.structure import Structure, Lattice, PeriodicSite
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
from pymatgen.analysis.structure_matcher import StructureMatcher

__author__ = "Jianli CHENG and Kesong YANG"
__copyright__ = "Copyright 2018 University of California San Diego"
__maintainer__ = "Jianli CHENG"
__email__ = "jic198@ucsd.edu"
__status__ = "Production"
__date__ = "January 26, 2018"


class Grain(Structure):
    """
    We use the Structure class from pymatgen and add several new functions.
    """
    def __init__(self, lattice, species, coords, charge=None,
                 validate_proximity=False, to_unit_cell=False,
                 coords_are_cartesian=False, site_properties=None):

        super(Structure, self).__init__(lattice, species, coords, charge=charge,
                                        validate_proximity=validate_proximity,
                                        to_unit_cell=to_unit_cell,
                                        coords_are_cartesian=coords_are_cartesian,
                                        site_properties=site_properties)

        self._sites = list(self._sites)

    @staticmethod
    def get_b_from_a(grain_a, csl):
        grain_b = grain_a.copy()
        csl_t = csl.transpose()
        if sum(abs(csl_t[0]) - abs(csl_t[1])) > 0:
            axis = (1, 0, 0)
        else:
            axis = (0, 1, 0)
        anchor = grain_b.lattice.get_cartesian_coords(np.array([.0, .0, .0]))
        # print(axis, anchor)
        # exit()
        grain_b.rotate_sites(theta=np.radians(180), axis=axis, anchor=anchor)
        return grain_b

    @classmethod
    def from_mp_id(cls, mp_id):
        """
        Get a structure from Materials Project database.
        Args:
            mp_id (str): Materials Project ID.

        Returns:
            A structure object.
        """
        from mp_api.client import MPRester

        mpr = MPRester()
        s = mpr.get_structure_by_material_id(mp_id, conventional_unit_cell=True)
        return cls.from_dict(s.as_dict())

    def make_supercell(self, scaling_matrix):
        """
        Create a supercell. Very similar to pymatgen's Structure.make_supercell
        However, we need to make sure that all fractional coordinates that equal
        to 1 will become 0 and the lattice are redefined so that x_c = [0, 0, c]
    
        Args:
            scaling_matrix (3x3 matrix): The scaling matrix to make supercell.
        """
        s = self * scaling_matrix
        for i, site in enumerate(s):
            f_coords = np.mod(site.frac_coords, 1)
            # The following for loop is probably not necessary. But I will leave
            # it here for now.
            for j, v in enumerate(f_coords):
                if abs(v - 1) < 1e-6:
                    f_coords[j] = 0
            s[i] = PeriodicSite(site.specie, f_coords, site.lattice,
                                properties=site.properties)
        self._sites = s.sites
        self._lattice = s.lattice
        new_lat = Lattice.from_parameters(*s.lattice.parameters)
        self.lattice = new_lat

    def delete_bt_layer(self, bt, tol=0.25, axis=2):
        """
        Delete bottom or top layer of the structure.
        Args:
            bt (str): Specify whether it's a top or bottom layer delete. "b"
                means bottom layer and "t" means top layer.
            tol (float), Angstrom: Tolerance factor to determine whether two
                atoms are at the same plane.
                Default to 0.25
            axis (int): The direction of top and bottom layers. 0: x, 1: y, 2: z

        """
        if bt == "t":
            l1, l2 = (-1, -2)
        else:
            l1, l2 = (0, 1)

        l = self.lattice.abc[axis]
        layers = self.sort_sites_in_layers(tol=tol, axis=axis)
        l_dist = abs(layers[l1][0].coords[axis] - layers[l2][0].coords[axis])
        l_vector = [1, 1]
        l_vector.insert(axis, (l - l_dist) / l)
        new_lat = Lattice(self.lattice.matrix * np.array(l_vector)[:, None])

        layers.pop(l1)
        sites = reduce(lambda x, y: np.concatenate((x, y), axis=0), layers)
        new_sites = []
        l_dist = 0 if bt == "t" else l_dist
        l_vector = [0, 0]
        l_vector.insert(axis, l_dist)
        for i in sites:
            new_sites.append(PeriodicSite(i.specie, i.coords - l_vector,
                                          new_lat, coords_are_cartesian=True))
        self._sites = new_sites
        self._lattice = new_lat

    def sort_sites_in_layers(self, tol=0.25, axis=2):
        """
        Sort the sites in a structure layer by layer.

        Args:
            tol (float): tolerance factor when determine whether two atoms are
                are at the same plane. Angstrom
            axis (int): The direction of top and bottom layers. 0: x, 1: y, 2: z

        Returns:
            Lists with the sites in the same plane as one list.
        """
        new_atoms = sorted(self, key=lambda x: x.frac_coords[axis])
        layers = []
        for k, g in groupby(new_atoms, key=lambda x: x.frac_coords[axis]):
            layers.append(list(g))
        new_layers = []
        k = -1
        for i in range(len(layers)):
            if i > k:
                tmp = layers[i]
                for j in range(i + 1, len(layers)):
                    if self.lattice.abc[axis] * abs(
                                    layers[j][0].frac_coords[axis] -
                                    layers[i][0].frac_coords[axis]) < tol:
                        tmp.extend(layers[j])
                        k = j
                    else:
                        break
                new_layers.append(sorted(tmp))
        # check if the 1st layer and last layer are actually the same layer
        # use the fractional as cartesian doesn't work for unorthonormal
        if self.lattice.abc[axis] * abs(
                                new_layers[0][0].frac_coords[axis] + 1 -
                                new_layers[-1][0].frac_coords[axis]) < tol:
            tmp = new_layers[0] + new_layers[-1]
            new_layers = new_layers[1:-1]
            new_layers.append(sorted(tmp))
        return new_layers

    def set_orthogonal_grain(self):
        a, b, c = self.lattice.abc
        self.lattice = Lattice.orthorhombic(a, b, c)

    def build_grains(self, csl, gb_direction, uc_a=1, uc_b=1):
        """
        Build structures for grain A and B from CSL matrix, number of unit cell
        of grain A and number of unit cell of grain B. Each grain is essentially
        a supercell for the initial structure.

        Args:
            csl (3x3 matrix): CSL matrix (scaling matrix)
            gb_direction (int): The direction of GB. 0: x, 1: y, 2: z
            uc_a (int): Number of unit cell of grain A. Default to 1.
            uc_b (int): Number of unit cell of grain B. Default to 1.

        Returns:
            Grain objects for grain A and B
        """
        csl_t = csl.transpose()
        # rotate along a longer axis between a and b
        grain_a = self.copy()
        grain_a.make_supercell(csl_t)
        # grain_a.to(filename='POSCAR')
        # exit()

        if not grain_a.lattice.is_orthogonal:
            warnings.warn("The lattice system of the grain is not orthogonal. "
                          "aimsgb will find a supercell of the grain structure "
                          "that is orthogonalized. This may take a while. ")
            cst = CubicSupercellTransformation(force_90_degrees=True,
                                               min_length=min(grain_a.lattice.abc))
            _s = grain_a.copy()
            _s = cst.apply_transformation(_s)
            _matrix = [reduce_vector(i) for i in cst.transformation_matrix]
            _s = grain_a.copy()
            _s.make_supercell(_matrix)
            sm = StructureMatcher(attempt_supercell=True, primitive_cell=False)
            _matrix = sm.get_supercell_matrix(_s, self)
            matrix = [reduce_vector(i) for i in _matrix]
            grain_a = self.copy()
            grain_a.make_supercell(matrix)
            # grain_a.make_supercell(get_sc_fromstruct(grain_a, min_length=min(grain_a.lattice.abc),
            #                                          force_diagonal=True))
            # grain_a.make_supercell(get_sc_fromstruct(grain_a).transpose())
            # grain_a.set_orthogonal_grain()
            # grain_b = grain_b.set_orthogonal_grain()

        # grain_a.to(filename='POSCAR')
        # exit()
        temp_a = grain_a.copy()
        scale_vector = [1, 1]
        scale_vector.insert(gb_direction, uc_b)
        temp_a.make_supercell(scale_vector)
        grain_b = self.get_b_from_a(temp_a, csl)
        # make sure that all fractional coordinates that equal to 1 will become 0
        grain_b.make_supercell([1, 1, 1])

        scale_vector = [1, 1]
        scale_vector.insert(gb_direction, uc_a)
        grain_a.make_supercell(scale_vector)

        # grain_b.to(filename='POSCAR')
        # exit(0)
        return grain_a, grain_b

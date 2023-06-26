import re
import warnings
import numpy as np
from numpy import sin, radians
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
    
    def add_selective_dynamics(self, fix_list, tol=0.25, axis=2):
        """
        Add selective dynamics properties for sites sorted by layers
        Args:
            fix_list (list): A list of layer indices
            tol (float): Tolerance factor in Angstrom to determnine if sites are 
                in the same layer. Default to 0.25.

        Returns: A Structure object with selective dynamics properties

        """
        layers = self.sort_sites_in_layers(tol=tol, axis=axis)
        sd_sites = []
        for i, l in enumerate(layers):
            if i in fix_list:
                sd_sites.extend(zip([[False, False, False]] * len(l), [_i[1] for _i in l]))
            else:
                sd_sites.extend(zip([[True, True, True]] * len(l), [_i[1] for _i in l]))
        values = [i[0] for i in sorted(sd_sites, key=lambda x: x[1])]
        self.add_site_property("selective_dynamics", values)

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
            tol (float): Tolerance factor in Angstrom to determnine if sites are 
                in the same layer. Default to 0.25.
            axis (int): The direction of top and bottom layers. 0: x, 1: y, 2: z

        """
        if bt == "t":
            l1, l2 = (-1, -2)
        else:
            l1, l2 = (0, 1)

        l = self.lattice.abc[axis]
        layers = self.sort_sites_in_layers(tol=tol, axis=axis)
        l_dist = abs(layers[l1][0][0].coords[axis] - layers[l2][0][0].coords[axis])
        l_vector = [1, 1]
        l_vector.insert(axis, (l - l_dist) / l)
        new_lat = Lattice(self.lattice.matrix * np.array(l_vector)[:, None])

        layers.pop(l1)
        sites = reduce(lambda x, y: np.concatenate((x, y), axis=0), layers)
        new_sites = []
        l_dist = 0 if bt == "t" else l_dist
        l_vector = [0, 0]
        l_vector.insert(axis, l_dist)
        for site, _ in sites:
            new_sites.append(PeriodicSite(site.specie, site.coords - l_vector,
                                          new_lat, coords_are_cartesian=True))
        self._sites = new_sites
        self._lattice = new_lat

    def sort_sites_in_layers(self, tol=0.25, axis=2):
        """
        Sort the sites in a structure layer by layer.

        Args:
            tol (float): Tolerance factor in Angstrom to determnine if sites are 
                in the same layer. Default to 0.25.
            axis (int): The direction of top and bottom layers. 0: x, 1: y, 2: z

        Returns:
            Lists with a list of (site, index) in the same plane as one list.
        """
        sites_indices = sorted(zip(self.sites, range(len(self))), 
                               key=lambda x: x[0].frac_coords[axis])
        layers = []
        for k, g in groupby(sites_indices, key=lambda x: x[0].frac_coords[axis]):
            layers.append(list(g))
        new_layers = []
        k = -1
        for i in range(len(layers)):
            if i > k:
                tmp = layers[i]
                for j in range(i + 1, len(layers)):
                    if self.lattice.abc[axis] * abs(
                                    layers[j][0][0].frac_coords[axis] -
                                    layers[i][0][0].frac_coords[axis]) < tol:
                        tmp.extend(layers[j])
                        k = j
                    else:
                        break
                new_layers.append(sorted(tmp))
        # check if the 1st layer and last layer are actually the same layer
        # use the fractional as cartesian doesn't work for unorthonormal
        if self.lattice.abc[axis] * abs(
                                new_layers[0][0][0].frac_coords[axis] + 1 -
                                new_layers[-1][0][0].frac_coords[axis]) < tol:
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

    @classmethod
    def stack_grains(cls, grain_a, grain_b, vacuum=0.0, gap=0.0, direction=2,
                     delete_layer="0b0t0b0t", tol=0.25, to_primitive=True):
        """
        Build an interface structure by stacking two grains along a given direction.
        The grain_b a- and b-vectors will be forced to be the grain_a's
        a- and b-vectors.
        Args:
            grain_a (Grain): Substrate for the interface structure
            grain_b (Grain): Film for the interface structure
            vacuum (float): Vacuum space above the film in Angstroms. Default to 0.0
            gap (float): Gap between substrate and film in Angstroms. Default to 0.0
            direction (int): Stacking direction of the interface structure. 0: x, 1: y, 2: z.
            delete_layer (str): Delete top and bottom layers of the substrate and film.
                8 characters in total. The first 4 characters is for the substrate and
                the other 4 is for the film. "b" means bottom layer and "t" means
                top layer. Integer represents the number of layers to be deleted.
                Default to "0b0t0b0t", which means no deletion of layers. The
                direction of top and bottom layers is based on the given direction.
            tol (float): Tolerance factor in Angstrom to determnine if sites are 
                in the same layer. Default to 0.25.
            to_primitive (bool): Whether to get primitive structure of GB. Default to true.
        Returns:
             GB structure (Grain)
        """
        delete_layer = delete_layer.lower()
        delete = re.findall('(\d+)(\w)', delete_layer)
        if len(delete) != 4:
            raise ValueError(f"'{delete_layer}' is not supported. Please make sure the format "
                             "is 0b0t0b0t.")
        for i, v in enumerate(delete):
            for j in range(int(v[0])):
                if i <= 1:
                    grain_a.delete_bt_layer(v[1], tol, direction)
                else:
                    grain_b.delete_bt_layer(v[1], tol, direction)
        abc_a = list(grain_a.lattice.abc)
        abc_b, angles = np.reshape(grain_b.lattice.parameters, (2, 3))
        if direction == 1:
            l = (abc_a[direction] + gap) * sin(radians(angles[2]))
        else:
            l = abc_a[direction] + gap
        abc_a[direction] += abc_b[direction] + 2 * gap + vacuum
        new_lat = Lattice.from_parameters(*abc_a, *angles)
        a_fcoords = new_lat.get_fractional_coords(grain_a.cart_coords)

        grain_a = Grain(new_lat, grain_a.species, a_fcoords, site_properties=grain_a.site_properties)
        l_vector = [0, 0]
        l_vector.insert(direction, l)
        b_fcoords = new_lat.get_fractional_coords(
            grain_b.cart_coords + l_vector)
        grain_b = Grain(new_lat, grain_b.species, b_fcoords, site_properties=grain_b.site_properties)

        structure = Grain.from_sites(grain_a[:] + grain_b[:])
        structure = structure.get_sorted_structure()
        if to_primitive:
            structure = structure.get_primitive_structure()

        return cls.from_dict(structure.as_dict())

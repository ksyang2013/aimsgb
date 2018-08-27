import collections
import numpy as np

from aimsflow.core.composition import Composition
from aimsflow.core.periodic_table import get_el_sp
from aimsflow.util import pbc_diff


class Site(collections.Hashable):
    position_atol = 1e-5

    def __init__(self, atomic_symbol, coords, properties=None):
        if isinstance(atomic_symbol, Composition):
            self._species = atomic_symbol
            total_occu = atomic_symbol._atom_num
            if total_occu > 1 + Composition.amount_tolerance:
                raise ValueError("Species occupancies sum to more than 1!")
            self._is_ordered = total_occu == 1 and len(self._species._data) == 1
        else:
            try:
                self._species = Composition({get_el_sp(atomic_symbol): 1})
                self._is_ordered = True
            except TypeError:
                self._species = Composition(atomic_symbol)
                total_occu = self._species.num_atoms
                if total_occu > 1 + Composition.amount_tolerance:
                    raise ValueError("Species occupancies sum to more than 1!")
                self._is_ordered = total_occu == 1 and len(self._species._data) == 1

        self._coords = coords
        self._properties = properties if properties else {}

    def __hash__(self):
        return sum([el.Z for el in self._species.keys()])

    def __contains__(self, el):
        return el in self._species

    def __lt__(self, other):
        if self._species.average_electroneg < other._species.average_electroneg:
            return True
        if self._species.average_electroneg > other._species.average_electroneg:
            return False
        if self.species_string < other.species_string:
            return True
        if self.species_string > other.species_string:
            return False
        return False

    def __eq__(self, other):
        if other is None:
            return False
        return self._species == other._species and\
               np.allclose(self._coords, other._coords,
                           atol=Site.position_atol) and\
               self._properties == other._properties

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, el):
        return self._species[el]

    def distance(self, other):
        return np.linalg.norm(other.cart_coords - self._coords)

    @property
    def specie(self):
        if not self._is_ordered:
            raise AttributeError("Specie property only works for ordered sites!")
        return list(self._species.keys())[0]

    @property
    def species_and_occu(self):
        return self._species

    @property
    def species_string(self):
        if self._is_ordered:
            return list(self._species.keys())[0].__str__()

    @property
    def properties(self):
        return {k: v for k, v in self._properties.items()}

    @property
    def coords(self):
        return np.copy(self._coords)

    @property
    def is_ordered(self):
        return self._is_ordered


class PeriodicSite(Site):
    
    def __init__(self, atomic_symbol, coords, lattice, to_unit_cell=False,
                 coords_cartesian=False, properties=None):
        self._lattice = lattice
        if coords_cartesian:
            self._fcoords = self._lattice.get_frac_coords(coords)
            cart_coords = coords
        else:
            self._fcoords = coords
            cart_coords = self._lattice.get_cart_coords(coords)

        if to_unit_cell:
            self._fcoords = np.mod(self._fcoords, 1)
            for j, v in enumerate(self._fcoords):
                if abs(v - 1) < 1e-6:
                    self._fcoords[j] = 0
            cart_coords = self._lattice.get_cart_coords(self._fcoords)

        super(PeriodicSite, self).__init__(atomic_symbol, cart_coords, properties)

    def __repr__(self):
        return "PeriodicSite: {} ({:.4f}. {:.4f}, {:.4f}) [{:.4f}, {:.4f}, " \
               "{:.4f}]".format(self.species_string, self._coords[0],
                                self._coords[1], self._coords[2],
                                self._fcoords[0], self._fcoords[1],
                                self._fcoords[2])

    @property
    def lattice(self):
        return self._lattice

    @property
    def frac_coords(self):
        return np.copy(self._fcoords)

    @property
    def to_unit_cell(self):
        """
        Copy of PeriodicSite translated to the unit cell.
        """
        self._fcoords = np.mod(self._fcoords, 1)
        # for j, v in enumerate(self._fcoords):
        #     if abs(v - 1) < 1e-6:
        #         self._fcoords[j] = 0
        return PeriodicSite(self._species, self._fcoords, self._lattice,
                            properties=self._properties)

    def distance_and_image_from_frac_coords(self, fcoords, jimage=None):
        return self._lattice.get_distance_and_image(self._fcoords, fcoords,
                                                    jimage=jimage)

    def distance_and_image(self, other, jimage=None):
        return self.distance_and_image_from_frac_coords(other._fcoords, jimage)

    def distance(self, other, jimage=None):
        return self.distance_and_image(other, jimage)[0]

    def is_periodic_image(self, other, tolerance=1e-8, check_lattice=True):
        if check_lattice and self._lattice != other._lattice:
            return False
        if self._species != other._species:
            return False

        frac_diff = pbc_diff(self._fcoords, other._fcoords)
        return np.allclose(frac_diff, [0, 0, 0], atol=tolerance)
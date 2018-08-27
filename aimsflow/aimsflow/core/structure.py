import re
import os
import six
import warnings
import collections
import numpy as np
from fnmatch import fnmatch
from tabulate import tabulate
from itertools import groupby, product
from abc import ABCMeta, abstractmethod, abstractproperty

from aimsflow.core.lattice import Lattice
from aimsflow.core.composition import Composition
from aimsflow.core.sites import PeriodicSite
from aimsflow.symmetry.groups import SpaceGroup
from aimsflow.util import lattice_points_in_supercell, all_distances

try:
    from math import gcd
except ImportError:
    from fractions import gcd


class SiteCollection(six.with_metaclass(ABCMeta, collections.Sequence)):
    DISTANCE_TOLERANCE = 0.5

    @abstractproperty
    def sites(self):
        return

    @abstractmethod
    def get_distance(self, i, j):
        return

    def __contains__(self, site):
        return site in self.sites

    def __iter__(self):
        return self.sites.__iter__()

    def __getitem__(self, ind):
        return self.sites[ind]

    def __len__(self):
        return len(self.sites)

    def __hash__(self):
        # for now, just use the composition hash code.
        return self.composition.__hash__()

    @property
    def num_sites(self):
        return len(self)

    @property
    def cart_coords(self):
        return np.array([site.coords for site in self])

    @property
    def distance_matrix(self):
        return all_distances(self.cart_coords, self.cart_coords)

    @property
    def types_of_specie(self):
        return [a[0] for a in groupby(self.species)]

    @property
    def symbol_set(self):
        return tuple((specie.symbol for specie in self.types_of_specie))

    @property
    def species(self):
        return [site.specie for site in self]

    @property
    def species_and_occu(self):
        return [site.species_and_occu for site in self]

    @property
    def site_properties(self):
        props = {}
        prop_keys = set()
        for site in self:
            prop_keys.update(site.properties.keys())

        for k in prop_keys:
            props[k] = [site.properties.get(k, None) for site in self]
        return props

    @property
    def formula(self):
        return self.composition.formula

    @property
    def composition(self):
        """
        (Composition) Returns the composition
        """
        elmap = collections.defaultdict(float)
        for site in self:
            for species, occu in site.species_and_occu.items():
                elmap[species] += occu
        return Composition(elmap)

    @property
    def is_ordered(self):
        return all((site._is_ordered for site in self))


class IStructure(SiteCollection):
    def __init__(self, lattice, species, coords,
                 to_unit_cell=False, validate_proximity=False,
                 coords_cartesian=False, site_properties=None):
        if len(species) != len(coords):
            raise StructureError("The list of atomic species must be of the"
                                 " same length as the list of fractional"
                                 " coords.")

        if isinstance(lattice, Lattice):
            self._lattice = lattice
        else:
            self._lattice = Lattice(lattice)

        sites = []
        for i in range(len(species)):
            prop = None
            if site_properties:
                prop = {k: v[i] for k, v in site_properties.items()}

            sites.append(PeriodicSite(species[i], coords[i],
                                      self._lattice, to_unit_cell,
                                      coords_cartesian=coords_cartesian,
                                      properties=prop))
        self._sites = tuple(sites)
        if validate_proximity and not self.is_valid():
            raise StructureError(("Structure contains sites that are ",
                                  "less than 0.01 Angstrom apart!"))

    def __repr__(self):
        outs = ["Structure Summary", repr(self.lattice)]
        for s in self:
            outs.append(repr(s))
        return "\n".join(outs)

    def __str__(self):
        outs = ["Full Formula ({})".format(self.composition.formula),
                "Reduced Formula: {}".format(self.composition.reduced_formula)]
        to_s = lambda x: "%0.6f" % x
        outs.append("abc:\t" + " ".join([to_s(i).rjust(10)
                                          for i in self.lattice.abc]))
        outs.append("angles: " + " ".join([to_s(i).rjust(10)
                                           for i in self.lattice.angles]))
        outs.append("Sites ({})".format(len(self)))
        data = []
        props = self.site_properties
        keys = sorted(props.keys())
        for i, site in enumerate(self):
            row = [i, site.species_string]
            row.extend([to_s(j) for j in site.frac_coords])
            for k in keys:
                row.append(props[k][i])
            data.append(row)
        outs.append(tabulate(data, headers=["#", "SP", "a", "b", "c"] + keys))

        return "\n".join(outs)

    def __eq__(self, other):
        if other is None:
            return False
        if len(self) != len(other):
            return False
        if self.lattice != other.lattice:
            return False
        for site in self:
            if site not in other:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        # For now, just use the composition hash code.
        return self.composition.__hash__()

    def __mul__(self, scaling_matrix):
        scale_matrix = np.array(scaling_matrix, np.int16)
        if scale_matrix.shape != (3, 3):
            scale_matrix = np.array(scale_matrix * np.eye(3), np.int16)
        new_lattice = Lattice(np.dot(scale_matrix, self._lattice.matrix))

        f_lat = lattice_points_in_supercell(scale_matrix)
        c_lat = new_lattice.get_cart_coords(f_lat)

        new_sites = []
        for site in self:
            for v in c_lat:
                s = PeriodicSite(site.species_and_occu, site.coords + v,
                                 new_lattice, properties=site.properties,
                                 coords_cartesian=True, to_unit_cell=True)
                new_sites.append(s)
        return Structure.from_sites(new_sites)

    def __rmul__(self, scaling_matrix):
        return self.__mul__(scaling_matrix)

    @property
    def frac_coords(self):
        return np.array([site.frac_coords for site in self._sites])

    @property
    def lattice(self):
        return self._lattice

    @property
    def sites(self):
        return self._sites

    @staticmethod
    def get_heterostructure(structs, axis=2, delete_layer=None, dist=0, vacuum=0,
                            to_primitive=False, tol=0.25):
        """
        The heterostructure consists of various bulk materials. The surface
        terminations of each bulk are represented by two letters of if_mode,
        denoted by either "b" or "t". "b" means "bottom" and "t" means "top".
        So the number of letters in if_mode must be exactly two times to the
        number of bulk materials. For example, HS has two bulk materials and
        if_mode="btbt". This means a direct stacking without deleting the top
        or bottom layer. If, say "btbb", the substrate will keep as it is but
        top layer of film will be deleted.
        Args:
            structs (list): A list of Structure objects for bulk materials
            delete_layer (str): Delete top or bottom layers, which is represented by either
                "b" or "t". Default to None, which equals to 0b0t0b0t.
            dist (float): Extra distance at the interface
                Default to 0.0.
            vacuum (float): Add vacuum for heterostructure to make it a slab
                Default to 0.0.
            to_primitive (bool):  Whether to get primitive structure of GB.
                Default to False.
            tol (float), Angstroms: Tolerance for determining whether two
                atoms are at the same layer. Default to 0.25.
        :return:
            Structure object for HS
        """
        num_st = len(structs)
        if delete_layer is None:
            delete_layer = '0b0t' * num_st
        delete = re.findall('(\d+)(\w)', delete_layer)
        if len(delete) != 2 * num_st:
            raise ValueError("'%s' is not supported. Please make sure the "
                             "format is something like 1t1t1b1b.")
        # diff is particularly needed for superlattice HS with two bulk
        # materials. diff can make the two interfaces exactly the same
        diff = structs[0].get_layer_info(tol=tol, axis=axis)["dists"][0] - \
               structs[-1].get_layer_info(tol=tol, axis=axis)["dists"][0]
        for i, v in enumerate(delete):
            for j in range(int(v[0])):
                structs[i // 2].delete_bt_layer(v[1], tol, axis)
        abc, ang = structs[0].lattice.lengths_and_angles
        new_abc = list(abc)
        l_list = [i.lattice.abc[axis] for i in structs]
        new_abc[axis] = sum(l_list) + vacuum + (len(structs)) * dist + diff
        new_lat = Lattice.from_lengths_angles(new_abc, ang)
        l = 0
        new_sites = []
        # structs[0].to(filename="POSCAR")
        # exit(0)
        for i, s in enumerate(structs):
            f_coords = s.frac_coords
            f_coords[:, axis] = (s.cart_coords[:, axis] + l + dist * i) / new_abc[axis]
            for j, site in enumerate(s):
                new_sites.append(PeriodicSite(site.specie, f_coords[j], new_lat,
                                              properties=site.properties))
            l += l_list[i]

        new_s = Structure.from_sites(new_sites)
        new_s = new_s.get_sorted_structure()
        if to_primitive:
            new_s = new_s.get_primitive_structure()
        return new_s

    @staticmethod
    def get_sandwich(structs, delete_layer=None, dist=0, vacuum=0,
                     to_primitive=False, tol=0.25):
        """
        The sandwiched structure is essentially a stacking of two mirror
        symmetric heterostructures. Each HS consists of various bulk materials.
        The surface terminations of each bulk are represented by two letters
        of if_mode, denoted by either "b" or "t". "b" means "bottom" and "t"
        means "top". So the number of letters in if_mode must be exactly two
        times to the number of bulk materials. For example, HS has two bulk
        materials and if_mode="btbt". This means a direct stacking without
        deleting the top or bottom layer. If, say "btbb", the substrate will
        keep as it is but top layer of film will be deleted.
        Args:
            structs (list): A list of Structure objects for bulk materials
            delete_layer (str): Delete top or bottom layers, which is represented by either
                "b" or "t". Default to None, which equals to 0b0t0b0t.
            dist (float): Extra distance at the interface
                Default to 0.0.
            vacuum (float): Add vacuum for heterostructure to make it a slab
                Default to 0.0.
            to_primitive (bool):  Whether to get primitive structure of GB.
                Default to False.
            tol (float), Angstroms: Tolerance for determining whether two
                atoms are at the same layer. Default to 0.25.
        :return:
            Structure object for HS
        """
        hs1 = Structure.get_heterostructure(
            structs, delete_layer=delete_layer, dist=dist, vacuum=vacuum,
            to_primitive=to_primitive, tol=tol)
        f_coords = hs1.frac_coords
        f_coords[:, 2] = 1 - f_coords[:, 2]
        hs2 = Structure(hs1.lattice, hs1.species, f_coords)

        abc, ang = hs1.lattice.lengths_and_angles
        new_abc = list(abc)
        new_abc[2] *= 2
        new_lat = Lattice.from_lengths_angles(new_abc, ang)
        new_sites = []
        for i, s in enumerate(hs2[:] + hs1[:]):
            if i <= hs2.num_sites - 1:
                c = 0
            else:
                c = hs2.lattice.abc[2]
            new_sites.append(PeriodicSite(s.specie, s.coords + [0, 0, c],
                                          new_lat, coords_cartesian=True,
                                          properties=s.properties))
        new_s = Structure.from_sites(new_sites)
        new_s.merge_sites(mode="delete")
        new_s = new_s.get_sorted_structure()
        if to_primitive:
            new_s = new_s.get_primitive_structure()
        return new_s

    @classmethod
    def from_sites(cls, sites, validate_proximity=False, to_unit_cell=False):
        if len(sites) < 1:
            raise ValueError("You need at least one site to construct a %s" %
                             cls)
        if (not validate_proximity) and (not to_unit_cell):
            lattice = sites[0].lattice
            for s in sites[1:]:
                if s.lattice != lattice:
                    raise ValueError("Sites must belong to the same lattice")
            s_copy = cls(lattice=lattice, species=[], coords=[])
            s_copy._sites = list(sites)
            return s_copy

        prop_keys = []
        props = {}
        lattice = None
        for i, site in enumerate(sites):
            if not lattice:
                lattice = site.lattice
            elif site.lattice != lattice:
                raise ValueError("Sites must belong to the same lattice")
            for k, v in site.properties.items():
                if k not in prop_keys:
                    prop_keys.append(k)
                    props[k] = [None] * len(sites)
                props[k][i] = v
        for k, v in props.items():
            if any((vv is None for vv in v)):
                warnings.warn("Not all sites have property %s. Missing values "
                              "are set to None." % k)
        return cls(lattice, [site.species_and_occu for site in sites],
                   [site.frac_coords for site in sites],
                   site_properties=props,
                   validate_proximity=validate_proximity,
                   to_unit_cell=to_unit_cell)

    @classmethod
    def from_spacegroup(cls, sg, lattice, species, coords, site_properties=None,
                        coords_cartesian=False, tol=1e-5):
        try:
            i = int(sg)
            sgp = SpaceGroup.from_int_number(i)
        except ValueError:
            sgp = SpaceGroup(sg)

        if isinstance(lattice, Lattice):
            latt = lattice
        else:
            latt = Lattice(lattice)

        if not sgp.is_compatible(latt):
            raise ValueError(
                "Supplied lattice with parameters %s is incompatible with "
                "supplied spacegroup %s!" % (latt.lengths_and_angles,
                                             sgp.symbol)
            )

        if len(species) != len(coords):
            raise ValueError(
                "Supplied species and coords lengths (%d vs %d) are "
                "different!" % (len(species), len(coords))
            )

        frac_coords = coords if not coords_cartesian else \
            lattice.get_frac_coords(coords)

        props = {} if site_properties is None else site_properties
        all_sp = []
        all_coords = []
        all_site_properties = collections.defaultdict(list)
        for i, (sp, c) in enumerate(zip(species, frac_coords)):
            cc = sgp.get_orbit(c, tol=tol)
            all_sp.extend([sp] * len(cc))
            all_coords.extend(cc)
            for k, v in props.items():
                all_site_properties[k].extend([v[i]] * len(cc))

        return cls(latt, all_sp, all_coords,
                   site_properties=all_site_properties)

    @classmethod
    def from_str(cls, input_string, fmt, primitive=False, sort=False,
                 merge_tol=0.0, to_unit_cell=False):
        # from aimsflow.vasp_io.cif import CifParser
        from aimsflow.vasp_io import Poscar

        fmt = fmt.lower()
        #     parser = CifParser.from_string(input_string)
        if fmt == "poscar":
            s = Poscar.from_string(input_string, False).structure
        else:
            raise ValueError("Unrecognized format '%s'!" % fmt)
        if sort:
            s = s.get_sorted_structure()
        if merge_tol:
            s.merge_sites(merge_tol)
        return cls.from_sites(s, to_unit_cell=to_unit_cell)

    @classmethod
    def from_file(cls, filename, primitive=False, sort=False, merge_tol=0.0,
                  to_unit_cell=False):
        from aimsflow.util.io_utils import zopen

        fname = os.path.basename(filename)
        with zopen(filename) as f:
            contents = f.read()
        if fnmatch(fname.lower(), "*.cif*"):
            return cls.from_str(contents, fmt="cif",
                                primitive=primitive, sort=sort,
                                merge_tol=merge_tol)
        elif any([fnmatch(fname, file_type)
                  for file_type in ["*POSCAR*", "*CONTCAR*", "*vasp*"]]):
            s = cls.from_str(contents, fmt="poscar",
                             primitive=primitive, sort=sort,
                             merge_tol=merge_tol, to_unit_cell=to_unit_cell)
        # elif fnmatch(fname, "CHGCAR*") or fnmatch(fname, "LOCPOT*"):
        #     s = Chgcar.from_file(filename).structure
        # elif fnmatch(fname, "vasprun*.xml*"):
        #     s = Vasprun(filename).final_structure
        else:
            raise ValueError("Unrecognized file extension!")
        if sort:
            s = s.get_sorted_structure()
        if merge_tol:
            s.merge_sites(merge_tol)

        s.__class__ = cls
        return s

    def get_distance(self, i, j, jimage=None):
        return self[i].distance(self[j], jimage)

    def get_sorted_structure(self, key=None, reverse=False):
        sites = sorted(self, key=key, reverse=reverse)
        return self.__class__.from_sites(sites)

    def get_layered_structure(self, tol=1.2, reverse=False):
        new_sites = []
        layers = self.sort_sites_in_layers(tol, reverse)
        for l in layers:
            new_sites.extend(l)
        return self.__class__.from_sites(new_sites)

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
        l_dist = abs((layers[l1][0].frac_coords[axis] -
                      layers[l2][0].frac_coords[axis]) * l)
        l_vector = [1, 1]
        l_vector.insert(axis, (l - l_dist) / l)
        new_lat = Lattice(self.lattice.matrix * np.array(l_vector)[:, None])

        layers.pop(l1)
        sites = [site for l in layers for site in l]
        new_sites = []
        l_dist = 0 if bt == "t" else l_dist
        l_vector = [0, 0]
        l_vector.insert(axis, l_dist)
        for a in sites:
            new_sites.append(PeriodicSite(a.specie, a.coords - l_vector,
                                          new_lat, to_unit_cell=True,
                                          coords_cartesian=True))
        self._sites = sorted(new_sites)
        self._lattice = new_lat

    def sort_sites_in_layers(self, tol=1.2, reverse=False, axis=2):
        new_atoms = sorted(self, key=lambda x: x.frac_coords[axis],
                           reverse=reverse)
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
                                layers[0][0].frac_coords[axis] + 1 -
                                layers[-1][0].frac_coords[axis]) < tol:
            tmp = layers[0] + layers[-1]
            new_layers = new_layers[1:-1]
            new_layers.append(sorted(tmp))

        return list(reversed(new_layers)) if reverse else new_layers

    def get_layer_info(self, n=1, tol=1.2, reverse=False, axis=2):
        l = self.lattice.abc[axis]
        layers = self.sort_sites_in_layers(tol, reverse, axis)
        layer_num = len(layers)
        l_heights = []
        for i in layers:
            h_list = [s.frac_coords[axis] for s in i]
            # h_list at the boundary layer could have both + and - values
            # If so, add 1 for h_list which has value > 0.5
            if all(j < 0 for j in h_list) or all(j > 0 for j in h_list):
                l_heights.append(sum(h_list) / len(h_list) * l)
            else:
                multi = sum(j > 0.5 for j in h_list)
                l_heights.append((sum(h_list) + multi) / len(h_list) * l)
        dists = []
        for i in range(1, layer_num):
            dists.append(l_heights[i] - l_heights[i - 1])
        l_name = [Composition("".join(
            [i.species_string for i in j])).reduced_formula for j in layers]
        if_ind = []
        if_dist = []
        if_name = []
        for i in range(n, layer_num, n):
            if l_name[i:i + n] == l_name[i - n:i]:
                continue
            for j, v in enumerate(l_name[i:i + n]):
                if v not in l_name[i - n:i]:
                    axis = i + j
                    if if_ind and axis - 1 in if_ind[-1] or dists[axis - 1] > 4:
                        continue
                    if_ind.append((axis - 1, axis))
                    if_dist.append(dists[axis - 1])
                    if_name.append("/".join([l_name[axis - 1], l_name[axis]]))
                    break
        # Check whether the bottom and top layers are interfacial layers.
        # Here I put the maximum interlayer distance as 3 angstrom. If
        # extra is larger than 3, then they are separated by vacuum
        extra = l_heights[0] + l - l_heights[-1]
        if extra < 3 and l_name[0] not in l_name[-n:]:
            if_ind.append((layer_num - 1, 0))
            if_dist.append(extra)
            if_name.append("/".join([l_name[-1], l_name[0]]))
        return {"layers": layers, "dists": dists, "if_ind": if_ind,
                "if_dist": if_dist, "if_name": if_name,
                "l_name": l_name, "l_heights": l_heights}

    def copy(self, site_properties=None):
        if not site_properties:
            s_copy = Structure(lattice=self._lattice, species=[], coords=[])
            s_copy._sites = list(self._sites)
            return s_copy
        props = self.site_properties
        if site_properties:
            props.update(site_properties)

        return self.__class__(self._lattice,
                              self.species_and_occu,
                              self.frac_coords,
                              site_properties=props)

    def to(self, fmt="", filename="", **kwargs):
        from aimsflow.vasp_io import Poscar

        writer = Poscar(self)
        if filename:
            writer.write_file(filename, **kwargs)
        else:
            writer.__str__()


class Structure(IStructure, collections.MutableSequence):
    __hash__ = None

    def __init__(self, lattice, species, coords,
                 to_unit_cell=False, validate_proximity=False,
                 coords_cartesian=False, site_properties=None):
        super(Structure, self).__init__(lattice, species, coords,
                                        to_unit_cell=to_unit_cell,
                                        validate_proximity=validate_proximity,
                                        coords_cartesian=coords_cartesian,
                                        site_properties=site_properties)

        self._sites = list(self._sites)

    def __setitem__(self, i, site):
        if isinstance(site, PeriodicSite):
            if site.lattice != self._lattice:
                raise ValueError("PeriodicSite added must have same lattice "
                                 "as Structure!")
            self._sites[i] = site
        else:
            if isinstance(site, six.string_types) or (
                    not isinstance(site, collections.Sequence)):
                sp = site
                fract_coords = self._sites[i].fract_coords
                site_properties = self._sites[i].site_properties
            else:
                sp = site[0]
                fract_coords = site[1] if len(site) > 1 else self._sites[i] \
                    .fract_coords
                site_properties = site[2] if len(site) > 2 else self._sites[i] \
                    .site_properties

            self._sites[i] = PeriodicSite(sp, fract_coords, self._lattice,
                                          properties=site_properties)

    def __delitem__(self, i):
        self._sites.__delitem__(i)

    def append(self, species, coords, coords_are_cartesian=False,
               validate_proximity=False, site_properties=None):
        return self.insert(len(self), species, coords, coords_are_cartesian,
                           validate_proximity=validate_proximity,
                           site_properties=site_properties)

    def insert(self, i, species, coords, coords_are_cartesian=False,
               validate_proximity=False, site_properties=None):
        if not coords_are_cartesian:
            new_site = PeriodicSite(species, coords, self._lattice,
                                    properties=site_properties)
        else:
            fract_coords = self._lattice.get_frac_coords(coords)
            new_site = PeriodicSite(species, fract_coords, self._lattice,
                                    properties=site_properties)

        if validate_proximity:
            for site in self:
                if site.distance(new_site) < self.DISTANCE_TOLERANCE:
                    raise ValueError("New site is too close to an "
                                     "existing site!")

        self._sites.insert(i, new_site)

    def modify_lattice(self, new_lattice):
        self._lattice = new_lattice
        f_coords = new_lattice.get_frac_coords(self.cart_coords)
        new_sites = []
        for i, s in enumerate(self._sites):
            new_sites.append(PeriodicSite(
                s.specie, f_coords[i], self._lattice, properties=s.properties))
        self._sites = new_sites

    def add_strain(self, strain):
        s = (1 + np.array(strain)) * np.eye(3)
        self.modify_lattice(Lattice(np.dot(self._lattice.matrix.T, s).T))

    def make_supercell(self, scaling_matrix, to_unit_cell=True):
        """
        Args:
            scaling_matrix: A scaling matrix for transforming the lattice
                vectors. Has to be all integers. Several options are possible:

                a. A full 3x3 scaling matrix defining the linear combination
                   the old lattice vectors. E.g., [[2,1,0],[0,3,0],[0,0,
                   1]] generates a new structure with lattice vectors a' =
                   2a + b, b' = 3b, c' = c where a, b, and c are the lattice
                   vectors of the original structure.
                b. A sequence of three scaling factors. E.g., [2, 1, 1]
                   specifies that the supercell should have dimensions
                   2a x b x c.
                c. A number, which simply scales all lattice vectors by the
                   same factor.
        """
        s = self * scaling_matrix
        if to_unit_cell:
            for i, v in enumerate(s):
                s[i] = v.to_unit_cell
        self._sites = s.sites
        self._lattice = s.lattice

    def translate_sites(self, vector, indices=None, frac_coords=True,
                        to_unit_cell=True):
        if indices is None:
            indices = range(len(self))

        for i in indices:
            site = self._sites[i]
            if frac_coords:
                fcoords = site.frac_coords + vector
            else:
                fcoords = self._lattice.get_frac_coords(site.coords + vector)
            new_site = PeriodicSite(site.species_and_occu, fcoords,
                                    self._lattice, to_unit_cell=to_unit_cell,
                                    coords_cartesian=False,
                                    properties=site.properties)
            self._sites[i] = new_site

    def merge_sites(self, tol=0.01, mode="sum"):
        mode = mode.lower()[0]
        from scipy.spatial.distance import squareform
        from scipy.cluster.hierarchy import fcluster, linkage

        d = self.distance_matrix
        np.fill_diagonal(d, 0)
        clusters = fcluster(linkage(squareform((d + d.T) / 2)),
                            tol, "distance")
        sites = []
        for c in np.unique(clusters):
            inds = np.where(clusters == c)[0]
            species = self[inds[0]].species_and_occu
            coords = self[inds[0]].frac_coords
            for n, i in enumerate(inds[1:]):
                sp = self[i].species_and_occu
                if mode == "s":
                    species += sp
                offset = self[i].frac_coords - coords
                coords += ((offset - np.round(offset)) / (n + 2)).astype(
                    coords.dtype)
            sites.append(PeriodicSite(species, coords, self.lattice))

        self._sites = sites

    def replace_species(self, species_mapping):
        for k, v in species_mapping.items():
            for i, site in enumerate(self._sites):
                if k == site.species_string:
                    self.replace(i, v)

    def replace(self, i, species, coords=None, coords_cartesian=False,
                properties=None):
        if coords is None:
            frac_coords = self[i].frac_coords
        elif coords_cartesian:
            frac_coords = self._lattice.get_frac_coords(coords)
        else:
            frac_coords = coords

        new_site = PeriodicSite(species, frac_coords, self._lattice,
                                properties=properties)
        self._sites[i] = new_site

    def remove_sites(self, indices):
        self._sites = [s for i, s in enumerate(self._sites)
                       if i not in indices]

    def remove_species(self, species):
        self._sites = [s for i, s in enumerate(self._sites)
                       if s.species_string not in species]

    def rotate_sites(self, indices=None, theta=0,
                     axis=(0, 0, 1), anchor=(0, 0, 0)):
        from numpy.linalg import norm
        from numpy import cross, radians, eye
        from scipy.linalg import expm

        if indices is None:
            indices = range(len(self))
        anchor = self._lattice.get_cart_coords(np.array(anchor))
        axis = np.array(axis)
        theta = radians(theta)
        theta %= 2 * np.pi
        rm = expm(cross(eye(3), axis / norm(axis)) * theta)

        for i in indices:
            site = self._sites[i]
            s = ((rm * np.matrix(site.coords - anchor).T).T + anchor).A1
            new_site = PeriodicSite(site.species_and_occu, s, self._lattice,
                                    properties=site.properties,
                                    coords_cartesian=True)
            self._sites[i] = new_site

    def add_site_property(self, property_name, values):
        if len(values) != len(self._sites):
            raise ValueError("Values must be same length as sites.")
        for i in range(len(self._sites)):
            site = self._sites[i]
            props = site.properties
            if not props:
                props = {}
            props[property_name] = values[i]
            self._sites[i] = PeriodicSite(site.species_and_occu,
                                          site.frac_coords, self._lattice,
                                          properties=props)

    def get_sites_in_sphere(self, center, r, w_index=True):
        site_fcoords = np.mod(self.frac_coords, 1)
        outs = []
        for fcoord, dist, i in self._lattice.get_points_in_sphere(
                site_fcoords, center, r):
            site = PeriodicSite(self[i].species_and_occu, fcoord, self._lattice,
                                properties=self[i].properties)
            outs.append([site, dist, i] if w_index else [site, dist])
        return outs

    def get_neighbors(self, site, r, w_index=True):
        n = self.get_sites_in_sphere(site.coords, r, w_index=w_index)
        return [d for d in n if site != d[0]]

    def get_primitive_structure(self, tolerance=0.25):
        """
        This finds a smaller unit cell than the input. Sometimes it doesn"t
        find the smallest possible one, so this method is recursively called
        until it is unable to find a smaller cell.

        NOTE: if the tolerance is greater than 1/2 the minimum inter-site
        distance in the primitive cell, the algorithm will reject this lattice.

        Args:
            tolerance (float), Angstroms: Tolerance for each coordinate of a
                particular site. For example, [0.1, 0, 0.1] in cartesian
                coordinates will be considered to be on the same coordinates
                as [0, 0, 0] for a tolerance of 0.25. Defaults to 0.25.

        Returns:
            The most primitive structure found.
        """
        # group sites by species string
        sites = sorted(self, key=lambda x: x.species_string)
        grouped_atoms= [
            list(a[1]) for a in groupby(sites, key=lambda x: x.species_string)]

        grouped_fcoords = [np.array([a.frac_coords for a in g])
                           for g in grouped_atoms]

        # min_vecs are approximate periodicities of the cell. The exact
        # periodicities from the supercell matrices are checked against these
        # first
        min_fcoords = min(grouped_fcoords, key=lambda x: len(x))
        min_vecs = min_fcoords - min_fcoords[0]

        # fractional tolerance in the supercell
        super_ftol = np.divide(tolerance, self.lattice.abc)
        super_ftol_2 = super_ftol * 2

        def pbc_coord_intersection(fc1, fc2, tol):
            """
            Returns the fractional coords in fc1 that have coordinates
            within tolerance to some coordinate in fc2
            """
            d = fc1[:, None, :] - fc2[None, :, :]
            d -= np.round(d)
            np.abs(d, d)
            return fc1[np.any(np.all(d < tol, axis=-1), axis=-1)]

        # here we reduce the number of min_vecs by enforcing that every
        # vector in min_vecs approximately maps each site onto a similar site.
        # The subsequent processing is O(fu^3 * min_vecs) = O(n^4) if we do no
        # reduction.
        # This reduction is O(n^3) so usually is an improvement. Using double
        # the tolerance because both vectors are approximate
        for g in sorted(grouped_fcoords, key=lambda x: len(x)):
            for f in g:
                min_vecs = pbc_coord_intersection(min_vecs, g - f, super_ftol_2)

        def get_hnf(fu):
            """
            Returns all possible distinct supercell matrices given a
            number of formula units in the supercell. Batches the matrices
            by the values in the diagonal (for less numpy overhead).
            Computational complexity is O(n^3), and difficult to improve.
            Might be able to do something smart with checking combinations of a
            and b first, though unlikely to reduce to O(n^2).
            """

            def factors(n):
                for i in range(1, n + 1):
                    if n % i == 0:
                        yield i

            for det in factors(fu):
                if det == 1:
                    continue
                for a in factors(det):
                    for e in factors(det // a):
                        g = det // a // e
                        yield det, np.array(
                            [[[a, b, c], [0, e, f], [0, 0, g]]
                             for b, c, f in
                             product(range(a), range(a),
                                               range(e))])

        # we cant let sites match to their neighbors in the supercell
        grouped_non_nbrs = []
        for gfcoords in grouped_fcoords:
            fdist = gfcoords[None, :, :] - gfcoords[:, None, :]
            fdist -= np.round(fdist)
            np.abs(fdist, fdist)
            non_nbrs = np.any(fdist > 2 * super_ftol[None, None, :], axis=-1)
            # since we want sites to match to themselves
            np.fill_diagonal(non_nbrs, True)
            grouped_non_nbrs.append(non_nbrs)

        num_fu = six.moves.reduce(gcd, map(len, grouped_atoms))
        for size, ms in get_hnf(num_fu):
            inv_ms = np.linalg.inv(ms)

            # find sets of lattice vectors that are are present in min_vecs
            dist = inv_ms[:, :, None, :] - min_vecs[None, None, :, :]
            dist -= np.round(dist)
            np.abs(dist, dist)
            is_close = np.all(dist < super_ftol, axis=-1)
            any_close = np.any(is_close, axis=-1)
            ind = np.all(any_close, axis=-1)

            for inv_m, m in zip(inv_ms[ind], ms[ind]):
                new_m = np.dot(inv_m, self.lattice.matrix)
                ftol = np.divide(tolerance, np.sqrt(np.sum(new_m ** 2, axis=1)))

                valid = True
                new_coords = []
                new_sp = []
                new_prop = collections.defaultdict(list)
                for gatoms, gfcoords, non_nbrs in zip(grouped_atoms,
                                                      grouped_fcoords,
                                                      grouped_non_nbrs):
                    all_frac = np.dot(gfcoords, m)

                    # calculate grouping of equivalent sites, represented by
                    # adjacency matrix
                    fdist = all_frac[None, :, :] - all_frac[:, None, :]
                    fdist = np.abs(fdist - np.round(fdist))
                    close_in_prim = np.all(fdist < ftol[None, None, :], axis=-1)
                    groups = np.logical_and(close_in_prim, non_nbrs)

                    # check that groups are correct
                    if not np.all(np.sum(groups, axis=0) == size):
                        valid = False
                        break

                    # check that groups are all cliques
                    for g in groups:
                        if not np.all(groups[g][:, g]):
                            valid = False
                            break
                    if not valid:
                        break

                    # add the new sites, averaging positions
                    added = np.zeros(len(gatoms))
                    new_fcoords = all_frac % 1
                    for i, group in enumerate(groups):
                        if not added[i]:
                            added[group] = True
                            ind = np.where(group)[0]
                            coords = new_fcoords[ind[0]]
                            for n, j in enumerate(ind[1:]):
                                offset = new_fcoords[j] - coords
                                coords += (offset - np.round(offset)) / (n + 2)
                            new_sp.append(gatoms[ind[0]].specie)
                            for k in gatoms[ind[0]].properties:
                                new_prop[k].append(gatoms[ind[0]].properties[k])
                            new_coords.append(coords)

                if valid:
                    inv_m = np.linalg.inv(m)
                    new_l = Lattice(np.dot(inv_m, self.lattice.matrix))
                    s = Structure(new_l, new_sp, new_coords,
                                  site_properties=new_prop)

                    return s.get_primitive_structure(
                        tolerance=tolerance).get_reduced_structure()
        return self

    def get_reduced_structure(self, reduction_algo="niggli"):
        """
        Get a reduced structure.
        """
        if reduction_algo == "niggli":
            reduced_latt = self.lattice.get_niggli_reduced_lattice()
        elif reduction_algo == "LLL":
            reduced_latt = self._lattice.get_lll_reduced_lattice()
        else:
            raise ValueError("Invalid reduction algo : {}"
                             .format(reduction_algo))

        if reduced_latt != self.lattice:
            return self.__class__(reduced_latt, self.species, self.cart_coords,
                                  coords_cartesian=True, to_unit_cell=True,
                                  site_properties=self.site_properties)
        else:
            return self.copy()


class StructureError(Exception):
    pass

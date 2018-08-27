from __future__ import division

import re
import itertools
import numpy as np
from functools import reduce
try:
    from math import gcd
except ImportError:
    from fractions import gcd

from aimsflow.util.num_utils import lcm
from aimsflow.core.structure import Structure, Lattice


class Slab(Structure):
    def __init__(self, lattice, species, coords, miller_index,
                 oriented_unit_cell, validate_proximity=False,
                 to_unit_cell=False, coords_cartesian=False,
                 site_properties=None):
        self.oriented_unit_cell = oriented_unit_cell
        self.miller_index = tuple(miller_index)
        super(Slab, self).__init__(
            lattice, species, coords, validate_proximity=validate_proximity,
            to_unit_cell=to_unit_cell, coords_cartesian=coords_cartesian,
            site_properties=site_properties)


class SlabGenerator(object):
    def __init__(self, initial_strucutre, miller_index, slab_uc,
                 vacuum, lll_reduce=False, center_slab=False,
                 primitive=True, max_normal_search=None):
        latt = initial_strucutre.lattice
        miller_index = reduce_vector(miller_index)
        recp = latt.reciprocal_lattice_crystallographic
        normal = recp.get_cart_coords(miller_index)
        normal /= np.linalg.norm(normal)

        slab_scale_factor = []
        non_orth_ind = []
        eye = np.eye(3, dtype=np.int)
        for i, j in enumerate(miller_index):
            if j == 0:
                slab_scale_factor.append(eye[i])
            else:
                d = abs(np.dot(normal, latt.matrix[i])) / latt.abc[i]
                non_orth_ind.append((i, d))
        c_index, dist = max(non_orth_ind, key=lambda t: t[1])

        if len(non_orth_ind) > 1:
            lcm_miller = lcm(*[miller_index[i] for i, d in non_orth_ind])
            for (i, di), (j, dj) in itertools.combinations(non_orth_ind, 2):
                l = [0, 0, 0]
                l[i] = -int(round(lcm_miller / miller_index[i]))
                l[j] = int(round(lcm_miller / miller_index[j]))
                slab_scale_factor.append(l)
                if len(slab_scale_factor) == 2:
                    break

        if max_normal_search is None:
            slab_scale_factor.append(eye[c_index])
        else:
            index_range = sorted(
                reversed(range(-max_normal_search, max_normal_search + 1)),
                key=lambda x: abs(x))
            candidates = []
            for uvw in itertools.product(index_range, index_range, index_range):
                if (not any(uvw)) or abs(
                        np.linalg.det(slab_scale_factor + [uvw])) < 1e-8:
                    continue
                vec = latt.get_cart_coords(uvw)
                l = np.linalg.norm(vec)
                cosine = abs(np.dot(vec, normal) / l)
                candidates.append((uvw, cosine, l))
                if abs(abs(cosine) - 1) < 1e-8:
                    break
            uvw, cosine, l = max(candidates, key=lambda x: (x[1], x[2]))
            slab_scale_factor.append(uvw)

        slab_scale_factor = np.array(slab_scale_factor)

        if np.linalg.det(slab_scale_factor) < 0:
            slab_scale_factor *= -1

        reduced_scale_factor = [reduce_vector(v) for v in slab_scale_factor]
        slab_scale_factor = np.array(reduced_scale_factor)

        single = initial_strucutre.copy()
        single.make_supercell(slab_scale_factor)

        self.oriented_unit_cell = Structure.from_sites(single,
                                                       to_unit_cell=True)
        self.parent = initial_strucutre
        self.lll_reduce = lll_reduce
        self.center_slab = center_slab
        self.miller_index = miller_index
        self.vacuum = vacuum
        self.slab_uc = slab_uc
        self.primitive = primitive
        self._normal = normal
        a, b, c = self.oriented_unit_cell.lattice.matrix
        self._proj_height = abs(np.dot(normal, c))

    def get_slab(self, delete_layer="0b0t", tol=0.25):
        nlayers = self.slab_uc + self.vacuum / self._proj_height
        species = self.oriented_unit_cell.species_and_occu
        props = self.oriented_unit_cell.site_properties
        props = {k: v * self.slab_uc for k, v in props.items()}
        frac_coords = self.oriented_unit_cell.frac_coords
        frac_coords -= np.floor(frac_coords)
        a, b, c = self.oriented_unit_cell.lattice.matrix
        new_lattice = Lattice([a, b,  nlayers * c])
        abc, angles = new_lattice.lengths_and_angles
        new_lattice = Lattice.from_lengths_angles(abc, angles)
        frac_coords[:, 2] = frac_coords[:, 2] / nlayers
        all_coords = []
        for i in range(self.slab_uc):
            fcoords = frac_coords.copy()
            fcoords[:, 2] += float(i) / nlayers
            all_coords.extend(fcoords)
        slab = Structure(new_lattice, species * self.slab_uc, all_coords,
                         site_properties=props)
        delete = re.findall('(\d+)(\w)', delete_layer)
        if len(delete) != 2:
            raise ValueError("'%s' is not supported. Please make sure the "
                             "format is 0b0t.")
        for i, v in enumerate(delete):
            for j in range(int(v[0])):
                slab.delete_bt_layer(v[1], tol, axis=2)

        if self.center_slab:
            avg_c = np.average([c[2] for c in slab.frac_coords])
            slab.translate_sites((list(range(len(slab)))), [0, 0, 0.5 - avg_c])
        return Slab(slab.lattice, slab.species_and_occu,
                    slab.frac_coords, self.miller_index,
                    self.oriented_unit_cell, site_properties=slab.site_properties)


def reduce_vector(vector):
    d = abs(reduce(gcd, vector))
    vector = tuple([int(i / d) for i in vector])
    return vector

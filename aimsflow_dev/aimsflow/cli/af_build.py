from __future__ import division

import re
import copy
import numpy as np
from numpy import inner, sqrt
from itertools import groupby

from aimsflow import Structure, Lattice, PeriodicSite, SlabGenerator
from aimsflow.cli.af_calculate import oxi_displace
from aimsflow.util import float_filename, flatten_lists, parse_number


def build_hs(args):
    files = args.poscars
    uc = [int(i) for i in args.uc.split(",")]
    if len(uc) != len(files):
        raise IOError("Inconsistent number of POSCARs and uc.")
    structs = []
    for i, f in enumerate(files):
        s = Structure.from_file(f)
        s.make_supercell([1, 1, uc[i]])
        structs.append(s)

    if args.sw:
        hs = Structure.get_sandwich(
            structs, delete_layer=args.delete_layer, vacuum=args.vacuum,
            tol=args.tol, dist=args.extra_distance, to_primitive=args.primitive)
    else:
        hs = Structure.get_heterostructure(
            structs, delete_layer=args.delete_layer, vacuum=args.vacuum,
            tol=args.tol, dist=args.extra_distance, to_primitive=args.primitive)
        if args.sd:
            layers = hs.sort_sites_in_layers()
            sd_sites = []
            new_sites = []
            for i, l in enumerate(layers):
                if i < args.sd:
                    sd_sites.extend([[False, False, False]] * len(l))
                else:
                    sd_sites.extend([[True, True, True]] * len(l))
                new_sites.extend(l)
            hs = Structure.from_sites(new_sites)
            hs.add_site_property("selective_dynamics", sd_sites)
            hs = hs.get_sorted_structure()
    hs.to(filename="POSCAR")


def build_sc(args):
    s = Structure.from_file(args.poscar)
    scaling_matrix = np.array(args.scaling_matrix.split(","), np.int16)
    if len(scaling_matrix) == 9:
        scaling_matrix = scaling_matrix.reshape(3, 3)
    # print(scaling_matrix)
    # exit(0)
    s.make_supercell(scaling_matrix)
    s.to(filename="POSCAR")


def build_slab(args):
    s = Structure.from_file(args.poscar)
    miller_index = list(map(int, args.miller_index.split(",")))
    g = SlabGenerator(s, miller_index, args.unit_cell, args.vacuum,
                      lll_reduce=True, max_normal_search=max(miller_index))
    new_s = g.get_slab(args.delete_layer, args.tol)
    # print(type(new_s))
    # print(new_s.copy())
    new_s.to(filename="POSCAR")


def build_strain(args):
    strain = [float(i) for i in args.strain]
    structure = Structure.from_file(args.poscar)
    d = dict(zip(args.direction, [1] * len(args.direction)))
    direct = [d.get(i, 0) for i in "xyz"]

    for i in strain:
        struct = structure.copy()
        s = np.multiply(direct, i / 100)
        struct.add_strain(s)
        name = float_filename(i)
        filename = "%s_%s" % (args.poscar, name)
        struct.to(filename=filename)


def build_rotate(args):
    struct = Structure.from_file(args.poscar)
    site_num = [i - 1 for i in parse_number(args.site_number)]
    axis = list(map(int, args.axis.split(",")))
    anchor = list(map(float, args.anchor.split(",")))
    struct.rotate_sites(site_num, args.angle, axis, anchor)
    struct.to(filename="POSCAR")


def build_if(args):
    f = args.poscar
    struct = Structure.from_file(f)
    dist = args.dist
    layer_info = struct.get_layer_info(args.nlayers, args.tol)
    layers = layer_info["layers"]
    sorted_sites = flatten_lists(layers)
    it_l = layer_info["if_ind"]
    if not it_l:
        raise IOError("aimsflow cannot find an interface for %s" % f)
    abc, angles = struct.lattice.lengths_and_angles

    for d in dist:
        new_sites = []
        tmp_abc = copy.copy(list(abc))
        tmp_abc[2] += 2 * d

        new_lat = Lattice.from_lengths_angles(tmp_abc, angles)
        for i, s in enumerate(sorted_sites):
            if len(it_l) == 1:
                num = len(flatten_lists(layers[:it_l[0][1]]))
                if i <= num - 1:
                    l = 0
                else:
                    l = d
            elif len(it_l) == 2:
                num1 = len(flatten_lists(layers[:it_l[0][1]]))
                if it_l[1][1] == 0:
                    num2 = len(flatten_lists(layers[:it_l[1][0]])) + 1
                else:
                    num2 = len(flatten_lists(layers[:it_l[1][1]]))
                if i <= num1 - 1:
                    l = 0
                elif i <= num2 - 1:
                    l = d
                else:
                    l = 2 * d
            else:
                raise IOError("There are more than 2 interfaces.")
            new_sites.append(PeriodicSite(s.specie, s.coords + [0, 0, l],
                                          new_lat, coords_cartesian=True,
                                          properties=s.properties))
        new_s = Structure.from_sites(new_sites)
        new_s = new_s.get_sorted_structure()
        name = float_filename(d)
        new_s.to(filename="%s_%s" % (f, name))


def translate_sites(args):
    f = args.poscar
    struct = Structure.from_file(f)
    site_number = [i - 1 for i in parse_number(args.site_number)] \
        if args.site_number else None
    frac_coords = False if args.cartesian else True
    if args.vector == "center":
        avg = [np.average([c[i] for c in struct.frac_coords]) for i in range(3)]
        if args.direction:
            v = avg[args.direction]
            avg = [0.5, 0.5]
            avg.insert(args.direction, v)
        if frac_coords:
            vector = [0.5 - i for i in avg]
        else:
            cart_avg = struct.lattice.get_cart_coords(avg)

            vector = [i / 2 for i in struct.lattice.abc] - cart_avg
    else:
        vector = list(map(float, args.vector.split(",")))
    struct.translate_sites(vector, indices=site_number, frac_coords=frac_coords)
    struct.to(filename="POSCAR")


def remove(args):
    f = args.poscar
    struct = Structure.from_file(f)
    if args.site_number:
        site_number = [i - 1 for i in parse_number(args.site_number)]
        struct.remove_sites(site_number)
    elif args.species:
        struct.remove_species(args.species)
    else:
        number, bt = re.search('(\d+)(\w)', args.layers).groups()
        for i in range(int(number)):
            struct.delete_bt_layer(bt, tol=args.tol, axis=args.direction)
    struct.to(filename="POSCAR")


def build_ferro(args):
    # Note that this is mainly used to calculate the cation-aniion displacement
    # in the film of a perovskite oxide HS (btbt). Two surface layers and all
    # substrate layers are excluded.
    files = args.poscar
    tol = args.tol if args.tol else 1.2
    reverse = True if args.reverse else False
    sub_uc = args.sub_uc if args.sub_uc else 5
    new_struct = []
    if args.opposite:
        for f in files:
            s = Structure.from_file(f)
            layers = s.sort_sites_in_layers(tol, reverse)
            new_sites = []
            for l in layers:
                group_a = []
                for k, g in groupby(l, key=lambda x: x.species_string):
                    group_a.append(list(g))
                d_coord = [group_a[1][0].frac_coords[2], group_a[0][0].frac_coords[2]]
                for j in range(len(group_a)):
                    for i in group_a[j]:
                        frac = i.frac_coords
                        frac[2] = d_coord[j]
                        new_sites.append(PeriodicSite(
                            i.species_string, frac, i.lattice,
                             properties=i.properties))
            new_struct.append(Structure.from_sites(new_sites))
    else:
        try:
            pa_struct, fe_struct = (Structure.from_file(f).get_layered_structure()
                                    for f in files)
            coord_diff = fe_struct.cart_coords - pa_struct.cart_coords
            coord_diff[:, :2] = 0.0
        except ValueError:
            raise ValueError("Must be two structure files.")

        if args.number:
            struct_num = args.number
            coord_gap = coord_diff / (struct_num + 1)
            for i in range(1, struct_num + 1):
                new_coord = pa_struct.cart_coords + coord_gap * i
                new_struct.append(Structure(
                    pa_struct.lattice, pa_struct.species, new_coord,
                    coords_cartesian=True, site_properties=pa_struct.site_properties))
        elif args.transplant:
            op_struct = Structure.from_file(args.transplant).get_layered_structure()
            new_coord = op_struct.cart_coords + coord_diff
            new_struct.append(Structure(
                op_struct.lattice, op_struct.species, new_coord,
                coords_cartesian=True, site_properties=op_struct.site_properties))
        elif args.enhance:
            percent = args.enhance / 100
            new_coord = fe_struct.cart_coords + coord_diff * percent
            new_struct.append(Structure(
                fe_struct.lattice, fe_struct.species, new_coord,
                coords_cartesian=True, site_properties=fe_struct.site_properties))

    for s in new_struct:
        new_s = s.get_sorted_structure()
        da, db = oxi_displace(new_s, sub_uc, tol, reverse)
        name = float_filename(db)
        filename = "POSCAR_" + name
        new_s.to(filename=filename)
        print("'%s' is created" % filename)
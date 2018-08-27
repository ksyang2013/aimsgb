from __future__ import division

import os
import sys
import math
import glob
import numpy as np
from numpy.linalg import norm
from collections import OrderedDict

from aimsflow import Structure, Composition, Specie, eff_radii, bd_val, born
from aimsflow.util import file_to_lines, str_to_file, parse_number, str_delimited
from aimsflow.elect_struct.core import Orbital, OrbitalType
from aimsflow.vasp_io import Eigenval, parse_outcar


def tolerance_factor(args):
    def get_parameter(a, v_a, b, v_b, const=0.37, ref="b"):
        avail = bd_val[a][v_a][b][v_b][const]
        if ref in avail:
            return bd_val[a][v_a][b][v_b][const][ref]
        else:
            for i in ["b", "bj", "bs", "a", "al", "e", "ae", "ah"]:
                if i in avail:
                    return bd_val[a][v_a][b][v_b][const][i]

        try:
            return bd_val[a][v_a][b][v_b][const][ref]
        except KeyError:
            return bd_val[a][v_a][b][v_b][const]["a"]

    if args.file:
        data = file_to_lines(args.file)
        comps = []
        for i in data:
            comps.append(i.split()[0])
    else:
        comps = args.compound

    outs = []
    for c in comps:
        comp = Composition(c)
        ele = [str(i) for i in comp.elements]
        if len(ele) == 2:
            ele.insert(0, ele[0])
        val = [float(i) for i in args.valence.split("-")]
        val.append(-2.0)
        coord = [12, 6, 6]

        r = []
        if args.mode == "cry":
            for i in range(3):
                radius = Specie(ele[i], val[i]).ionic_radius
                if radius:
                    r.append(radius)
                else:
                    r = None
                    break
            if r:
                d_ao = r[0] + r[2]
                d_bo = r[1] + r[2]
                t = "\t".join("%.2f" % i for i in [d_ao / math.sqrt(2) / d_bo,
                                                   r[1] / r[2]])
            else:
                t = "No data"
        elif args.mode == "eff":
            for i in range(3):
                try:
                    r.append(eff_radii[ele[i]][val[i]][coord[i]])
                except KeyError:
                    r = None
                    break
            if r:
                d_ao = r[0] + r[2]
                d_bo = r[1] + r[2]
                t = "\t".join("%.2f" % i for i in [d_ao / math.sqrt(2) / d_bo,
                                                   r[1] / r[2]])
            else:
                t = "No data"
        else:
            try:
                p_ao = get_parameter(ele[0], val[0], ele[2], val[2])
                d_ao = p_ao - 0.37 * math.log(val[0] / coord[0])
                p_bo = get_parameter(ele[1], val[1], ele[2], val[2])
                d_bo = p_bo - 0.37 * math.log(val[1] / coord[1])
                t = "%.2f" % (d_ao / math.sqrt(2) / d_bo)
            except KeyError:
                t = "No data"

        outs.append("\t".join([c, t]))
    print("\n".join(outs))


def poly_fit(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    r_squared = ssreg / sstot
    return {"coeffs": coeffs,
            "r_squared": r_squared}


def emass(args):
    sb = None
    if args.specify_branch:
        try:
            sb = parse_number(args.specify_branch)
        except ValueError:
            sb = args.specify_branch
    num = args.number if args.number else None

    for f in args.source:
        path = os.path.dirname(os.path.abspath(f))
        v = Eigenval(f)
        kpoint_f = "KPOINTS.bands.bz2" if "bz2" in f else "KPOINTS"
        outcar_f = "OUTCAR.bands.bz2" if "bz2" in f else "OUTCAR"
        band = v.get_band_structure("%s/%s" % (path, kpoint_f),
                                    "%s/%s" % (path, outcar_f))
        bm = band.get_cbm() if args.type == "electron" else band.get_vbm()
        efermi = band.efermi
        distance = band.distance
        for spin, bands in band.bands.items():
            bm_ind = bm[spin]["kpoint_index"]
            for n, branch in enumerate(band.branches):
                s = branch["start_index"]
                e = branch["end_index"]
                if sb:
                    if isinstance(sb, list):
                        if n + 1 not in sb:
                            continue
                    else:
                        if not any([s <= i <= e for i in bm_ind]):
                            continue
                b = bands[:, s:e + 1]
                inds = []
                if args.type == "electron":
                    max_tmp = float("inf")
                    for i, j in zip(*np.where(b > efermi)):
                        if b[i, j] < max_tmp:
                            ind = i
                            max_tmp = float(b[i, j])
                else:
                    max_tmp = -float("inf")
                    for i, j in zip(*np.where(b < efermi)):
                        if b[i, j] > max_tmp:
                            ind = i
                            max_tmp = float(b[i, j])

                if band.is_metal()[spin]:
                    # if system is metallic, include all the bands that cross
                    # the fermi level (0 eV)
                    for i in range(ind - 1, -1, -1):
                        if all([j - efermi < 0 for j in b[i]]):
                            break
                        else:
                            inds.append(i)
                else:
                    inds.append(ind)

                for ind in inds:
                    bm_i = b[ind] / 27.21138505
                    dist_i = np.multiply(distance[s:e + 1], 0.529177108)
                    if bm_i[0] > bm_i[-1]:
                        dist_i = dist_i - dist_i[-1]
                        dist_i = np.concatenate((dist_i, -dist_i[-2::-1]))
                        bm_i = np.concatenate((bm_i, bm_i[-2::-1]))
                    else:
                        dist_i = dist_i - dist_i[0]
                        dist_i = np.concatenate((-dist_i[:0:-1], dist_i))
                        bm_i = np.concatenate((bm_i[:0:-1], bm_i))
                    num_pair = len(dist_i) // 2
                    if num:
                        if num > num_pair:
                            raise ValueError("Please enter a number smaller "
                                             "than %s!" % num_pair)
                        dist_i = dist_i[num_pair - num:num_pair + num + 1]
                        bm_i = bm_i[num_pair - num:num_pair + num + 1]
                    first_fit = poly_fit(dist_i, bm_i, 8)
                    z = first_fit["coeffs"]
                    if first_fit["r_squared"] < 0.998:
                        bm_i[:num_pair + 1] = bm_i[num_pair::-1]
                        bm_i[num_pair + 1:-1] = bm_i[-2:num_pair:-1]
                        bm_i[-1] = bm_i[0]
                        second_fit = poly_fit(dist_i, bm_i, 8)
                        if first_fit["r_squared"] < second_fit["r_squared"]:
                            z = second_fit["coeffs"]
                    effmass = 0.5 / z[6]
                    out = "%s: %.2f" % (branch["name"].replace("\\Gamma", "G"),
                                        effmass)
                    print(out)


def interface_dist(args):
    files = args.poscar
    for f in files:
        struct = Structure.from_file(f)
        layer_info = struct.get_layer_info(args.nlayers, args.tol)
        if_dist = ["%.4f" % i for i in layer_info["if_dist"]]
        if_name = layer_info["if_name"]
        print("%s:\t%s" % (f, "\t".join([": ".join(i)
                                         for i in zip(if_name, if_dist)])))


def if_bond_len(args):
    files = args.poscar
    for f in files:
        struct = Structure.from_file(f)
        layer_info = struct.get_layer_info(args.nlayers, args.tol)
        if_name = layer_info["if_name"]
        all_dist = []
        for i in layer_info["if_ind"]:
            l1, l2 = layer_info["layers"][i[0]], layer_info["layers"][i[1]]
            name_dist = []
            for s1 in l1:
                for s2 in l2:
                    # np.mod is because fractional ab=1 and ab=0 are the same
                    if np.allclose(np.mod(s1.frac_coords[:2], 1),
                                   np.mod(s2.frac_coords[:2], 1)):
                        bond_name = "-".join([s1.species_string,
                                              s2.species_string])
                        if s2.coords[2] < s1.coords[2]:
                            bond_dist = s2.coords[2] - s1.coords[2] + \
                                        struct.lattice.abc[2]
                        else:
                            bond_dist = s2.coords[2] - s1.coords[2]
                        name_dist.append("%s: %.4f" % (bond_name, bond_dist))
            if not name_dist:
                name_dist.append("No direct bond")
            all_dist.append("\t".join(name_dist))
        print("%s:\t%s" % (f, "\t".join([": ".join(i)
                                         for i in zip(if_name, all_dist)])))


def ki(args):
    dirs = args.directories
    inter_num = args.interface_number
    dir_number = len(dirs)
    if dir_number % 2 == 1:
        raise IOError("The number of input directories (%s) is not an "
                      "even number." % dir_number)
    path1, path2 = [dirs[:dir_number // 2], dirs[dir_number // 2:]]
    path_pairs = zip(path1, path2)
    for p1, p2 in path_pairs:
        out1 = parse_outcar(p1, parse_mag=True)
        out2 = parse_outcar(p2, parse_mag=True)
        if out1 is None or out2 is None:
            continue
        t1 = out1.total_energy
        t2 = out2.total_energy
        try:
            f = sorted(glob.glob("%s/POSCAR*" % p1))[-1]
        except IndexError:
            raise IOError("No POSCAR file in %s" % p1)
        struct = Structure.from_file(f)
        area = struct.lattice.abc[0] * struct.lattice.abc[1]
        try:
            const = 1.6021773e4 / (inter_num * area)
            tot_ediff = t1 - t2
            tot_ki = tot_ediff * const  # mJ/m2
            print("%s-%s:\tdE = %.3f meV\tKi = %.3f"
                  % (p1, p2, tot_ediff * 1000, tot_ki))
            if args.verbose:
                ion_ediff = np.subtract(out1.e_soc, out2.e_soc)
                ion_ediff *= tot_ediff / sum(ion_ediff)
                ion_ki = ion_ediff * const * inter_num
                for i, v in enumerate(ion_ki, 1):
                    print("Ion%s:\tdE = %.3f \tKi = %.3f"
                          % (i, ion_ediff[i - 1] * 1000, v))
        except TypeError:
            pass


def orb_ki(args):
    mat_str = {'p': ['p_y', 'p_z', 'p_x'],
               'd': ['d_{xy}', 'd_{yz}', 'd_{z^2}', 'd_{xz}', 'd_{x^2-y^2}']}
    p1, p2 = args.directories
    site_num = [i - 1 for i in parse_number(args.site_number)]
    orb = OrbitalType[args.orbital]
    if orb == 'f':
        orb_str = [i.name for i in Orbital if i.orbital_type == orb]
    else:
        orb_str = mat_str[orb.name]
    inter_num = args.interface_number
    out1 = parse_outcar(p1, parse_mag=True)
    out2 = parse_outcar(p2, parse_mag=True)
    t1 = out1.total_energy
    t2 = out2.total_energy
    orb_soc1 = out1.orb_soc
    orb_soc2 = out2.orb_soc
    try:
        f = sorted(glob.glob("%s/POSCAR*" % p1))[-1]
    except IndexError:
        raise IOError("No POSCAR file in %s" % p1)
    struct = Structure.from_file(f)
    area = struct.lattice.abc[0] * struct.lattice.abc[1]
    const = 1.6021773e4 / (inter_num * area)
    ratio = (t1 - t2) / sum(np.subtract(out1.e_soc, out2.e_soc))
    for i in site_num:
        s1 = orb_soc1[i][orb]
        s2 = orb_soc2[i][orb]
        outs = np.subtract(s1, s2) * ratio * const * inter_num
        outs = [["%.6f" % j for j in k] for k in outs]
        filename = "%s_%s_%s.dat" % (i + 1, struct[i].specie.symbol, orb.name)
        str_to_file(str_delimited(outs, header=orb_str), filename)


def berry_phase_polarization(args):
    paths = args.paths
    outcars = [parse_outcar(p) for p in paths]
    structs = [Structure.from_file("%s/CONTCAR" % p) for p in paths]
    pa_o = outcars[0]
    for i in range(1, len(outcars)):
        v = structs[i].lattice.volume
        fe_o = outcars[i]
        p_ion = fe_o.p_ion - pa_o.p_ion
        p_elc = fe_o.p_elc - pa_o.p_elc
        polz = norm(p_ion + p_elc) / v * 1.6e3
        print("%s: %.3f" % (paths[i], polz))


def polarization(args):
    files = args.poscar
    tol = args.tol if args.tol else 1.2
    reverse = True if args.reverse else False
    sub_uc = args.sub_uc if args.sub_uc else 0
    outs = abo3_polarize(files, tol, sub_uc, reverse)
    for k, v in outs.items():
        print("%s: %.2f" % (k, v))


def abo3_polarize(files, tol=1.2, sub_uc=0, reverse=False):
    all_s = [Structure.from_file(f) for f in files]
    num = len(all_s)
    pa_struct = all_s[0]
    pa_layer = pa_struct.sort_sites_in_layers(tol, reverse)[2 * sub_uc:]
    layer_num = len(pa_layer)
    assert layer_num % 2 == 0
    num_uc = layer_num // 2
    pa_layer = [np.concatenate((i[0], i[1]))
                for i in np.array(pa_layer).reshape(num_uc, 2)]
    if len(pa_layer[0]) % 5 != 0:
        raise RuntimeError("Please make sure the structure is ABO3 and with an "
                           "appropriate tol (current=%s" % tol)
    bulk = False
    if len(pa_layer) == 1:
        uc_num = 2
        bulk = True
    else:
        uc_num = len(pa_layer)

    u = len(pa_layer[0]) // 5
    comp = pa_layer[0]
    a = comp[0 * u].species_string
    o = comp[1 * u].species_string
    b = comp[2 * u].species_string
    name = a + b + o + "3"
    za = born[name]["AO"][a]
    zo1 = born[name]["AO"][o]
    zb = born[name]["BO"][b]
    zo2 = born[name]["BO"][o]
    outs = OrderedDict()
    for n in range(1, num):
        fe_struct = all_s[n]
        fe_layer = fe_struct.sort_sites_in_layers(tol, reverse)[2 * sub_uc:]
        if len(fe_layer) != layer_num:
            sys.stderr.write("Layer number in %s is different from %s. Try "
                             "increasing tol (current: %s)\n"
                             % (files[n], files[0], tol))
            continue
        fe_layer = [np.concatenate((i[0], i[1]))
                    for i in np.array(fe_layer).reshape(num_uc, 2)]

        p = 0
        for i in range(uc_num - 1):
            ba = 0
            bo1 = 0
            bb = 0
            bo2 = 0
            for j in range(u):
                ba += (pa_layer[i][0 * u + j].coords[2] - fe_layer[i][0 * u + j].coords[2]) * za
                bo1 += (pa_layer[i][1 * u + j].coords[2] - fe_layer[i][1 * u + j].coords[2]) * zo1
                bb += (pa_layer[i][2 * u + j].coords[2] - fe_layer[i][2 * u + j].coords[2]) * zb
                bo2 += (pa_layer[i][3 * u + j].coords[2] - fe_layer[i][3 * u + j].coords[2]) * zo2
            if uc_num == 2 and bulk:
                v = fe_struct.lattice.volume
            else:
                v = (fe_layer[i + 1][0].coords[2] - fe_layer[i][0].coords[2]) \
                    * fe_struct.lattice.abc[0] * fe_struct.lattice.abc[1]
            p += 1.6e3 / v * (ba + bo1 + bb + 2 * bo2)
        p /= uc_num - 1
        outs[files[n]] = p
    return outs


def displace(args):
    files = args.poscar
    tol = args.tol if args.tol else 0.25
    reverse = True if args.reverse else False
    sub_uc = args.sub_uc if args.sub_uc else 0
    for f in files:
        try:
            s = Structure.from_file(f)
        except IndexError:
            sys.stderr.write("Structure file error in %s\n" % f)
            continue
        da, db = oxi_displace(s, sub_uc, tol, reverse)
        print("%s\t%.2f\t%.2f" % (f, da, db))


def oxi_displace(struct, sub_uc, tol, reverse=False):
    layers = struct.sort_sites_in_layers(tol, reverse)[2 * sub_uc:-2]
    disp_all = []
    for l in layers:
        if l[0].species_string != "O":
            disp_all.append(l[-1].coords[2] - l[0].coords[2])
        else:
            disp_all.append(l[0].coords[2] - l[-1].coords[2])
    disp_a = []
    disp_b = []
    for i in range(len(disp_all)):
        if abs(disp_all[i]) > 1e-4:
            if i % 2 == 0:
                disp_a.append(disp_all[i])
            else:
                disp_b.append(disp_all[i])
    da = np.average(disp_a) if len(disp_a) > 0 else 0.0
    db = np.average(disp_b) if len(disp_b) > 0 else 0.0
    return da, db

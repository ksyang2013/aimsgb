import os
from fnmatch import fnmatch
from collections import OrderedDict

from aimsflow import Structure, Composition
from aimsflow.util import immed_files, parse_number
from aimsflow.elect_struct.plotter import DosPlotter, BSPlotter, \
    BSProjPlotter, LocpotPlotter
from aimsflow.vasp_io import Vasprun, Doscar, Eigenval, Procar, Locpot


def plot_band(args):
    f = args.source
    path = os.path.dirname(os.path.abspath(f))
    folders = path.split("/")
    name = "%s_%s" % (folders[-2], folders[-1])
    v = parse_source(f)
    kpoint_file = "KPOINTS.bands.bz2" if "bz2" in f else "KPOINTS"
    outcar_file = "OUTCAR.bands.bz2" if "bz2" in f else "OUTCAR"
    band = v.get_band_structure("%s/%s" % (path, kpoint_file),
                                "%s/%s" % (path, outcar_file))
    plt = BSPlotter(band)
    is_separate = True if args.separate else False
    plt.get_plot(name, xlim=args.xlim, ylim=args.ylim, is_separate=is_separate)


def plot_band_pro(args):
    f = os.path.join(args.directory, args.source)
    path = os.path.dirname(os.path.abspath(f))
    folders = path.split("/")
    name = "%s_%s" % (folders[-2], folders[-1])
    if args.site_number:
        ion_spec = args.site_number - 1
        name += "_%s" % args.site_number
    else:
        ion_spec = None
    v = Procar(f, ion_spec)
    orb_spec = args.specify_orbit or None
    kpoint_file = "KPOINTS.bands.bz2" if "bz2" in f else "KPOINTS"
    outcar_file = "OUTCAR.bands.bz2" if "bz2" in f else "OUTCAR"
    band = v.get_band_structure("%s/%s" % (path, kpoint_file),
                                "%s/%s" % (path, outcar_file), orb_spec=orb_spec)
    plt = BSProjPlotter(band)
    plt.get_proj_plot(name, max_ps=args.point_size,
                      xlim=args.xlim, ylim=args.ylim)


def plot_tdos(args):
    f = os.path.join(args.directory, args.source)
    v = parse_source(f)
    xlim = args.xlim
    ylim = args.ylim
    path = os.path.dirname(os.path.abspath(f))
    folders = path.split("/")

    all_dos = OrderedDict()
    dos = v.complete_dos
    if args.split_type == "spdf":
        all_dos = dos.get_tdos_spdf()
    elif args.split_type == "t2g_eg":
        all_dos = dos.get_tdos_t2g_eg()
    elif args.split_type == "combine":
        all_dos["TDOS"] = v.tdos
    plotter = DosPlotter()
    plotter.add_dos_dict(all_dos)
    sys_name = "TDOS_%s_%s" % (folders[-2], folders[-1])
    is_separate = True if args.separate else False
    plotter.get_plot(sys_name, xlim, ylim, is_separate=is_separate)


def plot_pdos(args):
    f = os.path.join(args.directory, args.source)
    v = parse_source(f)
    xlim = args.xlim
    ylim = args.ylim
    path = os.path.dirname(os.path.abspath(f))

    all_dos = OrderedDict()
    site_num = [i - 1 for i in parse_number(args.site_number)]
    dos = v.complete_dos
    for f in immed_files(path):
        if fnmatch(f, "*POSCAR*") or fnmatch(f, "*CONTCAR*"):
            struct = Structure.from_file(f)
            break

    is_separate = True if args.separate else False
    for ind in site_num:
        try:
            name = "%s-%s" % (ind + 1, struct[ind].specie.symbol)
        except UnboundLocalError:
            name = ind + 1
        if args.split_type == "combine":
            all_dos = dos.get_pdos_combine(ind)
        elif args.split_type == "full":
            all_dos = dos.get_pdos_full(ind)
        elif args.split_type == "spdf":
            all_dos = dos.get_pdos_spdf(ind)
        elif args.split_type == "t2g_eg":
            all_dos = dos.get_pdos_t2g_eg(ind)
        elif args.split_type == "pxyz":
            all_dos = dos.get_pdos_pxyz(ind)
        plotter = DosPlotter()
        plotter.add_dos_dict(all_dos)
        plotter.get_plot("PDOS_%s" % name, xlim, ylim, is_separate=is_separate)


def plot_ldos(args):
    struct = Structure.from_file(args.poscar)
    v = parse_source(args.source)
    tol = args.tol if args.tol else 1.0
    reverse = True if args.reverse else False
    xlim = args.xlim
    ylim = args.ylim

    dos = v.complete_dos
    layers = struct.sort_sites_in_layers(tol=tol, reverse=reverse)
    for n, l in enumerate(layers):
        site_nums = []
        site_names = []
        for s in l:
            for i in range(struct.num_sites):
                if s == struct[i]:
                    site_nums.append(i)
                    site_names.append(s.species_string)
                    break
        all_dos = dos.get_sites_pdos_spdf(site_nums)
        plotter = DosPlotter()
        plotter.add_dos_dict(all_dos)
        layer_name = Composition("".join(site_names)).reduced_formula
        sys_name = "PDOS_layer%s_%s" % (n, layer_name)
        plotter.get_plot(sys_name, xlim, ylim)


def plot_locpot(args):
    v = parse_source(args.source)
    path = os.path.abspath(args.directory)
    folders = path.split("/")
    sys_name = "%s_%s" % (folders[-2], folders[-1])
    all_pot, all_z = v.plan_average
    marc_pot, marc_z = v.marc_average(args.nlayers)
    pot_dict = {}
    for spin in all_pot:
        pot_dict[spin] = {"all_pot": all_pot[spin], "all_z": all_z,
                          "marc_pot": marc_pot[spin], "marc_z": marc_z[spin]}
    plotter = LocpotPlotter(pot_dict, v.structure, args.nlayers, args.tol)
    plotter.get_plot(sys_name)


def parse_source(filename):
    if fnmatch(filename, "*DOSCAR*"):
        return Doscar(filename)
    elif fnmatch(filename, "*vasprun.xml*"):
        return Vasprun(filename, parse_dos=True)
    elif fnmatch(filename, "*EIGENVAL*"):
        return Eigenval(filename)
    elif fnmatch(filename, "*LOCPOT*"):
        return Locpot(filename)
    else:
        raise IOError("The source file should be 'DOSCAR', 'vasprun.xml' "
                      "LOCPOT or EIGENVAL.")
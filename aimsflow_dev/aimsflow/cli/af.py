#!/usr/bin/env python
from __future__ import division

import sys
import json
import urllib
import argparse
import numpy as np
try:
    from urllib.request import urlopen, urlretrieve
except ImportError:
    from urllib import urlopen, urlretrieve


from aimsflow import Structure
from aimsflow.symmetry.bandstructure import HighSymmKpath
from aimsflow.vasp_io import BatchFile, Poscar, Potcar, Kpoints
from aimsflow.cli.af_config import configure_af
from aimsflow.cli.af_jobflow import vasp, analyze, jyaml
from aimsflow.cli.af_calculate import emass, tolerance_factor, interface_dist,\
    if_bond_len, ki, orb_ki, displace, polarization, berry_phase_polarization
from aimsflow.cli.af_plot import plot_tdos, plot_pdos, plot_ldos, plot_band,\
    plot_band_pro, plot_locpot
from aimsflow.cli.af_build import build_hs, build_strain, build_if, build_ferro,\
    build_sc, build_rotate, build_slab, translate_sites, remove
from aimsflow.util import str_to_file, file_to_str, Citation


def batch(args):
    all_f = args.file
    if not isinstance(all_f, list):
        all_f = [all_f]

    for f in all_f:
        try:
            script = BatchFile.from_file(f)
        except IOError:
            sys.stderr.write("No '%s' file is found\n" % f)
            continue

        if args.queue:
            script.change_queue(args.queue)

        if args.walltime:
            script.change_walltime(args.walltime)

        if args.processors:
            script.change_processors(args.processors)

        if args.jobname:
            script.change_jobname(args.jobname)

        if args.mail_type:
            script.change_mail_type(args.mail_type)

        if args.exe:
            script.change_exe(args.exe)

        if args.convert:
            script = BatchFile.convert_batch(script)

        script.write_file(f)


def combine_dos(args):
    files = args.data_file
    out_file = args.output
    skip = 0
    try:
        data = np.loadtxt(files[0], skiprows=skip)
    except ValueError:
        skip = 1
        data = np.loadtxt(files[0], skiprows=skip)
    energy = data[:, 0]
    dos_matrix = data[:, 1:]
    for f in files[1:]:
        dos_matrix += np.loadtxt(f, skiprows=skip)[:, 1:]
    dos_sum = np.c_[energy[:, None], dos_matrix]
    outs = '\n'.join(['\t'.join('%.6f' % i for i in row) for row in dos_sum])
    str_to_file(outs, out_file)


def potcar(args):
    if args.poscar:
        poscar = Poscar.from_string(file_to_str(args.poscar))
        elements = poscar.site_symbols
    elif args.symbol:
        elements = args.symbol
    pot = Potcar.from_elements(elements, args.functional)
    pot.write_file('POTCAR')


def format_citation(args):
    cite_files = args.files
    for f in cite_files:
        print(Citation(f))


def kpoints(args):
    s = Structure.from_file(args.poscar)
    if args.band:
        num_kpts = args.band
        k = HighSymmKpath(s)
        frac, labels = k.get_kpoints()
        k_path = ' '.join(['-'.join(i) for i in k.kpath['path']])
        comment = 'aimsflow generated KPOINTS for %s: %s' % \
                  (k.name, k_path.replace('\\Gamma', 'G'))
        kpt = Kpoints(comment=comment, num_kpts=num_kpts,
                      style=Kpoints.supported_modes.Line_mode,
                      kpts=frac, labels=labels, coord_type='Reciprocal')
    else:
        density = args.relax if args.relax else args.static
        kpt = Kpoints.automatic_density_by_vol(s, density)
    print(kpt)


def query_aflow(args):
    for url in args.aurl:
        query(url, args.values, args.files)


def query(aurl, values=None, files=None):
    aurl = 'http://aflowlib.duke.edu/AFLOWDATA/' + aurl + '/'
    entry = json.loads(urlopen(aurl + '?format=json').read().decode('utf-8'))
    if values:
        if values == ['all']:
            values = entry.keys()
        for v in values:
            print('%s = %s' % (v, entry[v]))
    if files:
        for f in files:
            try:
                urlretrieve(aurl + f, f)
            except urllib.error.HTTPError:
                raise RuntimeError("Cannot retrieve '%s'" % f)
            print("Retrieved '%s'" % f)


def main():
    parser = argparse.ArgumentParser(
        description='aimsflow is an efficient Python package for VASP job '
                    'management and calculation analysis. This script works '
                    'based on several sub-commands with their own options. '
                    'To see the options for thes sub-commands, type '
                    'aimsflow sub-command -h',
        formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers()
    dir_kwargs = {'metavar': 'dir', 'default': '.', 'type': str,
                  'help': 'Working directory (default: %(default)s)'}
    source_kwargs = {'default': 'DOSCAR', 'type': str, 'nargs': '?',
                     'help': "Source file is either 'DOSCAR' or 'vasprun.xml' "
                             "(default: %(default)s)"}
    ed_kwargs = {'type': float, 'default': 0.0, 'nargs': '?',
                 'help': 'Add or delete extra distance at interface.'}
    dl_kwargs = {'type': str, 'nargs': '?',
                 'help': 'Set delete top or bottom layers. (default: %(default)s)'}
    ps_kwargs = {'type': str, 'help': 'Structure file'}
    pss_kwargs = {'nargs': '+', 'type': str,
                  'help': 'Structure files'}
    jt_kwargs = {'type': str, 'nargs': '?', 'choices': ['r', 's', 'b', 'e'],
                 'const': 's'}

    sn_args = ['-sn', '--site_number']
    d_args = ['-d', '--direction']
    vt_args = ['-v', '--vacuum']
    vt_kwargs = {'type': float, 'default': 0.0, 'const': 0.0, 'nargs': '?',
                 'help': 'Set vacuum thickness for slab.'}
    xlim_args = ['-x', '--xlim']
    xlim_kwargs = {'default': [-4, 4], 'type': float, 'nargs': 2,
                   'help': 'Set limit for x-axis. (default: %(default)s)'}
    ylim_args = ['-y', '--ylim']
    ylim_kwargs = {'default': None, 'type': float, 'nargs': 2,
                   'help': 'Set limit for y-axis. (default: %(default)s)'}
    tol_args = ['-t', '--tol']
    tol_kwargs = {'type': float, 'default': 0.25, 'const': 0.25, 'nargs': '?',
                  'help': 'Tolerance factor to determine if two atoms are at '
                          'the same plane. (default: %(default)s)'}
    rev_args = ['-r', '--reverse']
    rev_kwargs = {'action': 'store_true', 'help': 'When sorting atoms in c-axis, '
                                                  'whether in descendant order.'}
    nl_args = ['-n', '--nlayers']
    nl_kwargs = {'type': int, 'default': 1,
                 'help': 'Number of layers in one unit cell. '
                         '(default: %(default)s)'}

    parser_config = subparsers.add_parser('config', help='Configuring aimsflow.',
                                          formatter_class=argparse.RawTextHelpFormatter)
    parser_config.add_argument('-a', '--add', dest='var_spec', nargs='+',
                               help='Variables to add in the form of space '
                                    'separated key value pairs.\nE.g. aimsflow '
                                    'config -a AF_VASP_PSP_DIR ~/psps')
    parser_config.set_defaults(func=configure_af)

    parser_vasp_job = subparsers.add_parser(
        'vasp', help='Prepare, submit clean, kill and continue VASP jobs.',
        formatter_class=argparse.RawTextHelpFormatter)
    parser_vasp_job.add_argument('directories', nargs='*', **dir_kwargs)
    group = parser_vasp_job.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--prepare',
                       choices=['r', 's', 'b', 'p', 'e', 'm', 'rs', 'sb', 'rsb'],
                       help='Prepare VASP files and folders according to different '
                            'types of calculation.\nE.g. aimsflow vasp -p rs')
    group.add_argument('-s', '--submit',
                       choices=['r', 's', 'b', 'p', 'e', 'rs', 'sb', 'rsb'],
                       help='Job type to be submitted.\nE.g. aimsflow vasp -s rs')
    group.add_argument('-c', '--clean', action='store_true',
                       help='Clean VASP files to only submission files.\n'
                            'E.g. aimsflow vasp -c')
    group.add_argument('-k', '--kill', action='store_true',
                       help="Kill submitted VASP job in 'ID_list'.\n"
                            "E.g. aimsflow vasp -k")
    group.add_argument('-cj', '--continue_job',
                       choices=['r', 's', 'b', 'p', 'e', 'm', 'rs', 'sb'],
                       help='Continue unfinished VASP calculations.\n'
                            'E.g. aimsflow vasp -cj r')
    group.add_argument('-aj', '--add_job', choices=['s', 'b', 'p', 'e'],
                       help='Add a continuous job based on the finished job.\n'
                            'E.g. aimsflow vasp -aj s')
    parser_vasp_job.add_argument('-df', '--directory_file', type=str,
                                 help='Working directories given by a file\n'
                                      'E.g. aimsflow vasp -c -df dirs.txt')
    parser_vasp_job.add_argument('-e', '--eint', metavar='EINT',
                                 type=float, nargs='+',
                                 help='Specify the energy range of the bands '
                                      'that are used for PARCHG calculation.\n'
                                      'E.g. aimsflow vasp -aj p -e -1.0 0.5')
    parser_vasp_job.add_argument('-f', '--functional', default='PBE', nargs='?',
                                 choices=['PBE', 'PBE_52', 'PBE_54', 'LDA',
                                          'PW91', 'LDA_US'], const='PBE',
                                 help='Set the functional for pseudopotential. '
                                      '(default: %(default)s)\n'
                                      'E.g. aimsflow vasp -p rs -f PBE_52')
    parser_vasp_job.set_defaults(func=vasp)

    parser_analyze = subparsers.add_parser(
        'analyze', help='Analyze VASP job', formatter_class=argparse.RawTextHelpFormatter)
    parser_analyze.add_argument('directories', nargs='*', **dir_kwargs)
    parser_analyze.add_argument('-df', '--directory_file', type=str,
                                help='Working directories given by a file\n'
                                     'E.g. aimsflow analyze -cv -df dirs.txt')
    parser_analyze.add_argument('-cv', '--converge', action='store_true',
                                help='Check whether calculation is converged\n'
                                     'E.g. aimsflow analyze -cv')
    parser_analyze.add_argument('-cc', '--collect_contcar', metavar='dest_folder',
                                type=str, help='Collect all the CONTCARS\nE.g. '
                                               'aimsflow analyze -cc contcars')
    parser_analyze.add_argument('-cu', '--collect_unconverge', metavar='dest_folder',
                                type=str, help='Collect all the unconverged '
                                               'folders\nE.g. aimsflow analyze '
                                               '-cu unconverge')
    parser_analyze.add_argument('-gt', '--get_toten',
                                help='Get total energy for a certain job type '
                                     '(default: s)\nE.g. aimsflow analyze -gt r',
                                **jt_kwargs)
    parser_analyze.add_argument('-gm', '--get_mag',
                                help='Get magnetization for a certain job type '
                                     '(default: s) along certain direction '
                                     '(default: z)\nE.g. aimsflow analyze -gm r -d x',
                                **jt_kwargs)
    parser_analyze.add_argument('-gom', '--get_orb_mag',
                                help='Get orbital moment of a certain job type '
                                     '(default: s) along certain direction '
                                     '(default: z)\nE.g. aimsflow analyze -gom r -d x',
                                **jt_kwargs)
    parser_analyze.add_argument('-gbc', '--get_born_charge', action='store_true',
                                help='Get born charge along certain direction '
                                     '(default: z)\nE.g. aimsflow analyze -gbc -d x')
    parser_analyze.add_argument('-gbg', '--get_band_gap', nargs='?', const='DOSCAR',
                                choices=['DOSCAR', 'EIGENVAL', 'PROCAR'],
                                help='Get the band gap from band calculation in '
                                     'either EIGENVAL or PROCAR (default: '
                                     'EIGENVAL)\nE.g. aimsflow analyze -gbg')
    parser_analyze.add_argument(*d_args, type=str, nargs='?', const='z',
                                choices=['x', 'y', 'z'], default='z',
                                help='Get results along certain direction '
                                     '(default: %(default)s)\nE.g. aimsflow '
                                     'analyze -gm -d x')
    parser_analyze.add_argument(*sn_args, type=str,
                                help='Site number\nE.g. aimsflow analyze '
                                     '-gm -sn 1,2-5')
    parser_analyze.set_defaults(func=analyze)

    parser_batch = subparsers.add_parser(
        'batch', help='Change batch script setting.', formatter_class=argparse.RawTextHelpFormatter)
    parser_batch.add_argument('file', default='runscript.sh', type=str,
                              nargs='*', help='Batch file for editing. '
                                              '(default: %(default)s)')
    parser_batch.add_argument('-q', '--queue', metavar='queue',
                              type=str, choices=['hotel', 'home', 'condo', 'glean'],
                              help='Change the queue name (PBS only)\n'
                                   'E.g. aimsflow batch runscript.sh -q condo')
    parser_batch.add_argument('-t', '--walltime', metavar='walltime', type=str,
                              help='Change job walltime\n'
                                   'E.g. aimsflow batch runscript.sh -t 8')
    parser_batch.add_argument('-p', '--processors', metavar='processors', type=int,
                              help='Change number processors per node\n'
                                   'E.g. aimsflow batch runscript.sh -p 28')
    parser_batch.add_argument('-n', '--jobname', metavar='jobname', type=str,
                              help='Change the job name\n'
                                   'E.g. aimsflow batch runscript.sh -n relax')
    parser_batch.add_argument('-m', '--mail_type', metavar='matil_type', type=str,
                              help='Change the mail type\n'
                                   'E.g. aimsflow batch runscript.sh -m abe')
    parser_batch.add_argument('-e', '--exe', metavar='exe', type=str,
                              help='Change the VASP executable file\nE.g. '
                                   'aimsflow batch runscript.sh -e mpivasp54s.rlxZ')
    parser_batch.add_argument('-c', '--convert', action='store_true',
                              help='Convert batch from PBS to SLURM or vice versa\n'
                                   'E.g. aimsflow batch runscript.sh -c')
    parser_batch.set_defaults(func=batch)

    parser_plot = subparsers.add_parser(
        'plot', help='Plot DOS, band, electrostatic potential.')
    plot_subparser = parser_plot.add_subparsers(title='action', dest='action command')

    tdos_parser = plot_subparser.add_parser(
        'tdos', help='Total DOS', formatter_class=argparse.RawTextHelpFormatter)
    tdos_parser.add_argument('directory', nargs='?', **dir_kwargs)
    tdos_parser.add_argument('source', **source_kwargs)
    tdos_parser.add_argument('-st', '--split_type', default='combine', nargs='?',
                             choices=['combine', 'full', 'spdf', 't2g_eg'],
                             help='Split type for orbitals (default: %(default)s)\n'
                                  'E.g. aimsflow plot tdos -st t2g_eg')
    tdos_parser.add_argument('-s', '--separate', action='store_true',
                             help='Plot spin up and down dos separately\n')
    tdos_parser.add_argument(*xlim_args, **xlim_kwargs)
    tdos_parser.add_argument(*ylim_args, **ylim_kwargs)
    tdos_parser.set_defaults(func=plot_tdos)

    pdos_parser = plot_subparser.add_parser(
        'pdos', help='Partial DOS', formatter_class=argparse.RawTextHelpFormatter)
    pdos_parser.add_argument('directory', nargs='?', **dir_kwargs)
    pdos_parser.add_argument('source', **source_kwargs)
    pdos_parser.add_argument(*sn_args, type=str, required=True,
                             help='Site number\nE.g. aimsflow plot pdos -sn 1,3-5')
    pdos_parser.add_argument('-st', '--split_type', default='spdf', nargs='?',
                             choices=['combine', 'full', 'spdf', 't2g_eg', 'pxyz'],
                             help='Split type for orbitals (default: %(default)s)\n'
                                  'E.g. aimsflow plot pdos -sn 1,3-5 -st t2g_eg')
    pdos_parser.add_argument('-s', '--separate', action='store_true',
                             help='Plot spin up and down dos separately\n')
    pdos_parser.add_argument(*xlim_args, **xlim_kwargs)
    pdos_parser.add_argument(*ylim_args, **ylim_kwargs)
    pdos_parser.set_defaults(func=plot_pdos)

    ldos_parser = plot_subparser.add_parser(
        'ldos', help='Layered DOS', formatter_class=argparse.RawTextHelpFormatter)
    ldos_parser.add_argument('poscar', default='POSCAR', nargs='?', **ps_kwargs)
    ldos_parser.add_argument('source', **source_kwargs)
    ldos_parser.add_argument(*tol_args, **tol_kwargs)
    ldos_parser.add_argument(*rev_args, **rev_kwargs)
    ldos_parser.add_argument(*xlim_args, **xlim_kwargs)
    ldos_parser.add_argument(*ylim_args, **ylim_kwargs)
    ldos_parser.set_defaults(func=plot_ldos)

    band_parser = plot_subparser.add_parser(
        'band', help='Band structure', formatter_class=argparse.RawTextHelpFormatter)
    band_parser.add_argument('directory', nargs='?', **dir_kwargs)
    band_parser.add_argument('source', type=str, nargs='?', default='EIGENVAL',
                             help='Source file for band. (default: %(default)s)')
    band_parser.add_argument(*xlim_args, type=int, nargs=2, default=None,
                             help='set brance limit k-path. (default: %(default)s)')
    band_parser.add_argument(*ylim_args, **ylim_kwargs)
    band_parser.add_argument('-s', '--separate', action='store_true',
                             help='Plot spin up and down bands separately\n'
                                  'E.g. aimsflow plot band -s')
    band_parser.set_defaults(func=plot_band)

    band_pro_parser = plot_subparser.add_parser(
        'band_pro', help='Projected band structure', formatter_class=argparse.RawTextHelpFormatter)
    band_pro_parser.add_argument('directory', nargs='?', **dir_kwargs)
    band_pro_parser.add_argument('source', type=str, nargs='?', default='PROCAR',
                                 help='Source file for band. (default: %(default)s)')
    band_pro_parser.add_argument(*xlim_args, type=int, nargs=2, default=None,
                                 help='set brance limit k-path. '
                                      '(default: %(default)s)')
    band_pro_parser.add_argument(*ylim_args, **ylim_kwargs)
    band_pro_parser.add_argument('-so', '--specify_orbit',
                                 choices=['s', 'p', 'd'],
                                 help='Set the specific orbital to plot\n'
                                      'E.g. aimsflow plot band_pro -so d')
    band_pro_parser.add_argument(*sn_args, type=int,
                                 help='Set the specific site number to plot\n'
                                      'E.g. aimsflow plot band_pro -sn 1')
    band_pro_parser.add_argument('-ps', '--point_size', type=float, default=5.0,
                                 help='Set the maximum point size to plot '
                                      '(default: %(default)s)\n'
                                      'E.g. aimsflow plot band_pro -ps 4')
    band_pro_parser.set_defaults(func=plot_band_pro)

    locpot_parser = plot_subparser.add_parser(
        'locpot', help='Local potential', formatter_class=argparse.RawTextHelpFormatter)
    locpot_parser.add_argument('directory', nargs='?', **dir_kwargs)
    locpot_parser.add_argument('source', type=str, nargs='?', default='LOCPOT',
                               help='Source file for potential. (default: %(default)s)')
    locpot_parser.add_argument(*nl_args, **nl_kwargs)
    locpot_parser.add_argument(*tol_args, **tol_kwargs)
    locpot_parser.set_defaults(func=plot_locpot)

    parser_combine_dos = subparsers.add_parser(
        'combine_dos', help='Combine multiple PDOS files into single one',
        formatter_class=argparse.RawTextHelpFormatter)
    parser_combine_dos.add_argument('data_file', type=str, nargs='+',
                                    help='About to be combined PDOS files.')
    parser_combine_dos.add_argument(
        '-o', '--output', type=str, default='PDOS_sum_1.dat',
        help='Output PDOS data to file. (default name: %(default)s)\n'
             'E.g. aimsflow combine_dos PDOS1 PDOS2 -o PDOS_sum')
    parser_combine_dos.set_defaults(func=combine_dos)

    parser_job_yaml= subparsers.add_parser(
        'jyaml', help='Generate and modify job_flow.yaml for VASP calculation',
        formatter_class=argparse.RawTextHelpFormatter)
    parser_job_yaml.add_argument('directory', metavar='dir', default='.',
                                 type=str, nargs='?',
                                 help='Directory to work on the job_flow.yaml '
                                      '(default: %(default)s)')
    group = parser_job_yaml.add_mutually_exclusive_group()
    group.add_argument('-g', '--generate',
                       choices=['r', 's', 'b', 'p', 'e', 'm', 'rs', 'sb', 'rsb'],
                       help='Generate job_flow.yaml file based on the given '
                            'job type.\nE.g. aimsflow jyaml -g rs')
    group.add_argument('-up', '--update_poscar',
                       type=str, nargs='+', metavar='POSCARs',
                       help='POSCAR files that are used to update the '
                            'job_flow.yaml\nE.g. aimsflow jyaml -up POSCAR*')
    group.add_argument('-cb', '--convert_batch', action='store_true',
                       help='Convert the batch setting in job_flow.yaml file\n'
                            'from PBS to SLURM or vice versa. '
                            'E.g. aimsflow jyaml -cb')
    parser_job_yaml.add_argument('-u', '--u_value', action='store_true',
                                 help='Add U value for VASP calculation\n'
                                      'E.g. aimsflow jyaml -g r -u')
    parser_job_yaml.add_argument('-m', '--magmom', action='store_true',
                                 help='Add magmom for VASP calculation\n'
                                      'E.g. aimsflow jyaml -g r -m')
    parser_job_yaml.add_argument('-q', '--queue', metavar='queue', type=str,
                                 choices=['hotel', 'home', 'condo', 'glean'],
                                 help='Change the queue name (PBS only).\n'
                                      'E.g. aimsflow jyaml -q home')
    parser_job_yaml.add_argument('-t', '--walltime', type=str,
                                 help='Change job walltime.\n'
                                      'E.g. aimsflow jyaml -t 8')
    parser_job_yaml.add_argument('-p', '--processors', type=int,
                                 help='Change number processors per node.\n'
                                      'E.g. aimsflow jyaml -p 16')
    parser_job_yaml.add_argument('-mail', '--mail', type=str,
                                 help='Change mail options.\n'
                                      'E.g. aimsflow jyaml -m abe')
    parser_job_yaml.add_argument('-hse', '--hse', action='store_true',
                                 help='Perform HSE calculation\n'
                                      'E.g. aimsflow jyaml -g s -hse')
    parser_job_yaml.add_argument('-soc', '--soc', metavar='degree', type=float,
                                 help='Perform SOC calculation\n'
                                      'E.g. aimsflow jyaml -g s -soc 90')
    parser_job_yaml.set_defaults(func=jyaml)

    parser_potcar = subparsers.add_parser(
        'potcar', help='Generate POTCAR from POSCAR or symbol',
        formatter_class=argparse.RawTextHelpFormatter)
    group = parser_potcar.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--poscar', **ps_kwargs)
    group.add_argument('-s', '--symbol', type=str, nargs='+',
                       help='Generate POTCAR from atomic symbol')
    parser_potcar.add_argument('-f', '--functional', default='PBE', nargs='?',
                               choices=['PBE', 'PBE_52', 'PBE_54', 'LDA',
                                        'PW91', 'LDA_US'], const='PBE',
                               help='Set the functional for pseudopotential. '
                                    '(default: %(default)s)\n'
                                    'E.g. aimsflow potcat -p POSCAR -f PBE_52')
    parser_potcar.set_defaults(func=potcar)

    parser_calculate = subparsers.add_parser(
        'calcu', help='Calculate tolerance factor, effective mass, interface '
                      'distance, interface bond length, cation-anion '
                      'displacement and polarization',
        formatter_class=argparse.RawTextHelpFormatter)
    calculate_subparser = parser_calculate.add_subparsers(
        title='action', dest='action command')

    tolfac_parser = calculate_subparser.add_parser(
        'tolfac', help='Tolerance factor',
        formatter_class=argparse.RawTextHelpFormatter)
    group = tolfac_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', type=str,
                       help='Compounds given by a file\n'
                            'E.g. aimsflow calcu tolfac -f compound.dat')
    group.add_argument('-c', '--compound', type=str, nargs='+',
                       help='Compounds given by user input\n'
                            'E.g. aimsflow calcu tolfac -c SrTiO3')
    tolfac_parser.add_argument('-v', '--valence', default='2-4', const='2-4',
                               nargs='?', choices=['1-5', '2-4', '3-3'],
                               help='Set the valences for A and B atoms. '
                                    '(default: %(default)s)\n'
                                    'E.g. aimsflow calcu tolfac -c LaAlO3 -v 3-3')
    tolfac_parser.add_argument('-m', '--mode', default='eff', const='eff',
                               nargs='?', choices=['eff', 'cry', 'bv'],
                               help='Choose the set of data: effective, crystal '
                                    'and bond valence (default: %(default)s)\n'
                                    'E.g. aimsflow calcu tolfac -c SrTiO3 -m bv')
    tolfac_parser.set_defaults(func=tolerance_factor)

    emass_parser = calculate_subparser.add_parser(
        'emass', help='Electron effecitve mass\n'
                      'E.g. aimsflow calcu emass EIGENVAL',
        formatter_class=argparse.RawTextHelpFormatter)
    emass_parser.add_argument('source', metavar='EIGENVAL', nargs='+',
                              help='EIGENVAL files')
    emass_parser.add_argument('-s', '--specify_branch', type=str,
                              help='Branch number or CBM\n'
                                   'E.g. aimsflow calcu emass EIGENVAL -s 1,2-5')
    emass_parser.add_argument('-n', '--number', type=int,
                              help='Number of eigenvalues in each branch\n'
                                   'E.g. aimsflow calcu emass EIGENVAL -n 10')
    emass_parser.add_argument('-t', '--type', default='electron', const='electron',
                              nargs='?', choices=['electron', 'hole'],
                              help='Specify electron or hole mass (default: %(default)s)\n'
                                   'E.g. aimsflow calcu emass EIGENVAL -t hole')
    emass_parser.set_defaults(func=emass)

    ifdist_parser = calculate_subparser.add_parser(
        'ifdist', help='Interface distance for HS\n'
                       'E.g. aimsflow calcu ifdist POSCAR -n 2',
        formatter_class=argparse.RawTextHelpFormatter)
    ifdist_parser.add_argument('poscar', **pss_kwargs)
    ifdist_parser.add_argument(*nl_args, **nl_kwargs)
    ifdist_parser.add_argument(*tol_args, **tol_kwargs)
    ifdist_parser.set_defaults(func=interface_dist)

    ifbond_parser = calculate_subparser.add_parser(
        'ifbond', help='Interface bond length for HS\n'
                       'E.g. aimsflow calcu ifbond POSCAR -n 2',
        formatter_class=argparse.RawTextHelpFormatter)
    ifbond_parser.add_argument('poscar', **pss_kwargs)
    ifbond_parser.add_argument(*nl_args, **nl_kwargs)
    ifbond_parser.add_argument(*tol_args, **tol_kwargs)
    ifbond_parser.set_defaults(func=if_bond_len)

    ki_parser = calculate_subparser.add_parser(
        'ki', help='ki values\nE.g. aimsflow calcu ki static1 static2',
        formatter_class=argparse.RawTextHelpFormatter)
    ki_parser.add_argument('directories', nargs='*', **dir_kwargs)
    ki_parser.add_argument('-in', '--interface_number', default=2, const=2,
                           nargs='?', type=int,
                           help='Set the number of interfaces (default: '
                                '%(default)s)\n E.g. aimsflow calcu ki static1 '
                                'static2 -in 1')
    ki_parser.add_argument('-v', '--verbose', action='store_true',
                           help='Whether to print ionic information')
    ki_parser.set_defaults(func=ki)

    orbki_parser = calculate_subparser.add_parser(
        'orbki', help='Orbital-resolved Ki values\nE.g. aimsflow calcu orbki '
                      '1,3-5 d',
        formatter_class=argparse.RawTextHelpFormatter)
    orbki_parser.add_argument('directories', nargs=2, type=str,
                              help='Directories for 001 and 100')
    orbki_parser.add_argument('site_number', type=str)
    orbki_parser.add_argument('orbital', type=str, choices=['p', 'd', 'f'])
    orbki_parser.add_argument('-in', '--interface_number',
                              nargs='?', type=int, default=2, const=2,
                              help='Set the number of interfaces (default: '
                                   '%(default)s)\n E.g. aimsflow calcu ki static1 '
                                   'static2 -in 1')
    orbki_parser.set_defaults(func=orb_ki)

    displace_parser= calculate_subparser.add_parser(
        'displace', help='Average off-plane displacement (positive value means '
                         'cation is lower than anion)\nE.g. aimsflow calcu '
                         'displace POSCAR -t 1 -s 5',
        formatter_class=argparse.RawTextHelpFormatter)
    displace_parser.add_argument('poscar', **pss_kwargs)
    displace_parser.add_argument(*tol_args, **tol_kwargs)
    displace_parser.add_argument(*rev_args, **rev_kwargs)
    displace_parser.add_argument('-s', '--sub_uc', type=int,
                                help='Specify the number of substrate unit cells')
    displace_parser.set_defaults(func=displace)

    born_polz_parser= calculate_subparser.add_parser(
        'polz', help='Polarization for perovskites (negative value means '
                     'polarization to vacuum)\nE.g. aimsflow calcu polz '
                     'POSCAR CONTCAR -s 5',
        formatter_class=argparse.RawTextHelpFormatter)
    born_polz_parser.add_argument('poscar', **pss_kwargs)
    born_polz_parser.add_argument(*tol_args, **tol_kwargs)
    born_polz_parser.add_argument(*rev_args, **rev_kwargs)
    born_polz_parser.add_argument('-s', '--sub_uc', type=int,
                                help='Specify the number of substrate unit cells')
    born_polz_parser.set_defaults(func=polarization)

    polz_parser= calculate_subparser.add_parser(
        'berry_polz', help='Polarization for perovskites (negative value means '
                           'polarization to vacuum)\nE.g. aimsflow calcu polz '
                           'POSCAR CONTCAR -s 5',
        formatter_class=argparse.RawTextHelpFormatter)
    polz_parser.add_argument('paths', type=str, nargs='+')
    polz_parser.set_defaults(func=berry_phase_polarization)

    parser_build = subparsers.add_parser(
        'build', help='Build HS, supercell, slab, strain, sandwich, IF and '
                      'ferro structures',
        formatter_class=argparse.RawTextHelpFormatter)
    build_subparser = parser_build.add_subparsers(title='action', dest='action command')

    hs_parser = build_subparser.add_parser(
        'hs', help='Heterostructure\nE.g. aimsflow build hs POSCAR_sub '
                   'POSCAR_film -u 3,3 -dl 1t1t1b1b -v 10 -sd 4',
        formatter_class=argparse.RawTextHelpFormatter)
    hs_parser.add_argument('poscars', **pss_kwargs)
    hs_parser.add_argument('-u', '--uc', type=str, required=True,
                           help='Unit cell for each POSCAR,\nE.g. 3,3,3')
    hs_parser.add_argument(*vt_args, **vt_kwargs)
    hs_parser.add_argument('-dl', '--delete_layer', default=None, **dl_kwargs)
    hs_parser.add_argument('-ed', '--extra_distance', **ed_kwargs)
    hs_parser.add_argument(*tol_args, **tol_kwargs)
    hs_parser.add_argument('-sd', '--sd', metavar='SD_layers', type=int,
                           help='Number of substrate layers to be fixed')
    hs_parser.add_argument('-sw', '--sw', action='store_true',
                           help='Whether HS is a sandwidch type\nE.g. aimsflow '
                                'build hs POSCAR_sub POSCAR_film -u 3,3 -dl '
                                '1t1t1b1b -v 10 -sw')
    hs_parser.add_argument('-p', '--primitive', action='store_true',
                           help='Whether to get primitive structure ')
    hs_parser.set_defaults(func=build_hs)

    sc_parser = build_subparser.add_parser(
        'sc', help='Supercell\nE.g. aimsflow sc POSCAR 1,1,4',
        formatter_class=argparse.RawTextHelpFormatter)
    sc_parser.add_argument('poscar', **ps_kwargs)
    sc_parser.add_argument('scaling_matrix', type=str,
                           help='Scaling matrix: (1) a number (2) a sequence '
                                'of three scaling factors (3) a full 3x3 '
                                'scaling matrix')
    sc_parser.set_defaults(func=build_sc)

    slab_parser = build_subparser.add_parser(
        'slab', help='Slab\nE.g. aimsflow slab POSCAR 1,1,1 -v 5 -u 2 -dl 1b1t',
        formatter_class=argparse.RawTextHelpFormatter)
    slab_parser.add_argument('poscar', **ps_kwargs)
    slab_parser.add_argument('miller_index', type=str,
                             help='Miller index of plane parallel to surface')
    slab_parser.add_argument('-u', '--unit_cell', type=int, nargs='?', default=1,
                             const=1, help='Number of unit cell of the slab')
    slab_parser.add_argument('-dl', '--delete_layer', default='0b0t',
                             help='Delete bottom or top layers for slab. '
                                  '(default: %(default)s)')
    slab_parser.add_argument(*vt_args, **vt_kwargs)
    slab_parser.add_argument(*tol_args, **tol_kwargs)
    slab_parser.set_defaults(func=build_slab)

    strain_parser = build_subparser.add_parser(
        'strain', help='Add strain\nE.g. aimsflow build strain POSCAR 0.1 -d x',
        formatter_class=argparse.RawTextHelpFormatter)
    strain_parser.add_argument('poscar', **ps_kwargs)
    strain_parser.add_argument('strain', type=float, nargs='+',
                               help='Strain value (%)')
    strain_parser.add_argument(*d_args, type=str,
                               default='xy', const='xy', nargs='?',
                               choices=['x', 'y', 'z', 'xz', 'yz', 'xy', 'xyz'],
                               help='Strain direction (default: %(default)s)')
    strain_parser.set_defaults(func=build_strain)

    rotate_parser = build_subparser.add_parser(
        'rotate', help='Rotate sites\nE.g. aimsflow build rotate POSCAR 9-16 '
                       '0,0,1 0.5,0.5,0.5 20',
        formatter_class=argparse.RawTextHelpFormatter)
    rotate_parser.add_argument('poscar', **ps_kwargs)
    rotate_parser.add_argument('site_number', type=str,
                               help='Site number\nE.g. 1,2-5')
    rotate_parser.add_argument('axis', type=str, help='Rotation axis')
    rotate_parser.add_argument('anchor', type=str, help='Rotation anchor')
    rotate_parser.add_argument('angle', type=float, help='Rotation angle')
    rotate_parser.set_defaults(func=build_rotate)

    translate_parser = build_subparser.add_parser(
        'translate', help='Translate sites\nE.g. aimsflow build translate '
                          'POSCAR -sn 9-16 -v 0.5,0.5,0.5\nE.g. aimsflow build '
                          'translate POSCAR -sn 9-16 -v center',
        formatter_class=argparse.RawTextHelpFormatter)
    translate_parser.add_argument('poscar', **ps_kwargs)
    translate_parser.add_argument(*sn_args, type=str,
                                  help='Site number\nE.g. aimsflow build '
                                       'translate POSCAR -sn 9-16 -v 0.5,0.5,0.5')
    translate_parser.add_argument('-v', '--vector', type=str, required=True,
                                  help='Translation vector. If center, the '
                                       'sites will be moved to the center')
    translate_parser.add_argument('-c', '--cartesian', action='store_true',
                                  help='The vector corresponds to cartesian '
                                       'coordinates')
    translate_parser.add_argument(*d_args, type=int, choices=[0, 1, 2],
                                  help='Specify the direction to center'
                                       '(default: %(default)s)\nE.g. aimsflow '
                                       'build translate POSCAR -v center -d 2')
    translate_parser.set_defaults(func=translate_sites)

    if_parser = build_subparser.add_parser(
        'if', help='Modify interface distance\nE.g. aimsflow build if POSCAR '
                   '0.04 -n 2 -t 1',
        formatter_class=argparse.RawTextHelpFormatter)
    if_parser.add_argument('poscar', **ps_kwargs)
    if_parser.add_argument('dist', type=float, nargs='+',
                           help='Interface distance')
    if_parser.add_argument(*nl_args, **nl_kwargs)
    if_parser.add_argument(*tol_args, **tol_kwargs)
    if_parser.set_defaults(func=build_if)

    remove_parser = build_subparser.add_parser(
        'remove', help='Remove species or sites\nE.g. aimsflow build '
                       'remove POSCAR -sn 9-16',
        formatter_class=argparse.RawTextHelpFormatter)
    remove_parser.add_argument('poscar', **ps_kwargs)
    group = remove_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(*sn_args, type=str,
                       help='Site number\nE.g. aimsflow build '
                            'remove POSCAR -sn 9-16')
    group.add_argument('-sp', '--species', type=str, nargs='+',
                       help='Species\nE.g. aimsflow build '
                            'remove POSCAR -sp Sr Ti')
    group.add_argument('-l', '--layers', type=str,
                       help='Layer number\nE.g. aimsflow build '
                            'remove POSCAR -l 2b')
    remove_parser.add_argument(*d_args, type=int, nargs='?', const=2,
                               choices=[0, 1, 2], default=2,
                               help='Remove layers along certain direction '
                                    '(default: %(default)s)\nE.g. aimsflow '
                                    'build remove -l 2b -d 1')
    remove_parser.add_argument(*tol_args, **tol_kwargs)
    remove_parser.set_defaults(func=remove)

    ferro_parser = build_subparser.add_parser(
        'ferro', help='Ferroelectric structure',
        formatter_class=argparse.RawTextHelpFormatter)
    ferro_parser.add_argument('poscar', **pss_kwargs)
    ferro_parser.add_argument(*rev_args, **rev_kwargs)
    ferro_parser.add_argument(*tol_args, **tol_kwargs)
    ferro_parser.add_argument('-s', '--sub_uc', type=int,
                                help='Specify the number of substrate unit cells.')
    group = ferro_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-o', '--opposite', action='store_true',
                       help='Generate the structure with same ferroelectric '
                            'strength but different direction\nE.g. aimsflow '
                            'build ferro CONTCAR -s 5 -o')
    group.add_argument('-n', '--number', type=int,
                       help='The number of intermediate structures\nE.g. '
                            'aimsflow build ferro POSCAR CONTCAR -s 5 -n 4')
    group.add_argument('-tp', '--transplant', type=str,
                       help='Transplant the ferroelectricity to another system '
                            'with same number of unit cells\nE.g. aimsflow build '
                            'ferro POSCAR CONTCAR -s 5 -tp POSCAR1')
    group.add_argument('-e', '--enhance', type=float,
                       help='Enhance the distortion by a given percentage\nE.g. '
                            'aimsflow build ferro POSCAR CONTCAR -s 5 -e 30')
    ferro_parser.set_defaults(func=build_ferro)

    parser_cite= subparsers.add_parser(
        'cite', help='Format bib files',
        formatter_class=argparse.RawTextHelpFormatter)
    parser_cite.add_argument('files', metavar='bib_files', nargs='+',
                             help='Citation files for formatting\n'
                                  'E.g. aimsflow cite nmat4966.ris')
    parser_cite.set_defaults(func=format_citation)

    parser_kpts = subparsers.add_parser(
        'kpt', help='Generate KPOINTS file from the given POSCAR',
        formatter_class=argparse.RawTextHelpFormatter)
    parser_kpts.add_argument('poscar', default='POSCAR', nargs='?', **ps_kwargs)
    group = parser_kpts.add_mutually_exclusive_group(required=True)
    group.add_argument('-r', '--relax', type=int, const=64, nargs='?',
                       help='Reciprocal density for relaxation\n'
                            'E.g. aimsflow kpt POSCAR -r')
    group.add_argument('-s', '--static', type=int, const=400, nargs='?',
                       help='Reciprocal density for static\n'
                            'E.g. aimsflow kpt POSCAR -s')
    group.add_argument('-b', '--band', type=int, const=15, nargs='?',
                       help='Grid density for each high symmetry k-path\n'
                            'E.g. aimsflow kpt POSCAR -b')
    parser_kpts.set_defaults(func=kpoints)

    parser_query = subparsers.add_parser(
        'query', help='Query the data and file from aflow.\nE.g. aimsflow query '
                      'ICSD_WEB/HEX/I1Li1_ICSD_414242 -v Egap -f CONTCAR.relax',
        formatter_class=argparse.RawTextHelpFormatter)
    parser_query.add_argument('aurl', type=str, nargs='+',
                              help='E.g. aimsflow query ICSD_WEB/HEX/I1Li1_ICSD_414242 '
                                   '-v Egap -f CONTCAR.relax')
    parser_query.add_argument('-v', '--values', type=str, nargs='+')
    parser_query.add_argument('-f', '--files', type=str, nargs='+')
    parser_query.set_defaults(func=query_aflow)

    args = parser.parse_args()

    try:
        getattr(args, 'func')
    except AttributeError:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == '__main__':
    main()

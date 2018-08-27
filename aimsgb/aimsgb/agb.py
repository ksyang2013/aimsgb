#!/usr/bin/env python
from __future__ import division

import sys
import argparse
import numpy as np

from aimsgb import Grain, GBInformation, GrainBoundary

__author__ = "Jianli Cheng, Kesong Yang"
__copyright__ = "Copyright 2018, Yanggroup"
__maintainer__ = "Jianli Cheng"
__email__ = "jic198@ucsd.edu"
__status__ = "Production"
__date__ = "January 26, 2018"


def gb_list(args):
    axis = list(map(int, args.axis))
    sigma = args.sigma
    print(GBInformation(axis, sigma).__str__())


def gb(args):
    axis = list(map(int, args.axis))
    plane = args.plane
    initial_struct = Grain.from_file(args.poscar)
    gb = GrainBoundary(axis, args.sigma, plane, initial_struct, args.uc_a, args.uc_b)
    to_primitive = False if args.conventional else True
    gb.build_gb(args.vacuum, args.add_if_dist, to_primitive, args.delete_layer,
                args.tol).to(filename=args.out, fmt=args.fmt)
    print("CSL Matrix (det=%.1f):\n%s" % (np.linalg.det(gb.csl), gb.csl))
    print("%s of sigma%s[%s]/(%s) is created"
          % (args.out, gb.sigma, args.axis, gb.plane_str))


def main():
    parser = argparse.ArgumentParser(description="AIMSgb command line tools",
                                     formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(help="command", dest="command")

    parser_gb_list = subparsers.add_parser(
        "list", help="Show the values of sigma, theta, GB plane and CSL matrix "
                     "from a given rotation axis\nE.g. agb list 001",
        formatter_class=argparse.RawTextHelpFormatter)
    parser_gb_list.add_argument("axis", metavar="rotation axis", type=str,
                                help="The rotation axis of GB\n"
                                     "E.g. agb list 001")
    parser_gb_list.add_argument("sigma", default=30, type=int, nargs="?",
                                help="Set the sigma limit for on screen output "
                                     "(default: %(default)s)\n"
                                     "E.g. agb list 001 100")
    parser_gb_list.set_defaults(func=gb_list)

    parser_gb = subparsers.add_parser(
        "gb", help="Build grain boundary based on rotation axis, sigma, GB plane, "
                   "and input structure file.\nThe user can also specify many "
                   "optional arguments, such grain size and interface terminations."
                   "\nE.g. agb gb 001 5 1 2 0 POSCAR"
                   "\nE.g. agb gb 001 5 1 2 0 POSCAR -ua 2 -ub 3"
                   "\nE.g. agb gb 001 5 1 2 0 POSCAR -ua 3 -ub 2 -dl 0b1t1b0t",
        formatter_class=argparse.RawTextHelpFormatter)
    parser_gb.add_argument("axis", metavar="rotation axis", type=str,
                           help="The rotation axis of GB, E.g. 110")
    parser_gb.add_argument("sigma", type=int,
                           help="The sigma value for grain boundary, E.g. 3")
    parser_gb.add_argument("plane", type=int, nargs=3,
                           help="The GB plane for grain boundary, E.g. 1 1 0")
    parser_gb.add_argument("poscar", type=str,
                           help="The initial structure file for grain boundary.")
    parser_gb.add_argument("out", type=str, default="POSCAR", nargs="?",
                           help="The output filename. (default: %(default)s)")
    parser_gb.add_argument("-ua", "--uc_a", default=1, type=int,
                           help="The size (uc) for grain A. (default: %(default)s)"
                                "\nE.g. agb gb 001 5 1 2 0 POSCAR -ua 2")
    parser_gb.add_argument("-ub", "--uc_b", default=1, type=int,
                           help="The size (uc) for grain B. (default: %(default)s)"
                                "\nE.g. agb gb 001 5 1 2 0 POSCAR -ub 2")
    parser_gb.add_argument("-dl", "--delete_layer", default="0b0t0b0t", type=str,
                           help="Delete bottom or top layers for each grain. "
                                "(default: %(default)s)"
                                "\nE.g. agb gb 001 5 1 2 0 POSCAR -dl 0b1t1b0t")
    parser_gb.add_argument("-v", "--vacuum", default=0.0, type=float,
                           help="Set vacuum thickness for grain boundary. "
                                "(default: %(default)s angstrom)"
                                "\nE.g. agb gb 001 5 1 2 0 POSCAR -v 20")
    parser_gb.add_argument("-t", "--tol", default=0.25, type=float,
                           help="Tolerance factor to determine if two "
                                "atoms are at the same plane. "
                                "(default: %(default)s angstrom)"
                                "\nE.g. agb gb 001 5 1 2 0 POSCAR -dl 0b1t1b0t -t 0.5")
    parser_gb.add_argument("-ad", "--add_if_dist", default=0.0, type=float,
                           help="Add extra distance between two grains. "
                                "(default: %(default)s angstrom)"
                                "\nE.g. agb gb 001 5 1 2 0 POSCAR -ad 0.5")
    parser_gb.add_argument("-c", "--conventional", action="store_true",
                           help="Get conventional GB, not primitive"
                                "\nE.g. agb gb 001 5 1 2 0 POSCAR -c")
    parser_gb.add_argument("-fmt", "--fmt", default="poscar", const="poscar",
                           nargs="?", choices=["poscar", "cif", "cssr", "json"],
                           help="Choose the output format. (default: %(default)s)"
                                "\nE.g. agb gb 001 5 1 2 0 POSCAR -fmt cif")
    parser_gb.set_defaults(func=gb)

    args = parser.parse_args()

    try:
        getattr(args, "func")
    except AttributeError:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
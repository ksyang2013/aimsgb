#!/usr/bin/env python
from __future__ import division

import sys
import argparse
import numpy as np

from aimsgb import Grain, GBInformation, GrainBoundary, SIGMA_SYMBOL

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
    print("%s of %s%s[%s]/(%s) is created"
          % (args.out, SIGMA_SYMBOL, gb.sigma, args.axis, gb.plane_str))


def main():
    parser = argparse.ArgumentParser(description="AIMSgb command line tools",
                                     formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(help="command", dest="command")

    parser_gb_list = subparsers.add_parser(
        "list", help="Show the values of %s, theta and GB plane\n"
                     "E.g. agb list 001" % SIGMA_SYMBOL)
    parser_gb_list.add_argument("axis", metavar="rotation-axis", type=str,
                                help="The rotation axis of GB")
    parser_gb_list.add_argument("sigma", default=30, type=int, nargs="?",
                                help="Set the %s limit for on screen output "
                                     % SIGMA_SYMBOL + "(default: %(default)s)")
    parser_gb_list.set_defaults(func=gb_list)

    parser_gb = subparsers.add_parser(
        "gb", help="Build grain boundary based on rotation axis, sigma, "
                   "GB plane, and grain size.\nE.g. agb gb 001 5 1 2 0 POSCAR")
    parser_gb.add_argument("axis", metavar="rotation-axis", type=str,
                           help="The rotation axis of GB")
    parser_gb.add_argument("sigma", type=int,
                           help="Set the %s value for grain boundary." % SIGMA_SYMBOL)
    parser_gb.add_argument("plane", type=int, nargs=3,
                           help="Set the GB plane for grain boundary.")
    parser_gb.add_argument("poscar", type=str,
                           help="Set crystal structure for grain boundary.")
    parser_gb.add_argument("out", type=str, default="POSCAR", nargs="?",
                           help="Set the output file. (default: %(default)s)")
    parser_gb.add_argument("-ua", "--uc_a", default=1, type=int,
                           help="Set the size (uc) for grain A. (default: %(default)s)")
    parser_gb.add_argument("-ub", "--uc_b", default=1, type=int,
                           help="Set the size (uc) for grain B. (default: %(default)s)")
    parser_gb.add_argument("-dl", "--delete_layer", default="0b0t0b0t", type=str,
                           help="Delete bottom or top layers for each grain. "
                                "(default: %(default)s)")
    parser_gb.add_argument("-v", "--vacuum", default=0.0, type=float,
                           help="Set vacuum thickness for grain boundary. "
                                "(default: %(default)s)")
    parser_gb.add_argument("-t", "--tol", default=0.25, type=float,
                           help="Tolerance factor to determine if two atoms are "
                                "at the same plane. (default: %(default)s)")
    parser_gb.add_argument("-ad", "--add_if_dist", default=0.0, type=float,
                           help="Add extra distance between two grains. "
                                "(default: %(default)s)")
    parser_gb.add_argument("-c", "--conventional", action="store_true",
                           help="Get conventional GB, not primitive")
    parser_gb.add_argument("-fmt", "--fmt", default="poscar", const="poscar",
                           nargs="?", choices=["poscar", "cif", "cssr", "json"],
                           help="Choose the output format. (default: %(default)s)")
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
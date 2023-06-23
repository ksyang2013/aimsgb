#!/usr/bin/env python  

"""
File Name: test_time.py
Created Time: 2018-06-18 17:52:06
Author: Prof. KESONG YANG,  UC San Diego  
Mail: kesong@ucsd.edu

Python source code - replace this with a description of the code and write the code below this text.
"""

import sys

import time
from aimsgb import GrainBoundary, Grain, GBInformation


if len(sys.argv) == 2 :
    filename = sys.argv[1]
else:
    sys.stderr.write("usage: python program.py POSCAR")
    exit(0)

s = Grain.from_file(filename)
axis = [0, 0, 1]
info = GBInformation(axis, 150)
sigmas = sorted(info.keys())
print('\t'.join([u'\u03A3', 'Time (sec)', 'No. sites']))
for sig in sigmas:
    start = time.time()
    gb = GrainBoundary(axis, sig, axis, s,)
    struct = gb.build_gb(to_primitive=False)
    #struct.to(filename="out/POSCAR")
    duration = '%.2f' % (time.time() - start)
    print(' &'.join(map(str, [sig, len(struct), duration])) + ' \\\\')


# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4


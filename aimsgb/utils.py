from __future__ import division

import numpy as np
from numpy.linalg import norm
from functools import wraps, reduce
try:
    from math import gcd as pygcd
except ImportError:
    from fractions import gcd as pygcd

__author__ = "Jianli Cheng, Kesong Yang"
__copyright__ = "Copyright 2018, Yanggroup"
__maintainer__ = "Jianli Cheng"
__email__ = "jic198@ucsd.edu"
__status__ = "Production"
__date__ = "January 26, 2018"


def reduce_vector(vector):
    d = abs(reduce(gcd, vector))
    vector = tuple([int(i / d) for i in vector])

    return vector


def co_prime(a, b):
    return gcd(a, b) in (0, 1)


def plus_minus_gen(start, end):
    for i in range(start, end):
        yield i
        yield -i


def is_integer(a, tol=1e-5):
    return norm(abs(a - np.round(a))) < tol


def get_smallest_multiplier(a, max_n=10000):
    a = np.array(a)
    for i in range(1, max_n):
        if is_integer(i * a):
            return i
    raise ValueError("Cannot find an integer matrix with multiplier "
                     "searched already up to {}".format(max_n))


def reduce_integer(integer):
    while gcd(integer, 2) != 1:
        integer /= 2
    return integer


def gcd(*numbers):
    n = numbers[0]
    for i in numbers:
        n = pygcd(n, i)
    return n


def transpose_matrix(func):
    """
    A decorator; transpose the first argument and the return value (both
    should be 3x3 arrays). This makes column operations easier
    """

    @wraps(func)
    def transpose(*args, **kwargs):
        args_list = list(args)
        args_list[0] = args_list[0].transpose()
        matrix = func(*args_list, **kwargs)
        return matrix.transpose()

    return transpose

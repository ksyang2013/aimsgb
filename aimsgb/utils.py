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
    """
    Reduce a vector
    """
    d = abs(reduce(gcd, vector))
    vector = tuple([int(i / d) for i in vector])

    return vector


def co_prime(a, b):
    """
    Check if two integers are co-prime
    """
    return gcd(a, b) in (0, 1)


def plus_minus_gen(start, end):
    """
    Generate a list of plus and minus alternating integers
    """
    for i in range(start, end):
        yield i
        yield -i


def is_integer(a, tol=1e-5):
    """
    Check whether the number is integer
    """
    return norm(abs(a - np.round(a))) < tol


def get_smallest_multiplier(a, max_n=10000):
    """
    Get the smallest multiplier to make the list with all integers
    Args:
        a (list): A list of numbers
        max_n (int): The up limit to search multiplier

    Returns:
        The smallest integer multiplier
    """
    a = np.array(a)
    for i in range(1, max_n):
        if is_integer(i * a):
            return i
    raise ValueError("Cannot find an integer matrix with multiplier "
                     "searched already up to %s" % max_n)


def reduce_integer(integer):
    """
    Get the odd number for an integer
    """
    while gcd(integer, 2) != 1:
        integer //= 2
    return integer


def gcd(*numbers):
    """
    Get a greatest common divisor for a list of numbers
    """
    n = numbers[0]
    for i in numbers:
        n = pygcd(n, i)
    return n


def transpose_matrix(func):
    """
    Transpose the first argument and the return value.
    Args:
        func: The function that uses transpose_matrix as a decorator.
    """

    @wraps(func)
    def transpose(*args, **kwargs):
        args_list = list(args)
        args_list[0] = args_list[0].transpose()
        matrix = func(*args_list, **kwargs)
        return matrix.transpose()

    return transpose

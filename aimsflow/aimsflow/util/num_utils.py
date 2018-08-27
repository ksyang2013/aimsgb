from __future__ import division

import re
import numpy as np
from functools import reduce
from datetime import datetime
from collections import OrderedDict, Callable
try:
    from math import gcd as pygcd
except ImportError:
    from fractions import gcd as pygcd


def abs_cap(val, max_abs_value=1):
    return max(min(val, max_abs_value), -max_abs_value)


def remove_non_ascii(s):
    return ''.join(i for i in s if ord(i) < 128)


def delete_dict_keys(keys, dictionary):
    for k in keys:
        if k in dictionary:
            del dictionary[k]


def parse_number(string):
    """
    Return a list of integer numbers.
    Such as "1,3:5" will be returned as [1, 3, 4, 5]

    :param string: A str, such as "1,3:5"
    :return: List of integers
    """
    site_num = re.split(",", string)
    num_list = []
    for number in site_num:
        try:
            start, end = map(int, number.split("-"))
            num_list.extend(range(start, end + 1))
        except ValueError:
            num_list.extend([int(number)])
    return num_list


def gcd(*numbers):
    n = numbers[0]
    for i in numbers:
        n = pygcd(n, i)
    return n


def lcm(*numbers):
    n = 1
    for i in numbers:
        n = (i * n) // gcd(i, n)
    return n


def time_to_second(time):
    time_list = [int(i) for i in time.split(":")]
    for i in range(3):
        try:
            time_list[i]
        except IndexError:
            time_list.append(0)
    return np.dot(time_list, [3600, 60, 1])


def second_to_time(seconds):
    tot_seconds = int(float(seconds))
    hour = tot_seconds // 3600
    minute = (tot_seconds - 3600 * hour) // 60
    second = tot_seconds - 3600 * hour - 60 * minute
    return ":".join([str(i) for i in [hour, minute, second]])


def time_lapse(old_date):
    date_format = "%Y.%m.%d %H:%M:%S"
    old = datetime.strptime(old_date, date_format)
    new = datetime.now()
    diff = new - old
    return diff.days * 24 * 3600 + diff.seconds


def column_stack(a, b):
    return np.column_stack((a, b))


def flatten_lists(lists):
    return reduce(lambda x, y: np.concatenate((x, y), axis=0), lists)


class OrderedDefaultDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))
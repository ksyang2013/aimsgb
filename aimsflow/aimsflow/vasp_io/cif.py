import re, six
from collections import deque, OrderedDict
from aimsflow.util.num_utils import remove_non_ascii

class CifBlock(object):

    @classmethod
    def _process_string(cls, string):
        string = re.sub("(\s|^)#.*$", '', string, flags=re.MULTILINE)
        string = re.sub("^\s*\n", '', string, flags=re.MULTILINE)
        string = remove_non_ascii(string)
        q = deque()
        multiline = False
        ml = []
        for l in string.splitlines():
            if l.startswith(';'):
                multiline = True
                ml.append(l[1:].strip())
            # else:



    @classmethod
    def from_string(cls, string):
        q = cls._process_string(string)


class CifFile(object):

    @classmethod
    def from_string(cls, string):
        d = OrderedDict()
        for x in re.split("^\s*data_", "x\n"+string,
                          flags=re.MULTILINE | re.DOTALL)[1:]:
            if "powder_pattern" in re.split("\n", x, 1)[0]:
                continue
            c = CifBlock.from_string("data_" + x)


class CifParser(object):
    def __init__(self, filename, occupancy_tolerance=1., site_tolerance=1e-4):
        self._occupancy_tolerance = occupancy_tolerance
        self._site_tolerance = site_tolerance
        if isinstance(filename, six.string_types):
            self._cif = CifFile.from_file(filename)
        else:
            self._cif = CifFile.from_string(filename.read())





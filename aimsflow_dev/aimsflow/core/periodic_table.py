import os
import re
import json
from io import open
from enum import Enum

from aimsflow.util import loadfn

with open(os.path.join(os.path.dirname(__file__),
                       "periodic_table.json"), "rt") as f:
    _pt_data = json.load(f)

atomic_toten = loadfn(os.path.join(os.path.dirname(__file__), "atomic_toten.yaml"))
eff_radii = loadfn(os.path.join(os.path.dirname(__file__), "shanno_radii.yaml"))
bd_val = loadfn(os.path.join(os.path.dirname(__file__), "bond_valence.yaml"))
born = loadfn(os.path.join(os.path.dirname(__file__), "born.yaml"))


class Element(Enum):
    H = "H"
    He = "He"
    Li = "Li"
    Be = "Be"
    B = "B"
    C = "C"
    N = "N"
    O = "O"
    F = "F"
    Ne = "Ne"
    Na = "Na"
    Mg = "Mg"
    Al = "Al"
    Si = "Si"
    P = "P"
    S = "S"
    Cl = "Cl"
    Ar = "Ar"
    K = "K"
    Ca = "Ca"
    Sc = "Sc"
    Ti = "Ti"
    V = "V"
    Cr = "Cr"
    Mn = "Mn"
    Fe = "Fe"
    Co = "Co"
    Ni = "Ni"
    Cu = "Cu"
    Zn = "Zn"
    Ga = "Ga"
    Ge = "Ge"
    As = "As"
    Se = "Se"
    Br = "Br"
    Kr = "Kr"
    Rb = "Rb"
    Sr = "Sr"
    Y = "Y"
    Zr = "Zr"
    Nb = "Nb"
    Mo = "Mo"
    Tc = "Tc"
    Ru = "Ru"
    Rh = "Rh"
    Pd = "Pd"
    Ag = "Ag"
    Cd = "Cd"
    In = "In"
    Sn = "Sn"
    Sb = "Sb"
    Te = "Te"
    I = "I"
    Xe = "Xe"
    Cs = "Cs"
    Ba = "Ba"
    La = "La"
    Ce = "Ce"
    Pr = "Pr"
    Nd = "Nd"
    Pm = "Pm"
    Sm = "Sm"
    Eu = "Eu"
    Gd = "Gd"
    Tb = "Tb"
    Dy = "Dy"
    Ho = "Ho"
    Er = "Er"
    Tm = "Tm"
    Yb = "Yb"
    Lu = "Lu"
    Hf = "Hf"
    Ta = "Ta"
    W = "W"
    Re = "Re"
    Os = "Os"
    Ir = "Ir"
    Pt = "Pt"
    Au = "Au"
    Hg = "Hg"
    Tl = "Tl"
    Pb = "Pb"
    Bi = "Bi"
    Po = "Po"
    At = "At"
    Rn = "Rn"
    Fr = "Fr"
    Ra = "Ra"
    Ac = "Ac"
    Th = "Th"
    Pa = "Pa"
    U = "U"
    Np = "Np"
    Pu = "Pu"
    Am = "Am"
    Cm = "Cm"
    Bk = "Bk"
    Cf = "Cf"
    Es = "Es"
    Fm = "Fm"
    Md = "Md"
    No = "No"
    Lr = "Lr"

    def __init__(self, symbol):
        self.symbol = "%s" % symbol
        d = _pt_data[symbol]
        self.Z = d["Atomic no"]
        self.X = d.get("X", 0)
        self._data = d

    def __eq__(self, other):
        return isinstance(other, Element) and self.Z == other.Z

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.Z

    def __repr__(self):
        return "Element " + self.symbol

    def __str__(self):
        return self.symbol

    def __lt__(self, other):
        """
        Sets a default sort order for atomic species by electronegativity. Very
        useful for getting correct formulas.  For example, FeO4PLi is
        automatically sorted into LiFePO4.
        """
        if self.X != other.X:
            return self.X < other.X
        else:
            # There are cases where the electronegativity are exactly equal.
            # We then sort by symbol.
            return self.symbol < other.symbol

    def __deepcopy__(self, memo):
        return Element(self.symbol)

    @property
    def ionic_radii(self):
        if "Ionic radii" in self._data:
            return {int(k): v for k, v in self._data["Ionic radii"].items()}
        else:
            return {}

    @property
    def max_oxidation_state(self):
        if self._data.get("Oxidation states"):
            return max(self._data["Oxidation states"])
        return 0

    @property
    def min_oxidation_state(self):
        if self._data.get("Oxidation states"):
            return min(self._data["Oxidation states"])
        return 0

    @property
    def oxidation_states(self):
        return self._data.get("Oxidation states")

    @property
    def common_oxidation_states(self):
        return self._data.get("Common oxidation states")

    @property
    def is_lanthanoid(self):
        return 56 < self.Z < 72

    @property
    def is_actinoid(self):
        return 88 < self.Z < 104

    @property
    def is_rare_earth_metal(self):
        return self.is_lanthanoid or self.is_actinoid

    @property
    def is_transition_metal(self):
        ns = list(range(21, 31))
        ns.extend(list(range(39, 49)))
        ns.append(57)
        ns.extend(list(range(72, 81)))
        ns.append(89)
        ns.extend(list(range(104, 113)))
        return self.Z in ns

    @staticmethod
    def from_z(z):
        for symbol, data in _pt_data.items():
            if data["Atomic no"] == z:
                return Element(symbol)
        raise ValueError("No element with this atomic number %s" % z)

    @staticmethod
    def is_valid_symbol(symbol):
        try:
            Element(symbol)
            return True
        except ValueError:
            return False


class Specie(object):
    cache = {}

    def __new__(cls, *args, **kwargs):
        key = (cls,) + args + tuple(kwargs.items())
        try:
            inst = Specie.cache.get(key, None)
        except TypeError:
            inst = key = None
        if inst is None:
            inst = object.__new__(cls)
            if key is not None:
                Specie.cache[key] = inst
        return inst

    supported_properties = ("spin",)

    def __init__(self, symbol, oxidation_state, properties=None):
        self._symbol = Element(symbol)
        self._oxi_state = oxidation_state
        self._properties = properties if properties else {}
        for k in self._properties.keys():
            if k not in Specie.supported_properties:
                raise ValueError("{} is not a supported property".format(k))

    def __getattr__(self, a):
        p = object.__getattribute__(self, "_properties")
        if a in p:
            return p[a]
        try:
            return getattr(self._symbol, a)
        except:
            raise AttributeError(a)

    # def __hash__(self):
    #     return self._el.Z * 1000 + int(self._oxi_state)

    def __eq__(self, other):
        return isinstance(other, Specie) and self.symbol == other.symbol \
               and self._oxi_state == other.oxi_state\
               and self._properties == other.properties

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if self.X != other.X:
            return self.X < other.X
        elif self.symbol != other.symbol:
            return self.symbol < other.symbol
        else:
            other_oxi = 0 if isinstance(other, Element) else other.oxi_state
            return self.oxi_state < other_oxi

    # def __deepcopy__(self, memo):
    #     return Specie(self.symbol, self.oxi_state, self._properties)
    @property
    def element(self):
        return self._symbol

    @property
    def ionic_radius(self):
        if self._oxi_state in self.ionic_radii:
            return self.ionic_radii[self._oxi_state]
        else:
            return None

    @property
    def oxi_state(self):
        return self._oxi_state

    @staticmethod
    def from_string(species_string):
        m = re.search(r"([A-Z][a-z]*)([0-9\.]*)([\+\-])(.*)", species_string)
        if m:
            sym = m.group(1)
            oxi = 1 if m.group(2) == "" else float(m.group(2))
            oxi = -oxi if m.group(3) == "-" else oxi
            properties = None
            if m.group(4):
                toks = m.group(4).split("=")
                properties = {toks[0]: float(toks[1])}
            return Specie(sym, oxi, properties)
        else:
            raise ValueError("Invalid Species String")


class DummySpecie(Specie):

    def __init__(self, symbol='X', oxidation_state=0, properties=None):
        for i in range(1, min(2, len(symbol)) + 1):
            if Element.is_valid_symbol(symbol[:i]):
                raise ValueError("{} contains {}, which is a valid element "
                                 "symbol.".format(symbol, symbol[:i]))
        self._symbol = symbol
        self._oxi_state = oxidation_state
        self._properties = properties if properties else {}
        for k in self._properties.keys():
            if k not in Specie.supported_properties:
                raise ValueError("{} is not a supported property".format(k))

    # def __hash__(self):
    #     return 1

    def __eq__(self, other):
        if not isinstance(other, DummySpecie):
            return False
        return self.symbol == other.symbol \
               and self.oxi_state == other.oxi_state

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if self.X != other.X:
            return self.X < other.X
        elif self.symbol != other.symbol:
            return self.symbol < other.symbol
        else:
            other_oxi = 0 if isinstance(other, Element) else other.oxi_state
            return self.oxi_state < other_oxi

    # def __deepcopy__(self, memo):
    #     return DummySpecie(self.symbol, self.oxi_state, self._properties)

    @property
    def symbol(self):
        return self._symbol

    @property
    def Z(self):
        return 0


def get_el_sp(obj):
    if isinstance(obj, (Element, Specie, DummySpecie)):
        return obj

    try:
        c = float(obj)
        i = int(c)
        i = i if i == c else None
    except (ValueError, TypeError):
        i = None

    if i is not None:
        return Element.from_z(i)

    try:
        return Specie.from_string(obj)
    except (ValueError, KeyError):
        return Element(obj)
import re
import six
import collections
from functools import total_ordering

from aimsflow.util.num_utils import gcd
from aimsflow.core.periodic_table import get_el_sp
from aimsflow.util.string_utils import format_float


@total_ordering
class Composition(collections.Hashable, collections.Mapping):
    amount_tolerance = 1e-8
    special_formulas = {"LiO": "Li2O2", "NaO": "Na2O2", "KO": "K2O2",
                        "HO": "H2O2", "CsO": "Cs2O2", "RbO": "Rb2O2",
                        "O": "O2",  "N": "N2", "F": "F2", "Cl": "Cl2",
                        "H": "H2"}

    def __init__(self, *args, **kwargs):
        self.allow_negative = kwargs.pop("allow_negative", False)
        if len(args) == 1 and isinstance(args[0], Composition):
            elmap = args[0]
        elif len(args) == 1 and isinstance(args[0], six.string_types):
            elmap = self._parse_formula(args[0])
        else:
            elmap = dict(*args, **kwargs)
        elamt = collections.OrderedDict()
        self._atom_num = 0
        for k, v in elmap.items():
            if v < -Composition.amount_tolerance and not self.allow_negative:
                raise CompositionError("Amounts in Composition cannot be "
                                       "negative!")
            if abs(v) >= Composition.amount_tolerance:
                elamt[get_el_sp(k)] = v
                self._atom_num += abs(v)
        self._data = elamt

    def __str__(self):
        return " ".join([
            "{}{}".format(k, format_float(v, no_one=False))
            for k, v in self.items()])

    def __repr__(self):
        return self.formula

    def __getitem__(self, item):
        try:
            sp = get_el_sp(item)
            return self._data.get(sp, 0)
        except ValueError as ex:
            raise TypeError("Invalid key {}, {} for Composition\n"
                            "ValueError exception:\n{}".format(item, type(item), ex))

    def __contains__(self, item):
        try:
            sp = get_el_sp(item)
            return sp in self._data
        except ValueError as ex:
            raise TypeError("Invalid key {}, {} for Composition\n"
                            "ValueError exception:\n{}".format(item, type(item), ex))

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return self._data.keys().__iter__()

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for el, v in self.items():
            if abs(v - other[el]) > Composition.amount_tolerance:
                return False
        return True

    def __ge__(self, other):
        for el in sorted(set(self.elements + other.elements)):
            if other[el] - self[el] >= Composition.amount_tolerance:
                return False
            elif self[el] - other[el] >= Composition.amount_tolerance:
                return True
        return True

    def __hash__(self):
        hashcode = 0
        for el, amt in self.items():
            if abs(amt) > Composition.amount_tolerance:
                hashcode += el.Z
        return hashcode

    @property
    def elements(self):
        return list(self.keys())

    @property
    def num_atoms(self):
        return self._atom_num

    @property
    def formula(self):
        sym_amt = self.get_el_amt_dict()
        syms = sorted(sym_amt.keys(), key=lambda sym: get_el_sp(sym).symbol)
        formula = [s + format_float(sym_amt[s], False) for s in syms]
        return "".join(formula)

    @property
    def reduced_formula(self):
        return self.get_reduced_formula_and_factor()[0]

    @property
    def average_electroneg(self):
        return sum((el.X * abs(amt) for el, amt in self.items())) / self.num_atoms

    def get_el_amt_dict(self):
        d = collections.defaultdict(float)
        for e, a in self.items():
            d[e.symbol] += a
        return d

    def get_reduced_formula_and_factor(self):
        all_int = all(abs(x - round(x)) < Composition.amount_tolerance
                      for x in self.values())
        if not all_int:
            return self.formula.replace(" ", ""), 1
        d = {k: int(round(v)) for k, v in self.get_el_amt_dict().items()}
        (formula, factor) = reduce_formula(d)

        if formula in Composition.special_formulas:
            formula = Composition.special_formulas[formula]
            factor /= 2

        return formula, factor

    def _parse_formula(self, formula):
        def get_symbol_dict(formula, factor):
            atom_dict = collections.OrderedDict()
            for m in re.finditer(r"([A-Z][a-z]*)([-*\.\d]*)", formula):
                symbol = m.group(1)
                atom_num = 1
                if m.group(2).strip() != '':
                    atom_num = float(m.group(2))
                if symbol in atom_dict:
                    atom_dict[symbol] += atom_num * factor
                else:
                    atom_dict[symbol] = atom_num * factor
                formula = formula.replace(m.group(), '', 1)
            if formula.strip():
                raise CompositionError("{0} is an invalid formula!".format(formula))
            return atom_dict

        for m in re.finditer(r"\((.*?)\)([\d\.]*)", formula):
            factor = 1
            if m.group(2) != "":
                factor = float(m.group(2))
            unit_symbol_dict = get_symbol_dict(m.group(1), factor)
            expand_unit = ''.join(["{0}{1}".format(symbol, atom_num)
                                  for symbol, atom_num in unit_symbol_dict.items()])
            expand_formula = formula.replace(m.group(), expand_unit)
            return self._parse_formula(expand_formula)
        return get_symbol_dict(formula, 1)


def reduce_formula(sym_amt):
    syms = sorted(sym_amt.keys(), key=lambda s: get_el_sp(s).X)
    syms = list(filter(lambda s: abs(sym_amt[s]) >
                                 Composition.amount_tolerance, syms))
    num_el = len(syms)
    contains_polyanion = (num_el >= 3 and
                          get_el_sp(syms[num_el - 1]).X
                          - get_el_sp(syms[num_el - 2]).X < 1.65)
    factor = 1
    if all((int(i) == i for i in sym_amt.values())):
        factor = abs(gcd(*(int(i) for i in sym_amt.values())))

    reduced_form = []
    n = num_el - 2 if contains_polyanion else num_el
    for i in range(n):
        s = syms[i]
        normamt = sym_amt[s] * 1.0 / factor
        reduced_form.append(s)
        reduced_form.append(format_float(normamt))

    if contains_polyanion:
        poly_sym_amt = {syms[i]: sym_amt[syms[i]] / factor
                        for i in range(n, num_el)}
        (poly_form, poly_factor) = reduce_formula(poly_sym_amt)

        if poly_factor != 1:
            reduced_form.append("({}){}".format(poly_form, int(poly_factor)))
        else:
            reduced_form.append(poly_form)

    reduced_form = "".join(reduced_form)

    return reduced_form, factor


class CompositionError(Exception):
    pass
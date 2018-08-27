import os
from abc import ABCMeta, abstractproperty

import numpy as np
from fractions import Fraction
from collections import Sequence

from aimsflow.core.operations import SymmOp
from aimsflow.util import loadfn

SYMM_DATA = loadfn(os.path.join(os.path.dirname(__file__), "symm_data.yaml"))

GENERATOR_MATRICES = SYMM_DATA["generator_matrices"]
POINT_GROUP_ENC = SYMM_DATA["point_group_encoding"]
SPACE_GROUP_ENC = SYMM_DATA["space_group_encoding"]
ABBREV_SPACE_GROUP_MAPPING = SYMM_DATA["abbreviated_spacegroup_symbols"]
TRANSLATIONS = {k: Fraction(v) for k, v in SYMM_DATA["translations"].items()}
FULL_SPACE_GROUP_MAPPING = {
    v["full_symbol"]: k for k, v in SYMM_DATA["space_group_encoding"].items()}
MAXIMAL_SUBGROUPS = {int(k): v
                     for k, v in SYMM_DATA["maximal_subgroups"].items()}


class SymmetryGroup(Sequence):
    __metaclass__ = ABCMeta

    @abstractproperty
    def symmetry_ops(self):
        pass

    # def __contains__(self, item):
    def __getitem__(self, item):
        return self.symmetry_ops[item]

    def __len__(self):
        return len(self.symmetry_ops)


class SpaceGroup(SymmetryGroup):
    SG_SYMBOLS = tuple(SPACE_GROUP_ENC.keys())

    def __init__(self, int_symbol):
        if int_symbol not in SPACE_GROUP_ENC and \
           int_symbol not in ABBREV_SPACE_GROUP_MAPPING and \
           int_symbol not in FULL_SPACE_GROUP_MAPPING:
            raise ValueError("Bad international symbol %s" % int_symbol)
        elif int_symbol in ABBREV_SPACE_GROUP_MAPPING:
            int_symbol = ABBREV_SPACE_GROUP_MAPPING[int_symbol]
        elif int_symbol in FULL_SPACE_GROUP_MAPPING:
            int_symbol = FULL_SPACE_GROUP_MAPPING[int_symbol]

        data = SPACE_GROUP_ENC[int_symbol]

        self.symbol = int_symbol
        enc = list(data["enc"])
        inversion = int(enc.pop(0))
        ngen = int(enc.pop(0))
        symm_ops = [np.eye(4)]
        if inversion:
            symm_ops.append(np.array(
                [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
        for i in range(ngen):
            m = np.eye(4)
            m[:3, :3] = GENERATOR_MATRICES[enc.pop(0)]
            m[0, 3] = TRANSLATIONS[enc.pop(0)]
            m[1, 3] = TRANSLATIONS[enc.pop(0)]
            m[2, 3] = TRANSLATIONS[enc.pop(0)]
            symm_ops.append(m)
        self.generatos = symm_ops
        self.full_symbol = data["full_symbol"]
        self.int_number = data["int_number"]
        self.order = data["order"]
        self.patterson_symmetry = data["patterson_symmetry"]
        self.point_group = data["point_group"]
        self._symmetry_ops = None

    def _generate_full_symmetry_ops(self):
        symm_ops = np.array(self.generatos)
        for op in symm_ops:
            op[:3, 3] = np.mod(op[:3, 3], 1)
        new_ops = symm_ops
        while len(new_ops) > 0 and len(symm_ops) < self.order:
            gen_ops = []
            for g in new_ops:
                temp_ops = np.einsum('ijk,kl', symm_ops, g)
                for op in temp_ops:
                    op[0:3, 3] = np.mod(op[0:3, 3], 1)
                    ind = np.where(np.abs(1 - op[0:3, 3]) < 1e-5)
                    op[ind, 3] = 0
                    if not in_array_list(symm_ops, op):
                        gen_ops.append(op)
                        symm_ops = np.append(symm_ops, [op], axis=0)
            new_ops = gen_ops
        assert len(symm_ops) == self.order
        return symm_ops

    def is_compatible(self, lattice, tol=1e-5, angle_tol=5):
        abc, angles = lattice.lengths_and_angles
        crys_system = self.crystal_system

        def check(param, ref, tolerance):
            return all([abs(i - j) < tolerance for i, j in zip(param, ref)
                       if j is not None])

        if crys_system == "cubic":
            a = abc[0]
            return check(abc, [a, a, a], tol) and \
                check(angles, [90, 90, 90], angle_tol)
        elif crys_system == "hexagonal" or (crys_system == "trigonal" and
                                            self.symbol.endswith('H')):
            a = abc[0]
            return check(abc, [a, a, None], tol) and \
                check(angles, [90, 90, 120], angle_tol)
        elif crys_system == "trigonal":
            a = abc[0]
            return check(abc, [a, a, a], tol)
        elif crys_system == "tetragonal":
            a = abc[0]
            return check(abc, [a, a, None], tol) and\
                check(angles, [90, 90, 90], angle_tol)
        elif crys_system == "orthorhombic":
            return check(angles, [90, 90, 90], angle_tol)
        elif crys_system == "monoclinic":
            return check(angles, [90, None, 90], angle_tol)
        return True

    def get_orbit(self, p, tol=1e-5):
        orbit = []
        for o in self.symmetry_ops:
            pp = o.operate(p)
            pp = np.mod(np.round(pp, decimals=10), 1)
            if not in_array_list(orbit, pp, tol=tol):
                orbit.append(pp)
        return orbit

    @property
    def crystal_system(self):
        i = self.int_number
        if i <= 2:
            return "triclinic"
        elif i <= 15:
            return "monoclinic"
        elif i <= 74:
            return "orthorhombic"
        elif i <= 142:
            return "tetragonal"
        elif i <= 167:
            return "trigonal"
        elif i <= 194:
            return "hexagonal"
        else:
            return "cubic"

    @property
    def symmetry_ops(self):
        if self._symmetry_ops is None:
            self._symmetry_ops = [
                SymmOp(m) for m in self._generate_full_symmetry_ops()]
        return self._symmetry_ops

    @classmethod
    def from_int_number(cls, int_number, hexagonal=True):
        return SpaceGroup(sg_symbol_from_int_number(int_number,
                                                    hexagonal=hexagonal))


def sg_symbol_from_int_number(int_number, hexagonal=True):
    syms = []
    for n, v in SPACE_GROUP_ENC.items():
        if v["int_number"] == int_number:
            syms.append(n)
    if len(syms) == 0:
        raise ValueError("Invalid international number!")
    if len(syms) == 2:
        if hexagonal:
            syms = filter(lambda s: s.endswith("H"), syms)
        else:
            syms = filter(lambda s: not s.endswith("H"), syms)
    return syms.pop()


def in_array_list(array_list, a, tol=1e-5):
    if len(array_list) == 0:
        return False
    axes = tuple(range(1, a.ndim + 1))
    if not tol:
        return np.any(np.all(np.equal(array_list, a[None, :]), axes))
    else:
        return np.any(np.sum(np.abs(array_list - a[None, :]), axes) < tol)
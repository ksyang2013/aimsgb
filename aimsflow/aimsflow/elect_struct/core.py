from enum import Enum, unique


@unique
class Spin(Enum):
    up, dn = (1, -1)

    def __int__(self):
        return self.value

    def __str__(self):
        return str(self.value)


@unique
class OrbitalType(Enum):
    s = 0
    p = 1
    d = 2
    f = 3

    def __str__(self):
        return self.name


@unique
class Orbital(Enum):
    s = 0
    py = 1
    pz = 2
    px = 3
    dxy = 4
    dyz = 5
    dz2 = 6
    dxz = 7
    dx2 = 8
    f_3 = 9
    f_2 = 10
    f_1 = 11
    f0 = 12
    f1 = 13
    f2 = 14
    f3 = 15

    def __int__(self):
        return self.value

    def __str__(self):
        return self.name

    @property
    def orbital_type(self):
        return OrbitalType[self.name[0]]


class Color(Enum):
    s = 0
    p = 1
    py = 2
    pz = 3
    px = 4
    d = 5
    dxy = 6
    dyz = 7
    dz2 = 8
    dxz = 9
    dx2 = 10
    f = 11

    def __int__(self):
        return self.value

    def __str__(self):
        return self.name
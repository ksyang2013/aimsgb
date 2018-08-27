import six
import numpy as np
from collections import OrderedDict

from aimsflow.elect_struct.core import OrbitalType


class Dos(object):
    def __init__(self, efermi, energies, densities):
        self.efermi = efermi
        self.energies = np.array(energies)
        self.densities = OrderedDict((k, np.array(d)) for k, d in densities.items())


class CompleteDos(Dos):
    def __init__(self, total_dos, pdoss):
        super(CompleteDos, self).__init__(
            total_dos.efermi, energies=total_dos.energies,
            densities=OrderedDict((k, np.array(d)) for k, d in total_dos.densities.items()))
        self.pdos = pdoss

    def get_pdos_combine(self, site_num):
        site_dos = six.moves.reduce(add_densities, self.pdos[site_num].values())
        ion_num = "Ion-%s" % (site_num + 1)
        return {ion_num: Dos(self.efermi, self.energies, site_dos)}

    def get_sites_pdos_spdf(self, site_nums):
        pdos_split = OrderedDict()
        for s in site_nums:
            for orb, pdos in self.pdos[s].items():
                orb_type = get_orb_type(orb)
                if orb_type not in pdos_split:
                    pdos_split[orb_type] = pdos
                else:
                    pdos_split[orb_type] = add_densities(pdos_split[orb_type], pdos)
        return OrderedDict((orb, Dos(self.efermi, self.energies, densities))
                           for orb, densities in pdos_split.items())

    def get_pdos_spdf(self, site_num):
        pdos_split = OrderedDict()
        for orb, pdos in self.pdos[site_num].items():
            orb_type = get_orb_type(orb)
            if orb_type not in pdos_split:
                pdos_split[orb_type] = pdos
            else:
                pdos_split[orb_type] = add_densities(pdos_split[orb_type], pdos)
        return OrderedDict((orb, Dos(self.efermi, self.energies, densities))
                           for orb, densities in pdos_split.items())

    def get_pdos_full(self, site_num):
        pdos_full = OrderedDict((orb, pdos)
                                for orb, pdos in self.pdos[site_num].items())
        return OrderedDict((orb, Dos(self.efermi, self.energies, densities))
                           for orb, densities in pdos_full.items())

    def get_pdos_t2g_eg(self, site_num):
        t2g_eg = OrderedDict()
        for orb, pdos in self.pdos[site_num].items():
            orb_type = get_orb_type(orb)
            if orb_type == OrbitalType.d:
                t2g_eg[orb] = pdos
        return OrderedDict((orb, Dos(self.efermi, self.energies, densities))
                           for orb, densities in t2g_eg.items())

    def get_pdos_pxyz(self, site_num):
        pxyz = OrderedDict()
        for orb, pdos in self.pdos[site_num].items():
            orb_type = get_orb_type(orb)
            if orb_type == OrbitalType.p:
                pxyz[orb] = pdos
        return OrderedDict((orb, Dos(self.efermi, self.energies, densities))
                           for orb, densities in pxyz.items())

    def get_tdos_spdf(self):
        spdf = OrderedDict()
        for atom_dos in self.pdos.values():
            for orb, pdos in atom_dos.items():
                orb_type = get_orb_type(orb)
                if orb_type not in spdf:
                    spdf[orb_type] = pdos
                else:
                    spdf[orb_type] = add_densities(spdf[orb_type], pdos)
        return OrderedDict((orb, Dos(self.efermi, self.energies, densities))
                           for orb, densities in spdf.items())

    def get_tdos_t2g_eg(self):
        t2g_eg = OrderedDict()
        for atom_dos in self.pdos.values():
            for orb, pdos in atom_dos.items():
                orb_type = get_orb_type(orb)
                if orb_type == OrbitalType.d:
                    if orb not in t2g_eg:
                        t2g_eg[orb] = pdos
                    else:
                        t2g_eg[orb] = add_densities(t2g_eg[orb], pdos)
        return OrderedDict((orb, Dos(self.efermi, self.energies, densities))
                           for orb, densities in t2g_eg.items())


def add_densities(density1, density2):
    return OrderedDict((spin, np.array(density1[spin]) + np.array(density2[spin]))
                       for spin in density1.keys())


def get_orb_type(orb):
    try:
        return orb.orbital_type
    except AttributeError:
        return orb
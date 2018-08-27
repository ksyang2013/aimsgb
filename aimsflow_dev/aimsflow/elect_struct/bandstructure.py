import math
import numpy as np
from numpy.linalg import norm
from collections import OrderedDict
from aimsflow.elect_struct.core import Spin
from aimsflow.util.num_utils import OrderedDefaultDict


class Kpoint(object):
    def __init__(self, coords, lattice, to_unit_cell=False,
                 coords_are_cartesian=False, label=None):
        self._lattice = lattice
        self._fcoords = lattice.get_frac_coords(coords) \
            if coords_are_cartesian else coords
        self._label = label

        if to_unit_cell:
            for i in range(len(self._fcoords)):
                self._fcoords[i] -= math.floor(self._fcoords[i])
        self._coords = lattice.get_cart_coords(self._fcoords)

    def __str__(self):
        return "{} {} {}".format(self.frac_coords, self.cart_coords, self.label)

    def __repr__(self):
        return self.__str__()

    @property
    def label(self):
        return self._label

    @property
    def cart_coords(self):
        return self._coords

    @property
    def frac_coords(self):
        return self._fcoords


class BandStructure(object):
    def __init__(self, kpoints, eigenvalues, lattice, efermi, labels_dict,
                 coords_are_cartesian=False):
        self.kpoints = []
        self.bands = eigenvalues
        self.lattice_rec = lattice
        self.efermi = efermi
        self.labels_dict = {}

        for k in kpoints:
            label = None
            for key, value in labels_dict.items():
                if norm(k - np.array(value)) < 1e-4:
                    label = key
                    self.labels_dict[label] = Kpoint(
                        k, lattice, label=label,
                        coords_are_cartesian=coords_are_cartesian)
            self.kpoints.append(Kpoint(k, lattice, label=label,
                                       coords_are_cartesian=coords_are_cartesian))
        self.num_bands = len(eigenvalues[Spin.up])
        self.is_spin_polarized = len(eigenvalues) == 2

        self.distance = []
        self.branches = []
        one_group = []
        branches_tmp = []
        previous_kpoint = self.kpoints[0]
        previous_distance = 0.0
        previous_label = self.kpoints[0].label
        for i, value in enumerate(self.kpoints):
            label = value.label
            if label is not None and previous_label is not None:
                self.distance.append(previous_distance)
            else:
                self.distance.append(
                    norm(value.cart_coords - previous_kpoint.cart_coords) +
                    previous_distance)
            previous_kpoint = value
            previous_distance = self.distance[i]
            if label and previous_label:
                if len(one_group) != 0:
                    branches_tmp.append(one_group)
                one_group = []
            previous_label = label
            one_group.append(i)

        if len(one_group) != 0:
            branches_tmp.append(one_group)
        for b in branches_tmp:
            self.branches.append({"start_index": b[0], "end_index": b[-1],
                                  "name": "%s-%s" % (self.kpoints[b[0]].label,
                                                     self.kpoints[b[-1]].label)})

    def is_metal(self):
        outs = OrderedDict()
        for spin, bands in self.bands.items():
            outs[spin] = False
            for band in bands:
                if np.any(band < self.efermi) and np.any(band > self.efermi):
                    outs[spin] = True
        return outs

    def get_cbm(self):
        ind = None
        energy = None
        cbm = OrderedDefaultDict(dict)
        for spin, v in self.bands.items():
            if self.is_metal()[spin]:
                cbm[spin]["enery"] = None
                cbm[spin] = {"energy": None, "label": None, "kpoint": None,
                             "band_index": [], "kpoint_index": []}
                continue
            ind_band = []
            ind_kpoint = []
            max_tmp = float("inf")
            for i, j in zip(*np.where(v > self.efermi)):
                if v[i, j] < max_tmp:
                    ind = j
                    energy = float(v[i, j])
                    kpoint_cbm = self.kpoints[j]
                    max_tmp = float(v[i, j])

            label_cbm = kpoint_cbm.label
            if label_cbm is not None:
                for i in range(len(self.kpoints)):
                    if self.kpoints[i].label == label_cbm:
                        ind_kpoint.append(i)
            else:
                ind_kpoint.append(ind)

            for i in range(self.num_bands):
                if math.fabs(self.bands[spin][i][ind] - max_tmp) < 1e-3:
                    ind_band.append(i)
            cbm[spin] = {"energy": energy, "label": label_cbm,
                         "kpoint": kpoint_cbm, "band_index": ind_band,
                         "kpoint_index": ind_kpoint}
        return cbm

    def get_vbm(self):
        ind = None
        energy = None
        vbm = OrderedDefaultDict(dict)
        for spin, v in self.bands.items():
            if self.is_metal()[spin]:
                vbm[spin]["enery"] = None
                continue
            ind_band = []
            ind_kpoint = []
            max_tmp = -float("inf")
            for i, j in zip(*np.where(v < self.efermi)):
                if v[i, j] > max_tmp:
                    ind = j
                    energy = float(v[i, j])
                    kpoint_cbm = self.kpoints[j]
                    max_tmp = float(v[i, j])

            label_cbm = kpoint_cbm.label
            if label_cbm is not None:
                for i in range(len(self.kpoints)):
                    if self.kpoints[i].label == label_cbm:
                        ind_kpoint.append(i)
            else:
                ind_kpoint.append(ind)

            for i in range(self.num_bands):
                if math.fabs(self.bands[spin][i][ind] - max_tmp) < 1e-3:
                    ind_band.append(i)
            vbm[spin] = {"energy": energy,
                         "label": label_cbm,
                         "kpoint": kpoint_cbm,
                         "band_index": ind_band,
                         "kpoint_index": ind_kpoint}
        return vbm

    def get_band_gap(self):
        cbm = self.get_cbm()
        vbm = self.get_vbm()
        band_gap = OrderedDefaultDict(dict)
        for spin in cbm:
            band_gap[spin]["energy"] = cbm[spin]["energy"] - vbm[spin]["energy"]
            vbm_label = vbm[spin]["label"] if vbm[spin]["label"] is not None else\
                str(vbm[spin]["kpoint_index"][0])
            cbm_label = cbm[spin]["label"] if cbm[spin]["label"] is not None else\
                str(cbm[spin]["kpoint_index"][0])
            band_gap[spin]["label"] = vbm_label + "-" + cbm_label
        return band_gap


class BandStructureProj(BandStructure):
    def __init__(self, kpoints, eigenvalues, lattice, efermi, labels_dict,
                 coords_are_cartesian=False, tot_proj=None):
        super(BandStructureProj, self).__init__(
            kpoints, eigenvalues, lattice, efermi, labels_dict,
            coords_are_cartesian)
        self.tot_proj = tot_proj
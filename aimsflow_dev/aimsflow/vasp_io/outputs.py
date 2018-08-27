from __future__ import division

import os
import re
import glob
import math
import warnings
import numpy as np
from functools import reduce
from collections import OrderedDict, defaultdict
import xml.etree.cElementTree as ET

from aimsflow import Structure, Lattice, Element
from aimsflow.util import OrderedDefaultDict, find_re_pattern, zopen, \
    file_to_lines, file_to_str, flatten_lists
from aimsflow.elect_struct.core import Spin, OrbitalType, Orbital
from aimsflow.elect_struct.dos import Dos, CompleteDos
from aimsflow.elect_struct.bandstructure import BandStructure, BandStructureProj
from aimsflow.vasp_io import Incar, Kpoints, Poscar


def parse_vasp_float(num):
    try:
        return float(num)
    except ValueError:
        parts = num.split(".")
        decimal_number = len(parts[-1])
        num1 = '.'.join([parts[0], parts[1][:decimal_number]])
        num2 = '.'.join([parts[1].replace(parts[1][:decimal_number], ""), parts[-1]])
    return [float(num1), float(num2)]


def _parse_varray(elem):
    return [[_vasprun_float(i) for i in v.text.split()] for v in elem]


def _parse_parameters(val_type, val):
    if val_type == "logical":
        return val == "T"
    elif val_type == "int":
        return int(val)
    elif val_type == "string":
        return val.strip()
    else:
        return float(val)


def _parse_v_parameters(val_type, val, filename, param_name):
    if val_type == "logical":
        val = [i == "T" for i in val.split()]
    elif val_type == "int":
        try:
            val = [int(i) for i in val.split()]
        except ValueError:
            # Fix for stupid error in vasprun sometimes which displays
            # LDAUL/J as 2****
            val = _parse_from_incar(filename, param_name)
            if val is None:
                raise IOError("Error in parsing vasprun.xml")
    elif val_type == "string":
        val = val.split()
    else:
        try:
            [float(i) for i in val.split()]
        except ValueError:
            # Fix for stupid error in vasprun sometimes which displays
            # MAGMOM as 2****
            val = _parse_from_incar(filename, param_name)
            if val is None:
                raise IOError("Error in parsing vasprun.xml")
    return val


def _parse_from_incar(filename, key):
    dirname = os.path.dirname(os.path.abspath(filename))
    for f in os.listdir(dirname):
        if re.search("INCAR", f):
            warnings.warn("INCAR found. Using " + key + " from INCAR.")
            incar = Incar.from_file(os.path.join(dirname, f))
            if key in incar:
                return incar[key]
            else:
                return None
    return None


def _vasprun_float(f):
    """
    Large numbers are often represented as ********* in the vasprun.
    This function parses these values as np.nan
    """
    try:
        return float(f)
    except ValueError as e:
        f = f.strip()
        if f == '*' * len(f):
            warnings.warn("Float overflow (*******) encountered in vasprun")
            return np.nan
        raise e


class Vasprun(object):
    def __init__(self, filename, parse_dos=False, parse_eigen=False):
        self.filename = filename
        with zopen(filename, "rt") as f:
            self._parse(f, parse_dos, parse_eigen)

    def _parse(self, stream, parse_dos, parse_eigen):
        for event, elem in ET.iterparse(stream):
            tag = elem.tag
            if tag == "generator":
                self.generator = self._parse_params(elem)
            elif tag == "incar":
                self.incar = self._parse_params(elem)
            elif parse_dos and tag == 'dos':
                self.tdos, self.pdos = self._parse_dos(elem)
                self.efermi = self.tdos.efermi
            elif tag == "atominfo":
                self.atomic_symbols, self.potcar_symbol = self._parse_atominfo(elem)
            elif tag == "structure" and elem.attrib.get("name") == "finalpos":
                self.struct = self._parse_structure(elem)
            # elif tag == "eigenvalues" and parse_eigen:

    def _parse_dos(self, elem):
        efermi = float(elem.find("i").text)
        energies = None
        tdensities = {}

        for s in elem.find("total").find("array").find("set").findall("set"):
            data = np.array(_parse_varray(s))
            energies = data[:, 0]
            spin = Spin.up if s.attrib["comment"] == "spin 1" else Spin.dn
            tdensities[spin] = data[:, 1]

        pdoss = []
        partial = elem.find("partial")
        orbs = [ss.text for ss in partial.find("array").findall("field")]
        orbs.pop(0)
        lm = any(['x' in s for s in orbs])
        for s in partial.find("array").find("set").findall("set"):
            pdos = defaultdict(dict)
            for ss in s.findall("set"):
                spin = Spin.up if ss.attrib["comment"] == "spin 1" else Spin.dn
                data = np.array(_parse_varray(ss))
                nrow, ncol = data.shape
                for j in range(1, ncol):
                    if lm:
                        orb = Orbital(j - 1)
                    else:
                        orb = OrbitalType(j - 1)
                    pdos[orb][spin] = data[:, j]
            pdoss.append(pdos)
        elem.clear()
        return Dos(efermi, energies, tdensities), pdoss

    def _parse_structure(self, elem):
        latt = _parse_varray(elem.find("crystal").find("varray"))
        pos = _parse_varray(elem.find("varray"))
        return Structure(latt, self.atomic_symbols, pos)

    def _parse_params(self, elem):
        params = {}
        for c in elem:
            name = c.attrib.get("name")
            if c.tag not in ("i", "v"):
                p = self._parse_params(c)
                if name == "response functions":
                    p = {k: v for k, v in p.items() if k not in params}
                params.update(p)
            else:
                ptype = c.attrib.get("type")
                val = c.text.strip() if c.text else ""
                if c.tag == "i":
                    params[name] = _parse_parameters(ptype, val)
                else:
                    params[name] = _parse_v_parameters(ptype, val,
                                                       self.filename, name)
        elem.clear()
        return Incar(params)

    def _parse_atominfo(self, elem):
        def parse_atomic_symbol(symbol):
            try:
                return str(Element(symbol))
            except ValueError as e:
                if symbol == "X":
                    return "Xe"
                elif symbol == "r":
                    return "Zr"
                raise e

        for a in elem.findall("array"):
            if a.attrib["name"] == "atoms":
                atomic_symbols = [rc.find("c").text.strip()
                                  for rc in a.find("set")]
            elif a.attrib["name"] == "atomtypes":
                potcar_symbols = [rc.findall("c")[4].text.strip()
                                  for rc in a.find("set")]
        return [parse_atomic_symbol(sym)
                for sym in atomic_symbols], potcar_symbols

    # def _parse_eigen(self, elem):
    @property
    def complete_dos(self):
        struct = self.struct
        pdoss = {struct[i]: pdos for i, pdos in enumerate(self.pdos)}
        return CompleteDos(self.tdos, pdoss)


class Doscar(object):
    def __init__(self, filename):
        self.filename = filename
        self.tdos, self.pdos = self.parse_dos()

    def parse_dos(self):
        chunks = file_to_lines(self.filename)[5:]
        emax, emin, nedos, efermi = map(float, chunks[0].split()[:-1])
        nedos = int(nedos)
        chunks.pop(0)
        data = np.array([[float(j) for j in i.split()] for i in chunks[:nedos]])
        chunks = chunks[nedos:]

        path = os.path.dirname(os.path.abspath(self.filename))
        try:
            incar = Incar.from_file(glob.glob("%s/INCAR*" % path)[-1])
        except IndexError:
            raise IndexError("Cannot find INCAR in %s" % path)
        is_spin = True if incar.get("ISPIN", 1) == 2 else False
        lm = True if incar.get("LORBIT", 0) == 11 else False
        is_soc = True if incar.get("LSORBIT") else False

        tdensities = OrderedDict()
        energies = data[:, 0]
        tdensities[Spin.up] = data[:, 1]
        if is_spin:
            tdensities[Spin.dn] = data[:, 2]

        pdoss = []
        num_sites = len(chunks) // (nedos + 1)
        ncol = len(chunks[1].split())

        for i in range(num_sites):
            chunks.pop(0)
            pdos = OrderedDefaultDict(dict)
            data = np.array([[float(k) for k in j.split()]
                             for j in chunks[:nedos]])
            if is_spin:
                for j in range(1, ncol, 2):
                    if lm:
                        orb = Orbital(j // 2)
                    else:
                        orb = OrbitalType(j // 2)
                    pdos[orb] = OrderedDict(((Spin.up, data[:, j]),
                                             (Spin.dn, data[:, j + 1])))
            elif is_soc:
                for j in range(1, ncol, 4):
                    if lm:
                        orb = Orbital(j // 4)
                    else:
                        orb = OrbitalType(j // 4)
                    pdos[orb][Spin.up] = data[:, j]
            else:
                for j in range(1, ncol):
                    if lm:
                        orb = Orbital(j - 1)
                    else:
                        orb = OrbitalType(j - 1)
                    pdos[orb][Spin.up] = data[:, j]
            pdoss.append(pdos)
            chunks = chunks[nedos:]
        return Dos(efermi, energies, tdensities), pdoss

    @property
    def complete_dos(self):
        pdoss = {i: pdos for i, pdos in enumerate(self.pdos)}
        return CompleteDos(self.tdos, pdoss)


class Locpot(object):
    def __init__(self, filename):
        self.filename = filename
        self.structure, self.dim, self.pot_data = self.parse_locpot()

    def parse_locpot(self):
        chunks = file_to_lines(self.filename)
        read_poscar = True
        poscar_string = []
        pot_data = {}
        read_pot = False
        dim = None
        dimline = None
        count = 0
        for line in chunks:
            line = line.strip()
            if read_poscar:
                if line != "":
                    poscar_string.append(line)
                else:
                    poscar = Poscar.from_string("\n".join(poscar_string))
                    read_poscar = False
            elif read_pot:
                spin = Spin.up if pot_data.get(Spin.up) is None else Spin.dn
                for i in line.split():
                    if count < pot_num:
                        # Fist fill x, then y, then z
                        x = count % dim[0]
                        y = int(math.floor(count / dim[0])) % dim[1]
                        z = int(math.floor(count / dim[0] / dim[1]))
                        data[x, y, z] = float(i)
                        count += 1
                if count >= pot_num:
                    read_pot = False
                    count = 0
                    pot_data[spin] = data
            elif dim is None:
                dim = [int(i) for i in line.split()]
                pot_num = reduce(lambda x, y: x * y, dim)
                dimline = line
                read_pot = True
                data = np.zeros(dim)
            elif line == dimline:
                read_pot = True
                data = np.zeros(dim)
        return poscar.structure, dim, pot_data

    @property
    def plan_average(self):
        abc = self.structure.lattice.abc
        da, db, dc = [l / n for l, n in zip(abc, self.dim)]
        s = abc[0] * abc[1]
        all_pot = defaultdict(list)
        for spin, pot in self.pot_data.items():
            for i in range(self.dim[2]):
                all_pot[spin].append(sum(flatten_lists(pot[:, :, i]))
                                     * da * db / s)
        all_z = [dc * i for i in range(self.dim[2])]
        return all_pot, all_z

    def marc_average(self, gap):
        ngz = self.dim[2]
        all_pot, all_z = self.plan_average

        marc_pot = defaultdict(list)
        marc_z = defaultdict(list)
        for spin, values in all_pot.items():
            get_min = True
            x = []
            y = []
            for i in range(0, ngz - 1):
                if get_min:
                    if values[i + 1] > values[i] and values[i] < 0:
                        y.append(i)
                        x.append(all_z[i])
                        get_min = False
                        if len(x) > 1 and x[-1] - x[-2] < 0.5:
                            del x[-1]
                            del y[-1]
                elif values[i + 1] < values[i]:
                    get_min = True
            for i in range((len(x) - 1) // gap):
                area = np.trapz(values[y[gap * i]:y[gap * i + gap] + 1],
                                all_z[y[gap * i]:y[gap * i + gap] + 1])
                marc_pot[spin].append(area / (all_z[y[gap * i + gap]] -
                                              all_z[y[gap * i]]))
                marc_z[spin].append(x[gap * i + 1])
        return marc_pot, marc_z


class Eigenval(object):
    def __init__(self, filename):
        self.filename = filename
        self.kpoints, self.eigenvalues = self.parse_eigen()

    def parse_eigen(self):
        chunks = file_to_lines(self.filename)[5:]
        nelect, num_kpts, num_bands = map(int, chunks[0].split())
        is_spin = True if len(chunks[3].split()) > 3 else False
        chunks.pop(0)
        kpoints = []
        eigenvalues = OrderedDefaultDict(list)
        for i in range(num_kpts):
            kpoints.append([float(i) for i in chunks[1].split()[:3]])
            chunks = chunks[2:]
            data = np.array([[float(k) for k in j.split()]
                             for j in chunks[:num_bands]])
            eigenvalues[Spin.up].append(data[:, 1])
            if is_spin:
                eigenvalues[Spin.dn].append(data[:, 2])
            chunks = chunks[num_bands:]
        return kpoints, eigenvalues

    def get_band_structure(self, kpoints_filename, outcar_filename):
        kpoints_file = Kpoints.from_file(kpoints_filename)
        labels_dict = dict(zip(kpoints_file.labels, kpoints_file.kpts))
        outcar_file = Outcar(outcar_filename)
        lattice_rec = outcar_file.lattice.reciprocal_lattice

        kpoints = [np.array(i) for i in self.kpoints]
        eigenvalues = OrderedDefaultDict(list)
        for spin, v in self.eigenvalues.items():
            eigenvalues[spin] = np.swapaxes(v, 0, 1)
        return BandStructure(kpoints, eigenvalues, lattice_rec,
                             outcar_file.efermi, labels_dict)


class Procar(object):
    def __init__(self, filename, ion_spec=None):
        self.filename = filename
        self.lm, self.kpoints, self.eigenvalues,\
        self.tot_proj, self.ion_proj = self.parse_projection(ion_spec)

    def parse_projection(self, ion_spec=None):
        pattern_vasp = {
            "num_kpts": re.compile('k-points:\s+([\d]+)'),
            "num_bands": re.compile('bands:\s+([\d]+)'),
            "kpoints": re.compile('k-point\s+\d+\s:\s+(\-*[\d\.]+)\s*'
                                  '(\-*[\d\.]+)\s*(\-*[\d\.]+)', re.DOTALL),
            "eigen": re.compile('energy\s+([\d\.\-+]+)', re.DOTALL),
            "tot_orb": re.compile('^\s*tot\s+(.*)', re.MULTILINE)
        }
        file_str = file_to_str(self.filename)
        lm = True if "lm decomposed" in file_str else False
        outs = find_re_pattern(pattern_vasp, file_str)
        is_spin = True if isinstance(outs['num_kpts'], list) else False
        try:
            num_kpts = int(outs['num_kpts'])
            num_bands = int(outs['num_bands'])
        except TypeError:
            num_kpts = int(outs['num_kpts'][0])
            num_bands = int(outs['num_bands'][0])
        # num_ions = int(outs["num_ions"])
        tot_eigen = num_kpts * num_bands
        kpoints = [np.array(i) for i in [[float(j) for j in a]
                                         for a in outs["kpoints"][:num_kpts]]]

        each_tot = []
        for i, v in enumerate(outs["tot_orb"]):
            tot_orb = OrderedDict()
            tmp = v.split()
            for k, vv in enumerate(tmp[:-1]):
                if lm:
                    orb = Orbital(k)
                else:
                    orb = OrbitalType(k)
                tot_orb[orb] = float(vv)
            if len(tot_orb) > 9:
                orb_f = sum([tot_orb[Orbital(k)] for k in range(9, 16)])
                for k in range(9, 16):
                    del tot_orb[Orbital(k)]
                tot_orb[OrbitalType(3)] = orb_f
            each_tot.append(float(tmp[-1]))
            tot_orb["tot"] = float(tmp[-1])
            outs["tot_orb"][i] = tot_orb
        tot_proj = OrderedDict()
        tot_proj[Spin.up] = np.asarray(outs["tot_orb"][:tot_eigen]).reshape(
            num_kpts, num_bands)

        eigenvalues = OrderedDict()
        eigen = np.asarray([float(i) for i in outs["eigen"]])
        eigenvalues[Spin.up] = eigen[:tot_eigen].reshape(num_kpts, num_bands)
        if is_spin:
            tot_proj[Spin.dn] = np.asarray(outs["tot_orb"][tot_eigen:]).reshape(
                num_kpts, num_bands)
            eigenvalues[Spin.dn] = eigen[tot_eigen:].reshape(num_kpts, num_bands)

        ion_proj = None
        if ion_spec:
            outs = re.findall('tot\n(.*?)\ntot', file_str, re.DOTALL)
            for i, v in enumerate(outs):
                ion = []
                vv = v.split("\n")[ion_spec]
                ion_orb = OrderedDict()
                tmp = vv.split()
                for k, vvv in enumerate(tmp[1:-1]):
                    if lm:
                        orb = Orbital(k)
                    else:
                        orb = OrbitalType(k)
                    ion_orb[orb] = float(vvv)
                if len(ion_orb) > 9:
                    orb_f = sum([ion_orb[Orbital(k)] for k in range(9, 16)])
                    for k in range(9, 16):
                        del ion_orb[Orbital(k)]
                    ion_orb[OrbitalType(3)] = orb_f
                ion_orb["tot"] = each_tot[i]
                ion.append(ion_orb)
                outs[i] = ion
            ion_proj = OrderedDict()
            ion_proj[Spin.up] = np.asarray(
                outs[:tot_eigen]).reshape(num_kpts, num_bands)
            if is_spin:
                ion_proj[Spin.dn] = np.asarray(
                    outs[tot_eigen:]).reshape(num_kpts, num_bands)
        return lm, kpoints, eigenvalues, tot_proj, ion_proj

    def get_band_structure(self, kpoints_filename, outcar_filename,
                           orb_spec=None):
        kpoints_file = Kpoints.from_file(kpoints_filename)
        labels_dict = dict(zip(kpoints_file.labels, kpoints_file.kpts))
        outcar_file = Outcar(outcar_filename)
        lattice_rec = outcar_file.lattice.reciprocal_lattice

        kpoints = [np.array(i) for i in self.kpoints]
        eigenvalues = OrderedDefaultDict(list)
        for spin, v in self.eigenvalues.items():
            eigenvalues[spin] = np.swapaxes(v, 0, 1)

        tot_proj = self.ion_proj or self.tot_proj

        if orb_spec:
            for spin, values in tot_proj.items():
                for i, value in enumerate(values):
                    for j, v in enumerate(value):
                        orb = OrderedDict([(k, v[k]) for k in v
                                           if str(k)[0] == orb_spec])
                        orb["tot"] = v["tot"]
                        tot_proj[spin][i][j] = orb

        return BandStructureProj(kpoints, eigenvalues, lattice_rec,
                                 outcar_file.efermi, labels_dict,
                                 tot_proj=tot_proj)


class Outcar(object):
    def __init__(self, filename, parse_mag=False):
        self.filename = filename
        date = None
        efermi = None
        lattice = []
        total_energy = None
        iteration = None
        e_change = None
        p_elc = None
        p_ion = None
        err_msg = []
        born_charge = []
        e_soc = []
        orb_soc = []
        mag = defaultdict(list)
        orb_mag = defaultdict(list)

        date_patt = re.compile("\sdate\s(\S+)\s+(\S+)")
        efermi_patt = re.compile("E-fermi :\s*(\S+)")
        toten_patt = re.compile("TOTEN\s+=\s+(-[\d\.]+)")
        iter_patt = re.compile("Iteration\s+\d\(\s*(\d+)")
        echg_patt = re.compile("energy-change.*:\s*(.*?)\s\((.*?)\)")
        e_soc_patt = re.compile("E_soc:\s+(-[\d\.]+)")
        p_ion_patt = re.compile("p\[ion\]=\(\s+([-\w\.]*)\s+([-\w\.]*)\s+([-\w\.]*)\s")
        p_elc_patt = re.compile("p\[elc\]=\(\s+([-\w\.]*)\s+([-\w\.]*)\s+([-\w\.]*)\s")
        chunks = file_to_lines(filename)
        for line in chunks:
            if "NIONS =" in line:
                natom = int(line.split("NIONS =")[1])
            elif "BORN EFFECTIVE CHARGES" in line:
                l = chunks.index(line) + 2
                axes = ("x", "y", "z")
                for i in range(natom):
                    tmp = {}
                    for n, j in enumerate(chunks[l + 1:l + 4]):
                        tmp[axes[n]] = [float(k) for k in j.split()[1:]]
                    born_charge.append(tmp)
                    l += 4
            elif "direct lattice vectors" in line:
                lattice = []
                l = chunks.index(line) + 1
                for i in chunks[l:l + 3]:
                    try:
                        lattice.append([float(j) for j in i.split()[:3]])
                    except ValueError:
                        num = [parse_vasp_float(j) for j in i.split()[0:2]]
                        lattice.append(list(reduce(lambda x, y: np.append(x, y),
                                                   num)))
            elif "decrease AMIN" in line:
                err_msg.append("decrease AMIN")
            elif "Error EDDDAV" in line:
                err_msg.append("Error EDDDAV")
            m = date_patt.search(line)
            if m:
                date = " ".join([m.group(1), m.group(2)])
            m = efermi_patt.search(line)
            if m:
                # Sometimes VASP output: E-fermi : ********
                try:
                    efermi = float(m.group(1))
                    continue
                except ValueError:
                    efermi = None
                    continue
            m = toten_patt.search(line)
            if m:
                total_energy = float(m.group(1))
                continue
            m = iter_patt.search(line)
            if m:
                iteration = int(m.group(1))
                continue
            m = echg_patt.search(line)
            if m:
                try:
                    e_change = [float(i) for i in [m.group(1), m.group(2)]]
                except IndexError:
                    continue
            m = p_elc_patt.search(line)
            if m:
                p_elc = np.array([float(i) for i in m.groups()])
            m = p_ion_patt.search(line)
            if m:
                p_ion = np.array([float(i) for i in m.groups()])

        if parse_mag:
            chunks.reverse()
            for i in range(len(chunks)):
                m = re.search("magnetization \((\w)", chunks[i])
                if m and not mag.get(m.group(1)):
                    chunk = chunks[i - natom - 5:i - 1]
                    chunk.reverse()
                    # If structure only has 1 atom, there is no tot line
                    if natom == 1:
                        chunk = chunk[:-2]
                    header = re.split("\s{2,}", chunk[0].strip())
                    header.pop(0)
                    for j in chunk[2:]:
                        if "---" not in j:
                            tmp = [float(k) for k in j.split()[1:]]
                            mag[m.group(1)].append(
                                dict(zip(header, tmp)))

                m = re.search("orbital moment \((\w)", chunks[i])
                if m and not orb_mag.get(m.group(1)):
                    chunk = chunks[i - natom - 5:i - 1]
                    chunk.reverse()
                    header = re.split("\s{2,}", chunk[0].strip())
                    header.pop(0)
                    for j in chunk[2:]:
                        if "---" not in j:
                            tmp = [float(k) for k in j.split()]
                            orb_mag[m.group(1)].append(
                                dict(zip(header, tmp)))

                m = e_soc_patt.search(chunks[i])
                if m:
                    tmp = {}
                    orb = None
                    e_soc.insert(0, float(m.group(1)))
                    for j in chunks[i - 1:i - 19:-1]:
                        n = re.search("l=\s+(\d)", j)
                        if n:
                            orb = OrbitalType(int(n.group(1)))
                            tmp[orb] = []
                        else:
                            tmp[orb].append(list(map(float, j.split())))
                    orb_soc.insert(0, tmp)

        self.date = date
        self.efermi = efermi
        self.lattice = Lattice(lattice) if lattice else None
        self.total_energy = total_energy
        self.iteration = iteration
        self.e_change = e_change
        self.e_soc = e_soc
        self.p_elc = p_elc
        self.p_ion = p_ion
        self.orb_soc = orb_soc
        self.err_msg = err_msg
        self.born_charge = born_charge
        self.mag = mag if mag else None
        self.orb_mag = orb_mag if orb_mag else None
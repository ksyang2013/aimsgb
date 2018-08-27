from __future__ import division

import re
import os
import sys
import math
import warnings
import itertools
import collections
import numpy as np
from enum import Enum
from numpy.linalg import det
from tabulate import tabulate

from aimsflow.util import clean_lines, zopen, file_to_str, str_to_file, zpath, \
    loadfn, time_to_second, second_to_time, str_delimited, format_float
from aimsflow import Structure, Element, Lattice, PSP_DIR, MANAGER, TIME_TAG

VASP_CONFIG = loadfn(os.path.join(os.path.dirname(__file__),
                                  "af_vasp.yaml"))


def parse_string(s):
    return "{}".format(s.strip())


def parse_bool(s):
    m = re.search("^\.?([T|F|t|f])[A-Za-z]*\.?", s)
    if m:
        if m.group(1) == "T" or m.group(1) == "t":
            return True
        else:
            return False
    raise ValueError(s + " should be a boolean type!")


def parse_float(s):
    return float(re.search("^-?\d*\.?\d*[e|E]?-?\d*", s).group(0))


def parse_int(s):
    return int(re.search("^-?[0-9]+", s).group(0))


def parse_list(s):
    return [float(y) for y in re.split("\s+", s.strip()) if not y.isalpha()]


class Poscar(object):
    def __init__(self, structure, comment=None, selective_dynamics=None,
                 true_names=True):
        if structure.is_ordered:
            site_properties = {}
            if selective_dynamics:
                site_properties["selective_dynamics"] = selective_dynamics
            self.structure = structure.copy(site_properties=site_properties)
            self.true_names = true_names
            self.comment = structure.formula if comment is None else comment
        else:
            raise ValueError("Structure with partial occupancies cannot be "
                             "convered into POSCAR!")

    def __repr__(self):
        return self.get_string()

    def __str__(self):
        return self.get_string()

    @property
    def site_symbols(self):
        syms = [site.specie.symbol for site in self.structure]
        return [a[0] for a in itertools.groupby(syms)]

    @property
    def num_sites(self):
        return sum(self.natoms)

    @property
    def natoms(self):
        syms = [site.specie.symbol for site in self.structure]
        return [len(tuple(a[1])) for a in itertools.groupby(syms)]

    @property
    def selective_dynamics(self):
        return self.structure.site_properties.get("selective_dynamics")

    @staticmethod
    def from_file(filename):
        return Poscar.from_string(file_to_str(filename))

    @staticmethod
    def from_string(data, default_symbol=None):
        chunks = re.split("\n\s*\n", data.rstrip())
        try:
            if chunks[0] == '':
                chunks.pop(0)
        except IndexError:
            raise ValueError("Empty POSCAR")
        lines = tuple(clean_lines(chunks[0].split('\n'), False))
        comment = lines[0]
        scale = float(lines[1])
        lattice = np.asarray([[float(j) for j in i.split()] for i in lines[2:5]])
        if scale < 0:
            vol = abs(det(lattice))
            lattice *= (-scale / vol) ** (1.0 / 3)
        else:
            lattice *= scale
        vasp5_symbols = False
        try:
            atom_nums = [int(i) for i in lines[5].split()]
            line_cursor = 6
        except ValueError:
            vasp5_symbols = True
            symbols = lines[5].split()
            atom_nums = [int(i) for i in lines[6].split()]
            atomic_symbols = []
            for i in range(len(atom_nums)):
                atomic_symbols.extend([symbols[i]] * atom_nums[i])
            line_cursor = 7

        poscar_type = lines[line_cursor].split()[0]
        sdynamics = False
        if poscar_type[0] in 'sS':
            sdynamics = True
            line_cursor += 1
            poscar_type = lines[line_cursor].split()[0]

        cart = poscar_type[0] in 'cCkK'
        num_sites = sum(atom_nums)

        if default_symbol:
            try:
                atomic_symbols = []
                for i in range(len(atom_nums)):
                    atomic_symbols.extend([default_symbol[i]] * atom_nums[i])
                vasp5_symbols = True
            except IndexError:
                pass

        if not vasp5_symbols:
            index = 3 if not sdynamics else 6
            try:
                atomic_symbols = [line.split()[index]
                                  for line in lines[line_cursor + 1: line_cursor + 1 + num_sites]]
                if not all([Element.is_valid_symbol(symbol)
                            for symbol in atomic_symbols]):
                    raise ValueError("Invalid symbol detected.")
                vasp5_symbols = True
            except (ValueError, IndexError):
                atomic_symbols = []
                for i in range(len(atom_nums)):
                    symbol = Element.from_z(i + 1).symbol
                    atomic_symbols.extend([symbol] * atom_nums[i])
                warnings.warn("Atomic symbol in POSCAR cannot be determined.\n"
                              "Defaulting to fake names {}.".format(" ".join(atomic_symbols)))
        coords = []
        select_dynamics = [] if sdynamics else None
        for i in range(num_sites):
            elements = lines[line_cursor + 1 + i].split()
            site_scale = scale if cart else 1
            coords.append([float(j) * site_scale for j in elements[:3]])
            if sdynamics:
                select_dynamics.append([j.upper()[0] == 'T'
                                        for j in elements[3:6]])

        struct = Structure(lattice, atomic_symbols, coords,
                           to_unit_cell=False, validate_proximity=False,
                           coords_cartesian=cart)

        return Poscar(struct, comment, select_dynamics, vasp5_symbols)

    def get_string(self, direct=True, vasp4_compatible=False,
                   significant_figures=6):
        latt = self.structure.lattice
        if np.linalg.det(latt.matrix) < 0:
            latt = Lattice(-latt.matrix)

        lines = [self.comment, "1.0", str(latt)]
        if self.true_names and not vasp4_compatible:
            lines.append(" ".join(self.site_symbols))
        lines.append(" ".join([str(i) for i in self.natoms]))
        if self.selective_dynamics:
            lines.append("Selective dynamics")
        coord_type = "direct" if direct else "cartesian"
        lines.append("{}({})".format(coord_type, self.num_sites))

        format_str = "{{:.{0}f}}".format(significant_figures)
        for i, site in enumerate(self.structure):
            coords = site.frac_coords if direct else site.coords
            line = " ".join([format_str.format(c) for c in coords])
            if self.selective_dynamics is not None:
                sd = ["T" if j else "F" for j in self.selective_dynamics[i]]
                line += 3 * " %s" % tuple(sd)
            line += " " + site.species_string
            lines.append(line)
        return "\n".join(lines) + "\n"

    def write_file(self, filename, **kwargs):
        with zopen(filename, "wt") as f:
            f.write(self.get_string(**kwargs))


class BatchFile(dict):
    """
    BatchFile object for reading and writing a batch script. Essentially is a
    dictionary with batch tags
    """
    def __init__(self, params=None):
        """
        Create BatchFile object
        :param params: (dict) A set of batch tags
        """
        super(BatchFile, self).__init__()
        if params:
            self.update(params)
            self.manager = "PBS" if params.get("walltime") else "SLURM"
        self.queue_limit = {"hotel": "168",
                            "condo": "8",
                            "comet": "48"
                            }
        self.queue_ppn = {"hotel": "16:sandy",
                          "glean": "16:sandy",
                          "home": "28:broadwell",
                          "condo": "24:haswell"
                          }

    def __str__(self):
        return self.get_string()

    @staticmethod
    def from_file(filename):
        return BatchFile.from_string(file_to_str(filename))

    @staticmethod
    def from_string(string):
        lines = list(clean_lines(string.splitlines(), remove_comment=False))
        params = collections.OrderedDict()
        manager = "PBS" if "#PBS" in string else "SLURM"
        command = ""
        others = ""
        others_str = ["module", "ibrun", "mpirun"]
        for line in lines:
            m = re.match("#(?:PBS|SBATCH)\s+(\-\-*\w+\-*\w*\-*\w*)\=*\s*(.*)", line)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                if "nodes" in val:
                    key = "nodes" if manager == "PBS" else "--nodes"
                    val = val.split("nodes=")[-1]
                if "walltime" in val:
                    key = "walltime"
                    val = val.split("walltime=")[-1]
                params[key] = val
            else:
                if "#!" in line:
                    params["header"] = line
                elif any([i in line for i in others_str]):
                    others += line + "\n"
                else:
                    command += line + "\n"
        try:
            exe = re.search("\s(\{*\w+\}*)\s*>\s*vasp.out", others).group(1)
        except AttributeError:
            exe = None
        params.update({"command": command, "others": others, "exe": exe})
        return BatchFile(params)

    @staticmethod
    def convert_batch(old_batch):
        if MANAGER == old_batch.manager:
            sys.stderr.write("The batch is already in %s format\n" % MANAGER)
            return old_batch

        batch_run = VASP_CONFIG[MANAGER]
        new_batch = BatchFile.from_string(batch_run)
        if MANAGER == "SLURM":
            walltime = time_to_second(old_batch["walltime"])
            if walltime > new_batch.time_limit("comet"):
                walltime = new_batch.time_limit("comet")
            new_batch.update({"--mail-user": old_batch.get("-M"),
                              "-J": old_batch.get("-N"),
                              "-t": second_to_time(walltime)})
        else:
            walltime = time_to_second(old_batch["-t"])
            if walltime > new_batch.time_limit("condo"):
                new_batch.change_queue("home")
            new_batch.update({"-M": old_batch.get("--mail-user"),
                              "-N": old_batch.get("-J"),
                              "walltime": old_batch.get("-t")})

        if "aimsflow" in old_batch["command"]:
            command = [i for i in new_batch["command"].split("\n")
                       if i.startswith("cd")]
            command.append(re.findall("(aimsflow.*\n)", old_batch["command"])[0])
            new_batch.update({"command": "\n".join(command), "others": ""})
            err = new_batch["-e"]
            out = new_batch["-o"]
            new_batch.update({"-e": err.replace("err.", "err.check."),
                              "-o": out.replace("out.", "out.check.")})
        else:
            command = "".join(re.findall("(cp.*\n)", old_batch["command"]))
            if "{command}" not in old_batch["command"]:
                new_batch["command"] = new_batch["command"].format(command=command)
        if old_batch["exe"] is not None:
            new_batch["others"] = re.sub("\{*\w+\}*(?=\s*>\s*vasp.out)",
                                         old_batch["exe"], new_batch["others"])
        return BatchFile(new_batch)

    def time_limit(self, queue):
        return time_to_second(self.queue_limit[queue])

    def get_string(self):
        keys = list(self)
        [keys.remove(i) for i in ["header", "others", "command", "exe"]]
        lines = []
        if self.manager == "PBS":
            for k in keys:
                if k in ["walltime", "nodes"]:
                    lines.append(["#PBS -l {}=".format(k), self[k]])
                else:
                    lines.append(["#PBS " + k, self[k]])
            lines.append([self["command"] + self["others"]])
            string = str_delimited(lines, header=[self["header"]], delimiter=" ")
            string = re.sub("nodes=\s+", "nodes=", string)
            return re.sub("walltime=\s+", "walltime=", string)
        else:
            for k in keys:
                lines.append(["#SBATCH " + k, self[k]])
            lines.append([self["command"] + self["others"]])
            return str_delimited(lines, header=[self["header"]], delimiter=" ")

    def write_file(self, filename):
        str_to_file(self.__str__(), filename)

    def change_queue(self, queue):
        if MANAGER == "SLURM":
            raise IOError("In SBATCH system, there is no changing "
                          "queue option.\n")

        ppn = self.queue_ppn[queue]
        self.update({"-q": queue})
        ppn = re.sub("ppn=.*", "ppn={}".format(ppn),
                     self.get("nodes", "1:ppn=16"))
        self.update({"nodes": ppn})
        if queue in self.queue_limit:
            walltime = self[TIME_TAG]
            if time_to_second(walltime) > self.time_limit(queue):
                queue_limit = self.queue_limit[queue]
                sys.stderr.write("Maximum walltime for %s is %s hrs. "
                                 "The current %s will be changed to %s hrs.\n"
                                 % (queue, queue_limit, walltime, queue_limit))
                self.update({TIME_TAG: "%s:00:00" % queue_limit})

    def change_walltime(self, walltime):
        seconds = time_to_second(walltime)
        if MANAGER == "PBS":
            queue = self["-q"]
            if queue in ["condo", "hotel"]:
                if seconds > self.time_limit(queue):
                    self.change_queue("home")
        elif seconds > self.time_limit("comet"):
            raise ValueError("Maximum Walltime for comet is 48 hrs.\n")
        self.update({TIME_TAG: second_to_time(seconds)})

    def change_processors(self, ppn):
        if MANAGER == "SLURM":
            self.update({"--ntasks-per-node": ppn})
        else:
            if ppn == 16:
                ppn = "{}:sandy".format(ppn)
            elif ppn == 24:
                ppn = "{}:haswell".format(ppn)
            elif ppn == 28:
                ppn = "{}:broadwell".format(ppn)
            ppn = re.sub("ppn=.*", "ppn={}".format(ppn),
                         self.get("nodes", "1:ppn=16"))
            self.update({"nodes": ppn})

    def change_jobname(self, jobname):
        if MANAGER == "SLURM":
            self.update({"-J": jobname})
        else:
            self.update({"-N": jobname})

    def change_mail_type(self, name):
        if MANAGER == "SLURM":
            m_type = ['BEGIN', 'END', 'FAIL', 'REQUEUE', 'STAGE_OUT', 'NONE',
                      'ALL', 'STAGE_OUT', 'TIME_LIMIT', 'TIME_LIMIT_90',
                      'TIME_LIMIT_80', 'TIME_LIMIT_50', 'ARRAY_TASKS']
            n_list = name.split(",")
            if all([i in m_type for i in n_list]):
                self.update({"--mail-type": name})
            else:
                raise IOError("%s is not supported in SLURM" % name)
        else:
            if all([i in ['a', 'b', 'e'] for i in name]):
                self.update({"-m": name})
            else:
                raise IOError("%s is not supported in PBS" % name)

    def change_exe(self, exe):
        self["others"] = re.sub("\{*\w+\}*(?=\s*>\s*vasp.out)",
                                exe, self["others"])


class Incar(dict):
    """
    Incar object for reading and writing a INCAR file. Essentially is a
    dictionary with VASP tags
    """
    def __init__(self, params=None):
        """
        Create an Incar object
        :param params: (dict) A set of VASP tags
        """
        super(Incar, self).__init__()
        if params:
            self.update(params)

    def __str__(self):
        return self.get_string(sort_keys=True, pretty=False)

    @staticmethod
    def from_file(filename):
        return Incar.from_string(file_to_str(filename))

    @staticmethod
    def from_string(string):
        lines = list(clean_lines(string.splitlines(), remove_comment=False))
        params = {}
        for line in lines:
            m = re.match("(\w+)\s*=\s*(.*)", line)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                val = Incar.proc_val(key, val)
                params[key] = val
        return Incar(params)

    @staticmethod
    def proc_val(key, val):
        list_keys = ("LDAUU", "LDAUL", "LDAUJ", "MAGMOM", "DIPOL")
        bool_keys = ("LDAU", "LWAVE", "LSCALU", "LCHARG", "LPLANE",
                     "LHFCALC", "ADDGRID", "LSORBIT", "LNONCOLLINEAR")
        float_keys = ("EDIFF", "SIGMA", "TIME", "ENCUTFOCK", "HFSCREEN",
                      "POTIM", "EDIFFG")
        int_keys = ("NSW", "NBANDS", "NELMIN", "ISIF", "IBRION", "ISPIN",
                    "ICHARG", "NELM", "ISMEAR", "NPAR", "LDAUPRINT", "LMAXMIX",
                    "ENCUT", "NSIM", "NKRED", "NUPDOWN", "ISPIND", "LDAUTYPE")

        def smart_int_or_float(numstr):
            if numstr.find(".") != -1 or numstr.lower().find("e") != -1:
                return float(numstr)
            else:
                return int(numstr)

        try:
            if key in list_keys:
                output = []
                toks = re.findall(r"(-?\d+\.?\d*)\*?(-?\d+\.?\d*)?\*?(-?\d+\.?\d*)?", val)
                for tok in toks:
                    if tok[2] and "3" in tok[0]:
                        output.extend(
                            [smart_int_or_float(tok[2])] * int(tok[0]) * int(tok[1]))
                    elif tok[1]:
                        output.extend([smart_int_or_float(tok[1])] * int(tok[0]))
                    else:
                        output.append(smart_int_or_float(tok[0]))
                return output
            if key in bool_keys:
                m = re.match(r"^\.?([T|F|t|f])[A-Za-z]*\.?", val)
                if m:
                    if m.group(1) == "T" or m.group(1) == "t":
                        return True
                    else:
                        return False
                raise ValueError(key + " should be a boolean type!")
            if key in float_keys:
                return float(re.search(r"^-?\d*\.?\d*[e|E]?-?\d*", val).group(0))
            if key in int_keys:
                return int(re.match(r"^-?[0-9]+", val).group(0))
        except ValueError:
            pass

        try:
            val = int(val)
            return val
        except ValueError:
            pass

        try:
            val = float(val)
            return val
        except ValueError:
            pass

        if "true" in val.lower():
            return True

        if "false" in val.lower():
            return False

        try:
            if key not in ("TITEL", "SYSTEM"):
                return re.search(r"^-?[0-9]+", val.capitalize()).group(0)
            else:
                return val.capitalize()
        except:
            return val.capitalize()

    def get_string(self, sort_keys=False, pretty=False):
        keys = self.keys()
        if sort_keys:
            keys = sorted(keys)
        lines = []
        for k in keys:
            if k == "MAGMOM" and isinstance(self[k], list):
                value = []
                if isinstance(self[k][0], list) and (self.get("LSORBIT") or
                                                     self.get("LNONCOLLINEAR")):
                    self[k] = [format_float(i, no_one=False)
                               for i in np.matrix(self[k]).A1]
                for m, g in itertools.groupby(self[k]):
                    value.append("{}*{}".format(len(tuple(g)), m))
                lines.append([k, " ".join(value)])
            elif isinstance(self[k], list):
                lines.append([k, " ".join([str(i) for i in self[k]])])
            else:
                lines.append([k, self[k]])

        if pretty:
            return str(tabulate([[l[0], "=", l[1]] for l in lines],
                                tablefmt="plain"))
        else:
            return str_delimited(lines, None, " = ") + "\n"

    def write_file(self, filename):
        str_to_file(self.__str__(), filename)


class PotcarSingle(object):
    functional_tags = {"PE": "PBE", "91": "PW91", "CA": "LDA"}
    parse_tags = {"LULTRA": parse_bool,
                  "LCOR": parse_bool,
                  "LPAW": parse_bool,
                  "EATOM": parse_float,
                  "RPACOR": parse_float,
                  "POMASS": parse_float,
                  "ZVAL": parse_float,
                  "RCORE": parse_float,
                  "RWIGS": parse_float,
                  "ENMAX": parse_float,
                  "ENMIN": parse_float,
                  "EAUG": parse_float,
                  "DEXC": parse_float,
                  "RMAX": parse_float,
                  "RAUG": parse_float,
                  "RDEP": parse_float,
                  "RDEPT": parse_float,
                  "QCUT": parse_float,
                  "QGAM": parse_float,
                  "RCLOC": parse_float,
                  "IUNSCR": parse_int,
                  "ICORE": parse_int,
                  "NDATA": parse_int,
                  "VRHFIN": parse_string,
                  "LEXCH": parse_string,
                  "TITEL": parse_string,
                  "STEP": parse_list,
                  "RRKJ": parse_list,
                  "GGA": parse_list}

    def __init__(self, data):
        self.data = data
        self.header = data.split("\n")[0].strip()
        search_lines = re.search("(parameters from.*?PSCTR-controll parameters)",
                                 data, re.S).group(1)
        self.keywords = {}
        for key, val in re.findall("(\S+)\s*=\s*(.*?)(?=;|$)",
                                   search_lines, re.M):
            self.keywords[key] = self.parse_tags[key](val)
        self.functional = self.functional_tags[self.keywords["LEXCH"]]
        if self.functional == "PBE":
            if "mkinetic" in data:
                self.functional = "PBE_52"
        elif self.functional == "LDA":
            if "US" in self.keywords["TITEL"]:
                self.functional = "LDA_US"

    def __str__(self):
        return self.data + "\n"

    @staticmethod
    def from_file(filename):
        return PotcarSingle(file_to_str(filename).decode("utf-8"))


class Potcar(list):
    def __init__(self, psps, functional="PBE"):
        super(Potcar, self).__init__()
        self.functional = functional
        if isinstance(psps, list):
            self.extend(psps)

    def __str__(self):
        return "\n".join([str(potcar).strip("\n") for potcar in self]) + "\n"

    def write_file(self, filename="POTCAR"):
        str_to_file(self.__str__(), filename)

    @staticmethod
    def from_file(filename):
        return Potcar.from_string(file_to_str(filename))

    @staticmethod
    def from_string(string):
        chuncks = re.findall("\n?(\s*.*?End of Dataset)", string, re.S)
        psps = [PotcarSingle(i) for i in chuncks]
        functional = psps[0].functional
        return Potcar(psps, functional)

    @staticmethod
    def from_elements(elements, functional="PBE"):
        try:
            d = PSP_DIR
        except KeyError:
            raise KeyError("Please set the AF_VASP_PSP_DIR environment in "
                           ".afrc.yaml. E.g. aimsflow config -a "
                           "AF_VASP_PSP_DIR ~/psps")
        psps = []
        potcar_setting = VASP_CONFIG["POTCAR"]
        fundir = VASP_CONFIG["FUNDIR"][functional]
        for el in elements:
            psp = []
            symbol = potcar_setting.get(el, el)
            paths_to_try = [os.path.join(PSP_DIR, fundir, "POTCAR.%s" % symbol),
                            os.path.join(PSP_DIR, fundir, symbol, "POTCAR")]
            for p in paths_to_try:
                p = os.path.expanduser(p)
                p = zpath(p)
                if os.path.exists(p):
                    psp = PotcarSingle.from_file(p)
                    break
            if psp:
                psps.append(psp)
            else:
                raise IOError("Cannot find the POTCAR with functional %s and "
                              "label %s" % (functional, symbol))
        return Potcar(psps, functional)


class Kpoints_supported_modes(Enum):
    Automatic = 0
    Gamma = 1
    MonKhorst = 2
    Line_mode = 3
    Cartesian = 4
    Reciprocal = 5

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        c = s.lower()[0]
        for m in Kpoints_supported_modes:
            if m.name.lower()[0] == c:
                return m
        raise ValueError("Can't interprete Kpoint mode %s" % s)


class Kpoints(object):
    supported_modes = Kpoints_supported_modes

    def __init__(self, comment="Default gamma", num_kpts=0,
                 style=supported_modes.Gamma,
                 kpts=((1, 1, 1),), kpts_shift=(0, 0, 0),
                 kpts_weights=None, coord_type=None, labels=None,
                 tet_number=0, tet_weight=0, tet_connections=None):
        if num_kpts > 0 and (not labels) and (not kpts_weights):
            raise ValueError("For explicit or line-mode kpoints, either the "
                             "labels or kpts_weights must be specified.")
        self.comment = comment
        self.num_kpts = num_kpts
        self.style = style
        self.kpts = kpts
        self.kpts_shift = kpts_shift
        self.kpts_weights = kpts_weights
        self.coord_type = coord_type
        self.labels = labels
        self.tet_number = tet_number
        self.tet_weight = tet_weight
        self.tet_connections = tet_connections

    def __str__(self):
        lines = [self.comment, str(self.num_kpts), str(self.style)]
        style = self.style.name.lower()[0]
        if style == "l":
            lines.append(self.coord_type)
        for i in range(len(self.kpts)):
            lines.append(" ".join([str(j) for j in self.kpts[i]]))
            if style == "l":
                lines[-1] += " ! " + self.labels[i]
                if i % 2 == 1:
                    lines[-1] += "\n"
            elif self.num_kpts > 0:
                if self.labels is not None:
                    lines[-1] += " %i %s" % (self.kpts_weights[i],
                                             self.labels[i])
                else:
                    lines[-1] += " %i" % self.kpts_weights[i]
        # Print tetrahedron parameters if the number of tetrahedrons > 0
        if style not in "lagm" and self.tet_number > 0:
            lines.append("Tetrahedron")
            lines.append("%d %f" % (self.tet_number, self.tet_weight))
            for sym_weight, vertices in self.tet_connections:
                lines.append("%d %s" % (sym_weight, " ".join(vertices)))
        # Print shifts for automatic kpoints types if not zero
        if self.num_kpts <= 0 and tuple(self.kpts_shift) != (0, 0, 0):
            lines.append(" ".join("%.3f" % f for f in self.kpts_shift))
        return "\n".join(lines) + "\n"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def automatic(subdivisions):
        return Kpoints("Fully automatic kpoint scheme", 0,
                       style=Kpoints_supported_modes.Automatic,
                       kpts=[[subdivisions]])

    @staticmethod
    def gamma_automatic(kpts=(1, 1, 1), shift=(0, 0, 0)):
        return Kpoints("Automatic kpoint scheme", 0,
                       Kpoints.supported_modes.Gamma, kpts=[kpts],
                       kpts_shift=shift)

    @staticmethod
    def monkhorst_automatic(kpts=(2, 2, 2), shift=(0, 0, 0)):
        return Kpoints("Automatic kpoint scheme", 0,
                       Kpoints.supported_modes.MonKhorst, kpts=[kpts],
                       kpts_shift=shift)

    @staticmethod
    def automatic_density(structure, kpts, force_gamma=False):
        """
        Returns an automatic Kpoint object based on a structure and a kpoint
        density. Uses Gamma centered meshes for hexagonal cells and
        Monkhorst-Pack grids otherwise.

        Algorithm:
            Uses a simple approach scaling the number of divisions along each
            reciprocal lattice vector proportional to its length.
        :param structure:
        :param kpts:
        :param force_gamma:
        :return:
        """
        comment = "aimsflow generated KPOINTS with grid density " \
                  "= %.0f / atom" % kpts
        latt = structure.lattice
        lengths = latt.abc
        ngrid = kpts / structure.num_sites
        mult = (ngrid * lengths[0] * lengths[1] * lengths[2]) ** (1 / 3)
        num_div = [int(math.floor(max(mult / l, 1))) for l in lengths]
        is_hax = latt.is_hex()
        has_odd = any([i % 2 == 1 for i in num_div])

        if has_odd or is_hax or force_gamma:
            style = Kpoints.supported_modes.Gamma
        else:
            style = Kpoints.supported_modes.MonKhorst

        return Kpoints(comment, 0, style, [num_div], [0, 0, 0])

    @staticmethod
    def automatic_density_by_vol(structure, density, force_gamma=False):
        """
        Returns an automatic Kpoint object based on a structure and a kpoint
        density per inverse Angstrom of reciprocal cell.

        :param structure: Structure object
        :param density: Grid density per Angstrom^(-3) of reciprocal cell
        :param force_gamma: Force a gamma centered mesh
        :return: Kpoints object
        """
        vol = structure.lattice.reciprocal_lattice.volume
        kpts = density * vol * structure.num_sites
        return Kpoints.automatic_density(structure, kpts,
                                         force_gamma=force_gamma)

    @staticmethod
    def from_file(filename):
        return Kpoints.from_string(file_to_str(filename))

    @staticmethod
    def from_string(string):
        lines = [line.strip() for line in string.splitlines()]
        comment = lines[0]
        num_kpts = int(lines[1].split()[0])
        style = lines[2].lower()[0]
        # Fully automatic KPOINTS
        if style == "a":
            return Kpoints.automatic(int(lines[3]))
        # Automatic gamma and Monk KPOINTS, with optional shift
        if style in ["g", "m"]:
            kpts = [int(i) for i in lines[3].split()]
            kpts_shift = (0, 0, 0)
            if len(lines) > 4:
                try:
                    kpts_shift = [int(i) for i in lines[4].split()]
                except ValueError:
                    pass
            return Kpoints.gamma_automatic(kpts, kpts_shift) if style == "g" \
                else Kpoints.monkhorst_automatic(kpts, kpts_shift)
        # Automatic kpoints with basis
        if num_kpts <= 0:
            style = Kpoints.supported_modes.Cartesian if style in "ck" \
                else Kpoints.supported_modes.Reciprocal
            kpts = [[float(j) for j in lines[i].split()] for i in range(3, 6)]
            kpts_shift = [float(i) for i in lines[6].split()]
            return Kpoints(comment, num_kpts, style, kpts, kpts_shift)
        # Line-mode KPOINTS, usually used with band structure
        if style == "l":
            coord_type = "Catesian" if lines[3].lower()[0] in "ck" \
                else "Reciprocal"
            style = Kpoints_supported_modes.Line_mode
            patt = re.findall("([\d\.*\-]+)\s+([\d\.*\-]+)\s+([\d\.*\-]+)\s+"
                              "!\s*(.*)", "\n".join(lines[4:]))
            patt = np.array(patt)
            kpts = [[float(j) for j in i] for i in patt[:, :3]]
            labels = [i.strip() for i in patt[:, 3]]
            return Kpoints(comment, num_kpts, style, kpts,
                           coord_type=coord_type, labels=labels)
        # Assume explicit KPOINTS if all else fails
        style = Kpoints.supported_modes.Cartesian if style in "ck" \
            else Kpoints.supported_modes.Reciprocal
        kpts = []
        kpts_weights = []
        labels = []
        tet_number = 0
        tet_weight = 0
        tet_connections = None

        for i in range(3, 3 + num_kpts):
            toks = lines[i].split()
            kpts.append([float(j) for j in toks[:3]])
            kpts_weights.append(float(toks[3]))
            if len(toks) > 4:
                labels.append(toks[4])
            else:
                labels.append(None)
        try:
            # Deal with tetrahedron method
            lines = lines[3 + num_kpts:]
            if lines[0].strip().lower()[0] == "t":
                toks = lines[1].split()
                tet_number = int(toks[0])
                tet_weight = float(toks[1])
                tet_connections = []
                for i in range(2, 2 + tet_number):
                    toks = lines[i].split()
                    tet_connections.append((int(toks[0]), [int(j) for j in toks[1:]]))
        except IndexError:
            pass
        return Kpoints(comment, num_kpts, style, kpts,
                       kpts_weights=kpts_weights, labels=labels,
                       tet_number=tet_number, tet_weight=tet_weight,
                       tet_connections=tet_connections)

    def write_file(self, filename):
        str_to_file(self.__str__(), filename)
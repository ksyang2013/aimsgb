import re
import copy
from getpass import getuser
from fnmatch import fnmatch
from math import sin, cos, radians
from collections import OrderedDict

from aimsflow import Structure
from aimsflow.symmetry.bandstructure import HighSymmKpath
from aimsflow.util import file_to_str, str_to_file, immed_file_paths, second_to_time,\
    ordered_loadfn, literal_dumpfn, make_path, format_float, time_to_second, \
    immed_subdir, delete_dict_keys
from aimsflow.vasp_io import Incar, BatchFile, Poscar, Potcar, Kpoints, TIME_TAG,\
    MANAGER, VASP_CONFIG

DIRNAME = {"r": "relax", "e": "born", "s": "static", "b": "band", "p": "pchg"}
CMD = {"r": "",
       "e": "cp ../static/POSCAR .",
       "s": "cp ../relax/CONTCAR POSCAR",
       "b": "cp ../relax/CONTCAR POSCAR\ncp ../static/CHG* .",
       "p": "cp ../static/WAVECAR .\ncp ../relax/CONTCAR POSCAR"
       }


class VaspYaml(OrderedDict):
    def __init__(self, params=None):
        super(VaspYaml, self).__init__()
        if params:
            self.update(params)

    @staticmethod
    def from_file(filename):
        return VaspYaml(ordered_loadfn(filename))

    @staticmethod
    def generate(work_dir, job_type, add_u=False, add_mag=False,
                 con_job=False, eint=-1.0, hse=False, soc=False):
        poscars = []
        job_name = []
        vasp_input = OrderedDict()
        for f in immed_file_paths(work_dir):
            f_name = f.split("/")[-1]
            if fnmatch(f_name, "*POSCAR*") or fnmatch(f_name, "*CONTCAR*"):
                poscars.append(Structure.from_file(f))
                if con_job:
                    # job_name.append(work_dir.split("/")[-2])
                    if job_type == "s":
                        job_name.append("static")
                    elif job_type == "b":
                        job_name.append("band")
                    elif job_type == "p":
                        job_name.append("pchg")
                    else:
                        job_name.append("born")
                    break
                else:
                    try:
                        name = re.search("^[P|C].*?[_|-](.*)", f_name).group(1)
                    except AttributeError:
                        if f_name in ["POSCAR", "POSCAR.vasp", "CONTCAR",
                                      "CONTCAR.vasp"]:
                            if job_type == "r":
                                name = "relax"
                            elif job_type == "s":
                                name = "static"
                            else:
                                name = "run_%s" % f_name
                        else:
                            name = re.sub("POSCAR|CONTCAR", "", f_name)
                    job_name.append(name)

        if not poscars:
            raise IOError("No POSCAR or CONTCAR found in %s" % work_dir)

        batch_run = BatchFile.from_string(VASP_CONFIG[MANAGER])
        if con_job:
            old_batch = BatchFile.from_file("%s/runscript.sh" % work_dir)
            batch_run[TIME_TAG] = old_batch[TIME_TAG]
            if MANAGER == "PBS":
                batch_run.update({"nodes": old_batch["nodes"],
                                  "-q": old_batch["-q"]})
        if job_type == "p":
            batch_run[TIME_TAG] = second_to_time(600)
        vasp_input["RUN_SCRIPT"] = batch_run.__str__()
        batch_check = copy.deepcopy(batch_run)
        command = [i for i in batch_run["command"].split("\n")
                   if i.startswith("cd")]
        command.append("aimsflow vasp -cj {job_type}\n")
        err = batch_check["-e"]
        out = batch_check["-o"]
        batch_check.update({"command": "\n".join(command),
                            "others": "", TIME_TAG: second_to_time(120),
                            "-e": err.replace("err.", "err.check."),
                            "-o": out.replace("out.", "out.check.")})
        vasp_input["CHECK_SCRIPT"] = batch_check.__str__()

        if con_job:
            incar = Incar.from_file("%s/INCAR" % work_dir)
            if incar.get("HFSCREEN"):
                hse = True
        elif hse:
            incar = VASP_CONFIG["INCAR_HSE"]
        else:
            incar = VASP_CONFIG["INCAR"]

        if not hse:
            vasp_input["ADD_U"] = add_u
        vasp_input["ADD_MAG"] = add_mag
        vasp_input["SOC"] = soc

        for each in job_type:
            job = {"EXE": VASP_CONFIG["EXE"]}
            incar_tmp = copy.deepcopy(incar)
            if each != "r":
                delete_dict_keys(["ISIF", "EDIFFG"], incar_tmp)
            if each == "s":
                incar_tmp.update({"NSW": 0, "EMIN": -25, "EMAX": 25, "NELM": 200,
                                  "NEDOS": 5000, "IBRION": -1, "LCHARG": True})
                if hse:
                    incar_tmp.update({"ISMEAR": 0, "ISYM": 3, "NELMIN": 5})
                else:
                    incar_tmp.update(
                        {"LWAVE": False, "LORBIT": 11, "ISMEAR": -5,
                         "EDIFF": 1e-6, "ALGO": "Normal", "ICHARG": 2})
                if soc is not False:
                    a = sin(radians(soc))
                    c = cos(radians(soc))
                    saxis = " ".join([format_float(i, no_one=False)
                                      for i in [a, 0, c]])
                    incar_tmp.update({"LSORBIT": True, "LORBMOM": True,
                                      "LNONCOLLINEAR": True, "ISPIN": 1,
                                      "SAXIS": saxis})
            elif each == "b":
                incar_tmp.update({"NSW": 0, "ISMEAR": 0, "EMIN": -25, "EMAX": 25,
                                  "NEDOS": 5000, "NELM": 200, "LCHARG": False})
                if hse:
                    incar_tmp.update({"ISYM": 3, "NELMIN": 5})
                else:
                    incar_tmp.update(
                        {"IBRION": -1, "LWAVE": False, "LORBIT": 11,
                         "ICHARG": 11, "ALGO": "Normal", "EDIFF": 1e-6})
            elif each == "p":
                incar_tmp.update({"NBMOD": -3, "EINT": eint, "LPARD": True,
                                  "LWAVE": False, "LCHARG": False})
            elif each == "e":
                incar_tmp["LEPSILON"] = True
                delete_dict_keys(["NPAR"], incar_tmp)

            job["INCAR"] = incar_tmp
            if each in ["p", "e"]:
                job["KPT"] = file_to_str("%s/KPOINTS" % work_dir)
            else:
                job["KPT"] = VASP_CONFIG["KPT"]
            vasp_input[DIRNAME[each].upper()] = job

        vasp_input["POSCAR"] = {}
        for i in range(len(poscars)):
            vasp_input["POSCAR"][job_name[i]] = str(Poscar(poscars[i]))
        return VaspYaml(vasp_input)

    def prepare_vasp(self, jt, work_dir, functional="PBE"):
        vasp_jobs = self.get("POSCAR")
        if vasp_jobs is None:
            raise IOError("No POSCARs in yaml file.")
        for name, poscar in vasp_jobs.items():
            jp = "%s/%s" % (work_dir, name)
            if len(jt) > 1:
                make_path(jp)
            for i, t in enumerate(jt):
                tmp_jp = jp
                if len(jt) > 1:
                    tmp_jp = "%s/%s" % (tmp_jp, DIRNAME[t])
                try:
                    next_jt = jt[i + 1]
                except IndexError:
                    next_jt = ""
                self.create_vasp_files(tmp_jp, t, next_jt, poscar,
                                       name, functional)

    def create_vasp_files(self, jp, jt, next_jt, poscar, name,
                          functional="PBE"):
        jt_up = DIRNAME[jt].upper()
        print("aimsflow is working on %s" % jp)
        make_path(jp)
        poscar = Poscar.from_string(poscar)
        poscar.write_file("%s/POSCAR" % jp)
        elements = poscar.site_symbols
        pot = Potcar.from_elements(elements, functional)
        pot.write_file("%s/POTCAR" % jp)
        try:
            incar = Incar(self["INCAR_%s" % jt_up])
            # if any([i > 20 for i in poscar.structure.lattice.abc]):
            #     incar["AMIN"] = 0.01
            if self.get("ADD_MAG"):
                mag = []
                magmom = VASP_CONFIG["MAGMOM"]
                for site in poscar.structure:
                    if str(site.specie) in magmom:
                        tmp = magmom.get(str(site.specie))
                        if incar.get("LSORBIT"):
                            theta = self["SOC"]
                            a = sin(radians(theta)) * tmp
                            c = cos(radians(theta)) * tmp
                            mag.append([a, 0, c])
                        else:
                            mag.append(magmom.get(str(site.specie)))
                    else:
                        if incar.get("LSORBIT"):
                            mag.append([0, 0, 0])
                        else:
                            mag.append(0)
                incar["MAGMOM"] = mag
            if self.get("ADD_U"):
                incar["LDAU"] = True
                for k in ("LDAUU", "LDAUJ", "LDAUL"):
                    incar[k] = [VASP_CONFIG[k].get(sym, 0)
                                for sym in poscar.site_symbols]
                comp = poscar.structure.composition
                if any([el.Z > 56 for el in comp]):
                    incar["LMAXMIX"] = 6
                elif any([el.Z > 20 for el in comp]):
                    incar["LMAXMIX"] = 4
            incar.write_file("%s/INCAR" % jp)
            try:
                if jt == "b":
                    k = HighSymmKpath(poscar.structure)
                    frac, labels = k.get_kpoints()
                    k_path = " ".join(["-".join(i) for i in k.kpath["path"]])
                    comment = "aimsflow generated KPOINTS for %s: %s" % \
                              (k.name, k_path.replace("\\Gamma", "G"))
                    kpt = Kpoints(comment=comment, kpts=frac, labels=labels,
                                  style=Kpoints.supported_modes.Line_mode,
                                  num_kpts=self["KPT_%s" % jt_up],
                                  coord_type="Reciprocal")
                else:
                    kpt = Kpoints.automatic_density_by_vol(
                        poscar.structure, self["KPT_%s" % jt_up])
                kpt.write_file("%s/KPOINTS" % jp)
            except TypeError:
                str_to_file(self["KPT_%s" % jt_up], "%s/KPOINTS" % jp)
        except KeyError:
            raise KeyError("Available job types in yaml file are incompatible "
                           "with requested job types.")

        user_name = getuser()
        job_name = "%s_%s" % (jt, name)
        if jt == "s" and jp.split("/")[-1] != "static":
            command = ""
        elif jt == "b" and incar.get("HFSCREEN"):
            # HSE band structures must be self-consistent in VASP.
            command = CMD["b"].split("\n")[0]
        else:
            command = CMD[jt]
        exe = self["EXE_%s" % jt_up]
        runscript = self["RUN_SCRIPT"].format(
            job_name=job_name, command=command, user_name=user_name, exe=exe)
        checkscript = self["CHECK_SCRIPT"].format(
            user_name=user_name, job_type=jt + next_jt,
            job_name="check_" + name)
        # reduce walltime by 50% if next_jt is "s"
        if jt == "s" and all([i in immed_subdir(jp.rstrip(DIRNAME[jt]))
                              for i in ["relax", "static"]]):
            new_script = BatchFile.from_string(runscript)
            relax_t = time_to_second(new_script[TIME_TAG])
            new_script[TIME_TAG] = second_to_time(relax_t * 0.5)
            runscript = new_script.__str__()

        str_to_file(runscript, "%s/runscript.sh" % jp)
        str_to_file(checkscript, "%s/checkscript.sh" % jp)

    def write_file(self, filename="job_flow.yaml"):
        literal_dumpfn(self, filename, default_flow_style=False)
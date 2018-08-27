import os
import re
import sys
import glob
import shutil
import subprocess
import numpy as np
from fnmatch import fnmatch
from collections import defaultdict

from aimsflow import Structure
from aimsflow.vasp_io import Incar, Outcar, Poscar, Potcar, BatchFile, \
    DIRNAME, TIME_TAG, MANAGER, Kpoints
from aimsflow.util import file_to_str, str_to_file, time_to_second, \
    second_to_time, time_lapse, immed_files

VASPFILES = ['INCAR', 'KPOINTS', 'POSCAR', 'POTCAR']


class VaspFlow(object):
    """
    Vasp flow object for submitting jobs, checking status, handling errors
    and restarting the job with modified parameters if necessary

    .. attribute:: folders
        A dict with job types pointing to a list of VASP directories
    """

    def __init__(self, directories):
        """
        Create a VaspFlow object.
        :param directories: (list) A list of working directories
        """
        folders = defaultdict(list)
        if not isinstance(directories, list):
            directories = [directories]
        for work_dir in directories:
            for root, dirs, files in os.walk(work_dir, topdown=False):
                if root != "./":
                    root = root.replace("./", "")
                if all(f in files for f in VASPFILES):
                    incar = Incar.from_file("%s/INCAR" % root)
                    if incar.get("NSW", 0) > 1:
                        folders["r"].append(root)
                    else:
                        if incar.get("ISMEAR", 1) != 0:
                            if incar.get("NBMOD"):
                                folders["p"].append(root)
                            elif incar.get("LEPSILON"):
                                folders["e"].append(root)
                            else:
                                folders["s"].append(root)
                        else:
                            kpt = Kpoints.from_file("%s/KPOINTS" % root)
                            if kpt.style.name == "Line_mode":
                                folders["b"].append(root)
                            else:
                                folders["s"].append(root)
        if folders == {}:
            raise IOError("aimsflow does not find any VASP folder. Make sure "
                          "you have all %s files" % " ".join(VASPFILES))
        else:
            self.folders = folders

    @property
    def job_status(self):
        """
        Get the job status
        :return: dict as {"converge", "un_converge"}
        - "converge": A dict with job type keys pointing to a list of converged
        VASP directories
        - "un_converge": A dict with job type keys pointing to a dict of
        un-converged VASP directory keys pointing to the un-converged reason
        """
        def hit_walltime(path, date):
            try:
                script = BatchFile.from_file("%s/runscript.sh" % path)
            except IOError:
                f = sorted(glob.glob("%s/*sh" % path))[-1]
                print("In %s aimsflow cannot find runscript.sh so will use "
                      "%s to get the calculation walltime." % (path, f))
                script = BatchFile.from_file(f)
            try:
                walltime = time_to_second(script[TIME_TAG])
            except KeyError:
                walltime = time_to_second(script["-t"])
            return time_lapse(date) > walltime

        js = defaultdict(dict)
        for k, v in self.folders.items():
            js["converge"][k] = []
            js["un_converge"][k] = {}
            for folder in v:
                vasp_out_file = sorted(glob.glob("%s/vasp.out*" % folder))
                outcar_file = "%s/OUTCAR" % folder
                if not vasp_out_file or not os.path.exists(outcar_file):
                    js["un_converge"][k][folder] = "Not started yet"
                    continue

                vasp_out = file_to_str(vasp_out_file[-1])
                if "POSCAR and POTCAR are incompatible" in vasp_out:
                    js["un_converge"][k][folder] = "POSCAR and POTCAR are " \
                                                   "incompatible"
                    continue
                m = re.search("vasp.out.(\d+)", vasp_out_file[-1])
                if m:
                    job_id = m.group(1)
                    try:
                        js_str = subprocess.check_output(
                            ["qstat", "-f", job_id], stderr=subprocess.STDOUT)
                        status = re.search("job_state = (\w)", js_str).group(1)
                    except (subprocess.CalledProcessError, OSError):
                        status = "C"
                else:
                    status = "C"
                if status == "R":
                    js["un_converge"][k][folder] = "Still running"
                    continue

                outcar = Outcar(outcar_file)
                e_change = outcar.e_change
                # Temporary solution for the following errors.
                if "inverse of rotation matrix was not found" in vasp_out:
                    message = "inverse of rotation matrix was not found " \
                              "(increase SYMPREC)"
                    js["un_converge"][k][folder] = message
                    continue
                elif "internal error in subroutine PRICEL" in vasp_out:
                    message = "internal error in subroutine PRICEL"
                    js["un_converge"][k][folder] = message
                    continue
                elif "Tetrahedron method fails for NKPT<4" in vasp_out:
                    message = "Tetrahedron method fails for NKPT<4"
                    js["un_converge"][k][folder] = message
                    continue

                incar = Incar.from_file("%s/INCAR" % folder)
                if k == "r":
                    ionic_step = re.findall("(\d+)\sF=", vasp_out)
                    try:
                        ionic_step = int(ionic_step[-1])
                    except IndexError:
                        if hit_walltime(folder, outcar.date):
                            js["un_converge"][k][folder] = "Hit walltime"
                        else:
                            js["un_converge"][k][folder] = "Stop running"
                        continue

                    nsw = incar.get("NSW", 0)
                    ediff = incar.get("EDIFF", 1E-4)
                    if all(abs(np.array(e_change)) < ediff) \
                            and "reached required accuracy" in vasp_out:
                        js["converge"][k].append(folder)
                    else:
                        if ionic_step == nsw:
                            message = "Hit NSW."
                        elif "reached required accuracy" in vasp_out:
                            message = "Fake convergence!"
                        elif "please rerun" in vasp_out:
                            message = "Copy CONTCAR to POSCAR and rerun."
                        elif "Routine ZPOTRF failed" in vasp_out:
                            message = "Routine ZPOTRF failed!"
                        elif "TOO FEW BANDS" in vasp_out:
                            message = "Too few bands!"
                        elif "FEXCP" in vasp_out:
                            message = "ERROR FEXCP: supplied " \
                                      "Exchange-correletion table is too small"
                        elif "FEXCF" in vasp_out:
                            message = "ERROR FEXCF: supplied " \
                                      "Exchange-correletion table is too small"
                        elif "ZBRENT: fatal" in vasp_out:
                            message = "ZBRENT error"
                        elif "Error EDDDAV" in vasp_out:
                            message = "Error EDDDAV: Call to ZHEGV failed"
                        elif "EDWAV: internal error" in vasp_out:
                            message = "EDWAV: internal error, the gradient " \
                                      "is not orthogonal"
                        elif "ERROR ROTDIA" in vasp_out:
                            message = "ERROR ROTDIA: Call to routine ZHEEV failed"
                        elif hit_walltime(folder, outcar.date):
                            message = "Hit walltime"
                        else:
                            message = "Stop running"
                        js["un_converge"][k][folder] = message
                elif k == "p":
                    if "partial charge" in vasp_out:
                        js["converge"][k].append(folder)
                    else:
                        js["un_converge"][k][folder] = "Calculation Error"
                else:
                    ediff = incar.get("EDIFF", 1E-4)
                    iteration = outcar.iteration
                    err_msg = outcar.err_msg
                    nelm = incar.get("NELM", 60)
                    if all(abs(np.array(e_change)) < ediff):
                        js["converge"][k].append(folder)
                    else:
                        if "decrease AMIN" in err_msg:
                            message = "decrease AMIN"
                        elif iteration == nelm:
                            message = "Hit NELM"
                        elif "Error EDDDAV" in err_msg:
                            message = "Error EDDDAV: Call to ZHEGV failed"
                        elif hit_walltime(folder, outcar.date):
                            message = "Hit walltime"
                        else:
                            message = "Stop running"
                        js["un_converge"][k][folder] = message
            if not js["converge"][k]:
                del js["converge"][k]
            if not js["un_converge"][k]:
                del js["un_converge"][k]
        return js

    def submit_job(self, jt):
        """
        Submit VASP jobs based on the job type
        :param jt: (str) job type
        """
        js = self.job_status
        if js["un_converge"] == {}:
            sys.stderr.write("All %s calculations are finished!\n" % DIRNAME[jt])
        else:
            try:
                ji_list = []
                for d in js["un_converge"][jt]:
                    message = js["un_converge"][jt][d]
                    if message == "Still running":
                        print("Calculation in '%s' is %s." % (d, message))
                    else:
                        ji_list.append(submit(d))

                if ji_list:
                    job_id_str = '\n'.join(ji_list)
                    str_to_file(job_id_str, "ID_list")
                else:
                    sys.stderr.write("No job is submitted.\n")
            except KeyError:
                sys.stderr.write("No un-converge %s is found.\n" % DIRNAME[jt])

    def clean_files(self):
        for k, v in self.folders.items():
            for path in v:
                for f in immed_files(path):
                    if f not in VASPFILES and not fnmatch(f, "*sh"):
                        os.remove("{}/{}".format(path, f))


def submit(work_dir):
    """
    Submit VASP job in a directory
    :param work_dir: (str) working directory
    :return: (str) Job ID for a submitted job
    """
    cur_dir = os.getcwd()
    os.chdir(work_dir)
    command = "qsub" if MANAGER == "PBS" else "sbatch"
    job_id_str = subprocess.check_output([command, "runscript.sh"])
    try:
        job_id = re.search('(\d+)', job_id_str).group(1)
        print("Successfully submit job in %s with ID: %s" % (work_dir, job_id))
    except IndexError:
        job_id = None
    os.chdir(cur_dir)
    return job_id


def continue_job(jt, folder, message="", max_run=5):
    if message == "Not started yet":
        submit(folder)
    elif message in ["Still running", "Stop running"]:
        print("Calculation in '%s' is %s." % (folder, message))
    else:
        modify_job(jt, folder, message=message, max_run=max_run)
        print(message)
        submit(folder)


def modify_job(jt, folder, message="", max_run=5):
    """
    Restart a VASP job with modified parameters according to the error message
    :param jt: (str) job type
    :param folder: (str) working directory
    :param message: (str) error message
    :param max_run: (int) maximum number of run times
    """
    def update_walltime(batch):
        old_time = time_to_second(batch[TIME_TAG])
        new_time = old_time * 1.5
        if MANAGER == "PBS":
            queue = batch["-q"]
            if queue in ["condo", "hotel"]:
                time_limit = batch.time_limit(queue)
                if new_time > time_limit:
                    sys.stderr.write("Cannot further increase walltime. "
                                     "Will use home queue.")
                    batch.change_queue("home")
        else:
            time_limit = batch.time_limit("comet")
            if new_time > time_limit:
                queue_limit = batch.queue_limit["comet"]
                sys.stderr.write("Cannot further increase walltime. "
                                 "Set walltime: %s hrs.\n" % queue_limit)
                new_time = time_limit
        batch.update({TIME_TAG: second_to_time(new_time)})
        return batch

    if message in ["Not started yet", "Still running"]:
        print(message)
        return
    run_times = len(glob.glob("%s/err.check*" % folder))
    if run_times >= max_run:
        raise RuntimeError("aimsflow has tried at least %s times rerun and "
                           "will stop trying in %s" % (max_run, folder))
    if message == "POSCAR and POTCAR are incompatible":
        old_pot = Potcar.from_file("%s/POTCAR" % folder)
        functional = old_pot.functional
        poscar = Poscar.from_file("%s/POSCAR" % folder)
        elements = poscar.site_symbols
        pot = Potcar.from_elements(elements, functional)
        pot.write_file("POTCAR")
        return

    incar = Incar.from_file("%s/INCAR" % folder)
    if jt == "r":
        if run_times > 1 or (run_times == 1 and incar.get("ALGO") == "Normal"):
            if "few bands" in message:
                outcar = file_to_str("%s/OUTCAR" % folder)
                nbands = int(re.findall("NBANDS=\s+(\d+)", outcar)[0])
                incar.update({"NBANDS": int(1.1 * nbands)})
            elif "ZPOTRF" in message:
                potim = incar.get("POTIM", 0.5) / 2.0
                incar.update({"POTIM": potim})
            elif "EDDDAV" in message:
                incar.update({"ALGO": "All"})
            elif any([i in message for i in ["EDWAV", "Tetrahedron"]]):
                incar.update({"ISMEAR": 0})
            elif any([i in message for i in ["FEXCP", "ROTDIA",
                                             "FEXCF", "ZBRENT"]]):
                incar.update({"IBRION": "1"})
            elif "SYMPREC" in message:
                incar.update({"SYMPREC": 1e-8})
            elif "PRICEL" in message:
                incar.update({"SYMPREC": 1e-8, "ISYM":0})
        else:
            incar.update({"ALGO": "Normal"})

        if any([i in message for i in ["Copy", "NSW", "Fake"]]):
            incar.update({"NELMIN": 5, "ALGO": "Normal"})
            if "POSCAR_orig" not in immed_files(folder):
                shutil.move("%s/POSCAR" % folder, "%s/POSCAR_orig" % folder)
            shutil.move("%s/CONTCAR" % folder, "%s/POSCAR" % folder)
            if run_times >= 3:
                incar.update({"IBRION": "1"})
        elif "walltime" in message:
            incar.update({"ALGO": "Normal"})
            script = BatchFile.from_file("%s/runscript.sh" % folder)
            script = update_walltime(script)
            script.write_file("%s/runscript.sh" % folder)

            if "POSCAR_orig" not in immed_files(folder):
                shutil.move("%s/POSCAR" % folder, "%s/POSCAR_orig" % folder)
            shutil.move("%s/CONTCAR" % folder, "%s/POSCAR" % folder)
    else:
        if "EDDDAV" in message:
            if incar.get("ALGO") == "Fast":
                incar.update({"ALGO": "Normal"})
            elif incar.get("ALGO") == "Normal":
                incar.update({"ALGO": "All"})
            else:
                raise RuntimeError("aimsflow failed to fix %s for job in %s"
                                   % (message, folder))
        else:
            if incar.get("HFSCREEN") is None:
                incar.update({"ALGO": "Normal"})
            if "NELM" in message:
                ediff = incar.get("EDIFF", 1e-4)
                if incar.get("NELMDL", -5) != -12:
                    incar.update({"NELMDL": -12})
                elif incar.get("AMIX") and incar.get("ALGO") != "All":
                    incar.update({"ALGO": "All", "ISMEAR": 0})
                elif not incar.get("AMIX"):
                    incar.update({"AMIX": 0.1})
                    incar.update({"BMIX": 0.0001})
                    incar.update({"AMIX_MAG": 0.4})
                    incar.update({"BMIX_MAG": 0.0001})
                elif ediff <= 1e-4:
                    incar.update({"EDIFF": ediff * 10})
                else:
                    raise RuntimeError("aimsflow failed to fix %s for job in "
                                       "%s" % (message, folder))
            elif any([i in message for i in ["ZPOTRF", "Tetrahedron"]]):
                incar.update({"ISYM": 0})
            elif "SYMPREC" in message:
                incar.update({"SYMPREC": 1e-8})
            elif "PRICEL" in message:
                incar.update({"SYMPREC": 1e-8, "ISYM": 0})
            elif "AMIN" in message:
                incar.update({"AMIN": 0.01})
            elif "walltime" in message:
                incar.update({"NELMDL": -12})
                script = BatchFile.from_file("%s/runscript.sh" % folder)
                script = update_walltime(script)
                script.write_file("%s/runscript.sh" % folder)
            else:
                raise RuntimeError("aimsflow failed to fix %s for job in %s"
                                   % (message, folder))

    incar.write_file("%s/INCAR" % folder)


def parse_outcar(folder, parse_mag=False):
    try:
        f = sorted(glob.glob("%s/OUTCAR*" % folder))[-1]
        return Outcar(f, parse_mag=parse_mag)
    except IndexError:
        sys.stderr.write("No OUTCAR in %s\n" % folder)


def get_toten(folder):
    return parse_outcar(folder).total_energy


def get_born(folder, axis="z", verbose=False):
    ind = ("x", "y", "z").index(axis)
    born = parse_outcar(folder).born_charge
    if born:
        s = Structure.from_file("%s/POSCAR" % folder)
        outs = []
        if verbose:
            for i, v in enumerate(s):
                outs.append("%s\t%s" % (v.species_string,
                                        "\t".join("%.2f" % j for j in born[i][axis])))
            return "\n" + "\n".join(outs)
        return "\t".join("%.2f" % j for j in [born[i][axis][ind]
                                              for i in range(s.num_sites)])
    else:
        sys.stderr.write("No born charge in OUTCAR in %s\n" % folder)


def get_mag(folder, ion_spec=None, axis="x", mag_tol=0.1, orb_mag_tol=0.001):
    outcar = parse_outcar(folder, parse_mag=True)
    mag = outcar.mag
    orb_mag = outcar.orb_mag
    if mag is None:
        sys.stderr.write("No magnetization in OUTCAR in %s\n" % folder)
    else:
        try:
            mag_tot = mag[axis][-1]["tot"]
            orb_mag_tot = orb_mag[axis][-1]["tot"] if orb_mag else None
            s = Structure.from_file("%s/POSCAR" % folder)
            mag_out = []
            orb_mag_out = []
            if ion_spec:
                sites = [s[i] for i in ion_spec]
                num_site = zip(ion_spec, sites)
            else:
                sites = s.sites
                num_site = zip(range(len(sites)), sites)
            for i, v in num_site:
                mag_tmp = mag[axis][i]["tot"]
                if abs(mag_tmp) > mag_tol:
                    name = v.species_string + str(i + 1)
                    mag_out.append((name, mag_tmp))
                if orb_mag:
                    mag_tmp = orb_mag[axis][i]["tot"]
                    name = v.species_string + str(i + 1)
                    if abs(mag_tmp) > orb_mag_tol:
                        orb_mag_out.append((name, mag_tmp))
            return {"mag": mag_out if mag_out else None, "mag_tot": mag_tot,
                    "orb_mag": orb_mag_out if orb_mag_out else None,
                    "orb_mag_tot": orb_mag_tot}
        except IndexError:
            sys.stderr.write("No magnetization (%s) for OUTCAR in %s\n"
                             % (axis, folder))

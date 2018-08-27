from __future__ import division

import re
import os
import sys
import shutil
import subprocess

from aimsflow import Structure
from aimsflow.elect_struct.core import Spin
from aimsflow.util import file_to_lines, str_to_file, immed_files, make_path,\
    immed_subdir, parse_number
from aimsflow.vasp_io import Eigenval, Procar, Doscar, BatchFile, Poscar, VaspYaml,\
    VaspFlow, continue_job, submit, get_toten, get_born, get_mag, DIRNAME, MANAGER


def vasp(args):
    directories = file_to_lines(args.directory_file) if args.directory_file \
        else args.directories
    if args.prepare:
        for d in directories:
            try:
                yaml = VaspYaml.from_file("%s/job_flow.yaml" % d)
            except IOError:
                raise IOError("No 'job_flow.yaml' file in '%s'" % d)
            yaml.prepare_vasp(args.prepare, d, args.functional)
    elif args.kill:
        for d in directories:
            try:
                job_id_list = file_to_lines("%s/ID_list" % d)
                for job_id in job_id_list:
                    subprocess.call(["qdel", job_id])
                    print("%s is killed" % job_id)
            except IOError:
                sys.stderr.write("No 'ID_list' file is found in %s" % d)
    else:
        flow = VaspFlow(directories)
        if args.submit:
            jt = args.submit[0]
            js = flow.job_status
            if js["un_converge"] == {}:
                sys.stderr.write("All %s calculations are finished!\n" % DIRNAME[jt])
            else:
                try:
                    ji_list = []
                    for d, message in js["un_converge"][jt].items():
                        if message == "Still running":
                            print("Calculation in '%s' is %s" % (d, message))
                        else:
                            ji_list.append(submit(d))

                    if ji_list:
                        job_id_str = '\n'.join(ji_list)
                        str_to_file(job_id_str, "ID_list")
                    else:
                        sys.stderr.write("No job is submitted\n")
                except KeyError:
                    sys.stderr.write("No un-converge %s is found\n" % DIRNAME[jt])
        elif args.clean:
            flow.clean_files()
        elif args.continue_job:
            js = flow.job_status
            jt = args.continue_job
            un_converge = js["un_converge"]
            # if first job type is not converged, then only submit them
            if un_converge.get(jt[0]):
                for d, message in un_converge[jt[0]].items():
                    continue_job(jt[0], d, message)
            elif len(jt) == 1:
                print("All calculations are finished!")
            else:
                # if first job type is converged, but second is not, then
                # only submit the second job type
                if un_converge.get(jt[1]):
                    for d, message in un_converge[jt[1]].items():
                        continue_job(jt[1], d, message)
                else:
                    d = os.getcwd().rsplit("/", 1)
                    # if the current directory contains "relax" or "static",
                    # aimsflow will go to the previous folder and submit the
                    # jobs with second job type
                    if any([i in d[1] for i in ["relax", "static"]]):
                        d = d[0]
                        js = VaspFlow([d]).job_status
                        un_converge = js["un_converge"]
                        try:
                            for d, message in un_converge[jt[1]].items():
                                continue_job(jt[1], d, message)
                        except KeyError:
                            print("All calculations are finished!")
                    else:
                        print("All calculations are finished!")
        elif args.add_job:
            cur_dir = os.getcwd()
            folders = flow.folders
            jt = args.add_job
            prev_jt = "r" if jt == "s" else "s"
            eint = args.eint if args.eint else -1.0
            for d in folders[prev_jt]:
                path = cur_dir if d == "." else os.path.join(cur_dir, d)
                if jt != "s":
                    path_p = path.rsplit("/", 1)[0]
                else:
                    path_p = os.path.join(path, "relax")
                    make_path(path_p)
                    for f in immed_files(path):
                        shutil.move(os.path.join(path, f), path_p)
                    tmp = path
                    path = path_p
                    path_p = tmp
                yaml = VaspYaml.generate_from_jobtype(path, jt, con_job=True, eint=eint)
                yaml.prepare_vasp(jt, path_p)


def analyze(args):
    def sorted_print(results, deli="\t"):
        sorted_keys = sorted(results.keys())
        for k in sorted_keys:
            print("%s:%s%s" % (k, deli, results[k]))

    directories = file_to_lines(args.directory_file) if args.directory_file \
        else args.directories
    flow = VaspFlow(directories)
    js = flow.job_status

    if args.converge:
        if js["un_converge"]:
            for k, v in js["un_converge"].items():
                if js["un_converge"][k]:
                    keys = sorted(js["un_converge"][k].keys())
                    print("\nFollowing %s calculations are not converged:\n"
                          "*****%s*****" % (DIRNAME[k], " ".join(keys)))
                    for key in keys:
                        print("%s: %s" % (key, js["un_converge"][k][key]))
                else:
                    print("All %s calculations are converged!" % DIRNAME[k])
        else:
            print("All calculations are finished!")

    if args.collect_contcar:
        def collect_structure(src, dst):
            for d in src:
                name = d.split("/")
                if len(name) > 1:
                    name = name[-2]
                else:
                    name = name[-1]
                if "CONTCAR" in immed_files(d):
                    shutil.copyfile("%s/CONTCAR" % d,
                                    "%s/POSCAR_%s" % (dst, name))
                else:
                    sys.stderr.write("No CONTCAR in %s\n" % d)

        dest = args.collect_contcar
        make_path(dest)
        try:
            collect_structure(js["converge"]["r"], dest)
        except KeyError:
            sys.stderr.write("No converged RELAX calculation. Will collect "
                             "CONTCARs from STATIC calculation.\n")
            try:
                collect_structure(js["converge"]["s"], dest)
            except KeyError:
                sys.stderr.write("No converged STATIC calculation. "
                                 "No CONTCAR is collected.\n")

    if args.collect_unconverge:
        dest = args.collect_unconverge
        make_path(dest)
        for k, v in js["un_converge"].items():
            for folder in v:
                shutil.move(folder, "{}/{}".format(dest, folder))
                name = folder.split("/")
                if len(name) > 1:
                    name = "/".join(name[:-1])
                else:
                    name = name[-1]
                if not immed_subdir(name):
                    os.rmdir(name)

    if args.get_toten:
        jt = args.get_toten
        toten = {}
        try:
            for folder in js["converge"][jt]:
                toten[folder] = get_toten(folder)
            for k, v in toten.items():
                if not v:
                    toten.pop(k, None)
            sorted_print(toten)
        except KeyError:
            sys.stderr.write("No converged %s calculation.\n" % DIRNAME[jt])

    if args.get_born_charge:
        born = {}
        try:
            for folder in js["converge"]["e"]:
                born[folder] = get_born(folder, axis=args.direction)
            sorted_print(born)
        except KeyError:
            sys.stderr.write("No converged BORN calculation.\n")

    if args.get_mag:
        mag = {}
        jt = args.get_mag
        sn = [i - 1 for i in parse_number(args.site_number)]\
            if args.site_number else None
        try:
            for folder in js["converge"][jt]:
                outs = get_mag(folder, ion_spec=sn, axis=args.direction)
                if outs is None or outs["mag"] is None:
                    mag[folder] = "No magnetic moment"
                else:
                    mag[folder] = "\t".join([": ".join([i[0], str(i[1])])
                                             for i in outs["mag"]])
                    mag[folder] += "\ttot: %s" % outs["mag_tot"]
            sorted_print(mag)
        except KeyError:
            sys.stderr.write("No converged %s calculation.\n" % DIRNAME[jt])

    if args.get_orb_mag:
        mag = {}
        jt = args.get_orb_mag
        sn = [i - 1 for i in parse_number(args.site_number)]\
            if args.site_number else None
        try:
            for folder in js["converge"][jt]:
                outs = get_mag(folder, ion_spec=sn, axis=args.direction)
                if outs is None or outs["orb_mag"] is None:
                    mag[folder] = "No orbital magnetic moment"
                else:
                    mag[folder] = "\t".join([": ".join([i[0], str(i[1])])
                                             for i in outs["orb_mag"]])
                    mag[folder] += "\ttot: %s" % outs["orb_mag_tot"]
            sorted_print(mag)
        except KeyError:
            sys.stderr.write("No converged %s calculation.\n" % DIRNAME[jt])

    if args.get_band_gap:
        def get_gap(vbm_p, energy, states, tol):
            if abs(states[vbm_p + 1]) > tol:
                return 0
            while abs(states[vbm_p - 1]) < tol:
                vbm_p -= 1
            for i in range(vbm_p, len(states)):
                if abs(states[i]) > tol:
                    cbm = energy[i] + abs(energy[vbm_p])
                    return cbm

        f = args.get_band_gap
        outs = {}
        for path in directories:
            f_path = "%s/%s" % (path, f)
            try:
                if f == "DOSCAR":
                    d = Doscar(f_path)
                    energy = d.tdos.energies - d.tdos.efermi
                    states = d.tdos.densities
                    for k, v in enumerate(energy):
                        if v > 0:
                            vbm_p = k
                            break
                    cbm = []
                    for spin in [Spin.up, Spin.dn]:
                        if spin in states:
                            cbm.append(get_gap(vbm_p, energy, states[spin], 1e-3))
                    outs[path] = "\t".join("%.3f" % i for i in cbm)
                else:
                    if f == "EIGENVAL":
                        v = Eigenval(f_path)
                    else:
                        v = Procar(f_path)
                    kpoint_file = "KPOINTS.bands.bz2" if "bz2" in f else "KPOINTS"
                    outcar_file = "OUTCAR.bands.bz2" if "bz2" in f else "OUTCAR"
                    band = v.get_band_structure("%s/%s" % (path, kpoint_file),
                                                "%s/%s" % (path, outcar_file))
                    b = band.get_band_gap()
                    outs[path] = "\t".join(
                        "%s\t%.3f\t%s" % (spin.name, b[spin]["energy"],
                                          b[spin]["label"]) for spin in b)
            except IOError:
                sys.stderr.write("No %s in %s\n" % (f, path))
        sorted_print(outs)


def jyaml(args):
    work_dir = args.directory
    filename = "%s/job_flow.yaml" % work_dir
    add_u = True if args.u_value else False
    add_mag = True if args.magmom else False
    hse = True if args.hse else False
    soc = args.soc if args.soc is not None else False

    if args.generate:
        vasp_input = VaspYaml.generate_from_jobtype(
            work_dir, args.generate, add_u, add_mag, hse=hse, soc=soc)
    else:
        vasp_input = VaspYaml.from_file(filename)
        if args.update_poscar:
            poscar_files = args.update_poscar
            vasp_input["POSCAR"] = {}
            for f in poscar_files:
                poscar = Structure.from_file(f)
                tmp = f.split("/")[-1]
                try:
                    name = re.search("^[P|C].*?_(.*)", tmp).group(1)
                except AttributeError:
                    name = "run_%s" % tmp
                vasp_input["POSCAR"][name] = str(Poscar(poscar))
        elif args.convert_batch:
            run_script = BatchFile.from_string(vasp_input["RUN_SCRIPT"])
            check_script = BatchFile.from_string(vasp_input["CHECK_SCRIPT"])
            if MANAGER == run_script.manager:
                sys.stderr.write("No need to convert job_flow.yaml\n")
                exit(0)
            run_script = BatchFile.convert_batch(run_script)
            check_script = BatchFile.convert_batch(check_script)
            vasp_input["RUN_SCRIPT"] = str(run_script)
            vasp_input["CHECK_SCRIPT"] = str(check_script)

        if add_u:
            vasp_input["ADD_U"] = True
        if add_mag:
            vasp_input["ADD_MAG"] = True

    script = BatchFile.from_string(vasp_input["RUN_SCRIPT"])
    if args.queue:
        script.change_queue(args.queue)
    if args.walltime:
        script.change_walltime(args.walltime)
    if args.processors:
        script.change_processors(args.processors)
    if args.mail:
        script.change_mail_type(args.mail)
    vasp_input["RUN_SCRIPT"] = str(script)
    if soc:
        vasp_input["SOC"] = soc

    vasp_input.write_file(filename)
    print("'job_flow.yaml' file is created in %s" % work_dir)
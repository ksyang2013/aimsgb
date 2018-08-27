import sys
import subprocess
import numpy as np
from collections import OrderedDict, defaultdict
from functools import reduce

from aimsflow.elect_struct.core import Spin, Color
from aimsflow.util.io_utils import str_to_file
from aimsflow.util.num_utils import column_stack


class DosPlotter(object):
    def __init__(self, zero_at_efermi=True):
        self.zero_at_efermi = zero_at_efermi
        self._doses = OrderedDict()

    def add_dos_dict(self, dos_dict):
        for label in dos_dict.keys():
            dos = dos_dict[label]
            energies = dos.energies - dos.efermi if self.zero_at_efermi \
                else dos.energies
            self._doses[label] = {"energies": energies, "densities": dos.densities,
                                  "efermi": dos.efermi}

    def get_energy_density(self):
        allenergies = []
        alldensities = []

        for key, dos in self._doses.items():
            energies = dos["energies"]
            densities = dos["densities"]
            newdens = OrderedDict()
            for spin in [Spin.up, Spin.dn]:
                if spin in densities:
                    newdens[spin] = densities[spin]
            allenergies.append(energies)
            alldensities.append(newdens)
        return allenergies, alldensities

    def get_plot(self, sys_name, xlim=None, ylim=None, is_separate=False):
        allenergies, alldensities = self.get_energy_density()
        outs = [list(allenergies[0])]
        is_spin = False
        if len(alldensities[0]) == 2:
            is_spin = True
            energies = list(allenergies[0])
            energies.reverse()
            outs[0].extend(energies)

        for i in alldensities:
            tmp = list(i[Spin.up])
            if is_spin:
                densities = list(-i[Spin.dn])
                densities.reverse()
                tmp.extend(densities)
            outs.append(tmp)

        legend = list(map(str, self._doses.keys()))
        legend.insert(0, "Energy")
        outs = reduce(lambda a, b: np.column_stack((a, b)), outs)
        if is_separate and is_spin:
            ndos = len(outs) // 2
            outs = {Spin.up: outs[:ndos], Spin.dn: outs[ndos:]}
            outs[Spin.dn][:, 1:] = abs(outs[Spin.dn][:, 1:])
            for spin in [Spin.up, Spin.dn]:
                files = file_name_list(sys_name + "_" + spin.name)
                np.savetxt(files["dos"], outs[spin], fmt="%10.5f",
                           header="\t\t".join(legend), comments="")
                str_to_file(self.generate_gp(files, xlim, ylim), files["gp"])
                execute_gnuplot(files)
        else:
            files = file_name_list(sys_name)
            np.savetxt(files["dos"], outs, fmt="%10.5f",
                       header="\t".join(legend), comments="")
            str_to_file(self.generate_gp(files, xlim, ylim), files["gp"])
            execute_gnuplot(files)

    def generate_gp(self, files, xlim=None, ylim=None):
        gp_script = """
set term postscript eps enhanced color font "Times-Roman, 80" size 18, 10.125
set output "{eps_file}"
set xlabel 'Energy (eV)'
set ylabel 'DOS (States/eV)'
set arrow from first 0, graph 0 to first 0, graph 1 nohead lt 0 lw 4
set label "E_F" at 0.02, graph 0.9
set key autotitle columnheader samplen 2
set style line 2  lc 'forest-green' lt 1 lw 5
set style line 3  lc 'web-blue' lt 1 lw 5
set style line 4  lc 'red' lt 1 lw 5
set style line 5  lc 'dark-violet' lt 1 lw 5
set style line 6  lc 'orange' lt 1 lw 5
set style line 7  lc 'greenyellow' lt 1 lw 5
set style line 8  lc 'dark-cyan' lt 1 lw 5
set style line 9  lc 'orchid' lt 1 lw 5
set style line 10  lc 'coral' lt 1 lw 5

plot [{xlim}][{ylim}] for [i=2:{col_num}] "{dos_file}" u 1:i w l ls i,\\
    """
        xlim = "" if xlim is None else ":".join("%.3f" % i for i in xlim)
        ylim = "" if ylim is None else ":".join("%.3f" % i for i in ylim)
        gp_script = gp_script.format(eps_file=files["eps"], xlim=xlim, ylim=ylim,
                                     col_num=len(self._doses.keys()) + 1,
                                     dos_file=files["dos"])
        return gp_script


class BSPlotter(object):
    def __init__(self, bs, zero_to_efermi=True, ):
        self.bs = bs
        self.num_bands = bs.num_bands

        self.zero_energy = OrderedDict()
        for spin in self.bs.bands:
            if not zero_to_efermi:
                self.zero_energy[spin] = 0.0
            elif self.bs.is_metal()[spin]:
                self.zero_energy[spin] = self.bs.efermi
            else:
                self.zero_energy[spin] = self.bs.get_vbm()[spin]["energy"]

    def get_plot_data(self, xlim=None, ylim=None, is_separate=False):
        if xlim:
            s_branch = self.bs.branches[xlim[0] - 1]
            e_branch = self.bs.branches[xlim[1] - 1]
            s_ind = s_branch["start_index"]
            e_ind = e_branch["end_index"]
        else:
            s_ind = self.bs.branches[0]["start_index"]
            e_ind = self.bs.branches[-1]["end_index"]

        bands = OrderedDict()
        num_bands = {}
        band_ind = {}
        for spin, values in self.bs.bands.items():
            if ylim:
                tmp = []
                ind = []
                for n, v in enumerate(values - self.zero_energy[spin]):
                    if any([ylim[0] < i < ylim[1] for i in v]):
                       tmp.append(v[s_ind:e_ind + 1])
                       ind.append(n)
                bands[spin] = tmp
                num_bands[spin] = len(tmp)
                band_ind[spin] = [ind[0], ind[-1]]
            else:
                bands[spin] = [v[s_ind:e_ind + 1]
                               for v in values - self.zero_energy[spin]]
                num_bands[spin] = self.num_bands
                band_ind[spin] = [0, self.num_bands]

        def get_edges(sp, band):
            outs = []
            for i, v in enumerate(band[sp]["kpoint_index"]):
                try:
                    x = eigen_data[sp][:, 0][v]
                except IndexError:
                    x = eigen_data[:, 0][v]
                y = band[sp]["energy"] - self.zero_energy[sp]
                outs.append("\t".join([str(j) for j in [x, y]]))
            return outs

        cbm = self.bs.get_cbm()
        # print(self.zero_energy)
        # exit(0)
        vbm = self.bs.get_vbm()
        if is_separate and self.bs.is_spin_polarized:
            eigen_data = OrderedDict()
            edges = defaultdict(list)
            for spin, values in bands.items():
                eigen_data[spin] = [self.bs.distance[s_ind:e_ind + 1]]
                for v in values:
                    eigen_data[spin].append(v)
                eigen_data[spin] = reduce(column_stack, eigen_data[spin])
                if self.bs.is_metal()[spin] is False:
                    edges[spin].extend(get_edges(spin, cbm))
                    edges[spin].extend(get_edges(spin, vbm))
        else:
            eigen_data = [self.bs.distance[s_ind:e_ind + 1]]
            for spin, values in bands.items():
                for v in values:
                    eigen_data.append(v)
            eigen_data = reduce(column_stack, eigen_data)
            edges = []
            for spin in self.zero_energy:
                if self.bs.is_metal()[spin] is False:
                    edges.extend(get_edges(spin, cbm))
                    edges.extend(get_edges(spin, vbm))
        return {"edges": edges, "eigen": eigen_data,
                "num": num_bands, "band_ind": band_ind}

    def get_plot(self, sys_name, xlim=None, ylim=None, is_separate=False):
        data = self.get_plot_data(xlim, ylim, is_separate)
        if isinstance(data["eigen"], OrderedDict):
            for spin, values in data["eigen"].items():
                files = file_name_list(sys_name + "_" + spin.name)
                np.savetxt(files["eig"], values, fmt="%10.3f")
                str_to_file(self.generate_gp(
                    files, data["edges"][spin], data["num"][spin],
                    xlim, ylim, is_separate), files["gp"])
                execute_gnuplot(files)
        else:
            files = file_name_list(sys_name)
            np.savetxt(files["eig"], data["eigen"], fmt="%10.3f")
            num_bands = data["num"][Spin.up] if len(data["num"]) == 1 \
                else data["num"]
            str_to_file(self.generate_gp(
                files, data["edges"], num_bands, xlim, ylim), files["gp"])
            execute_gnuplot(files)

    def get_ticks(self, xlim=None):
        tick_distance = [self.bs.distance[0]]
        tick_label = [self.bs.kpoints[0].label]
        for branch in self.bs.branches:
            s_ind = branch["start_index"]
            e_ind = branch["end_index"]
            s_label = self.bs.kpoints[s_ind].label
            if s_label != tick_label[-1]:
                tick_label[-1] += "|" + s_label
            tick_label.append(self.bs.kpoints[e_ind].label)
            tick_distance.append(self.bs.distance[e_ind])
        if xlim:
            tick_distance = tick_distance[xlim[0] - 1:xlim[1] + 1]
            tick_label = tick_label[xlim[0] - 1:xlim[1] + 1]
            if "|" in tick_label[-1]:
                tick_label[-1] = tick_label[-1].split("|")[0]
        for i, v in enumerate(tick_label):
            if v.startswith("\\") and len(v) > 3:
                tick_label[i] = "/Symbol G"
        return {"distance": tick_distance, "label": tick_label}

    def generate_gp(self, files, edges, num_bands, xlim=None, ylim=None,
                    is_separate=False):
        ticks = self.get_ticks(xlim)
        gp_script = """
set term postscript eps enhanced color font "Times-Roman, 80" size 18, 10.125
set output "{eps_file}"
set xtics ({x_labels})
set ytics scale 3
set ylabel 'Energy (eV)'
set arrow from graph 0, first 0 to graph 1, first 0 nohead dt 6 lw 1
{vertical_line}
unset key
plot [][{ylim}] for [i=2:{bands}] "{eig_file}" u 1:i w l lt 8 lw 2,\\\n"""
        if not is_separate and self.bs.is_spin_polarized:
            tot_bands = sum(num_bands.values())
            num_bands = num_bands[Spin.up]
            gp_script += 'for [i=%d+1:%d-1] "{eig_file}" u 1:i w l lt 7 ' \
                         'lw 2,\\\n' % (num_bands, tot_bands)
        vertical_line = ""
        x_labels = ""
        for i in range(len(ticks["label"])):
            vertical_line += "set arrow from first {0:.3f}, graph 0 to " \
                             "first {0:.3f}, graph 1 nohead " \
                             "lt 8 lw 1\n".format(ticks["distance"][i])
            x_labels += ''', "{{{0}}}" {1:.3f}'''.format(ticks["label"][i],
                                                         ticks["distance"][i])
        x_labels = x_labels[2:]
        ylim = "" if ylim is None else ":".join("%.3f" % i for i in ylim)
        gp_script += ", ".join([''''-' w p pt 7 ps 3 lc "black"'''] * len(edges))
        for i in edges:
            gp_script += "\n%s\ne" % i
        return gp_script.format(eps_file=files["eps"], x_labels=x_labels,
                                ylim=ylim, vertical_line=vertical_line,
                                bands=num_bands + 1, eig_file=files["eig"])


class BSProjPlotter(BSPlotter):
    def __init__(self, bs):
        super(BSProjPlotter, self).__init__(bs)

    def get_proj_data(self, band_ind, max_ps=5.0, xlim=None):
        if xlim:
            s_branch = self.bs.branches[xlim[0] - 1]
            e_branch = self.bs.branches[xlim[1] - 1]
            s_ind = s_branch["start_index"]
            e_ind = e_branch["end_index"]
        else:
            s_ind = self.bs.branches[0]["start_index"]
            e_ind = self.bs.branches[-1]["end_index"]

        tot_proj = OrderedDict()
        for spin, values in self.bs.tot_proj.items():
            s, e = band_ind[spin]
            tot_proj[spin] = values[s_ind:e_ind + 1, s:e + 1]

        proj_data = OrderedDict()
        for spin, each_spin in tot_proj.items():
            proj_data[spin] = []
            for each_kpt in each_spin:
                kpt_proj = []
                for each_proj in each_kpt:
                    tot = each_proj["tot"]
                    v = np.array(list(each_proj.values())[:-1])
                    v = v / tot * max_ps if tot else v
                    color_ind = [Color[orb.name].value
                                 for orb in each_proj.keys() if orb != "tot"]
                    v_orb = sorted(zip(v, color_ind),
                                   key=lambda x: x[0], reverse=True)
                    kpt_proj.append(np.reshape(np.array(v_orb), -1))
                proj_data[spin].extend(kpt_proj)
        return proj_data, color_ind

    def get_proj_plot(self, sys_name, max_ps=5.0, xlim=None, ylim=None):
        data = self.get_plot_data(xlim, ylim, is_separate=True)
        proj_data, color_ind = self.get_proj_data(data["band_ind"], max_ps, xlim)
        if isinstance(data["eigen"], OrderedDict):
            for spin in [Spin.up, Spin.dn]:
                files = file_name_list(sys_name + "_" + spin.name)
                dist_eigen = []
                for i in data["eigen"][spin]:
                    for j in i[1:]:
                        dist_eigen.append([i[0], j])
                proj = column_stack(dist_eigen, proj_data[spin])
                np.savetxt(files["eig"], data["eigen"][spin], fmt="%10.3f")
                np.savetxt(files["pro"], proj, fmt="%10.3f")
                str_to_file(self.generate_proj_gp(
                    files, color_ind, data["num"][spin], xlim, ylim), files["gp"])
                execute_gnuplot(files)
        else:
            files = file_name_list(sys_name)
            dist_eigen = []
            for i in data["eigen"]:
                for j in i[1:]:
                    dist_eigen.append([i[0], j])
            proj = column_stack(dist_eigen, proj_data[Spin.up])
            np.savetxt(files["eig"], data["eigen"], fmt="%10.3f")
            np.savetxt(files["pro"], proj, fmt="%10.3f")
            num_bands = sum(data["num"].values())
            str_to_file(self.generate_proj_gp(
                files, color_ind, num_bands, xlim, ylim), files["gp"])
            execute_gnuplot(files)

    def generate_proj_gp(self, files, color_ind, num_bands, xlim=None, ylim=None):
        ticks = self.get_ticks(xlim)
        gp_script = """
set term postscript eps enhanced color font "Times-Roman, 80" size 18, 10.125
set output "{eps_file}"
set multiplot layout 1, 2 title "" columnsfirst
set size 0.9, 1.0
set xtics ({x_labels})
set ytics scale 3
set ylabel 'Energy (eV)'
set palette defined ({color_list})
set arrow from graph 0, first 0 to graph 1, first 0 nohead dt 6 lw 1
{vertical_line}
unset key
unset colorbox
plot [][{ylim}] for [i=1:{bands}] "{eig_file}" u 1:(column(i+1)) w l lt 8 lw 2,\\
for [i=1:{orbs}] "{pro_file}" u 1:2:(column(2*i+1)):(column(2*i+2)) w points pt 7 ps variable lc palette

reset
set size 0.15, 1.0
set origin 0.85, 0
unset key
unset border
unset xtics
unset ytics
set yrange[0:3]
set xrange[0.0:1.5]
"""
        vertical_line = ""
        x_labels = ""
        for i in range(len(ticks["label"])):
            vertical_line += "set arrow from first {0:.3f}, graph 0 to " \
                             "first {0:.3f}, graph 1 nohead " \
                             "lt 8 lw 1\n".format(ticks["distance"][i])
            x_labels += '"{{{0}}}" {1:.3f}, '.format(ticks["label"][i],
                                                     ticks["distance"][i])
        x_labels = x_labels[:-2]
        ylim = "" if ylim is None else ":".join("%.3f" % i for i in ylim)

        is_single_orb = True if len(color_ind) == 1 else False
        color_dict = {0: "greenyellow", 1: "skyblue", 2: "skyblue", 3: "orchid",
                      4: "dark-cyan", 5: "forest-green", 6: "forest-green",
                      7: "web-blue", 8: "red", 9: "dark-violet",
                      10: "gold", 11: "coral"}
        color_list = ""
        for i in color_ind:
            color_list += '%s "%s", ' % (i, color_dict[i])
        if is_single_orb:
            color_list += '%s "%s", ' % (color_ind[0], color_dict[color_ind[0]])
        color_list = color_list[:-2]

        legend_place = 2.5
        num_orbs = len(color_ind)
        for i in color_ind:
            legend_place -= 0.2
            gp_script += 'set label "%s" at 0.0, %s font "Times-Roman, 60"\n'\
                         % (Color(i).name, legend_place)

        gp_script += "plot "
        for i in color_ind:
            gp_script += ''''-' w points pt 7 ps 5 lc "%s", ''' % color_dict[i]

        legend_place = 2.5
        for i in range(num_orbs):
            legend_place -= 0.2
            gp_script += "\n1.1 %s\ne" % legend_place

        return gp_script.format(x_labels=x_labels, vertical_line=vertical_line,
                                eps_file=files["eps"], color_list=color_list,
                                bands=num_bands, orbs=num_orbs, ylim=ylim,
                                eig_file=files["eig"], pro_file=files["pro"])


class LocpotPlotter(object):
    def __init__(self, pot_dict, structure, n, tol=0.25):
        self.pot_dict = pot_dict
        self.structure = structure
        layer_info = self.structure.get_layer_info(n, tol)
        self.layer_names = layer_info["l_name"]
        self.layer_positions = [sum([j.coords[2] for j in i]) / len(i)
                                for i in layer_info["layers"]]

    def get_plot(self, sys_name):
        try:
            from itertools import zip_longest
        except ImportError:
            from itertools import izip_longest as zip_longest

        is_spin = True if len(self.pot_dict.keys()) == 2 else False
        for spin, values in self.pot_dict.items():
            outs = list(zip_longest(*[values["all_z"], values["all_pot"],
                                       values["marc_z"], values["marc_pot"]]))
            if is_spin:
                files = file_name_list(sys_name + "_" + spin.name)
            else:
                files = file_name_list(sys_name)
            for i, v in enumerate(outs):
                outs[i] = "\t".join(["%.4f" % j for j in v if j is not None])
            str_to_file("\n".join(outs), files["pot"])
            str_to_file(self.generate_gp(files, values["all_z"][-1]), files["gp"])
            execute_gnuplot(files)

    def generate_gp(self, files, xmax):
        gp_script = """
set term postscript eps enhanced color font "Times-Roman, 80" size 18, 10.25
set output "{eps_file}"
set key top left samplen 1
set xtics rotate scale 3 font ", 50" ({x_labels})
set ytics scale 3
set ylabel 'Average POT (ev)'

plot [0:{xmax}][] "{data}" u 1:2 w l lt 1 lc rgb 'red' lw 5 title "Planar average",\\
"{data}" u 3:4 w l lt 5 lc rgb 'dark-green' lw 7 title "Macroscopic average"
"""
        x_labels = ""
        for i in range(len(self.layer_names)):
            x_labels += ''', "{{{0}}}" {1:.3f}'''.format(self.layer_names[i],
                                                         self.layer_positions[i])
        x_labels = x_labels[2:]
        return gp_script.format(eps_file=files["eps"], data=files["pot"],
                                xmax=xmax, x_labels=x_labels)


def file_name_list(sys_name):
    files = {"dos": "{}_dos.dat",
             "eig": "{}_eig.dat",
             "pro": "{}_pro.dat",
             "pot": "{}_pot.dat",
             "gp": "{}.gp",
             "eps": "{}.eps"
             }
    return {k: v.format(sys_name) for k, v in files.items()}


def execute_gnuplot(files):
    try:
        subprocess.check_output(["gnuplot", files["gp"]])
    except subprocess.CalledProcessError:
        raise RuntimeError("aimsflow encounts a gnuplot error. Make sure your "
                           "gnuplot is at least 5.0")
    print("Successfully plot %s" % files["eps"])

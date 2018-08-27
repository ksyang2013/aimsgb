import os
import re
import sys
import bz2
import gzip
import shutil
import warnings
from collections import OrderedDict, defaultdict

try:
    import yaml

    try:
        from yaml import CLoader as Loader, CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper
except ImportError:
    yaml = None

PY_VERSION = sys.version_info


def get_str_btw_lines(in_str, start_str, end_str):
    target_str = re.findall(r"^.+?{}.+?\n((?:.*\n)+?).+?{}".format(start_str, end_str),
                            in_str, re.M)
    return target_str


def clean_lines(line_list, remove_newline=True, remove_comment=True):
    for i in line_list:
        clean_line = i
        if '#' in clean_line and remove_comment:
            index = clean_line.index('#')
            clean_line = clean_line[:index]
        clean_line = clean_line.strip()
        if (not remove_newline) or clean_line != '':
            yield clean_line


def zopen(filename, *args, **kwargs):
    file_ext = filename.split('.')[-1].upper()
    if file_ext == "BZ2":
        if PY_VERSION[0] >= 3:
            return bz2.open(filename, *args, **kwargs)
        else:
            args = list(args)
            if len(args) > 0:
                args[0] = ''.join([c for c in args[0] if c != 't'])
            if "mode" in kwargs:
                kwargs["mode"] = ''.join([c for c in kwargs["mode"]
                                          if c != 't'])
            return bz2.BZ2File(filename, *args, **kwargs)
    elif file_ext in ("GZ", 'Z'):
        return gzip.open(filename, *args, **kwargs)
    else:
        return open(filename, *args, **kwargs)


def file_to_str(filename):
    with zopen(filename) as f:
        return f.read().rstrip()


def str_to_file(string, filename):
    with zopen(filename, 'wt') as f:
        f.write(string)


def file_to_lines(filename, no_emptyline=False):
    with zopen(filename) as f:
        if no_emptyline:
            return [i.strip("\n") for i in f.readlines() if i != '\n']
        return [i.strip("\n") for i in f.readlines()]


def immed_subdir(work_dir):
    return sorted([name for name in os.listdir(work_dir)
                   if os.path.isdir(os.path.join(work_dir, name))])


def immed_subdir_paths(work_dir):
    return sorted([os.path.join(work_dir, name) for name in os.listdir(work_dir)
                   if os.path.isdir(os.path.join(work_dir, name))])


def immed_files(work_dir):
    return sorted([name for name in os.listdir(work_dir)
                   if not os.path.isdir(os.path.join(work_dir, name))])


def immed_file_paths(folder):
    return sorted([os.path.join(folder, name) for name in os.listdir(folder)
                   if not os.path.isdir(os.path.join(folder, name))])


def make_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def loadfn(fn, *args, **kwargs):
    with zopen(fn) as fp:
        if "yaml" in fn.lower():
            if yaml is None:
                raise RuntimeError("Loading of YAML files is not "
                                   "possible as PyYAML is not installed.")
            if "Loader" not in kwargs:
                kwargs["Loader"] = Loader
            return yaml.load(fp, *args, **kwargs)


def ordered_loadfn(fn, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader): pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return loadfn(fn, Loader=OrderedLoader)


class Literal(str): pass


def literal_presenter(dumper, data):
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:%s'
                                   % type(data).__name__, data)


def dict_presenter(dumper, data):
    return dumper.represent_dict(data.items())


def literal_dumpfn(obj, fn, *args, **kwargs):

    def process_literal(obj):
        for key, value in obj.items():
            try:
                process_literal(value)
            except AttributeError:
                try:
                    if "\n" in value:
                        obj[key] = Literal(value)
                except TypeError:
                    pass
        return obj

    obj = process_literal(obj)
    return dumpfn(obj, fn, *args, **kwargs)


def dumpfn(obj, fn, *args, **kwargs):
    with zopen(fn, "wt") as fp:
        if "yaml" in fn.lower():
            if yaml is None:
                raise RuntimeError("Loading of YAML files is not "
                                   "possible as PyYaml is not installed.")
            if "Dumper" not in kwargs:
                kwargs["Dumper"] = Dumper
            yaml.add_representer(Literal, literal_presenter)
            yaml.add_representer(OrderedDict, dict_presenter)
            yaml.add_representer(defaultdict, dict_presenter)
            yaml.add_representer(type(obj), dict_presenter)
            yaml.dump(obj, fp, *args, **kwargs)


def zpath(filename):
    for ext in ["", ".gz", ".GZ", ".bz2", ".BZ2", ".z", ".Z"]:
        zfilename = "{}{}".format(filename, ext)
        if os.path.exists(zfilename):
            return zfilename
    return filename


def copy_r(src, dst):
    abs_src = os.path.abspath(src)
    abs_dst = os.path.abspath(dst)
    make_path(abs_dst)
    for f in os.listdir(abs_src):
        f_path = os.path.join(abs_src, f)
        if os.path.isfile(f_path):
            shutil.copy(f_path, abs_dst)
        elif not abs_dst.startswith(f_path):
            copy_r(f_path, os.path.join(abs_dst, f))
        else:
            warnings.warn("Cannot copy %s to itself" % f_path)

# def get_layer_info(self, tol=0.25, reverse=False, complex_layer=False):
#     layers = self.sort_sites_in_layers(tol, reverse)
#     dists = []
#     for i in range(1, len(layers)):
#         l1 = [s.coords[2] for s in layers[i - 1]]
#         l2 = [s.coords[2] for s in layers[i]]
#         dists.append(sum(l2) / len(l2) - sum(l1) / len(l1))
#     l_name = [Composition("".join([i.species_string for i in j]))
#               for j in layers]
#     if_ind = []
#     if_dist = []
#     if_name = []
#     for i in range(2, len(l_name)):
#         if not complex_layer:
#             if l_name[i] in l_name[i - 2:i]:
#                 continue
#         else:
#             if l_name[i] in l_name[i - 2:i] or \
#                     (i != len(l_name) - 1 and
#                              l_name[i + 1] in l_name[i - 1:i + 1]):
#                 continue
#         if_ind.append((i - 1, i))
#         if_dist.append(dists[i - 1])
#         if_name.append("/".join([l_name[i - 1].reduced_formula,
#                                  l_name[i].reduced_formula]))
#
#     return {"layers": layers, "dists": dists, "if_ind": if_ind,
#             "if_dist": if_dist, "if_name": if_name}
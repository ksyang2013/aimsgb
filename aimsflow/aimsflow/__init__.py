import os
from aimsflow.util.io_utils import loadfn

SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".afrc.yaml")


def _load_aimsflow_settings():
    try:
        import yaml
        with open(SETTINGS_FILE, "r") as f:
            d = yaml.load(f)
    except IOError:
        d = {}
        for k, v in os.environ.items():
            if k.startswith("AF_"):
                d[k] = v
            elif k in ["VASP_PSP_DIR", "DEFAULT_FUNCTIONAL"]:
                d["AF_" + k] = v
    clean_d = {}
    for k, v in d.items():
        if not k.startswith("AF_"):
            clean_d["AF_" + k] = v
        else:
            clean_d[k] = v
    try:
        PSP_DIR = clean_d["AF_VASP_PSP_DIR"]
    except KeyError:
        raise KeyError("Please set the AF_VASP_PSP_DIR environment in ~/.afrc.yaml "
                       "E.g. AF_VASP_PSP_DIR: ~/psp")
    try:
        MANAGER = clean_d["AF_MANAGER"]
        TIME_TAG = '-t' if MANAGER == 'SLURM' else 'walltime'
    except KeyError:
        raise KeyError("Please set the AF_MANAGER environment in ~/.afrc.yaml "
                       "E.g. AF_MANAGER: SLURM")
    return PSP_DIR, MANAGER, TIME_TAG


PSP_DIR, MANAGER, TIME_TAG = _load_aimsflow_settings()

from aimsflow.core import *

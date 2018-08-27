import os
import sys
import shutil
from aimsflow import SETTINGS_FILE
from aimsflow.util import loadfn, dumpfn


def add_config_var(args):
    d = {}
    if os.path.exists(SETTINGS_FILE):
        shutil.copy(SETTINGS_FILE, SETTINGS_FILE + ".bak")
        sys.stdout.write("Existing %s backed up to %s\n"
                         % (SETTINGS_FILE, SETTINGS_FILE + ".bak"))
        d = loadfn(SETTINGS_FILE)
    toks = args.var_spec
    if len(toks) % 2 != 0:
        sys.stderr.write("Bad variable specification!\n")
        sys.exit(0)
    for i in range(int(len(toks) / 2)):
        d[toks[2 * i]] = toks[2 * i + 1]
    dumpfn(d, SETTINGS_FILE, default_flow_style=False)
    sys.stdout.write("New %s written!\n" % (SETTINGS_FILE))


def configure_af(args):
    if args.var_spec:
        add_config_var(args)
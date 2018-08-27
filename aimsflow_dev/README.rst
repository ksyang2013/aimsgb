aimsflow is a simple and efficient VASP work flow framework written in
Python. aimsflow performs automatic error checking, job management and error
recovery, reducing the repetitive work as much as possible.

aimsflow also provides handy calculation analysis and structure building tools.
Currently, the package is still under development, so it is likey to be buggy

Using pip to install aimsflow on any system::

    pip install -e .

If get permission error, use::

    pip install -e . --user

Usage
=====

aimsflow allows you to do fine-grained control of jobs using a yaml spec file.
Let's say you need to do SOC calculations for BCC Fe with angle 0 and 90.

(1) Prepare the POSCAR. You can use aims, aimsflow and aimsgb. The POSCAR for
BCC Fe is in the `example <https://github.com/ksyang2013/AIMS.nano_yang_group_ucsd/tree/master/aimsflow/examples>`_ folder.
(2) Generate job_flow.yaml::

    aimsflow jyaml -g rs -soc 0

(3) Modify job_flow.yaml to add static calculation for angle 90. Simply add the
following two lines in the STATIC section::

     EXTRA
     - {"SOC": 90, SUFFIX: 90}

Here "SUFFIX" means extra folder name. In this case, aimsflow will create
"static" and "static90" folders. You can rename "SUFFIX" to whatever you want,
but you must have it there. If you want to do more angles, you can do::

 EXTRA:
 - {"SOC": 90, SUFFIX: 90}
 - {"SOC": 45, SUFFIX: 45}

If you want to change the INCAR setting or KPOINTS for static, you can do
(be aware of the format, support two formats, see yaml files in `example <https://github.com/ksyang2013/AIMS.nano_yang_group_ucsd/tree/master/aimsflow/examples>`_)::

 EXTRA:
 - {"SOC": 90, SUFFIX: 90, “INCAR”: {“ENCUT”: 500}}
 - {"SOC": 45, SUFFIX: 45, “INCAR”: {“ENCUT”: 500}, KPT: 600}
You can do the same thing for extra relax calculations.
(4) Generate VASP files::

 aimsflow vasp -p rs

(5) Submit VASP jobs::

 aimsflow vasp -s rs

aimsflow will do all the relaxation and static automatically.

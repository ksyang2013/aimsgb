from setuptools import setup, find_packages


long_desc = """
Introduction
============
aimsgb, an efficient and open-source Python library for generating atomic coordinates in periodic grain boundary models. It is designed to
construct various grain boundary structures from cubic and non-cubic initial
configurations. A convenient command line tool has also been provided to enable
easy and fast construction of tilt and twist boundaries by assigining the degree
of fit (Σ), rotation axis, grain boundary plane and initial crystal structure.
aimsgb is expected to greatly accelerate the theoretical investigation of
grain boundary properties and facilitate the experimental analysis of grain
boundary structures as well.

A reference for the usage of aimsGB software is:
Jianli Cheng, Jian Luo, and Kesong Yang, Aimsgb: An Algorithm and OPen-Source Python Library to Generate Periodic Grain Boundary Structures, Comput. Mater. Sci. 155, 92-103, (2018). 
DOI:10.1016/j.commatsci.2018.08.029  

Install aimsgb
==============
1. Clone the latest version from github::

    git clone git@github.com:ksyang2013/aimsgb.git

2. Navigate to aimsgb folder::

    cd aimsgb

3. Type in the root of the repo::

    pip install .

4. or to install the package in development mode::

    pip install -e .


How to cite aimsgb
==================

If you use aimsgb in your research, please consider citing the following work:

    Jianli Cheng, Jian Luo, Kesong Yang. *Aimsgb: An algorithm and open-source python
    library to generate periodic grain boundary structures.* Computational Materials
    Science, 2018, 155, 92-103. `doi:10.1016/j.commatsci.2018.08.029
    <https://doi.org/10.1016/j.commatsci.2018.08.029>`_


Copyright
=========
Copyright (C) 2018 The Regents of the University of California

All Rights Reserved. Permission to copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice, this paragraph and the following three paragraphs appear in all copies. Permission to make commercial use of this software may be obtained by contacting:

Office of Innovation and Commercialization
9500 Gilman Drive, Mail Code 0910
University of California
La Jolla, CA 92093-0910
(858) 534-5815
innovation@ucsd.edu

This software program and documentation are copyrighted by The Regents of the University of California. The software program and documentation are supplied “as is”, without any accompanying services from The Regents. The Regents does not warrant that the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason.

IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN “AS IS” BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.


Authors
=======
Dr. Jianli Cheng (jic198@ucsd.edu)
New Email: jianlicheng@lbl.gov

Prof. Kesong Yang  (kesong@ucsd.edu)

About the aimsgb Development Team
=================================
http://materials.ucsd.edu/
"""

setup(
    name="aimsgb",
    packages=find_packages(),
    version="0.1.3",
    setup_requires=["setuptools>=18.0"],
    install_requires=["pymatgen", "mp_api", "numpy"],
    include_package_data=True,
    author="Jianli Cheng and Kesong YANG",
    maintainer="Jianli Cheng, Sicong JIANG, and Kesong YANG",
    maintainer_email="chengjianli90@gmail.com, sij014@ucsd.edu, kesong@ucsd.edu",
    url="http://aimsgb.org",
    description="aimsgb is a python library for generatng the atomic "
                "coordinates of periodic grain boundaries."
                "Copyright © 2018 The Regents of the University of California."
                "All Rights Reserved. See more in Copyright.",
    long_description=long_desc,
    keywords=["material science", "grain boundary", "molecular simulation"],
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    entry_points={
        'console_scripts': [
            'aimsgb = aimsgb.agb:main',
        ]
    }
)

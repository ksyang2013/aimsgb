from io import open
from setuptools import setup, find_packages

with open("README.rst") as f:
    long_desc = f.read()

setup(
    name="aimsgb",
    packages=find_packages(),
    version="0.1.0",
    setup_requires=["setuptools>=18.0"],
    install_requires=["pymatgen"],
    include_package_data=True,
    author="Jianli Cheng and Kesong YANG",
    Principal Investigator (PI) = "Kesong YANG"
    PI_email="kesong@ucsd.edu",
    affliation="University of California San Diego"
    address="9500 Gilman Dr., MC 0448, La Jolla, CA, 92093-0448, USA"
    maintainer="Jianli Cheng",
    maintainer_email="jic198@ucsd.edu",
    url="https://github.com/ksyang2013/aimsgb"
    copyright_notice="This software is Copyright © 2018 The Regents of the University of California."
                     "All Rights Reserved. See more in LICENSE." 
    description="aimsgb is a python library for generatng the atomic "
                "coordinates of periodic grain boundaries.",
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
            'agb = aimsgb.agb:main',
            'aimsgb = aimsgb.agb:main',
        ]
    }
)

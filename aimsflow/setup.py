from io import open
from setuptools import setup, find_packages

with open("README.rst") as f:
    long_desc = f.read()

setup(
    name="aimsflow",
    packages=find_packages(),
    version="0.1.0",
    setup_requires=['numpy', 'setuptools>=18.0'],
    install_requires=["numpy>=1.9", "six>=1.10.0", "pyyaml>=3.11",
                      "scipy>=0.14", "tabulate", "spglib>=1.9.9.44",
                      "enum34>=1.1.6"],
    package_data={"aimsflow.core": ["*.json"],
                  "aimsflow.vasp_io": ["*.yaml"]},
    include_package_data=True,
    author="Jianli Cheng",
    author_email="jic198@ucsd.edu",
    maintainer="Jianli Cheng",
    url="https://github.com/jic198/aimsflow",
    description="aimsflow is a python package developed based on pymatgen.",
    long_description=long_desc,
    keywords=["vasp", "job", "management", "analysis"],
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
            'aimsflow = aimsflow.cli.af:main',
        ]
    }
)

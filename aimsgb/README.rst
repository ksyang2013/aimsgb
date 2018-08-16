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


Cite
====

If you use aimsgb for your work, please cite the paper:

License
=======

Aimsgb is released under the MIT License. The terms of the license are as
follows::

    The MIT License (MIT)
    Copyright (c) 2018 Prof. Kesong YANG Group at University of California San Diego

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
    the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Contributing
============

We try to follow the coding style used by pymatgen(PEP8):

http://pymatgen.org/contributing.html#coding-guidelines


Authors
=======
Dr. Jianli Cheng (jianlicheng@lbl.gov)

Prof. Kesong Yang  (kesong@ucsd.edu)

About the aimsgb Development Team
=================================
http://materials.ucsd.edu/

#!/usr/bin/python3
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of webloader (see TBD).
# See the LICENSE file for licensing terms (BSD-style).
#

#from __future__ import print_function

import glob
import sys
from distutils.core import setup  # , Extension, Command

if sys.version_info < (3, 6):
    sys.exit("Python versions less than 3.6 are not supported")

scripts = []

prereqs = """
    braceexpand
    numpy
    six
    Pillow
    simplejson
    msgpack
""".split()

setup(
    name='webdataset',
    version='v0.0',
    author="Thomas Breuel",
    description="Record sequential storage for deep learning.",
    packages=["webdataset"],
    scripts=scripts,
    install_requires=prereqs,
)

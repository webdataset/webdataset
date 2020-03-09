#!/usr/bin/python
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of the objectio library.
# See the LICENSE file for licensing terms (BSD-style).
#



def checktype(value, types, msg=""):
    """Type check value; raise ValueError if fails."""
    if not isinstance(value, types):
        raise ValueError(f"ERROR {msg}: {value} should be of type {types}")


def checkmember(value, values, msg=""):
    """Check value for membership; raise ValueError if fails."""
    if value not in values:
        raise ValueError(f"ERROR {msg}: {value} should be in {values}")


def checkrange(value, lo, hi, msg=""):
    """Check value for membership; raise ValueError if fails."""
    if value < lo or value > hi:
        raise ValueError(f"ERROR {msg}: {value} should be in range {lo} {hi}")


def check(value, msg=""):
    """Check value for membership; raise ValueError if fails."""
    if not value:
        raise ValueError(f"ERROR {msg}: {value} should be true")


def checkcallable(value, msg=""):
    """Check value for membership; raise ValueError if fails."""
    if not callable(value):
        raise ValueError(f"ERROR {msg}: {value} should be callable")


def checknotnone(value, msg=""):
    """Check value for membership; raise ValueError if fails."""
    if value is None:
        raise ValueError(f"ERROR {msg}: {value} should not be None")

#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""Miscellaneous utility functions."""

import sys
import re
import importlib
import itertools as itt
from typing import Any, Union, Optional, Callable, Iterator


def identity(x: Any) -> Any:
    """Return the argument as is."""
    return x


def safe_eval(s: str, expr: str = "{}"):
    """Evaluate the given expression more safely."""
    if re.sub("[^A-Za-z0-9_]", "", s) != s:
        raise ValueError(f"safe_eval: illegal characters in: '{s}'")
    return eval(expr.format(s))


def lookup_sym(sym: str, modules: list):
    """Look up a symbol in a list of modules."""
    for mname in modules:
        module = importlib.import_module(mname, package="webdataset")
        result = getattr(module, sym, None)
        if result is not None:
            return result
    return None


def repeatedly0(loader: Iterator, nepochs: int = sys.maxsize, nbatches: int = sys.maxsize):
    """Repeatedly returns batches from a DataLoader."""
    for epoch in range(nepochs):
        for sample in itt.islice(loader, nbatches):
            yield sample


def guess_batchsize(batch: Union[tuple, list]):
    """Guess the batch size by looking at the length of the first element in a tuple."""
    return len(batch[0])


def repeatedly(
    source: Iterator,
    nepochs: int = None,
    nbatches: int = None,
    nsamples: int = None,
    batchsize: Callable[..., int] = guess_batchsize,
):
    """Repeatedly yield samples from an iterator."""
    epoch = 0
    batch = 0
    total = 0
    while True:
        for sample in source:
            yield sample
            batch += 1
            if nbatches is not None and batch >= nbatches:
                return
            if nsamples is not None:
                total += guess_batchsize(sample)
                if total >= nsamples:
                    return
        epoch += 1
        if nepochs is not None and epoch >= nepochs:
            return

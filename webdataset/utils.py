#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""Miscellaneous utility functions."""

import fnmatch
import functools
import glob
import importlib
import itertools as itt
import os
import re
import sys
import warnings
from typing import Any, Callable, Iterator, Union

import numpy as np


def glob_with_braces(pattern):
    """Apply glob to patterns with braces by pre-expanding the braces."""
    expanded = braceexpand.braceexpand(pattern)
    return [f for pat in expanded for f in glob.glob(pat)]


def fnmatch_with_braces(filename, pattern):
    """Apply fnmatch to patterns with braces by pre-expanding the braces."""
    expanded = braceexpand.braceexpand(pattern)
    for pat in expanded:
        if fnmatch.fnmatch(filename, pat):
            return True
    return any(fnmatch.fnmatch(filename, pat) for pat in expanded)


def make_seed(*args):
    seed = 0
    for arg in args:
        seed = (seed * 31 + hash(arg)) & 0x7FFFFFFF
    return seed


def is_iterable(obj):
    if isinstance(obj, str):
        return False
    if isinstance(obj, bytes):
        return False
    if isinstance(obj, list):
        return True
    if isinstance(obj, Iterator):
        return True
    if isinstance(obj, Iterable):
        return True
    return False


class PipelineStage:
    def invoke(self, *args, **kw):
        raise NotImplementedError


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


def repeatedly0(
    loader: Iterator, nepochs: int = sys.maxsize, nbatches: int = sys.maxsize
):
    """Repeatedly returns batches from a DataLoader."""
    for _ in range(nepochs):
        yield from itt.islice(loader, nbatches)


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


def pytorch_worker_info(group=None):  # sourcery skip: use-contextlib-suppress
    """Return node and worker info for PyTorch and some distributed environments."""
    rank = 0
    world_size = 1
    worker = 0
    num_workers = 1
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        try:
            import torch.distributed

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                group = group or torch.distributed.group.WORLD
                rank = torch.distributed.get_rank(group=group)
                world_size = torch.distributed.get_world_size(group=group)
        except ModuleNotFoundError:
            pass
    if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
        worker = int(os.environ["WORKER"])
        num_workers = int(os.environ["NUM_WORKERS"])
    else:
        try:
            import torch.utils.data

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker = worker_info.id
                num_workers = worker_info.num_workers
        except ModuleNotFoundError:
            pass

    return rank, world_size, worker, num_workers


def pytorch_worker_seed(group=None):
    """Compute a distinct, deterministic RNG seed for each worker and node."""
    rank, world_size, worker, num_workers = pytorch_worker_info(group=group)
    return rank * 1000 + worker


def deprecated(arg=None):
    if callable(arg):
        # The decorator was used without arguments
        func = arg
        reason = None
    else:
        # The decorator was used with arguments
        func = None
        reason = arg

    def decorator(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            msg = f"Call to deprecated function {func.__name__}."
            if reason is not None:
                msg += " Reason: " + reason
            warnings.warn(
                msg,
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return new_func

    if func is None:
        # The decorator was used with arguments
        return decorator
    else:
        # The decorator was used without arguments
        return decorator(func)


def obsolete(func=None, *, reason=None):
    if func is None:
        return functools.partial(obsolete, reason=reason)

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        if int(os.environ.get("ALLOW_OBSOLETE", "0")):
            pass
        else:
            msg = f"Call to obsolete function {func.__name__}. Set env ALLOW_OBSOLETE=1 to permit."
            if reason is not None:
                msg += " Reason: " + reason
            raise Exception(msg)
        return func(*args, **kwargs)

    return new_func


def compute_sample_weights(n_w_pairs):
    ns = np.array([p[0] for p in n_w_pairs])
    ws = np.array([p[1] for p in n_w_pairs])
    weighted = ns * ws
    ps = weighted / np.amax(weighted)
    return ps

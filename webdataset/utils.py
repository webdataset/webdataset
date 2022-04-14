#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""Miscellaneous utility functions."""

import importlib
import itertools as itt
import os
import re
import sys
from typing import Any, Callable, Iterator, Optional, Union


def make_seed(*args):
    seed = 0
    for arg in args:
        seed = (seed * 31 + hash(arg)) & 0x7FFFFFFF
    return seed


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


def pytorch_worker_info(group=None):
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

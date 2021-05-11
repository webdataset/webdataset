#!/usr/bin/python
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""Functions related to splitting datasets by node and worker.
This follows the PyTorch model of magic global functions and environment
settings and supplies the default node and worker splitting functions.
This is provided mainly because PyTorch users expect something like this
to exist. The cleaner and safer way of dealing with node and worker splitting
is via explicit functions.
"""

import os
import random
import warnings
import itertools as itt
import socket
import braceexpand

from . import tariterators
from . import iterators
from . import autodecode
from . import shardcache
from . import dbcache
from . import utils
from . import gopen
from .utils import reraise_exception, lookup_sym, safe_eval

worker_environment = None


too_few_shards_warning = int(os.environ.get("WDS_WARNINGS", 1))


def worker_id():
    return socket.gethostname(), os.getpid()


class WorkerEnvironment:
    def __init__(self, rank=0, world_size=1, worker=0, nworkers=1):
        self.identity = worker_id()
        self.rank = rank
        self.world_size = world_size
        self.worker = worker
        self.nworkers = nworkers

    def __str__(self):
        return (
            f"WorkerEnvironment(rank={self.rank}, world_size={self.world_size}, "
            + f"worker={self.worker}, nworkers={self.nworkers})"
        )


class TorchWorkerEnvironment(WorkerEnvironment):
    def __init__(self, group=None):
        import torch
        import torch.distributed

        super().__init__()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if group is None:
                group = torch.distributed.group.WORLD
            self.rank = torch.distributed.get_rank(group=group)
            self.world_size = torch.distributed.get_world_size(group=group)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            self.worker = worker_info.id
            self.nworkers = worker_info.num_workers


def get_worker_environment():
    global worker_environment
    if worker_environment is not None and worker_environment.identity == worker_id():
        return worker_environment
    try:
        worker_environment = TorchWorkerEnvironment()
        return worker_environment
    except ModuleNotFoundError:
        pass
    worker_environment = WorkerEnvironment()
    return worker_environment


def split_by_node(urls, env=None):
    """Selects a subset of urls based on node info.

    Used as a shard selection function in Dataset."""
    env = env or get_worker_environment()
    urls = [url for url in urls]
    assert isinstance(urls, list)
    import socket

    gopen.info["rank"] = env.rank
    gopen.info["size"] = env.world_size
    gopen.info["host"] = socket.gethostname()
    gopen.info["pid"] = os.getpid()
    if too_few_shards_warning and env.rank == 0 and len(urls) < env.world_size:
        warnings.warn(f"world_size {env.world_size} > num_shards {len(urls)}")
    return urls[env.rank :: env.world_size]


def split_by_worker(urls, env=None):
    """Selects a subset of urls based on worker info.

    Used as a shard selection function in Dataset."""
    env = env or get_worker_environment()
    urls = [url for url in urls]
    assert isinstance(urls, list)
    gopen.info["worker_id"] = env.worker
    gopen.info["num_workers"] = env.nworkers
    if too_few_shards_warning and env.worker == 0 and len(urls) < env.nworkers:
        warnings.warn(f"num_workers {env.nworkers} > num_shards {len(urls)}")
    return urls[env.worker :: env.nworkers]

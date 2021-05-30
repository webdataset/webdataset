#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
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

import sys
import os
import warnings
import socket

from . import gopen
from itertools import islice
from functools import partial


worker_environment = None


too_few_shards_warning = int(os.environ.get("WDS_WARNINGS", 1))


def worker_id():
    """Return an identifier for the current worker."""
    return socket.gethostname(), os.getpid()


class WorkerEnvironment:
    """Encapsulates the runtime environment of the worker."""

    def __init__(self, rank=0, world_size=1, worker=0, nworkers=1):
        """Initialize the worker environment."""
        self.identity = worker_id()
        self.rank = rank
        self.world_size = world_size
        self.worker = worker
        self.nworkers = nworkers
        self.rank_init = False
        self.worker_init = False

    def __str__(self):
        """__str__."""
        return (
            f"WorkerEnvironment(rank={self.rank}, world_size={self.world_size}, {self.rank_init}, "
            + f"worker={self.worker}, nworkers={self.nworkers}, {self.worker_init})"
        )


class TorchWorkerEnvironment(WorkerEnvironment):
    """TorchWorkerEnvironment."""

    def __init__(self, group=None):
        """Initialize the worker environment for Torch.

        :param group: torch.distributed group for determining rank/size
        """
        import torch
        import torch.distributed

        super().__init__()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if group is None:
                group = torch.distributed.group.WORLD
            self.rank_init = True
            self.rank = torch.distributed.get_rank(group=group)
            self.world_size = torch.distributed.get_world_size(group=group)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            self.worker_init = True
            self.worker = worker_info.id
            self.nworkers = worker_info.num_workers
            

def get_worker_environment():
    """Get the current worker environment."""
    global worker_environment
    if worker_environment is not None and worker_environment.identity == worker_id():
        print("cached", worker_environment)
        return worker_environment
    try:
        worker_environment = TorchWorkerEnvironment()
        print("torch", worker_environment)
        return worker_environment
    except ModuleNotFoundError:
        pass
    print("default", worker_environment)
    worker_environment = WorkerEnvironment()
    return worker_environment


def nodeslice(source, env=None):
    """Slice the source based on the rank and worker number."""
    env = env or get_worker_environment()
    offset = env.rank * env.nworkers + env.worker
    step = env.world_size * env.nworkers
    yield from islice(source, offset, sys.maxsize, step)


def split_by_node(urls, env=None):
    """Select a subset of urls based on node info.

    Used as a shard selection function in Dataset.
    """
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
    """Select a subset of urls based on worker info.

    Used as a shard selection function in Dataset.
    """
    env = env or get_worker_environment()
    urls = [url for url in urls]
    assert isinstance(urls, list)
    gopen.info["worker_id"] = env.worker
    gopen.info["num_workers"] = env.nworkers
    if too_few_shards_warning and env.worker == 0 and len(urls) < env.nworkers:
        warnings.warn(f"num_workers {env.nworkers} > num_shards {len(urls)}")
    return urls[env.worker :: env.nworkers]


class RandomNodeSplitter:
    """Randomly split shards between nodes.

    Used as a shard selection function in Dataset.
    In epoch-based training loops, this splitter can be used to ensure shards are distributed
    differently among nodes at each epoch. A typical use is to call `set_seed(epoch)` before
    iterating over the dataset in each epoch.
    """

    def __init__(self, start_seed=0):
        """Initialize the node splitter with a fixed seed."""
        self.set_seed(start_seed)

    def set_seed(self, seed):
        """Change the seed to change the node splitting on the next epoch."""
        self.seed = seed

    def __call__(self, urls, env=None):
        """Perform both per-epoch shuffling and node splitting."""
        urls = urls.copy()
        random.Random(self.seed).shuffle(list(urls))
        return split_by_node(urls, env)

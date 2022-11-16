#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Train PyTorch models directly from POSIX tar archive.

Code works locally or over HTTP connections.
"""

import os
import random
import sys
import time
from dataclasses import dataclass, field
from itertools import islice
from typing import List

import braceexpand
import yaml

from .composable import Composable
from .filters import Curried
from .pytorch import IterableDataset


class SimpleShardList(IterableDataset, Composable):
    """An iterable dataset yielding a list of urls."""

    def __init__(self, urls, seed=None):
        """Iterate through the list of shards.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        if isinstance(urls, str):
            urls = list(braceexpand.braceexpand(urls))
        else:
            urls = list(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.seed = seed

    def __len__(self):
        return len(self.urls)

    def __iter__(self):
        """Return an iterator over the shards."""
        urls = self.urls.copy()
        if self.seed is not None:
            random.Random(self.seed).shuffle(urls)
        for url in urls:
            yield dict(url=url)


def split_by_node(src, group=None):
    import torch.distributed

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        group = group or torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        for s in islice(src, rank, None, size):
            yield s
    else:
        for s in src:
            yield s

def split_by_node_sm(src, group=None):
    # for SageMaker
    import smdistributed.dataparallel.torch.distributed as dist
    
    if dist.is_available() and dist.is_initialized():
        group = group or dist.group.WORLD
        rank = dist.get_rank(group=group)
        size = dist.get_world_size(group=group)
        for s in islice(src, rank, None, size):
            yield s
    else:
        for s in src:
            yield s

def split_by_worker(src):
    import torch.utils.data

    winfo = torch.utils.data.get_worker_info()
    if winfo is None:
        for s in src:
            yield s
    else:
        for s in islice(src, winfo.id, None, winfo.num_workers):
            yield s


def resampled_(src, n=sys.maxsize):
    import random

    seed = time.time_ns()
    try:
        seed = open("/dev/random", "rb").read(20)
    except Exception as exn:
        print(repr(exn)[:50], file=sys.stderr)
    rng = random.Random(seed)
    print(f"# resampled loading", file=sys.stderr)
    items = list(src)
    print(f"# resampled got {len(items)} samples, yielding {n}", file=sys.stderr)
    for i in range(n):
        yield rng.choice(items)


resampled = Curried(resampled_)


def non_empty(src):
    count = 0
    for s in src:
        yield s
        count += 1
    if count == 0:
        raise ValueError("pipeline stage received no data at all and this was declared as an error")


# BELOW IS DEPRECATED ###


class PytorchEnv:
    """A class encapsulating the PyTorch node/worker environment.
      
       2021.12.24
       sagemaker の smdistributedを利用できるようにオプションを追加
    """
    
    def __init__(self, group=None, sagemaker=False):
        """Initialize rank/worker information."""
        import socket

        super().__init__()
        self.rank = None
        self.worker = None
        self.group = group
        self.sagemaker = sagemaker
        self.nodeinfo = (socket.gethostname(), os.getpid())
        if sagemaker:
            # sagemaker 利用のオプション押下時
            self.update_sm_env()
        else:
            self.update_env()

    def update_env(self):
        """Update information about node and worker environment.

        This code is written this way because the torch.distributed info is
        available only in the environment where the loader is created.
        This class retains that environment info when it is serialized.
        """

        from . import gopen

        try:
            import torch
            import torch.distributed
        except Exception:
            return

        if self.rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                group = self.group or torch.distributed.group.WORLD
                self.rank = torch.distributed.get_rank(group=group), torch.distributed.get_world_size(
                    group=group
                )

        if self.worker is None:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                self.worker = worker_info.id, worker_info.num_workers

        gopen.info["nodeinfo"] = self.nodeinfo
        gopen.info["rank"], gopen.info["size"] = self.rank or (-1, -1)
        gopen.info["worker_id"], gopen.info["num_workers"] = self.worker or (-1, -1)

    def update_sm_env(self):
        """smdistributed.dataparallel.torch.distributed を利用してupdate_env と同じ動作を実現"""
        from . import gopen
        
        try:
            import torch
            import smdistributed.dataparallel.torch.distributed as dist
        except Exception:
            return

        if self.rank is None:
            if dist.is_available() and dist.is_initialized():
                group = self.group or dist.group.WORLD
                self.rank = dist.get_rank(group=group), dist.get_world_size(
                    group=group
                )

        if self.worker is None:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                self.worker = worker_info.id, worker_info.num_workers

        gopen.info["nodeinfo"] = self.nodeinfo
        gopen.info["rank"], gopen.info["size"] = self.rank or (-1, -1)
        gopen.info["worker_id"], gopen.info["num_workers"] = self.worker or (-1, -1)

        
class ShardSample:
    pass


class SimpleShardSample(ShardSample):
    def __init__(self, urls):
        if isinstance(urls, str):
            urls = list(braceexpand.braceexpand(urls))
        else:
            urls = list(urls)
        self.urls = list(urls)
        assert isinstance(self.urls[0], str)

    def sample(self):
        return self.urls.copy()


@dataclass
class MSSource:
    """Class representing a data source."""

    name: str = ""
    perepoch: int = -1
    resample: bool = False
    urls: List[str] = field(default_factory=list)


default_rng = random.Random()


def expand(s):
    return os.path.expanduser(os.path.expandvars(s))


class MultiShardSample(ShardSample):
    def __init__(self, fname):
        """Construct a shardlist from multiple sources using a YAML spec."""
        self.rng = default_rng  # capture default_rng if we fork
        with open(fname) as stream:
            spec = yaml.safe_load(stream)
        assert set(spec.keys()).issubset(set("prefix datasets".split()))
        prefix = expand(spec.get("prefix", ""))
        self.sources = []
        for ds in spec["datasets"]:
            assert set(ds.keys()).issubset(set("buckets name shards perepoch choose".split()))
            buckets = [expand(s) for s in ds.get("buckets", [""])]
            assert len(buckets) == 1, "FIXME support for multiple buckets unimplemented"
            bucket = buckets[0]
            name = ds.get("name", "@" + bucket)
            urls = ds["shards"]
            urls = [u for url in urls for u in braceexpand.braceexpand(url)]
            urls = [prefix + bucket + u for url in urls for u in braceexpand.braceexpand(url)]
            resample = ds.get("choose", -1)
            nsample = ds.get("perepoch", -1)
            if nsample > len(urls):
                raise ValueError(f"perepoch {nsample} must be no greater than the number of shards")
            if (nsample > 0) and (resample > 0):
                raise ValueError("specify only one of perepoch or choose")
            entry = MSSource(name=name, urls=urls, perepoch=nsample, resample=resample)
            self.sources.append(entry)
            print(f"# {name} {len(urls)} {nsample}", file=sys.stderr)

    def set_epoch(self, seed):
        """Set the current epoch (for consistent shard selection among nodes)."""
        self.rng = random.Random(seed)

    def sample(self):
        result = []
        for source in self.sources:
            if source.resample > 0:
                # sample with replacement
                l = self.rng.choices(source.urls, k=source.resample)
            elif source.perepoch > 0:
                # sample without replacement
                l = list(source.urls)
                self.rng.shuffle(l)
                l = l[: source.perepoch]
            else:
                l = list(source.urls)
            result += l
        self.rng.shuffle(result)
        return result


class PytorchShardList(IterableDataset, PytorchEnv, Composable):
    """An iterable dataset yielding a list of urls.

    This understands the PyTorch distributed and worker APIs and splits shards
    accordingly.
    """

    def __init__(
            self,
            urls,
            epoch_shuffle=False,
            shuffle=True,
            split_by_worker=True,
            split_by_node=True,
            verbose=False,
            sagemaker=False
    ):
        """Create a ShardList.

        :param urls: a list of URLs as a Python list or brace notation string
        :param shuffle: shuffle samples before iterating
        :param split_by_node: split shards by node if True
        :param split_by_worker: split shards by worker if True
        :param group: group used for determining rank/world_size

        If WDS_SHUFFLE is in the environment, it is used for shuffling shards prior
        to splitting; this assigns different shards to different nodes on each epoch.
        """
        # 継承元クラスの初期化の方法を変更する
        # PytorchEnv のみ sagemaker のオプションを渡す
        IterableDataset.__init__(self)
        PytorchEnv.__init__(self, sagemaker=sagemaker)
        Composable.__init__(self)
        # super().__init__()
        
        self.verbose = verbose
        if self.verbose:
            print("PytorchShardList init")
        self.epoch = -1
        self.epoch_shuffle = epoch_shuffle
        self.shuffle = shuffle
        self.split_by_worker = split_by_worker
        self.split_by_node = split_by_node_sm if sagemaker else split_by_node
        if not isinstance(urls, ShardSample):
            urls = SimpleShardSample(urls)
        self.shardsample = urls

    def set_epoch(self, epoch):
        """Set the current epoch. Used for per-node shuffling."""
        self.epoch = epoch - 1

    def __iter__(self):
        """Return an iterator over the shards."""
        self.epoch += 1
        if hasattr(self.shardsample, "set_epoch"):
            self.shardsample.set_epoch(self.epoch)
        self.update_env()
        urls = self.shardsample.sample()
        if self.epoch_shuffle:
            if "WDS_EPOCH" not in os.environ:
                raise ValueError(
                    "when specifying epoch_shuffle, you must provide the epoch in the WDS_EPOCH environment variable"
                )
            epoch = int(os.environ["WDS_EPOCH"])
            if self.verbose:
                print(f"PytorchShardList epochshuffle {epoch}")
            random.Random(epoch).shuffle(urls)
        if self.split_by_node:
            rank, world = self.rank or (0, 1)
            if self.verbose:
                print(f"PytorchShardList rank {rank} of {world}")
            urls = urls[rank::world]
        if self.split_by_worker:
            worker, nworkers = self.worker or (0, 1)
            if self.verbose:
                print(f"PytorchShardList worker {worker} of {nworkers}")
            urls = urls[worker::nworkers]
        if self.shuffle:
            random.Random(self.epoch + 17).shuffle(urls)
        if self.verbose:
            print(f"PytorchShardList got {len(urls)} urls")
        for url in urls:
            yield dict(
                url=url,
                __url__=url,
                __worker__=str(self.worker),
                __rank__=str(self.rank),
                __nodeinfo__=str(self.nodeinfo),
            )


class ResampledShards(IterableDataset, Composable):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        env=None,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        if isinstance(urls, str):
            urls = list(braceexpand.braceexpand(urls))
        else:
            urls = list(urls)
        self.urls = urls
        self.nshards = nshards
        self.env = env or PytorchEnv()
        self.rng = random.Random()
        assert isinstance(self.urls[0], str)

    def __iter__(self):
        """Return an iterator over the shards."""
        rank, world = self.env.rank or (0, 1)
        worker, nworkers = self.env.worker or (0, 1)
        seed = (rank, worker, os.getpid(), time.time())
        self.rng.seed(seed)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))

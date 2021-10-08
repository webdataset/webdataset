#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Train PyTorch models directly from POSIX tar archive.

Code works locally or over HTTP connections.
"""

import os
import sys
import yaml
from dataclasses import dataclass

import braceexpand

from .composable import Composable, Shorthands
from .utils import lookup_sym, safe_eval
from .handlers import reraise_exception
from .pytorch import IterableDataset
from .shardlists import PytorchShardList, ResampledShards
from typing import Iterator, Any, Optional

default_cache_dir = os.path.expanduser(os.environ.get("WEBDATASET_CACHE", ""))
default_cache_name = lookup_sym(os.environ.get("WEBDATASET_CACHE_NAME", "shard_uuid"), ".shardcache".split())
default_cache_verbose = int(safe_eval(os.environ.get("WEBDATASET_CACHE_VERBOSE", "1")))
default_cache_size = int(float(safe_eval(os.environ.get("WEBDATASET_CACHE_SIZE", "1e15"))))


@dataclass
class Source:
    """Class representing a data source."""

    dataset: IterableDataset
    probability: float = 1.0
    source: Optional[Iterator[Any]] = None
    comment: str = ""


class RoundRobin(IterableDataset, Composable, Shorthands):
    """Iterate through datasets in a round-robin way."""

    def __init__(self, sources=None):
        """Initialize from a set of sources."""
        super().__init__()
        self.sources = sources if sources is not None else []

    def add_dataset(self, dataset, probability=1.0, comment=""):
        self.sources.append(Source(dataset=dataset, probability=probability, comment=comment))

    def __iter__(self):
        """Iterate through the list of sources in a round-robin way until all sources have been exhausted."""
        index = 0
        iters = [s for s in self.sources]
        for s in iters:
            s.source = iter(s.dataset)
        while len(iters) > 0:
            try:
                sample = next(iters[index].source)
                yield sample
            except StopIteration:
                del iters[index]
            index += 1
            if index >= len(iters):
                index = 0

    def __str__(self):
        return f"RoundRobin({self.sources})"


def check_allowed(d, allowed, name="dictionary"):
    allowed = set(allowed.split())
    actual = set(d.keys())
    extra = actual.difference(allowed)
    if extra != set():
        raise ValueError(f"{name} has extra keys {extra}")


def construct_dataset(
    fname,
    cache_dir=default_cache_dir,
    cache_size=default_cache_size,
    cache_name=default_cache_name,
    cache_verbose=default_cache_verbose,
    chunksize=10,
    handler=reraise_exception,
    repeat=False,
):
    """Construct a composite dataset from multiple sources using a YAML spec.

    This function gets invoked when you construct a WebDataset from a
    ".ds.yml" specification file.

    This is an experimental function that constructs composite datasets.
    You may want to opt for the simpler .shards.yml spec, which specifies
    combining datasets at the shard level; it interacts more simply with
    workers and distributed training.
    """
    from .dataset import WebDataset, WebLoader
    with open(fname) as stream:
        spec = yaml.safe_load(stream)
    result = []
    check_allowed(spec, "prefix datasets epoch", "top datasets spec")
    prefix = spec.get("prefix", "")
    for ds in spec["datasets"]:
        check_allowed(
            ds,
            """
            name buckets shards resampled epoch_shuffle shuffle split_by_worker
            split_by_node cachedir cachesize cachename cacheverbose subsample
            shuffle epoch chunksize nworkers probability
        """,
            "dataset spec",
        )
        buckets = ds.get("buckets", [""])
        assert len(buckets) == 1, "FIXME support for multiple buckets unimplemented"
        bucket = buckets[0]
        urls = ds["shards"]
        urls = [u for url in urls for u in braceexpand.braceexpand(url)]
        urls = [prefix + bucket + u for u in urls]
        print(
            f"# input {ds.get('name', '')} {prefix+bucket+str(ds['shards'])} {len(urls)} "
            + f"{ds.get('epoch')} {ds.get('resampled')}",
            file=sys.stderr,
        )
        if ds.get("resampled", False):
            urls = ResampledShards(urls)
        else:
            urls = PytorchShardList(
                urls,
                epoch_shuffle=ds.get("epoch_shuffle", False),
                shuffle=ds.get("shuffle", True),
                split_by_worker=ds.get("split_by_worker", True),
                split_by_node=ds.get("split_by_node", True),
            )
        dataset = WebDataset(
            urls,
            ds.get("cachedir", cache_dir),
            ds.get("cachesize", cache_size),
            ds.get("cachename", cache_name),
            ds.get("cacheverbose", cache_verbose),
        )
        if "subsample" in ds:
            dataset = dataset.rsample(ds["subsample"])
        if "shuffle" in ds:
            dataset = dataset.shuffle(ds["shuffle"])
        if "epoch" in ds:
            dataset = dataset.with_epoch(ds["epoch"])
        bs = ds.get("chunksize", chunksize)
        if bs > 0:
            dataset = dataset.listed(bs)
        nworkers = ds.get("nworkers", 0)
        if nworkers >= 0:
            dataset = WebLoader(dataset, num_workers=nworkers, batch_size=None, collate_fn=list)
        p = ds.get("probability", 1.0)
        result.append(Source(dataset=dataset, probability=p))
    if len(result) > 1:
        result = RoundRobin(result)
    else:
        result = result[0].dataset
    if bs > 0:
        result = result.unlisted()
    if "epoch" in spec:
        result = result.with_epoch(spec["epoch"]).with_length(spec["epoch"])
    return result

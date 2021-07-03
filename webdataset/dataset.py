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

from . import shardcache, tariterators, utils
from .composable import Composable, Shorthands, Processor
from .utils import lookup_sym, safe_eval
from .handlers import reraise_exception
from .pytorch import IterableDataset, DataLoader
from .shardlists import PytorchShardList, ResampledShards, MultiShardSample

default_cache_dir = os.path.expanduser(os.environ.get("WEBDATASET_CACHE", ""))
default_cache_name = lookup_sym(os.environ.get("WEBDATASET_CACHE_NAME", "shard_uuid"), ".shardcache".split())
default_cache_verbose = int(safe_eval(os.environ.get("WEBDATASET_CACHE_VERBOSE", "1")))
default_cache_size = int(float(safe_eval(os.environ.get("WEBDATASET_CACHE_SIZE", "1e15"))))


@dataclass
class Source:
    """Class representing a data source."""

    dataset: IterableDataset
    probability: float = 1.0
    source: iter = None


class RoundRobin(IterableDataset, Composable, Shorthands):
    """Iterate through datasets in a round-robin way."""

    def __init__(self, sources):
        """Initialize from a set of sources."""
        super().__init__()
        self.sources = sources

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
    """Construct a composite dataset from multiple sources using a YAML spec."""
    with open(fname) as stream:
        spec = yaml.safe_load(stream)
    result = []
    assert set(spec.keys()).issubset(set("prefix datasets epoch".split())), list(spec.keys())
    prefix = spec.get("prefix", "")
    for ds in spec["datasets"]:
        assert set(ds.keys()).issubset(set("name buckets shards resampled epoch_shuffle shuffle split_by_worker split_by_node cachedir cachesize cachename cacheverbose subsample shuffle epoch chunksize nworkers probability".split())), list(ds.keys())
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


def WebDataset(
    urls,
    cache_dir=default_cache_dir,
    cache_size=default_cache_size,
    cache_name=default_cache_name,
    cache_verbose=default_cache_verbose,
    handler=reraise_exception,
    repeat=False,
):
    """Return a pipeline for WebDataset-style data files.

    This is a convenience function for constructing a partial pipeline
    that reads from a set of sharded tar files, extracts the individual
    files, and groups them together into samples (dictionaries).

    You can use all the methods from `Composable` (`then`, `compose`) and
    from `Shorthands` (`batched`, `unbatched`, `decode`, `shuffle`, etc.)
    on the result.

    The recommended way of specifying novel ways of splitting shards is
    via writing a new shardlist class.

    :param urls: the source URLs: a string, a list, or an IterableDataset
    :param handler: an error handler
    :param cache_dir: when set, caches shards in this directory
    :param cache_size: when set, specifies a maximum size for the shard cache
    :param cache_name: when set, specifies how shards should be named in the cache
    :param cache_verbose: when set, prints information about caching
    :param repeat: repeat infinitely if True
    """
    if isinstance(urls, str) and urls.endswith(".ds.yml"):
        return construct_dataset(
            urls,
            cache_dir=cache_dir,
            cache_size=cache_size,
            cache_name=cache_name,
            cache_verbose=cache_verbose,
            handler=handler,
            repeat=repeat,
        )
    if isinstance(urls, str):
        if urls.endswith(".shards.yml"):
            urls = MultiShardSample(urls)
        result = PytorchShardList(urls)
    elif isinstance(urls, list):
        result = PytorchShardList(urls)
    elif isinstance(urls, str) and os.path.splitext(urls)[1] in ["yml", "yaml", "json"]:
        raise ValueError("bad shard spec (only '.shards.yml' supported right now)")
    elif isinstance(urls, Composable):
        result = urls
    elif isinstance(urls, IterableDataset):
        result = urls
    else:
        return ValueError(f"{type(urls)}: unknown shard list type")
    result = result.then(tariterators.url_opener, handler=handler)
    if cache_dir != "":
        result = result.then(
            shardcache.cache_shards,
            cache_dir=cache_dir,
            cache_size=cache_size,
            cache_name=cache_name,
            verbose=cache_verbose,
        )
    result = result.then(tariterators.tar_file_expander, handler=handler)
    result = result.then(tariterators.group_by_keys)
    if repeat:
        result = result.repeat()
    return result


def WebLoader(*args, **kw):
    """Return a small wrapper around torch.utils.data.DataLoader.

    This wrapper works identically to the original `DataLoader`, but adds
    alls the convenience functions and filters for WebDataset.

    You can use all the methods from `Composable` (`then`, `compose`) and
    from `Shorthands` (`batched`, `unbatched`, `decode`, `shuffle`, etc.)
    on the result.

    :param args: forwarded to `DataLoader`
    :param kw: forwarded to `DataLoader`
    """
    return Processor(DataLoader(*args, **kw), utils.identity)

#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Train PyTorch models directly from POSIX tar archive.

Code works locally or over HTTP connections.
"""

import os

from . import shardcache, tariterators, utils
from .composable import Composable, Processor
from .utils import lookup_sym, safe_eval
from .handlers import reraise_exception
from .pytorch import IterableDataset, DataLoader
from .shardlists import PytorchShardList, MultiShardSample, ResampledShards
from .dsspecs import construct_dataset

default_cache_dir = os.path.expanduser(os.environ.get("WEBDATASET_CACHE", ""))
default_cache_name = lookup_sym(os.environ.get("WEBDATASET_CACHE_NAME", "shard_uuid"), ".shardcache".split())
default_cache_verbose = int(safe_eval(os.environ.get("WEBDATASET_CACHE_VERBOSE", "1")))
default_cache_size = int(float(safe_eval(os.environ.get("WEBDATASET_CACHE_SIZE", "1e15"))))


def WebDataset(
    urls,
    cache_dir=default_cache_dir,
    cache_size=default_cache_size,
    cache_name=default_cache_name,
    cache_verbose=default_cache_verbose,
    handler=reraise_exception,
    resampled=False,
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
    :param resampled: use shard resampling
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
    if resampled:
        result = ResampledShards(urls)
    elif isinstance(urls, str):
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

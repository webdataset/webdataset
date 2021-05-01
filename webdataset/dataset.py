#!/usr/bin/python
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Train PyTorch models directly from POSIX tar archive, locally
or over HTTP connections.
"""

__all__ = """Dataset tariterator default_handlers imagehandler
reraise_exception ignore_and_continue warn_and_continue ignore_and_stop warn_and_stop
""".split()

import os
import random
import warnings
import itertools as itt

import braceexpand

from . import tariterators
from . import iterators
from . import autodecode
from . import shardcache
from . import dbcache
from . import utils
from . import gopen
from .utils import reraise_exception, lookup_sym, safe_eval


try:
    from torch.utils.data import IterableDataset, DataLoader
except:
    from .mock import IterableDataset, DataLoader


default_cache_dir = os.path.expanduser(os.environ.get("WEBDATASET_CACHE", ""))
default_cache_name = lookup_sym(
    os.environ.get("WEBDATASET_CACHE_NAME", "shard_uuid"), ".shardcache".split()
)
default_cache_verbose = int(safe_eval(os.environ.get("WEBDATASET_CACHE_VERBOSE", "1")))
default_cache_size = int(
    float(safe_eval(os.environ.get("WEBDATASET_CACHE_SIZE", "1e15")))
)


class Composable:
    def __init__(self):
        super().__init__()

    def source_(self, source):
        self.source = source
        return self

    def then(self, f, *args, length=True, **kw):
        """Compose this processor with a new processor defined by a function.

        The function is of the form:

            def my_process(source, ...):
                for sample in source:
                    ...
                    result = ...
                    yield result
        """
        assert callable(f)
        assert "source" not in kw
        # print("Processor", args, kw)
        return Processor(self, f, *args, **kw)

    def compose(self, constructor, *args, **kw):
        """Compose this processor with another IterableDataset.

        The constructor should be of the form `__init__(self, source_dataset, ...)`
        """
        assert callable(constructor)
        return constructor(*args, **kw).source_(self)


class SplitByNode:

    def __init__(self, group=None):
        self.rank = -1
        self.size = -1
        try:
            import torch
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                return
        except Exception as e:
            print(e)
            return
        if group is None:
            # group = torch.distributed.group.WORLD
            try:
                # some versions of torch don't like group=None
                import torch.distributed.distributed_c10d
                group = torch.distributed.distributed_c10d._default_pg
            except:
                pass
        self.rank = torch.distributed.get_rank(group=group)
        self.size = torch.distributed.get_world_size(group=group)

    def __call__(self, urls):
        urls = [url for url in urls]
        assert isinstance(urls, list)
        if self.size > 1:
            import socket
            gopen.info["rank"] = self.rank
            gopen.info["size"] = self.size
            gopen.info["host"] = socket.gethostname()
            gopen.info["pid"] = os.getpid()
            if self.rank == 0 and len(urls) < self.size:
                warnings.warn(f"world_size {self.size} > num_shards {len(urls)}")
            return urls[self.rank::self.size]
        else:
            return urls


def split_by_worker(urls):
    """Selects a subset of urls based on Torch get_worker_info.

    Used as a shard selection function in Dataset."""
    import torch

    urls = [url for url in urls]

    assert isinstance(urls, list)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        wid = worker_info.id
        num_workers = worker_info.num_workers
        gopen.info["worker_id"] = wid
        gopen.info["num_workers"] = num_workers
        if wid == 0 and len(urls) < num_workers:
            warnings.warn(f"num_workers {num_workers} > num_shards {len(urls)}")
        return urls[wid::num_workers]
    else:
        return urls


class ShardList(IterableDataset, Composable):
    def __init__(
        self,
        urls,
        shuffle=False,
        nodesplitter=True,
        splitter=split_by_worker,
        length=None,
    ):
        super().__init__()
        self.shuffle = shuffle
        self.length = length
        if nodesplitter is True:
            nodesplitter = SplitByNode()
        self.nodesplitter = nodesplitter
        self.splitter = splitter
        if isinstance(urls, str):
            urls = list(braceexpand.braceexpand(urls))
        else:
            urls = list(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)

    def __iter__(self):
        urls = list(self.urls)
        if self.nodesplitter is not None:
            urls = list(self.nodesplitter(urls))
        if self.splitter is not None:
            urls = list(self.splitter(urls))
        if callable(self.shuffle):
            self.shuffle(urls)
        elif self.shuffle:
            random.shuffle(urls)
        for url in urls:
            yield dict(url=url)

    def __len__(self):
        if self.length is None:
            raise ValueError(
                "length requested, but no length specified for ShardIterator"
            )
        return self.length


class Shorthands:
    def __init__(self):
        super().__init__()

    def batched(self, batchsize, collation_fn=iterators.default_collation_fn, partial=True):
        return self.then(iterators.batched, batchsize=batchsize, collation_fn=collation_fn, partial=partial)

    def unbatched(self):
        return self.then(iterators.unbatched)

    def shuffle(self, size, **kw):
        if size < 1:
            return self
        return self.then(iterators.shuffle, size, **kw)

    def map(self, f, handler=reraise_exception):
        return self.then(iterators.map, f, handler=handler)

    def decode(
        self,
        *args,
        pre=None,
        post=None,
        only=None,
        handler=reraise_exception,
    ):
        # for backwards compatibility
        handlers = [
            autodecode.ImageHandler(h) if isinstance(h, str) else h for h in args
        ]
        decoder = autodecode.Decoder(handlers, pre=pre, post=post, only=only)
        return self.map(decoder, handler=handler)

    def rename(self, handler=reraise_exception, **kw):
        return self.then(iterators.rename, handler=handler, _kwa=kw)

    def map_dict(self, handler=reraise_exception, **kw):
        return self.then(iterators.map_dict, handler=handler, _kwa=kw)

    def select(self, predicate, **kw):
        return self.then(iterators.select, predicate, _kwa=kw)

    def to_tuple(self, *args, handler=reraise_exception):
        return self.then(iterators.to_tuple, *args, handler=handler)

    def map_tuple(self, *args, handler=reraise_exception):
        return self.then(iterators.map_tuple, *args, handler=handler)

    def pipe(self, f, *args, **kw):
        return self.then(f, *args, _kwa=kw)

    def dbcache(self, fname, size):
        return self.compose(dbcache.DBCache, fname, size)

    def slice(self, *args):
        return self.pipe(itt.islice, *args)

    def repeat(
        self,
        nepochs=None,
        nbatches=None,
        nsamples=None,
        batchsize=utils.guess_batchsize,
    ):
        return self.compose(
            Repeatedly,
            nepochs=nepochs,
            nbatches=nbatches,
            nsamples=nsamples,
            batchsize=batchsize,
        )


class Repeatedly(IterableDataset, Composable, Shorthands):
    def __init__(self, **kw):
        self.kw = kw

    def __iter__(self):
        return utils.repeatedly(self.source, **self.kw)

    def __len__(self):
        return len(self.source)


class Processor(IterableDataset, Composable, Shorthands):
    def __init__(self, source, f, *args, _kwa={}, length=True, **kw):
        super().__init__()
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = dict(_kwa)
        self.kw.update(kw)
        self.length = length

    def source_(self, source):
        self.source = source
        return self

    def __iter__(self):
        assert (
            self.source is not None
        ), f"must set source before calling iter {self.f} {self.args} {self.kw}"
        assert callable(self.f), self.f
        return self.f(iter(self.source), *self.args, **self.kw)

    def __len__(self):
        if self.length is True:
            return len(self.source)
        elif isinstance(self.length, int):
            return self.length
        elif callable(self.length):
            return self.length(self.source)
        else:
            raise ValueError(f"{self.length}: not a valid length specification")


def WebDataset(
    urls,
    shardshuffle=True,
    cache_dir=default_cache_dir,
    cache_size=default_cache_size,
    cache_name=default_cache_name,
    cache_verbose=default_cache_verbose,
    splitter=split_by_worker,
    nodesplitter=True,
    handler=reraise_exception,
    length=None,
):
    result = ShardList(
        urls,
        shuffle=shardshuffle,
        splitter=splitter,
        nodesplitter=nodesplitter,
        length=length,
    )
    result = result.then(tariterators.url_opener, handler=handler)
    if cache_dir != "":
        result = result.then(
            shardcache.cache_shards,
            cache_dir=cache_dir,
            cache_size=cache_size,
            cache_name=cache_name,
            verbose=cache_verbose,
        )
    result = result.then(tariterators.tar_file_expander, length=None, handler=handler)
    result = result.then(tariterators.group_by_keys, length=length)
    return result


def WebLoader(*args, **kw):
    return Processor(DataLoader(*args, **kw), utils.identity)


class ResizedDataset(IterableDataset):
    """Change the actual and nominal length of an IterableDataset.

    :param dataset: IterableDataset
    :param length: declared length of the dataset
    :param nominal: nominal length of dataset (if different from declared)

    This will continuously iterate through the original dataset, but
    impose new epoch boundaries at the given length/nominal.
    This exists mainly as a workaround for the odd logic in DataLoader.
    It is also useful for choosing smaller nominal epoch sizes with
    very large datasets.

    """

    def __init__(self, dataset, length=None, nominal=None):
        super().__init__()
        self.dataset = dataset
        if length is None:
            length = len(dataset)
        self.length = length
        self.nominal = self.length if nominal is None else nominal
        self.source = None

    def __len__(self):
        return self.nominal

    def __getstate__(self):
        result = dict(self.__dict__)
        result["source"] = None
        return result

    def __iter__(self):
        if self.source is None:
            self.source = iter(self.dataset)
        for i in range(self.length):
            try:
                sample = next(self.source)
            except StopIteration:
                self.source = iter(self.dataset)
                sample = next(self.source)
            yield sample


ChoppedDataset = ResizedDataset

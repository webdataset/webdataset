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

import random
import warnings

import braceexpand
from torch.utils.data import IterableDataset

from . import tariterators
from . import iterators
from . import autodecode
from . import shardcache
from .utils import reraise_exception


class Composable:

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
        return Processor(f, *args, length=length, source=self, **kw)

    def compose(self, constructor, *args, **kw):
        """Compose this processor with another IterableDataset.

        The constructor should be of the form `__init__(self, source_dataset, ...)`
        """
        assert callable(constructor)
        return constructor(*args, **kw)(self)


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
        if wid == 0 and len(urls) < num_workers:
            warnings.warn(f"num_workers {num_workers} > num_shards {len(urls)}")
        return urls[wid::num_workers]
    else:
        return urls


class ShardList(IterableDataset, Composable):
    def __init__(self, urls, shuffle=False, splitter=split_by_worker):
        self.shuffle = shuffle
        if isinstance(urls, str):
            urls = list(braceexpand.braceexpand(urls))
        else:
            urls = list(urls)
        if splitter is not None:
            self.urls = list(splitter(urls))
        else:
            self.urls = urls
        assert isinstance(self.urls[0], str)

    def __iter__(self):
        urls = list(self.urls)
        if self.shuffle:
            random.shuffle(urls)
        for url in urls:
            yield dict(url=url)


class Shorthands:
    def batched(self, batchsize, partial=True):
        return self.then(iterators.batched, batchsize=batchsize, partial=partial)

    def unbatched(self):
        return self.then(iterators.unbatched)

    def shuffle(self, size, **kw):
        return self.then(iterators.shuffle, size, **kw)

    def map(self, f, handler=reraise_exception):
        return self.then(iterators.map, f, handler=handler)

    def decode(
        self,
        *args,
        pre=None,
        post=None,
        handler=reraise_exception,
    ):
        # for backwards compatibility
        handlers = [
            autodecode.ImageHandler(h) if isinstance(h, str) else h for h in args
        ]
        decoder = autodecode.Decoder(handlers, pre=pre, post=post)
        return self.map(decoder, handler=handler)

    def rename(self, handler=reraise_exception, **kw):
        return self.then(iterators.rename, handler=handler, **kw)

    def map_dict(self, handler=reraise_exception, **kw):
        return self.then(iterators.map_dict, handler=handler, **kw)

    def select(self, predicate, **kw):
        return self.then(iterators.select, predicate, **kw)

    def to_tuple(self, *args, handler=reraise_exception):
        return self.then(iterators.to_tuple, *args, handler=handler)

    def map_tuple(self, *args, handler=reraise_exception):
        return self.then(iterators.map_tuple, *args, handler=handler)

    def pipe(self, f, *args, **kw):
        return self.then(f, *args, **kw)


class Processor(IterableDataset, Composable, Shorthands):
    def __init__(self, f, *args, length=True, source=None, **kw):
        super().__init__()
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw
        self.length = length

    def on(self, source):
        self.source = source
        return self

    def __call__(self, source):
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
    cache_dir=None,
    cache_size=1e15,
    cache_name=shardcache.shard_uuid,
    cache_verbose=True,
    splitter=split_by_worker,
    handler=reraise_exception,
    length=None,
):
    result = ShardList(urls, shuffle=shardshuffle, splitter=splitter)
    result = result.then(tariterators.url_opener, handler=handler)
    if cache_dir is not None:
        result = result.then(shardcache.cache_shards, cache_dir=cache_dir, cache_size=cache_size, cache_name=cache_name, verbose=cache_verbose)
    result = result.then(tariterators.tar_file_expander, length=None, handler=handler)
    result = result.then(tariterators.group_by_keys, length=length)
    return result


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

#!/usr/bin/python
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""A fluid interface for constructing datasets.
"""

__all__ = """FluidPipes Dataset""".split()

import random

import braceexpand
from torch.utils.data import IterableDataset

from . import autodecode
from . import iterators
from . import tariterators
from . import dataset
from .utils import reraise_exception, ignore_and_continue, ignore_and_stop, warn_and_continue

class FluidPipes(IterableDataset):
    def __init__(self, initial):
        super().__init__()
        self.dataset = initial

    def __len__(self):
        """Return the nominal length of the dataset."""
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataset)

    def pipe(self, f, *args, **kw):
        self.dataset = self.dataset.then(f, *args, **kw)
        return self

    def batched(self, batchsize, partial=True):
        self.dataset = self.dataset.then(
            iterators.batched, batchsize=batchsize, partial=partial
        )
        return self

    def unbatched(self):
        self.dataset = self.dataset.then(iterators.unbatched)
        return self

    def shuffle(self, size, **kw):
        self.dataset = self.dataset.then(iterators.shuffle, size, **kw)
        return self

    def map(self, f, handler=reraise_exception):
        self.dataset = self.dataset.then(iterators.map, f, handler=handler)
        return self

    def decode(
        self,
        *args,
        pre=[autodecode.gzfilter],
        post=[autodecode.basichandlers],
        handler=reraise_exception,
    ):
        handlers = [
            autodecode.ImageHandler(h) if isinstance(h, str) else h for h in args
        ]
        decoder = autodecode.Decoder(pre + handlers + post)
        return self.map(decoder, handler=handler)

    def rename(self, handler=reraise_exception, **kw):
        self.dataset = self.dataset.then(iterators.rename, handler=handler, **kw)
        return self

    def map_dict(self, handler=reraise_exception, **kw):
        self.dataset = self.dataset.then(iterators.map_dict, handler=handler, **kw)
        return self

    def select(self, predicate, **kw):
        self.dataset = self.dataset.then(iterators.select, predicate, **kw)
        return self

    def to_tuple(self, *args, handler=reraise_exception):
        self.dataset = self.dataset.then(iterators.to_tuple, *args, handler=handler)
        return self

    def map_tuple(self, *args, handler=reraise_exception):
        self.dataset = self.dataset.then(iterators.map_tuple, *args, handler=handler)
        return self


class Dataset(FluidPipes):
    def __init__(
        self,
        urls,
        *,
        length=True,
        splitter=tariterators.split_by_worker,
        handler=reraise_exception,
        shardshuffle=False,
    ):
        shardlist = dataset.ShardList(urls, shuffle=shardshuffle)
        FluidPipes.__init__(self, shardlist)
        self.shardlist = shardlist
        self.pipe(splitter)
        self.pipe(tariterators.url_opener, handler=handler)
        self.pipe(tariterators.tar_file_expander, length=None, handler=handler)
        self.pipe(tariterators.group_by_keys, length=length)

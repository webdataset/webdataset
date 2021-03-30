#!/usr/bin/python
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""A fluid interface for constructing datasets.
"""

__all__ = """FluidPipes Dataset""".split()

from torch.utils.data import IterableDataset

from . import autodecode
from . import iterators
from . import tariterators
from .dataset import (
    WebDataset,
    split_by_worker,
    default_cache_dir,
    default_cache_name,
    default_cache_size,
    default_cache_verbose
)
from .utils import reraise_exception


class Dataset(IterableDataset):
    def __init__(
        self,
        urls,
        *,
        length=True,
        splitter=split_by_worker,
        handler=reraise_exception,
        shuffle=False,
        cache_dir=default_cache_dir,
        cache_size=default_cache_size,
        cache_name=default_cache_name,
        cache_verbose=default_cache_verbose
    ):
        super().__init__()
        self._dataset = WebDataset(
            urls,
            shardshuffle=shuffle,
            splitter=splitter,
            handler=handler,
            length=length,
            cache_dir=cache_dir,
            cache_size=cache_size,
            cache_name=cache_name,
            cache_verbose=cache_verbose
        )

    def __getattr__(self, name):
        if not hasattr(self._dataset, name):
            raise AttributeError()
        def f(*args, **kw):
            self._dataset = getattr(self._dataset, name)(*args, **kw)
            return self
        return f

    def __iter__(self):
        return iter(self._dataset)

    def __len__(self):
        return len(self._dataset)

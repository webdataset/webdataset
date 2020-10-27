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
from . import dataset
from .utils import reraise_exception



class Dataset(IterableDataset):
    def __init__(
        self,
        urls,
        *,
        length=True,
        splitter=dataset.split_by_worker,
        handler=reraise_exception,
        shuffle=False,
    ):
        super().__init__()
        self.dataset = dataset.WebDataset(
            urls,
            shardshuffle=shuffle,
            splitter=splitter,
            handler=handler,
            length=length,
        )

    def __getattr__(self, name):
        if not hasattr(self.dataset, name):
            raise AttributeError()
        def f(*args, **kw):
            self.dataset = getattr(self.dataset, name)(*args, **kw)
            return self
        return f

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)

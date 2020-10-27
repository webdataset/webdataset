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

import braceexpand
from torch.utils.data import IterableDataset

from . import tariterators
from .utils import reraise_exception


class ShardList(IterableDataset):
    def __init__(self, urls, shuffle=False):
        self.shuffle = shuffle
        if isinstance(urls, str):
            self.urls = list(braceexpand.braceexpand(urls))
        else:
            self.urls = list(urls)
        assert isinstance(self.urls[0], str)

    def __iter__(self):
        urls = list(self.urls)
        if self.shuffle:
            random.shuffle(urls)
        for url in urls:
            yield dict(url=url)

    def then(self, f, *args, length=True, **kw):
        assert callable(f)
        assert "source" not in kw
        return Processor(f, *args, length=length, source=self, **kw)


class Processor(IterableDataset):
    def __init__(self, f, *args, length=True, source=None, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw
        self.length = length

    def __call__(self, source):
        self.source = source

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
        return constructor(self, *args, **kw)


def WebDataset(
    urls,
    shardshuffle=True,
    splitter=tariterators.split_by_worker,
    handler=reraise_exception,
    length=None,
):
    return (
        ShardList(urls, shuffle=shardshuffle)
        .then(splitter)
        .then(tariterators.url_opener, handler=handler)
        .then(tariterators.tar_file_expander, length=None, handler=handler)
        .then(tariterators.group_by_keys, length=length)
    )


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

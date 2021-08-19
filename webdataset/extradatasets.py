#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Train PyTorch models directly from POSIX tar archive.

Code works locally or over HTTP connections.
"""

import itertools as itt
import os
import sys
import random

import braceexpand

from .pytorch import IterableDataset
from .composable import Composable, Shorthands
from . import dataset
from . import utils

class MockDataset(IterableDataset, Composable, Shorthands):
    """MockDataset.

    A mock dataset for performance testing and unit testing.
    """

    def __init__(self, sample, length):
        """Create a mock dataset instance.

        :param sample: the sample to be returned repeatedly
        :param length: the length of the mock dataset
        """
        self.sample = sample
        self.length = length

    def __iter__(self):
        """Return an iterator over this mock dataset."""
        for i in range(self.length):
            yield self.sample


class Repeatedly(IterableDataset, Composable, Shorthands):
    """Repeatedly yield samples from a dataset."""

    def __init__(self, nepochs=None, nbatches=None, length=None):
        """Create an instance of Repeatedly.

        :param nepochs: repeat for a maximum of nepochs
        :param nbatches: repeat for a maximum of nbatches
        """
        self.length = length
        self.nepochs = nepochs
        self.nbatches = nbatches

    def __iter__(self):
        """Return an iterator that iterates repeatedly over a source."""
        return utils.repeatedly(
            self.source,
            nepochs=self.nepochs,
            nbatches=self.nbatches,
        )


class DatasetTest(IterableDataset, Composable, Shorthands):
    """Perform final checks on an IterableDataset and permit easy mock tests.

    This is the implementation of the `dataset.Shorthands.test` method; you usually
    do not need to construct it explicitly.
    """

    def __init__(self, length=None, checker=None, mock_sample=None, mock_length=10000, mock=False):
        """Create a DatasetTest.

        :param length: length of the dataset
        :param checker: any kind of final checking function you want to run over samples
        :param mock_sample: mock sample
        :param mock_length: size of mocked dataset
        :param mock: turning mocking on/off
        """
        super().__init__()
        self.source = None
        self.length = length
        self.checker = checker
        self.mock = mock
        self.mock_length = mock_length
        self.mock_sample = mock_sample


    def __iter__(self):
        """Return an iterator either over the mock object or the underlying dataset."""
        if self.mock:
            if not callable(self.mock_sample):
                for i in range(self.mock_length):
                    yield self.mock_sample
            else:
                return self.mock_sample()
        else:
            for sample in self.source:
                if self.checker is not None:
                    self.checker(sample)
                yield sample


class ChoppedDataset(IterableDataset, Composable, Shorthands):
    """Change the actual and nominal length of an IterableDataset.

    This will continuously iterate through the original dataset, but
    impose new epoch boundaries at the given length/nominal.
    This exists mainly as a workaround for the odd logic in DataLoader.
    It is also useful for choosing smaller nominal epoch sizes with
    very large datasets.

    """

    def __init__(self, dataset, length=None, nominal=None):
        """Create a ChoppedDataset.

        :param dataset: IterableDataset
        :param length: declared length of the dataset
        :param nominal: nominal length of dataset (if different from declared)
        """
        super().__init__()
        self.dataset = dataset
        if length is None:
            length = len(dataset)
        self.length = length
        self.nominal = self.length if nominal is None else nominal
        self.source = None

    def __getstate__(self):
        """Return the pickled state of the dataset.

        This resets the dataset iterator, since that can't be pickled.
        """
        result = dict(self.__dict__)
        result["source"] = None
        return result

    def __iter__(self):
        """Return an iterator over the dataset.

        This iterator returns as many samples as given by the `length` parameter.
        """
        if self.source is None:
            self.source = iter(self.dataset)
        for i in range(self.length):
            try:
                sample = next(self.source)
            except StopIteration:
                self.source = iter(self.dataset)
                try:
                    sample = next(self.source)
                except StopIteration:
                    return
            yield sample


class FakeLength(IterableDataset, Composable, Shorthands):
    """Repeatedly yield samples from a dataset."""

    def __init__(self, dataset, length):
        """Create an instance of Repeatedly.

        :param dataset: source dataset
        :param length: stated length
        """
        super().__init__()
        self.dataset = dataset
        self.length = length

    def __iter__(self):
        """Return an iterator that iterates repeatedly over a source."""
        return iter(self.dataset)

    def __len__(self):
        """Return the user specified length."""
        return self.length

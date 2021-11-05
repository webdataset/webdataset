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


class with_epoch(IterableDataset, Composable, Shorthands):
    """Change the actual and nominal length of an IterableDataset.

    This will continuously iterate through the original dataset, but
    impose new epoch boundaries at the given length/nominal.
    This exists mainly as a workaround for the odd logic in DataLoader.
    It is also useful for choosing smaller nominal epoch sizes with
    very large datasets.

    """

    def __init__(self, dataset, length):
        """Chop the dataset to the given length.

        :param dataset: IterableDataset
        :param length: declared length of the dataset
        :param nominal: nominal length of dataset (if different from declared)
        """
        super().__init__()
        self.dataset = dataset
        self.length = length
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


class with_length(IterableDataset, Composable, Shorthands):
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

#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""Classes for mixing samples from multiple sources."""

import random

import numpy as np

from .pytorch import IterableDataset


def round_robin_shortest(*sources):
    """Yield samples from multiple sources in a round-robin fashion until the shortest source is exhausted.

    Args:
        *sources: Iterable sources to draw samples from.

    Yields:
        Sample from one of the sources.
    """
    i = 0
    while True:
        try:
            sample = next(sources[i % len(sources)])
            yield sample
        except StopIteration:
            break
        i += 1


def round_robin_longest(*sources):
    """Yield samples from multiple sources in a round-robin fashion until all sources are exhausted.

    Args:
        *sources: Iterable sources to draw samples from.

    Yields:
        Sample from one of the sources.
    """
    sources = list(sources)
    i = 0
    while len(sources) > 0:
        i %= len(sources)
        try:
            sample = next(sources[i])
            i += 1
            yield sample
        except StopIteration:
            del sources[i]


class RoundRobin(IterableDataset):
    """Iterate over multiple datasets in a round-robin fashion."""

    def __init__(self, datasets, longest=False):
        """Initialize the RoundRobin iterator.

        Args:
            datasets (list): List of datasets to iterate over.
            longest (bool): If True, continue until the longest dataset is exhausted.
        """
        self.datasets = datasets
        self.longest = longest

    def __iter__(self):
        """Return an iterator over the sources.

        Returns:
            iterator: An iterator that yields samples from the datasets in a round-robin fashion.
        """
        sources = [iter(d) for d in self.datasets]
        if self.longest:
            return round_robin_longest(*sources)
        else:
            return round_robin_shortest(*sources)


def random_samples(sources, probs=None, longest=False):
    """Yield samples randomly from multiple sources based on given probabilities.

    Args:
        sources (list): List of iterable sources to draw samples from.
        probs (list, optional): List of probabilities for each source. Defaults to None.
        longest (bool): If True, continue until all sources are exhausted. Defaults to False.

    Yields:
        Sample randomly selected from one of the sources.
    """
    if probs is None:
        probs = [1] * len(sources)
    else:
        probs = list(probs)
    while len(sources) > 0:
        cum = (np.array(probs) / np.sum(probs)).cumsum()
        r = random.random()
        i = np.searchsorted(cum, r)
        try:
            yield next(sources[i])
        except StopIteration:
            if longest:
                del sources[i]
                del probs[i]
            else:
                break


class RandomMix(IterableDataset):
    """Iterate over multiple datasets by randomly selecting samples based on given probabilities."""

    def __init__(self, datasets, probs=None, longest=False):
        """Initialize the RandomMix iterator.

        Args:
            datasets (list): List of datasets to iterate over.
            probs (list, optional): List of probabilities for each dataset. Defaults to None.
            longest (bool): If True, continue until all datasets are exhausted. Defaults to False.
        """
        self.datasets = datasets
        self.probs = probs
        self.longest = longest

    def __iter__(self):
        """Return an iterator over the sources.

        Returns:
            iterator: An iterator that yields samples randomly from the datasets.
        """
        sources = [iter(d) for d in self.datasets]
        return random_samples(sources, self.probs, longest=self.longest)

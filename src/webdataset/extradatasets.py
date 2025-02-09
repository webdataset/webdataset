#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Train PyTorch models directly from POSIX tar archive.

Code works locally or over HTTP connections.
"""


from . import utils
from .pytorch import IterableDataset
from .utils import PipelineStage


class MockDataset(IterableDataset):
    """Create a mock dataset for performance testing and unit testing.

    Args:
        sample: The sample to be returned repeatedly.
        length (int): The length of the mock dataset.
    """

    def __init__(self, sample, length):
        self.sample = sample
        self.length = length

    def __iter__(self):
        """Yield samples from the mock dataset.

        Returns:
            Iterator: An iterator that yields the same sample repeatedly.
        """
        for _ in range(self.length):
            yield self.sample


class repeatedly(IterableDataset, PipelineStage):
    """Repeatedly yield samples from a dataset.

    Args:
        source: The source dataset to repeat.
        nepochs (int, optional): Maximum number of epochs to repeat.
        nbatches (int, optional): Maximum number of batches to repeat.
        length (int, optional): Length of the repeated dataset.
    """

    def __init__(self, source, nepochs=None, nbatches=None, length=None):
        self.source = source
        self.length = length
        self.nbatches = nbatches

    def invoke(self, source):
        """Return an iterator that iterates repeatedly over a source.

        Args:
            source: The source dataset to repeat.

        Returns:
            Iterator: An iterator that repeatedly yields samples from the source.
        """
        return utils.repeatedly(
            source,
            nepochs=self.nepochs,
            nbatches=self.nbatches,
        )


class with_epoch(IterableDataset):
    """Change the actual and nominal length of an IterableDataset.

    This will continuously iterate through the original dataset, but
    impose new epoch boundaries at the given length/nominal.
    This exists mainly as a workaround for the odd logic in DataLoader.
    It is also useful for choosing smaller nominal epoch sizes with
    very large datasets.

    Args:
        dataset: The source IterableDataset.
        length (int): Declared length of the dataset.
    """

    def __init__(self, dataset, length):
        super().__init__()
        self.length = length
        self.source = None

    def __getstate__(self):
        """Return the pickled state of the dataset.

        This resets the dataset iterator, since that can't be pickled.

        Returns:
            dict: A dictionary representing the pickled state of the dataset.
        """
        result = dict(self.__dict__)
        result["source"] = None
        return result

    def invoke(self, dataset):
        """Return an iterator over the dataset.

        This iterator returns as many samples as given by the `length` parameter.

        Args:
            dataset: The source dataset to iterate over.

        Yields:
            Sample: The next sample from the dataset.
        """
        if self.source is None:
            self.source = iter(dataset)
        for _ in range(self.length):
            try:
                sample = next(self.source)
            except StopIteration:
                self.source = iter(dataset)
                try:
                    sample = next(self.source)
                except StopIteration:
                    return
            yield sample
        self.source = None


class with_length(IterableDataset, PipelineStage):
    """Repeatedly yield samples from a dataset with a specified length.

    Args:
        dataset: The source dataset.
        length (int): The stated length of the dataset.
    """

    def __init__(self, dataset, length):
        super().__init__()
        self.dataset = dataset
        self.length = length

    def invoke(self, dataset):
        """Return an iterator that iterates over the source dataset.

        Args:
            dataset: The source dataset to iterate over.

        Returns:
            Iterator: An iterator over the source dataset.
        """
        return iter(dataset)

    def __len__(self):
        """Return the user specified length.

        Returns:
            int: The specified length of the dataset.
        """
        return self.length

import copy
import sys
import warnings
from itertools import islice

from .pytorch import DataLoader, IterableDataset
from .utils import PipelineStage


def add_length_method(obj):
    """Add a length method to the given object.

    Args:
        obj: The object to which the length method will be added.

    Returns:
        The modified object with a new length method.
    """

    def length(self):
        return self.size

    Combined = type(
        obj.__class__.__name__ + "_Length",
        (obj.__class__, IterableDataset),
        {"__len__": length},
    )
    obj.__class__ = Combined
    return obj


class DataPipeline(IterableDataset, PipelineStage):
    """A pipeline starting with an IterableDataset and a series of filters.

    Args:
        *args: Variable length argument list of pipeline stages.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pipeline = []
        self.length = -1
        self.repetitions = 1
        self.nsamples = -1
        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, list):
                self.pipeline.extend(arg)
            else:
                self.pipeline.append(arg)

    def close(self):
        """Close the pipeline and release resources."""
        for step in self.pipeline:
            if hasattr(step, "close"):
                step.close()
        del self.pipeline

    def invoke(self, f, *args, **kwargs):
        """Apply a pipeline stage, possibly to the output of a previous stage.

        Args:
            f: The pipeline stage to invoke.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The result of invoking the pipeline stage.

        Raises:
            ValueError: If the pipeline stage is not valid.
        """
        if isinstance(f, (IterableDataset, DataLoader)) and len(args) == 0:
            return iter(f)
        if isinstance(f, PipelineStage):
            return f.run(*args, **kwargs)
        if isinstance(f, list):
            return iter(f)
        if callable(f):
            result = f(*args, **kwargs)
            return result
        raise ValueError(f"{f}: not a valid pipeline stage")

    def iterator1(self):
        """Create an iterator through one epoch in the pipeline.

        Returns:
            An iterator for one epoch of the pipeline.
        """
        source = self.invoke(self.pipeline[0])
        for step in self.pipeline[1:]:
            source = self.invoke(step, source)
        return source

    def iterator(self):
        """Create an iterator through the entire dataset, using the given number of repetitions.

        Yields:
            Samples from the dataset.
        """
        for _ in range(self.repetitions):
            count = 0
            for sample in self.iterator1():
                yield sample
                count += 1
            if count == 0:
                # if the dataset is empty, don't keep looping
                break

    def __iter__(self):
        """Create an iterator through the pipeline, repeating and slicing as requested.

        Returns:
            An iterator through the pipeline.
        """
        if self.repetitions != 1:
            if self.nsamples > 0:
                return islice(self.iterator(), self.nsamples)
            else:
                return self.iterator()
        else:
            return self.iterator()

    def stage(self, i):
        """Return pipeline stage i.

        Args:
            i: The index of the pipeline stage to return.

        Returns:
            The pipeline stage at index i.
        """
        return self.pipeline[i]

    def append(self, f):
        """Append a pipeline stage (modifies the object).

        Args:
            f: The pipeline stage to append.
        """
        self.pipeline.append(f)

    def compose(self, *args):
        """Append pipeline stages to a copy of the pipeline and return the copy.

        Args:
            *args: Variable length argument list of pipeline stages to append.

        Returns:
            A new DataPipeline object with the appended stages.
        """
        result = copy.copy(self)
        result.pipeline = copy.copy(result.pipeline)
        for arg in args:
            result.append(arg)
        return result

    def with_length(self, n, silent=False):
        """Add a __len__ method returning the desired value.

        This does not change the actual number of samples in an epoch.
        PyTorch IterableDataset should not have a __len__ method.
        This is provided only as a workaround for some broken training environments
        that require a __len__ method.

        Args:
            n: The length value to set.
            silent: If True, suppress the warning message.

        Returns:
            The modified DataPipeline object with a __len__ method.
        """
        if not silent:
            warnings.warn(
                ".with_length() only sets the value of __len__ for compatibility with some training environments. It does not change the number of samples in an epoch."
            )
        self.size = n
        return add_length_method(self)

    def with_epoch(self, nsamples=-1, nbatches=-1):
        """Change the epoch to return the given number of samples/batches.

        Args:
            nsamples: The number of samples per epoch.
            nbatches: The number of batches per epoch.

        Returns:
            The modified DataPipeline object.
        """
        self.repetitions = sys.maxsize
        self.nsamples = max(nsamples, nbatches)
        return self

    def repeat(self, nepochs=-1, nbatches=-1):
        """Repeat iterating through the dataset for the given number of epochs up to the given number of samples.

        Args:
            nepochs: The number of epochs to repeat.
            nbatches: The number of batches to limit per repetition.

        Returns:
            The modified DataPipeline object.
        """
        if nepochs > 0:
            self.repetitions = nepochs
            self.nsamples = nbatches
        else:
            self.repetitions = sys.maxsize
            self.nsamples = nbatches
        return self

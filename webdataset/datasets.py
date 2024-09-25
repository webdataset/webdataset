import os
import random
import warnings
from types import SimpleNamespace
from urllib.parse import urlparse

import yaml

from . import autodecode, cache, filters, shardlists, utils
from .filters import pipelinefilter, reraise_exception
from .pipeline import DataPipeline
from .pytorch import DataLoader
from .tariterators import group_by_keys, tar_file_expander


def run_pipeline(pipeline):
    """Run a list of iterators as a pipeline.

    This can be so much shorter than the Pipeline class
    because for these classes, the pipelines are fixed
    and only created once inside the constructor. Users
    never use it.

    Args:
        pipeline (list): List of callables returning iterators.
    """
    source = pipeline[0]
    for filter in pipeline[1:]:
        source = filter(source)
    for sample in source:
        yield sample


def set_pipeline_epochs(pipeline, epoch):
    """Set the epoch for all stages in the pipeline.

    For any stage that has a set_epoch method, call it with the epoch number.

    Args:
        pipeline (list): List of callables.
        epoch (int): Epoch number.
    """
    for stage in pipeline:
        if hasattr(stage, "set_epoch"):
            stage.set_epoch(epoch)


def apply_transformations(transformations, x):
    """Apply a list of transformations to a sample.

    Args:
        transformations (list): List of callables.
        x (dict): Sample.
    """
    if not isinstance(transformations, list):
        transformations = [transformations]
    for transformation in transformations:
        x = transformation(x)
    return x


def interpret_transformations(transformations):
    """Interpret the transformations argument.

    This takes care of transformations specified as string shortcuts
    and returns a list of callables.
    """
    if not isinstance(transformations, list):
        transformations = [transformations]

    result = []

    for transformation in transformations:
        if transformation == "PIL":
            transformation = autodecode.Decoder("PIL")
        elif transformation == "numpy":
            transformation = autodecode.Decoder("numpy")
        else:
            assert callable(transformation)
        result.append(transformation)

    return result


class IterableWebDataset(IterableDataset):
    def __init__(
        self,
        shards,
        *,
        cache_size=int(1e12),
        cache_dir=None,
        lru_size=10,
        dataset_name=None,
        localname=None,
        transformations="PIL",
        keep=False,
        base=None,
        options=None,
        select_files=None,
        rename_files=None,
    ):
        if options is None:
            options = {}
        self.total_size = -1
        self.transformations = interpret_transformations(transformations)
        source = self.create_url_iterator(shars)
        self.init_pipeline(source)
        self.epoch = -1

    def create_url_iterator(self, shards):
        """Create an iterator over the shards."""
        if isinstance(shards, str) and shard.endswith(".json"):
            shards, total_size = read_shards_from_json(shards)
            self.total_size = total_size
        if isinstance(shards, (str, list)):
            return shardlists.SimpleShardList(shards)
        raise ValueError("unknown shard list type")

    def init_pipeline(self, source):
        # FIXME: check for multinode here
        self.pipeline = [source, shardlists.split_by_worker]
        if cache_dir is None:
            self.pipeline.append(cache.StreamingOpen(handler=handler))
        else:
            self.pipeline.append(
                cache.FileCache(
                    cache_dir=cache_dir, cache_size=cache_size, handler=handler
                )
            )
        self.pipeline.append(
            partial(
                tariterators.tar_file_expander,
                select_files=select_files,
                rename_files=rename_files,
                handler=handler,
            )
        )
        self.pipeline.append(partial(group_by_keys, handler=handler))
        self.pipeline.append(check_empty)

    def add_transform(self, transform):
        """Add a transformation to the dataset."""
        self.transformations.append(transform)
        return self

    def get_stats(self):
        """Return the number of cache accesses and misses."""
        return self.cache.accesses, self.cache.misses

    def check_cache_misses(self):
        """Check if the cache miss rate is too high."""
        accesses, misses = self.get_stats()
        if accesses > 100 and misses / accesses > 0.3:
            # output a warning only once
            self.check_cache_misses = lambda: None
            print(
                "Warning: ShardListDataset has a cache miss rate of {:.1%}%".format(
                    misses * 100.0 / accesses
                )
            )

    def __iter__(self):
        """Iterate over the dataset."""
        self.epoch += 1
        set_pipeline_epochs(self.pipeline, self.epoch)

        for sample in run_pipeline(self.pipeline):
            transformed = apply_transformations(self.transformations, sample)
            yield transformed

    def set_size(self, n):
        """Set the size of the dataset."""
        self.total_size = n

    def size(self):
        """Return the number of samples in the dataset.

        This is not called __len__ because some PyTorch code checks for the presence
        of that method to determine if the dataset is indexable. Furthermore, the length
        need not be accurate, and for some datasets, we do not know the length.
        """
        return self.total_size

    def close(self):
        """ "Close the dataset."""
        for stage in self.pipeline[::-1]:
            if hasattr(stage, "close"):
                stage.close()
            del stage
        self.cache.clear()

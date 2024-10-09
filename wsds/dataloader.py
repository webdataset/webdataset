import dataclasses
import importlib
import io
import sys
import warnings
from typing import Any, Callable, Optional, Union

import braceexpand
import yaml
from icecream import ic
from torch.utils.data import DataLoader, IterableDataset

from webdataset import filters, shardlists

from . import datasets


def single_node_only(src, group=None):
    """Ensure the input sequence is not split for multi-node training.

    Args:
        src: The input sequence.
        group: The process group for distributed training.

    Yields:
        Elements from the input sequence.

    Raises:
        ValueError: If multi-node training is detected.
    """
    rank, world_size, worker, num_workers = utils.pytorch_worker_info(group=group)
    if world_size > 1:
        raise ValueError(
            "you need to add an explicit nodesplitter to your input pipeline for multi-node training"
        )
    yield from src


def split_by_worker(src):
    """Split the input sequence by PyTorch DataLoader worker.

    When used with multinode training, this will use all shards on each node.
    The result is that each epoch will be num_workers times larger than
    the dataset. It also guarantees that epochs are the same size on all workers.

    Args:
        src: The input sequence to be split.

    Yields:
        Elements from the input sequence based on the worker's ID.
    """
    rank, world_size, worker, num_workers = utils.pytorch_worker_info()
    if num_workers > 1:
        yield from islice(src, worker, None, num_workers)
    else:
        yield from src


def split_by_node_and_worker(src, group=None):
    """Split the input sequence by PyTorch distributed rank.

    When used with multinode training, this will split shards by node and by
    worker. This often means that different nodes will get different numbers
    of samples.

    Args:
        src: The input sequence to be split.
        group: The process group for distributed training.

    Yields:
        Elements from the input sequence based on the node's rank.
    """
    rank, world_size, worker, num_workers = utils.pytorch_worker_info(group=group)
    assert num_workers >= 1
    assert world_size >= 1
    if world_size > 1 or num_workers > 1:
        yield from islice(
            src, rank + world_size * worker, None, world_size * num_workers
        )
    else:
        yield from src


@dataclasses.dataclass
class DataloaderSpec:
    loader_class: str = "SingleNodeLoader"
    batch_size: Optional[int] = None
    reshuffle_size: Optional[int] = None
    collation_fn: Optional[Union[Callable, str]] = None
    num_workers: int = 4
    force_size: int = -1


def read_yaml_spec(spec, which):
    if spec.startswith("---\n"):
        spec_data = yaml.safe_load(io.StringIO(spec))
    else:
        with gopen(spec) as stream:
            spec_data = yaml.safe_load(stream)
    assert spec_data["__kind__"] == "webdataset-spec-v1"
    if which is None:
        spec_data = spec_data.get("train") or spec_data.get("default")
        assert spec_data is not None, "spec does not contain train or default"
    else:
        spec_data = spec_data.get(which)
        assert spec_data is not None, f"spec does not contain {which}"
    return spec_data


class GenericLoader:
    def __init__(self, *, dataset_spec=None, loader_spec=None):
        loader_spec.shard_split_fn = datasets.get_callable(loader_spec.shard_split_fn)
        self.dataset = datasets.SequentialDataset(dataset_spec)
        self.loader = DataLoader(
            self.dataset, batch_size=None, num_workers=loader_spec.num_workers
        )
        self.pipeline = [self.loader.__iter__]
        if loader_spec.reshuffle_size is not None:
            batch_size = loader_spec.batch_size or dataset_spec.batch_size
            if not batch_size:
                raise ValueError(
                    "dataset or loader batch_size must be specified when using reshuffle_size"
                )
            self.pipeline.append(filters.unbatched())
            self.pipeline.append(filters.shuffle(loader_spec.reshuffle_size))
            self.pipeline.append(filters.batched(batch_size))
        else:
            if loader_spec.batch_size is not None and loader_spec.batch_size > 0:
                warnings.warn(
                    "loader batch_size is ignored when reshuffle_size is not specified"
                )

    def __iter__(self):
        return datasets.run_pipeline(self.pipeline)


class SingleNodeLoader(GenericLoader):
    def __init__(self, *, dataset_spec=None, loader_spec=None):
        dataset_spec = dataset_spec.copy()
        dataset_spec.shard_split_fn = single_node_only
        super().__init__(dataset_spec=dataset_spec, loader_spec=loader_spec)


class ResamplingLoaderNoSplit(GenericLoader):
    def __init__(self, *, dataset_spec=None, loader_spec=None):
        dataset_spec = dataset_spec.copy()
        dataset_spec.resampling = True
        dataset_spec.shard_split_fn = split_by_worker
        super().__init__(dataset_spec=dataset_spec, loader_spec=loader_spec)


class ResamplingLoaderSplit:
    def __init__(self, *, dataset_spec=None, loader_spec=None):
        dataset_spec = dataset_spec.copy()
        dataset_spec.resampling = True
        dataset_spec.shard_split_fn = split_by_node_and_worker
        super().__init__(dataset_spec=dataset_spec, loader_spec=loader_spec)


class MultiNodeAllShardsLoader:
    def __init__(self, *, dataset_spec=None, loader_spec=None):
        dataset_spec = dataset_spec.copy()
        dataset_spec.resampling = False
        dataset_spec.shard_split_fn = split_by_worker
        super().__init__(dataset_spec=dataset_spec, loader_spec=loader_spec)


class MultiNodeSplitShardsLoader:
    def __init__(self, *, dataset_spec=None, loader_spec=None):
        dataset_spec = dataset_spec.copy()
        dataset_spec.resampling = False
        dataset_spec.shard_split_fn = split_by_node_and_worker
        super().__init__(dataset_spec=dataset_spec, loader_spec=loader_spec)


def make_dataset(spec, which="train", **kw):
    if isinstance(spec, str):
        spec = read_yaml_spec(spec, which)
    if isinstance(spec, dict):
        spec = datasets.DatasetSpec(**spec["sequential"])
    return datasets.SequentialDataset(spec)


def make_loader(spec, which="train"):
    if isinstance(spec, str):
        spec = read_yaml_spec(spec, which)
    if isinstance(spec, dict):
        spec = (
            datasets.DatasetSpec(**spec["sequential"]),
            DataloaderSpec(**spec["loader"]),
        )
    dataset_spec, loader_spec = spec
    if loader_spec.num_workers == -1:
        return datasets.SequentialDataset(dataset_spec)
    loader_class = getattr(sys.modules[__name__], loader_spec.loader_class)
    assert isinstance(loader_class, type)
    return loader_class(dataset_spec=dataset_spec, loader_spec=loader_spec)

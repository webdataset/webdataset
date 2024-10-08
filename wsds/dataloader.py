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


class SingleNodeLoader:
    def __init__(self, *, dataset_spec=None, loader_spec=None):
        loader_spec.shard_split_fn = shardlists.single_node_only
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


class MultiNodeResamplingLoader:
    def __init__(self, *, dataset_spec=None, loader_spec=None):
        raise NotImplementedError("MultiNodeResamplingLoader is not implemented yet")


class MultiNodeAllShardsLoader:
    def __init__(self, *, dataset_spec=None, loader_spec=None):
        raise NotImplementedError("MultiNodeResamplingLoader is not implemented yet")


class MultiNodeSplitShardsLoader:
    def __init__(self, *, dataset_spec=None, loader_spec=None):
        raise NotImplementedError("MultiNodeResamplingLoader is not implemented yet")


def make_dataset(spec, which="train"):
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
    loader_class = getattr(sys.modules[__name__], loader_spec.loader_class)
    assert isinstance(loader_class, type)
    return loader_class(dataset_spec=dataset_spec, loader_spec=loader_spec)

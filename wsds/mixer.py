import dataclasses
import importlib
import io
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import braceexpand
import yaml
from icecream import ic
from torch.utils.data import DataLoader, IterableDataset

from webdataset import filters, shardlists

from . import datasets
from .dataloader import make_dataset, make_loader


@dataclasses.dataclass
class MixComponent:
    # the source is either a dataset or a dataloader
    dataset_spec: Any = None
    # use either a dataloader as a source or a dataset directly
    use_loader: bool = True
    # frequency with which samples are selected; <1.0: random choice, >1: use every nth sample
    frequency: Union[int, float] = 1.0
    # number of repeats of the source
    repeats: int = 1
    # force the number of samples from the source
    force_size: int = -1
    rebatch: int = -1
    rename_fields: List[Tuple[str, str]] = dataclasses.field(default_factory=list)
    transformations: List[Union[str, Callable]] = dataclasses.field(
        default_factory=list
    )
    # actual source (either a SequentialDataset or a DataLoader)
    source: Any = None
    # names and comments
    name: Optional[str] = None
    comment: Optional[str] = None


@dataclasses.dataclass
class MixerSpec:
    loaders: List[MixComponent] = dataclasses.field(default_factory=list)
    # wrap the mixer in a dataloader if num_workers > 0
    num_workers: int = -1


def read_yaml_spec(spec, which):
    if spec.startswith("---\n"):
        spec_data = yaml.safe_load(io.StringIO(spec))
    else:
        with gopen(spec) as stream:
            spec_data = yaml.safe_load(stream)
    assert spec_data["__kind__"] == "wsds-mixer-v1"
    if which is None:
        spec_data = spec_data.get("train") or spec_data.get("default")
        assert spec_data is not None, "spec does not contain train or default"
    else:
        spec_data = spec_data.get(which)
        assert spec_data is not None, f"spec does not contain {which}"
    return spec_data


def get_mixer_spec(spec, which):
    mixer_spec = read_yaml_spec(spec, which)
    components = [MixComponent(**x) for x in mixer_spec["loaders"]]
    del mixer_spec["loaders"]
    mixer = MixerSpec(**mixer_spec)
    for c in components:
        if c.use_loader:
            if isinstance(c.dataset_spec, str):
                c.source = make_loader(c.dataset_spec)
            else:
                raise ValueError(
                    f"unknown type for dataset_spec: {type(c.dataset_spec)}"
                )
        else:
            if isinstance(c.dataset_spec, str):
                c.source = make_dataset(c.dataset_spec)
            else:
                raise ValueError(
                    f"unknown type for dataset_spec: {type(c.dataset_spec)}"
                )
    mixer.loaders = components
    return mixer


class Repeater:
    def __init__(self, source, repeats=1, force_size=-1, frequency=1.0):
        self.source = source
        self.repeats = repeats
        self.force_size = force_size
        assert frequency >= 0.0
        self.frequency = frequency

    def __iter__(self):
        count = 0
        for i in range(self.repeats):
            for x in self.source:
                if self.force_size > 0 and count >= self.force_size:
                    return
                if self.frequency > 1 and count % self.frequency != 0:
                    continue
                elif self.frequency < 1.0 and random.random() > self.frequency:
                    continue
                count += 1
                yield x


class Mixer:
    def __init__(self, mixer_spec, which):
        self.mixer = get_mixer_spec(mixer_spec, which)
        self.pipeline = []
        self.transformations = []

    def make_iterator(self, component):
        if (
            component.repeats > 1
            or component.force_size > 0
            or component.frequency != 1.0
        ):
            return iter(
                Repeater(
                    component.source,
                    component.repeats,
                    component.force_size,
                    component.frequency,
                )
            )
        else:
            return iter(component.source)

    def __iter__(self):
        iterators = [make_iterator(component) for component in self.mixer.loaders]


def make_mixer(mixer_spec, which):
    mixer = get_mixer_spec(mixer_spec, which)
    return mixer

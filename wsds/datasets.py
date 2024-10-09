import dataclasses
import fnmatch
import importlib
import io
import json
import os
import random
import re
import warnings
from functools import partial
from itertools import islice
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse

import braceexpand
import yaml

from webdataset import autodecode, cache, filters, gopen, shardlists, tariterators
from wids import wids_decode

try:
    from torch.utils.data import IterableDataset
except ImportError:

    class IterableDataset:
        pass


def apply_regex_list(patterns, s):
    if patterns is None:
        return s
    for k, v in patterns:
        s = re.sub(k, v, s)
    return s


def run_pipeline(pipeline):
    """Run a list of iterators as a pipeline.

    This can be so much shorter than the Pipeline class
    because for these classes, the pipelines are fixed
    and only created once inside the constructor. Users
    never use it.

    Args:
        pipeline (list): List of callables returning iterators.
    """
    source = pipeline[0]()
    for filter in pipeline[1:]:
        assert source is not None
        source = filter(source)
        assert source is not None, filter
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
    if transformations is None or transformations == []:
        return x
    if callable(transformations):
        return transformations(x)
    if isinstance(transformations, list):
        for transformation in transformations:
            assert callable(transformation), transformation
            x = transformation(x)
        return x
    raise ValueError(f"bad transformations: {transformations}")


def fix_dots(sample):
    for k in list(sample.keys()):
        if k.startswith("__") or k.startswith("."):
            continue
        sample["." + k] = sample[k]
        del sample[k]


def map_stream(source, f=None):
    for x in source:
        yield f(x)


def map_expand(source, f=None):
    for x in source:
        y = f(x)
        if isinstance(y, Iterator):
            yield from y
        elif isinstance(y, (dict, tuple)):
            yield y
        else:
            raise ValueError(f"function {f} returned unexpected type {type(y)}")


def default_handler(exn):
    raise exn


def default_transformations():
    return ["gz", "basic"]


@dataclasses.dataclass
class DatasetSpec:
    # basic dataset info
    shards: List[str] = dataclasses.field(default_factory=list)
    transformations: List[Union[str, Callable]] = dataclasses.field(
        default_factory=default_transformations
    )
    handler: Callable = default_handler
    check_empty: bool = False
    file_fn: Optional[Callable] = None
    repeats: int = 1
    force_size: Optional[int] = None
    total_size: int = -1

    # shard splitting
    resampling: bool = False
    shard_split_fn: Optional[callable] = None

    # renaming
    rename_fields: List[Tuple[str, str]] = dataclasses.field(default_factory=list)

    # shuffle options
    shuffle_size: int = -1
    shard_shuffle_size: int = 10000

    # batching options
    batch_size: Optional[int] = None
    batch_partial: bool = True
    collation_fn: Optional[Union[Callable, str]] = filters.default_collation_fn

    # JSON specification files
    dataset_name: Optional[str] = None
    base_url: Optional[Any] = None
    override_options: Optional[Dict[str, Any]] = None

    # caching related options
    cache_size: int = int(1e12)
    cache_dir: Optional[str] = None
    localname_fn: Optional[Callable] = None
    lru_size: int = 10
    keep_downloaded: bool = False

    # debugging
    log_shards: Optional[str] = None
    log_keys: Optional[str] = None


def get_callable(x):
    if callable(x):
        return x
    if isinstance(x, str):
        result = lookup_qualified_python_symbols(x)
        assert callable(result), f"lookup of {x} did not result in a callable"
        return result
    raise ValueError(f"bad transformations: {x}")


def lookup_qualified_python_symbols(sym: str):
    args = None
    if isinstance(sym, dict):
        args = dict(sym)
        del args["fn"]
        sym = sym["fn"]
    symbol_name = sym.split(".")[-1]
    module_path = sym.split(".")[:-1]
    symbol_value = getattr(importlib.import_module(".".join(module_path)), symbol_name)
    if args is not None:
        return partial(symbol_value, **args)
    else:
        return symbol_value


def interpret_transformations(transformations):
    """Interpret a list of transformations.

    This resolves strings into functions.
    """
    result = []
    if not isinstance(transformations, list):
        transformations = [transformations]
    for t in transformations:
        if isinstance(t, str):
            if t.lower() in ["gz"]:
                t = wids_decode.decode_all_gz
            elif t.lower() in ["", "basic"]:
                t = wids_decode.decode_basic
            elif t.lower() == "pil":
                t = wids_decode.decode_images_to_pil
            elif t.lower() == "rgb":
                t = wids_decode.decode_image_to_numpy
            else:
                t = lookup_qualified_python_symbols(t)
        elif isinstance(t, dict):
            t = lookup_qualified_python_symbols(t)
        assert callable(t), t
        result.append(t)
    return result


def add_len_method(obj):
    """Add a fake __len__ method to an object.

    This is useful for frameworks that happen to work with
    IterableDataset but still use __len__ to determine the
    length of the dataset. This is not usually recommended
    because PyTorch does not expect __len__ on IterableDataset.
    """

    def fake_len(self):
        return self.total_size

    obj.__len__ = fake_len


def read_shards_from_json(filename, base_url=None, check=True):
    with gopen(filename) as stream:
        shard_info = json.load(stream)
    if base_url is None:
        base_url = filename
    urls = [x["url"] for x in shard_info["shardlist"]]
    if base_url is not None:
        urls = [urljoin(base_url, url) for url in urls]
    if check:
        assert (
            shard_info["__kind__"] == "wids-shard-index-v1"
            or shard_info["wids_version"] == 1
        )
        assert "shardlist" in shard_info
    total_size = sum(x["nsamples"] for x in shard_info["shardlist"])
    return urls, total_size


def read_yaml_spec(spec, which):
    if spec.startswith("---\n"):
        spec_data = yaml.safe_load(io.StringIO(spec))
    else:
        with gopen(spec) as stream:
            spec_data = yaml.safe_load(stream)
    assert spec_data is not None, "spec is None"
    if which is None:
        spec_data = spec_data.get("train") or spec_data.get("default")
        assert spec_data is not None, "spec does not contain train or default"
    else:
        spec_data = spec_data.get(which)
        assert spec_data is not None, f"spec does not contain {which}"
    assert "sequential" in spec_data, spec_data
    args = DatasetSpec()
    result = dataclasses.replace(args, **spec_data["sequential"])
    return result


class SequentialDataset(IterableDataset):
    def __init__(
        self,
        spec=None,
        which=None,
        **kw,
    ):
        if isinstance(spec, str):
            self.args = read_yaml_spec(spec, which)
        elif isinstance(spec, dict):
            self.args = DatasetSpec(**spec)
        elif isinstance(spec, DatasetSpec):
            self.args = spec
        else:
            self.args = DatasetSpec()
        self.args = dataclasses.replace(self.args, **kw)
        self.total_size = self.args.total_size
        self.epoch = -1
        self.shardlist = None
        self.log_shards_stream = None
        self.log_keys_stream = None
        self.init_pipeline()
        if self.args.cache_dir is not None:
            self.cache = cache.FileCache(
                cache_dir=self.args.cache_dir,
                cache_size=self.args.cache_size,
            )
        else:
            self.cache = None
        self.read_shardlist()
        self.transformations = interpret_transformations(
            self.args.transformations or []
        )
        if not isinstance(self.transformations, list):
            assert callable(self.transformations)
            self.transformations = [self.transformations]

    def add_transform(self, transformation):
        self.transformations.append(transformation)
        return self

    def open_log(self, dest):
        if dest is None:
            stream = None
        elif dest == "-":
            stream = sys.stderr
        elif isinstance(dest, str):
            stream = open(dest, "w")
        else:
            stream = dest
        return stream

    def log_shards(self, source):
        if self.log_shards_stream is None:
            self.log_shards_stream = self.open_log(self.args.log_shards)
        for x in source:
            if self.log_shards_stream is not None:
                self.log_shards_stream.write(x["__url__"])
            yield x

    def log_keys(self, source):
        if self.log_keys_stream is None:
            self.log_keys_stream = self.open_log(self.args.log_keys)
        for x in source:
            if self.log_keys_stream is not None:
                self.log_keys_stream.write(x["__key__"])
            yield x

    def rename_fields(self, source):
        for x in source:
            if self.args.rename_fields is not None:
                keys = list(x.keys())
                for k in keys:
                    if k.startswith("_"):
                        continue
                    k_new = apply_regex_list(self.args.rename_fields, k)
                    if k_new != k:
                        x[k_new] = x[k]
                        del x[k]
            yield x

    def init_pipeline(self):
        self.pipeline = [
            self.iterate_shards,
            self.split_shards,
        ]
        if self.args.resampling:
            self.pipeline += [
                self.resample_shards,
            ]
        else:
            self.pipeline += [
                self.repeat_shards,
                self.shuffle_shards,
            ]
        self.pipeline += [
            self.open_shards,
            self.log_shards,
            self.iterate_tar_files,
            self.rename_files,
            self.group_by_keys,
            self.rename_fields,
            self.log_keys,
            self.shuffle_samples,
            self.transform_samples,
            self.limit_size,
            self.batch_samples,
        ]

    def read_shardlist(self):
        """Get a list of shards from a string, list, or JSON file."""
        shards = self.args.shards
        if isinstance(shards, str) and shards.endswith(".json"):
            shards, total_size = read_shards_from_json(
                shards, base_url=self.args.base_url
            )
            self.shardlist = shards
            self.total_size = total_size
        elif isinstance(shards, str):
            self.shardlist = list(braceexpand.braceexpand(shards))
        elif isinstance(shards, list):
            self.shardlist = shards
        else:
            raise ValueError(f"unknown shard list type {shards}")
        assert self.shardlist is not None
        assert len(self.shardlist) > 0, f"{self.args.shards = }"

    def iterate_shards(self):
        """Iterate over the shardlist."""
        for i, shard in enumerate(self.shardlist):
            yield dict(url=shard, shard_num=i)

    def split_shards(self, source):
        """Split the shardlist according to a custom function.

        This is used for multiple workers and/or multiple nodes.
        """
        if self.args.shard_split_fn:
            yield from self.args.shard_split_fn(source)
        else:
            yield from source

    def repeat_shards(self, source):
        """Repeat the shards according to the repeats parameter.

        This takes place after splitting the shards, so the repeats
        are per worker or per node.
        """
        shards = list(source)
        for i in range(self.args.repeats):
            for shard in shards:
                yield shard

    def resample_shards(self, source):
        """Return an endless stream of shard samples from the source.

        This takes place after splitting the shards, so the repeats
        are per worker or per node.
        """
        shards = list(source)
        while True:
            yield random.choice(shards)

    def shuffle_shards(self, source):
        """Shuffle the shards."""
        n = self.args.shard_shuffle_size
        yield from filters.shuffle(bufsize=n, initial=n)(source)

    def open_shards(self, source):
        """Open the shards and yield url+stream dicts.

        This optionally uses a cache to store the shards if a
        cache_dir is provided.
        """
        if self.args.cache_dir is None:
            yield from tariterators.url_opener(source)
        else:
            yield from self.cache(source)

    def iterate_tar_files(self, source):
        """Iterate over the files in the tar archives."""
        yield from tariterators.tar_file_expander(source)

    def rename_files(self, source):
        """Rename files according to a custom function."""
        if self.args.file_fn:
            yield from map_expand(source, self.args.file_fn)
        else:
            yield from source

    def group_by_keys(self, source):
        """Group samples by keys using WebDataset conventions."""
        for i, sample in enumerate(tariterators.group_by_keys(source)):
            fix_dots(sample)
            sample["__epoch__"] = self.epoch
            sample["__count__"] = i
            yield sample

    def shuffle_samples(self, source):
        """Shuffle the samples within the stream using shuffle_size."""
        n = self.args.shuffle_size
        yield from filters.shuffle(n)(source)

    def transform_sample(self, sample):
        """Apply the given transformations to the sample."""
        return apply_transformations(self.transformations, sample)

    def transform_samples(self, source):
        """Apply the transformations to the stream of samples.

        This can add samples (by returning an iterator) or
        remove samples (by returning None).
        """
        yield from map_expand(source, self.transform_sample)

    def limit_size(self, source):
        """Limit the size of the dataset to args.force_size."""
        if self.args.force_size is not None:
            yield from islice(source, self.args.force_size)
        else:
            yield from source

    def batch_samples(self, source):
        """Batch the samples if a batch_size is given."""
        if self.args.batch_size is None:
            yield from source
            return
        batcher = filters.batched(
            self.args.batch_size, collation_fn=get_callable(self.args.collation_fn)
        )
        yield from batcher(source)

    def get_stats(self):
        """Return the number of cache accesses and misses."""
        if self.cache is None:
            return 0, 0
        return self.cache.accesses, self.cache.misses

    def check_cache_misses(self):
        """Check if the cache miss rate is too high."""
        if self.cache is None:
            return
        accesses, misses = self.get_stats()
        if accesses > 100 and misses / accesses > 0.3:
            warnings.warn(
                "ShardListDataset has a cache miss rate of {:.1%}%".format(
                    misses * 100.0 / accesses
                )
            )

    def add_transform(self, transformation):
        """Add a transformation to the dataset."""
        self.transformations.append(transformation)

    def __iter__(self):
        """Iterate over the dataset."""
        self.epoch += 1
        set_pipeline_epochs(self.pipeline, self.epoch)
        yield from run_pipeline(self.pipeline)

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
        del self.cache

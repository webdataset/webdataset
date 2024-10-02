import webdataset.typecheck  # isort:skip
import os
import tempfile
import time
from urllib.parse import urlparse

import pytest

import wsds
from tests.conftest import local_data, remote_loc, remote_shards
from wsds.datasets import DatasetSpec, SequentialDataset


def summarize_sample(sample):
    for k, v in sample.items():
        print(k, type(v), repr(v)[:80])


def test_smoke():
    ds = wsds.SequentialDataset(shards=[local_data])


def test_all():
    ds = wsds.SequentialDataset(shards=[local_data])
    for i in range(1, len(ds.pipeline)):
        ds = wsds.SequentialDataset(shards=[local_data])
        print("testing pipeline", i, ds.pipeline[i - 1])
        ds.pipeline = ds.pipeline[:i]
        for sample in ds:
            pass


def test_simple():
    ds = wsds.SequentialDataset(shards=[local_data])
    ds.read_shardlist()
    assert ds.size() == -1
    count = 0
    # required_keys = "__key__ __url__ __local_path__ .cls .png .wnid .xml".split()
    required_keys = "__key__ __url__ .cls .png .wnid .xml".split()
    keys = set()
    for sample in ds:
        assert set(sample.keys()) >= set(required_keys)
        keys.add(sample["__key__"])
        assert isinstance(sample[".png"], bytes)
        count += 1
    assert count == 47
    assert len(keys) == 47


def test_repeats():
    ds = wsds.SequentialDataset(shards=[local_data], repeats=2)
    ds.read_shardlist()
    assert ds.size() == -1
    count = 0
    # required_keys = "__key__ __url__ __local_path__ .cls .png .wnid .xml".split()
    required_keys = "__key__ __url__ .cls .png .wnid .xml".split()
    keys = set()
    for sample in ds:
        assert set(sample.keys()) >= set(required_keys)
        keys.add(sample["__key__"])
        assert isinstance(sample[".png"], bytes)
        count += 1
    assert count == 47 * 2
    assert len(keys) == 47


def test_force_size():
    ds = wsds.SequentialDataset(shards=[local_data], repeats=2, force_size=70)
    ds.read_shardlist()
    assert ds.size() == -1
    count = 0
    # required_keys = "__key__ __url__ __local_path__ .cls .png .wnid .xml".split()
    required_keys = "__key__ __url__ .cls .png .wnid .xml".split()
    keys = set()
    for sample in ds:
        assert set(sample.keys()) >= set(required_keys)
        keys.add(sample["__key__"])
        assert isinstance(sample[".png"], bytes)
        count += 1
    assert count == 70
    assert len(keys) == 47


def test_transforms():
    import numpy as np

    def my_transform(sample):
        wsds.decode_all_gz(sample)
        wsds.decode_basic(sample)
        wsds.decode_images_to_numpy(sample)
        return sample

    ds = wsds.SequentialDataset(shards=[local_data], transformations=my_transform)
    for sample in ds:
        assert isinstance(sample[".png"], np.ndarray)


def test_iterate_shards():
    shards = ["shard1.tar", "shard2.tar", "shard3.tar"]
    dataset = SequentialDataset(shards=shards)
    dataset.read_shardlist()
    assert [x["url"] for x in dataset.iterate_shards()] == shards


def test_split_shards():
    def custom_split(source):
        for i, shard in enumerate(source):
            if i % 2 == 0:
                yield shard

    shards = ["shard1.tar", "shard2.tar", "shard3.tar", "shard4.tar"]
    dataset = SequentialDataset(shards=shards, shard_split_fn=custom_split)
    print(dataset.shardlist)
    dataset.read_shardlist()
    print(dataset.shardlist)
    assert list(dataset.split_shards(shards)) == ["shard1.tar", "shard3.tar"]


def test_shuffle_shards():
    shards = [{"shard": f"shard{i}"} for i in range(100)]
    dataset = SequentialDataset(shards=shards, shard_shuffle_size=10)
    shuffled = list(dataset.shuffle_shards(iter(shards)))
    assert len(shuffled) == len(shards)
    assert shuffled != shards


def test_rename_files():
    def custom_rename(sample):
        sample["renamed"] = sample["original"] + "_renamed"
        return sample

    samples = [{"original": f"file{i}"} for i in range(5)]
    dataset = SequentialDataset(shards=["dummy.tar"], file_fn=custom_rename)
    renamed = list(dataset.rename_files(samples))
    assert all("renamed" in sample for sample in renamed)
    assert all(
        sample["renamed"] == sample["original"] + "_renamed" for sample in renamed
    )


def test_group_by_keys():
    samples = [
        {"fname": "a/b.png", "data": b"", "__url__": "http://foo/a/b.png"},
        {"fname": "a/b.cls", "data": b"", "__url__": "http://foo/a/b.png"},
        {"fname": "c/d.png", "data": b"", "__url__": "http://foo/a/b.png"},
        {"fname": "c/d.cls", "data": b"", "__url__": "http://foo/a/b.png"},
    ]
    dataset = SequentialDataset(shards=["dummy.tar"])
    grouped = list(dataset.group_by_keys(samples))
    assert len(grouped) == 2
    assert grouped[0]["__key__"] == "a/b"
    assert grouped[1]["__key__"] == "c/d"


def test_transform_sample():
    def custom_transform(sample):
        sample["transformed"] = sample["original"] * 2
        return sample

    dataset = SequentialDataset(
        shards=["dummy.tar"], transformations=[custom_transform]
    )
    sample = {"original": 5}
    transformed = dataset.transform_sample(sample)
    assert transformed["transformed"] == 10


def test_transform_samples():
    def custom_transform(sample):
        sample["transformed"] = sample["original"] * 2
        return sample

    samples = [{"original": i} for i in range(5)]
    dataset = SequentialDataset(
        shards=["dummy.tar"], transformations=[custom_transform]
    )
    transformed = list(dataset.transform_samples(samples))
    assert all(
        sample["transformed"] == sample["original"] * 2 for sample in transformed
    )


def test_batch_samples():
    samples = [{"value": i} for i in range(10)]
    dataset = SequentialDataset(shards=["dummy.tar"], batch_size=3)
    batches = list(dataset.batch_samples(samples))
    assert len(batches) == 4
    shapes = [batch["value"].shape for batch in batches]
    assert all(shape == (3,) for shape in shapes[:-1]), shapes
    assert batches[-1]["value"].shape == (1,)

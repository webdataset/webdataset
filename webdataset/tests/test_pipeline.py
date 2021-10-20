import io
import os
import numpy as np
import PIL
import pytest
import torch
import pickle

import webdataset as wds
from webdataset.pipeline import DataPipeline, stage
from webdataset import filters
from webdataset import iterators
from webdataset import tariterators
from webdataset import autodecode
from webdataset import PytorchShardList, SimpleShardList


def test_trivial():
    dataset = DataPipeline(
        lambda: iter([1, 2, 3, 4])
    )
    result = list(iter(dataset))
    assert result == [1, 2, 3, 4]

def test_trivial_map():
    dataset = DataPipeline(
        lambda: iter([1, 2, 3, 4]),
        filters.map(lambda x: x + 1)
    )
    result = list(iter(dataset))
    assert result == [2, 3, 4, 5]

def test_trivial_map2():
    dataset = DataPipeline(
        lambda: iter([1, 2, 3, 4]),
        lambda src: iterators.map(src, lambda x: x + 1)
    )
    result = list(iter(dataset))
    assert result == [2, 3, 4, 5]


def test_trivial_map3():
    dataset = DataPipeline(
        lambda: iter([1, 2, 3, 4]),
        stage(iterators.map, lambda x: x + 1)
    )
    result = list(iter(dataset))
    assert result == [2, 3, 4, 5]

def adder4(src):
    for x in src:
        yield x + 4

def test_trivial_map4():
    dataset = DataPipeline(
        lambda: iter([1, 2, 3, 4]),
        adder4,
    )
    result = list(iter(dataset))
    assert result == [5, 6, 7, 8]

def test_pytorchshardlist():
    dataset = DataPipeline(
        wds.SimpleShardList("test-{000000..000099}.tar"),
    )
    result = list(iter(dataset))
    assert len(result) == 100

def test_composable():
    dataset = DataPipeline(
        wds.SimpleShardList("test-{000000..000099}.tar"),
    )
    result = list(iter(dataset))
    assert len(result) == 100

def test_reader1():
    dataset = DataPipeline(
        wds.SimpleShardList("testdata/imagenet-000000.tgz"),
        wds.tarfile_samples,
        wds.decode(autodecode.ImageHandler("rgb")),
    )
    result = list(iter(dataset))
    keys = list(result[0].keys())
    assert "__key__" in keys
    assert "cls" in keys
    assert "png" in keys
    assert isinstance(result[0]["cls"], int)
    assert isinstance(result[0]["png"], np.ndarray)
    assert result[0]["png"].shape == (793, 600, 3)
    assert len(result) == 47
    

def test_reader2():
    dataset = DataPipeline(
        wds.SimpleShardList(["testdata/imagenet-000000.tgz"]*10),
        wds.shuffle(3),
        wds.tarfile_samples,
        wds.shuffle(100),
        wds.decode(autodecode.ImageHandler("rgb")),
        wds.to_tuple("png", "cls")
    )
    result = list(iter(dataset))
    assert len(result[0]) == 2
    assert isinstance(result[0][0], np.ndarray)
    assert isinstance(result[0][1], int)
    assert len(result) == 470


def resampled_(src, n):
    import random
    items = list(src)
    for i in range(n):
        yield random.choice(items)

resampled = wds.Curried(resampled_)

def test_reader3():
    dataset = DataPipeline(
        wds.SimpleShardList(["testdata/imagenet-000000.tgz"]*3),
        resampled(10),
        wds.tarfile_samples,
        wds.shuffle(100),
        wds.decode(autodecode.ImageHandler("rgb")),
        wds.to_tuple("png", "cls")
    )
    result = list(iter(dataset))
    assert len(result[0]) == 2
    assert isinstance(result[0][0], np.ndarray)
    assert isinstance(result[0][1], int)
    assert len(result) == 470


def test_splitting():
    dataset = DataPipeline(
        wds.SimpleShardList(list(map(str, range(10)))),
        wds.split_by_node,
        wds.split_by_worker,
    )
    result = list(iter(dataset))
    assert len(result) == 10
    assert result[0]["url"] == "0"


def test_seed():
    dataset = DataPipeline(
        wds.SimpleShardList(list(map(str, range(10)))),
        wds.split_by_node,
        wds.split_by_worker,
    )
    result = list(iter(dataset))
    assert len(result) == 10
    assert result[0]["url"] == "0"
    epoch = 17
    dataset.stage(0).seed = epoch
    result = list(iter(dataset))
    assert len(result) == 10
    assert result[0]["url"] == "7"

def test_nonempty():
    dataset = DataPipeline(
        wds.SimpleShardList(list(map(str, range(10)))),
        wds.split_by_node,
        wds.split_by_worker,
        wds.non_empty,
    )
    result = list(iter(dataset))
    assert len(result) == 10
    assert result[0]["url"] == "0"


def test_nonempty2():
    dataset = DataPipeline(
        wds.SimpleShardList(list(map(str, range(10)))),
        lambda src: iter([]),
        wds.non_empty,
    )
    with pytest.raises(ValueError):
        list(iter(dataset))



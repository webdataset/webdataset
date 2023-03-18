import numpy as np

import webdataset as wds


def test_reader1():
    dataset = wds.DataPipeline(
        wds.SimpleShardList("testdata/imagenet-000000.tgz"),
        wds.tarfile_samples,
    )
    result = list(iter(dataset))
    assert len(result) == 47


def test_rr():
    dataset = wds.DataPipeline(
        wds.SimpleShardList("testdata/imagenet-000000.tgz"),
        wds.tarfile_samples,
    )
    mix = wds.RoundRobin([dataset, dataset])
    result = list(iter(mix))
    assert len(result) == 47 * 2
    dataset = wds.DataPipeline(
        wds.SimpleShardList("testdata/imagenet-000000.tgz"),
        wds.tarfile_samples,
    )
    mix = wds.RoundRobin([dataset, dataset, dataset])
    result = list(iter(mix))
    assert len(result) == 47 * 3
    mix = wds.FluidWrapper(wds.RoundRobin([dataset, dataset, dataset])).shuffle(10)
    result = list(iter(mix))
    assert len(result) == 47 * 3
    mix = wds.RoundRobin([np.full(2, 2), np.full(1, 1), np.full(3, 3)], longest=True)
    result = list(iter(mix))
    assert (result == np.array([2, 1, 3, 2, 3, 3])).all()


def test_rs():
    dataset = wds.DataPipeline(
        wds.SimpleShardList("testdata/imagenet-000000.tgz"),
        wds.tarfile_samples,
    )
    mix = wds.RandomMix([dataset, dataset], longest=True)
    result = list(iter(mix))
    assert len(result) == 47 * 2

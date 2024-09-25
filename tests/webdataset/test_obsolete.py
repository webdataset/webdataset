import os
import pickle
from io import StringIO

import numpy as np
import PIL
import pytest
import torch
import yaml

import webdataset as wds
from tests.conftest import (
    compressed,
    local_data,
    remote_loc,
    remote_pattern,
    remote_sample,
    remote_shard,
    remote_shards,
)


@pytest.mark.skip(reason="ddp_equalize is obsolete")
def test_ddp_equalize():
    ds = wds.WebDataset(local_data, shardshuffle=False).ddp_equalize(733)
    assert count_samples_tuple(ds) == 733


@pytest.mark.skip(reason="not implemented")
def test_log_keys(tmp_path):
    tmp_path = str(tmp_path)
    fname = tmp_path + "/test.ds.yml"
    ds = wds.WebDataset(local_data, shardshuffle=100).log_keys(fname)
    result = [x for x in ds]
    assert len(result) == 47
    with open(fname) as stream:
        lines = stream.readlines()
    assert len(lines) == 47


@pytest.mark.skip(reason="FIXME")
def test_length():
    ds = wds.WebDataset(local_data, shardshuffle=100)
    with pytest.raises(TypeError):
        len(ds)
    dsl = ds.with_length(1793)
    assert len(dsl) == 1793
    dsl2 = dsl.repeat(17).with_length(19)
    assert len(dsl2) == 19


@pytest.mark.skip(reason="need to figure out unraisableexceptionwarning")
def test_rgb8_np_vs_torch():
    import warnings

    warnings.filterwarnings("error")
    ds = (
        wds.WebDataset(local_data, shardshuffle=100)
        .decode("rgb8")
        .to_tuple("png;jpg", "cls")
    )
    image, cls = next(iter(ds))
    assert isinstance(image, np.ndarray), type(image)
    assert isinstance(cls, int), type(cls)
    ds = (
        wds.WebDataset(local_data, shardshuffle=100)
        .decode("torchrgb8")
        .to_tuple("png;jpg", "cls")
    )
    image2, cls2 = next(iter(ds))
    assert isinstance(image2, torch.Tensor), type(image2)
    assert isinstance(cls, int), type(cls)
    assert (image == image2.permute(1, 2, 0).numpy()).all, (image.shape, image2.shape)
    assert cls == cls2


@pytest.mark.skip(reason="fixme")
def test_associate():
    with open("testdata/imagenet-extra.json") as stream:
        extra_data = simplejson.load(stream)

    def associate(key):
        return dict(MY_EXTRA_DATA=extra_data[key])

    ds = wds.WebDataset(local_data, shardshuffle=100).associate(associate)

    for sample in ds:
        assert "MY_EXTRA_DATA" in sample.keys()
        break


@pytest.mark.skip(reason="fixme")
def test_container_mp():
    ds = wds.WebDataset(
        "testdata/mpdata.tar", container="mp", decoder=None, shardshuffle=100
    )
    assert count_samples_tuple(ds) == 100
    for sample in ds:
        assert isinstance(sample, dict)
        assert set(sample.keys()) == set("__key__ x y".split()), sample


@pytest.mark.skip(reason="fixme")
def test_container_ten():
    ds = wds.WebDataset(
        "testdata/tendata.tar", container="ten", decoder=None, shardshuffle=100
    )
    assert count_samples_tuple(ds) == 100
    for xs, ys in ds:
        assert xs.dtype == np.float64
        assert ys.dtype == np.float64
        assert xs.shape == (28, 28)
        assert ys.shape == (28, 28)


@pytest.mark.skip(reason="fixme")
def test_multimode():
    import torch

    urls = [local_data] * 8
    nsamples = 47 * 8

    shardlist = wds.PytorchShardList(
        urls, verbose=True, epoch_shuffle=True, shuffle=True
    )
    os.environ["WDS_EPOCH"] = "7"
    ds = wds.WebDataset(shardlist, shardshuffle=100)
    dl = torch.utils.data.DataLoader(ds, num_workers=4)
    count = count_samples_tuple(dl)
    assert count == nsamples, count
    del os.environ["WDS_EPOCH"]

    shardlist = wds.PytorchShardList(urls, verbose=True, split_by_worker=False)
    ds = wds.WebDataset(shardlist, shardshuffle=100)
    dl = torch.utils.data.DataLoader(ds, num_workers=4)
    count = count_samples_tuple(dl)
    assert count == 4 * nsamples, count

    shardlist = shardlists.ResampledShards(urls)
    ds = wds.WebDataset(shardlist, shardshuffle=100).slice(170)
    dl = torch.utils.data.DataLoader(ds, num_workers=4)
    count = count_samples_tuple(dl)
    assert count == 170 * 4, count


@pytest.mark.skip(reason="obsolete")
def test_ddp_equalize():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data), wds.tarfile_to_samples(), wds.ddp_equalize(773)
    )
    assert count_samples_tuple(ds) == 733


@pytest.mark.skip(reason="untested")
def test_container_mp():
    ds = wds.WebDataset(
        "testdata/mpdata.tar", container="mp", decoder=None, shardshuffle=100
    )
    assert count_samples_tuple(ds) == 100
    for sample in ds:
        assert isinstance(sample, dict)
        assert set(sample.keys()) == set("__key__ x y".split()), sample


@pytest.mark.skip(reason="untested")
def test_container_ten():
    ds = wds.WebDataset(
        "testdata/tendata.tar", container="ten", decoder=None, shardshuffle=100
    )
    assert count_samples_tuple(ds) == 100
    for xs, ys in ds:
        assert xs.dtype == np.float64
        assert ys.dtype == np.float64
        assert xs.shape == (28, 28)
        assert ys.shape == (28, 28)


@pytest.mark.skip(reason="fix this some time")
def test_opener():
    def opener(url):
        print(url, file=sys.stderr)
        cmd = "curl -s '{}{}'".format(remote_loc, remote_pattern.format(url))
        print(cmd, file=sys.stderr)
        return subprocess.Popen(
            cmd, bufsize=1000000, shell=True, stdout=subprocess.PIPE
        ).stdout

    ds = (
        wds.WebDataset("{000000..000099}", open_fn=opener, shardshuffle=100)
        .shuffle(100)
        .to_tuple("jpg;png", "json")
    )
    assert count_samples_tuple(ds, n=10) == 10


@pytest.mark.skip(reason="failing for unknown reason")
def test_pipe():
    ds = (
        wds.WebDataset(
            f"pipe:curl -s -L '{remote_loc}{remote_shards}'", shardshuffle=100
        )
        .shuffle(100)
        .to_tuple("jpg;png", "json")
    )
    assert count_samples_tuple(ds, n=10) == 10


@pytest.mark.skip(reason="need to figure out unraisableexceptionwarning")
def test_rgb8_np_vs_torch():
    import warnings

    warnings.filterwarnings("error")
    ds = (
        wds.WebDataset(local_data, shardshuffle=100)
        .decode("rgb8")
        .to_tuple("png;jpg", "cls")
    )
    image, cls = next(iter(ds))
    assert isinstance(image, np.ndarray), type(image)
    assert isinstance(cls, int), type(cls)
    ds = (
        wds.WebDataset(local_data, shardshuffle=100)
        .decode("torchrgb8")
        .to_tuple("png;jpg", "cls")
    )
    image2, cls2 = next(iter(ds))
    assert isinstance(image2, torch.Tensor), type(image2)
    assert isinstance(cls, int), type(cls)
    assert (image == image2.permute(1, 2, 0).numpy()).all, (image.shape, image2.shape)
    assert cls == cls2


@pytest.mark.skip(reason="untested")
def test_associate():
    """Test associating extra data with samples."""
    with open("testdata/imagenet-extra.json") as stream:
        extra_data = simplejson.load(stream)

    def associate(key):
        return dict(MY_EXTRA_DATA=extra_data[key])

    ds = wds.WebDataset(local_data, shardshuffle=100).associate(associate)

    for sample in ds:
        assert "MY_EXTRA_DATA" in sample.keys()
        break

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
from webdataset import compat


def identity(x):
    return x


def count_samples_tuple(source: iter, *args: callable, n: int = 10000) -> int:
    """
    Counts the number of samples from an iterable source that pass a set of conditions specified as callables.

    Args:
        source: An iterable source containing samples to be counted.
        *args: A set of callables representing conditions that a candidate
            sample has to meet to be considered for the count. These functions accept a single argument,
            which is the sample being considered, and return True or False indicating whether the condition is met.
        n: Maximum number of samples to consider for the count. Defaults to 10000.

    Returns:
        Number of samples from `source` that meet all conditions.

    Raises:
        AssertionError: If any of the samples from `source` is not tuple, dict, or list
            or if any of the callable arguments returns False when applied on any of the samples.

    """

    count: int = 0

    for i, sample in enumerate(source):
        if i >= n:
            break
        assert isinstance(sample, (tuple, dict, list)), (type(sample), sample)
        for f in args:
            assert f(sample)
        count += 1

    return count


def count_samples(source: iter, *args: callable, n: int = 1000) -> int:
    """
    Counts the number of samples from an iterable source that pass a set of conditions specified as callables.

    Args:
        source: An iterable source containing samples to be counted.
        *args: A set of callables representing conditions that a candidate sample has to meet to be considered for
            the count. These functions accept a single argument, which is the sample being considered, and
            return True or False indicating whether the condition is met.
        n: Maximum number of samples to consider for the count. Defaults to 1000.

    Returns:
        Number of samples from `source` that meet all conditions.

    Raises:
        AssertionError: If any of the samples from `source` fails to meet any of the callable arguments.
    """

    count: int = 0

    for i, sample in enumerate(source):
        if i >= n:
            break
        for f in args:
            assert f(sample)
        count += 1

    return count


def test_dataset():
    """
    Tests that the WebDataset object created from locally hosted data contains the expected number of samples.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the number of samples in the WebDataset object does not match the expected value.
    """
    ds = wds.WebDataset(local_data, shardshuffle=False)
    assert count_samples_tuple(ds) == 47


def test_dataset_resampled():
    """
    Tests that the WebDataset object created from resampled locally hosted data contains the expected number of samples.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the number of samples in the WebDataset object does not match the expected value.
    """
    ds = wds.WebDataset(local_data, resampled=True)
    assert count_samples_tuple(ds, n=100) == 100


@pytest.mark.quick
def test_length():
    """Test the with_length and repeat methods."""
    ds = wds.WebDataset(local_data, shardshuffle=False)
    # ensure that the dataset does not have a length property
    with pytest.raises(TypeError):
        len(ds)
    # ensure that the dataset has a length property after setting it
    dsl = ds.with_length(1793)
    assert len(dsl) == 1793
    # repeat the dataset 17 times
    dsl2 = ds.repeat(17)
    # ensure that the dataset has a length property after setting it
    dsl3 = dsl2.with_length(19)
    assert len(dsl3) == 19


def test_mock():
    """Test that MockDataset works as expected."""
    ds = wds.MockDataset((True, True), 193)
    assert count_samples_tuple(ds) == 193
    assert next(iter(ds)) == (True, True)


def test_dataset_shuffle_extract():
    """Basic WebDataset usage: shuffle, extract, and count samples."""
    ds = (
        wds.WebDataset(local_data, shardshuffle=False)
        .shuffle(5)
        .to_tuple("png;jpg cls")
    )
    assert count_samples_tuple(ds) == 47


@pytest.mark.quick
def test_dataset_context():
    """Basic WebDataset usage: shuffle, extract, and count samples."""
    with (
        wds.WebDataset(local_data, shardshuffle=100)
        .shuffle(5)
        .to_tuple("png;jpg cls") as ds
    ):
        assert count_samples_tuple(ds) == 47


@pytest.mark.quick
def test_dataset_pipe_cat():
    """Test that WebDataset can read from a pipe."""
    ds = (
        wds.WebDataset(f"pipe:cat {local_data}", shardshuffle=100)
        .shuffle(5)
        .to_tuple("png;jpg cls")
    )
    assert count_samples_tuple(ds) == 47


def test_slice():
    """Test the slice method."""
    ds = wds.WebDataset(local_data, shardshuffle=100).slice(10)
    assert count_samples_tuple(ds) == 10


@pytest.mark.quick
def test_dataset_eof():
    """Test that truncated tar files raise an error."""
    import tarfile

    with pytest.raises(tarfile.ReadError):
        ds = wds.WebDataset(
            f"pipe:dd if={local_data} bs=1024 count=10", shardshuffle=100
        ).shuffle(5)
        assert count_samples(ds) == 47


def test_dataset_eof_handler():
    """Test that we can ignore EOF errors by using a handler."""
    ds = wds.WebDataset(
        f"pipe:dd if={local_data} bs=1024 count=10",
        handler=wds.ignore_and_stop,
        shardshuffle=100,
    )
    assert count_samples(ds) < 47


def test_dataset_decode_nohandler():
    """Test that errors in a custom decoder without handler are raised."""
    count = [0]

    def faulty_decoder(key, data):
        if count[0] % 2 == 0:
            raise ValueError("nothing")
        else:
            return data
        count[0] += 1

    with pytest.raises(wds.DecodingError):
        ds = wds.WebDataset(local_data, shardshuffle=100).decode(faulty_decoder)
        count_samples_tuple(ds)


def test_dataset_missing_totuple_raises():
    """Test that missing keys in to_tuple raise an error."""
    with pytest.raises(ValueError):
        ds = wds.WebDataset(local_data, shardshuffle=100).to_tuple("foo", "bar")
        count_samples_tuple(ds)


def test_dataset_missing_rename_raises():
    """Test that missing keys in rename raise an error."""
    with pytest.raises(ValueError):
        ds = wds.WebDataset(local_data, shardshuffle=100).rename(x="foo", y="bar")
        count_samples_tuple(ds)


def getkeys(sample):
    return set(x for x in sample.keys() if not x.startswith("_"))


def test_dataset_rename_keep():
    """Test the keep option of rename.

    This option determines whether the original keys are kept or only the renamed keys.
    """
    ds = wds.WebDataset(local_data, shardshuffle=100).rename(image="png", keep=False)
    sample = next(iter(ds))
    assert getkeys(sample) == set(["image"]), getkeys(sample)
    ds = wds.WebDataset(local_data, shardshuffle=100).rename(image="png")
    sample = next(iter(ds))
    assert getkeys(sample) == set("cls image wnid xml".split()), getkeys(sample)


def test_dataset_rsample():
    """Test the rsample method.

    The rsample method selects samples from a stream with a given probability."""
    ds = wds.WebDataset(local_data, shardshuffle=100).rsample(1.0)
    assert count_samples_tuple(ds) == 47

    ds = wds.WebDataset(local_data, shardshuffle=100).rsample(0.5)
    result = [count_samples_tuple(ds) for _ in range(300)]
    assert np.mean(result) >= 0.3 * 47 and np.mean(result) <= 0.7 * 47, np.mean(result)


def test_dataset_decode_handler():
    """Test that we can handle a faulty decoder with a handler."""
    count = [0]
    good = [0]

    def faulty_decoder(key, data):
        if "png" not in key:
            return data
        count[0] += 1
        if count[0] % 2 == 0:
            raise ValueError("nothing")
        else:
            good[0] += 1
            return data

    ds = wds.WebDataset(local_data, shardshuffle=100).decode(
        faulty_decoder, handler=wds.ignore_and_continue
    )
    result = count_samples_tuple(ds)
    assert count[0] == 47
    assert good[0] == 24
    assert result == 24


def test_dataset_rename_handler():
    """Test basic rename functionality."""

    ds = wds.WebDataset(local_data, shardshuffle=100).rename(image="png;jpg", cls="cls")
    count_samples_tuple(ds)

    with pytest.raises(ValueError):
        ds = wds.WebDataset(local_data, shardshuffle=100).rename(
            image="missing", cls="cls"
        )
        count_samples_tuple(ds)


def test_dataset_map_handler():
    """Test the map method on a dataset, including error handling."""

    def f(x):
        assert isinstance(x, dict)
        return x

    def g(x):
        raise ValueError()

    ds = wds.WebDataset(local_data, shardshuffle=100).map(f)
    count_samples_tuple(ds)

    with pytest.raises(ValueError):
        ds = wds.WebDataset(local_data, shardshuffle=100).map(g)
        count_samples_tuple(ds)


def test_dataset_map_dict_handler():
    """Test the map_dict method on a dataset, including error handling."""

    ds = wds.WebDataset(local_data, shardshuffle=100).map_dict(
        png=identity, cls=identity
    )
    count_samples_tuple(ds)

    with pytest.raises(KeyError):
        ds = wds.WebDataset(local_data, shardshuffle=100).map_dict(
            png=identity, cls2=identity
        )
        count_samples_tuple(ds)

    def g(x):
        raise ValueError()

    with pytest.raises(ValueError):
        ds = wds.WebDataset(local_data, shardshuffle=100).map_dict(png=g, cls=identity)
        count_samples_tuple(ds)


def test_dataset_shuffle_decode_rename_extract():
    """Test the basic shuffle-decode-rename-to_tuple pipeline."""
    ds = (
        wds.WebDataset(local_data, shardshuffle=100)
        .shuffle(5)
        .decode("rgb")
        .rename(image="png;jpg", cls="cls")
        .to_tuple("image", "cls")
    )
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, np.ndarray), image
    assert isinstance(cls, int), type(cls)


def test_rgb8():
    """Test decoding to RGB8 numpy arrays."""
    ds = (
        wds.WebDataset(local_data, shardshuffle=100)
        .decode("rgb8")
        .to_tuple("png;jpg", "cls")
    )
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, np.ndarray), type(image)
    assert image.dtype == np.uint8, image.dtype
    assert isinstance(cls, int), type(cls)


def test_pil():
    """Test decoding to PIL images."""
    ds = (
        wds.WebDataset(local_data, shardshuffle=100)
        .decode("pil")
        .to_tuple("jpg;png", "cls")
    )
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, PIL.Image.Image)


def test_raw():
    """Test absence of decoding."""
    ds = wds.WebDataset(local_data, shardshuffle=100).to_tuple("jpg;png", "cls")
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, bytes)
    assert isinstance(cls, bytes)


def test_only1():
    """Test partial decoding using the only option to decode."""
    ds = (
        wds.WebDataset(local_data, shardshuffle=100)
        .decode(only="cls")
        .to_tuple("jpg;png", "cls")
    )
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, bytes)
    assert isinstance(cls, int)

    ds = (
        wds.WebDataset(local_data, shardshuffle=100)
        .decode("l", only=["jpg", "png"])
        .to_tuple("jpg;png", "cls")
    )
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, np.ndarray)
    assert isinstance(cls, bytes)


def test_gz():
    """Test chained decoding: txt.gz is first decompressed then decoded."""
    ds = wds.WebDataset(compressed, shardshuffle=100).decode()
    sample = next(iter(ds))
    print(sample)
    assert sample["txt.gz"] == "hello\n", sample


def test_float_np_vs_torch():
    """Compare decoding to numpy and to torch and ensure that they give the same results."""
    ds = (
        wds.WebDataset(local_data, shardshuffle=100)
        .decode("rgb")
        .to_tuple("png;jpg", "cls")
    )
    image, cls = next(iter(ds))
    ds = (
        wds.WebDataset(local_data, shardshuffle=100)
        .decode("torchrgb")
        .to_tuple("png;jpg", "cls")
    )
    image2, cls2 = next(iter(ds))
    assert (image == image2.permute(1, 2, 0).numpy()).all(), (image.shape, image2.shape)
    assert cls == cls2


def test_decoder():
    """Test a custom decoder function."""

    def mydecoder(key, sample):
        return len(sample)

    ds = (
        wds.WebDataset(remote_loc + remote_shard, shardshuffle=100)
        .decode(mydecoder)
        .to_tuple("jpg;png", "json")
    )
    for sample in ds:
        assert isinstance(sample[0], int)
        break


def test_cache_dir(tmp_path):
    """Test a custom decoder function."""

    ds = wds.WebDataset(remote_sample, cache_dir=tmp_path, shardshuffle=100)

    count = 0
    for epoch in range(3):
        for sample in ds:
            assert set(sample.keys()) == set(
                "__key__ __url__ cls __local_path__ png".split()
            )
            assert sample["__key__"] == "10"
            assert sample["cls"] == b"0"
            assert sample["png"].startswith(b"\x89PNG\r\n\x1a\n\x00\x00\x00")
            assert sample["__local_path__"].startswith(str(tmp_path))
            break


def test_shard_syntax():
    """Test that remote shards are correctly handled."""
    print(remote_loc, remote_shards)
    ds = (
        wds.WebDataset(remote_loc + remote_shards, shardshuffle=100)
        .decode()
        .to_tuple("jpg;png", "json")
    )
    assert count_samples_tuple(ds, n=10) == 10


def test_torchvision():
    """Test that torchvision transforms work correctly when used with WebDataset and map_tuple."""
    import torch
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    preproc = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    ds = (
        wds.WebDataset(remote_loc + remote_shards, shardshuffle=100)
        .decode("pil")
        .to_tuple("jpg;png", "json")
        .map_tuple(preproc, identity)
    )
    for sample in ds:
        assert isinstance(sample[0], torch.Tensor), type(sample[0])
        assert tuple(sample[0].size()) == (3, 224, 224), sample[0].size()
        assert isinstance(sample[1], list), type(sample[1])
        break


def test_batched():
    """Test batching with WebDataset and batched(n) method."""
    import torch
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    preproc = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    raw = wds.WebDataset(remote_loc + remote_shards, shardshuffle=100)
    ds = (
        raw.decode("pil")
        .to_tuple("jpg;png", "json")
        .map_tuple(preproc, identity)
        .batched(7)
    )
    for sample in ds:
        assert isinstance(sample[0], torch.Tensor), type(sample[0])
        assert tuple(sample[0].size()) == (7, 3, 224, 224), sample[0].size()
        assert isinstance(sample[1], list), type(sample[1])
        break
    pickle.dumps(ds)


def test_unbatched():
    """Test unbatching with WebDataset and unbatched() method."""
    import torch
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    preproc = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    ds = (
        wds.WebDataset(remote_loc + remote_shards, shardshuffle=100)
        .decode("pil")
        .to_tuple("jpg;png", "json")
        .map_tuple(preproc, identity)
        .batched(7)
        .unbatched()
    )
    for sample in ds:
        assert isinstance(sample[0], torch.Tensor), type(sample[0])
        assert tuple(sample[0].size()) == (3, 224, 224), sample[0].size()
        assert isinstance(sample[1], list), type(sample[1])
        break
    pickle.dumps(ds)


def test_with_epoch():
    """Test the with_epoch(n) method, forcing epochs of a given size."""
    ds = wds.WebDataset(local_data, shardshuffle=100)
    for _ in range(10):
        assert count_samples_tuple(ds) == 47
    be = ds.with_epoch(193)
    for _ in range(10):
        assert count_samples_tuple(be) == 193
    be = ds.with_epoch(2)
    for _ in range(10):
        assert count_samples_tuple(be) == 2


def test_repeat():
    """Test the repeatn(n) method, repeating the dataset n times."""
    ds = wds.WebDataset(local_data, shardshuffle=100)
    assert count_samples_tuple(ds.repeat(nepochs=2)) == 47 * 2


def test_repeat2():
    """Testing the repeat(nbatches=n) method, repeating the dataset n batches."""
    ds = wds.WebDataset(local_data, shardshuffle=100).to_tuple("png", "cls").batched(2)
    assert count_samples_tuple(ds.repeat(nbatches=20)) == 20


def test_mcached():
    shardname = "testdata/imagenet-000000.tgz"
    dataset = wds.DataPipeline(
        wds.SimpleShardList([shardname]),
        wds.tarfile_to_samples(),
        wds.Cached(),
    )
    result1 = list(iter(dataset))
    result2 = list(iter(dataset))
    assert len(result1) == len(result2)


def test_lmdb_cached(tmp_path):
    shardname = "testdata/imagenet-000000.tgz"
    dest = os.path.join(tmp_path, "test.lmdb")
    assert not os.path.exists(dest)
    dataset = wds.DataPipeline(
        wds.SimpleShardList([shardname]),
        wds.tarfile_to_samples(),
        wds.LMDBCached(dest),
    )
    result1 = list(iter(dataset))
    assert os.path.exists(dest)
    result2 = list(iter(dataset))
    assert os.path.exists(dest)
    assert len(result1) == len(result2)
    del dataset
    dataset = wds.DataPipeline(
        wds.SimpleShardList([shardname]),
        wds.tarfile_to_samples(),
        wds.LMDBCached(dest),
    )
    result3 = list(iter(dataset))
    assert len(result1) == len(result3)


def test_shuffle_seed():
    """Test that shuffle is deterministic for a given seed."""

    def make_shuffle_only_ds(seed=0):
        ds = compat.WebDataset(
            "shard-{000000..000999}.tar", shardshuffle=True, seed=seed
        )
        index = ["shuffle" in str(stage) for stage in ds.pipeline].index(True)
        del ds.pipeline[index + 1 :]
        return ds

    ds1 = make_shuffle_only_ds(seed=0)
    ds2 = make_shuffle_only_ds(seed=0)
    assert list(ds1) == list(ds2)

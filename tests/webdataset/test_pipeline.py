import io
import os
import pickle
from io import StringIO
from itertools import islice

import numpy as np
import PIL
import pytest
import yaml

import webdataset as wds
from tests.conftest import (
    compressed,
    count_samples_tuple,
    local_data,
    remote_loc,
    remote_shards,
)
from webdataset import autodecode, handlers, shardlists


def identity(x):
    return x


def test_trivial():
    dataset = wds.DataPipeline(lambda: iter([1, 2, 3, 4]))
    result = list(iter(dataset))
    assert result == [1, 2, 3, 4]


def test_trivial_map():
    dataset = wds.DataPipeline(lambda: iter([1, 2, 3, 4]), wds.map(lambda x: x + 1))
    result = list(iter(dataset))
    assert result == [2, 3, 4, 5]


def test_trivial_map2():
    dataset = wds.DataPipeline(lambda: iter([1, 2, 3, 4]), wds.map(lambda x: x + 1))
    result = list(iter(dataset))
    assert result == [2, 3, 4, 5]


def mymap(src, f):
    for x in src:
        yield f(x)


def adder4(src):
    for x in src:
        yield x + 4


def test_trivial_map4():
    dataset = wds.DataPipeline(
        lambda: iter([1, 2, 3, 4]),
        adder4,
    )
    result = list(iter(dataset))
    assert result == [5, 6, 7, 8]


def test_pytorchshardlist():
    dataset = wds.DataPipeline(
        wds.SimpleShardList("test-{000000..000099}.tar"),
    )
    result = list(iter(dataset))
    assert len(result) == 100


def test_composable():
    dataset = wds.DataPipeline(
        wds.SimpleShardList("test-{000000..000099}.tar"),
    )
    result = list(iter(dataset))
    assert len(result) == 100


def select_png(name):
    return name.endswith(".png")


def test_select_files():
    dataset = wds.DataPipeline(
        wds.shardspec("testdata/imagenet-000000.tgz"),
        wds.tarfile_to_samples(select_files=select_png),
        wds.decode(autodecode.ImageHandler("rgb")),
    )
    result = list(iter(dataset))
    keys = list(result[0].keys())
    assert "__key__" in keys
    assert "__url__" in keys
    assert "cls" not in keys
    assert "png" in keys
    assert isinstance(result[0]["png"], np.ndarray)
    assert result[0]["png"].shape == (793, 600, 3)
    assert len(result) == 47


def rename_cls(name):
    if name.endswith(".cls"):
        return name[:-4] + ".txt"
    return name


def test_rename_files():
    dataset = wds.DataPipeline(
        wds.shardspec("testdata/imagenet-000000.tgz"),
        wds.tarfile_to_samples(rename_files=rename_cls),
        wds.decode(autodecode.ImageHandler("rgb")),
    )
    result = list(iter(dataset))
    keys = list(result[0].keys())
    assert "__key__" in keys
    assert "__url__" in keys
    assert "cls" not in keys
    assert "txt" in keys
    assert "png" in keys
    assert isinstance(result[0]["txt"], str)
    assert int(result[0]["txt"]) >= 0  # also tests format
    assert isinstance(result[0]["png"], np.ndarray)
    assert result[0]["png"].shape == (793, 600, 3)
    assert len(result) == 47


def test_sep():
    dataset = wds.DataPipeline(
        wds.shardspec("testdata/imagenet-000000.tgz::testdata/imagenet-000000.tgz"),
        wds.tarfile_samples,
    )
    result = list(iter(dataset))
    assert len(result) == 47 * 2


def test_reader1():
    dataset = wds.DataPipeline(
        wds.SimpleShardList("testdata/imagenet-000000.tgz"),
        wds.tarfile_samples,
        wds.decode(autodecode.ImageHandler("rgb")),
    )
    result = list(iter(dataset))
    keys = list(result[0].keys())
    assert "__key__" in keys
    assert "__url__" in keys
    assert "cls" in keys
    assert "png" in keys
    assert isinstance(result[0]["cls"], int)
    assert isinstance(result[0]["png"], np.ndarray)
    assert result[0]["png"].shape == (793, 600, 3)
    assert len(result) == 47


def test_reader2():
    dataset = wds.DataPipeline(
        wds.SimpleShardList(["testdata/imagenet-000000.tgz"] * 10),
        wds.shuffle(3),
        wds.tarfile_samples,
        wds.shuffle(100),
        wds.decode(autodecode.ImageHandler("rgb")),
        wds.to_tuple("png", "cls"),
    )
    result = list(iter(dataset))
    assert len(result[0]) == 2
    assert isinstance(result[0][0], np.ndarray)
    assert isinstance(result[0][1], int)
    assert len(result) == 470


def test_reader3():
    dataset = wds.DataPipeline(
        wds.SimpleShardList(["testdata/imagenet-000000.tgz"] * 3),
        wds.resampled(10),
        wds.tarfile_samples,
        wds.shuffle(100),
        wds.decode(autodecode.ImageHandler("rgb")),
        wds.to_tuple("png", "cls"),
    )
    result = list(iter(dataset))
    assert len(result[0]) == 2
    assert isinstance(result[0][0], np.ndarray)
    assert isinstance(result[0][1], int)
    assert len(result) == 470
    result = list(iter(dataset))
    assert len(result[0]) == 2
    assert isinstance(result[0][0], np.ndarray)
    assert isinstance(result[0][1], int)
    assert len(result) == 470


def test_splitting():
    dataset = wds.DataPipeline(
        wds.SimpleShardList(list(map(str, range(10)))),
        wds.split_by_node,
        wds.split_by_worker,
    )
    result = list(iter(dataset))
    assert len(result) == 10
    assert result[0]["url"] == "0"


def test_seed():
    dataset = wds.DataPipeline(
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
    dataset = wds.DataPipeline(
        wds.SimpleShardList(list(map(str, range(10)))),
        wds.split_by_node,
        wds.split_by_worker,
        wds.non_empty,
    )
    result = list(iter(dataset))
    assert len(result) == 10
    assert result[0]["url"] == "0"


def test_nonempty2():
    dataset = wds.DataPipeline(
        wds.SimpleShardList(list(map(str, range(10)))),
        lambda src: iter([]),
        wds.non_empty,
    )
    with pytest.raises(ValueError):
        list(iter(dataset))


def test_resampled():
    dataset = wds.DataPipeline(
        wds.SimpleShardList(list(map(str, range(10)))),
        wds.resampled(27),
    )
    result = list(iter(dataset))
    assert len(result) == 27


def test_slice():
    dataset = wds.DataPipeline(
        wds.SimpleShardList(list(map(str, range(200)))),
        wds.slice(29),
    )
    result = list(iter(dataset))
    assert len(result) == 29


def count_samples(source, *args, n=1000):
    count = 0
    for i, sample in enumerate(iter(source)):
        if i >= n:
            break
        for f in args:
            assert f(sample)
        count += 1
    return count


def test_dataset():
    ds = wds.DataPipeline(wds.SimpleShardList(local_data), wds.tarfile_to_samples())
    assert count_samples_tuple(ds) == 47


def test_expandvars():
    os.environ["WDS_TESTDATA"] = "testdata"
    with_var = "${TESTDATA}/imagenet-000000.tgz"
    ds = wds.DataPipeline(wds.SimpleShardList(with_var), wds.tarfile_to_samples())
    assert count_samples_tuple(ds) == 47
    del os.environ["WDS_TESTDATA"]


def test_dataset_resampled():
    ds = wds.DataPipeline(wds.ResampledShards(local_data), wds.tarfile_to_samples())
    assert count_samples_tuple(ds, n=100) == 100


def test_mock():
    ds = wds.MockDataset((True, True), 193)
    assert count_samples_tuple(ds) == 193


def test_dataset_shuffle_extract():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.to_tuple("png;jpg cls"),
    )
    assert count_samples_tuple(ds) == 47


def test_dataset_pipe_cat():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.shuffle(5),
        wds.to_tuple("png;jpg cls"),
    )
    assert count_samples_tuple(ds) == 47


def test_dataset_extract_keys():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.shuffle(5),
        wds.extract_keys("*.png;*.jpg", "*.cls"),
    )
    assert count_samples_tuple(ds) == 47


def test_slice():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data), wds.tarfile_to_samples(), wds.slice(10)
    )
    assert count_samples_tuple(ds) == 10


def test_dataset_eof():
    import tarfile

    with pytest.raises(tarfile.ReadError):
        ds = wds.DataPipeline(
            wds.SimpleShardList(f"pipe:dd if={local_data} bs=1024 count=10"),
            wds.tarfile_to_samples(),
            wds.shuffle(5),
        )
        assert count_samples(ds) == 47


def test_dataset_eof_handler():
    ds = wds.DataPipeline(
        wds.SimpleShardList(f"pipe:dd if={local_data} bs=1024 count=10"),
        wds.tarfile_to_samples(handler=handlers.ignore_and_stop),
        wds.shuffle(5),
    )
    assert count_samples(ds) < 47


def test_dataset_decode_nohandler():
    count = [0]

    def faulty_decoder(key, data):
        if count[0] % 2 == 0:
            raise ValueError("nothing")
        else:
            return data
        count[0] += 1

    with pytest.raises(wds.DecodingError):
        ds = wds.DataPipeline(
            wds.SimpleShardList(local_data),
            wds.tarfile_to_samples(),
            wds.decode(faulty_decoder),
        )
        count_samples_tuple(ds)


def test_dataset_missing_totuple_raises():
    with pytest.raises(ValueError):
        ds = wds.DataPipeline(
            wds.SimpleShardList(f"pipe:dd if={local_data} bs=1024 count=10"),
            wds.tarfile_to_samples(handler=handlers.ignore_and_stop),
            wds.to_tuple("foo", "bar"),
        )
        count_samples_tuple(ds)


def test_dataset_missing_rename_raises():
    with pytest.raises(ValueError):
        ds = wds.DataPipeline(
            wds.SimpleShardList(f"pipe:dd if={local_data} bs=1024 count=10"),
            wds.tarfile_to_samples(handler=handlers.ignore_and_stop),
            wds.rename(x="foo", y="bar"),
        )
        count_samples_tuple(ds)


def getkeys(sample):
    return set(x for x in sample.keys() if not x.startswith("_"))


def test_dataset_rename_keep():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.rename(image="png", keep=False),
    )
    sample = next(iter(ds))
    assert getkeys(sample) == set(["image"]), getkeys(sample)
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.rename(image="png"),
    )
    sample = next(iter(ds))
    assert getkeys(sample) == set("cls image wnid xml".split()), getkeys(sample)


def test_dataset_rename_keys():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.rename_keys(image="png", keep_unselected=True),
    )
    sample = next(iter(ds))
    assert getkeys(sample) == set("cls image wnid xml".split()), getkeys(sample)


def test_dataset_rsample():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data), wds.tarfile_to_samples(), wds.rsample(1.0)
    )
    assert count_samples_tuple(ds) == 47

    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data), wds.tarfile_to_samples(), wds.rsample(0.5)
    )
    result = [count_samples_tuple(ds) for _ in range(300)]
    assert np.mean(result) >= 0.3 * 47 and np.mean(result) <= 0.7 * 47, np.mean(result)


def test_dataset_decode_handler():
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

    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.decode(faulty_decoder, handler=handlers.ignore_and_continue),
    )
    result = count_samples_tuple(ds)
    assert count[0] == 47
    assert good[0] == 24
    assert result == 24


def test_dataset_map():
    def f(x):
        assert isinstance(x, dict)
        return x

    def g(x):
        raise ValueError()

    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.map(f),
    )
    count_samples_tuple(ds)

    with pytest.raises(ValueError):
        ds = wds.DataPipeline(
            wds.SimpleShardList(local_data),
            wds.tarfile_to_samples(),
            wds.map(g),
        )
        count_samples_tuple(ds)


def test_dataset_map_dict_handler():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.map_dict(png=identity, cls=identity),
    )
    count_samples_tuple(ds)

    with pytest.raises(KeyError):
        ds = wds.DataPipeline(
            wds.SimpleShardList(local_data),
            wds.tarfile_to_samples(),
            wds.map_dict(png=identity, cls2=identity),
        )
        count_samples_tuple(ds)

    def g(x):
        raise ValueError()

    with pytest.raises(ValueError):
        ds = wds.DataPipeline(
            wds.SimpleShardList(local_data),
            wds.tarfile_to_samples(),
            wds.map_dict(png=g, cls2=identity),
        )
        count_samples_tuple(ds)


def test_dataset_shuffle_decode_rename_extract():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.shuffle(5),
        wds.decode("rgb"),
        wds.rename(image="png;jpg", cls="cls"),
        wds.to_tuple("image", "cls"),
    )
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, np.ndarray), image
    assert isinstance(cls, int), type(cls)


def test_rgb8():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.shuffle(5),
        wds.decode("rgb8"),
        wds.rename(image="png;jpg", cls="cls"),
        wds.to_tuple("image", "cls"),
    )
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, np.ndarray), type(image)
    assert image.dtype == np.uint8, image.dtype
    assert isinstance(cls, int), type(cls)


def test_pil():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.shuffle(5),
        wds.decode("pil"),
        wds.rename(image="png;jpg", cls="cls"),
        wds.to_tuple("image", "cls"),
    )
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, PIL.Image.Image)


def test_raw():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.rename(image="png;jpg", cls="cls"),
        wds.to_tuple("image", "cls"),
    )
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, bytes)
    assert isinstance(cls, bytes)


def IGNORE_test_only1():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.decode(only="cls"),
        wds.to_tuple("png;jpg", "cls"),
    )
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, bytes)
    assert isinstance(cls, int)

    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.decode(only=["png", "jpg"]),
        wds.to_tuple("jpg;png", "cls"),
    )
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, np.ndarray)
    assert isinstance(cls, bytes)


def test_gz():
    ds = wds.DataPipeline(
        wds.SimpleShardList(compressed),
        wds.tarfile_to_samples(),
        wds.decode(),
    )
    sample = next(iter(ds))
    print(sample)
    assert sample["txt.gz"] == "hello\n", sample
    assert "__url__" in sample, sample.keys()


def test_float_np_vs_torch():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.decode("rgb"),
        wds.to_tuple("png;jpg", "cls"),
    )
    image, cls = next(iter(ds))
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.decode("torchrgb"),
        wds.to_tuple("png;jpg", "cls"),
    )
    image2, cls2 = next(iter(ds))
    assert (image == image2.permute(1, 2, 0).numpy()).all(), (image.shape, image2.shape)
    assert cls == cls2


def test_tenbin():
    from webdataset import tenbin

    for d0 in [0, 1, 2, 10, 100, 1777]:
        for d1 in [0, 1, 2, 10, 100, 345]:
            for t in [np.uint8, np.float16, np.float32, np.float64]:
                a = np.random.normal(size=(d0, d1)).astype(t)
                a_encoded = tenbin.encode_buffer([a])
                (a_decoded,) = tenbin.decode_buffer(a_encoded)
                print(a.shape, a_decoded.shape)
                assert a.shape == a_decoded.shape
                assert a.dtype == a_decoded.dtype
                assert (a == a_decoded).all()


def test_tenbin_dec():
    ds = wds.DataPipeline(
        wds.SimpleShardList("testdata/tendata.tar"),
        wds.tarfile_to_samples(),
        wds.decode(),
        wds.to_tuple("ten"),
    )
    assert count_samples_tuple(ds) == 100
    for sample in ds:
        xs, ys = sample[0]
        assert xs.dtype == np.float64
        assert ys.dtype == np.float64
        assert xs.shape == (28, 28)
        assert ys.shape == (28, 28)


def test_dataloader():
    import torch

    ds = wds.DataPipeline(
        wds.SimpleShardList(remote_loc + remote_shards),
        wds.tarfile_to_samples(),
        wds.decode("torchrgb"),
        wds.to_tuple("jpg;png", "json"),
    )
    dl = torch.utils.data.DataLoader(ds, num_workers=4)
    assert count_samples_tuple(dl, n=100) == 100


def test_resampled_initialization():
    shardlist = shardlists.ResampledShards([str(i) for i in range(100000)])
    list1 = [x for x in islice(shardlist, 0, 10)]
    list2 = [x for x in islice(shardlist, 0, 10)]
    assert list1 != list2
    shardlist2 = shardlists.ResampledShards([str(i) for i in range(100000)])
    list3 = [x for x in islice(shardlist2, 0, 10)]
    assert list1 != list3


def test_decode_handlers():
    def mydecoder(data):
        return PIL.Image.open(io.BytesIO(data)).resize((128, 128))

    ds = wds.DataPipeline(
        wds.SimpleShardList(remote_loc + remote_shards),
        wds.tarfile_to_samples(),
        wds.decode(
            wds.handle_extension("jpg", mydecoder),
            wds.handle_extension("png", mydecoder),
        ),
        wds.to_tuple("jpg;png", "json"),
    )

    for sample in ds:
        assert isinstance(sample[0], PIL.Image.Image)
        break


def test_decoder():
    def mydecoder(key, sample):
        return len(sample)

    ds = wds.DataPipeline(
        wds.SimpleShardList(remote_loc + remote_shards),
        wds.tarfile_to_samples(),
        wds.decode(mydecoder),
        wds.to_tuple("jpg;png", "json"),
    )

    for sample in ds:
        assert isinstance(sample[0], int)
        break


def test_pipe():
    ds = wds.DataPipeline(
        wds.SimpleShardList(f"pipe:curl -s '{remote_loc}{remote_shards}' || true"),
        wds.tarfile_to_samples(),
        wds.shuffle(100),
        wds.to_tuple("jpg;png", "json"),
    )
    assert count_samples_tuple(ds, n=10) == 10


def test_torchvision():
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
    ds = wds.DataPipeline(
        wds.SimpleShardList(f"{remote_loc}{remote_shards}"),
        wds.tarfile_to_samples(),
        wds.decode("pil"),
        wds.to_tuple("jpg;png", "json"),
        wds.map_tuple(preproc, None),
    )
    for sample in ds:
        assert isinstance(sample[0], torch.Tensor), type(sample[0])
        assert tuple(sample[0].size()) == (3, 224, 224), sample[0].size()
        assert isinstance(sample[1], list), type(sample[1])
        break


def test_batched():
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
    ds = wds.DataPipeline(
        wds.SimpleShardList(f"{remote_loc}{remote_shards}"),
        wds.tarfile_to_samples(),
        wds.decode("pil"),
        wds.to_tuple("jpg;png", "json"),
        wds.map_tuple(preproc, None),
        wds.batched(7),
    )
    for sample in ds:
        assert isinstance(sample[0], torch.Tensor), type(sample[0])
        assert tuple(sample[0].size()) == (7, 3, 224, 224), sample[0].size()
        assert isinstance(sample[1], list), type(sample[1])
        break
    # make sure the batched dataset can be pickled
    pickle.dumps(ds)


def test_unbatched():
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
    ds = wds.DataPipeline(
        wds.SimpleShardList(remote_loc + remote_shards),
        wds.tarfile_to_samples(),
        wds.decode("pil"),
        wds.to_tuple("jpg;png", "json"),
        wds.map_tuple(preproc, None),
        wds.batched(7),
        wds.unbatched(),
    )
    for sample in ds:
        assert isinstance(sample[0], torch.Tensor), type(sample[0])
        assert tuple(sample[0].size()) == (3, 224, 224), sample[0].size()
        assert isinstance(sample[1], list), type(sample[1])
        break
    # make sure the batched dataset can be pickled
    pickle.dumps(ds)


def test_with_epoch():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
    )
    for _ in range(10):
        assert count_samples_tuple(ds) == 47
    be = ds.with_epoch(193)
    for _ in range(10):
        assert count_samples_tuple(be) == 193
    be = ds.with_epoch(2)
    for _ in range(10):
        assert count_samples_tuple(be) == 2


def test_repeat():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
    )
    ds = ds.repeat(nepochs=2)
    assert count_samples_tuple(ds) == 47 * 2


def test_repeat2():
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.tarfile_to_samples(),
        wds.to_tuple("cls"),
        wds.batched(2),
    )
    ds = ds.with_epoch(20)
    assert count_samples_tuple(ds) == 20


def test_tenbin():
    """Test tensor binary encoding."""
    from webdataset import tenbin

    for d0 in [0, 1, 2, 10, 100, 1777]:
        for d1 in [0, 1, 2, 10, 100, 345]:
            for t in [np.uint8, np.float16, np.float32, np.float64]:
                a = np.random.normal(size=(d0, d1)).astype(t)
                a_encoded = tenbin.encode_buffer([a])
                (a_decoded,) = tenbin.decode_buffer(a_encoded)
                print(a.shape, a_decoded.shape)
                assert a.shape == a_decoded.shape
                assert a.dtype == a_decoded.dtype
                assert (a == a_decoded).all()


def test_tenbin_dec():
    """Test tensor binary decoding."""
    ds = (
        wds.WebDataset("testdata/tendata.tar", shardshuffle=100)
        .decode()
        .to_tuple("ten")
    )
    assert count_samples_tuple(ds) == 100
    for sample in ds:
        xs, ys = sample[0]
        assert xs.dtype == np.float64
        assert ys.dtype == np.float64
        assert xs.shape == (28, 28)
        assert ys.shape == (28, 28)

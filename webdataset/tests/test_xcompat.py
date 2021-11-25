import io
import os
import numpy as np
import PIL
import pytest
import torch
import pickle
import yaml
from io import StringIO

import webdataset as wds


local_data = "testdata/imagenet-000000.tgz"
compressed = "testdata/compressed.tar"
remote_loc = "http://storage.googleapis.com/nvdata-openimages/"
remote_shards = "openimages-train-0000{00..99}.tar"
remote_shard = "openimages-train-000321.tar"
remote_pattern = "openimages-train-{}.tar"


def identity(x):
    return x


def count_samples_tuple(source, *args, n=10000):
    count = 0
    for i, sample in enumerate(iter(source)):
        if i >= n:
            break
        assert isinstance(sample, (tuple, dict, list)), (type(sample), sample)
        for f in args:
            assert f(sample)
        count += 1
    return count


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
    ds = wds.WebDataset(local_data)
    assert count_samples_tuple(ds) == 47


def test_dataset_resampled():
    ds = wds.WebDataset(local_data, resampled=True)
    assert count_samples_tuple(ds, n=100) == 100


shardspec = """
datasets:

  - name: CDIP
    perepoch: 10
    buckets:
      - ./gs/nvdata-ocropus/words/
    shards:
    - cdipsub-{000000..000092}.tar

  - name: Google 1000 Books
    perepoch: 20
    buckets:
      - ./gs/nvdata-ocropus/words/
    shards:
      - gsub-{000000..000167}.tar

  - name: Internet Archive Sample
    perepoch: 30
    buckets:
      - ./gs/nvdata-ocropus/words/
    shards:
      - ia1-{000000..000033}.tar
"""


yaml3_data = """
prefix: pipe:curl -s -L http://storage.googleapis.com/
buckets: nvdata-ocropus-words
datasets:
  - shards: uw3-word-0000{00..21}.tar
  - shards: ia1-{000000..000033}.tar
  - shards: gsub-{000000..000167}.tar
  - shards: cdipsub-{000000..000092}.tar
"""


def test_yaml3():
    spec = yaml.safe_load(StringIO(yaml3_data))
    ds = wds.WebDataset(spec)
    next(iter(ds))


def test_length():
    ds = wds.WebDataset(local_data)
    with pytest.raises(TypeError):
        len(ds)
    dsl = ds.with_length(1793)
    assert len(dsl) == 1793
    dsl2 = ds.repeat(17)
    dsl3 = dsl2.with_length(19)
    assert len(dsl3) == 19


def test_mock():
    ds = wds.MockDataset((True, True), 193)
    assert count_samples_tuple(ds) == 193


def IGNORE_test_ddp_equalize():
    ds = wds.WebDataset(local_data).ddp_equalize(733)
    assert count_samples_tuple(ds) == 733


def test_dataset_shuffle_extract():
    ds = wds.WebDataset(local_data).shuffle(5).to_tuple("png;jpg cls")
    assert count_samples_tuple(ds) == 47


def test_dataset_pipe_cat():
    ds = wds.WebDataset(f"pipe:cat {local_data}").shuffle(5).to_tuple("png;jpg cls")
    assert count_samples_tuple(ds) == 47


def test_slice():
    ds = wds.WebDataset(local_data).slice(10)
    assert count_samples_tuple(ds) == 10


def test_dataset_eof():
    import tarfile

    with pytest.raises(tarfile.ReadError):
        ds = wds.WebDataset(f"pipe:dd if={local_data} bs=1024 count=10").shuffle(5)
        assert count_samples(ds) == 47


def test_dataset_eof_handler():
    ds = wds.WebDataset(f"pipe:dd if={local_data} bs=1024 count=10", handler=wds.ignore_and_stop)
    assert count_samples(ds) < 47


def test_dataset_decode_nohandler():
    count = [0]

    def faulty_decoder(key, data):
        if count[0] % 2 == 0:
            raise ValueError("nothing")
        else:
            return data
        count[0] += 1

    with pytest.raises(ValueError):
        ds = wds.WebDataset(local_data).decode(faulty_decoder)
        count_samples_tuple(ds)


def test_dataset_missing_totuple_raises():
    with pytest.raises(ValueError):
        ds = wds.WebDataset(local_data).to_tuple("foo", "bar")
        count_samples_tuple(ds)


def test_dataset_missing_rename_raises():
    with pytest.raises(ValueError):
        ds = wds.WebDataset(local_data).rename(x="foo", y="bar")
        count_samples_tuple(ds)


def getkeys(sample):
    return set(x for x in sample.keys() if not x.startswith("_"))


def test_dataset_rename_keep():
    ds = wds.WebDataset(local_data).rename(image="png", keep=False)
    sample = next(iter(ds))
    assert getkeys(sample) == set(["image"]), getkeys(sample)
    ds = wds.WebDataset(local_data).rename(image="png")
    sample = next(iter(ds))
    assert getkeys(sample) == set("cls image wnid xml".split()), getkeys(sample)


def test_dataset_rsample():

    ds = wds.WebDataset(local_data).rsample(1.0)
    assert count_samples_tuple(ds) == 47

    ds = wds.WebDataset(local_data).rsample(0.5)
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

    ds = wds.WebDataset(local_data).decode(faulty_decoder, handler=wds.ignore_and_continue)
    result = count_samples_tuple(ds)
    assert count[0] == 47
    assert good[0] == 24
    assert result == 24


def test_dataset_rename_handler():

    ds = wds.WebDataset(local_data).rename(image="png;jpg", cls="cls")
    count_samples_tuple(ds)

    with pytest.raises(ValueError):
        ds = wds.WebDataset(local_data).rename(image="missing", cls="cls")
        count_samples_tuple(ds)


def test_dataset_map_handler():
    def f(x):
        assert isinstance(x, dict)
        return x

    def g(x):
        raise ValueError()

    ds = wds.WebDataset(local_data).map(f)
    count_samples_tuple(ds)

    with pytest.raises(ValueError):
        ds = wds.WebDataset(local_data).map(g)
        count_samples_tuple(ds)


def test_dataset_map_dict_handler():

    ds = wds.WebDataset(local_data).map_dict(png=identity, cls=identity)
    count_samples_tuple(ds)

    with pytest.raises(KeyError):
        ds = wds.WebDataset(local_data).map_dict(png=identity, cls2=identity)
        count_samples_tuple(ds)

    def g(x):
        raise ValueError()

    with pytest.raises(ValueError):
        ds = wds.WebDataset(local_data).map_dict(png=g, cls=identity)
        count_samples_tuple(ds)


def test_dataset_shuffle_decode_rename_extract():
    ds = (
        wds.WebDataset(local_data)
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
    ds = wds.WebDataset(local_data).decode("rgb8").to_tuple("png;jpg", "cls")
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, np.ndarray), type(image)
    assert image.dtype == np.uint8, image.dtype
    assert isinstance(cls, int), type(cls)


def test_pil():
    ds = wds.WebDataset(local_data).decode("pil").to_tuple("jpg;png", "cls")
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, PIL.Image.Image)


def test_raw():
    ds = wds.WebDataset(local_data).to_tuple("jpg;png", "cls")
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, bytes)
    assert isinstance(cls, bytes)


def test_only1():
    ds = wds.WebDataset(local_data).decode(only="cls").to_tuple("jpg;png", "cls")
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, bytes)
    assert isinstance(cls, int)

    ds = wds.WebDataset(local_data).decode("l", only=["jpg", "png"]).to_tuple("jpg;png", "cls")
    assert count_samples_tuple(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, np.ndarray)
    assert isinstance(cls, bytes)


def test_gz():
    ds = wds.WebDataset(compressed).decode()
    sample = next(iter(ds))
    print(sample)
    assert sample["txt.gz"] == "hello\n", sample


@pytest.mark.skip(reason="need to figure out unraisableexceptionwarning")
def test_rgb8_np_vs_torch():
    import warnings

    warnings.filterwarnings("error")
    ds = wds.WebDataset(local_data).decode("rgb8").to_tuple("png;jpg", "cls")
    image, cls = next(iter(ds))
    assert isinstance(image, np.ndarray), type(image)
    assert isinstance(cls, int), type(cls)
    ds = wds.WebDataset(local_data).decode("torchrgb8").to_tuple("png;jpg", "cls")
    image2, cls2 = next(iter(ds))
    assert isinstance(image2, torch.Tensor), type(image2)
    assert isinstance(cls, int), type(cls)
    assert (image == image2.permute(1, 2, 0).numpy()).all, (image.shape, image2.shape)
    assert cls == cls2


def test_float_np_vs_torch():
    ds = wds.WebDataset(local_data).decode("rgb").to_tuple("png;jpg", "cls")
    image, cls = next(iter(ds))
    ds = wds.WebDataset(local_data).decode("torchrgb").to_tuple("png;jpg", "cls")
    image2, cls2 = next(iter(ds))
    assert (image == image2.permute(1, 2, 0).numpy()).all(), (image.shape, image2.shape)
    assert cls == cls2


# def test_associate():
#     with open("testdata/imagenet-extra.json") as stream:
#         extra_data = simplejson.load(stream)

#     def associate(key):
#         return dict(MY_EXTRA_DATA=extra_data[key])

#     ds = wds.WebDataset(local_data).associate(associate)

#     for sample in ds:
#         assert "MY_EXTRA_DATA" in sample.keys()
#         break


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
    ds = wds.WebDataset("testdata/tendata.tar").decode().to_tuple("ten")
    assert count_samples_tuple(ds) == 100
    for sample in ds:
        xs, ys = sample[0]
        assert xs.dtype == np.float64
        assert ys.dtype == np.float64
        assert xs.shape == (28, 28)
        assert ys.shape == (28, 28)


# def test_container_mp():
#     ds = wds.WebDataset("testdata/mpdata.tar", container="mp", decoder=None)
#     assert count_samples_tuple(ds) == 100
#     for sample in ds:
#         assert isinstance(sample, dict)
#         assert set(sample.keys()) == set("__key__ x y".split()), sample


# def test_container_ten():
#     ds = wds.WebDataset("testdata/tendata.tar", container="ten", decoder=None)
#     assert count_samples_tuple(ds) == 100
#     for xs, ys in ds:
#         assert xs.dtype == np.float64
#         assert ys.dtype == np.float64
#         assert xs.shape == (28, 28)
#         assert ys.shape == (28, 28)


def test_dataloader():
    import torch

    ds = wds.WebDataset(remote_loc + remote_shards)
    dl = torch.utils.data.DataLoader(ds, num_workers=4)
    assert count_samples_tuple(dl, n=100) == 100


def test_handlers():
    def mydecoder(data):
        return PIL.Image.open(io.BytesIO(data)).resize((128, 128))

    ds = (
        wds.WebDataset(remote_loc + remote_shard)
        .decode(
            wds.handle_extension("jpg", mydecoder),
            wds.handle_extension("png", mydecoder),
        )
        .to_tuple("jpg;png", "json")
    )

    for sample in ds:
        assert isinstance(sample[0], PIL.Image.Image)
        break


def test_decoder():
    def mydecoder(key, sample):
        return len(sample)

    ds = wds.WebDataset(remote_loc + remote_shard).decode(mydecoder).to_tuple("jpg;png", "json")
    for sample in ds:
        assert isinstance(sample[0], int)
        break


def test_shard_syntax():
    print(remote_loc, remote_shards)
    ds = wds.WebDataset(remote_loc + remote_shards).decode().to_tuple("jpg;png", "json")
    assert count_samples_tuple(ds, n=10) == 10


# def test_opener():
#     def opener(url):
#         print(url, file=sys.stderr)
#         cmd = "curl -s '{}{}'".format(remote_loc, remote_pattern.format(url))
#         print(cmd, file=sys.stderr)
#         return subprocess.Popen(
#             cmd, bufsize=1000000, shell=True, stdout=subprocess.PIPE
#         ).stdout
#
#     ds = (
#         wds.WebDataset("{000000..000099}", open_fn=opener)
#         .shuffle(100)
#         .to_tuple("jpg;png", "json")
#     )
#     assert count_samples_tuple(ds, n=10) == 10


def test_pipe():
    ds = (
        wds.WebDataset(f"pipe:curl -s '{remote_loc}{remote_shards}'").shuffle(100).to_tuple("jpg;png", "json")
    )
    assert count_samples_tuple(ds, n=10) == 10


def test_torchvision():
    import torch
    from torchvision import transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preproc = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    ds = (
        wds.WebDataset(remote_loc + remote_shards)
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
    import torch
    from torchvision import transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preproc = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    raw = wds.WebDataset(remote_loc + remote_shards)
    ds = raw.decode("pil").to_tuple("jpg;png", "json").map_tuple(preproc, identity).batched(7)
    for sample in ds:
        assert isinstance(sample[0], torch.Tensor), type(sample[0])
        assert tuple(sample[0].size()) == (7, 3, 224, 224), sample[0].size()
        assert isinstance(sample[1], list), type(sample[1])
        break
    pickle.dumps(ds)


def test_unbatched():
    import torch
    from torchvision import transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preproc = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    ds = (
        wds.WebDataset(remote_loc + remote_shards)
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
    ds = wds.WebDataset(local_data)
    for _ in range(10):
        assert count_samples_tuple(ds) == 47
    be = ds.with_epoch(193)
    for _ in range(10):
        assert count_samples_tuple(be) == 193
    be = ds.with_epoch(2)
    for _ in range(10):
        assert count_samples_tuple(be) == 2


def test_repeat():
    ds = wds.WebDataset(local_data)
    assert count_samples_tuple(ds.repeat(nepochs=2)) == 47 * 2


def test_repeat2():
    ds = wds.WebDataset(local_data).to_tuple("png", "cls").batched(2)
    assert count_samples_tuple(ds.repeat(nbatches=20)) == 20


def test_webloader():
    ds = wds.WebDataset(local_data)
    dl = wds.WebLoader(ds, num_workers=4, batch_size=3)
    nsamples = count_samples_tuple(dl)
    assert nsamples == (47 + 2) // 3, nsamples


def test_webloader_repeat():
    ds = wds.WebDataset(local_data)
    dl = wds.WebLoader(ds, num_workers=4, batch_size=3).repeat(nepochs=2)
    nsamples = count_samples_tuple(dl)
    assert nsamples == 2 * (47 + 2) // 3, nsamples


def test_webloader_unbatched():
    ds = wds.WebDataset(local_data).to_tuple("png", "cls")
    dl = wds.WebLoader(ds, num_workers=4, batch_size=3).unbatched()
    nsamples = count_samples_tuple(dl)
    assert nsamples == 47, nsamples

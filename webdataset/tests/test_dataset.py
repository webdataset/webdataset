import io
import subprocess
import sys

import numpy as np
import PIL
import simplejson

import webdataset.dataset as wds
from webdataset import autodecode
import torch

local_data = "testdata/imagenet-000000.tgz"
remote_loc = "http://storage.googleapis.com/lpr-imagenet/"


def count_samples(source, *args, n=1000):
    count = 0
    for i, sample in enumerate(iter(source)):
        if i >= n:
            break
        assert isinstance(sample, (tuple, dict, list)), (type(sample), sample)
        for f in args:
            assert f(sample)
        count += 1
    return count


def test_dataset_nogrouping():
    ds = wds.Dataset(local_data, initial_pipeline=[])
    assert count_samples(ds) == 188


def test_dataset():
    ds = wds.Dataset(local_data)
    assert count_samples(ds) == 47


def test_dataset_shuffle_extract():
    ds = wds.Dataset(local_data).shuffle(5).extract("png;jpg cls")
    assert count_samples(ds) == 47


def test_dataset_shuffle_decode_rename_extract():
    ds = (
        wds.Dataset(local_data)
        .shuffle(5)
        .decode("rgb")
        .rename(image="png;jpg", cls="cls")
        .extract("image", "cls")
    )
    assert count_samples(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, np.ndarray), image
    assert isinstance(cls, int)


def test_dataset_len():
    ds = wds.Dataset(local_data, length=100)
    assert len(ds) == 100


def test_rgb8():
    ds = wds.Dataset(local_data).decode("rgb8").extract("png;jpg", "cls")
    assert count_samples(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, np.ndarray), type(image)
    assert image.dtype == np.uint8, image.dtype
    assert isinstance(cls, int), type(cls)


def test_pil():
    ds = wds.Dataset(local_data).decode("pil").extract("jpg;png", "cls")
    assert count_samples(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, PIL.Image.Image)


def test_raw():
    ds = wds.Dataset(local_data).extract("jpg;png", "cls")
    assert count_samples(ds) == 47
    image, cls = next(iter(ds))
    assert isinstance(image, bytes)
    assert isinstance(cls, bytes)


def test_rgb8_np_vs_torch():
    ds = wds.Dataset(local_data).decode("rgb8").extract("png;jpg", "cls")
    image, cls = next(iter(ds))
    assert isinstance(image, np.ndarray), type(image)
    assert isinstance(cls, int), type(cls)
    ds = wds.Dataset(local_data).decode("torchrgb8").extract("png;jpg", "cls")
    image2, cls2 = next(iter(ds))
    assert isinstance(image2, torch.Tensor), type(image2)
    assert isinstance(cls, int), type(cls)
    assert (image == image2.permute(1, 2, 0).numpy()).all, (image.shape, image2.shape)
    assert cls == cls2


def test_float_np_vs_torch():
    ds = wds.Dataset(local_data).decode("rgb").extract("png;jpg", "cls")
    image, cls = next(iter(ds))
    ds = wds.Dataset(local_data).decode("torchrgb").extract("png;jpg", "cls")
    image2, cls2 = next(iter(ds))
    assert (image == image2.permute(1, 2, 0).numpy()).all(), (image.shape, image2.shape)
    assert cls == cls2


def test_associate():
    with open("testdata/imagenet-extra.json") as stream:
        extra_data = simplejson.load(stream)

    def associate(key):
        return dict(MY_EXTRA_DATA=extra_data[key])

    ds = wds.Dataset(local_data).associate(associate)

    for sample in ds:
        assert "MY_EXTRA_DATA" in sample.keys()
        break


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
    ds = wds.Dataset("testdata/tendata.tar").decode().extract("ten")
    assert count_samples(ds) == 100
    for sample in ds:
        xs, ys = sample[0]
        assert xs.dtype == np.float64
        assert ys.dtype == np.float64
        assert xs.shape == (28, 28)
        assert ys.shape == (28, 28)


# def test_container_mp():
#     ds = wds.WebDataset("testdata/mpdata.tar", container="mp", decoder=None)
#     assert count_samples(ds) == 100
#     for sample in ds:
#         assert isinstance(sample, dict)
#         assert set(sample.keys()) == set("__key__ x y".split()), sample


# def test_container_ten():
#     ds = wds.WebDataset("testdata/tendata.tar", container="ten", decoder=None)
#     assert count_samples(ds) == 100
#     for xs, ys in ds:
#         assert xs.dtype == np.float64
#         assert ys.dtype == np.float64
#         assert xs.shape == (28, 28)
#         assert ys.shape == (28, 28)


def test_dataloader():
    import torch

    ds = wds.Dataset(remote_loc + "imagenet_train-{0000..0147}.tgz")
    dl = torch.utils.data.DataLoader(ds, num_workers=4)
    assert count_samples(dl, n=100) == 100


def test_handlers():
    handlers = dict(autodecode.default_handlers["rgb"])

    def decode_jpg_and_resize(data):
        return PIL.Image.open(io.BytesIO(data)).resize((128, 128))

    handlers["jpg"] = decode_jpg_and_resize

    ds = (
        wds.Dataset(remote_loc + "imagenet_train-0050.tgz")
        .decode(handlers)
        .extract("jpg;png", "cls")
    )

    for sample in ds:
        assert isinstance(sample[0], PIL.Image.Image)
        break


def test_decoder():
    def mydecoder(sample):
        return {k: len(v) for k, v in sample.items()}

    ds = (
        wds.Dataset(remote_loc + "imagenet_train-0050.tgz")
        .decode(mydecoder)
        .extract("jpg;png", "cls")
    )
    for sample in ds:
        assert isinstance(sample[0], int)
        break


def test_shard_syntax():
    ds = (
        wds.Dataset(remote_loc + "imagenet_train-{0000..0147}.tgz")
        .decode()
        .extract("jpg;png", "cls")
        .shuffle(0)
    )
    assert count_samples(ds, n=10) == 10


def test_opener():
    def opener(url):
        print(url, file=sys.stderr)
        cmd = "curl -s '{}imagenet_train-{}.tgz'".format(remote_loc, url)
        return subprocess.Popen(
            cmd, bufsize=1000000, shell=True, stdout=subprocess.PIPE
        ).stdout

    ds = wds.WebDataset(
        "{0000..0147}", extensions="jpg;png cls", shuffle=100, opener=opener
    )
    assert count_samples(ds, n=10) == 10


def test_pipe():
    ds = wds.WebDataset(
        f"pipe:curl -s '{remote_loc}" + "imagenet_train-{0000..0147}.tgz'",
        extensions="jpg;png cls",
        shuffle=100,
    )
    assert count_samples(ds, n=10) == 10


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
    ds = wds.WebDataset(
        remote_loc + "imagenet_train-{0000..0147}.tgz",
        decoder="pil",
        extensions="jpg;png cls",
        transforms=[preproc, lambda x: x - 1, lambda x: x],
    )
    for sample in ds:
        assert isinstance(sample[0], torch.Tensor)
        assert tuple(sample[0].size()) == (3, 224, 224)
        assert isinstance(sample[1], int)
        break

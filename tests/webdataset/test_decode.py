import io

import numpy as np
import PIL
import pytest
import torch
from imageio.v3 import imread

import webdataset as wds
from tests.conftest import remote_loc, remote_shard
from webdataset import autodecode


@pytest.mark.quick
def test_xdecode():
    dataset = wds.DataPipeline(
        wds.shardspec("testdata/imagenet-000000.tgz"),
        wds.tarfile_samples,
        wds.xdecode(
            png=imread,
            cls=lambda stream: int(stream.read()),
            must_decode=False,
        ),
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


def test_decoders():
    ref = None
    for spec in autodecode.imagespecs.keys():
        print(spec)
        shardname = "testdata/imagenet-000000.tgz"
        dataset = wds.DataPipeline(
            wds.SimpleShardList([shardname]),
            wds.tarfile_to_samples(),
            wds.decode(autodecode.ImageHandler(spec)),
            wds.to_tuple("png", "cls"),
        )
        out = list(iter(dataset))
        if "8" in spec:
            for x in out:
                assert x[0].dtype in [np.uint8, torch.uint8], (x[0].dtype, spec)
        elif not spec.startswith("pil"):
            for x in out:
                assert x[0].dtype in [np.float32, torch.float32], (x[0].dtype, spec)
        if spec in ["l", "l8", "torchl", "torchl8"]:
            for x in out:
                assert x[0].ndim == 2, (spec, x[0].shape)
        shapes = [x[0].shape for x in out] if not spec.startswith("pil") else None
        if ref is None:
            ref = shapes
        else:
            if spec.startswith("torch"):
                for x, y in zip(ref, shapes):
                    assert x[-2:] == y[-2:], (x, y, spec)
            elif spec.startswith("pil"):
                pass
            else:
                for x, y in zip(ref, shapes):
                    assert x[:2] == y[:2], (x, y, spec)


@pytest.mark.quick
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

import webdataset.typecheck  # isort:skip

import os
import tempfile
import time
from urllib.parse import urlparse

import wsds
from tests.conftest import local_data, remote_loc, remote_shards


def summarize_sample(sample):
    for k, v in sample.items():
        print(k, type(v), repr(v)[:80])


def test_smoke():
    ds = wsds.SimpleDataset([local_data])


def test_simple():
    ds = wsds.SimpleDataset([local_data])
    assert ds.size() == -1
    count = 0
    required_keys = "__key__ __url__ __local_path__ .cls .png .wnid .xml".split()
    keys = set()
    for sample in ds:
        assert set(sample.keys()) == set(required_keys)
        keys.add(sample["__key__"])
        assert isinstance(sample[".png"], bytes)
        count += 1
    assert count == 47
    assert len(keys) == 47


def test_transforms():
    import numpy as np

    def my_transform(sample):
        wsds.decode_all_gz(sample)
        wsds.decode_basic(sample)
        wsds.decode_images_to_numpy(sample)
        return sample

    ds = wsds.SimpleDataset([local_data], transformations=my_transform)
    for sample in ds:
        assert isinstance(sample[".png"], np.ndarray)

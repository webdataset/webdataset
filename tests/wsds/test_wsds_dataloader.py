import webdataset.typecheck  # isort:skip
import os
import tempfile
import time
from urllib.parse import urlparse

import pytest
from icecream import ic

ic.configureOutput(argToStringFunction=lambda obj: repr(obj)[:100])

import wsds
from tests.conftest import local_data, remote_loc, remote_shards
from wsds.datasets import DatasetSpec, SequentialDataset


def summarize_sample(sample):
    for k, v in sample.items():
        print(k, type(v), repr(v)[:80])


spec = f"""---
__kind__: "webdataset-spec-v1"
train:
    sequential:
        shards: {local_data}
        batch_size: 4
        transformations:
        - wsds.decode_all_gz
        - wsds.decode_basic
        - wsds.decode_images_to_pil
        - fn: wsds.pil_resize 
          key: .png
          shape: [224, 224]
    loader:
        reshuffle_size: 1000
        batch_size: 5
        num_workers: 4
"""


def test_smoke():
    loader = wsds.make_loader(spec)
    sample = next(iter(loader))
    assert set(".wnid __url__ .cls .xml .png".split()) <= set(sample.keys())
    assert len(sample["__url__"]) == 5
    assert len(sample[".png"]) == 5
    assert sample[".png"][0].size == (224, 224)

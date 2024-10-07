import webdataset.typecheck  # isort:skip
import os
import tempfile
import time
from urllib.parse import urlparse

import pytest
from icecream import ic

import wsds
from tests.conftest import local_data, remote_loc, remote_shards
from wsds.datasets import DatasetSpec, SequentialDataset


def summarize_sample(sample):
    for k, v in sample.items():
        print(k, type(v), repr(v)[:80])


spec = f"""---
train:
    sequential:
        shards: {local_data}
        batch_size: 4
    loader:
        reshuffle_size: 1000
        batch_size: 5
        num_workers: 4
"""


def test_smoke():
    loader = wsds.make_loader(spec)
    sample = next(iter(loader))
    ic(sample)
    raise Exception("test_smoke")

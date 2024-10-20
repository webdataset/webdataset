import webdataset.typecheck  # isort:skip
import os
import tempfile
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import pytest
from icecream import ic

import wsds
from tests.conftest import local_data, remote_loc, remote_shards
from wsds.datasets import DatasetSpec, SequentialDataset
from wsds.mixer import make_mixer

# ic.configureOutput(argToStringFunction=lambda obj: repr(obj)[:100])


def summarize_sample(sample):
    for k, v in sample.items():
        print(k, type(v), repr(v)[:80])


spec = f"""---
__kind__: "wsds-mixer-v1"
train:
    loaders: 
      - comment: unusually, an inline spec
        dataset_spec: |
            ---
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
    num_workers: -1
"""


def test_smoke():
    result = make_mixer(spec, "train")
    ic(result)
    raise Exception("test_smoke")

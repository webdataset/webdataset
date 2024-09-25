import webdataset.typecheck  # isort:skip

import os
import tempfile
import time
from urllib.parse import urlparse

import wsds.datasets
from tests.conftest import local_data, remote_loc, remote_shards


def test_smoke():
    ds = wsds.datasets.SimpleDataset([local_data])

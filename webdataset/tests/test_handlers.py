import io
import os
import pickle
import time

import numpy as np
import PIL
import pytest
import torch
from torch.utils.data import DataLoader
from io import StringIO
import yaml
from itertools import islice
from imageio import imread

import webdataset as wds
import webdataset.extradatasets as eds
from webdataset import (
    SimpleShardList,
    autodecode,
    filters,
    handlers,
    shardlists,
    tariterators,
)


def test_missing_throws(tmp_path):
    path = os.path.join(tmp_path, "missing.tar")
    ds = wds.WebDataset(path)
    with pytest.raises(IOError):
        for sample in ds:
            pass

def test_missing_throws2(tmp_path):
    # path = os.path.join("http://storage.googleapis.com/torch-ml/vision/imagenet", "missing.tar)
    path = "http://storage.googleapis.com/nvdata-openimages/missing.tar"
    ds = wds.WebDataset(path)
    with pytest.raises(IOError):
        for sample in ds:
            pass

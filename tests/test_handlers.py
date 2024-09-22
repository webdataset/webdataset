import os

import pytest

import webdataset as wds


@pytest.mark.quick
def test_missing_throws(tmp_path):
    path = os.path.join(tmp_path, "missing.tar")
    ds = wds.WebDataset(path, shardshuffle=False)
    with pytest.raises(IOError):
        for sample in ds:
            pass


def test_missing_throws2(tmp_path):
    # path = os.path.join("http://storage.googleapis.com/torch-ml/vision/imagenet", "missing.tar)
    path = "http://storage.googleapis.com/nvdata-openimages/missing.tar"
    ds = wds.WebDataset(path, shardshuffle=False)
    with pytest.raises(IOError):
        for sample in ds:
            pass

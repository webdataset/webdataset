import os
import time

import numpy as np

import webdataset as wds
from webdataset import (
    autodecode,
)


def test_mcached():
    shardname = "testdata/imagenet-000000.tgz"
    dataset = wds.DataPipeline(
        wds.SimpleShardList([shardname]),
        wds.tarfile_to_samples(),
        wds.Cached(),
    )
    result1 = list(iter(dataset))
    result2 = list(iter(dataset))
    assert len(result1) == len(result2)


def test_lmdb_cached(tmp_path):
    shardname = "testdata/imagenet-000000.tgz"
    dest = os.path.join(tmp_path, "test.lmdb")
    assert not os.path.exists(dest)
    dataset = wds.DataPipeline(
        wds.SimpleShardList([shardname]),
        wds.tarfile_to_samples(),
        wds.LMDBCached(dest),
    )
    result1 = list(iter(dataset))
    assert os.path.exists(dest)
    result2 = list(iter(dataset))
    assert os.path.exists(dest)
    assert len(result1) == len(result2)
    del dataset
    dataset = wds.DataPipeline(
        wds.SimpleShardList([shardname]),
        wds.tarfile_to_samples(),
        wds.LMDBCached(dest),
    )
    result3 = list(iter(dataset))
    assert len(result1) == len(result3)


def test_cached(tmp_path):
    shardname = "testdata/imagenet-000000.tgz"
    dest = os.path.join(tmp_path, shardname)
    assert not os.path.exists(dest)
    dataset = wds.DataPipeline(
        wds.SimpleShardList([shardname] * 3),
        wds.resampled(10),
        wds.cached_tarfile_to_samples(cache_dir=tmp_path, verbose=True, always=True),
        wds.shuffle(100),
        wds.decode(autodecode.ImageHandler("rgb")),
        wds.to_tuple("png", "cls"),
    )
    result = list(iter(dataset))
    assert os.path.exists(dest)
    assert os.system(f"cmp {shardname} {dest}") == 0
    assert len(result[0]) == 2
    assert isinstance(result[0][0], np.ndarray)
    assert isinstance(result[0][1], int)
    assert len(result) == 470
    result = list(iter(dataset))
    assert len(result[0]) == 2
    assert isinstance(result[0][0], np.ndarray)
    assert isinstance(result[0][1], int)
    assert len(result) == 470


def test_lru_cleanup(tmp_path):
    for i in range(20):
        fname = os.path.join(tmp_path, "%06d" % i)
        with open(fname, "wb") as f:
            f.write(b"x" * 4096)
        print(fname, os.path.getctime(fname))
        time.sleep(0.1)
    assert "000000" in os.listdir(tmp_path)
    assert "000019" in os.listdir(tmp_path)
    total_before = sum(
        os.path.getsize(os.path.join(tmp_path, fname)) for fname in os.listdir(tmp_path)
    )
    wds.lru_cleanup(tmp_path, total_before / 2, verbose=True)
    total_after = sum(
        os.path.getsize(os.path.join(tmp_path, fname)) for fname in os.listdir(tmp_path)
    )
    assert total_after <= total_before * 0.5
    assert "000000" not in os.listdir(tmp_path)
    assert "000019" in os.listdir(tmp_path)

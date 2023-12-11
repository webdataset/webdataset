import os
import time

import webdataset as wds
from webdataset.cache import FileCache, LRUCleanup, StreamingOpen


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


class TestStreamingOpen:
    def setup_method(self):
        self.stream_open = StreamingOpen()

    def test_local_file(self, tmp_path):
        # Create a temporary file
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, World!")

        # Test opening the local file
        for file in self.stream_open([str(file_path)]):
            assert file.read().decode() == "Hello, World!"

    def test_remote_file(self):
        # Test opening a remote file
        url = "https://storage.googleapis.com/webdataset/testdata/imagenet-000000.tgz"
        for file in self.stream_open([url]):
            assert (
                file.read(1) == b"\x1f"
            )  # Check that the file starts with the expected gzip magic number


class TestFileCache:
    def setup_method(self):
        self.file_cache = FileCache()

    def test_local_file(self, tmp_path):
        # Create a temporary file
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, World!")

        # Test opening the local file
        with self.file_cache.open_file(str(file_path)) as file:
            assert file.read().decode() == "Hello, World!"

    def test_remote_file(self):
        # Test opening a remote file
        url = "https://storage.googleapis.com/webdataset/testdata/imagenet-000000.tgz"
        with self.file_cache.open_file(url) as file:
            assert (
                file.read(1) == b"\x1f"
            )  # Check that the file starts with the expected gzip magic number


def test_lru_cleanup(tmp_path):
    lru_cleanup = LRUCleanup(
        tmp_path, interval=None
    )  # create an instance of the LRUCleanup class

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

    lru_cleanup.cache_size = (
        total_before * 0.5
    )  # set the cache size to 50% of the total size

    lru_cleanup.cleanup()  # use the cleanup method of the LRUCleanup class

    total_after = sum(
        os.path.getsize(os.path.join(tmp_path, fname)) for fname in os.listdir(tmp_path)
    )

    assert total_after <= total_before * 0.5
    assert "000000" not in os.listdir(tmp_path)
    assert "000019" in os.listdir(tmp_path)

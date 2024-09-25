import webdataset.typecheck  # isort:skip

import os
import tempfile
import time
from urllib.parse import urlparse

import pytest

from webdataset.cache import FileCache, LRUCleanup, StreamingOpen, url_to_cache_name


def test_url_to_cache_name():
    assert url_to_cache_name("http://example.com/path/to/file.txt") == "file.txt"
    assert (
        url_to_cache_name("http://example.com/path/to/file.txt", ndir=1)
        == "to/file.txt"
    )
    assert (
        url_to_cache_name("http://example.com/path/to/file.txt", ndir=2)
        == "path/to/file.txt"
    )
    assert (
        url_to_cache_name("http://example.com/path/to/file.txt", ndir=3)
        == "path/to/file.txt"
    )
    assert url_to_cache_name("http://example.com/") == ""
    assert url_to_cache_name("http://example.com", ndir=1) == ""
    assert url_to_cache_name("file:///path/to/file.txt") == "file.txt"
    assert url_to_cache_name("file:///path/to/file.txt", ndir=1) == "to/file.txt"
    assert url_to_cache_name("ftp://example.com/path/to/file.txt") == "file.txt"
    assert url_to_cache_name("gs://example.com/path/to/file.txt") == "file.txt"
    assert url_to_cache_name("s3://example.com/path/to/file.txt") == "file.txt"
    assert url_to_cache_name("ais://example.com/path/to/file.txt") == "file.txt"
    assert (
        url_to_cache_name("unknown://example.com/path/to/file.txt")
        == "unknown%3A%2F%2Fexample.com%2Fpath%2Fto%2Ffile.txt"
    )


def test_url_to_cache_name_non_string_input():
    with pytest.raises(AssertionError):
        url_to_cache_name(123)


class TestStreamingOpen:
    def setup_method(self):
        self.stream_open = StreamingOpen()

    def test_local_file(self, tmp_path):
        # Create a temporary file
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, World!")

        # Test opening the local file
        for result in self.stream_open([str(file_path)]):
            assert result["url"] == str(file_path)
            assert "local_path" in result
            assert result["local_path"] == str(file_path)
            assert result["stream"].read().decode() == "Hello, World!"

    def test_remote_file(self):
        # Test opening a remote file
        url = "https://storage.googleapis.com/webdataset/testdata/imagenet-000000.tgz"
        for result in self.stream_open([url]):
            assert result["url"] == url
            assert "local_path" not in result
            assert (
                result["stream"].read(1) == b"\x1f"
            )  # Check that the file starts with the expected gzip magic number


class TestFileCache:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_cache = FileCache(cache_dir=self.temp_dir.name, validator=None)
        yield
        self.temp_dir.cleanup()

    def test_local_file(self):
        # Create a temporary file
        file_path = tempfile.mktemp(dir=self.temp_dir.name)
        with open(file_path, "w") as f:
            f.write("Hello, World!")

        # Test opening the local file
        result = next(self.file_cache([{"url": file_path}]))
        assert result["url"] == file_path
        assert "local_path" in result
        assert result["local_path"] == file_path
        with result["stream"] as file:
            assert file.read().decode() == "Hello, World!"

    def test_remote_file(self):
        # Test opening a remote file
        url = "https://storage.googleapis.com/webdataset/testdata/imagenet-000000.tgz"
        result = next(self.file_cache([{"url": url}]))
        assert result["url"] == url
        assert "local_path" in result
        assert os.path.exists(result["local_path"])
        assert os.path.dirname(result["local_path"]) == self.temp_dir.name
        assert os.path.basename(result["local_path"]) == os.path.basename(
            urlparse(url).path
        )
        with result["stream"] as file:
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

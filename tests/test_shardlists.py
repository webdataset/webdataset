import glob
import os
import tempfile
from itertools import islice
from pathlib import Path

import pytest

from webdataset.shardlists import DirectoryShardList, ResampledShards, SimpleShardList


class TestSimpleShardList:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.urls = [
            "http://example.com/file1",
            "http://example.com/file2",
            "http://example.com/file3",
        ]

    def test_init_with_seed(self):
        seed = 0
        ssl = SimpleShardList(self.urls, seed)
        assert ssl.urls == self.urls
        assert ssl.seed == seed

    def test_init_without_seed(self):
        ssl = SimpleShardList(self.urls)
        assert ssl.urls == self.urls
        assert ssl.seed is None

    def test_iter_with_seed(self):
        seed = 0
        ssl = SimpleShardList(self.urls, seed)
        assert set(shard["url"] for shard in ssl) == set(self.urls)

    def test_iter_without_seed(self):
        ssl = SimpleShardList(self.urls)
        shards = list(ssl)
        assert set(shard["url"] for shard in shards) == set(self.urls)


class TestResampledShards:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.urls = [
            "http://example.com/file1",
            "http://example.com/file2",
            "http://example.com/file3",
        ]
        self.nshards = 2
        self.seed = 0
        self.worker_seed = lambda: 1
        self.deterministic = False
        self.max_urls = int(1e6)

    def test_init(self):
        rs = ResampledShards(
            self.urls,
            self.nshards,
            self.seed,
            self.worker_seed,
            self.deterministic,
            self.max_urls,
        )
        assert rs.urls == self.urls
        assert rs.nshards == self.nshards
        assert rs.seed == self.seed
        assert rs.worker_seed == self.worker_seed
        assert rs.deterministic == self.deterministic
        assert rs.epoch == -1

    def test_iter(self):
        rs = ResampledShards(
            self.urls,
            self.nshards,
            self.seed,
            self.worker_seed,
            self.deterministic,
            self.max_urls,
        )
        shards = list(rs)
        assert len(shards) == self.nshards
        for shard in shards:
            assert shard["url"] in self.urls


class TestDirectoryShardList:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = self.test_dir.name + "/"
        yield
        self.test_dir.cleanup()

    def create_test_files(self, num_files, start=0):
        for i in range(start, start + num_files):
            Path(self.test_path + f"file{i}.tar").touch()

    def test_random_selection(self):
        self.create_test_files(5)
        ds = DirectoryShardList(
            self.test_path, pattern="*.tar", select="random", poll=0.01
        )
        files = []
        for file in islice(ds, 5):
            files.append(file["url"])
        assert len(files) == 5

    def test_oldest_selection(self):
        self.create_test_files(5)
        ds = DirectoryShardList(
            self.test_path, pattern="*.tar", select="oldest", mode="unlink", poll=None
        )
        files = [file["url"] for file in islice(ds, 3)]
        assert len(files) == 3
        assert len(glob.glob(self.test_path + "*.tar")) == 2
        assert files == [
            self.test_path + f"file{i}.tar._{os.getpid()}_" for i in range(3)
        ]

    def test_adding_files(self):
        self.create_test_files(5)
        ds = DirectoryShardList(
            self.test_path,
            pattern="*.tar",
            select="random",
            mode="unlink",
            poll=None,
        )
        files = set()
        src = iter(ds)
        for _ in range(3):
            files.add(next(src)["url"])
        assert len(glob.glob(self.test_path + "*.tar")) == 2
        self.create_test_files(5, 10)  # add more files
        assert len(glob.glob(self.test_path + "*.tar")) == 7
        for file in ds:
            files.add(file["url"])
        assert len(files) == 10
        assert len(glob.glob(self.test_path + "*.tar")) == 0

    def test_removing_files(self):
        self.create_test_files(5)
        ds = DirectoryShardList(
            self.test_path, pattern="*.tar", select="random", mode="unlink", poll=None
        )
        files = set()
        for _ in range(5):
            files.add(next(iter(ds))["url"])
        for file in files:
            os.remove(file)  # remove the files
        with pytest.raises(StopIteration):
            next(iter(ds))

    def test_recycle_unlink(self):
        self.create_test_files(1)
        ds = DirectoryShardList(
            self.test_path, pattern="*.tar", select="random", poll=None, mode="unlink"
        )
        src = iter(ds)
        file = next(src)["url"]
        with pytest.raises(StopIteration):
            next(src)
        assert not Path(file).exists()

    def test_recycle_keep(self):
        self.create_test_files(1)
        files = glob.glob(self.test_path + "*.tar*")
        assert len(files) == 1
        original = files[0]
        ds = DirectoryShardList(
            self.test_path, pattern="*.tar", select="random", poll=None, mode="keep"
        )
        src = iter(ds)
        file = next(src)["url"]
        assert Path(file).exists()
        with pytest.raises(StopIteration):
            next(src)
        assert not Path(file).exists()
        assert Path(original + "._done_").exists()

    def test_recycle_resample(self):
        self.create_test_files(1)
        ds = DirectoryShardList(
            self.test_path, pattern="*.tar", select="random", poll=None, mode="resample"
        )
        file = next(iter(ds))["url"]
        assert Path(file).exists()

    def test_cleanup_files_without_processes(self):
        self.create_test_files(1)
        ds = DirectoryShardList(
            self.test_path, pattern="*.tar", select="random", poll=None
        )
        file = next(iter(ds))["url"]
        os.rename(file, file + "._99999999_")
        ds.cleanup_files_without_processes()
        assert not Path(file + "._99999999_").exists()

import pytest
from wids import wids
from wids import wids_specs
import os
import textwrap
import json
import io


class TestIndexedTarSamples:
    def setup_class(self):
        tar_file = "testdata/ixtest.tar"
        md5sum = "3b3c0afe31e45325b7c4e6dec5235d13"
        expected_size = 10
        self.indexed_samples = wids.IndexedTarSamples(tar_file, md5sum, expected_size)

    def test_length(self):
        assert (
            len(self.indexed_samples) == 10
        )  # Update with the expected number of samples

    def test_getitem(self):
        sample = self.indexed_samples[0]  # Update with the desired sample index

        assert isinstance(sample, dict)
        assert "__key__" in sample
        assert ".cls" in sample or ".jpg" in sample


class TestLRUShards:
    def test_add(self, tmpdir):
        lru_shards = wids.LRUShards(2, localname=wids.default_localname(tmpdir))
        assert len(lru_shards) == 0
        shard = lru_shards.get_shard("testdata/ixtest.tar")
        assert len(shard) == 10
        assert len(lru_shards) == 1
        path = shard.path
        assert os.path.exists(path)
        assert os.stat(path).st_nlink == 2
        shard = lru_shards.get_shard("testdata/ixtest.tar")
        assert os.stat(path).st_nlink == 2
        # lru_shards.release(shard)
        # assert not os.path.exists(path)

class TestGz:
    def test_gz(self):
        dataset = wids.ShardListDataset([dict(url="testdata/testgz.tar", nsamples=1000)])
        assert len(dataset) == 1000
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "__key__" in sample
        assert ".txt.gz" in sample
        assert sample[".txt.gz"] == ", or more info on the item.", sample

class TestShardListDataset:
    @pytest.fixture(scope="class")
    def class_tmpdir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("class_tmpdir")

    @pytest.fixture(scope="function")
    def shard_list_dataset(self, tmpdir):
        # Define example shards and other necessary variables for creating the dataset
        shards = [
            dict(url="testdata/mpdata.tar", nsamples=100),
            dict(url="testdata/tendata.tar", nsamples=100),
            dict(url="testdata/compressed.tar", nsamples=3),
        ]

        dataset = wids.ShardListDataset(
            shards, cache_size=2, localname=wids.default_localname(str(tmpdir))
        )
        dataset.tmpdir = str(tmpdir)

        yield dataset

        # Clean up any resources used by the dataset after running the tests

    def test_initialization(self, shard_list_dataset):
        assert len(shard_list_dataset.shards) == 3
        assert shard_list_dataset.lengths == [100, 100, 3]
        assert shard_list_dataset.total_length == 203

    def test_length(self, shard_list_dataset):
        assert len(shard_list_dataset) == 203

    def test_getshard(self, shard_list_dataset):
        shard, _, _ = shard_list_dataset.get_shard(0)
        assert os.path.exists(shard.path)
        assert os.stat(shard.path).st_nlink == 2

    def test_getitem(self, shard_list_dataset):
        # access sample in the first shard
        sample = shard_list_dataset[17]
        assert isinstance(sample, dict)
        assert "__key__" in sample
        assert sample["__key__"] == "000017"
        assert ".mp" in sample
        assert len(shard_list_dataset.cache) == 1
        cache = set(shard_list_dataset.cache.keys())
        assert cache == set("testdata/mpdata.tar".split()), cache

        # access sample in the second shard
        sample = shard_list_dataset[190]
        assert isinstance(sample, dict)
        assert "__key__" in sample
        assert sample["__key__"] == "000090"
        assert ".ten" in sample
        assert len(shard_list_dataset.cache) == 2
        cache = set(shard_list_dataset.cache.keys())
        assert cache == set("testdata/tendata.tar testdata/mpdata.tar".split()), cache

        # access sample in the third shard
        sample = shard_list_dataset[200]
        assert isinstance(sample, dict)
        assert "__key__" in sample
        assert sample["__key__"] == "compressed/0001"
        assert ".txt.gz" in sample
        assert len(shard_list_dataset.cache) == 2
        cache = set(shard_list_dataset.cache.keys())
        assert cache == set(
            "testdata/tendata.tar testdata/compressed.tar".split()
        ), cache

        # access sample in the third shard
        sample = shard_list_dataset[201]
        assert isinstance(sample, dict)
        assert "__key__" in sample
        assert sample["__key__"] == "compressed/0002"
        assert ".txt.gz" in sample
        assert len(shard_list_dataset.cache) == 2
        cache = set(shard_list_dataset.cache.keys())
        assert cache == set(
            "testdata/tendata.tar testdata/compressed.tar".split()
        ), cache

        # access sample in the first shard
        sample = shard_list_dataset[0]
        assert isinstance(sample, dict)
        assert "__key__" in sample
        assert sample["__key__"] == "000000"
        assert ".mp" in sample
        assert len(shard_list_dataset.cache) == 2
        cache = set(shard_list_dataset.cache.keys())
        assert cache == set(
            "testdata/mpdata.tar testdata/compressed.tar".split()
        ), cache

        assert shard_list_dataset.get_stats() == (5, 4)

class TestSpecs:
    def test_spec_parsing(self):
        spec = textwrap.dedent("""
        {
            "__kind__": "wids-shard-index-v1",
            "name": "train",
            "wids_version": 1,
            "shardlist": [
                {
                    "url": "testdata/mpdata.tar",
                    "nsamples": 10
                }
            ]
        }
        """)
        spec = json.loads(spec)
        shardlist = wids_specs.extract_shardlist(spec)
        assert len(shardlist) == 1

    def test_spec_parsing(self):
        spec = textwrap.dedent("""
        {
            "__kind__": "wids-shard-index-v1",
            "name": "train",
            "wids_version": 1,
            "shardlist": [
                {
                    "url": "testdata/mpdata.tar",
                    "nsamples": 10
                }
            ]
        }
        """)
        stream = io.StringIO(spec)
        dataset = wids.ShardListDataset(stream)
        assert len(dataset) == 10


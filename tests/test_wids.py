import io
import json
import os
import random
import textwrap
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from wids import DistributedChunkedSampler, wids, wids_specs
from wids.wids import ChunkedSampler, ShardedSampler, ShardListDataset


@pytest.mark.quick
class TestIndexedTarSamples:
    def setup_class(self):
        tar_file = "testdata/ixtest.tar"
        md5sum = "3b3c0afe31e45325b7c4e6dec5235d13"
        expected_size = 10
        self.indexed_samples = wids.IndexedTarSamples(
            path=tar_file, md5sum=md5sum, expected_size=expected_size
        )

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
    def test_add(self, tmpdir: str):
        lru_shards = wids.LRUShards(2, localname=wids.default_localname(tmpdir))
        assert len(lru_shards) == 0
        shard = lru_shards.get_shard("testdata/ixtest.tar")
        assert len(shard) == 10
        assert len(lru_shards) == 1
        path = shard.path
        assert os.path.exists(path)
        shard = lru_shards.get_shard("testdata/ixtest.tar")
        # lru_shards.release(shard)
        # assert not os.path.exists(path)


class TestGz:
    def test_gz(self):
        dataset = wids.ShardListDataset(
            [dict(url="testdata/testgz.tar", nsamples=1000)]
        )
        assert len(dataset) == 1000
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "__key__" in sample
        assert ".txt.gz" in sample
        assert sample[".txt.gz"] == ", or more info on the item.", sample


class TestShardListDataset:
    @pytest.fixture(scope="class")
    def class_tmpdir(self, tmp_path_factory: pytest.TempPathFactory):
        return tmp_path_factory.mktemp("class_tmpdir")

    @pytest.fixture(scope="function")
    def shard_list_dataset(self, tmpdir: str):
        # Define example shards and other necessary variables for creating the dataset
        shards = [
            dict(url="testdata/mpdata.tar", nsamples=100),
            dict(url="testdata/tendata.tar", nsamples=100),
            dict(url="testdata/compressed.tar", nsamples=3),
        ]

        dataset = wids.ShardListDataset(
            shards, lru_size=2, localname=wids.default_localname(str(tmpdir))
        )
        dataset.tmpdir = str(tmpdir)

        yield dataset

        # Clean up any resources used by the dataset after running the tests

    def test_initialization(
        self, shard_list_dataset: wids.ShardListDataset[list[dict[str, str | int]]]
    ):
        assert len(shard_list_dataset.shards) == 3
        assert shard_list_dataset.lengths == [100, 100, 3]
        assert shard_list_dataset.total_length == 203

    def test_length(self, shard_list_dataset: ShardListDataset):
        assert len(shard_list_dataset) == 203

    def test_getshard(self, shard_list_dataset: ShardListDataset):
        shard, _, _ = shard_list_dataset.get_shard(0)
        assert os.path.exists(shard.path)

    def test_getitem(self, shard_list_dataset: ShardListDataset):
        # make sure this was set up correctly
        assert shard_list_dataset.cache.lru.capacity == 2

        # access sample in the first shard
        sample = shard_list_dataset[17]
        assert isinstance(sample, dict)
        assert "__key__" in sample
        assert sample["__key__"] == "000017"
        assert ".mp" in sample
        # assert len(shard_list_dataset.cache) == 1
        cache = set(shard_list_dataset.cache.lru.keys())
        assert cache == set("testdata/mpdata.tar".split()), cache

        # access sample in the second shard
        sample = shard_list_dataset[190]
        assert isinstance(sample, dict)
        assert "__key__" in sample
        assert sample["__key__"] == "000090"
        assert ".ten" in sample
        # assert len(shard_list_dataset.cache) == 2
        cache = set(shard_list_dataset.cache.lru.keys())
        assert cache == set("testdata/tendata.tar testdata/mpdata.tar".split()), cache

        # access sample in the third shard
        sample = shard_list_dataset[200]
        assert isinstance(sample, dict)
        assert "__key__" in sample
        assert sample["__key__"] == "compressed/0001"
        assert ".txt.gz" in sample
        # assert len(shard_list_dataset.cache) == 2
        cache = set(shard_list_dataset.cache.lru.keys())
        assert cache == set(
            "testdata/tendata.tar testdata/compressed.tar".split()
        ), cache

        # access sample in the third shard
        sample = shard_list_dataset[201]
        assert isinstance(sample, dict)
        assert "__key__" in sample
        assert sample["__key__"] == "compressed/0002"
        assert ".txt.gz" in sample
        # assert len(shard_list_dataset.cache) == 2
        cache = set(shard_list_dataset.cache.lru.keys())
        assert cache == set(
            "testdata/tendata.tar testdata/compressed.tar".split()
        ), cache

        # access sample in the first shard
        sample = shard_list_dataset[0]
        assert isinstance(sample, dict)
        assert "__key__" in sample
        assert sample["__key__"] == "000000"
        assert ".mp" in sample
        # assert len(shard_list_dataset.cache) == 2
        cache = set(shard_list_dataset.cache.lru.keys())
        assert cache == set(
            "testdata/mpdata.tar testdata/compressed.tar".split()
        ), cache

        assert shard_list_dataset.get_stats() == (5, 4)


class TestSpecs:
    def test_spec_parsing(self):
        spec = textwrap.dedent(
            """
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
        """
        )
        spec = json.loads(spec)
        shardlist = wids_specs.extract_shardlist(spec)
        assert len(shardlist) == 1

    def test_spec_parsing(self):
        spec = textwrap.dedent(
            """
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
        """
        )
        stream = io.StringIO(spec)
        dataset = wids.ShardListDataset(stream)
        assert len(dataset) == 10


class TestShardedSampler:
    @pytest.fixture(scope="function")
    def sharded_sampler(self):
        dataset = wids.ShardListDataset(
            [
                dict(url="testdata/mpdata.tar", nsamples=100),
                dict(url="testdata/tendata.tar", nsamples=100),
                dict(url="testdata/compressed.tar", nsamples=3),
            ]
        )
        sampler = wids.ShardedSampler(dataset)
        yield sampler

    def test_initialization(self, sharded_sampler: wids.ShardedSampler):
        assert len(sharded_sampler.ranges) == 3
        assert sharded_sampler.ranges == [(0, 100), (100, 200), (200, 203)]

    def test_iter(self, sharded_sampler: ShardedSampler):
        indexes = list(sharded_sampler)
        assert len(indexes) == 203
        assert min(indexes) == 0
        assert max(indexes) == 202
        assert len(set(indexes)) == 203  # Ensure all indexes are unique

    def test_iter_with_shuffle(self, sharded_sampler: ShardedSampler):
        indexes1 = list(sharded_sampler)
        indexes2 = list(sharded_sampler)
        assert indexes1 != indexes2  # Ensure order changes with each iteration


class TestChunkedSampler:
    def setup_method(self):
        self.dataset = list(range(10000))
        self.sampler = ChunkedSampler(
            self.dataset,
            num_samples=5000,
            chunksize=1000,
            seed=0,
            shuffle=True,
            shufflefirst=False,
        )

    def test_init(self):
        assert self.sampler.ranges == [
            (0, 1000),
            (1000, 2000),
            (2000, 3000),
            (3000, 4000),
            (4000, 5000),
        ]
        assert self.sampler.seed == 0
        assert self.sampler.shuffle == True
        assert self.sampler.shufflefirst == False
        assert self.sampler.epoch == 0

    def test_set_epoch(self):
        self.sampler.set_epoch(5)
        assert self.sampler.epoch == 5

    def test_iter(self):
        random.seed(0)
        samples = list(iter(self.sampler))
        assert len(samples) == 5000
        assert samples != list(range(5000))  # The samples should be shuffled

    def test_iter_no_shuffle(self):
        self.sampler.shuffle = False
        samples = list(iter(self.sampler))
        assert len(samples) == 5000
        assert samples == list(range(5000))  # The samples should not be shuffled

    def test_iter_full_range(self):
        self.sampler = ChunkedSampler(
            self.dataset,
            num_samples=5000,
            chunksize=1000,
            seed=0,
            shuffle=True,
            shufflefirst=False,
        )
        samples = list(iter(self.sampler))
        assert set(samples) == set(
            range(5000)
        )  # The samples should cover the full range

    def test_iter_full_range_no_shuffle(self):
        self.sampler = ChunkedSampler(
            self.dataset,
            num_samples=5000,
            chunksize=1000,
            seed=0,
            shuffle=False,
            shufflefirst=False,
        )
        samples = list(iter(self.sampler))
        assert set(samples) == set(
            range(5000)
        )  # The samples should cover the full range

    def test_num_samples_range(self):
        self.sampler = ChunkedSampler(
            self.dataset,
            num_samples=(1111, 2222),
            chunksize=1000,
            seed=0,
            shuffle=True,
            shufflefirst=False,
        )
        samples = list(iter(self.sampler))
        assert set(samples) == set(
            range(1111, 2222)
        )  # The samples should cover the range from 1111 to 2222


# Fixture for mocking the distributed environment
@pytest.fixture
def mock_distributed_env():
    def _mock_distributed_env(rank, world_size):
        with patch("torch.distributed.init_process_group"):
            with patch("torch.distributed.get_rank", return_value=rank):
                with patch("torch.distributed.get_world_size", return_value=world_size):
                    yield

    return _mock_distributed_env


# Context manager for mocking the distributed environment
@contextmanager
def mock_distributed_env(rank, world_size):
    with (
        patch("torch.distributed.init_process_group"),
        patch("torch.distributed.get_rank", return_value=rank),
        patch("torch.distributed.get_world_size", return_value=world_size),
        patch("torch.distributed.is_initialized", return_value=True),
    ):
        yield


class TestDistributedChunkedSampler:
    def setup_method(self, method):
        self.dataset = list(range(10000))
        with mock_distributed_env(0, 2):
            self.sampler = DistributedChunkedSampler(
                self.dataset,
                num_samples=5000,
                chunksize=1000,
                seed=0,
                shuffle=True,
                shufflefirst=False,
            )

    def test_init(self):
        assert self.sampler.ranges == [(0, 1000), (1000, 2000), (2000, 2500)]
        assert self.sampler.seed == 0
        assert self.sampler.shuffle == True
        assert self.sampler.shufflefirst == False
        assert self.sampler.epoch == 0

    def test_set_epoch(self):
        self.sampler.set_epoch(5)
        assert self.sampler.epoch == 5

    def test_iter(self):
        with mock_distributed_env(0, 2):
            sampler = DistributedChunkedSampler(
                self.dataset,
                num_samples=5000,
                chunksize=1000,
                seed=0,
                shuffle=True,
                shufflefirst=False,
            )
            samples = list(iter(sampler))
            assert len(samples) == 2500
            assert samples != list(range(2500))  # The samples should be shuffled
            assert set(samples) == set(
                range(2500)
            )  # The samples should cover the full range
        with mock_distributed_env(1, 2):
            sampler = DistributedChunkedSampler(
                self.dataset,
                num_samples=5000,
                chunksize=1000,
                seed=0,
                shuffle=True,
                shufflefirst=False,
            )
            samples = list(iter(sampler))
            assert len(samples) == 2500
            assert samples != list(range(2500, 5000))  # The samples should be shuffled
            assert set(samples) == set(
                range(2500, 5000)
            )  # The samples should cover the full range

    def test_iter_no_shuffle(self):
        with mock_distributed_env(0, 2):
            sampler = DistributedChunkedSampler(
                self.dataset,
                num_samples=5000,
                chunksize=1000,
                seed=0,
                shuffle=False,
                shufflefirst=False,
            )
            samples = list(iter(sampler))
            assert len(samples) == 2500
            assert samples == list(range(2500))  # The samples should not be shuffled

    def test_disjoint_samples(self):
        with mock_distributed_env(0, 2):
            sampler1 = DistributedChunkedSampler(
                self.dataset,
                num_samples=5000,
                chunksize=1000,
                seed=0,
                shuffle=True,
                shufflefirst=False,
            )
            samples1 = set(iter(sampler1))

        with mock_distributed_env(1, 2):
            sampler2 = DistributedChunkedSampler(
                self.dataset,
                num_samples=5000,
                chunksize=1000,
                seed=0,
                shuffle=True,
                shufflefirst=False,
            )
            samples2 = set(iter(sampler2))

        assert set(samples1) == set(range(2500))
        assert set(samples2) == set(range(2500, 5000))
        assert (
            samples1.intersection(samples2) == set()
        )  # The samples should be disjoint

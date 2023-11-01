import hashlib
import io
import os
import re
import tarfile
from urllib.parse import urlparse
from typing import Any, Dict, Optional, Union
import pickle
from functools import partial
import json
import io

import numpy as np
from torch.utils.data import Dataset

from .wids_dl import ConcurrentDownloader
from .wids_lru import LRUCache
from .wids_tar import TarFileReader, find_index_file
from .wids_mmtar import MMIndexedTar
from .wids_specs import load_remote_shardlist


def compute_file_md5sum(fname, chunksize=1000000):
    """Compute the md5sum of a file in chunks."""
    md5 = hashlib.md5()
    if isinstance(fname, str):
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(chunksize), b""):
                md5.update(chunk)
    else:
        fname.seek(0)
        for chunk in iter(lambda: fname.read(chunksize), b""):
            md5.update(chunk)
    return md5.hexdigest()


def compute_num_samples(fname):
    ds = IndexedTarSamples(fname)
    return len(ds)


def splitname(fname):
    """Returns the basename and extension of a filename"""
    assert "." in fname, "Filename must have an extension"
    basename, extension = re.match(r"^((?:.*/)?.*?)(\..*)$", fname).groups()
    return basename, extension


def group_by_key(names):
    """Group the file names by key.

    Args:
        names: A list of file names.

    Returns:
        A list of lists of indices, where each sublist contains indices of files
        with the same key.
    """
    groups = []
    last_key = None
    current = []
    for i, fname in enumerate(names):
        # Ignore files that are not in a subdirectory.
        if "." not in fname:
            print(fname)
            continue
        key, ext = splitname(fname)
        if key != last_key:
            if current:
                groups.append(current)
            current = []
            last_key = key
        current.append(i)
    if current:
        groups.append(current)
    return groups


def default_decoder(sample: Dict[str, Any], format: Optional[Union[bool, str]] = True):
    """A default decoder for webdataset.

    This handles common file extensions: .txt, .cls, .cls2,
        .jpg, .png, .json, .npy, .mp, .pt, .pth, .pickle, .pkl.
    These are the most common extensions used in webdataset.
    For other extensions, users can provide their own decoder.

    Args:
        sample: sample, modified in place
    """
    sample = dict(sample)
    for key, stream in sample.items():
        extension = key.split(".")[-1]
        if key.startswith("__"):
            continue
        elif extension in ["txt", "text"]:
            value = stream.read()
            sample[key] = value.decode("utf-8")
        elif extension in ["cls", "cls2"]:
            value = stream.read()
            sample[key] = int(value.decode("utf-8"))
        elif extension in ["jpg", "png", "ppm", "pgm", "pbm", "pnm"]:
            if format == "PIL":
                import PIL.Image

                sample[key] = PIL.Image.open(stream)
            elif format == "numpy":
                import numpy as np

                sample[key] = np.asarray(PIL.Image.open(stream))
            else:
                raise ValueError(f"Unknown format: {format}")
        elif extension == "json":
            import json

            value = stream.read()
            sample[key] = json.loads(value)
        elif extension == "npy":
            import numpy as np

            sample[key] = np.load(stream)
        elif extension == "mp":
            import msgpack

            value = stream.read()
            sample[key] = msgpack.unpackb(value, raw=False)
        elif extension in ["pt", "pth"]:
            import torch

            sample[key] = torch.load(stream)
        elif extension in ["pickle", "pkl"]:
            import pickle

            sample[key] = pickle.load(stream)
    return sample


class IndexedTarSamples:
    """A class that accesses samples in a tar file. The tar file must follow
    WebDataset conventions. The tar file is indexed when the IndexedTarSamples
    object is created. The samples are accessed by index using the __getitem__
    method. The __getitem__ method returns a dictionary containing the files
    for the sample. The key for each file is the extension of the file name.
    The key "__key__" is reserved for the key of the sample (the basename of
    each file without the extension). For example, if the tar file contains
    the files "sample1.jpg" and "sample1.txt", then the sample with key
    "sample1" will be returned as the dictionary {"jpg": ..., "txt": ...}.
    """

    def __init__(
        self,
        tar_file,
        md5sum=None,
        expected_size=None,
        source=None,
        use_mmap=True,
        index_file=find_index_file,
    ):
        # Create TarFileReader object to read from tar_file
        self.source = source
        self.path = tar_file
        if use_mmap:
            self.reader = MMIndexedTar(tar_file)
        else:
            self.reader = TarFileReader(tar_file, index_file=index_file)
        # Get list of all files in tar_file
        all_files = self.reader.names()
        # Group files by key into samples
        self.samples = group_by_key(all_files)
        if md5sum is not None:
            got = compute_file_md5sum(tar_file)
            assert got == md5sum, f"MD5 sum mismatch: expected {md5sum}, got {got}"
        if expected_size is not None:
            assert (
                len(self) == expected_size
            ), f"Expected {expected_size} samples, got {len(self)}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get indexes of files for the sample at index idx
        indexes = self.samples[idx]
        sample = {}
        key = None
        for i in indexes:
            # Get filename and data for the file at index i
            fname, data = self.reader.get_file(i)
            # Split filename into key and extension
            k, ext = splitname(fname)
            # Make sure all files in sample have same key
            key = key or k
            assert key == k
            sample[ext] = data
        # Add key to sample
        sample["__key__"] = key
        return sample


def default_localname(dldir="/tmp/wids"):
    os.makedirs(dldir, exist_ok=True)

    def f(shard):
        """Given a URL, return a local name for the shard. The local name contains
        no directory components."""
        if shard.startswith("pipe:"):
            # uuencode the entire URL string
            return os.path.join(
                dldir, base64.urlsafe_b64encode(shard.encode()).decode()
            )
        else:
            # use urlparse to get the filename component of the URL
            return os.path.join(dldir, os.path.basename(urlparse(shard).path))

    return f


class LRUShards:
    """A class that manages a cache of shards. The cache is a LRU cache that
    stores the local names of the shards as keys and the downloaded paths as
    values. The shards are downloaded to a directory specified by dldir.
    The local name of a shard is computed by the localname function, which
    takes the shard URL as an argument. If keep is True, the downloaded files
    are not deleted when they are no longer needed.
    """

    def __init__(self, num_shards, keep=False, localname=default_localname()):
        self.localname = localname
        # the cache contains the local name as the key and the downloaded path as the value
        self.lru = LRUCache(num_shards, release_handler=self.release_handler)
        # the downloader ensures that if multiple processes download the same file on this
        # machine, only a single download takes place
        self.downloader = ConcurrentDownloader(keep=False)
        # keep statistics
        self.reset_stats()

    def reset_stats(self):
        self.accesses = 0
        self.misses = 0

    def __len__(self):
        return len(self.lru)

    def items(self):
        return self.lru.items()

    def keys(self):
        return self.lru.keys()

    def values(self):
        return self.lru.values()

    def release_handler(self, key, value):
        # called back from the LRUCache when an object is released;
        # this tells the downloader that the file is no longer needed
        self.downloader.release(value.path)

    def release(self, itf):
        for k, v in self.lru.items():
            if v is itf:
                self.release_handler(k, v)
                return
        raise ValueError("Shard not found")

    def clear(self):
        self.lru.clear()

    def get_shard(self, url):
        self.accesses += 1
        if url not in self.lru:
            local = self.localname(url)
            downloaded = self.downloader.download(url, local)
            print(url, local, downloaded)
            itf = IndexedTarSamples(downloaded, source=url)
            self.lru[url] = itf
            self.misses += 1
        return self.lru[url]




class ShardListDataset(Dataset):
    """An indexable dataset based on a list of shards.

    The shards are specified as a list of (filename, length) pairs.
    The filename can be a local file or a URL. The length is the
    number of samples in the shard. The shards are downloaded to
    a local directory and cached there."""

    def __init__(
        self,
        shards,
        cache_size=10,
        localname=default_localname(),
        transformations="PIL",
        keep=False,
    ):
        """Create a ShardListDataset.

        Args:
            shards: a list of (filename, length) pairs or a URL pointing to a JSON descriptor file
            cache_size: the number of shards to keep in the cache
            localname: a function that maps URLs to local filenames
        """
        super(ShardListDataset, self).__init__()
        # shards is a list of (filename, length) pairs. We'll need to
        # keep track of the lengths and cumulative lengths to know how
        # to map indices to shards and indices within shards.
        self.shards = (
            load_remote_shardlist(shards) if isinstance(shards, (str, io.IOBase)) else shards
        )
        print(self.shards)
        self.lengths = [shard["nsamples"] for shard in self.shards]
        self.cum_lengths = np.cumsum(self.lengths)
        self.total_length = self.cum_lengths[-1]

        if transformations == "PIL":
            self.transformations = [partial(default_decoder, format="PIL")]
        elif transformations == "numpy":
            self.transformations = [partial(default_decoder, format="numpy")]
        else:
            if not isinstance(transformations, list):
                transformations = [transformations]
            for transform in transformations:
                assert callable(transform)
            self.transformations = transformations

        self.cache = LRUShards(cache_size, localname=localname, keep=keep)

    def add_transform(self, transform):
        """Add a transformation to the dataset."""
        self.transformations.append(transform)
        return self

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.total_length

    def get_stats(self):
        """Return the number of cache accesses and misses."""
        return self.cache.accesses, self.cache.misses

    def check_cache_misses(self):
        """Check if the cache miss rate is too high."""
        accesses, misses = self.get_stats()
        if accesses > 100 and misses / accesses > 0.3:
            # output a warning only once
            self.check_cache_misses = lambda: None
            print(
                "Warning: ShardListDataset has a cache miss rate of {:.1%}%".format(
                    misses * 100.0 / accesses
                )
            )

    def get_shard(self, index):
        """Get the shard and index within the shard corresponding to the given index."""
        # Find the shard corresponding to the given index.
        shard_idx = np.searchsorted(self.cum_lengths, index, side="right")

        # Figure out which index within the shard corresponds to the
        # given index.
        if shard_idx == 0:
            inner_idx = index
        else:
            inner_idx = index - self.cum_lengths[shard_idx - 1]

        # Get the shard and return the corresponding element.
        url = self.shards[shard_idx]["url"]
        shard = self.cache.get_shard(url)
        return shard, inner_idx

    def __getitem__(self, index):
        """Return the sample corresponding to the given index."""
        shard, inner_idx = self.get_shard(index)
        sample = shard[inner_idx]

        # Check if we're missing the cache too often.
        self.check_cache_misses()

        # Apply transformations
        for transform in self.transformations:
            sample = transform(sample)

        return sample

    def close(self):
        """Close the dataset."""
        self.cache.clear()


class ShardedSampler:
    """A sampler that samples consistent with a ShardListDataset.

    This sampler is used to sample from a ShardListDataset in a way that
    preserves locality.

    This returns a permutation of the indexes by shard, then a permutation of
    indexes within each shard. This ensures that the data is accessed in a
    way that preserves locality.

    Note that how this ends up splitting data between multiple workers ends up
    on the details of the DataLoader. Generally, it will likely load samples from the
    same shard in each worker.

    Other more sophisticated shard-aware samplers are possible and will likely
    be added.
    """

    def __init__(self, dataset, lengths=None, batch_size=1, shuffle=False):
        if lengths is None:
            lengths = list(dataset.lengths)
        self.ranges = []
        start = 0
        for i, l in enumerate(lengths):
            self.ranges.append((start, start + l))
            start += l

    def __iter__(self):
        shardperm = torch.randperm(len(self.ranges))
        for shard in shardperm:
            start, end = self.ranges[shard]
            yield from (int(x) for x in torch.randperm(end - start) + start)

#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""Implement caching for shards."""

import sys
import os.path
import io
import uuid


def guess_shard(path):
    """Guess the shard from a given path."""
    return os.path.split(path.split()[-1])[1]


def shard_uuid(path):
    """Compute a UUID for a shard path."""
    return str(uuid.uuid3(uuid.NAMESPACE_URL, path))


class CacheStream(io.RawIOBase):
    """Cache raw IO stream."""

    def __init__(self, fname, stream, verbose=False):
        """Create a shard cache.

        :param fname: file name for the cache file
        :param stream: stream to be cached
        :param verbose: verbose output on progress
        """
        super().__init__()
        self.verbose = verbose
        self.stream = stream
        self.fname = fname
        self.tempname = self.fname + ".~" + str(os.getpid()) + "~"
        os.unlink(fname) if os.path.exists(fname) else None
        os.unlink(self.tempname) if os.path.exists(self.tempname) else None
        self.cache = open(self.tempname, "wb")
        if verbose:
            print(
                "[caching",
                stream,
                "at",
                self.tempname,
                "]",
                file=sys.stderr,
                flush=True,
            )

    def close(self, complete=False):
        """Close both the cache file and the original stream.

        :param complete: indicate whether the stream was fully read (if not, the cache file is discarded)
        """
        self.stream.close()
        if self.cache is not None:
            self.cache.close()
            self.cache = None
            if complete:
                assert os.path.exists(self.tempname), self.tempname
                os.rename(self.tempname, self.fname)
                if self.verbose:
                    print("[done caching", self.fname, "]", file=sys.stderr, flush=True)
        else:
            os.remove(self.tempname)

    def read(self, n):
        """Read n bytes from the stream and write them to the cache file.

        :param n: number of bytes
        """
        data = self.stream.read(n)
        self.cache.write(data)
        if data is None or len(data) < n:
            self.close(complete=True)
        self.last = ("read", n, len(data) if data is not None else None)
        return data

    def readinto(self, b):
        """Read data into a buffer.

        :param b: buffer
        """
        n = self.stream.readinto(b)
        self.cache.write(b[:n])
        if n == 0:
            self.close(complete=True)
        self.last = ("readinto", n)
        return n


def cache_shards(urls, cache_dir="./data", cache_size=1e15, cache_name=guess_shard, verbose=False):
    """Implement shard caching.

    When caching is off, just iterates through the list of shards.

    When caching is on (cache_dir is not None), opens each shard with caching
    an returns a dictionary consisting of a URL and a stream.

    :param urls: list of URLs
    :param cache_dir: directory used for caching
    :param cache_size: cache size
    :param cache_name: function computing cache names
    :param verbose: verbose caching info
    """
    global _cache
    if cache_dir is None:
        yield from urls
        return
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    for shard in urls:
        url = shard["url"]
        stream = shard["stream"]
        cache_path = os.path.join(cache_dir, cache_name(url))
        if not os.path.exists(cache_path):
            _cache = CacheStream(cache_path, stream, verbose=verbose)
            yield dict(url=url, stream=_cache)
        else:
            if verbose:
                print("[opening cached", cache_path, "]", file=sys.stderr, flush=True)
            yield dict(url=url, stream=open(cache_path, "rb"))
    if verbose:
        print(f"[finished {cache_path}]", file=sys.stderr, flush=True)

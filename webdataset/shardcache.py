import os.path
import io
import uuid
from itertools import islice


def guess_shard(path):
    return os.path.split(path.split()[-1])[1]


def shard_uuid(path):
    return str(uuid.uuid3(uuid.NAMESPACE_URL, path))


class CacheStream(io.RawIOBase):
    def __init__(self, fname, stream, tempsuffix=".TEMP", verbose=False):
        super().__init__()
        self.verbose = verbose
        self.stream = stream
        self.fname = fname
        self.tempsuffix = tempsuffix
        if verbose:
            print("[caching", stream, "at", fname, "]", file=sys.stderr)
        os.unlink(fname) if os.path.exists(fname) else None
        os.unlink(fname + tempsuffix) if os.path.exists(fname + tempsuffix) else None
        self.cache = open(fname + tempsuffix, "wb")

    def close(self, complete=False):
        self.stream.close()
        if self.cache is not None:
            self.cache.close()
            self.cache = None
        if complete:
            assert os.path.exists(self.fname + self.tempsuffix), (
                self.fname,
                self.tempsuffix,
            )
            os.rename(self.fname + self.tempsuffix, self.fname)
            if self.verbose:
                print("[done caching", self.fname, "]")
        else:
            os.remove(self.fname + self.tempsuffix)

    def read(self, n):
        data = self.stream.read(n)
        self.cache.write(data)
        if data is None or len(data) < n:
            self.close(complete=True)
        self.last = ("read", n, len(data) if data is not None else None)
        return data

    def readinto(self, b):
        n = self.stream.readinto(b)
        self.cache.write(b[:n])
        if n == 0:
            self.close(complete=True)
        self.last = ("readinto", n)
        return n


def cache_shards(urls, cachedir="./data", encode=guess_shard, verbose=False):
    global _cache
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    for shard in urls:
        url = shard["url"]
        stream = shard["stream"]
        cachepath = os.path.join(cachedir, encode(url))
        print(cachepath)
        if not os.path.exists(cachepath):
            _cache = CacheStream(cachepath, stream, verbose=verbose)
            yield dict(url=url, stream=_cache)
        else:
            if verbose:
                print("[opening cached", cachepath, "]", file=sys.stderr)
            yield dict(url=url, stream=open(cachepath, "rb"))
    print("streamcache done")

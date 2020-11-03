import sys
import os.path
import io
import uuid


def guess_shard(path):
    return os.path.split(path.split()[-1])[1]


def shard_uuid(path):
    return str(uuid.uuid3(uuid.NAMESPACE_URL, path))


class CacheStream(io.RawIOBase):
    def __init__(self, fname, stream, verbose=False):
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


def cache_shards(
    urls, cache_dir="./data", cache_size=1e15, cache_name=guess_shard, verbose=False
):
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

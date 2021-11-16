import itertools
import os
import random
import re
import sys

from urllib.parse import urlparse
from .handlers import reraise_exception
from . import filters, gopen
from .tariterators import tar_file_expander, group_by_keys


def lru_cleanup(cache_dir, cache_size, keyfn=os.path.getctime, verbose=False):
    """Performs cleanup of the file cache in cache_dir using an LRU strategy,
    keeping the total size of all remaining files below cache_size."""
    if not os.path.exists(cache_dir):
        return
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, filename))
    if total_size <= cache_size:
        return
    # sort files by last access time
    files = []
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    files.sort(key=keyfn, reverse=True)
    # delete files until we're under the cache size
    while len(files) > 0 and total_size > cache_size:
        fname = files.pop()
        total_size -= os.path.getsize(fname)
        if verbose:
            print("# deleting %s" % fname, file=sys.stderr)
        os.remove(fname)


def download(url, dest, chunk_size=1024**2, verbose=False):
    """Download a file from `url` to `dest`."""
    if verbose:
        print("# downloading %s to %s" % (url, dest), file=sys.stderr)
    with gopen.gopen(url) as stream:
        with open(dest, "wb") as f:
            while data := stream.read(chunk_size):
                f.write(data)


def pipe_cleaner(spec):
    """Guess the actual URL from a "pipe:" specification."""
    if spec.startswith("pipe:"):
        spec = spec[5:]
        words = spec.split(" ")
        for word in words:
            if re.match(r"^(https?|gs|ais|s3)", word):
                return word
    return spec
    

def cached_url_opener(data, handler=reraise_exception, cache_size=1e10, cache_dir="./_shardcache", url_to_name=pipe_cleaner, verbose=False):
    """Given a stream of url names (packaged in `dict(url=url)`), yield opened streams."""
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        parsed = urlparse(url)
        path = url_to_name(parsed.path)
        dirname, filename = os.path.split(path)
        dirname = re.sub(r"\W", "_", dirname)
        destdir = os.path.join(cache_dir, dirname)
        os.makedirs(destdir, exist_ok=True)
        dest = os.path.join(cache_dir, dirname, filename)
        try:
            if not os.path.exists(dest):
                lru_cleanup(cache_dir, cache_size, verbose=verbose)
                download(url, dest, verbose=verbose)
            assert os.path.exists(dest)
            stream = open(dest, "rb")
            sample.update(stream=stream)
            yield sample
        except Exception as exn:
            exn.args = exn.args + (url,)
            if handler(exn):
                continue
            else:
                break


def cached_tarfile_samples(src, handler=reraise_exception, cache_size=1e10, cache_dir="./data", verbose=False):
    streams = cached_url_opener(src, handler=handler, cache_size=cache_size, cache_dir=cache_dir, verbose=verbose, url_to_name=pipe_cleaner)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys(files, handler=handler)
    return samples


cached_tarfile_to_samples = filters.pipelinefilter(cached_tarfile_samples)
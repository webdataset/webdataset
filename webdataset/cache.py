"""Code related to caching files downloaded from storage
servers, object servers, and web servers.
"""

import io
import os
import random
import re
import sys
import time
import urllib.parse
from typing import Callable, Iterable, Optional
from urllib.parse import urlparse

import webdataset.gopen as gopen

from . import filters
from .handlers import reraise_exception
from .tariterators import group_by_keys, tar_file_expander
from .utils import obsolete

default_cache_dir = os.environ.get("WDS_CACHE", "./_cache")
default_cache_size = float(os.environ.get("WDS_CACHE_SIZE", "1e18"))

verbose_cache = int(os.environ.get("WDS_VERBOSE_CACHE", "0"))


def islocal(url):
    """Check whether a URL is a local file."""
    parsed = urlparse(url)
    return parsed.scheme in ["", "file"]


def get_filetype(fname: str):
    """Get the file type of a file."""
    assert os.path.exists(fname), fname
    assert os.system("file . > /dev/null") == 0, "UNIX/Linux file command not available"
    with os.popen("file '%s'" % fname) as f:
        ftype = f.read()
    return ftype


def check_tar_format(fname: str):
    """Check whether a file is a tar archive."""
    assert os.path.exists(fname), fname
    ftype = get_filetype(fname)
    return "tar archive" in ftype or "gzip compressed" in ftype


@obsolete
def pipe_cleaner(spec):
    """Guess the actual URL from a "pipe:" specification."""
    if spec.startswith("pipe:"):
        spec = spec[5:]
        words = spec.split(" ")
        for word in words:
            if re.match(r"^(https?|hdfs|gs|ais|s3):", word):
                return word
    return spec


def url_to_cache_name(url, ndir=0):
    """Guess the cache name from a URL."""
    assert isinstance(url, str)
    parsed = urlparse(url)
    if parsed.scheme in [
        None,
        "",
        "file",
        "http",
        "https",
        "ftp",
        "ftps",
        "gs",
        "s3",
        "ais",
    ]:
        path = parsed.path
        path = path.lstrip("/")  # always relative
        list_of_directories = path.split("/")
        return "/".join(list_of_directories[-1 - ndir :])
    else:
        # don't try to guess, just urlencode the whole thing with "/" and ":"
        # quoted using the urllib.quote function
        quoted = urllib.parse.quote(url, safe="_+{}*,-")
        quoted = quoted[-128:]
        return quoted


class LRUCleanup:
    """Perform LRU cleanup on a cache directory."""

    def __init__(
        self,
        cache_dir=None,
        cache_size=int(1e12),
        keyfn=os.path.getctime,
        verbose=False,
        interval=30,
    ):
        """Initialize the LRU cleanup object."""
        self.cache_dir = cache_dir
        self.cache_size = cache_size
        self.keyfn = keyfn
        self.verbose = verbose
        self.interval = interval
        self.last_run = 0

    def set_cache_dir(self, cache_dir):
        """Set the cache directory."""
        self.cache_dir = cache_dir

    def cleanup(self):
        """Performs cleanup of the file cache in cache_dir using an LRU strategy,
        keeping the total size of all remaining files below cache_size.
        """
        if not os.path.exists(self.cache_dir):
            return
        if self.interval is not None and time.time() - self.last_run < self.interval:
            return
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.cache_dir):
                for filename in filenames:
                    total_size += os.path.getsize(os.path.join(dirpath, filename))
            if total_size <= self.cache_size:
                return
            # sort files by last access time
            files = []
            for dirpath, dirnames, filenames in os.walk(self.cache_dir):
                for filename in filenames:
                    files.append(os.path.join(dirpath, filename))
            files.sort(key=self.keyfn, reverse=True)
            # delete files until we're under the cache size
            while len(files) > 0 and total_size > self.cache_size:
                fname = files.pop()
                total_size -= os.path.getsize(fname)
                if self.verbose:
                    print("# deleting %s" % fname, file=sys.stderr)
                os.remove(fname)
        except (OSError, FileNotFoundError):
            # files may be deleted by other processes between walking the directory and getting their size/deleting them
            pass
        self.last_run = time.time()


def download(url, dest, chunk_size=1024**2, verbose=False):
    """Download a file from `url` to `dest`."""
    temp = dest + f".temp{os.getpid()}"
    with gopen.gopen(url) as stream:
        with open(temp, "wb") as f:
            while True:
                data = stream.read(chunk_size)
                if not data:
                    break
                f.write(data)
    os.rename(temp, dest)


class StreamingOpen:
    """Open a stream from a URL."""

    def __init__(self, verbose=False, handler=reraise_exception):
        """Initialize the streaming open object."""
        self.verbose = verbose
        self.handler = handler

    def __call__(self, urls):
        """Open a stream from a URL."""
        for url in urls:
            if isinstance(url, dict):
                url = url["url"]
            parsed = urlparse(url)
            try:
                if parsed.scheme in ["", "file"]:
                    stream = open(parsed.path, "rb")
                    yield dict(url=url, stream=stream, local_path=parsed.path)
                else:
                    stream = gopen.gopen(url)
                    yield dict(url=url, stream=stream)
            except Exception as exn:
                if self.handler(exn):
                    continue
                else:
                    break


class FileCache:
    """Cache files from URLs.

    This class provides functionality to download and cache files from URLs,
    with options for validation, error handling, and cache management.

    Args:
        cache_dir (Optional[str]): The directory to use for caching. Defaults to None.
        url_to_name (Callable[[str], str]): Function to convert URLs to cache names.
        verbose (bool): Whether to print verbose output. Defaults to False.
        validator (Callable[[str], bool]): Function to validate downloaded files.
        handler (Callable[[Exception], bool]): Function to handle exceptions.
        cache_size (int): Maximum size of the cache in bytes. Defaults to -1 (unlimited).
        cache_cleanup_interval (int): Interval between cache cleanup operations in seconds.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        *,
        url_to_name: Callable[[str], str] = url_to_cache_name,
        verbose: bool = False,
        validator: Callable[[str], bool] = check_tar_format,
        handler: Callable[[Exception], bool] = reraise_exception,
        cache_size: int = -1,
        cache_cleanup_interval: int = 30,
    ):
        self.url_to_name = url_to_name
        self.validator = validator
        self.handler = handler
        if cache_dir is None:
            self.cache_dir = default_cache_dir
        else:
            self.cache_dir = cache_dir
        self.verbose = verbose
        if cache_size > 0:
            self.cleaner = LRUCleanup(
                self.cache_dir,
                cache_size,
                verbose=self.verbose,
                interval=cache_cleanup_interval,
            )
        else:
            self.cleaner = None

    def get_file(self, url: str) -> str:
        """Download a file from a given URL and return the path to the downloaded file.

        Args:
            url (str): The URL of the file to download.

        Returns:
            str: The path to the downloaded file.

        Raises:
            ValueError: If the downloaded file fails validation.
        """
        assert isinstance(url, str)
        if islocal(url):
            return urlparse(url).path
        cache_name = self.url_to_name(url)
        assert "/" not in cache_name, f"bad cache name {cache_name} for {url}"
        destdir = os.path.join(self.cache_dir, os.path.dirname(cache_name))
        os.makedirs(destdir, exist_ok=True)
        dest = os.path.join(self.cache_dir, cache_name)
        if not os.path.exists(dest):
            if self.verbose:
                print("# downloading %s to %s" % (url, dest), file=sys.stderr)
            if self.cleaner is not None:
                self.cleaner.cleanup()
            download(url, dest, verbose=self.verbose)
            if self.validator:
                if not self.validator(dest):
                    ftype = get_filetype(dest)
                    with open(dest, "rb") as f:
                        data = f.read(200)
                    os.remove(dest)
                    raise ValueError(
                        "%s (%s) is not a tar archive, but a %s, contains %s"
                        % (dest, url, ftype, repr(data))
                    )
        return dest

    def __call__(self, urls: Iterable[str]) -> Iterable[io.IOBase]:
        """Download files from a list of URLs and yield file streams.

        Args:
            urls (Iterable[str]): An iterable of URLs to download files from.

        Yields:
            dict: A dictionary containing the URL, file stream, and local path of each downloaded file.

        Raises:
            Exception: If there's an error downloading or opening a file.
        """
        for url in urls:
            if isinstance(url, dict):
                url = url["url"]
            for _ in range(10):
                try:
                    dest = self.get_file(url)
                    stream = open(dest, "rb")
                except Exception as e:
                    if self.handler(e):
                        continue
                    else:
                        break
                yield dict(url=url, stream=stream, local_path=dest)
                break


@obsolete
def cached_url_opener(
    data,
    handler=reraise_exception,
    cache_size=-1,
    cache_dir=None,
    url_to_name=pipe_cleaner,
    validator=check_tar_format,
    verbose=False,
    always=False,
):
    """Open streams for a sequence of URLs, with caching.

    Given a stream of URL names (packaged in `dict(url=url)`), yield opened streams.

    Args:
        data: An iterable of dictionaries containing URLs.
        handler: Function to handle exceptions. Defaults to reraise_exception.
        cache_size (int): Maximum size of the cache in bytes. Defaults to -1 (unlimited).
        cache_dir (Optional[str]): The directory to use for caching. Defaults to None.
        url_to_name: Function to convert URLs to cache names. Defaults to pipe_cleaner.
        validator: Function to validate downloaded files. Defaults to check_tar_format.
        verbose (bool): Whether to print verbose output. Defaults to False.
        always (bool): Whether to always download files, even if they exist locally. Defaults to False.

    Yields:
        dict: A dictionary containing the original sample data and an opened file stream.

    Raises:
        ValueError: If a downloaded file fails validation.
        Exception: For any other errors during download or file opening.
    """
    verbose = verbose or verbose_cache
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        attempts = 5
        try:
            if not always and os.path.exists(url):
                dest = url
            else:
                dest = get_file_cached(
                    url,
                    cache_size=cache_size,
                    cache_dir=cache_dir,
                    url_to_name=url_to_name,
                    verbose=verbose,
                )
            if verbose:
                print("# opening %s" % dest, file=sys.stderr)
            assert os.path.exists(dest)
            if not validator(dest):
                ftype = get_filetype(dest)
                with open(dest, "rb") as f:
                    data = f.read(200)
                os.remove(dest)
                raise ValueError(
                    "%s (%s) is not a tar archive, but a %s, contains %s"
                    % (dest, url, ftype, repr(data))
                )
            try:
                stream = open(dest, "rb")
                sample.update(stream=stream)
                yield sample
            except FileNotFoundError as exn:
                # dealing with race conditions in lru_cleanup
                attempts -= 1
                if attempts > 0:
                    time.sleep(random.random() * 10)
                    continue
                raise exn
        except Exception as exn:
            exn.args = exn.args + (url,)
            if handler(exn):
                continue
            else:
                break


@obsolete
def cached_tarfile_samples(
    src,
    handler=reraise_exception,
    cache_size=-1,
    cache_dir=None,
    verbose=False,
    url_to_name=pipe_cleaner,
    always=False,
    select_files=None,
    rename_files=None,
):
    """Process and yield samples from cached tar files.

    This function is obsolete.

    Args:
        src: An iterable source of URLs or dictionaries containing URLs.
        handler: Function to handle exceptions. Defaults to reraise_exception.
        cache_size (int): Maximum size of the cache in bytes. Defaults to -1 (unlimited).
        cache_dir (Optional[str]): The directory to use for caching. Defaults to None.
        verbose (bool): Whether to print verbose output. Defaults to False.
        url_to_name: Function to convert URLs to cache names. Defaults to pipe_cleaner.
        always (bool): Whether to always download files, even if they exist locally. Defaults to False.
        select_files: Function to select specific files from the tar archive. Defaults to None.
        rename_files: Function to rename files from the tar archive. Defaults to None.

    Returns:
        An iterable of samples extracted from the cached tar files.
    """
    verbose = verbose or int(os.environ.get("GOPEN_VERBOSE", 0))
    streams = cached_url_opener(
        src,
        handler=handler,
        cache_size=cache_size,
        cache_dir=cache_dir,
        verbose=verbose,
        url_to_name=url_to_name,
        always=always,
    )
    files = tar_file_expander(
        streams, handler=handler, select_files=select_files, rename_files=rename_files
    )
    samples = group_by_keys(files, handler=handler)
    return samples


cached_tarfile_to_samples = filters.pipelinefilter(cached_tarfile_samples)

"""Code related to caching files downloaded from storage
servers, object servers, and web servers.
"""

import io
import os
import re
import subprocess
import sys
import time
import urllib.parse
from typing import Callable, Iterable, Optional
from urllib.parse import urlparse

import webdataset.gopen as gopen

from .handlers import reraise_exception
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
    # Check if 'file' command is available
    try:
        subprocess.run(["file", "."], stdout=subprocess.DEVNULL, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        raise AssertionError("UNIX/Linux file command not available")
    # Run file command on the specified file
    result = subprocess.run(["file", fname], stdout=subprocess.PIPE, text=True, check=True)
    return result.stdout


def magic_filetype(fname: str):
    """Determine file type by checking magic numbers.

    It checks common formats used with WebDataset: tar archives and gzip files.

    Args:
        fname (str): Path to the file to check

    Returns:
        str: Description of the file type
    """
    with open(fname, "rb") as f:
        # Read the first 512 bytes for header checks
        header = f.read(512)

        # Check for gzip signature (begins with 1F 8B)
        if len(header) >= 2 and header[0:2] == b"\x1f\x8b":
            return f"{fname}: gzip compressed data"

        # Check for tar file signature (ustar at position 257-261)
        if len(header) > 261 and header[257:262] == b"ustar":
            return f"{fname}: POSIX tar archive"

        # Standard tar has a checksum at 148-156 and magic at 257-263
        # If we can't identify it specifically, return generic data
        return f"{fname}: data"


def check_tar_format(fname: str):
    """Check whether a file is a tar archive."""
    assert os.path.exists(fname), fname

    # Always use magic_filetype for file format detection
    ftype = magic_filetype(fname)
    return "tar archive" in ftype.lower() or "gzip compressed" in ftype.lower()


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

        This is a simple implementation that scans the directory twice - once to compute
        the total size and once to build a list of files for potential deletion. While
        not theoretically optimal for extremely large caches, it is efficient enough
        for practical purposes with typical cache sizes and file counts.
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
            if self.validator and not self.validator(dest):
                ftype = get_filetype(dest)
                with open(dest, "rb") as f:
                    data = f.read(200)
                os.remove(dest)
                raise ValueError("%s (%s) is not a tar archive, but a %s, contains %s" % (dest, url, ftype, repr(data)))
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
            delay = 1.0
            for _ in range(10):
                try:
                    dest = self.get_file(url)
                    stream = open(dest, "rb")
                except Exception as e:
                    if self.handler(e):
                        time.sleep(delay)
                        delay *= 1.5
                        continue
                    else:
                        break
                yield dict(url=url, stream=stream, local_path=dest)
                break

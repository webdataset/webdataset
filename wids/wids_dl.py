import fcntl
import os
import shutil
import sys
import time
from collections import deque
from urllib.parse import urlparse

import numpy as np

recent_downloads = deque(maxlen=1000)


class ULockFile:
    """A simple locking class. We don't need any of the third
    party libraries since we rely on POSIX semantics for linking
    below anyway."""

    def __init__(self, path):
        self.lockfile_path = path
        self.lockfile = None

    def __enter__(self):
        self.lockfile = open(self.lockfile_path, "w")
        fcntl.flock(self.lockfile.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        fcntl.flock(self.lockfile.fileno(), fcntl.LOCK_UN)
        self.lockfile.close()
        self.lockfile = None
        try:
            os.unlink(self.lockfile_path)
        except FileNotFoundError:
            pass


def pipe_download(remote, local):
    """Perform a download for a pipe: url."""
    assert remote.startswith("pipe:")
    cmd = remote[5:]
    cmd = cmd.format(local=local)
    assert os.system(cmd) == 0, "Command failed: %s" % cmd


def copy_file(remote, local):
    remote = urlparse(remote)
    assert remote.scheme in ["file", ""]
    # use absolute path
    remote = os.path.abspath(remote.path)
    local = urlparse(local)
    assert local.scheme in ["file", ""]
    local = os.path.abspath(local.path)
    if remote == local:
        return
    # check if the local file exists
    shutil.copyfile(remote, local)


verbose_cmd = int(os.environ.get("WIDS_VERBOSE_CMD", "0"))


def vcmd(flag, verbose_flag=""):
    return verbose_flag if verbose_cmd else flag


default_cmds = {
    "posixpath": copy_file,
    "file": copy_file,
    "pipe": pipe_download,
    "http": "curl " + vcmd("-s") + " -L {url} -o {local}",
    "https": "curl " + vcmd("-s") + " -L {url} -o {local}",
    "ftp": "curl " + vcmd("-s") + " -L {url} -o {local}",
    "ftps": "curl " + vcmd("-s") + " -L {url} -o {local}",
    "gs": "gsutil " + vcmd("-q") + " cp {url} {local}",
    "s3": "aws s3 cp {url} {local}",
}


def download_file_no_log(remote, local, handlers=default_cmds):
    """Download a file from a remote url to a local path.
    The remote url can be a pipe: url, in which case the remainder of
    the url is treated as a command template that is executed to perform the download.
    """

    if remote.startswith("pipe:"):
        schema = "pipe"
    else:
        schema = urlparse(remote).scheme
    if schema is None or schema == "":
        schema = "posixpath"
    # get the handler
    handler = handlers.get(schema)
    if handler is None:
        raise ValueError("Unknown schema: %s" % schema)
    # call the handler
    if callable(handler):
        handler(remote, local)
    else:
        assert isinstance(handler, str)
        cmd = handler.format(url=remote, local=local)
        assert os.system(cmd) == 0, "Command failed: %s" % cmd
    return local


def download_file(remote, local, handlers=default_cmds, verbose=False):
    start = time.time()
    try:
        return download_file_no_log(remote, local, handlers=handlers)
    finally:
        recent_downloads.append((remote, local, time.time(), time.time() - start))
        if verbose:
            print(
                "downloaded",
                remote,
                "to",
                local,
                "in",
                time.time() - start,
                "seconds",
                file=sys.stderr,
            )


def download_and_open(remote, local, mode="rb", handlers=default_cmds, verbose=False):
    with ULockFile(local + ".lock"):
        if not os.path.exists(local):
            if verbose:
                print("downloading", remote, "to", local, file=sys.stderr)
            download_file(remote, local, handlers=handlers)
        else:
            if verbose:
                print("using cached", local, file=sys.stderr)
        return open(local, mode)


def keep_most_recent_files(directory, maxsize=int(1e12), maxfiles=1000):
    """Keep the most recent files in a directory, deleting the rest.

    The maxsize is the maximum size of the directory in bytes. The maxfiles is
    the maximum number of files to keep. The files are sorted by modification
    time, and the most recent files are kept. If the directory is already
    smaller than maxsize, then no files are deleted. If there are fewer than
    maxfiles, then no files are deleted."""

    # get the list of files in the directory
    files = os.listdir(directory)
    # compute a list of (mtime, fname, size) triples
    files = [
        (
            os.stat(os.path.join(directory, fname)).st_mtime,
            fname,
            os.stat(os.path.join(directory, fname)).st_size,
        )
        for fname in files
    ]
    # sort the list by mtime, most recent first
    files.sort(reverse=True)
    # compute an accumulated total of the file sizes in order using np.cumsum
    sizes = np.cumsum([size for mtime, fname, size in files])
    # compute a cutoff index based on maxsize
    cutoff = np.searchsorted(sizes, maxsize)
    # compute a cutoff index based on maxfiles
    cutoff = min(cutoff, maxfiles)
    # delete the files above the cutoff in reverse order
    for mtime, fname, size in files[cutoff:][::-1]:
        try:
            os.unlink(os.path.join(directory, fname))
        except FileNotFoundError:
            pass


class DirectoryCleanup:
    def __init__(self, directory, every=10, maxsize=int(1e12), maxfiles=100000):
        self.directory = directory
        self.maxsize = maxsize
        self.maxfiles = maxfiles
        self.every = every
        # create a .last_cleanup file whose mtime reflects the last
        # time we ran a cleanup
        self.last_cleanup = os.path.join(directory, ".last_cleanup")
        if not os.path.exists(self.last_cleanup):
            with open(self.last_cleanup, "w"):
                pass
    def run_cleanup(self):
        """Run a cleanup if the .last_cleanup file is old enough."""
        if time.time() - os.stat(self.last_cleanup).st_mtime > self.every:
            keep_most_recent_files(
                self.directory, maxsize=self.maxsize, maxfiles=self.maxfiles
            )
            with open(self.last_cleanup, "w"):
                pass
        

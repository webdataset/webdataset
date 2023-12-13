import multiprocessing
import os
import random
import sys
import time
from typing import List, Optional

import braceexpand
import typer

from . import gopen
from .utils import fnmatch_with_braces, glob_with_braces

app = typer.Typer()


def download_file(url, filename):
    """Download a file from a URL."""
    return wids_dl.download_file(url, filename)


def download_file(url, filename):
    """Download a file from a URL."""
    with gopen.gopen(url, "rb") as stream:
        with open(filename, "wb") as out:
            while True:
                chunk = stream.read(1024 * 1024)
                if len(chunk) == 0:
                    break
                out.write(chunk)


def download_with(command):
    """Return a function that downloads a file with the given command.

    The command must contain {url} and {output} placeholders.
    """

    def download(url, filename):
        return subprocess.check_call(
            command.format(url=url, output=filename), shell=True
        )

    return download


def total_file_size(files):
    """Return the total size of a list of files."""
    return sum(os.path.getsize(f) for f in files)


def file_of_tempfile(tempfile):
    """Return the file name corresponding to a tempfile."""
    assert tempfile.endswith("_") and "." in tempfile
    return tempfile.rsplit(".", 1)[0]


def get_oldest_file(files):
    """Return the oldest file in a list of files."""
    return min(files, key=os.path.getmtime)


class RandomShardDownloader:
    """Download shards randomly from a source to directory.

    This can be run in one of two modes:

    - update_every: keep filling the directory with shards until it contains nshards shards;
                    does not remove shards (gpu job removes shards)
    - replace_every: keep filling the directory with shards until it contains nshards shards;
                    removes a shard every polling period (gpu job samples with replacement)
    """

    def __init__(
        self,
        shards,
        nshards,
        *,
        directory=None,
        pattern="*.{tar,tgz,tar.gz}",
        increment=999999,
        maxsize=999999999999,
        verbose=False,
        download=None,
        errors="ignore",  # ignore, warn, fail
    ):
        """Initialize the downloader with the given parameters."""
        self.shards = shards
        self.directory = directory
        self.nshards = nshards
        self.pattern = pattern
        self.increment = increment
        self.errors = errors
        self.maxsize = maxsize
        self.verbose = verbose
        if isinstance(download, str):
            download = download_with(download)
        self.download = download or download_file
        # check that the file name components of the shards match the glob pattern; use fnmatch
        for shard in shards:
            assert fnmatch_with_braces(
                os.path.basename(shard), pattern
            ), f"shard {os.path.basename(shard)} does not match pattern {pattern}"

    def list_files(self, inactive=False):
        """Return a list of files matching the given pattern.

        Files with temporary file name patterns ("*._*_") are also included and
        mapped to their names without the temporary extension.
        """
        files = glob_with_braces(os.path.join(self.directory, self.pattern))
        if not inactive:
            tempfiles = glob_with_braces(os.path.join(self.directory, "*._*_"))
            files += [file_of_tempfile(tempfile) for tempfile in tempfiles]
        return list(set(files))

    def set_directory(self, directory):
        """Set the directory to download to."""
        self.directory = directory

    def update(self):
        """Download shards randomly from a source to directory.

        Ensure that there are nshards shards in the directory.
        If there are fewer, download random shards from the source.
        Use the basename of the shards as the local file name.
        """
        assert self.directory is not None, "directory must be set"
        files = self.list_files()
        start = len(files)
        for _ in range(10 * self.nshards):
            files = self.list_files()
            total_size = total_file_size(files)
            if (
                len(files) >= min(self.nshards, start + self.increment)
                or total_size > self.maxsize
            ):
                return
            shard = random.choice(self.shards)
            filename = os.path.basename(shard)
            if filename in files:
                continue
            tempdest = os.path.join(self.directory, filename + f"._{os.getpid()}_")
            if self.verbose:
                print(f"downloading {shard} to {tempdest}", file=sys.stderr)
            try:
                self.download(shard, tempdest)
            except Exception as exn:
                print(f"download failed: {exn}", file=sys.stderr)
                if self.errors == "ignore":
                    continue
                if self.errors == "warn":
                    print(f"ignoring error {exn}", file=sys.stderr)
                    continue
                raise
            try:
                os.rename(tempdest, os.path.join(self.directory, filename))
            except FileExistsError:
                # some other process downloaded the same file
                os.unlink(tempdest)

        raise RuntimeError(f"unable to download {self.nshards} shards")

    def sleep(self, poll=10):
        delta = poll * random.uniform(0.7, 1.3)
        time.sleep(delta)

    def update_every(self, poll=10):
        """Repeatedly call update with the given delay."""
        while True:
            self.update()
            delta = poll * random.uniform(0.7, 1.3)
            time.sleep(delta)

    def maybe_remove(self, strategy="oldest"):
        files = self.list_files()
        if len(files) > self.nshards:
            inactive = self.list_files(inactive=True)
            if len(inactive) == 0:
                return False
            if strategy == "oldest":
                selected = get_oldest_file(inactive)
            elif strategy == "random":
                selected = random.choice(inactive)
            else:
                raise ValueError(f"unknown strategy {strategy}")
            try:
                os.unlink(selected)
                return True
            except FileNotFoundError:
                return False
        return False

    def replace_every(self, poll=60, strategy="oldest"):
        """Repeatedly call update with the given delay."""
        while len(self.list_files()) >= self.nshards:
            if self.maybe_remove(strategy=strategy):
                self.update()
            self.sleep(poll)

    def run_job(self, poll=10, mode="update", strategy="oldest"):
        if mode == "update":
            self.update_every(poll)
        elif mode == "replace":
            self.replace_every(poll, strategy=strategy)
        else:
            raise ValueError(f"unknown mode {mode}")


@app.command()
def random_downloader(
    shards: List[str],
    *,
    directory: Optional[str] = None,
    nshards: int = 10,
    command: Optional[str] = None,
    pattern: str = "*.{tar,tgz,tar.gz}",
    increment: int = 999999,
    maxsize: int = 999999999999,
    njobs: int = 1,
    poll: float = 10,
    mode: str = "update",
    errors: str = "ignore",
    verbose: bool = False,
):
    """Start njobs jobs to download shards randomly from the given list of shards."""
    assert directory is not None
    shards = [fname for shard in shards for fname in braceexpand.braceexpand(shard)]
    print(f"got {len(shards)} shards", file=sys.stderr)
    if njobs > 1:
        pool = multiprocessing.Pool(njobs)
        for _ in range(njobs):
            pool.apply_async(
                RandomShardDownloader(
                    shards,
                    nshards,
                    directory=directory,
                    pattern=pattern,
                    increment=increment,
                    maxsize=maxsize,
                    download=command,
                    errors=errors,
                    verbose=verbose,
                ).run_job,
                kwds=dict(poll=poll, mode=mode),
            )
    else:
        RandomShardDownloader(
            shards,
            nshards,
            directory=directory,
            pattern=pattern,
            increment=increment,
            maxsize=maxsize,
            download=command,
            errors=errors,
            verbose=verbose,
        ).run_job(poll=poll, mode=mode)


if __name__ == "__main__":
    app()

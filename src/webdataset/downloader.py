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
    """Download a file from a URL.

    Args:
        url (str): The URL of the file to download.
        filename (str): The local path where the file will be saved.

    Returns:
        None
    """
    with gopen.gopen(url, "rb") as stream:
        with open(filename, "wb") as out:
            while True:
                chunk = stream.read(1024 * 1024)
                if len(chunk) == 0:
                    break
                out.write(chunk)


def download_with(command):
    """Create a download function using a custom command.

    Args:
        command (str): The command to use for downloading, containing {url} and {output} placeholders.

    Returns:
        function: A function that takes a URL and filename as arguments and downloads the file.
    """

    def download(url, filename):
        return subprocess.check_call(
            command.format(url=url, output=filename), shell=True
        )

    return download


def total_file_size(files):
    """Calculate the total size of a list of files.

    Args:
        files (list): A list of file paths.

    Returns:
        int: The total size of all files in bytes.
    """
    return sum(os.path.getsize(f) for f in files)


def file_of_tempfile(tempfile):
    """Get the original file name from a temporary file name.

    Args:
        tempfile (str): The temporary file name.

    Returns:
        str: The original file name without the temporary suffix.

    Raises:
        AssertionError: If the tempfile doesn't end with '_' or doesn't contain a period.
    """
    assert tempfile.endswith("_") and "." in tempfile
    return tempfile.rsplit(".", 1)[0]


def get_oldest_file(files):
    """Find the oldest file in a list of files.

    Args:
        files (list): A list of file paths.

    Returns:
        str: The path of the oldest file.
    """
    return min(files, key=os.path.getmtime)


class RandomShardDownloader:
    """Download shards randomly from a source to a directory.

    This class can be run in two modes:
    - update_every: Keep filling the directory with shards until it contains nshards shards.
    - replace_every: Keep filling the directory with shards, removing a shard every polling period.

    Args:
        shards (list): List of shard URLs to download from.
        nshards (int): Number of shards to maintain in the directory.
        directory (str, optional): Directory to download shards to.
        pattern (str, optional): Glob pattern for matching shard files.
        increment (int, optional): Maximum number of shards to add in one update.
        maxsize (int, optional): Maximum total size of downloaded shards in bytes.
        verbose (bool, optional): Whether to print verbose output.
        download (function, optional): Custom download function to use.
        errors (str, optional): Error handling strategy ('ignore', 'warn', or 'fail').

    Raises:
        AssertionError: If a shard filename doesn't match the given pattern.
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
        """List files in the download directory matching the given pattern.

        Args:
            inactive (bool, optional): Whether to include only inactive (non-temporary) files.

        Returns:
            list: A list of file paths matching the pattern.
        """
        files = glob_with_braces(os.path.join(self.directory, self.pattern))
        if not inactive:
            tempfiles = glob_with_braces(os.path.join(self.directory, "*._*_"))
            files += [file_of_tempfile(tempfile) for tempfile in tempfiles]
        return list(set(files))

    def set_directory(self, directory):
        """Set the directory to download shards to.

        Args:
            directory (str): The path to the download directory.
        """
        self.directory = directory

    def update(self):
        """Download shards randomly from the source to the directory.

        Ensures that there are nshards shards in the directory. If there are fewer,
        download random shards from the source.

        Raises:
            RuntimeError: If unable to download the required number of shards.
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
        """Sleep for a randomized duration based on the poll interval.

        Args:
            poll (float, optional): The base polling interval in seconds.
        """
        delta = poll * random.uniform(0.7, 1.3)
        time.sleep(delta)

    def update_every(self, poll=10):
        """Repeatedly call update with a given delay.

        Args:
            poll (float, optional): The polling interval in seconds.
        """
        while True:
            self.update()
            delta = poll * random.uniform(0.7, 1.3)
            time.sleep(delta)

    def maybe_remove(self, strategy="oldest"):
        """Attempt to remove a shard if the number of shards exceeds the limit.

        Args:
            strategy (str, optional): The strategy for selecting which file to remove ('oldest' or 'random').

        Returns:
            bool: True if a file was removed, False otherwise.

        Raises:
            ValueError: If an unknown strategy is provided.
        """
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
        """Repeatedly update and remove shards to maintain the desired number.

        Args:
            poll (float, optional): The polling interval in seconds.
            strategy (str, optional): The strategy for selecting which file to remove.
        """
        while len(self.list_files()) >= self.nshards:
            if self.maybe_remove(strategy=strategy):
                self.update()
            self.sleep(poll)

    def run_job(self, poll=10, mode="update", strategy="oldest"):
        """Run the downloader job in the specified mode.

        Args:
            poll (float, optional): The polling interval in seconds.
            mode (str, optional): The mode to run in ('update' or 'replace').
            strategy (str, optional): The strategy for selecting which file to remove in replace mode.

        Raises:
            ValueError: If an unknown mode is provided.
        """
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
    """Start multiple jobs to download shards randomly from the given list of shards.

    Args:
        shards (List[str]): List of shard URLs or patterns to download from.
        directory (Optional[str]): Directory to download shards to.
        nshards (int): Number of shards to maintain in the directory.
        command (Optional[str]): Custom download command to use.
        pattern (str): Glob pattern for matching shard files.
        increment (int): Maximum number of shards to add in one update.
        maxsize (int): Maximum total size of downloaded shards in bytes.
        njobs (int): Number of parallel download jobs to run.
        poll (float): Polling interval in seconds.
        mode (str): Mode to run in ('update' or 'replace').
        errors (str): Error handling strategy ('ignore', 'warn', or 'fail').
        verbose (bool): Whether to print verbose output.

    Raises:
        AssertionError: If the directory is not provided.
    """
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

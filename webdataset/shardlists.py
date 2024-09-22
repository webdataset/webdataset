#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Train PyTorch models directly from POSIX tar archive.

Code works locally or over HTTP connections.
"""

import glob
import os
import os.path
import random
import re
import sys
import time
from dataclasses import dataclass, field
from itertools import islice
from typing import List

import braceexpand
import yaml

from . import utils
from .filters import pipelinefilter
from .pytorch import IterableDataset
from .utils import obsolete


def envlookup(m):
    """Look up match in the environment with prefix WDS_.

    Args:
        m: A match object.

    Returns:
        str: The value of the environment variable WDS_<m.group(1)>.

    Raises:
        AssertionError: If the environment variable is not found.
    """
    key = m.group(1)
    key = "WDS_" + key
    assert key in os.environ, f"missing environment variable wds_{key}"
    return os.environ[key]


def envsubst(s):
    """Substitute ${var} with the value of the environment variable WDS_var.

    Args:
        s (str): String to be substituted.

    Returns:
        str: The substituted string.
    """
    return re.sub(r"\$\{(\w+)\}", envlookup, s)


def split_by_node(src, group=None):
    """Split the input sequence by PyTorch distributed rank.

    Args:
        src: The input sequence to be split.
        group: The process group for distributed training.

    Yields:
        Elements from the input sequence based on the node's rank.
    """
    rank, world_size, worker, num_workers = utils.pytorch_worker_info(group=group)
    if world_size > 1:
        yield from islice(src, rank, None, world_size)
    else:
        yield from src


def single_node_only(src, group=None):
    """Ensure the input sequence is not split for multi-node training.

    Args:
        src: The input sequence.
        group: The process group for distributed training.

    Yields:
        Elements from the input sequence.

    Raises:
        ValueError: If multi-node training is detected.
    """
    rank, world_size, worker, num_workers = utils.pytorch_worker_info(group=group)
    if world_size > 1:
        raise ValueError(
            "you need to add an explicit nodesplitter to your input pipeline for multi-node training"
        )
    yield from src


def split_by_worker(src):
    """Split the input sequence by PyTorch DataLoader worker.

    Args:
        src: The input sequence to be split.

    Yields:
        Elements from the input sequence based on the worker's ID.
    """
    rank, world_size, worker, num_workers = utils.pytorch_worker_info()
    if num_workers > 1:
        yield from islice(src, worker, None, num_workers)
    else:
        yield from src


def expand_urls(urls):  # sourcery skip: for-index-underscore, last-if-guard
    """Expand the urls if they are a string.

    If input is a string:
    - split on '::'
    - expand environment variables (using WDS_ prefix)
    - expand braces

    Otherwise:
    - return the input as a list

    Args:
        urls (str or List[str]): URL list or URL string.

    Returns:
        List[str]: List of expanded URLs.
    """
    urllist = urls.split("::")
    result = []
    for url in urllist:
        for i in range(10):
            last = url
            url = envsubst(url)
            if url == last:
                break
        result.extend(braceexpand.braceexpand(url))
    return result


def expand_source(source, max_urls=int(1e9)):
    """Expand the given source into a list of URLs.

    Args:
        source (str or List[str] or Iterable): The source to be expanded.
        max_urls (int): Maximum number of URLs to return.

    Returns:
        List[str]: List of expanded URLs.

    Raises:
        ValueError: If the source type is not supported.
    """
    if isinstance(source, str):
        return expand_urls(source)
    elif isinstance(source, list):
        return source
    elif is_iterable(source):
        return list(islice(source, max_urls))
    else:
        raise ValueError(f"cannot handle {type(source)}")


class SimpleShardList(IterableDataset):
    """An iterable dataset yielding a list of URLs."""

    def __init__(self, urls, seed=None):
        """Initialize the SimpleShardList.

        Args:
            urls (str or List[str]): A list of URLs as a Python list or brace notation string.
            seed (int or bool or None): Random seed for shuffling; if None, no shuffling is done,
                if True, a random seed is generated.
        """
        super().__init__()
        if isinstance(urls, str):
            urls = expand_urls(urls)
        else:
            urls = list(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        if seed is True:
            seed = time.time()
        self.seed = seed

    def __len__(self):
        """Return the number of URLs in the list.

        Returns:
            int: The number of URLs.
        """
        return len(self.urls)

    def __iter__(self):
        """Return an iterator over the shards.

        Yields:
            dict: A dictionary containing the URL of each shard.
        """
        urls = self.urls.copy()
        if self.seed is not None:
            random.Random(self.seed).shuffle(urls)
        for url in urls:
            yield dict(url=url)


def resampled_(src, n=sys.maxsize):
    """Resample items from the source with replacement.

    Args:
        src: The source iterable.
        n (int): The number of items to yield. Defaults to sys.maxsize.

    Yields:
        Randomly chosen items from the source.
    """
    import random

    seed = time.time()
    try:
        seed = open("/dev/random", "rb").read(20)
    except Exception as exn:
        print(repr(exn)[:50], file=sys.stderr)
    rng = random.Random(seed)
    print("# resampled loading", file=sys.stderr)
    items = list(src)
    print(f"# resampled got {len(items)} samples, yielding {n}", file=sys.stderr)
    for _ in range(n):
        yield rng.choice(items)


resampled = pipelinefilter(resampled_)


def non_empty(src):
    """Ensure the source yields at least one item.

    Args:
        src: The source iterable.

    Yields:
        Items from the source.

    Raises:
        ValueError: If the source yields no items.
    """
    count = 0
    for s in src:
        yield s
        count += 1
    if count == 0:
        raise ValueError(
            "pipeline stage received no data at all and this was declared as an error"
        )


@dataclass
class MSSource:
    """Class representing a data source."""

    name: str = ""
    perepoch: int = -1
    resample: bool = False
    urls: List[str] = field(default_factory=list)


default_rng = random.Random()


def expand(s):
    """Expand user and environment variables in a string.

    Args:
        s (str): The string to expand.

    Returns:
        str: The expanded string.
    """
    return os.path.expanduser(os.path.expandvars(s))


class ResampledShards(IterableDataset):
    """An iterable dataset yielding a list of URLs with resampling."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        seed=0,
        worker_seed=None,
        deterministic=False,
        max_urls=int(1e6),
        empty_check=True,
    ):
        """Initialize the ResampledShards.

        Args:
            urls: A list of URLs as a Python list or brace notation string.
            nshards (int): The number of shards to yield. Defaults to sys.maxsize.
            seed (int): The seed for random number generation.
            worker_seed (Callable or None): A function to generate worker-specific seeds.
            deterministic (bool): Whether to use deterministic sampling.
            max_urls (int): Maximum number of URLs to consider.
            empty_check (bool): Whether to check for empty URL list.

        Raises:
            ValueError: If empty_check is True and no shards are found.
        """
        super().__init__()
        self.urls = expand_source(urls, max_urls)
        if empty_check:
            if len(self.urls) == 0:
                raise ValueError(
                    "empty_check=True, but no shards found in ResampledShards"
                )
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.worker_seed = (
            utils.pytorch_worker_seed if worker_seed is None else worker_seed
        )
        self.deterministic = deterministic
        self.seed = seed
        self.epoch = -1

    def __iter__(self):
        """Return an iterator over the shards.

        Yields:
            dict: A dictionary containing the URL of each shard.
        """
        self.epoch += 1
        if self.deterministic:
            seed = utils.make_seed(self.worker_seed(), self.epoch, self.seed)
        else:
            seed = utils.make_seed(
                self.worker_seed(),
                self.epoch,
                self.seed,
                os.getpid(),
                time.time_ns(),
                os.urandom(4),
            )
        if os.environ.get("WDS_SHOW_SEED", "0") == "1":
            print(f"# ResampledShards seed {seed}")
        self.rng = random.Random(seed)
        for _ in range(self.nshards):
            index = self.rng.randint(0, len(self.urls) - 1)
            yield dict(url=self.urls[index])


ResampledShardList = ResampledShards


def check_pid_is_running(pid):
    """Check for the existence of a Unix PID.

    Args:
        pid (int): The process ID to check.

    Returns:
        bool: True if the process is running, False otherwise.
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def without_last_extension(fname):
    """Remove the last extension from a filename.

    Args:
        fname (str): The filename to process.

    Returns:
        str: The filename without the last extension.
    """
    return re.sub(r"\.[^.]*$", "", fname)


def get_pid_from_filename(fname):
    """Get the PID from a filename.

    Args:
        fname (str): The filename to process.

    Returns:
        int or None: The PID if found in the filename, None otherwise.
    """
    match = re.match(r"^(.*)\._(\d+)_$", fname)
    if not match:
        return None
    return int(match.group(2))


class DirectoryShardList(IterableDataset):
    """An iterable dataset that yields shards from a directory."""

    def __init__(
        self,
        path,
        pattern="*.{tar,tgz,tar.tgz}",
        poll=1,
        timeout=1e12,
        mode="resample",
        select="random",
        fate=None,
    ):
        """Initialize the DirectoryShardList.

        Args:
            path (str): The directory path to monitor for shards.
            pattern (str): The glob pattern to match shard files.
            poll (int): The polling interval in seconds.
            timeout (float): The maximum time to wait for new shards.
            mode (str): The mode for handling processed shards.
            select (str): The strategy for selecting shards.
            fate (Any): Currently unused parameter.
        """
        assert path.endswith("/")
        assert os.path.isdir(path)
        self.path = path
        self.poll = poll
        self.pattern = pattern
        self.mode = mode
        self.select = select
        self.fate = fate
        self.timeout = timeout

    def recycle(self, activename):
        """Recycle a processed shard based on the current mode.

        Args:
            activename (str): The name of the active shard file.
        """
        if self.mode == "unlink":
            os.unlink(activename)
        elif self.mode == "keep":
            os.rename(activename, without_last_extension(activename) + "._done_")
        elif self.mode == "resample":
            os.rename(activename, without_last_extension(activename))

    def cleanup_files_without_processes(self):
        """Clean up shard files associated with non-existent processes."""
        for fname in glob.glob(os.path.join(self.path, "*._*_")):
            pid = get_pid_from_filename(fname)
            if pid is None:
                continue
            if not check_pid_is_running(pid):
                self.recycle(fname)

    def __iter__(self):
        """Iterate over the shards in the directory.

        Yields:
            dict: A dictionary containing the URL of each shard.
        """
        last = time.time()
        while time.time() - last < self.timeout:
            candidates = sorted(glob.glob(self.path + self.pattern))
            if len(candidates) == 0:
                if self.poll is None:
                    return
                time.sleep(self.poll)
                continue

            if self.select == "oldest":
                candidate = min(candidates, key=lambda fn: os.stat(fn).st_mtime)
            elif self.select == "random":
                candidate = random.choice(candidates)
            else:
                raise ValueError(f"unknown selection strategy {self.select}")

            activename = candidate + f"._{os.getpid()}_"
            try:
                os.rename(candidate, activename)
            except FileNotFoundError as exn:
                time.sleep(self.poll)
                continue

            yield dict(url=activename)

            self.recycle(activename)
            self.cleanup_files_without_processes()


class MultiShardSample(IterableDataset):
    """An iterable dataset that samples from multiple shard sources."""

    @obsolete(reason="this is going to be replaced with the WIDS JSON format")
    def __init__(self, fname):
        """Initialize the MultiShardSample.

        Args:
            fname (str or dict): The filename of the YAML spec or a dictionary containing the spec.
        """
        self.epoch = -1
        self.parse_spec(fname)

    def parse_spec(self, fname):
        """Parse the specification for multiple shard sources.

        Args:
            fname (str or dict): The filename of the YAML spec or a dictionary containing the spec.
        """
        self.rng = default_rng  # capture default_rng if we fork
        if isinstance(fname, dict):
            spec = fname
            fname = "{dict}"
        else:
            with open(fname) as stream:
                spec = yaml.safe_load(stream)
        assert set(spec.keys()).issubset(set("prefix datasets buckets".split())), list(
            spec.keys()
        )
        prefix = expand(spec.get("prefix", ""))
        self.sources = []
        for ds in spec["datasets"]:
            assert set(ds.keys()).issubset(
                set("buckets name shards resample choose".split())
            ), list(ds.keys())
            buckets = ds.get("buckets", spec.get("buckets", []))
            if isinstance(buckets, str):
                buckets = [buckets]
            buckets = [expand(s) for s in buckets]
            if buckets == []:
                buckets = [""]
            assert (
                len(buckets) == 1
            ), f"{buckets}: FIXME support for multiple buckets unimplemented"
            bucket = buckets[0]
            name = ds.get("name", "@" + bucket)
            urls = ds["shards"]
            if isinstance(urls, str):
                urls = [urls]
            # urls = [u for url in urls for u in braceexpand.braceexpand(url)]
            urls = [
                prefix + os.path.join(bucket, u)
                for url in urls
                for u in braceexpand.braceexpand(expand(url))
            ]
            resample = ds.get("resample", -1)
            nsample = ds.get("choose", -1)
            if nsample > len(urls):
                raise ValueError(
                    f"perepoch {nsample} must be no greater than the number of shards"
                )
            if (nsample > 0) and (resample > 0):
                raise ValueError("specify only one of perepoch or choose")
            entry = MSSource(name=name, urls=urls, perepoch=nsample, resample=resample)
            self.sources.append(entry)
            print(f"# {name} {len(urls)} {nsample}", file=sys.stderr)

    def set_epoch(self, seed):
        """Set the current epoch for consistent shard selection among nodes.

        Args:
            seed (int): The seed for the random number generator.
        """
        self.rng = random.Random(seed)

    def get_shards_for_epoch(self):
        """Get the list of shards for the current epoch.

        Returns:
            list: A list of shard URLs for the current epoch.
        """
        result = []
        for source in self.sources:
            if source.resample > 0:
                # sample with replacement
                l = self.rng.choices(source.urls, k=source.resample)
            elif source.perepoch > 0:
                # sample without replacement
                l = list(source.urls)
                self.rng.shuffle(l)
                l = l[: source.perepoch]
            else:
                l = list(source.urls)
            result += l
        self.rng.shuffle(result)
        return result

    def __iter__(self):
        """Iterate over the shards for the current epoch.

        Yields:
            dict: A dictionary containing the URL of each shard.
        """
        shards = self.get_shards_for_epoch()
        for shard in shards:
            yield dict(url=shard)


def shardspec(spec):
    """Create a shard list based on the given specification.

    Args:
        spec (str): The specification for creating the shard list.

    Returns:
        IterableDataset: Either a MultiShardSample or a SimpleShardList based on the spec.
    """
    if spec.endswith(".yaml"):
        return MultiShardSample(spec)
    else:
        return SimpleShardList(spec)

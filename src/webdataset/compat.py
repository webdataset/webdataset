import os
import random
import warnings
from types import SimpleNamespace
from urllib.parse import urlparse

import yaml

from . import autodecode, cache, filters, shardlists, utils
from .filters import pipelinefilter, reraise_exception
from .pipeline import DataPipeline
from .pytorch import DataLoader
from .tariterators import group_by_keys, tar_file_expander


class FluidInterface:
    def batched(
        self, batchsize, collation_fn=filters.default_collation_fn, partial=True
    ):
        """Create batches of the given size.

        This method forwards to the filters.batched function.

        Args:
            batchsize (int): Target batch size.
            collation_fn (callable, optional): Function to collate samples into a batch.
                Defaults to filters.default_collation_fn.
            partial (bool, optional): Whether to return partial batches. Defaults to True.

        Returns:
            FluidInterface: Updated pipeline with batched filter.
        """
        return self.compose(
            filters.batched(batchsize, collation_fn=collation_fn, partial=partial)
        )

    def unbatched(self):
        """Turn batched data back into unbatched data.

        This method forwards to the filters.unbatched function.

        Returns:
            FluidInterface: Updated pipeline with unbatched filter.
        """
        return self.compose(filters.unbatched())

    def listed(self, batchsize, partial=True):
        """Create lists of samples without collation.

        This method forwards to the filters.batched function with collation_fn set to None.

        Args:
            batchsize (int): Target list size.
            partial (bool, optional): Whether to return partial lists. Defaults to True.

        Returns:
            FluidInterface: Updated pipeline with listed filter.
        """
        return self.compose(filters.batched(batchsize=batchsize, collation_fn=None))

    def unlisted(self):
        """Turn listed data back into individual samples.

        This method forwards to the filters.unlisted function.

        Returns:
            FluidInterface: Updated pipeline with unlisted filter.
        """
        return self.compose(filters.unlisted())

    def log_keys(self, logfile=None):
        """Log keys of samples passing through the pipeline.

        This method forwards to the filters.log_keys function.

        Args:
            logfile (str, optional): Path to the log file. If None, logging is disabled.

        Returns:
            FluidInterface: Updated pipeline with log_keys filter.
        """
        return self.compose(filters.log_keys(logfile))

    def shuffle(self, size, **kw):
        """Shuffle the data in the stream.

        This method forwards to the filters.shuffle function if size > 0.

        Args:
            size (int): Buffer size for shuffling.
            **kw: Additional keyword arguments for filters.shuffle.

        Returns:
            FluidInterface: Updated pipeline with shuffle filter, or self if size < 1.
        """
        if size < 1:
            return self
        else:
            return self.compose(filters.shuffle(size, **kw))

    def map(self, f, handler=reraise_exception):
        """Apply a function to each sample in the stream.

        This method forwards to the filters.map function.

        Args:
            f (callable): Function to apply to each sample.
            handler (callable, optional): Exception handler. Defaults to reraise_exception.

        Returns:
            FluidInterface: Updated pipeline with map filter.
        """
        return self.compose(filters.map(f, handler=handler))

    def decode(
        self,
        *args,
        pre=None,
        post=None,
        only=None,
        partial=False,
        handler=reraise_exception,
    ):
        """Decode data based on the decoding functions given as arguments.

        This method creates a decoder using autodecode.Decoder and applies it using filters.map.

        Args:
            *args: Decoding functions or strings representing image handlers.
            pre (callable, optional): Pre-processing function.
            post (callable, optional): Post-processing function.
            only (list, optional): List of keys to decode.
            partial (bool, optional): Whether to allow partial decoding. Defaults to False.
            handler (callable, optional): Exception handler. Defaults to reraise_exception.

        Returns:
            FluidInterface: Updated pipeline with decode filter.
        """
        handlers = [
            autodecode.ImageHandler(x) if isinstance(x, str) else x for x in args
        ]
        decoder = autodecode.Decoder(
            handlers, pre=pre, post=post, only=only, partial=partial
        )
        return self.map(decoder, handler=handler)

    def map_dict(self, handler=reraise_exception, **kw):
        """Map the entries in a dict sample with individual functions.

        This method forwards to the filters.map_dict function.

        Args:
            handler (callable, optional): Exception handler. Defaults to reraise_exception.
            **kw: Mapping of keys to functions to apply.

        Returns:
            FluidInterface: Updated pipeline with map_dict filter.
        """
        return self.compose(filters.map_dict(handler=handler, **kw))

    def select(self, predicate, **kw):
        """Select samples based on a predicate.

        This method forwards to the filters.select function.

        Args:
            predicate (callable): Function that returns True for samples to keep.
            **kw: Additional keyword arguments for filters.select.

        Returns:
            FluidInterface: Updated pipeline with select filter.
        """
        return self.compose(filters.select(predicate, **kw))

    def to_tuple(self, *args, **kw):
        """Convert dict samples to tuples.

        This method forwards to the filters.to_tuple function.

        Args:
            *args: Keys to extract from the dict.
            **kw: Additional keyword arguments for filters.to_tuple.

        Returns:
            FluidInterface: Updated pipeline with to_tuple filter.
        """
        return self.compose(filters.to_tuple(*args, **kw))

    def map_tuple(self, *args, handler=reraise_exception):
        """Map the entries of a tuple with individual functions.

        This method forwards to the filters.map_tuple function.

        Args:
            *args: Functions to apply to each element of the tuple.
            handler (callable, optional): Exception handler. Defaults to reraise_exception.

        Returns:
            FluidInterface: Updated pipeline with map_tuple filter.
        """
        return self.compose(filters.map_tuple(*args, handler=handler))

    def slice(self, *args):
        """Slice the data stream.

        This method forwards to the filters.slice function.

        Args:
            *args: Arguments for slicing (start, stop, step).

        Returns:
            FluidInterface: Updated pipeline with slice filter.
        """
        return self.compose(filters.slice(*args))

    def rename(self, **kw):
        """Rename samples based on keyword arguments.

        This method forwards to the filters.rename function.

        Args:
            **kw: Mapping of old names to new names.

        Returns:
            FluidInterface: Updated pipeline with rename filter.
        """
        return self.compose(filters.rename(**kw))

    def rsample(self, p=0.5):
        """Randomly subsample a stream of data.

        This method forwards to the filters.rsample function.

        Args:
            p (float, optional): Probability of keeping each sample. Defaults to 0.5.

        Returns:
            FluidInterface: Updated pipeline with rsample filter.
        """
        return self.compose(filters.rsample(p))

    def rename_keys(self, *args, **kw):
        """Rename keys in samples based on patterns.

        This method forwards to the filters.rename_keys function.

        Args:
            *args: Positional arguments for filters.rename_keys.
            **kw: Keyword arguments for filters.rename_keys.

        Returns:
            FluidInterface: Updated pipeline with rename_keys filter.
        """
        return self.compose(filters.rename_keys(*args, **kw))

    def extract_keys(self, *args, **kw):
        """Extract specific keys from samples.

        This method forwards to the filters.extract_keys function.

        Args:
            *args: Keys or patterns to extract.
            **kw: Additional keyword arguments for filters.extract_keys.

        Returns:
            FluidInterface: Updated pipeline with extract_keys filter.
        """
        return self.compose(filters.extract_keys(*args, **kw))

    def xdecode(self, *args, **kw):
        """Decode data based on file extensions.

        This method forwards to the filters.xdecode function.

        Args:
            *args: Positional arguments for filters.xdecode.
            **kw: Keyword arguments for filters.xdecode.

        Returns:
            FluidInterface: Updated pipeline with xdecode filter.
        """
        return self.compose(filters.xdecode(*args, **kw))

    def mcached(self):
        """Cache samples in memory.

        This method forwards to the filters.Cached class.

        Returns:
            FluidInterface: Updated pipeline with memory caching.
        """
        return self.compose(filters.Cached())

    def lmdb_cached(self, *args, **kw):
        """Cache samples using LMDB.

        This method forwards to the filters.LMDBCached class.

        Args:
            *args: Positional arguments for filters.LMDBCached.
            **kw: Keyword arguments for filters.LMDBCached.

        Returns:
            FluidInterface: Updated pipeline with LMDB caching.
        """
        return self.compose(filters.LMDBCached(*args, **kw))


def check_empty(source):
    """Check if the dataset is empty and yield samples.

    Args:
        source: An iterable source of samples.

    Yields:
        The samples from the source.

    Raises:
        ValueError: If no samples are found in the dataset.
    """
    count = 0
    for sample in source:
        yield sample
        count += 1
    if count == 0:
        raise ValueError(
            "No samples found in dataset; perhaps you have fewer shards than workers.\n"
            + "Turn off using empty_check=False in the WebDataset constructor."
        )


class WebDataset(DataPipeline, FluidInterface):
    """Create a WebDataset pipeline for efficient data loading.

    This class sets up a data pipeline for loading and processing WebDataset-format data.
    It handles URL generation, shard shuffling, caching, and sample grouping.

    Args:
        urls: The source URLs or specifications for the dataset.
        handler: Function to handle exceptions. Defaults to reraise_exception.
        mode: The mode of operation. Defaults to None.
        resampled: Whether to use resampled mode. Defaults to False.
        repeat: Whether to repeat the dataset. Defaults to False.
        shardshuffle: The number of shards to shuffle, or None. Defaults to None.
        cache_size: The size of the cache in bytes. Defaults to -1 (unlimited).
        cache_dir: The directory to use for caching. Defaults to None.
        url_to_name: Function to convert URLs to cache names. Defaults to pipe_cleaner.
        detshuffle: Whether to use deterministic shuffling. Defaults to False.
        nodesplitter: Function to split data by node. Defaults to single_node_only.
        workersplitter: Function to split data by worker. Defaults to split_by_worker.
        select_files: Function to select files from tar archives. Defaults to None.
        rename_files: Function to rename files from tar archives. Defaults to None.
        empty_check: Whether to check for empty datasets. Defaults to True.
        verbose: Whether to print verbose output. Defaults to False.
        seed: Random seed for shuffling. Defaults to None.

    Raises:
        ValueError: If the cache directory does not exist or if the URL type is not supported.
    """

    def __init__(
        self,
        urls,
        handler=reraise_exception,
        mode=None,
        resampled=False,
        repeat=False,
        shardshuffle=None,
        cache_size=-1,
        cache_dir=None,
        url_to_name=cache.pipe_cleaner,
        detshuffle=False,
        nodesplitter=shardlists.single_node_only,
        workersplitter=shardlists.split_by_worker,
        select_files=None,
        rename_files=None,
        empty_check=True,
        verbose=False,
        seed=None,
    ):
        super().__init__()
        if resampled:
            mode = "resampled"
        if mode == "resampled" and shardshuffle not in (False, None):
            warnings.warn(
                "WebDataset(shardshuffle=...) is ignored for resampled datasets"
            )
        elif shardshuffle is None:
            warnings.warn(
                "WebDataset(shardshuffle=...) is None; set explicitly to False or a number"
            )
        if shardshuffle is True:
            warnings.warn(
                "set WebDataset(shardshuffle=...) to a positive integer or 0 or False"
            )
            shardshuffle = 100
        args = SimpleNamespace(**locals())
        self.seed = (
            os.environ.get("WDS_SEED", random.randint(0, 1000000))
            if seed is None
            else seed
        )
        self.update_cache_info(args)

        # first, we add a generator for the urls to used
        # this generates a stream of dict(url=...)
        self.create_url_iterator(args)

        # split by node (for distributed processing)
        if nodesplitter is not None:
            self.append(nodesplitter)

        # split by worker (for DataLoader)
        if workersplitter:
            self.append(workersplitter)

        # add a shard shuffler
        if args.shardshuffle is not None:
            if args.detshuffle:
                self.append(filters.detshuffle(args.shardshuffle, seed=self.seed))
            else:
                self.append(filters.shuffle(args.shardshuffle, seed=self.seed))

        # next, we select a URL opener, either with or without caching
        # this generates a stream of dict(url=..., stream=...)
        if cache_dir is None or cache_size == 0:
            opener = cache.StreamingOpen(handler=handler)
        else:
            opener = cache.FileCache(
                cache_dir=cache_dir, cache_size=cache_size, handler=handler
            )
        self.append(opener)

        # now we need to open each stream and read the tar files contained in it
        # this generates a stream of dict(fname=..., data=...) objects
        expander = pipelinefilter(tar_file_expander)
        self.append(
            expander(
                handler=handler, select_files=select_files, rename_files=rename_files
            )
        )

        # finally, the files need to be groups into samples
        # this generates a stream of dict(__key__=..., ...=...) objects
        grouper = pipelinefilter(group_by_keys)
        self.append(grouper(handler=handler))

        # check for empty datasets
        if empty_check:
            self.append(check_empty)

    def update_cache_info(self, args):
        """Update cache information based on arguments and environment variables.

        Args:
            args: A SimpleNamespace object containing the arguments.

        Raises:
            ValueError: If the specified cache directory does not exist.
        """
        args.cache_size = int(os.environ.get("WDS_CACHE_SIZE", args.cache_size))
        args.cache_dir = os.environ.get("WDS_CACHE", args.cache_dir)
        if args.cache_dir is not None:
            args.cache_dir = os.path.expanduser(args.cache_dir)
            if not os.path.exists(args.cache_dir):
                raise ValueError(f"cache directory {args.cache_dir} does not exist")

    def create_url_iterator(self, args):
        """Create an appropriate URL iterator based on the input type.

        This method determines the type of URL input and creates the corresponding
        iterator for the dataset.

        Args:
            args: A SimpleNamespace object containing the arguments.

        Raises:
            ValueError: If the URL type is not supported or implemented.
        """
        urls = args.urls

        # .yaml specification files
        if isinstance(urls, str) and (urls.endswith(".yaml") or urls.endswith(".yml")):
            with open(args.urls) as stream:
                spec = yaml.safe_load(stream)
            assert "datasets" in spec
            self.append(shardlists.MultiShardSample(spec))
            return

        # .yaml specifications already loaded as dictionaries
        if isinstance(args.urls, dict):
            assert "datasets" in args.urls
            self.append(shardlists.MultiShardSample(args.urls))
            return

        # .json specification files (from wids)
        if isinstance(urls, str) and urls.endswith(".json"):
            raise ValueError("unimplemented")

        # any URL ending in "/" is assumed to be a directory
        if isinstance(urls, str) and urlparse(urls).path.endswith("/"):
            self.append(shardlists.DirectoryShardList(urls, mode=args.mode))
            return

        # the rest is either a shard list or a resampled shard list
        if isinstance(args.urls, str) or utils.is_iterable(args.urls):
            if args.mode == "resampled":
                self.append(shardlists.ResampledShardList(args.urls))
            else:
                self.append(shardlists.SimpleShardList(args.urls))
            return

        raise ValueError(f"cannot handle urls of type {type(args.urls)}")

    def __enter__(self):
        """Enter the runtime context for the WebDataset.

        Returns:
            self: The WebDataset instance.
        """
        return self

    def __exit__(self, *args):
        """Exit the runtime context for the WebDataset.

        Args:
            *args: Exception type, value, and traceback if an exception occurred.
        """
        self.close()


class FluidWrapper(DataPipeline, FluidInterface):
    """Small fluid-interface wrapper for DataPipeline."""

    def __init__(self, initial):
        super().__init__()
        self.append(initial)


class WebLoader(DataPipeline, FluidInterface):
    """A wrapper for DataLoader that adds a fluid interface."""

    def __init__(self, *args, **kw):
        super().__init__(DataLoader(*args, **kw))

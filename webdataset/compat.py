import os
import random
from types import SimpleNamespace
from urllib.parse import urlparse
import warnings

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
        return self.compose(
            filters.batched(batchsize, collation_fn=collation_fn, partial=partial)
        )

    def unbatched(self):
        return self.compose(filters.unbatched())

    def listed(self, batchsize, partial=True):
        return self.compose(filters.batched(batchsize=batchsize, collation_fn=None))

    def unlisted(self):
        return self.compose(filters.unlisted())

    def log_keys(self, logfile=None):
        return self.compose(filters.log_keys(logfile))

    def shuffle(self, size, **kw):
        if size < 1:
            return self
        else:
            return self.compose(filters.shuffle(size, **kw))

    def map(self, f, handler=reraise_exception):
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
        handlers = [
            autodecode.ImageHandler(x) if isinstance(x, str) else x for x in args
        ]
        decoder = autodecode.Decoder(
            handlers, pre=pre, post=post, only=only, partial=partial
        )
        return self.map(decoder, handler=handler)

    def map_dict(self, handler=reraise_exception, **kw):
        return self.compose(filters.map_dict(handler=handler, **kw))

    def select(self, predicate, **kw):
        return self.compose(filters.select(predicate, **kw))

    def to_tuple(self, *args, **kw):
        return self.compose(filters.to_tuple(*args, **kw))

    def map_tuple(self, *args, handler=reraise_exception):
        return self.compose(filters.map_tuple(*args, handler=handler))

    def slice(self, *args):
        return self.compose(filters.slice(*args))

    def rename(self, **kw):
        return self.compose(filters.rename(**kw))

    def rsample(self, p=0.5):
        return self.compose(filters.rsample(p))

    def rename_keys(self, *args, **kw):
        return self.compose(filters.rename_keys(*args, **kw))

    def extract_keys(self, *args, **kw):
        return self.compose(filters.extract_keys(*args, **kw))

    def xdecode(self, *args, **kw):
        return self.compose(filters.xdecode(*args, **kw))

    def mcached(self):
        return self.compose(filters.Cached())

    def lmdb_cached(self, *args, **kw):
        return self.compose(filters.LMDBCached(*args, **kw))
    

def check_empty(source):
    count = 0
    for sample in source:
        yield sample
        count += 1
    if count == 0:
        raise ValueError("No samples found in dataset; perhaps you have fewer shards than workers.\n" +
                         "Turn off using empty_check=False in the WebDataset constructor.")


class WebDataset(DataPipeline, FluidInterface):
    """Small fluid-interface wrapper for DataPipeline."""

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
        if shardshuffle is None:
            warnings.warn("WebDataset(shardshuffle=...) is None; set explicitly to False or a number")
        if shardshuffle is True:
            shardshuffle = 100
        args = SimpleNamespace(**locals())
        self.seed = seed or os.environ.get("WDS_SEED", random.randint(0, 1000000))
        self.update_cache_info(args)

        # first, we add a generator for the urls to used
        # this generates a stream of dict(url=...)
        self.create_url_iterator(args)

        # split by node (for distributed processing)
        if nodesplitter is not None:
            self.append(nodesplitter)

        # split by worker (for DataLoader)
        if workersplitter:
            self.append(shardlists.split_by_worker)

        # add a shard shuffler
        if args.shardshuffle is not None:
            if args.detshuffle:
                self.append(filters.detshuffle(args.shardshuffle, seed=args.seed))
            else:
                self.append(filters.shuffle(args.shardshuffle, seed=args.seed))

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
        """Compute the correct cache directory and size from the arguments and environment."""
        args.cache_size = int(os.environ.get("WDS_CACHE_SIZE", args.cache_size))
        args.cache_dir = os.environ.get("WDS_CACHE", args.cache_dir)
        if args.cache_dir is not None:
            args.cache_dir = os.path.expanduser(args.cache_dir)
            if not os.path.exists(args.cache_dir):
                raise ValueError(f"cache directory {args.cache_dir} does not exist")

    def create_url_iterator(self, args):
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
            self.append(shardlists.DirectoryShardlist(urls, mode=args.mode))
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
        return self

    def __exit__(self, *args):
        self.close()


class FluidWrapper(DataPipeline, FluidInterface):
    """Small fluid-interface wrapper for DataPipeline."""

    def __init__(self, initial):
        super().__init__()
        self.append(initial)


class WebLoader(DataPipeline, FluidInterface):
    def __init__(self, *args, **kw):
        super().__init__(DataLoader(*args, **kw))

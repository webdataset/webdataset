import os
import random

import yaml

from . import autodecode, cache, filters, shardlists
from .filters import pipelinefilter, reraise_exception
from .pipeline import DataPipeline
from .pytorch import DataLoader, IterableDataset
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
        return self.compose(filters.batched(), batchsize=batchsize, collation_fn=None)

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

    def to_tuple(self, *args, handler=reraise_exception):
        return self.compose(filters.to_tuple(*args, handler=handler))

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


class WebDataset(DataPipeline, FluidInterface):
    """Small fluid-interface wrapper for DataPipeline."""

    def __init__(
        self,
        urls,
        handler=reraise_exception,
        resampled=False,
        repeat=False,
        shardshuffle=None,
        cache_size=-1,
        cache_dir=None,
        url_to_name=cache.pipe_cleaner,
        detshuffle=False,
        nodesplitter=shardlists.single_node_only,
        select_files=None,
        rename_files=None,
        verbose=False,
        seed=None,
    ):
        super().__init__()
        self.seed = seed or os.environ.get("WDS_SEED", random.randint(0, 1000000))
        cache_size = int(os.environ.get("WDS_CACHE_SIZE", cache_size))
        cache_dir = os.environ.get("WDS_CACHE", cache_dir)
        if cache_dir is not None:
            cache_dir = os.path.expanduser(cache_dir)
            if not os.path.exists(cache_dir):
                raise ValueError(f"cache directory {cache_dir} does not exist")
        if isinstance(urls, IterableDataset):
            assert not resampled
            self.append(urls)
        elif isinstance(urls, str) and (
            urls.endswith(".yaml") or urls.endswith(".yml")
        ):
            with open(urls) as stream:
                spec = yaml.safe_load(stream)
            assert "datasets" in spec
            self.append(shardlists.MultiShardSample(spec))
        elif isinstance(urls, dict):
            assert "datasets" in urls
            self.append(shardlists.MultiShardSample(urls))
        elif resampled:
            self.append(shardlists.ResampledShards(urls, seed=self.seed))
        else:
            self.append(shardlists.SimpleShardList(urls))
            self.append(nodesplitter)
            self.append(shardlists.split_by_worker)
            if shardshuffle is True:
                shardshuffle = 100
            if shardshuffle is not None:
                if detshuffle:
                    self.append(filters.detshuffle(shardshuffle, seed=self.seed))
                else:
                    self.append(filters.shuffle(shardshuffle, seed=self.seed))
        # self.append(filters.info(name="shard"))
        if cache_dir is None or cache_size == 0:
            self.append(cache.StreamingOpen(handler=handler))
        else:
            self.append(cache.FileCache(cache_dir, cache_size, handler=handler))
        # self.append(filters.info(name="opened"))
        self.append(
            pipelinefilter(tar_file_expander)(
                handler=handler, select_files=select_files, rename_files=rename_files
            )
        )
        # self.append(filters.info(name="expanded"))
        self.append(pipelinefilter(group_by_keys)(handler=handler))

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

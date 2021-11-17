import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List
from itertools import islice

import braceexpand
import yaml
import copy

from . import filters, shardlists, tariterators, autodecode, cache
from .filters import reraise_exception
from .pytorch import DataLoader, IterableDataset
from .pipeline import DataPipeline


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
        return self.compose(iterators.batched, batchsize=batchsize, collation_fn=None)

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

    def decode(self, *args, pre=None, post=None, only=None, handler=reraise_exception):
        handlers = [
            autodecode.ImageHandler(x) if isinstance(x, str) else x for x in args
        ]
        decoder = autodecode.Decoder(handlers, pre=pre, post=post, only=only)
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


class WebDataset(DataPipeline, FluidInterface):
    """Small fluid-interface wrapper for DataPipeline."""

    def __init__(
        self,
        urls,
        handler=reraise_exception,
        resampled=False,
        repeat=False,
        shardshuffle=None,
        caching=None,
        detshuffle=False,
        nodesplitter=shardlists.single_node_only,
        verbose=False,
    ):
        super().__init__()
        if isinstance(urls, IterableDataset):
            assert not resampled
            self.append(urls)
        elif resampled:
            self.append(shardlists.ResampledShards(urls))
        else:
            self.append(shardlists.SimpleShardList(urls))
            self.append(nodesplitter)
            self.append(shardlists.split_by_worker)
            if shardshuffle is not None:
                if detshuffle:
                    self.append(filters.detshuffle(shardshuffle))
                else:
                    self.append(filters.shuffle(shardshuffle))
        if caching is None or caching is False:
            self.append(tariterators.tarfile_to_samples(handler=handler))
        elif caching is True:
            self.append(
                cache.cached_tarfile_to_samples(handler=handler, verbose=verbose)
            )
        else:
            dir, size = caching
            self.append(
                cache.cached_tarfile_to_samples(
                    handler=handler,
                    cache_dir=dir,
                    cache_size=size,
                    verbose=verbose,
                )
            )


class WebLoader(DataPipeline, FluidInterface):
    def __init__(self, *args, **kw):
        super().__init__(DataLoader(*args, **kw))

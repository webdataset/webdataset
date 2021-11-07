#%%
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List

import braceexpand
import yaml

from . import filters
from .filters import pipelinefilter, reraise_exception
from .pytorch import DataLoader, IterableDataset


def stage(f, *args, **kw):
    return pipelinefilter(f)(*args, **kw)


class DataPipeline(IterableDataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pipeline = []
        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, list):
                self.pipeline.extend(arg)
            else:
                self.pipeline.append(arg)

    def invoke(self, f, *args, **kwargs):
        if isinstance(f, (IterableDataset, DataLoader)) and len(args) == 0:
            return iter(f)
        if isinstance(f, list):
            return iter(f)
        if callable(f):
            result = f(*args, **kwargs)
            return result
        raise ValueError(f"{f}: not a valid pipeline stage")

    def __iter__(self):
        source = self.invoke(self.pipeline[0])
        for step in self.pipeline[1:]:
            source = self.invoke(step, source)
        return source

    def stage(self, i):
        return self.pipeline[i]

    def append(self, f):
        self.pipeline.append(f)

    def compose(self, f):
        result = copy.copy(self)
        result.append(f)
        return result

class WebDataset(DataPipeline):
    def __init__(self, urls, handler=reraise_exception, resampled=False, repeat=False, shardshuffle=None):
        super().__init__()
        if resampled:
            self.append(ResampledShards(urls))
        else:
            self.append(shardlists.SimpleShardList(urls))
            self.append(shardlists.split_by_node)
            self.append(shardlists.split_by_worker)
            if shardshuffle is not None:
                self.append(filters.shuffle(shardshuffle))
        self.append(tariterators.tarfile_samples)

    def batched(self, batchsize, collation_fn=filters.default_collation_fn, partial=True):
        return self.compose(filters.batch(batchsize, collation_fn=collation_fn, partial=partial))

    def unbatched(self):
        return self.compose(filters.unbatch())

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
        decoder = autodecode.Decoder(handlers, pre=pre, post=post, only=only)
        return self.map(decoder, handler=handler)

    def map_dict(self, handler=reraise_exception, **kw):
        return self.compose(iterators.map_dict, handler=handler, _kwa=kw)

    def select(self, predicate, **kw):
        return self.compose(filters.select(predicate, **kw))

    def to_tuple(self, *args, handler=reraise_exception):
        return self.compose(filters.to_tuple(*args, handler=handler))

    def map_tuple(self, *args, handler=reraise_exception):
        return self.compose(filters.map_tuple(*args, handler=handler))

    def slice(self, *args):
        return self.compose(filters.slice(*args))

    def rsample(self, p=0.5):
        return self.compose(filters.rsample(p))

    def repeat(self, nepochs=None, nbatches=None):
        return self.compose(Repeatedly(nepochs, nbatches))

    def with_epoch(self, n):
        return eds.with_epoch(self, n)

    def with_length(selfn, n):
        return eds.with_length(self, n)

#%%
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

from . import extradatasets as eds
from . import filters, shardlists, tariterators, autodecode
from .filters import pipelinefilter, reraise_exception
from .pytorch import DataLoader, IterableDataset
from .utils import PipelineStage


def stage(f, *args, **kw):
    return pipelinefilter(f)(*args, **kw)


def add_length_method(obj):
    def length(self):
        return self.size
    Combined = type(obj.__class__.__name__ + "_Length", (obj.__class__, IterableDataset), {"__len__": length})
    obj.__class__ = Combined
    return obj

class DataPipeline(IterableDataset, PipelineStage):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pipeline = []
        self.length = -1
        self.repetitions = 1
        self.nsamples = -1
        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, list):
                self.pipeline.extend(arg)
            else:
                self.pipeline.append(arg)

    def invoke(self, f, *args, **kwargs):
        if isinstance(f, PipelineStage):
            return f.run(*args, **kwargs)
        if isinstance(f, (IterableDataset, DataLoader)) and len(args) == 0:
            return iter(f)
        if isinstance(f, list):
            return iter(f)
        if callable(f):
            result = f(*args, **kwargs)
            return result
        raise ValueError(f"{f}: not a valid pipeline stage")

    def iterator1(self):
        source = self.invoke(self.pipeline[0])
        for step in self.pipeline[1:]:
            source = self.invoke(step, source)
        return source

    def iterator(self):
        for i in range(self.repetitions):
            for sample in self.iterator1():
                yield sample

    def __iter__(self):
        if self.repetitions != 1:
            if self.nsamples > 0:
                return islice(self.iterator(), self.nsamples)
            else:
                return self.iterator()
        else:
            return self.iterator()

    def stage(self, i):
        return self.pipeline[i]

    def append(self, f):
        self.pipeline.append(f)

    def compose(self, f):
        result = copy.copy(self)
        result.append(f)
        return result

    def with_length(self, n):
        self.size = n
        return add_length_method(self)

    def with_epoch(self, nsamples):
        self.repetitions = sys.maxsize
        self.nsamples = nsamples
        return self

    def repeat(self, nepochs=-1, nbatches=-1):
        if nepochs > 0:
            self.repetitions = nepochs
            self.nsamples = nbatches
        else:
            self.repetitions = sys.maxsize
            self.nsamples = nbatches
        return self


class ResampledDataset(DataPipeline):
    def __init__(self, urls):
        pipeline = [eds.ResampledShards(urls), tariterators.tarfile_samples]
        super().__init__(*pipeline)


class TarShards(DataPipeline):
    def __init__(self, urls, shuffle=None):
        pipeline = [
            shardlists.SimpleShardList(urls),
            tariterators.tarfile_samples,
            shardlists.split_by_node,
            shardlists.split_by_worker,
        ]
        if shuffle:
            pipeline.append(filters.shuffle(shuffle))
        super().__init__(*pipeline)


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
        handlers = [autodecode.ImageHandler(x) if isinstance(x, str) else x for x in args]
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
    ):
        super().__init__()
        if isinstance(urls, IterableDataset):
            assert not resampled
            self.append(urls)
        elif resampled:
            self.append(shardlists.ResampledShards(urls))
        else:
            self.append(shardlists.SimpleShardList(urls))
            self.append(shardlists.split_by_node)
            self.append(shardlists.split_by_worker)
            if shardshuffle is not None:
                self.append(filters.shuffle(shardshuffle))
        self.append(tariterators.tarfile_to_samples(handler=handler))


class WebLoader(DataPipeline, FluidInterface):
    def __init__(self, *args, **kw):
        super().__init__(DataLoader(*args, **kw))

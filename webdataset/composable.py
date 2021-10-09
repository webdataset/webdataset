#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Train PyTorch models directly from POSIX tar archive.

Code works locally or over HTTP connections.
"""

import itertools as itt
import os
import sys

from . import autodecode, dbcache, iterators
from .utils import lookup_sym, safe_eval
from .handlers import reraise_exception
from .pytorch import IterableDataset

default_cache_dir = os.path.expanduser(os.environ.get("WEBDATASET_CACHE", ""))
default_cache_name = lookup_sym(os.environ.get("WEBDATASET_CACHE_NAME", "shard_uuid"), ".shardcache".split())
default_cache_verbose = int(safe_eval(os.environ.get("WEBDATASET_CACHE_VERBOSE", "1")))
default_cache_size = int(float(safe_eval(os.environ.get("WEBDATASET_CACHE_SIZE", "1e15"))))


class Composable:
    """A mixin implementing composability of data pipelines."""

    def __init__(self):
        """Initialize the composable mixin."""
        super().__init__()

    def source_(self, source):
        """Set the source for this dataset.

        :param source: source dataset, should be an IterableDataset instance
        """
        self.source = source
        return self

    def then(self, f, *args, **kw):
        """Compose this processor with a new processor defined by a function.

        The function is of the form:

            def my_process(source, ...):
                for sample in source:
                    ...
                    result = ...
                    yield result
        """
        assert callable(f)
        assert "source" not in kw
        # print("Processor", args, kw)
        return Processor(self, f, *args, **kw)

    def compose(self, constructor, *args, **kw):
        """Compose this processor with another IterableDataset.

        The constructor should be of the form `__init__(self, source_dataset, ...)`
        """
        assert callable(constructor)
        return constructor(*args, **kw).source_(self)


class Shorthands:
    """A convenient set of shorthands for common data transformations."""

    def batched(self, batchsize, collation_fn=iterators.default_collation_fn, partial=True):
        """Compute batches for the given dataset.

        :param batchsize: desired batchsize
        :param collation_fn: collation function to turn list of objects into batches
        :param partial: return partial batches
        """
        return self.then(iterators.batched, batchsize=batchsize, collation_fn=collation_fn, partial=partial)

    def unbatched(self):
        """Take a stream of batches and turn it back into a stream of samples."""
        return self.then(iterators.unbatched)

    def listed(self, batchsize, partial=True):
        """Compute batches by just putting collections of samples into a list; analogous to batched."""
        return self.batched(batchsize, collation_fn=None, partial=partial)

    def unlisted(self):
        """Take a stream of batches and turn it back into a stream of samples."""
        return self.then(iterators.unlisted)

    def log_keys(self, logfile=None):
        """Log keys from current samples to the given logfile (used for debugging)."""
        return self.then(iterators.log_keys, logfile=logfile)

    def shuffle(self, size, **kw):
        """Shuffle the dataset using an internal shuffle buffer.

        This will buffer up `initial` samples. Then it will add new samples to
        the internal buffer and return random samples from the buffer, simultaneously
        filling up the buffer to the given size.

        Using initial < size will result in less initial randomness but faster
        startups.

        :param size: size of the shuffle buffer
        :param initial: buffer this many samples before yield training samples
        :param handler: The exception handling strategy.
        :param kw: other keywords for iterators.shuffle
        """
        if size < 1:
            return self
        return self.then(iterators.shuffle, size, **kw)

    def map(self, f, handler=reraise_exception):
        """Map a function over a stream of samples.

        This may be a tuple stream or a stream of dicts.

        :param f: The function to be mapped.
        :param handler: The exception handling strategy.
        """
        return self.then(iterators.map, f, handler=handler)

    def decode(
        self,
        *args,
        pre=None,
        post=None,
        only=None,
        handler=reraise_exception,
    ):
        """Decode samples.

        This is a special form of mapping over samples given as dicts.
        A list of all decoders is formed from `pre + args + post`.
        For each dict entry, the decoders on that list are invoked in sequence
        until one of them decodes the sample. That decoded value is then stored
        in the dictionary and the next dictionary entry is decoded.

        The `pre` and `post` decoder lists are set to common defaults (including `.gz` decoding).
        You can specify decoders for your application in the `args` argument.
        String arguments like "pil" are a shorthand for image decoder functions like
        `webdataset.imagehandler("pil")`. All other decoders must be specified as
        functions.

        :param args: list of decoder functions; a string like "pil" is a shorthand for common image decoders
        :param pre: a list of decoder functions that is always carried out before args
        :param post: a list of decoder functions that is always carried out after args
        :param only: limit decoding to the list of these fields
        :param handler: exception handler
        """
        # for backwards compatibility
        handlers = [autodecode.ImageHandler(h) if isinstance(h, str) else h for h in args]
        decoder = autodecode.Decoder(handlers, pre=pre, post=post, only=only)
        return self.map(decoder, handler=handler)

    def rename(self, handler=reraise_exception, **kw):
        """Rename fields in a dictionary based sample.

        This works on dictionary input streams. A keyword argument like
        `new="old"` renames extension/key "old" to "new".

        :param handler: exception handler
        :param kw: list of renames
        """
        return self.then(iterators.rename, handler=handler, _kwa=kw)

    def map_dict(self, handler=reraise_exception, **kw):
        """Map the fields of a dictionary.

        :param handler: exeption handler
        :param kw: list of key=function mappers
        """
        return self.then(iterators.map_dict, handler=handler, _kwa=kw)

    def select(self, predicate, **kw):
        """Select samples matching some predicate.

        :param predicate: predicate used to select samples
        """
        return self.then(iterators.select, predicate, _kwa=kw)

    def to_tuple(self, *args, handler=reraise_exception, **kw):
        """Convert a dictionary-based sample to a tuple.

        Field names to be extracted can be specified as a Python list
        or as a string. "__key__ jpg;png cls" will extract a triple, with the
        first entry being the key, the second being a JPEG or PNG image, and
        the third being the contents of the cls file.

        :param args: field names
        :param handler: exception handler
        :param missing_is_error: whether to ignore fields missing from samples and replace them by None
        :param none_is_error: whether reading a None triggers an exception, defaults to missing_is_error
        """
        return self.then(iterators.to_tuple, *args, handler=handler, **kw)

    def map_tuple(self, *args, handler=reraise_exception):
        """Map a tuple.

        :param args: List of functions corresponding to the fields of the tuple.
        :param handler: exception handler
        """
        return self.then(iterators.map_tuple, *args, handler=handler)

    def dbcache(self, fname, size):
        """Cache training samples in an SQLite database.

        This is useful for testing and for running validation tests.

        :param fname: filename for the sqlite database
        :param size: number of samples to be cached
        """
        return self.compose(dbcache.DBCache, fname, size)

    def associate(self, associator):
        """Slice the stream of training samples.

        Associates information from the associator with the current sample.
        The associator should either be a function or a hash table. It is
        invoked with the sample key as an argument and must return a dictionary
        of information that is merged with the sample.

        :param associator: callable or dictionary-like object
        """
        return self.then(iterators.associate, associator)

    def slice(self, *args):
        """Slice the stream of training samples.

        This takes the usual islice arguments of ([start], stop, [step])

        :param args: arguments to itertools.islice
        """
        if len(args) == 0:
            return self
        start = 0
        stop = sys.maxsize
        step = 1
        if len(args) == 1:
            (stop,) = args
        elif len(args) == 2:
            start, stop = args
        elif len(args) == 3:
            start, stop, step = args
        result = self.then(itt.islice, *args)
        return result

    def rsample(self, p=0.5):
        """Randomly subsample a stream of samples.

        :param args: probability of including a sample in the output stream.
        """
        return self.then(iterators.rsample, p)

    def repeat(
        self,
        nepochs=None,
        nbatches=None,
    ):
        """Repeat samples from the source dataset iterator.

        With no arguments, repeat infinitely.

        :param nepochs: maximum number of epochs
        :param nbatches: maximum number of batches
        """
        from .extradatasets import Repeatedly

        return self.compose(
            Repeatedly,
            nepochs=nepochs,
            nbatches=nbatches,
        )

    def with_epoch(self, length, by_node=False):
        """Override the epoch size by repeating/slicding the dataset."""
        from .extradatasets import ChoppedDataset

        if by_node:
            import torch.distributed

            if torch.distributed.is_initialized():
                world_size = torch.distributed.world_size()
                length = length // world_size
        return ChoppedDataset(self, length)

    def with_length(self, length):
        """Return an IterableDataset with a __len__ method."""
        from .extradatasets import FakeLength

        return FakeLength(self, length)

    def ddp_equalize(self, length, with_length=False):
        """Equalize number of training samples in DistributedDataParallel training.

        Torch's DistributedDataParallel requires the same number of samples in
        all participating compute nodes.

        Use with `loader = loader.ddp_equalize(number_of_batches)`


        You need to specify the number of batches you want to equalize to.
        This is usually the number of samples in the dataset divided by the batch size.

        :param length: number of batches in the dataset
        """
        import torch.distributed

        world_size = 1
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        numbatches = length // world_size
        result = self.repeat(sys.maxsize).with_epoch(numbatches)
        if with_length:
            result = result.with_length(numbatches)
        return result


class Processor(IterableDataset, Composable, Shorthands):
    """A class that turns a function into an IterableDataset."""

    def __init__(self, source, f, *args, _kwa={}, **kw):
        """Create a processor.

        The function should take an iterator as an argument and yield
        processed samples. The function is invoked as `f(source, *args, **kw)`.

        :param source: source dataset, an IterableDataset
        :param f: function implementing the processor
        :param args: extra arguments to the processor after the source iterator
        :param _kwa: keyword arguments
        :param kw: extra keyword arguments
        """
        super().__init__()
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = dict(_kwa)
        self.kw.update(kw)

    def source_(self, source):
        """Set the source dataset.

        :param source: source dataset
        """
        self.source = source
        return self

    def __iter__(self):
        """Return an iterator over the source dataset processed by the given function."""
        assert self.source is not None, f"must set source before calling iter {self.f} {self.args} {self.kw}"
        assert callable(self.f), self.f
        return self.f(iter(self.source), *self.args, **self.kw)

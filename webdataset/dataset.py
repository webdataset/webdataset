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
import random
import warnings

import braceexpand

from . import autodecode, dbcache, iterators, shardcache, tariterators, utils
from .utils import lookup_sym, safe_eval
from .handlers import reraise_exception
from .workerenv import split_by_node, split_by_worker, get_worker_environment

try:
    from torch.utils.data import IterableDataset, DataLoader
except ModuleNotFoundError:
    from .mock import IterableDataset, DataLoader


default_cache_dir = os.path.expanduser(os.environ.get("WEBDATASET_CACHE", ""))
default_cache_name = lookup_sym(os.environ.get("WEBDATASET_CACHE_NAME", "shard_uuid"), ".shardcache".split())
default_cache_verbose = int(safe_eval(os.environ.get("WEBDATASET_CACHE_VERBOSE", "1")))
default_cache_size = int(float(safe_eval(os.environ.get("WEBDATASET_CACHE_SIZE", "1e15"))))


class MockDataset(IterableDataset):
    """MockDataset.

    A mock dataset for performance testing and unit testing.
    """

    def __init__(self, sample, length):
        """Create a mock dataset instance.

        :param sample: the sample to be returned repeatedly
        :param length: the length of the mock dataset
        """
        self.sample = sample
        self.length = length

    def __len__(self):
        """Return the length of this mock dataset."""
        return self.length

    def __iter__(self):
        """Return an iterator over this mock dataset."""
        for i in range(self.length):
            yield self.sample


def warn_no_samples(data):
    """Warn if the iterator yields no samples."""
    count = 0
    for sample in data:
        yield sample
        count += 1
    if count == 0:
        env = get_worker_environment()
        if env.rank > 0:
            warnings.warn(f"no samples at all in node {env.rank}, worker {env.worker}")


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

    def then(self, f, *args, length=True, **kw):
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
        return Processor(self, f, length=length, *args, **kw)

    def compose(self, constructor, *args, **kw):
        """Compose this processor with another IterableDataset.

        The constructor should be of the form `__init__(self, source_dataset, ...)`
        """
        assert callable(constructor)
        return constructor(*args, **kw).source_(self)


class ShardList(IterableDataset, Composable):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        shuffle=False,
        nodesplitter=True,
        splitter=True,
        length=None,
    ):
        """Create a ShardList.

        :param urls: a list of URLs as a Python list or brace notation string
        :param shuffle: whether to shuffle the URLs
        :param nodesplitter: function for splitting urls across nodes (None: don't split)
        :param splitter: function for splitting urls across workers (None: don't split)
        :param length: user-specified length; this is returned unchanged by the len() function
        """
        super().__init__()
        self.shuffle = shuffle
        self.length = length
        if nodesplitter is None:
            self.nodesplitter = utils.identity
        elif nodesplitter is True:
            self.nodesplitter = split_by_node
        else:
            assert callable(nodesplitter)
            self.nodesplitter = nodesplitter
        if splitter is None:
            self.splitter = utils.identity
        elif splitter is True:
            self.splitter = split_by_worker
        else:
            assert callable(splitter)
            self.splitter = splitter
        if isinstance(urls, str):
            urls = list(braceexpand.braceexpand(urls))
        else:
            urls = list(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)

    def __iter__(self):
        """Return an iterator over the shards."""
        urls = list(self.urls)
        urls = list(self.nodesplitter(urls))
        urls = list(self.splitter(urls))
        if callable(self.shuffle):
            self.shuffle(urls)
        elif self.shuffle:
            random.shuffle(urls)
        for url in urls:
            yield dict(url=url)

    def __len__(self):
        """Return the user-specified length of this dataset."""
        if self.length is None:
            raise ValueError("length requested, but no length specified for ShardIterator")
        return self.length


class BatchedLength:
    """Compute the batched length of a dataset.

    We make this a class rather than a closure so that it can be pickled.
    """

    def __init__(self, batchsize, partial: bool):
        """Initialize.

        :param batchsize: batch size
        :param partial: allow partial batches
        """
        self.batchsize = batchsize
        self.partial = partial

    def __call__(self, length):
        """Compute the number of batches for the given length.

        :param length: number of samples
        """
        # Add +1 when partial batch is allowed
        partial_batch = len(length) % self.batchsize > 0  # True or False
        return len(length) // self.batchsize + (1 if self.partial and partial_batch else 0)


class Shorthands:
    """A convenient set of shorthands for common data transformations."""

    def batched(self, batchsize, collation_fn=iterators.default_collation_fn, partial=True):
        """Compute batches for the given dataset.

        :param batchsize: desired batchsize
        :param collation_fn: collation function to turn list of objects into batches
        :param partial: return partial batches
        """
        length = BatchedLength(batchsize, partial)
        return self.then(
            iterators.batched, length=length, batchsize=batchsize, collation_fn=collation_fn, partial=partial
        )

    def unbatched(self, length=None):
        """Take a stream of batches and turn it back into a stream of samples.

        :param length: user-supplied length for the unbatched dataset.
        """
        return self.then(iterators.unbatched, length=length)

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

    def to_tuple(self, *args, handler=reraise_exception):
        """Convert a dictionary-based sample to a tuple.

        Field names to be extracted can be specified as a Python list
        or as a string. "__key__ jpg;png cls" will extract a triple, with the
        first entry being the key, the second being a JPEG or PNG image, and
        the third being the contents of the cls file.

        :param args: field names
        :param handler: exception handler
        """
        return self.then(iterators.to_tuple, *args, handler=handler)

    def map_tuple(self, *args, handler=reraise_exception):
        """Map a tuple.

        :param args: List of functions corresponding to the fields of the tuple.
        :param handler: exception handler
        """
        return self.then(iterators.map_tuple, *args, handler=handler)

    def pipe(self, f, *args, **kw):
        """Pipe the sample stream through the given function.

        :param f: function obtaining an iterator as a sample and yielding samples
        :param args: arguments to the function
        :param kw: keyword arguments
        """
        return self.then(f, *args, _kwa=kw)

    def dbcache(self, fname, size):
        """Cache training samples in an SQLite database.

        This is useful for testing and for running validation tests.

        :param fname: filename for the sqlite database
        :param size: number of samples to be cached
        """
        return self.compose(dbcache.DBCache, fname, size)

    def slice(self, *args):
        """Slice the stream of training samples.

        This takes the usual islice arguments of ([start], stop, [step])

        :param args: arguments to itertools.islice
        """
        return self.pipe(itt.islice, *args)

    def repeat(
        self,
        nepochs=None,
        nbatches=None,
        nsamples=None,
        batchsize=utils.guess_batchsize,
    ):
        """Repeat samples from the source dataset iterator.

        With no arguments, repeat infinitely.

        :param nepochs: maximum number of epochs
        :param nbatches: maximum number of batches
        :param nsamples: maximum number of samples
        :param batchsize: integer giving batchsize, or function to compute it
        """
        return self.compose(
            Repeatedly,
            nepochs=nepochs,
            nbatches=nbatches,
            nsamples=nsamples,
            batchsize=batchsize,
        )

    def test(self, length=None, checker=None, mock_sample=None, mock_length=None, mock=False):
        """A quick and simple way of switching to a mock dataset at the end of a pipeline.

        Use with `loader = loader.test(mock_sample=..., mock_length=...)
        You can turn on mocking with `loader.mock = True`

        :param length: length of the dataset
        :param checker: any kind of final checking function you want to run over samples
        :param mock_sample: mock sample
        :param mock_length: size of mocked dataset
        :param mock: turning mocking on/off
        """
        return self.compose(
            DatasetTest,
            length=length,
            checker=checker,
            mock_sample=mock_sample,
            mock_length=mock_length,
            mock=mock,
        )

    def ddp_equalize(self, length):
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
        result = self.repeat(999999999).slice(numbatches)
        result.length = numbatches
        return result


class Repeatedly(IterableDataset, Composable, Shorthands):
    """Repeatedly yield samples from a dataset."""

    def __init__(self, nepochs=None, nbatches=None, nsamples=None, batchsize=None, length=None):
        """Create an instance of Repeatedly.

        :param nepochs: repeat for a maximum of nepochs
        :param nbatches: repeat for a maximum of nbatches
        :param nsamples: repeat for a maximum of nsamples (requires batchsize)
        :param batchsize: integer or function of sample returning batch size
        """
        self.length = length
        self.nepochs = nepochs
        self.nbatches = nbatches
        self.nsamples = nsamples
        self.batchsize = batchsize

    def __iter__(self):
        """Return an iterator that iterates repeatedly over a source."""
        return utils.repeatedly(self.source, nepochs=self.nepochs, nbatches=self.nbatches, nsamples=self.nsamples, batchsize=self.batchsize)

    def __len__(self):
        """Return the length of the source."""
        if callable(self.length):
            return self.length(self.source)
        if self.length is not None:
            return self.length
        if self.nepochs is not None:
            return len(self.source) * self.nepochs
        if self.nbatches is not None:
            return self.nbatches
        if self.nsamples is not None:
            raise ValueError("can't compute size for nsamples; please specify with length= argument")


class Processor(IterableDataset, Composable, Shorthands):
    """A class that turns a function into an IterableDataset."""

    def __init__(self, source, f, *args, _kwa={}, length=True, **kw):
        """Create a processor.

        The function should take an iterator as an argument and yield
        processed samples. The function is invoked as `f(source, *args, **kw)`.

        The `length` can be specified as `True`, in which case the value
        is taken from the source dataset, as a callable, in which case
        the length is the result of applying the callable to the source
        dataset, or as an integer, in which case the length returned by
        `__len__` is that integer.

        :param source: source dataset, an IterableDataset
        :param f: function implementing the processor
        :param args: extra arguments to the processor after the source iterator
        :param _kwa: keyword arguments
        :param length: specified length for the output
        :param kw: extra keyword arguments
        """
        super().__init__()
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = dict(_kwa)
        self.kw.update(kw)
        self.length = length

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

    def __len__(self):
        """Return the length of this dataset; see above how this is computed."""
        if self.length is True:
            return len(self.source)
        elif isinstance(self.length, int):
            return self.length
        elif callable(self.length):
            return self.length(self.source)
        else:
            raise ValueError(f"{self.length}: not a valid length specification")


def WebDataset(
    urls,
    shardshuffle=True,
    cache_dir=default_cache_dir,
    cache_size=default_cache_size,
    cache_name=default_cache_name,
    cache_verbose=default_cache_verbose,
    splitter=split_by_worker,
    nodesplitter=True,
    handler=reraise_exception,
    length=None,
    warn_empty=True,
):
    """Return a pipeline for WebDataset-style data files.

    This is a convenience function for constructing a partial pipeline
    that reads from a set of sharded tar files, extracts the individual
    files, and groups them together into samples (dictionaries).

    You can use all the methods from `Composable` (`then`, `compose`) and
    from `Shorthands` (`batched`, `unbatched`, `decode`, `shuffle`, etc.)
    on the result.

    :param urls: the source URLs, specified either as a list or as a brace-expanded string
    :param shardshuffle: boolean indicating whether the shards should be shuffled or not
    :param splitter: a function called for splitting shards among workers (True: PyTorch default, None: no splitting)
    :param nodesplitter: a function called for splitting shards among nodes (True: PyTorch default, None: no splitting)
    :param handler: an error handler
    :param length: the length of this dataset, should be an integer
    :param cache_dir: when set, caches shards in this directory
    :param cache_size: when set, specifies a maximum size for the shard cache
    :param cache_name: when set, specifies how shards should be named in the cache
    :param cache_verbose: when set, prints information about caching
    :param warn_empty: warn when no samples are generated at all
    """
    result = ShardList(
        urls,
        shuffle=shardshuffle,
        splitter=splitter,
        nodesplitter=nodesplitter,
        length=length,
    )
    result = result.then(tariterators.url_opener, handler=handler)
    if cache_dir != "":
        result = result.then(
            shardcache.cache_shards,
            cache_dir=cache_dir,
            cache_size=cache_size,
            cache_name=cache_name,
            verbose=cache_verbose,
        )
    result = result.then(tariterators.tar_file_expander, length=None, handler=handler)
    result = result.then(tariterators.group_by_keys, length=length)
    if warn_empty:
        result = result.then(warn_no_samples)
    return result


def WebLoader(*args, **kw):
    """Return a small wrapper around torch.utils.data.DataLoader.

    This wrapper works identically to the original `DataLoader`, but adds
    alls the convenience functions and filters for WebDataset.

    You can use all the methods from `Composable` (`then`, `compose`) and
    from `Shorthands` (`batched`, `unbatched`, `decode`, `shuffle`, etc.)
    on the result.

    :param args: forwarded to `DataLoader`
    :param kw: forwarded to `DataLoader`
    """
    return Processor(DataLoader(*args, **kw), utils.identity)


class DatasetTest(IterableDataset, Composable, Shorthands):
    """Perform final checks on an IterableDataset and permit easy mock tests.

    This is the implementation of the `Shorthands.test` method; you usually
    do not need to construct it explicitly.
    """

    def __init__(self, length=None, checker=None, mock_sample=None, mock_length=10000, mock=False):
        """Create a DatasetTest.

        :param length: length of the dataset
        :param checker: any kind of final checking function you want to run over samples
        :param mock_sample: mock sample
        :param mock_length: size of mocked dataset
        :param mock: turning mocking on/off
        """
        super().__init__()
        self.source = None
        self.length = length
        self.checker = checker
        self.mock = mock
        self.mock_length = mock_length
        self.mock_sample = mock_sample

    def __len__(self):
        """Return the length of the test object.

        This is either the length of the mock object when in mock mode,
        otherwise the length of the underlying dataset/data loader.
        """
        if self.mock:
            return self.mock_length
        elif self.length is True:
            return len(self.source)
        elif isinstance(self.length, int):
            return self.length
        elif callable(self.length):
            return self.length(self.source)
        else:
            raise ValueError(f"{self.length}: not a valid length specification")

    def __iter__(self):
        """Return an iterator either over the mock object or the underlying dataset."""
        if self.mock:
            if not callable(self.mock_sample):
                for i in range(self.mock_length):
                    yield self.mock_sample
            else:
                return self.mock_sample()
        else:
            for sample in self.source:
                if self.checker is not None:
                    self.checker(sample)
                yield sample


class ChoppedDataset(IterableDataset):
    """Change the actual and nominal length of an IterableDataset.

    This will continuously iterate through the original dataset, but
    impose new epoch boundaries at the given length/nominal.
    This exists mainly as a workaround for the odd logic in DataLoader.
    It is also useful for choosing smaller nominal epoch sizes with
    very large datasets.

    """

    def __init__(self, dataset, length=None, nominal=None):
        """Create a ChoppedDataset.

        :param dataset: IterableDataset
        :param length: declared length of the dataset
        :param nominal: nominal length of dataset (if different from declared)
        """
        super().__init__()
        self.dataset = dataset
        if length is None:
            length = len(dataset)
        self.length = length
        self.nominal = self.length if nominal is None else nominal
        self.source = None

    def __len__(self):
        """Return the length of the dataset."""
        return self.nominal

    def __getstate__(self):
        """Return the pickled state of the dataset.

        This resets the dataset iterator, since that can't be pickled.
        """
        result = dict(self.__dict__)
        result["source"] = None
        return result

    def __iter__(self):
        """Return an iterator over the dataset.

        This iterator returns as many samples as given by the `length` parameter.
        """
        if self.source is None:
            self.source = iter(self.dataset)
        for i in range(self.length):
            try:
                sample = next(self.source)
            except StopIteration:
                self.source = iter(self.dataset)
                sample = next(self.source)
            yield sample


ResizedDataset = ChoppedDataset

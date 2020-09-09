#!/usr/bin/python
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Train PyTorch models directly from POSIX tar archive, locally
or over HTTP connections.
"""

__all__ = """Dataset tariterator default_handlers imagehandler
reraise_exception ignore_and_continue warn_and_continue ignore_and_stop warn_and_stop
""".split()

import gc
import os
import random
import re
import tarfile
import time
import warnings
from builtins import range

import braceexpand
from torch.utils.data import IterableDataset

from . import filters, gopen, autodecode

# from functools import wraps
trace = False

debug_dataset = os.environ.get("WDS_DEBUG", 0)
popen_bufsize = int(os.environ.get("WDS_BUFSIZE", "2000000"))

meta_prefix = "__"
meta_suffix = "__"

collection_counter = 0
collection_frequency = 50000


def reraise_exception(exn):
    """Called in an exception handler to re-raise the exception."""
    raise exn


def ignore_and_continue(exn):
    """Called in an exception handler to ignore any exception and continue."""
    return True


def warn_and_continue(exn):
    """Called in an exception handler to ignore any exception, isssue a warning, and continue."""
    warnings.warn(repr(exn))
    time.sleep(0.5)
    return True


def ignore_and_stop(exn):
    """Called in an exception handler to ignore any exception and stop further processing."""
    return False


def warn_and_stop(exn):
    """Called in an exception handler to ignore any exception and stop further processing."""
    warnings.warn(repr(exn))
    time.sleep(0.5)
    return False


def add_hook(fs, f):
    assert callable(f)
    if fs is None:
        return [f]
    if isinstance(fs, list):
        return fs + [f]
    assert callable(fs)
    return [fs, f]


def call_hook(fs, *args, **kw):
    if fs is None:
        return
    if not isinstance(fs, list):
        fs = [fs]
    for f in fs:
        f(*args, **kw)


def identity(x):
    return x


def do_nothing(*args, **kw):
    """Do nothing function."""
    pass


class Shuffler:
    """Make a shuffle function (avoid nesting for pickle)."""
    def __init__(self, rng):
        self.rng = rng

    def __call__(self, lst):
        lst = list(lst)
        self.rng.shuffle(lst)
        return lst


def maybe_collect():
    """Running in notebooks, we tend to run out of memory due to
    weak references, and the collector doesn't seem to get triggered
    in time automatically. This function periodically calls the Python
    garbage collector."""
    global collection_counter, collection_frequency  # skipcq: PYL-W0603
    if collection_counter >= collection_frequency == 0:
        gc.collect()
        collection_counter = 0
    collection_counter += 1


def base_plus_ext(path):
    """Helper method that splits off all extension.

    Returns base, allext.

    path: path with extensions
    returns: path with all extensions removed

    """
    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def valid_sample(sample):
    """Check whether a sample is valid.

        sample: sample to be checked

    """
    return (
        sample is not None
        and isinstance(sample, dict)
        and len(list(sample.keys())) > 0
        and not sample.get("__bad__", False)
    )


def group_by_keys_(data, keys=base_plus_ext, lcase=True, suffixes=None):
    """Returns function over iterator that groups key, value pairs into samples.

    keys: function that splits the key into key and extension (base_plus_ext)
    lcase: convert suffixes to lower case (Default value = True)

    """

    current_sample = None
    for fname, value in data:
        prefix, suffix = keys(fname)
        if trace:
            print(
                prefix,
                suffix,
                current_sample.keys() if isinstance(current_sample, dict) else None,
            )
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        if current_sample is None or prefix != current_sample["__key__"]:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix)
        if suffix in current_sample:
            raise ValueError(
                f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}"
            )
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


group_by_keys = filters.Curried(group_by_keys_)


def tardata(fileobj, skip_meta=r"__[^/]*__($|/)", handler=reraise_exception):
    """Iterator yielding filename, content pairs for the given tar stream.

    :param fileobj: byte stream suitable for tarfile
    :param skip_meta: regexp for keys that are skipped entirely (Default value = r"__[^/]*__($|/)")

    """
    try:
        stream = tarfile.open(fileobj=fileobj, mode="r|*")
        for tarinfo in stream:
            try:
                if not tarinfo.isreg():
                    continue
                fname = tarinfo.name
                if fname is None:
                    continue
                if (
                    "/" not in fname
                    and fname.startswith(meta_prefix)
                    and fname.endswith(meta_suffix)
                ):
                    # skipping metadata for now
                    continue
                if skip_meta is not None and re.match(skip_meta, fname):
                    continue
                data = stream.extractfile(tarinfo).read()
                yield fname, data
            except Exception as exn:
                if handler(exn):
                    continue
                else:
                    break
        del stream
    except Exception as exn:
        handler(exn)


class Pipeline:
    """Simple fluid pipeline builder.

    This is a convenience class that allows common processing
    pipelines to be built up with a fluid interface.
    """

    def __init__(self):
        self.pipeline = []
        self.rng = None
        self.shard_shuffle = None

    def pipe(self, stage):
        """Add a pipline stage (a function taking an iterator and returning another iterator)."""
        self.pipeline.append(stage)
        return self

    def batched(
        self, batchsize, partial=True,
    ):
        self.pipeline.append(filters.batched(batchsize=batchsize, partial=True,))
        return self

    def unbatched(self):
        self.pipeline.append(filters.unbatched())
        return self

    def shuffle(self, size, rng=None, **kw):
        """Shuffle the data."""
        if size == 0:
            return self
        if rng is None:
            rng = random.Random()
        self.rng = rng
        self.reseed_hook = self.reseed_rng
        self.shard_shuffle = Shuffler(rng)
        self.pipeline.append(filters.shuffle(size, rng=rng, **kw))
        return self

    def reseed_rng(self):
        seed = random.SystemRandom().random()
        self.rng.seed(seed)

    def decode(self, *args, handler=reraise_exception):
        """Decode the data with the given decoder."""
        handlers = list(args)
        # special case for images (backwards compatibility)
        for i in range(len(handlers)):
            if isinstance(handlers[i], tuple) and isinstance(handlers[i][0], str):
                assert callable(handlers[i][1]), handlers[i][1]
                handlers[i] = autodecode.handle_extension(handlers[i][0], handlers[i][1])
            elif isinstance(handlers[i], str):
                handlers[i] = autodecode.ImageHandler(handlers[i])
        for f in handlers:
            assert callable(f), f"unknown handler {f} in `Dataset.decode`"
        # always provide basichandlers; if you don't want it,
        # either map Decoder yourself, or override the types
        # you don't want decoded
        handlers += [autodecode.basichandlers]
        decoder = autodecode.Decoder(handlers)
        self.pipeline.append(filters.map(decoder, handler=handler))
        return self

    def map(self, f, handler=reraise_exception):
        """Apply function `f` to each sample."""
        self.pipeline.append(filters.map(f, handler=handler))
        return self

    def rename(self, handler=reraise_exception, **kw):
        """Rename fields in the sample, dropping all unmatched fields."""
        self.pipeline.append(filters.rename(handler=handler, **kw))
        return self

    def map_dict(self, handler=reraise_exception, **kw):
        """Transform each sample by applying functions to corresponding fields."""
        self.pipeline.append(filters.map_dict(handler=handler, **kw))
        return self

    def select(self, predicate, **kw):
        """Select samples based on a predicate."""
        self.pipeline.append(filters.select(predicate, **kw))
        return self

    def to_tuple(self, *args, handler=reraise_exception):
        """Extract fields from the sample in order and yield tuples."""
        self.pipeline.append(filters.to_tuple(*args, handler=handler))
        return self

    def map_tuple(self, *args, handler=reraise_exception):
        """Apply a list of functions to the tuple."""
        self.pipeline.append(filters.map_tuple(*args, handler=handler))
        return self


def make_opener(open_fn):
    if isinstance(open_fn, str):
        return gopen.command_pipe(open_fn)
    elif callable(open_fn):
        return open_fn
    else:
        raise ValueError(f"{open_fn}: must be either str or callable")


class SampleIterator(Pipeline):
    """Iterates over the samples of webdatasets using a given processing pipeline."""

    def __init__(
        self, initial_pipeline=None, tarhandler=reraise_exception, open_fn=gopen.reader
    ):
        Pipeline.__init__(self)
        self.open_fn = make_opener(open_fn)
        self.tarhandler = tarhandler
        self.shard_hook = do_nothing
        if initial_pipeline is None:
            initial_pipeline = [group_by_keys()]
        for stage in initial_pipeline:
            self.pipe(stage)

    def raw_samples(self, urls):
        assert isinstance(urls, list)
        for url in urls:
            self.shard_hook()
            stream = None
            try:
                with self.open_fn(url) as stream:
                    files_of_archive = tardata(stream, handler=self.tarhandler)
                    for fname, content in files_of_archive:
                        yield fname, content
                    maybe_collect()
            except Exception as exn:
                if self.tarhandler(exn):
                    continue
                else:
                    break

    def samples(self, urls):
        if isinstance(urls, str):
            urls = [urls]
        assert isinstance(urls, list)
        self.sample_urls = urls
        source = self.raw_samples(urls)
        return filters.pipeline(source, *self.pipeline)


def all_urls(urls):
    """Returns all URLs.

    Used as a shard selection function in Dataset."""
    assert isinstance(urls, list)
    assert isinstance(urls[0], str)
    return urls


def worker_urls(urls):
    """Selects a subset of urls based on Torch get_worker_info.

    Used as a shard selection function in Dataset."""
    import torch

    assert isinstance(urls, list)
    assert isinstance(urls[0], str)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        wid = worker_info.id
        num_workers = worker_info.num_workers
        if wid == 0 and len(urls) < num_workers:
            warnings.warn(f"num_workers {num_workers} > num_shards {len(urls)}")
        return urls[wid::num_workers]
    else:
        return urls


class Dataset(IterableDataset, SampleIterator):
    """Iterate over sharded datasets.

    This class combines several function: it is a container for a list of
    shards, it is a container for a processing pipelines, and it handles
    some bookkeeping related to DataLoader.
    """

    def __init__(
        self,
        urls,
        *,
        length=None,
        open_fn=gopen.reader,
        handler=reraise_exception,
        tarhandler=None,
        prepare_for_worker=True,
        initial_pipeline=None,
        shard_selection=worker_urls,
    ):
        tarhandler = handler if tarhandler is None else tarhandler
        IterableDataset.__init__(self)
        SampleIterator.__init__(
            self,
            initial_pipeline=initial_pipeline,
            tarhandler=tarhandler,
            open_fn=open_fn,
        )
        if isinstance(urls, str):
            urls = list(braceexpand.braceexpand(urls))
        self.urls = urls
        self.length = length
        self.handler = handler
        self.total = 0
        self.reseed_hook = do_nothing
        self.node_selection = identity
        self.shard_selection = shard_selection
        self.shard_shuffle = identity

    def __len__(self):
        """Return the nominal length of the dataset."""
        return self.length

    def shard_fn(self):
        urls = list(self.urls)
        self.reseed_hook()
        urls = self.node_selection(urls)
        urls = self.shard_selection(urls)
        urls = self.shard_shuffle(urls)
        return urls

    def __iter__(self):
        urls = self.shard_fn()
        return self.samples(urls)


class ResizedDataset(IterableDataset):
    """Change the actual and nominal length of an IterableDataset.

    :param dataset: IterableDataset
    :param length: declared length of the dataset
    :param nominal: nominal length of dataset (if different from declared)

    This will continuously iterate through the original dataset, but
    impose new epoch boundaries at the given length/nominal.
    This exists mainly as a workaround for the odd logic in DataLoader.
    It is also useful for choosing smaller nominal epoch sizes with
    very large datasets.

    """

    def __init__(self, dataset, length=None, nominal=None):
        self.dataset = dataset
        if length is None:
            length = len(dataset)
        self.length = length
        self.nominal = self.length if nominal is None else nominal
        self.source = None

    def __len__(self):
        return self.nominal

    def __getstate__(self):
        result = dict(self.__dict__)
        result["source"] = None
        return result

    def __iter__(self):
        if self.source is None:
            self.source = iter(self.dataset)
        for i in range(self.length):
            try:
                sample = next(self.source)
            except StopIteration:
                self.source = iter(self.dataset)
                sample = next(self.source)
            yield sample


ChoppedDataset = ResizedDataset

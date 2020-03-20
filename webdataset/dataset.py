#!/usr/bin/python
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Train PyTorch models directly from POSIX tar archive, locally
or over HTTP connections.
"""

__all__ = """Dataset WebDataset tariterator default_handlers imagehandler
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

from . import gopen
from . import filters
from .checks import checkcallable
from .webdataset import WebDataset  # noqa: F401

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


def group_by_keys(keys=base_plus_ext, lcase=True, suffixes=None):
    """Returns function over iterator that groups key, value pairs into samples.

    keys: function that splits the key into key and extension (base_plus_ext)
    lcase: convert suffixes to lower case (Default value = True)

    """

    def iterator(data):
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

    return iterator


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


class Dataset(IterableDataset):
    """Iterate over sharded datasets.

    :param urls: shard spec or list of shards
    :param prepare_for_worker: callable called in each worker before anything else is done

    """

    def __init__(
        self,
        urls,
        *,
        keys=base_plus_ext,
        suffixes=None,
        length=None,
        epochs=1,
        opener=gopen.reader,
        handler=reraise_exception,
        shuffle=False,
        prepare_for_worker=True,
        initial_pipeline=None,
    ):
        if isinstance(opener, str):
            self.opener = gopen.command_pipe(opener)
        elif callable(opener):
            self.opener = opener
        else:
            raise ValueError(f"{opener}: must be either str or callable")
        checkcallable(self.opener)
        if isinstance(urls, str):
            urls = list(braceexpand.braceexpand(urls))
        self.full_urls = self.urls = urls
        self.length = length
        self.epochs = epochs
        self.keys = keys
        self.suffixes = suffixes
        self.subset = None
        self.do_shuffle = shuffle
        self.handler = handler
        if prepare_for_worker is True:
            self.prepare_for_worker = self.shard_selection
        elif prepare_for_worker is False:
            self.prepare_for_worker = lambda: None
        else:
            self.prepare_for_worker = prepare_for_worker
        self.pipeline = (
            initial_pipeline if initial_pipeline is not None else [group_by_keys()]
        )

    def __len__(self):
        return self.length

    def shard_selection(self):
        """Contains the logic for self.subset shard selection."""
        import torch

        index, total = None, None
        if self.subset is not None:
            index, total = self.subset  # skipcq: PYL-E0633
        else:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                index = worker_info.id
                total = worker_info.num_workers
        if total is None:
            self.urls = self.full_urls
            return
        if index == 0 and len(self.full_urls) < total:
            warnings.warn(f"num_workers {total} > num_shards {len(self.full_urls)}")
        self.urls = self.full_urls[index::total]

    def raw_iter(self):
        """Iterate over samples."""
        self.prepare_for_worker()
        if self.do_shuffle:
            random.shuffle(self.urls)
        self.sample = 0
        urls = self.urls
        for epoch in range(self.epochs):
            for url in urls:
                stream = None
                with self.opener(url) as stream:
                    files_of_archive = tardata(stream, handler=self.handler)
                    for fname, content in files_of_archive:
                        yield fname, content
                    maybe_collect()

    def __iter__(self):
        return filters.pipeline(self.raw_iter(), *self.pipeline)

    def pipe(self, stage):
        """Add a pipline stage (a function taking an iterator and returning another iterator)."""
        self.pipeline.append(stage)
        return self

    def shuffle(self, size, **kw):
        """Shuffle the data."""
        if size == 0:
            return self
        self.do_shuffle = True
        self.pipeline.append(filters.shuffle(size, **kw))
        return self

    def decode(self, decoder="rgb", handler=reraise_exception):
        """Decode the data with the given decoder."""
        self.pipeline.append(filters.decode(decoder, handler=handler))
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

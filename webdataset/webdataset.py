#!/usr/bin/python
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

# THIS CODE IS DEPRECATED AND WILL BE REMOVED SOON
# USE webdataset.Dataset INSTEAD


"""Train PyTorch models directly from POSIX tar archive, locally
or over HTTP connections.
"""

__all__ = "Dataset WebDataset tariterator default_handlers imagehandler".split()

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
from . import autodecode
from . import filters
from .checks import checktype, checkcallable

trace = False

debug_dataset = os.environ.get("WDS_DEBUG", 0)
popen_bufsize = int(os.environ.get("WDS_BUFSIZE", "2000000"))

meta_prefix = "__"
meta_suffix = "__"

collection_counter = 0
collection_frequency = 50000


def reraise_exception(exn):
    raise exn


def ignore_and_continue(exn):
    return True


def ignore_and_stop(exn):
    return False


def warn_and_continue(exn):
    warnings.warn(repr(exn))
    time.sleep(0.5)
    return True


def warn_and_stop(exn):
    warnings.warn(repr(exn))
    time.sleep(0.5)
    return True


def listify(x):
    """Turn a value into a list.

    Lists and tuples are turned into lists, everything else is turned
    into a one element list.

    """
    if x is None:
        return None
    elif isinstance(x, (list, tuple)):
        return x
    else:
        return [x]


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


class NoException(Exception):
    pass


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


def maybe_decode(s, mode="ascii"):
    if isinstance(s, str):
        return s
    elif isinstance(s, bytes):
        return s.decode(mode)
    else:
        raise ValueError(f"{type(s)}: wrong type for maybe_decode")


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


def tariterator(
    fileobj,
    keys=base_plus_ext,
    decoder=True,
    suffixes=None,
    tar_errors=reraise_exception,
    decode_errors=reraise_exception,
):
    """
    Iterate through training samples stored in a sharded tar file.

    :param fileobj: a Python file-like object
    :param check_sorted:  check whether the input is actually properly sorted (Default value = False)
    :param keys:  key extraction function (Default value = base_plus_ext)
    :param decoder: value decoding function (Default value = True)

    The key extraction function takes a string representing a pathname and
    returns a pair (__key__, suffix).

    The decoder takes the entire sample as a dict and returns the
    decoded sample as a dict.
    """
    content = tardata(fileobj, handler=tar_errors)
    samples = group_by_keys(keys=keys, suffixes=suffixes)(content)
    decoder = autodecode.make_decoder(decoder)
    samples = filters.map_stream(decoder, handler=decode_errors)(samples)
    return samples


class WebDataset(IterableDataset):
    """Iterate over sharded datasets.

    :param urls: shard spec or list of shards
    :param extensions: extensions to extract (Default value = None, can be either list of lists or "a;b c")
    :param decode: decoder to apply to files in tarfiles (Default value = True, based on extension)
    :param transforms: list of functions to apply to unbatched samples (Default value = None)
    :param pipeline: function that maps the iterator, e.g. for batching
    :param opener: either a function that returns a stream or a string that is invoked via Popen
    :param verbose: verbose output
    :param shuffle: if >0, then shuffle shards, and shuffle samples with a buffer of the given size
    :param associate: a callable or dictionary that returns additional information to associate with each sample
    :param prepare_for_worker: callable called in each worker before anything else is done
    :param extra_meta: associates subset info with each sample record

    The decoder can be True (default decoder), False (no decoder), a callable (called
    decode the sample, or a dictionary mapping filename extensions to callables for
    the decoding.
    """

    def __init__(
        self,
        urls,
        *,
        extensions=None,
        decoder="rgb",
        transforms=None,
        pipeline=None,
        epochs=1,
        keys=base_plus_ext,
        opener=gopen.reader,
        verbose=False,
        shuffle=0,
        associate=None,
        prepare_for_worker=True,
        length=None,
        handler=reraise_exception,
    ):
        if isinstance(opener, str):
            self.opener = gopen.command_pipe(opener)
        elif callable(opener):
            self.opener = opener
        else:
            raise ValueError(f"{opener}: must be either str or callable")
        checkcallable(self.opener)
        self.decoder = decoder
        self.transforms = listify(transforms)
        self.epochs = epochs
        self.verbose = verbose
        self.keys = keys
        self.handler = handler
        self.associate = associate
        self.pipeline = pipeline
        self.length = length
        if isinstance(urls, str):
            urls = list(braceexpand.braceexpand(urls))
        checktype(urls, list)
        self.full_urls = urls
        self.urls = urls
        self.shuffle = shuffle
        if extensions is not None:
            if isinstance(extensions, str):
                extensions = [f.split(";") for f in extensions.split()]
            for f in extensions:
                checktype(f, list)
            self.extensions = extensions
            self.suffixes = {x for l in extensions for x in l}
        else:
            self.extensions = None
            self.suffixes = None
        if prepare_for_worker is True:
            self.prepare_for_worker = self.shard_selection
        elif prepare_for_worker is False:
            self.prepare_for_worker = lambda: None
        else:
            self.prepare_for_worker = prepare_for_worker
        self.subset = None
        self.extra_meta = False
        self.sample = None

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

    def __iter__(self):
        """Iterate over samples."""
        self.prepare_for_worker()
        if self.shuffle > 0:
            random.shuffle(self.urls)
        self.sample = 0
        urls = self.urls
        for epoch in range(self.epochs):
            for url in urls:
                stream = None
                try:
                    with self.opener(url) as stream:
                        source = tariterator(
                            stream,
                            keys=self.keys,
                            suffixes=self.suffixes,
                            decoder=self.decoder,
                            tar_errors=self.handler,
                            decode_errors=self.handler,
                        )
                        source = iter(source)
                        if self.associate is not None:
                            source = filters.associate(self.associate)(source)
                        if self.extensions is not None:
                            source = filters.to_tuple(*self.extensions)(source)
                        if self.shuffle > 1:
                            source = filters.shuffle(self.shuffle)(source)
                        if self.transforms is not None:
                            source = filters.map_stream(
                                filters.transformer(self.transforms)
                            )(source)
                        if self.pipeline is not None:
                            source = self.pipeline(source)
                        for sample in source:
                            if self.extra_meta and isinstance(sample, dict):
                                sample["__webdataset__"] = (self.subset,)
                            if isinstance(sample, list):
                                sample = tuple(sample)
                            yield sample
                            maybe_collect()
                except Exception as exn:  # skipcq: PYL-W0703
                    if self.handler(exn):
                        continue
                    else:
                        break

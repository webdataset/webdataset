#!/usr/bin/python
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""A collection of iterators implementing useful functionality for
transforming datasets in processing pipelines.

These functions are plain iterator functions. You can find curried versions
in webdataset.filters, and you can find IterableDataset wrappers in
webdataset.processing.
"""

__all__ = "WebDataset tariterator default_handlers imagehandler".split()

import random
import sys
from functools import reduce

import numpy as np

from . import autodecode
from .checks import checktype

try:
    from torch import Tensor as TorchTensor
except ModuleNotFoundError:

    class TorchTensor:
        pass


###
# Helpers
###


def reraise_exception(exn):
    raise exn


def identity(x):
    return x


def compose2(f, g):
    """Compose two functions, g(f(x))"""
    return lambda x: g(f(x))


def compose(*args):
    """Compose a sequence of functions (left-to-right)"""
    return reduce(compose2, args)


def pipeline(source, *args):
    """Write an input pipeline; first argument is source, rest are filters."""
    if len(args) == 0:
        return source
    return compose(*args)(source)


def getfirst(a, keys, default=None, missing_is_error=True):
    """Get the first matching key from a dictionary.

    Keys can be specified as a list, or as a string of keys separated by ';'.
    """
    if isinstance(keys, str):
        assert " " not in keys
        keys = keys.split(";")
    for k in keys:
        if k in a:
            return a[k]
    if missing_is_error:
        raise ValueError(f"didn't find {keys} in {list(a.keys())}")
    return default


def parse_field_spec(fields):
    """Parse a specification for a list of fields to be extracted.

    Keys are separated by spaces in the spec. Each key can itself
    be composed of key alternatives separated by ';'.
    """
    if isinstance(fields, str):
        fields = fields.split()
    return [field.split(";") for field in fields]


def transform_with(sample, transformers):
    """Transform a list of values using a list of functions.

    sample: list of values
    transformers: list of functions

    If there are fewer transformers than inputs, or if a transformer
    function is None, then the identity function is used for the
    corresponding sample fields.
    """
    checktype(sample, (tuple, list))
    if transformers is None or len(transformers) == 0:
        return sample
    result = list(sample)
    assert len(transformers) <= len(sample)
    for i in range(len(transformers)):  # skipcq: PYL-C0200
        f = transformers[i]
        if f is not None:
            result[i] = f(sample[i])
    return result


def transformer(transformers):
    """Curried version of `transform_with`.

    transformers :

    """

    def f(x):
        return transform_with(x, transformers)

    return f


###
# Iterators
###


# def map_stream(data, f=None, handler=reraise_exception):
#     """Map entire samples using the given function.
# 
#     data: iterator
#     f: function from samples to samples
#     returns: iterator over transformed samples
# 
#     """
# 
#     if f is None:
# 
#         def f(x):  # skipcq: PYL-E0102
#             return x
# 
#     for sample in data:
#         try:
#             result = f(sample)
#         except Exception as exn:
#             if handler(exn):
#                 continue
#             else:
#                 break
#         if isinstance(sample, dict) and isinstance(result, dict):
#             result["__key__"] = sample.get("__key__")
#         yield result


def info(data, fmt=None, n=3, every=-1, width=50, stream=sys.stderr, name=""):
    for i, sample in enumerate(data):
        if i < n or (every > 0 and (i + 1) % every == 0):
            if fmt is None:
                print("---", name, file=stream)
                for k, v in sample.items():
                    print(k, repr(v)[:width], file=stream)
            else:
                print(fmt.format(**sample), file=stream)
        yield sample


def shuffle(data, bufsize=1000, initial=100, rng=random, handler=None):
    """Shuffle the data in the stream.

    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.

    data: iterator
    bufsize: buffer size for shuffling
    returns: iterator
    rng: either random module or random.Random instance

    """
    initial = min(initial, bufsize)
    buf = []
    startup = True
    for sample in data:
        if len(buf) < bufsize:
            try:
                buf.append(next(data))  # skipcq: PYL-R1708
            except StopIteration:
                pass
        k = rng.randint(0, len(buf) - 1)
        sample, buf[k] = buf[k], sample
        if startup and len(buf) < initial:
            buf.append(sample)
            continue
        startup = False
        yield sample
    for sample in buf:
        yield sample


def select(data, predicate):
    for sample in data:
        if predicate(sample):
            yield sample


def decode(data, *args, handler=reraise_exception, **kw):
    f = autodecode.Decoder(list(args), **kw)

    for sample in data:
        assert isinstance(sample, dict), sample
        try:
            decoded = f(sample)
        except Exception as exn:  # skipcq: PYL-W0703
            if handler(exn):
                continue
            else:
                break
        yield decoded


def map(data, f, handler=reraise_exception):
    for sample in data:
        try:
            result = f(sample)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
        if isinstance(sample, dict) and isinstance(result, dict):
            result["__key__"] = sample.get("__key__")
        yield result


def rename(data, handler=reraise_exception, **kw):
    for sample in data:
        try:
            yield {k: getfirst(sample, v, missing_is_error=True) for k, v in kw.items()}
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


def associate(data, associator, **kw):
    for sample in data:
        if callable(associator):
            extra = associator(sample["__key__"])
        else:
            extra = associator.get(sample["__key__"], {})
        sample.update(extra)  # destructive
        yield sample


def map_dict(data, handler=reraise_exception, **kw):
    assert len(list(kw.keys())) > 0
    for key, f in kw.items():
        assert callable(f), (key, f)

    for sample in data:
        assert isinstance(sample, dict)
        try:
            for k, f in kw.items():
                sample[k] = f(sample[k])
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
        yield sample


def to_tuple(data, *args, handler=reraise_exception):
    if len(args) == 1 and isinstance(args[0], str) and " " in args[0]:
        args = args[0].split()

    for sample in data:
        try:
            yield tuple([getfirst(sample, f, missing_is_error=True) for f in args])
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


def map_tuple(data, *args, handler=reraise_exception):
    for f in args:
        assert callable(f), f
    for sample in data:
        assert isinstance(sample, (list, tuple))
        sample = list(sample)
        n = min(len(args), len(sample))
        try:
            for i in range(n):
                sample[i] = args[i](sample[i])
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
        yield tuple(sample)


def default_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a collection of samples (dictionaries) and create a batch.

    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.

    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict

    """
    batched = list(zip(*samples))
    result = []
    for b in batched:
        if isinstance(b[0], (int, float)):
            if combine_scalars:
                b = np.array(list(b))
        elif isinstance(b[0], TorchTensor):
            if combine_tensors:
                import torch

                b = torch.stack(list(b))
        elif isinstance(b[0], np.ndarray):
            if combine_tensors:
                b = np.array(list(b))
        else:
            b = list(b)
        result.append(b)
    return result


def batched(
    data, batchsize=20, collation_fn=default_collation_fn, partial=True,
):
    """Create batches of the given size.

    :param data: iterator
    :param batchsize: target batch size
    :param tensors: automatically batch lists of ndarrays into ndarrays
    :param partial: return partial batches
    :returns: iterator

    """

    batch = []
    for sample in data:
        if len(batch) >= batchsize:
            yield collation_fn(batch)
            batch = []
        batch.append(sample)
    if len(batch) == 0:
        return
    elif len(batch) == batchsize or partial:
        yield collation_fn(batch)


def unbatched(data):
    for sample in data:
        assert isinstance(sample, (tuple, list)), sample
        assert len(sample) > 0
        for i in range(len(sample[0])):
            yield tuple(x[i] for x in sample)

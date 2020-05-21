#!/usr/bin/python
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Train PyTorch models directly from POSIX tar archive, locally
or over HTTP connections.
"""

__all__ = "WebDataset tariterator default_handlers imagehandler".split()

import sys
import random
from functools import reduce, wraps
import numpy as np

from .checks import checktype
from . import autodecode


try:
    from torch import Tensor as TorchTensor
except ModuleNotFoundError:
    class TorchTensor:
        pass


def reraise_exception(exn):
    raise exn


def curried(f):
    """A decorator for currying functions in the first argument."""

    @wraps(f)
    def wrapper(*args, **kw):
        def g(x):
            return f(x, *args, **kw)

        return g

    return wrapper


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
    if isinstance(fields, str):
        fields = fields.split()
    return [field.split(";") for field in fields]


def transform_with(sample, transformers):
    """Transform a list of values using a list of functions.

    sample: list of values
    transformers: list of functions

    """
    checktype(sample, (tuple, list))
    if transformers is None or len(transformers) == 0:
        return sample
    result = list(sample)
    ntransformers = len(transformers)
    for i in range(len(sample)):  # skipcq: PYL-C0200
        f = transformers[i % ntransformers]
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


# @curried
# def associate(data, associator):
#     """Extract the given fields and return a tuple.
#     """
#     for sample in data:
#         if callable(associator):
#             extra = associator(sample["__key__"])
#         else:
#             extra = associator.get(sample["__key__"], {})
#         sample.update(extra)
#         yield sample


# @curried
# def extract(data, *fields):
#     """Extract the given fields and return a tuple.
#     """
#     for sample in data:
#         if fields is None:
#             yield sample
#         else:
#             yield [getfirst(sample, f) for f in fields]


@curried
def map_stream(data, f=None, handler=reraise_exception):
    """Map entire samples using the given function.

    data: iterator
    f: function from samples to samples
    returns: iterator over transformed samples

    """

    if f is None:

        def f(x):  # skipcq: PYL-E0102
            return x

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


@curried
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


@curried
def shuffle(data, bufsize=1000, initial=100, rng=random):
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


@curried
def select(data, predicate):
    for sample in data:
        if predicate(sample):
            yield sample


def decode(decoder="rgb", handler=reraise_exception):
    f = autodecode.make_decoder(decoder)

    def stage(data):
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

    return stage


def map(f, handler=reraise_exception):
    def stage(data):
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

    return stage


def rename(handler=reraise_exception, **kw):
    def stage(data):
        for sample in data:
            try:
                yield {
                    k: getfirst(sample, v, missing_is_error=True) for k, v in kw.items()
                }
            except Exception as exn:
                if handler(exn):
                    continue
                else:
                    break

    return stage


def associate(associator, **kw):
    def stage(data):
        for sample in data:
            if callable(associator):
                extra = associator(sample["__key__"])
            else:
                extra = associator.get(sample["__key__"], {})
            sample.update(extra)  # destructive
            yield sample

    return stage


def map_dict(handler=reraise_exception, **kw):
    assert len(list(kw.keys())) > 0
    for f in kw.values():
        assert callable(f)

    def stage(data):
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

    return stage


def to_tuple(*args, handler=reraise_exception):
    if len(args) == 1 and isinstance(args[0], str) and " " in args[0]:
        args = args[0].split()

    def stage(data):
        for sample in data:
            try:
                yield tuple([getfirst(sample, f, missing_is_error=True) for f in args])
            except Exception as exn:
                if handler(exn):
                    continue
                else:
                    break

    return stage


def map_tuple(*args, handler=reraise_exception):
    def stage(data):
        for f in args:
            assert callable(f)
        for sample in data:
            assert isinstance(sample, (list, tuple))
            assert len(args) == len(
                sample
            ), f"len(args) {len(args)} != len(sample) {len(sample)}"
            sample = list(sample)
            try:
                for i in range(min(len(args), len(sample))):
                    sample[i] = args[i](sample[i])
            except Exception as exn:
                if handler(exn):
                    continue
                else:
                    break
            yield tuple(sample)

    return stage


def batch_tensors(l, expand=False):
    if isinstance(l[0], np.ndarray):
        if not expand:
            return np.ndarray(list(l))
        else:
            raise Exception("unimplemented")
    elif isinstance(l[0], TorchTensor):
        import torch
        if not expand:
            return torch.stack(list(l))
        else:
            raise Exception("unimplemented")
    else:
        raise ValueError(f"{l}: cannot combine")


def samples_to_batch(samples, combine_tensors=True, combine_scalars=True, expand=False):
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
        elif isinstance(b[0], (TorchTensor, np.ndarray)):
            if combine_tensors:
                b = batch_tensors(b, expand=expand)
        else:
            b = list(b)
        result.append(b)
    return result


def batched(
    batchsize=20, combine_tensors=True, combine_scalars=True, partial=True, expand=False
):
    """Create batches of the given size.

    :param data: iterator
    :param batchsize: target batch size
    :param tensors: automatically batch lists of ndarrays into ndarrays
    :param partial: return partial batches
    :returns: iterator

    """

    def stage(data):
        batch = []
        for sample in data:
            if len(batch) >= batchsize:
                yield samples_to_batch(
                    batch,
                    combine_tensors=combine_tensors,
                    combine_scalars=combine_scalars,
                    expand=expand,
                )
                batch = []
            batch.append(sample)
        if len(batch) == 0:
            return
        elif len(batch) == batchsize or partial:
            yield samples_to_batch(
                batch,
                combine_tensors=combine_tensors,
                combine_scalars=combine_scalars,
                expand=expand,
            )

    return stage

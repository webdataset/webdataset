#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""A collection of iterators for data transformations.

These functions are plain iterator functions. You can find curried versions
in webdataset.filters, and you can find IterableDataset wrappers in
webdataset.processing.
"""

import os
import time
import random
import sys
from functools import reduce

import numpy as np

from . import autodecode
from . import utils
from .checks import checktype

try:
    from torch import Tensor as TorchTensor
except ModuleNotFoundError:

    class TorchTensor:
        """TorchTensor."""

        pass


###
# Helpers
###


def reraise_exception(exn):
    """Reraises the given exception; used as a handler.

    :param exn: exception
    """
    raise exn


def identity(x):
    """Return the argument."""
    return x


def compose2(f, g):
    """Compose two functions, g(f(x))."""
    return lambda x: g(f(x))


def compose(*args):
    """Compose a sequence of functions (left-to-right)."""
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


###
# Iterators
###


def info(data, fmt=None, n=3, every=-1, width=50, stream=sys.stderr, name=""):
    """Print information about the samples that are passing through.

    :param data: source iterator
    :param fmt: format statement (using sample dict as keyword)
    :param n: when to stop
    :param every: how often to print
    :param width: maximum width
    :param stream: output stream
    :param name: identifier printed before any output
    """
    for i, sample in enumerate(data):
        if i < n or (every > 0 and (i + 1) % every == 0):
            if fmt is None:
                print("---", name, file=stream)
                for k, v in sample.items():
                    print(k, repr(v)[:width], file=stream)
            else:
                print(fmt.format(**sample), file=stream)
        yield sample


shuffle_rng = random.Random()
shuffle_rng.seed((os.getpid(), time.time()))


def shuffle(data, bufsize=1000, initial=100, rng=shuffle_rng, handler=None):
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
    """Select samples based on a predicate.

    :param data: source iterator
    :param predicate: predicate (function)
    """
    for sample in data:
        if predicate(sample):
            yield sample


def log_keys(data, logfile=None):
    import fcntl
    if logfile is None or logfile == "":
        for sample in data:
            yield sample
    else:
        with open(logfile, "a") as stream:
            for i, sample in enumerate(data):
                buf = f"{i}\t{sample.get('__worker__')}\t{sample.get('__rank__')}\t{sample.get('__key__')}\n"
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                    stream.write(buf)
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                yield sample


def decode(data, *args, handler=reraise_exception, **kw):
    """Decode data based on the decoding functions given as arguments."""
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
    """Map samples."""
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


def rename(data, handler=reraise_exception, keep=True, **kw):
    """Rename samples based on keyword arguments."""
    for sample in data:
        try:
            if not keep:
                yield {k: getfirst(sample, v, missing_is_error=True) for k, v in kw.items()}
            else:
                def listify(v):
                    return v.split(";") if isinstance(v, str) else v
                to_be_replaced = {x for v in kw.values() for x in listify(v)}
                result = {k: v for k, v in sample.items() if k not in to_be_replaced}
                result.update({k: getfirst(sample, v, missing_is_error=True) for k, v in kw.items()})
                yield result
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


def associate(data, associator, **kw):
    """Associate additional data with samples."""
    for sample in data:
        if callable(associator):
            extra = associator(sample["__key__"])
        else:
            extra = associator.get(sample["__key__"], {})
        sample.update(extra)  # destructive
        yield sample


def map_dict(data, handler=reraise_exception, **kw):
    """Map the entries in a dict sample with individual functions."""
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


def to_tuple(data, *args, handler=reraise_exception, missing_is_error=True, none_is_error=None):
    """Convert dict samples to tuples."""
    if none_is_error is None:
        none_is_error = missing_is_error
    if len(args) == 1 and isinstance(args[0], str) and " " in args[0]:
        args = args[0].split()

    for sample in data:
        try:
            result = tuple([getfirst(sample, f, missing_is_error=missing_is_error) for f in args])
            if none_is_error and any(x is None for x in result):
                raise ValueError(f"to_tuple {args} got {sample.keys()}")
            yield result
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


def map_tuple(data, *args, handler=reraise_exception):
    """Map the entries of a tuple with individual functions."""
    args = [f if f is not None else utils.identity for f in args]
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
    assert isinstance(samples[0], (list, tuple)), type(samples[0])
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
    data,
    batchsize=20,
    collation_fn=default_collation_fn,
    partial=True,
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
            if collation_fn is not None:
                batch = collation_fn(batch)
            yield batch
            batch = []
        batch.append(sample)
    if len(batch) == 0:
        return
    elif len(batch) == batchsize or partial:
        if collation_fn is not None:
            batch = collation_fn(batch)
        yield batch


def unlisted(data):
    """Turn batched data back into unbatched data."""
    for batch in data:
        assert isinstance(batch, list), sample
        for sample in batch:
            yield sample


def unbatched(data):
    """Turn batched data back into unbatched data."""
    for sample in data:
        assert isinstance(sample, (tuple, list)), sample
        assert len(sample) > 0
        for i in range(len(sample[0])):
            yield tuple(x[i] for x in sample)


def rsample(data, p=0.5):
    """Randomly subsample a stream of data."""
    assert p >= 0.0 and p <= 1.0
    for sample in data:
        if random.uniform(0.0, 1.0) < p:
            yield sample

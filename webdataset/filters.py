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

import io
from fnmatch import fnmatch
import re
import itertools, os, random, sys, time
from functools import reduce, wraps
import pickle

import numpy as np

from . import autodecode, utils
from .pytorch import TorchTensor
from .utils import PipelineStage


class FilterFunction(object):
    """Helper class for currying pipeline stages.

    We use this roundabout construct becauce it can be pickled.
    """

    def __init__(self, f, *args, **kw):
        """Create a curried function."""
        self.f = f
        self.args = args
        self.kw = kw

    def __call__(self, data):
        """Call the curried function with the given argument."""
        return self.f(data, *self.args, **self.kw)

    def __str__(self):
        """Compute a string representation."""
        return f"<{self.f.__name__} {self.args} {self.kw}>"

    def __repr__(self):
        """Compute a string representation."""
        return f"<{self.f.__name__} {self.args} {self.kw}>"


class RestCurried(object):
    """Helper class for currying pipeline stages.

    We use this roundabout construct because it can be pickled.
    """

    def __init__(self, f):
        """Store the function for future currying."""
        self.f = f

    def __call__(self, *args, **kw):
        """Curry with the given arguments."""
        return FilterFunction(self.f, *args, **kw)


def pipelinefilter(f):
    """Turn the decorated function into one that is partially applied for
    all arguments other than the first."""
    result = RestCurried(f)
    return result


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


def _info(data, fmt=None, n=3, every=-1, width=50, stream=sys.stderr, name=""):
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


info = pipelinefilter(_info)


def pick(buf, rng):
    k = rng.randint(0, len(buf) - 1)
    sample = buf[k]
    buf[k] = buf[-1]
    buf.pop()
    return sample


def _shuffle(data, bufsize=1000, initial=100, rng=None, handler=None):
    """Shuffle the data in the stream.

    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.

    data: iterator
    bufsize: buffer size for shuffling
    returns: iterator
    rng: either random module or random.Random instance

    """
    if rng is None:
        rng = random.Random(int((os.getpid() + time.time()) * 1e9))
    initial = min(initial, bufsize)
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) < bufsize:
            try:
                buf.append(next(data))  # skipcq: PYL-R1708
            except StopIteration:
                pass
        if len(buf) >= initial:
            yield pick(buf, rng)
    while len(buf) > 0:
        yield pick(buf, rng)


shuffle = pipelinefilter(_shuffle)


class detshuffle(PipelineStage):
    def __init__(self, bufsize=1000, initial=100, seed=0, epoch=-1):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        self.epoch += 1
        rng = random.Random()
        rng.seed(self.seed + self.epoch)
        return _shuffle(src, self.bufsize, self.initial, rng)


def _select(data, predicate):
    """Select samples based on a predicate.

    :param data: source iterator
    :param predicate: predicate (function)
    """
    for sample in data:
        if predicate(sample):
            yield sample


select = pipelinefilter(_select)


def _log_keys(data, logfile=None):
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


log_keys = pipelinefilter(_log_keys)


def _decode(data, *args, handler=reraise_exception, **kw):
    """Decode data based on the decoding functions given as arguments."""

    decoder = lambda x: autodecode.imagehandler(x) if isinstance(x, str) else x
    handlers = [decoder(x) for x in args]
    f = autodecode.Decoder(handlers, **kw)

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


decode = pipelinefilter(_decode)


def _map(data, f, handler=reraise_exception):
    """Map samples."""
    for sample in data:
        try:
            result = f(sample)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
        if result is None:
            continue
        if isinstance(sample, dict) and isinstance(result, dict):
            result["__key__"] = sample.get("__key__")
        yield result


map = pipelinefilter(_map)


def _rename(data, handler=reraise_exception, keep=True, **kw):
    """Rename samples based on keyword arguments."""
    for sample in data:
        try:
            if not keep:
                yield {
                    k: getfirst(sample, v, missing_is_error=True) for k, v in kw.items()
                }
            else:

                def listify(v):
                    return v.split(";") if isinstance(v, str) else v

                to_be_replaced = {x for v in kw.values() for x in listify(v)}
                result = {k: v for k, v in sample.items() if k not in to_be_replaced}
                result.update(
                    {
                        k: getfirst(sample, v, missing_is_error=True)
                        for k, v in kw.items()
                    }
                )
                yield result
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


rename = pipelinefilter(_rename)


def _associate(data, associator, **kw):
    """Associate additional data with samples."""
    for sample in data:
        if callable(associator):
            extra = associator(sample["__key__"])
        else:
            extra = associator.get(sample["__key__"], {})
        sample.update(extra)  # destructive
        yield sample


associate = pipelinefilter(_associate)


def _map_dict(data, handler=reraise_exception, **kw):
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


map_dict = pipelinefilter(_map_dict)


def _to_tuple(
    data, *args, handler=reraise_exception, missing_is_error=True, none_is_error=None
):
    """Convert dict samples to tuples."""
    if none_is_error is None:
        none_is_error = missing_is_error
    if len(args) == 1 and isinstance(args[0], str) and " " in args[0]:
        args = args[0].split()

    for sample in data:
        try:
            result = tuple(
                [getfirst(sample, f, missing_is_error=missing_is_error) for f in args]
            )
            if none_is_error and any(x is None for x in result):
                raise ValueError(f"to_tuple {args} got {sample.keys()}")
            yield result
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


to_tuple = pipelinefilter(_to_tuple)


def _map_tuple(data, *args, handler=reraise_exception):
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


map_tuple = pipelinefilter(_map_tuple)


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


def _batched(
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


batched = pipelinefilter(_batched)


def _unlisted(data):
    """Turn batched data back into unbatched data."""
    for batch in data:
        assert isinstance(batch, list), sample
        for sample in batch:
            yield sample


unlisted = pipelinefilter(_unlisted)


def _unbatched(data):
    """Turn batched data back into unbatched data."""
    for sample in data:
        assert isinstance(sample, (tuple, list)), sample
        assert len(sample) > 0
        for i in range(len(sample[0])):
            yield tuple(x[i] for x in sample)


unbatched = pipelinefilter(_unbatched)


def _rsample(data, p=0.5):
    """Randomly subsample a stream of data."""
    assert p >= 0.0 and p <= 1.0
    for sample in data:
        if random.uniform(0.0, 1.0) < p:
            yield sample


rsample = pipelinefilter(_rsample)

slice = pipelinefilter(itertools.islice)


def _extract_keys(source, *patterns, duplicate_is_error=True, ignore_missing=False):
    for sample in source:
        result = []
        for pattern in patterns:
            pattern = pattern.split(";") if isinstance(pattern, str) else pattern
            matches = [
                x for x in sample.keys() if any(fnmatch("." + x, p) for p in pattern)
            ]
            if len(matches) == 0:
                if ignore_missing:
                    continue
                else:
                    raise ValueError(
                        f"Cannot find {pattern} in sample keys {sample.keys()}."
                    )
            if len(matches) > 1 and duplicate_is_error:
                raise ValueError(
                    f"Multiple sample keys {sample.keys()} match {pattern}."
                )
            value = sample[matches[0]]
            result.append(value)
        yield tuple(result)


extract_keys = pipelinefilter(_extract_keys)


def _rename_keys(
    source, *args, keep_unselected=False, must_match=True, duplicate_is_error=True, **kw
):
    renamings = [(pattern, output) for output, pattern in args]
    renamings += [(pattern, output) for output, pattern in kw.items()]
    for sample in source:
        new_sample = {}
        matched = {k: False for k, _ in renamings}
        for path, value in sample.items():
            fname = re.sub(r".*/", "", path)
            new_name = None
            for pattern, name in renamings[::-1]:
                if fnmatch(fname.lower(), pattern):
                    matched[pattern] = True
                    new_name = name
                    break
            if new_name is None:
                if keep_unselected:
                    new_sample[path] = value
                continue
            if new_name in new_sample:
                if duplicate_is_error:
                    raise ValueError(
                        f"Duplicate value in sample {sample.keys()} after rename."
                    )
                continue
            new_sample[new_name] = value
        if must_match and not all(matched.values()):
            raise ValueError(
                f"Not all patterns ({matched}) matched sample keys ({sample.keys()})."
            )

        yield new_sample


rename_keys = pipelinefilter(_rename_keys)


def decode_bin(stream):
    return stream.read()


def decode_text(stream):
    binary = stream.read()
    return binary.decode("utf-8")


def decode_pickle(stream):
    return pickle.load(stream)


default_decoders = [
    ("*.bin", decode_bin),
    ("*.txt", decode_text),
    ("*.pyd", decode_pickle),
]


def find_decoder(decoders, path):
    fname = re.sub(r".*/", "", path)
    if fname.startswith("__"):
        return lambda x: x
    for pattern, fun in decoders[::-1]:
        if fnmatch(fname.lower(), pattern) or fnmatch("." + fname.lower(), pattern):
            return fun
    return None


def _xdecode(
    source,
    *args,
    must_decode=True,
    defaults=default_decoders,
    **kw,
):
    decoders = list(defaults) + list(args)
    decoders += [("*." + k, v) for k, v in kw.items()]
    for sample in source:
        new_sample = {}
        for path, data in sample.items():
            if path.startswith("__"):
                new_sample[path] = data
                continue
            decoder = find_decoder(decoders, path)
            if decoder is False:
                value = data
            elif decoder is None:
                if must_decode:
                    raise ValueError(f"No decoder found for {path}.")
                value = data
            else:
                if isinstance(data, bytes):
                    data = io.BytesIO(data)
                value = decoder(data)
            new_sample[path] = value
        yield new_sample


xdecode = pipelinefilter(_xdecode)


class Cached(PipelineStage):
    def __init__(self):
        super().__init__()
        self.cached = None

    def run(self, source):
        if self.cached is None:
            self.temp = []
            for sample in source:
                self.temp.append(sample)
                yield sample
            self.cached = self.temp
        else:
            for sample in self.cached:
                yield sample


class LMDBCached(PipelineStage):
    def __init__(self, fname, map_size=1e12, pickler=pickle, chunksize=500):
        import lmdb

        self.db = lmdb.open(fname, readonly=False, map_size=int(map_size))
        self.pickler = pickler
        self.chunksize = chunksize

    def is_complete(self):
        with self.db.begin(write=False) as txn:
            return txn.get(b"_") is not None

    def add_samples(self, samples):
        with self.db.begin(write=True) as txn:
            for key, sample in samples:
                txn.put(key.encode(), self.pickler.dumps(sample))

    def run(self, source):
        if self.is_complete():
            with self.db.begin(write=False) as txn:
                for key, value in txn.cursor():
                    if key == b"_":
                        continue
                    yield self.pickler.loads(value)
        else:
            buffer = []
            for i, sample in enumerate(source):
                key = (isinstance(sample, dict) and sample.get("__key__")) or str(i)
                buffer.append((key, sample))
                if len(buffer) >= self.chunksize:
                    self.add_samples(buffer)
                    buffer = []
                yield sample
            if len(buffer) > 0:
                self.add_samples(buffer)
            with self.db.begin(write=True) as txn:
                txn.put(b"_", b"1")

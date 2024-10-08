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

import functools
import io
import itertools
import os
import pickle
import random
import re
import sys
import time
from fnmatch import fnmatch
from functools import reduce

import numpy as np

from . import autodecode, utils
from .pytorch import TorchTensor
from .utils import PipelineStage


class FilterFunction(object):
    """
    Helper class for currying pipeline stages.

    This class is used to create a curried function that can be pickled.

    Attributes:
        f: The function to be curried.
        args: Positional arguments for the function.
        kw: Keyword arguments for the function.
    """

    def __init__(self, f, *args, **kw):
        """
        Create a curried function.

        Args:
            f: The function to be curried.
            *args: Positional arguments for the function.
            **kw: Keyword arguments for the function.
        """
        self.f = f
        self.args = args
        self.kw = kw

    def __call__(self, data):
        """
        Call the curried function with the given argument.

        Args:
            data: The data to be processed by the curried function.

        Returns:
            The result of calling the curried function with the given data and stored arguments.
        """
        return self.f(data, *self.args, **self.kw)

    def __str__(self):
        """
        Compute a string representation.

        Returns:
            str: A string representation of the FilterFunction object.
        """
        return f"<{self.f.__name__} {self.args} {self.kw}>"

    def __repr__(self):
        """
        Compute a string representation.

        Returns:
            str: A string representation of the FilterFunction object.
        """
        return f"<{self.f.__name__} {self.args} {self.kw}>"


class RestCurried(object):
    """
    Helper class for currying pipeline stages.

    This class is used to create a curried function that can be pickled.

    Attributes:
        f: The function to be curried.
    """

    def __init__(self, f):
        """
        Store the function for future currying.

        Args:
            f: The function to be curried.
        """
        self.f = f

    def __call__(self, *args, **kw):
        """
        Curry with the given arguments.

        Args:
            *args: Positional arguments for the function.
            **kw: Keyword arguments for the function.

        Returns:
            FilterFunction: A FilterFunction object with the curried function and arguments.
        """
        return FilterFunction(self.f, *args, **kw)


def pipelinefilter(f):
    """
    Turn the decorated function into one that is partially applied for all arguments other than the first.

    Args:
        f: The function to be decorated.

    Returns:
        RestCurried: A RestCurried object that can be used to create a FilterFunction.
    """
    result = RestCurried(f)
    functools.update_wrapper(result, f)
    return result


def reraise_exception(exn):
    """
    Reraise the given exception.

    Args:
        exn: The exception to be reraised.

    Raises:
        The input exception.
    """
    raise exn


def identity(x):
    """
    Return the argument unchanged.

    Args:
        x: The input value.

    Returns:
        The input value unchanged.
    """
    return x


def compose2(f, g):
    """
    Compose two functions, g(f(x)).

    Args:
        f: The first function to be composed.
        g: The second function to be composed.

    Returns:
        function: A new function that applies f and then g to its input.
    """
    return lambda x: g(f(x))


def compose(*args):
    """
    Compose a sequence of functions (left-to-right).

    Args:
        *args: Functions to be composed.

    Returns:
        function: A new function that applies all input functions in sequence.
    """
    return reduce(compose2, args)


def pipeline(source, *args):
    """
    Write an input pipeline; first argument is source, rest are filters.

    Args:
        source: The data source for the pipeline.
        *args: Filters to be applied to the data.

    Returns:
        The result of applying all filters to the source data.
    """
    if len(args) == 0:
        return source
    return compose(*args)(source)


def getfirst(a, keys, default=None, missing_is_error=True):
    """
    Get the first matching key from a dictionary.

    Keys can be specified as a list, or as a string of keys separated by ';'.

    Args:
        a (dict): The dictionary to search.
        keys (str or list): The keys to search for.
        default: The default value to return if no key is found.
        missing_is_error (bool): If True, raise an error when no key is found.

    Returns:
        The value of the first matching key found in the dictionary.

    Raises:
        ValueError: If no matching key is found and missing_is_error is True.
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
    """
    Parse a specification for a list of fields to be extracted.

    Keys are separated by spaces in the spec. Each key can itself
    be composed of key alternatives separated by ';'.

    Args:
        fields (str or list): The field specification to parse.

    Returns:
        list: A list of parsed field specifications.
    """
    if isinstance(fields, str):
        fields = fields.split()
    return [field.split(";") for field in fields]


def transform_with(sample, transformers):
    """
    Transform a list of values using a list of functions.

    If there are fewer transformers than inputs, or if a transformer
    function is None, then the identity function is used for the
    corresponding sample fields.

    Args:
        sample (list): List of values to transform.
        transformers (list): List of functions to apply to the sample.

    Returns:
        list: The transformed sample.
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
    """
    Print information about the samples that are passing through.

    Args:
        data: Source iterator.
        fmt (str): Format statement (using sample dict as keyword).
        n (int): When to stop printing.
        every (int): How often to print.
        width (int): Maximum width for printed values.
        stream: Output stream.
        name (str): Identifier printed before any output.

    Yields:
        The samples from the input iterator.
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
    """
    Pick a random item from the buffer and remove it.

    Args:
        buf (list): The buffer to pick from.
        rng: Random number generator.

    Returns:
        The randomly picked item.
    """
    k = rng.randint(0, len(buf) - 1)
    sample = buf[k]
    buf[k] = buf[-1]
    buf.pop()
    return sample


def _shuffle(data, bufsize=1000, initial=100, rng=None, seed=None, handler=None):
    """
    Shuffle the data in the stream.

    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.

    Args:
        data: Iterator to shuffle.
        bufsize (int): Buffer size for shuffling.
        initial (int): Initial buffer size before yielding.
        rng: Random number generator.
        seed: Seed for the random number generator.
        handler: Exception handler.

    Yields:
        Shuffled items from the input iterator.
    """
    if seed is not None:
        assert rng is None
        rng = random.Random(seed)
    elif rng is None:
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
    """
    A deterministic shuffling stage for the pipeline.

    This class provides a reproducible shuffling mechanism based on a seed and epoch.

    Attributes:
        bufsize (int): Size of the buffer for shuffling.
        initial (int): Initial number of samples to collect before shuffling.
        seed (int): Seed for the random number generator.
        epoch (int): Current epoch number.
    """

    def __init__(self, bufsize=1000, initial=100, seed=0, epoch=-1):
        """
        Initialize the detshuffle stage.

        Args:
            bufsize (int): Size of the buffer for shuffling.
            initial (int): Initial number of samples to collect before shuffling.
            seed (int): Seed for the random number generator.
            epoch (int): Starting epoch number.
        """
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        """
        Run the shuffling process on the input source.

        Args:
            src: Input data source to be shuffled.

        Returns:
            Iterator: Shuffled data iterator.
        """
        self.epoch += 1
        rng = random.Random()
        rng.seed(self.seed + self.epoch)
        return _shuffle(src, self.bufsize, self.initial, rng)


def _select(data, predicate):
    """
    Select samples based on a predicate.

    Args:
        data: Source iterator.
        predicate: Function that returns True for samples to be selected.

    Yields:
        Samples that satisfy the predicate.
    """
    for sample in data:
        if predicate(sample):
            yield sample


select = pipelinefilter(_select)


def _log_keys(data, logfile=None):
    """
    Log keys of the samples passing through the pipeline.

    Args:
        data: Source iterator.
        logfile (str): Path to the log file.

    Yields:
        Samples from the input iterator.
    """
    import fcntl

    if logfile is None or logfile == "":
        yield from data
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
    """
    Decode data based on the decoding functions given as arguments.

    Args:
        data: Source iterator.
        *args: Decoding functions to be applied.
        handler: Exception handler function.
        **kw: Additional keyword arguments for the Decoder.

    Yields:
        Decoded samples.

    Raises:
        Exception: If the handler doesn't handle an exception.
    """
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
    """
    Map samples through a function.

    Args:
        data: Source iterator.
        f: Function to apply to each sample.
        handler: Exception handler function.

    Yields:
        Processed samples.

    Raises:
        Exception: If the handler doesn't handle an exception.
    """
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
    """
    Rename samples based on keyword arguments.

    Args:
        data: Source iterator.
        handler: Exception handler function.
        keep (bool): Whether to keep original keys not being renamed.
        **kw: Mapping of new names to old names.

    Yields:
        Samples with renamed keys.

    Raises:
        Exception: If the handler doesn't handle an exception.
    """
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
    """
    Associate additional data with samples.

    Args:
        data: Source iterator.
        associator: Function or dictionary to associate extra data.
        **kw: Additional keyword arguments.

    Yields:
        Samples with associated data.
    """
    for sample in data:
        if callable(associator):
            extra = associator(sample["__key__"])
        else:
            extra = associator.get(sample["__key__"], {})
        sample.update(extra)  # destructive
        yield sample


associate = pipelinefilter(_associate)


def _map_dict(data, handler=reraise_exception, **kw):
    """
    Map the entries in a dict sample with individual functions.

    Args:
        data: Source iterator of dictionary samples.
        handler: Exception handler function.
        **kw: Mapping of keys to functions to apply.

    Yields:
        Samples with mapped values.

    Raises:
        Exception: If the handler doesn't handle an exception.
    """
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
    """
    Convert dict samples to tuples.

    Args:
        data: Source iterator of dictionary samples.
        *args: Keys to extract from the dictionaries.
        handler: Exception handler function.
        missing_is_error (bool): Whether missing keys should raise an error.
        none_is_error (bool): Whether None values should raise an error.

    Yields:
        Tuples of extracted values.

    Raises:
        Exception: If the handler doesn't handle an exception.
    """
    if none_is_error is None:
        none_is_error = missing_is_error
    if len(args) == 1 and isinstance(args[0], str) and " " in args[0]:
        args = args[0].split()

    for sample in data:
        try:
            result = tuple(
                getfirst(sample, f, missing_is_error=missing_is_error) for f in args
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
    """
    Map the entries of a tuple with individual functions.

    Args:
        data: Source iterator of tuple samples.
        *args: Functions to apply to each element of the tuples.
        handler: Exception handler function.

    Yields:
        Tuples with mapped values.

    Raises:
        Exception: If the handler doesn't handle an exception.
    """
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


def combine_values(b, combine_tensors=True, combine_scalars=True):
    if isinstance(b[0], (int, float)):
        if combine_scalars:
            b = np.array(list(b))
    elif isinstance(b[0], TorchTensor):
        if combine_tensors:
            import torch

            shapes = set(x.shape for x in b)
            assert (
                len(shapes) == 1
            ), f"all shapes must be equal in collation, got {shapes}"
            b = torch.stack(list(b))
    elif isinstance(b[0], np.ndarray):
        if combine_tensors:
            shapes = set(x.shape for x in b)
            assert (
                len(shapes) == 1
            ), f"all shapes must be equal in collation, got {shapes}"
            b = np.array(list(b))
    else:
        b = list(b)
    return b


def tuple2dict(l):
    if isinstance(l, dict):
        return l
    return {i: d for i, d in enumerate(l)}


def dict2tuple(d):
    return tuple(d[i] for i in range(1 + max(d.keys())))


def default_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """
    Take a collection of samples (dictionaries or tuples) and create a batch.

    Args:
        samples (list): List of samples to be batched.
        combine_tensors (bool): Whether to combine tensor-like objects into batches.
        combine_scalars (bool): Whether to combine scalar values into numpy arrays.

    Returns:
        list: A batch of samples.
    """
    rows = [tuple2dict(x) for x in samples]
    keys = set(rows[0].keys())
    for row in rows[1:]:
        assert set(row.keys()) == keys, "keys don't match in different samples"
    result = {
        k: combine_values([row[k] for row in rows], combine_tensors, combine_scalars)
        for k in keys
    }
    if isinstance(samples[0], (list, tuple)):
        return dict2tuple(result)
    else:
        return result


def _batched(
    data,
    batchsize=20,
    collation_fn=default_collation_fn,
    partial=True,
):
    """
    Create batches of the given size.

    Args:
        data: Iterator of samples.
        batchsize (int): Target batch size.
        collation_fn (callable): Function to use for collating samples into a batch.
        partial (bool): Whether to return partial batches at the end.

    Yields:
        Batches of samples.
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
    """
    Turn batched data back into unbatched data.

    Args:
        data: Iterator of batches.

    Yields:
        Individual samples from the batches.
    """
    for batch in data:
        assert isinstance(batch, list), batch
        yield from batch


unlisted = pipelinefilter(_unlisted)


def _unbatched(data):
    """
    Turn batched data back into unbatched data.

    Args:
        data: Iterator of batches.

    Yields:
        Individual samples from the batches.
    """
    for sample in data:
        if isinstance(sample, (list, tuple)):
            for i in range(len(sample[0])):
                yield tuple(x[i] for x in sample)
        elif isinstance(sample, dict):
            lengths = {len(v) for v in sample.values()}
            assert len(lengths) == 1, lengths
            n = list(lengths)[0]
            for i in range(n):
                yield {k: v[i] for k, v in sample.items()}
        else:
            raise ValueError(f"unknown sample type: {type(sample)}")


unbatched = pipelinefilter(_unbatched)


def _rsample(data, p=0.5):
    """
    Randomly subsample a stream of data.

    Args:
        data: Iterator of samples.
        p (float): Probability of keeping each sample.

    Yields:
        Randomly selected samples from the input stream.
    """
    assert p >= 0.0 and p <= 1.0
    for sample in data:
        if random.uniform(0.0, 1.0) < p:
            yield sample


rsample = pipelinefilter(_rsample)

slice = pipelinefilter(itertools.islice)


def _extract_keys(source, *patterns, duplicate_is_error=True, ignore_missing=False):
    """
    Extract values from samples based on key patterns.

    Args:
        source: Iterator of dictionary samples.
        *patterns: Patterns to match keys against.
        duplicate_is_error (bool): Whether multiple matches for a pattern should raise an error.
        ignore_missing (bool): Whether to ignore patterns that don't match any keys.

    Yields:
        Tuples of extracted values.

    Raises:
        ValueError: If a pattern matches multiple keys and duplicate_is_error is True,
                    or if a pattern doesn't match any keys and ignore_missing is False.
    """
    for sample in source:
        result = []
        for pattern in patterns:
            pattern = pattern.split(";") if isinstance(pattern, str) else pattern
            matches = [
                x
                for x in sample.keys()
                if any((fnmatch("." + x, p) or fnmatch(x, p)) for p in pattern)
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
    """
    Rename keys in dictionary samples based on patterns.

    Args:
        source: Iterator of dictionary samples.
        *args: Tuples of (new_name, pattern) for renaming.
        keep_unselected (bool): Whether to keep keys that don't match any patterns.
        must_match (bool): Whether all patterns must match at least one key.
        duplicate_is_error (bool): Whether multiple matches for a pattern should raise an error.
        **kw: Keyword arguments of the form new_name=pattern for renaming.

    Yields:
        Dictionary samples with renamed keys.

    Raises:
        ValueError: If a pattern matches multiple keys and duplicate_is_error is True,
                    or if not all patterns match and must_match is True.
    """
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
    """
    Decode binary data from a stream.

    Args:
        stream: A file-like object containing binary data.

    Returns:
        bytes: The binary data read from the stream.
    """
    return stream.read()


def decode_text(stream):
    """
    Decode text data from a stream.

    Args:
        stream: A file-like object containing text data.

    Returns:
        str: The decoded text data.
    """
    binary = stream.read()
    return binary.decode("utf-8")


def decode_pickle(stream):
    """
    Decode pickle data from a stream.

    Args:
        stream: A file-like object containing pickle data.

    Returns:
        The unpickled object.
    """
    return pickle.load(stream)


default_decoders = [
    ("*.bin", decode_bin),
    ("*.txt", decode_text),
    ("*.pyd", decode_pickle),
]


def find_decoder(decoders, path):
    """
    Find the appropriate decoder for a given path.

    Args:
        decoders: List of (pattern, decoder_function) pairs.
        path: The path to find a decoder for.

    Returns:
        callable: The decoder function for the given path, or None if no match is found.
    """
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
    """
    Decode data in samples using specified decoders.

    Args:
        source: Iterator of dictionary samples.
        *args: Additional (pattern, decoder_function) pairs.
        must_decode (bool): Whether all data must be decoded.
        defaults: Default decoders to use.
        **kw: Additional decoders specified as key=decoder_function.

    Yields:
        Dictionary samples with decoded data.

    Raises:
        ValueError: If no decoder is found for a key and must_decode is True.
    """
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
    """
    A pipeline stage that caches its output.

    This stage will cache all samples that pass through it, allowing subsequent
    iterations to use the cached data instead of recomputing it.
    """

    def __init__(self):
        """Initialize the Cached pipeline stage."""
        super().__init__()
        self.cached = None

    def run(self, source):
        """
        Run the caching process on the input source.

        Args:
            source: Input data source to be cached.

        Yields:
            Samples from the source, caching them for future use.
        """
        if self.cached is None:
            self.temp = []
            for sample in source:
                self.temp.append(sample)
                yield sample
            self.cached = self.temp
        else:
            yield from self.cached


class LMDBCached(PipelineStage):
    """
    A pipeline stage that caches its output in an LMDB database.

    This stage will cache all samples that pass through it in an LMDB database,
    allowing subsequent iterations to use the cached data instead of recomputing it.
    """

    def __init__(self, fname, map_size=1e12, pickler=pickle, chunksize=500):
        """
        Initialize the LMDBCached pipeline stage.

        Args:
            fname (str): Filename for the LMDB database.
            map_size (int): Maximum size database may grow to.
            pickler: Module to use for pickling (default is Python's pickle module).
            chunksize (int): Number of samples to write in each transaction.
        """
        import lmdb

        self.db = lmdb.open(fname, readonly=False, map_size=int(map_size))
        self.pickler = pickler
        self.chunksize = chunksize

    def is_complete(self):
        """
        Check if the database is complete.

        Returns:
            bool: True if the database is complete, False otherwise.
        """
        with self.db.begin(write=False) as txn:
            return txn.get(b"_") is not None

    def add_samples(self, samples):
        """
        Add samples to the database.

        Args:
            samples: Iterable of (key, sample) pairs to add to the database.
        """
        with self.db.begin(write=True) as txn:
            for key, sample in samples:
                txn.put(key.encode(), self.pickler.dumps(sample))

    def run(self, source):
        """
        Run the caching process on the input source.

        If the database is complete, yield samples from the database.
        Otherwise, yield samples from the source and cache them in the database.

        Args:
            source: Input data source to be cached.

        Yields:
            Samples from the source or the database.
        """
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

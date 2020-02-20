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

import gc
import os
import pickle
import random
import re
import tarfile
import time
import warnings
from builtins import range
from functools import wraps

import braceexpand
import numpy as np
import PIL
import PIL.Image
import simplejson
import six
from torch.utils.data import IterableDataset

from . import io
from .checks import checkmember, checktype, checkrange, checknotnone, checkcallable

trace = False

debug_dataset = os.environ.get("WDS_DEBUG", 0)
popen_bufsize = int(os.environ.get("WDS_BUFSIZE", "2000000"))

meta_prefix = "__"
meta_suffix = "__"

collection_counter = 0
collection_frequency = 50000


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


def curried(f):
    """A decorator for currying functions in the first argument."""
    @wraps(f)
    def wrapper(*args, **kw):
        def g(x):
            return f(x, *args, **kw)
        return g
    return wrapper


imagespecs = {
    "l8": ("numpy", "uint8", "l"),
    "rgb8": ("numpy", "uint8", "rgb"),
    "rgba8": ("numpy", "uint8", "rgba"),
    "l": ("numpy", "float", "l"),
    "rgb": ("numpy", "float", "rgb"),
    "rgba": ("numpy", "float", "rgba"),
    "torchl8": ("torch", "uint8", "l"),
    "torchrgb8": ("torch", "uint8", "rgb"),
    "torchrgba8": ("torch", "uint8", "rgba"),
    "torchl": ("torch", "float", "l"),
    "torchrgb": ("torch", "float", "rgb"),
    "torch": ("torch", "float", "rgb"),
    "torchrgba": ("torch", "float", "rgba"),
    "pill": ("pil", None, "l"),
    "pil": ("pil", None, "rgb"),
    "pilrgb": ("pil", None, "rgb"),
    "pilrgba": ("pil", None, "rgba"),
}


def imagehandler(data, imagespec):
    """Decode image data using the given `imagespec`.

    The `imagespec` specifies whether the image is decoded
    to numpy/torch/pi, decoded to uint8/float, and decoded
    to l/rgb/rgba:

    - l8: numpy uint8 l
    - rgb8: numpy uint8 rgb
    - rgba8: numpy uint8 rgba
    - l: numpy float l
    - rgb: numpy float rgb
    - rgba: numpy float rgba
    - torchl8: torch uint8 l
    - torchrgb8: torch uint8 rgb
    - torchrgba8: torch uint8 rgba
    - torchl: torch float l
    - torchrgb: torch float rgb
    - torch: torch float rgb
    - torchrgba: torch float rgba
    - pill: pil None l
    - pil: pil None rgb
    - pilrgb: pil None rgb
    - pilrgba: pil None rgba

    """
    checkmember(imagespec, list(imagespecs.keys()), "unknown image specification")
    atype, etype, mode = imagespecs[imagespec.lower()]
    with six.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        img = img.convert(mode.upper())
    if atype == "pil":
        return img
    elif atype == "numpy":
        result = np.asarray(img)
        checkmember(result.dtype, [np.uint8])
        if etype == "uint8":
            return result
        else:
            return result.astype("f") / 255.0
    elif atype == "torch":
        import torch
        result = np.asarray(img)
        checkmember(result.dtype, [np.uint8])
        if etype == "uint8":
            result = result.transpose(2, 0, 1)
            return torch.tensor(result)
        else:
            result = result.transpose(2, 0, 1)
            return torch.tensor(result).type(torch.float) / 255.0
    return None


def maybe_int(data):
    """Try to turn data into an int; if it fails, return data."""
    try:
        return int(data)
    except ValueError:
        return data


def make_handlers(imagetype):
    """Preload the default_handlers table."""
    handlers = {}
    for extension in ["cls", "cls2", "class", "count", "index", "inx", "id"]:
        handlers[extension] = maybe_int
    for extension in ["txt", "text", "transcript"]:
        handlers[extension] = lambda x: x.decode("utf-8")
    for extension in ["png", "jpg", "jpeg", "img", "image",
                      "pbm", "pgm", "ppm"]:
        handlers[extension] = lambda data: imagehandler(data, imagetype)
    for extension in ["pyd", "pickle"]:
        handlers[extension] = pickle.loads
    for extension in ["json", "jsn"]:
        handlers[extension] = simplejson.loads
    for extension in ["ten", "tb"]:
        from . import tenbin
        handlers[extension] = tenbin.decode_buffer
    try:
        import msgpack
        for extension in ["mp", "msgpack", "msg"]:
            handlers[extension] = msgpack.unpackb
    except ImportError:
        pass
    return handlers


default_handlers = {key: make_handlers(key) for key in imagespecs.keys()}
"""A mapping of filename extensions to loading functions.

You can modify this to suit your needs.

E.g.,

```Python
    default_handlers["mp4"] = my_mp4_decoder
```

will call `my_mp4_decoder` in order to decode files ending in `.mp4`.
The decoder takes a single argument, a bytestring, and returns the decoded
object that is returned as part of a sample by `WebDataset`.
"""


def decode_item_based_on_extension(data, tname, handlers):
    # Unicode change. If it is alread an unicode string,
    # no decoding (Byte->Unicode req)
    if isinstance(data, (int, float, str)):
        return data
    checktype(data, bytes)
    checktype(tname, str)
    extension = re.sub(r".*\.", "", tname).lower()
    decoder = handlers.get(extension)
    if decoder is None:
        return data
    else:
        return decoder(data)


def decode_sample_based_on_extensions(sample, handlers):
    """Autodecode a sample, using extensions as guide for how to decode.

    Args:
    sample: dictionary representing sample
    imagetype: format for images (gray, rgb, rgba, PIL; rgb8=8 bit)
    """
    result = {}
    for k, v in list(sample.items()):
        if k[0] == "_":
            if isinstance(v, bytes):
                v = v.decode('utf-8')
            result[k] = v
            continue
        checknotnone(v)
        result[k] = decode_item_based_on_extension(v, k, handlers=handlers)
    return result


def getfirst(a, keys, default=None):
    if isinstance(keys, str):
        keys = keys.split(";")
    for k in keys:
        result = a.get(k)
        if result is not None:
            return result
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
    def f(x): return transform_with(x, transformers)
    return f


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


@curried
def associate(data, associator):
    """Extract the given fields and return a tuple.
    """
    for sample in data:
        if callable(associator):
            extra = associator(sample["__key__"])
        else:
            extra = associator.get(sample["__key__"], {})
        sample.update(extra)
        yield sample


@curried
def extract(data, *fields):
    """Extract the given fields and return a tuple.
    """
    for sample in data:
        if fields is None:
            yield sample
        else:
            yield [getfirst(sample, f) for f in fields]


@curried
def transform(data, f=None):
    """Map entire samples using the given function.

    data: iterator
    f: function from samples to samples
    returns: iterator over transformed samples

    """

    if f is None:
        def f(x): return x  # skipcq: PYL-E0102
    for sample in data:
        result = f(sample)
        if isinstance(sample, dict) and isinstance(result, dict):
            result["__key__"] = sample.get("__key__")
        yield result


@curried
def shuffle(data, bufsize=1000, initial=100):
    """Shuffle the data in the stream.

    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.

    data: iterator
    bufsize: buffer size for shuffling
    returns: iterator

    """
    checkrange(initial, 0, bufsize+1)
    buf = []
    startup = True
    for sample in data:
        if len(buf) < bufsize:
            buf.append(next(data))  # skipcq: PYL-R1708
        k = random.randint(0, len(buf) - 1)
        sample, buf[k] = buf[k], sample
        if startup and len(buf) < initial:
            buf.append(sample)
            continue
        startup = False
        yield sample
    for sample in buf:
        yield sample


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
    return (sample is not None and
            isinstance(sample, dict) and
            len(list(sample.keys())) > 0 and
            not sample.get("__bad__", False))


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
                print(prefix, suffix,
                      current_sample.keys()
                      if isinstance(current_sample, dict) else None)
            if prefix is None:
                continue
            if lcase:
                suffix = suffix.lower()
            if current_sample is None or prefix != current_sample["__key__"]:
                if valid_sample(current_sample):
                    yield current_sample
                current_sample = dict(__key__=prefix)
            if suffix in current_sample:
                raise ValueError(f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}")
            if suffixes is None or suffix in suffixes:
                current_sample[suffix] = value
        if valid_sample(current_sample):
            yield current_sample
    return iterator


def make_unique(keys):
    """Given a list of keys, ensures that they are all unique"."""
    result = []
    for i, k in enumerate(keys):
        if k is None or k == "" or k in result:
            result.append(f"_{i}")
        else:
            result.append(k)
    return result


def maybe_decode(s, mode="ascii"):
    if isinstance(s, str):
        return s
    elif isinstance(s, bytes):
        return s.decode(mode)
    else:
        raise ValueError(f"{type(s)}: wrong type for maybe_decode")


def extract_container(container):
    """Extracts an embedded container format from a tar file.

        container: suffix for container file format

    """
    def iterator(data):
        for fname, value in data:
            if fname.endswith("." + container):
                if container.endswith("mp"):
                    import msgpack
                    sample = msgpack.unpackb(value)
                    sample = {maybe_decode(k, "ascii"): v for k, v in sample.items()}
                elif container.endswith("json"):
                    sample = simplejson.loads(value)
                elif container.endswith("pyd"):
                    sample = pickle.loads(value)  # skipcq: BAN-B301
                elif container.endswith("ten"):
                    from . import tenbin
                    sample = tenbin.decode_buffer(value, infos=False)
                if isinstance(sample, dict):
                    sample["__key__"] = fname
                if isinstance(sample, list) or valid_sample(sample):
                    yield sample
    return iterator


def tardata(fileobj, skip_meta=r"__[^/]*__($|/)"):
    """Iterator yielding filename, content pairs for the given tar stream.

    :param fileobj: byte stream suitable for tarfile
    :param skip_meta: regexp for keys that are skipped entirely (Default value = r"__[^/]*__($|/)")

    """
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    for tarinfo in stream:
        if not tarinfo.isreg():
            continue
        fname = tarinfo.name
        if fname is None:
            continue
        if "/" not in fname and fname.startswith(meta_prefix) and fname.endswith(meta_suffix):
            # skipping metadata for now
            continue
        if skip_meta is not None and re.match(skip_meta, fname):
            continue
        data = stream.extractfile(tarinfo).read()
        yield fname, data
    del stream


def make_decoder(spec):
    if spec is True:
        spec = "rgb"
    if spec is False or spec is None:
        def decoder(x): return x
    elif callable(spec):
        decoder = spec
    elif isinstance(spec, dict):
        def decoder(sample): return decode_sample_based_on_extensions(
            sample, spec)
    elif isinstance(spec, str):
        handlers = default_handlers.get(spec)
        checknotnone(handlers, spec)
        def decoder(sample): return decode_sample_based_on_extensions(
            sample, handlers)
    else:
        raise ValueError(f"{spec}: unknown decoder spec")
    checkcallable(decoder, "could not make a callable decoder")
    return decoder


def apply_decoder(decoder, errors=True):
    """Decode samples by invoking the decoder with error handling.

        decode: decoder function
        errors: True, "warn", or False

    """
    def iterator(data):
        for sample in data:
            try:
                decoded = decoder(sample)
            except Exception as exn:  # skipcq: PYL-W0703
                if errors == "warn":
                    warnings.warn("apply_decoder " + repr(exn))
                    time.sleep(0.5)
                elif errors:
                    raise exn
                else:
                    continue
            yield decoded

    return iterator


def tariterator(fileobj, keys=base_plus_ext, decoder=True, suffixes=None,
                errors=True, container=None):
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
    decoder = make_decoder(decoder)
    content = tardata(fileobj)
    if container is not None:
        samples = extract_container(container)(content)
    else:
        samples = group_by_keys(keys=keys, suffixes=suffixes)(content)
    if container != "ten":
        samples = apply_decoder(decoder=decoder, errors=errors)(samples)
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
    :param container: if given, treats the tar file as a record file of containers (protobufs, msgpack, etc.)
    :param extra_meta: associates subset info with each sample record

    The decoder can be True (default decoder), False (no decoder), a callable (called
    decode the sample, or a dictionary mapping filename extensions to callables for
    the decoding.
    """

    def __init__(self, urls, *, extensions=None, decoder="rgb",
                 transforms=None, pipeline=None,
                 keys=base_plus_ext, opener=io.reader,
                 errors=True, verbose=False, shuffle=0, associate=None,
                 prepare_for_worker=True, container=None):
        self.opener = opener if callable(opener) else io.command_pipe(opener)
        checkcallable(self.opener)
        self.decoder = decoder
        self.transforms = listify(transforms)
        self.verbose = verbose
        self.keys = keys
        self.container = container
        self.errors = errors
        self.associate = associate
        self.pipeline = pipeline
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
            warnings.warn(
                f"num_workers {total} > num_shards {len(self.full_urls)}")
        self.urls = self.full_urls[index::total]

    def __iter__(self):
        """Iterate over samples."""
        self.prepare_for_worker()
        if self.shuffle > 0:
            random.shuffle(self.urls)
        self.sample = 0
        urls = self.urls
        for url in urls:
            stream = None
            try:
                with self.opener(url) as stream:
                    source = tariterator(stream,
                                         keys=self.keys,
                                         suffixes=self.suffixes,
                                         decoder=self.decoder,
                                         container=self.container,
                                         errors=self.errors)
                    if self.container == "ten":
                        for sample in source:
                            checktype(sample, list)
                            yield tuple(sample)
                        continue
                    if self.associate is not None:
                        source = associate(self.associate)(source)
                    if self.extensions is not None:
                        source = extract(*self.extensions)(source)
                    if self.shuffle > 1:
                        source = shuffle(self.shuffle)(source)
                    if self.transforms is not None:
                        source = transform(
                            transformer(self.transforms))(source)
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
                if self.errors == "warn":
                    warnings.warn("dataset __iter__ " + url + " " + repr(exn))
                    time.sleep(0.5)
                elif self.errors:
                    raise exn

#!/usr/bin/python
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of webloader (see TBD).
# See the LICENSE file for licensing terms (BSD-style).
#

from __future__ import absolute_import, division, print_function

import argparse
import collections
import io
import logging
import os
import re
import sys
import time
from builtins import range
from functools import wraps
import braceexpand
from subprocess import PIPE, Popen, check_call
import warnings
import tarfile
import simplejson
import pickle
import random

import numpy as np
import PIL
import PIL.Image
import six
import urllib.parse

from future import standard_library
standard_library.install_aliases()

try:
    from torch.utils.data import IterableDataset
except:
    class IterableDataset(object):
        pass

debug_dataset = os.environ.get("WDS_DEBUG", 0)
popen_bufsize = int(os.environ.get("WDS_BUFSIZE", "2000000"))

meta_prefix = "__"
meta_suffix = "__"

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


if sys.version_info[0] == 3:
    from builtins import str
    unicode = str
    buffer = str

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
    assert imagespec in imagespecs, imagespecs.keys()
    atype, etype, mode = imagespecs[imagespec.lower()]
    with six.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        img = img.convert(mode.upper())
    if atype == "pil":
        return img
    elif atype == "numpy":
        import numpy as np
        result = np.asarray(img)
        assert result.dtype == np.uint8, (image, result.dtype)
        if etype == "uint8":
            return result
        else:
            return result.astype("f") / 255.0
    elif atype == "torch":
        import torch
        from torchvision import transforms
        result = transforms.ToTensor()(img)
        if etype == "uint8":
            return result.type(torch.uint8)
        else:
            return result.type(torch.float) / 255.0

def maybe_int(data):
    try:
        return int(data)
    except ValueError:
        return data

def make_handlers(imagetype):
    handlers = {}
    for extension in ["cls", "cls2", "class", "count", "index", "inx", "id"]:
        handlers[extension] = maybe_int
    for extension in ["txt", "text", "transcript"]:
        handlers[extension] = lambda x: x.decode("utf-8")
    for extension in ["png", "jpg", "jpeg", "img", "image", "pbm", "pgm", "ppm"]:
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

default_handlers = { key: make_handlers(key) for key in imagespecs.keys() }

def decode_based_on_extension1(data, tname, handlers):
    # Unicode change. If it is alread an unicode string, no decoding (Byte->Unicode req)
    if isinstance(data, (int, float, unicode)):
        return data
    assert isinstance(data, bytes), type(data)
    assert isinstance(tname, str), tname
    extension = re.sub(r".*\.", "", tname).lower()
    decoder = handlers.get(extension)
    if decoder is None:
        return data
    else:
        return decoder(data)

def decode_based_on_extension(sample, handlers):
    """Autodecode a sample, using extensions as guide for how to decode.

    :param sample: dictionary representing sample
    :param imagetype: format for images (gray, rgb, rgba, PIL; rgb8=8 bit)
    """
    result = {}
    for k, v in list(sample.items()):
        if k[0] == "_":
            if isinstance(v, bytes):
                v = v.decode('utf-8')
            result[k] = v
            continue
        assert v is not None, (k, sample)
        result[k] = decode_based_on_extension1(v, k, handlers=handlers)
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

    :param sample: list of values
    :param transformers: list of functions

    """
    assert not isinstance(sample, dict)
    assert isinstance(sample, (tuple, list))
    if transformers is None or len(transformers) == 0:
        return sample
    result = list(sample)
    ntransformers = len(transformers)
    for i in range(len(sample)):
        f = transformers[i % ntransformers]
        if f is not None:
            result[i] = f(sample[i])
    return result


def transformer(transformers):
    """Curried version of `transform_with`.

    :param transformers: 

    """
    def f(x): return transform_with(x, transformers)
    return f


def listify(x):
    """Turn a value into a list.

    Lists and tuples are turned into lists, everything else is turned
    into a one element list.

    :param x: value to be converted
    :param x): return transform_with(x: 
    :param transformers)return flistify(x: 

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

    :param data: iterator
    :param f: function from samples to samples
    :returns: iterator over transformed samples

    """

    if f is None:
        def f(x): return x
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

    :param data: iterator
    :param bufsize: buffer size for shuffling
    :returns: iterator

    """
    import random
    assert initial <= bufsize
    buf = []
    startup = True
    for sample in data:
        if len(buf) < bufsize:
            buf.append(next(data))
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

    :param path: path with extensions
    :returns: path with all extensions removed

    """
    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def valid_sample(sample):
    """Check whether a sample is valid.

    :param sample: sample to be checked

    """
    return (sample is not None and
            isinstance(sample, dict) and
            len(list(sample.keys())) > 0 and
            not sample.get("__bad__", False))


def group_by_keys(keys=base_plus_ext, lcase=True, suffixes=None):
    """Returns function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (Default value = base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)

    """
    def iterator(data):
        current_sample = None
        for fname, value in data:
            prefix, suffix = keys(fname)
            if prefix is None:
                continue
            if current_sample is not None and prefix == current_sample["__key__"]:
                current_sample[suffix] = value
                continue
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix)
            if lcase:
                suffix = suffix.lower()
            if suffixes is None or suffix in suffixes:
                current_sample[suffix] = value
        if valid_sample(current_sample):
            yield current_sample
    return iterator


def make_unique(keys):
    """Given a list of keys, ensures that they are all unique"."""
    result = []
    for i, k in enumerate(keys):
        if k is None or k=="" or k in result:
            result.append(f"_{i}")
        else:
            result.append(k)
    return result


def extract_container(container):
    """Extracts an embedded container format from a tar file.

    :param container: suffix for container file format

    """
    def iterator(data):
        for fname, value in data:
            if fname.endswith("."+container):
                if container.endswith("mp"):
                    import msgpack
                    sample = msgpack.unpackb(value)
                    sample = {k.decode("ascii"): v for k, v in sample.items()}
                elif container.endswith("json"):
                    import simplejson
                    sample = simplejson.loads(value)
                elif container.endswith("pyd"):
                    sample = pickle.loads(value)
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
        decoder = lambda x: x
    elif callable(spec):
        decoder = spec
    elif isinstance(spec, dict):
        decoder = lambda sample: decode_based_on_extension(sample, spec)
    elif isinstance(spec, str):
        handlers = default_handlers.get(spec)
        assert handlers is not None, spec
        decoder = lambda sample: decode_based_on_extension(sample, handlers)
    else:
        raise ValueError(f"{spec}: unknown decoder spec")
    assert callable(decoder), (spec, decoder)
    return decoder

def apply_decoder(decoder, errors=True):
    """Decode samples by invoking the decoder with error handling.

    :param decode: decoder function
    :param errors: True, "warn", or False

    """
    def iterator(data):
        for sample in data:
            try:
                decoded = decoder(sample)
            except NoException as exn:
                if errors=="warn":
                    warnings.warn(repr(exn))
                    time.sleep(0.5)
                elif errors:
                    raise exn
                else:
                    continue
            yield decoded

    return iterator


def tariterator1(fileobj, keys=base_plus_ext, decoder=True, suffixes=None, errors=True, container=None):
    """Iterate through training samples stored in a sharded tar file.

    :param fileobj:
    :param check_sorted:  (Default value = False)
    :param keys:  (Default value = base_plus_ext)
    :param decode:  (Default value = True)

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


class Pipe(object):
    def __init__(self, *args, raise_errors=True, **kw):
        self.open(*args, **kw)
        self.raise_errors = raise_errors
    def open(self, *args, **kw):
        self.proc = Popen(*args, **kw)
        self.args = (args, kw)
        self.stream = self.proc.stdout
        assert self.stream is not None
        self.status = None
        return self
    def read(self, *args, **kw):
        result = self.stream.read(*args, **kw)
        self.status = self.proc.poll()
        if self.status is not None:
            self.status = self.proc.wait()
            if self.status != 0 and self.raise_errors:
                raise Exception(f"{self.args}: exit {self.status} (read)")
        return result
    def readLine(self, *args, **kw):
        result = self.stream.readLine(*args, **kw)
        self.status = self.proc.poll()
        if self.status is not None:
            self.status = self.proc.wait()
            if self.status != 0 and self.raise_errors:
                raise Exception(f"{self.args}: exit {self.status} (readLine)")
    def close(self):
        self.stream.close()
        self.status = self.proc.wait()
        if self.raise_errors == "all":
            if self.status != 0 and self.raise_errors:
                raise Exception(f"{self.args}: exit {self.status} (close)")
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        self.close()


def command_pipe(cmd, bufsize=popen_bufsize):
    def f(url):
        stream = Pipe(cmd.format(url), stdout=PIPE, shell=True, bufsize=bufsize)
        return stream
    return f


def generic_opener(url):
    if url.startswith("gs://"):
        cmd = "gsutil cat -q '{}'"
    elif url.startswith("http://") or url.startswith("https://"):
        cmd = "curl --fail -s '{}' --output -"
    else:
        cmd = "cat '{}'"
    return command_pipe(cmd)(url)

def base_of_url(url):
    components = urllib.parse.urlparse(url)
    path = components.path
    base = re.sub(r"[.].*$", "", re.sub(r".*/", "", path))
    return base

def shard_of_url(url):
    components = urllib.parse.urlparse(url)
    path = components.path
    base =  re.sub(r".*/", "", path)
    match = re.search(r"[0-9][0-9-]*[0-9]", base)
    if match:
        return match.group(0)
    else:
        return None

def size_loader(indexurl, opener):
    """Return a length function based on a JSON file.

    The JSON file should contain a dictionary that maps
    either full shard paths or basenames to the number of
    samples in that shard.

    The number of samples should be given either by:

        json[url]
        json[basename]
        json[url]["num_samples"]
        json[basename]["num_samples"]
    """
    with opener(indexurl) as stream:
        lengths = simplejson.loads(stream.read())
    assert hasattr(lengths, "__getitem__")
    def f(url):
        attempts = []
        for extractor in (lambda x: x, base_of_url, shard_of_url):
            key = extractor(url)
            if key is not None:
                attempts.append(key)
                result = lengths.get(key)
                if result is not None:
                    if not isinstance(result, int):
                        result = result["num_samples"]
                    return result
        raise ValueError(f"{attempts}: no num_samples in {indexurl}")
    return f


class WebDataset(IterableDataset):
    """Iterate over sharded datasets."""

    def __init__(self, urls, sizefun=None, extensions=None, decoder="rgb", 
                 transforms=None, pipeline=None,
                 epochs=1, keys=base_plus_ext, opener=generic_opener,
                 errors=True, verbose=False, shuffle=0, associate=None,
                 prepare_for_worker=True, container=None, extra_meta=False):
        """Create a WebLoader

        :param urls: shard spec or list of shards
        :param extensions: extensions to extract (Default value = None, can be either list of lists or "a;b c")
        :param decode: decoder to apply to tarfiles (Default value = True)
        :param transforms: list of functions to apply to unbatched samples (Default value = None)
        :param pipeline: function that maps the iterator, e.g. for batching
        :param opener: either a function that returns a stream or a string that is invoked via Popen
        """

        self.opener = opener if callable(opener) else command_pipe(opener)
        assert callable(self.opener), opener
        self.decoder = decoder
        self.transforms = listify(transforms)
        self.verbose = verbose
        self.keys = keys
        self.container = container
        self.errors = errors
        self.associate = associate
        self.pipeline = pipeline
        if callable(sizefun):
            self.sizefun = sizefun
        else:
            self.sizefun = lambda _:sizefun
        if isinstance(urls, str):
            urls = list(braceexpand.braceexpand(urls))
        #urls = list(urls)
        assert isinstance(urls, list)
        self.full_urls = urls
        self.urls = urls
        self.shuffle = shuffle
        if extensions is not None:
            if isinstance(extensions, str):
                extensions = [f.split(";") for f in extensions.split()]
            for f in extensions:
                assert isinstance(f, list), (extensions, f)
            self.extensions = extensions
            self.suffixes = {x for l in extensions for x in l}
        else:
            self.extensions = None
            self.suffixes = None
        if prepare_for_worker is True:
            self.prepare_for_worker = self.shard_selection
        elif prepare_for_worker is False:
            self.prepare_for_worker = lambda:None
        else:
            self.prepare_for_worker = prepare_for_worker
        self.subset = None
        self.extra_meta = False

    def shard_selection(self):
        import torch
        index, total = None, None
        if self.subset is not None:
            index, total = self.subset
        else:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                index = worker_info.id
                total = worker_info.num_workers
        if total is None:
            self.urls = self.full_urls
            return
        if index==0 and len(self.full_urls)<total:
            warnings.warn(f"num_workers {total} > num_shards {len(self.full_urls)}")
        self.urls = self.full_urls[index::total]

    def __iter__(self):
        """Iterate over samples."""
        self.prepare_for_worker()
        if self.shuffle > 0:
            random.shuffle(self.urls)
        finished = False
        self.sample = 0
        urls = self.urls
        for url in urls:
            stream = None
            try:
                with self.opener(url) as stream:
                    source = tariterator1(stream,
                                          keys=self.keys,
                                          suffixes=self.suffixes,
                                          decoder=self.decoder,
                                          container=self.container,
                                          errors=self.errors)
                    if self.container=="ten":
                        for sample in source:
                            assert isinstance(sample, list)
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
            except NoException as exn:
                if self.errors=="warn":
                    warnings.warn(repr(exn))
                    time.sleep(0.5)
                elif self.errors:
                    raise exn
    def size(self):
        """Return the length specified at initialization."""
        return sum(self.sizefun(url) for url in self.urls)

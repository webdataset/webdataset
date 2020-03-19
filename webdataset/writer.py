#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

__all__ = "TarWriter ShardWriter".split()

import io
import pickle
import re
import tarfile
import time

import numpy as np
import PIL
import simplejson

from . import gopen


def imageencoder(image, format="PNG"):  # skipcq: PYL-W0622
    """Compress an image using PIL and return it as a string.

    Can handle float or uint8 images.

    :param image: ndarray representing an image
    :param format: compression format (PNG, JPEG, PPM)

    """
    if isinstance(image, np.ndarray):
        if image.dtype in [np.dtype("f"), np.dtype("d")]:
            if not (np.amin(image) > -0.001 and np.amax(image) < 1.001):
                raise ValueError(
                    f"image values out of range {np.amin(image)} {np.amax(image)}"
                )
            image = np.clip(image, 0.0, 1.0)
            image = np.array(image * 255.0, "uint8")
        image = PIL.Image.fromarray(image)
    if format.upper() == "JPG":
        format = "JPEG"
    elif format.upper() in ["IMG", "IMAGE"]:
        format = "PPM"
    if format == "JPEG":
        opts = dict(quality=100)
    else:
        opts = {}
    with io.BytesIO() as result:
        image.save(result, format=format, **opts)
        return result.getvalue()


def bytestr(data):
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("ascii")
    return str(data).encode("ascii")


def make_handlers():
    handlers = {}
    for extension in ["cls", "cls2", "class", "count", "index", "inx", "id"]:
        handlers[extension] = lambda x: str(x).encode("ascii")
    for extension in ["txt", "text", "transcript"]:
        handlers[extension] = lambda x: x.encode("utf-8")
    for extension in ["png", "jpg", "jpeg", "img", "image", "pbm", "pgm", "ppm"]:

        def f(extension_):
            handlers[extension] = lambda data: imageencoder(data, extension_)

        f(extension)
    for extension in ["pyd", "pickle"]:
        handlers[extension] = pickle.dumps
    for extension in ["json", "jsn"]:
        handlers[extension] = lambda x: simplejson.dumps(x).encode("utf-8")
    for extension in ["ten", "tb"]:
        from . import tenbin

        def f(x):  # skipcq: PYL-E0102
            if isinstance(x, list):
                return tenbin.encode_buffer(x)
            else:
                return tenbin.encode_buffer([x])

        handlers[extension] = f
    try:
        import msgpack

        for extension in ["mp", "msgpack", "msg"]:
            handlers[extension] = msgpack.packb
    except ImportError:
        pass
    return handlers


default_handlers = {"default": make_handlers()}


def encode_based_on_extension1(data, tname, handlers):
    if tname[0] == "_":
        if not isinstance(data, str):
            raise ValueError("the values of metadata must be of string type")
        return data
    extension = re.sub(r".*\.", "", tname).lower()
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    handler = handlers.get(extension)
    if handler is None:
        raise ValueError(f"no handler found for {extension}")
    return handler(data)


def encode_based_on_extension(sample, handlers):
    return {
        k: encode_based_on_extension1(v, k, handlers) for k, v in list(sample.items())
    }


def make_encoder(spec):
    if spec is False or spec is None:

        def encoder(x):
            return x

    elif callable(spec):
        encoder = spec
    elif isinstance(spec, dict):

        def encoder(sample):
            return encode_based_on_extension(sample, spec)

    elif isinstance(spec, str) or spec is True:
        if spec is True:
            spec = "default"
        handlers = default_handlers.get(spec)
        if handlers is None:
            raise ValueError(f"no handler found for {spec}")

        def encoder(sample):
            return encode_based_on_extension(sample, handlers)

    else:
        raise ValueError(f"{spec}: unknown decoder spec")
    if not callable(encoder):
        raise ValueError(f"{spec} did not yield a callable encoder")
    return encoder


class TarWriter:
    """A class for writing dictionaries to tar files.

    :param fileobj: fileobj: file name for tar file (.tgz/.tar) or open file descriptor
    :param encoder: sample encoding (Default value = True)
    :param compress:  (Default value = None)

    `True` will use an encoder that behaves similar to the automatic
    decoder for `Dataset`. `False` disables encoding and expects byte strings
    (except for metadata, which must be strings). The `encoder` argument can
    also be a `callable`, or a dictionary mapping extensions to encoders.

    The following code will add two file to the tar archive: `a/b.png` and
    `a/b.output.png`.

    ```Python
        tarwriter = TarWriter(stream)
        image = imread("b.jpg")
        image2 = imread("b.out.jpg")
        sample = {"__key__": "a/b", "png": image, "output.png": image2}
        tarwriter.write(sample)
    ```
    """

    def __init__(
        self,
        fileobj,
        user="bigdata",
        group="bigdata",
        mode=0o0444,
        compress=None,
        encoder=True,
        keep_meta=False,
    ):
        if isinstance(fileobj, str):
            if compress is False:
                tarmode = "w|"
            elif compress is True:
                tarmode = "w|gz"
            else:
                tarmode = "w|gz" if fileobj.endswith("gz") else "w|"
            fileobj = gopen.gopen(fileobj, "wb")
            self.own_fileobj = fileobj
        else:
            tarmode = "w|gz" if compress is True else "w|"
            self.own_fileobj = None
        self.encoder = make_encoder(encoder)
        self.keep_meta = keep_meta
        self.stream = fileobj
        self.tarstream = tarfile.open(fileobj=fileobj, mode=tarmode)

        self.user = user
        self.group = group
        self.mode = mode
        self.compress = compress

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the tar file."""
        self.tarstream.close()
        if self.own_fileobj is not None:
            self.own_fileobj.close()
            self.own_fileobj = None

    def dwrite(self, key, **kw):
        """Convenience function for `write`.

        Takes key as the first argument and key-value pairs for the rest.
        Replaces "_" with ".".
        """
        obj = dict(__key__=key)
        obj.update({k.replace("_", "."): v for k, v in kw.items()})
        self.write(obj)

    def write(self, obj):
        """Write a dictionary to the tar file.

        :param obj: dictionary of objects to be stored
        :returns: size of the entry

        """
        total = 0
        obj = self.encoder(obj)
        if "__key__" not in obj:
            raise ValueError(f"object must contain a __key__")
        for k, v in list(obj.items()):
            if k[0] == "_":
                continue
            if not isinstance(v, bytes):
                raise ValueError(
                    f"{k} doesn't map to a bytes after encoding ({type(v)})"
                )
        key = obj["__key__"]
        for k in sorted(obj.keys()):
            if k == "__key__":
                continue
            if not self.keep_meta and k[0] == "_":
                continue
            v = obj[k]
            if isinstance(v, str):
                v = v.encode("utf-8")
            now = time.time()
            ti = tarfile.TarInfo(key + "." + k)
            ti.size = len(v)
            ti.mtime = now
            ti.mode = self.mode
            ti.uname = self.user
            ti.gname = self.group
            if not isinstance(v, (bytes)):
                raise ValueError(f"converter didn't yield bytes: {k}, {type(v)}")
            stream = io.BytesIO(v)
            self.tarstream.addfile(ti, stream)
            total += ti.size
        return total


class ShardWriter:
    """Like TarWriter but splits into multiple shards.

    :param pattern: output file pattern
    :param maxcount: maximum number of records per shard (Default value = 100000)
    :param maxsize: maximum size of each shard (Default value = 3e9)
    :param kw: other options passed to TarWriter

    """

    def __init__(self, pattern, maxcount=100000, maxsize=3e9, post=None, **kw):
        self.verbose = 1
        self.kw = kw
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.post = post

        self.tarstream = None
        self.shard = 0
        self.pattern = pattern
        self.total = 0
        self.count = 0
        self.size = 0
        self.fname = None
        self.next_stream()

    def next_stream(self):
        self.finish()
        self.fname = self.pattern % self.shard
        if self.verbose:
            print(
                "# writing",
                self.fname,
                self.count,
                "%.1f GB" % (self.size / 1e9),
                self.total,
            )
        self.shard += 1
        stream = open(self.fname, "wb")
        self.tarstream = TarWriter(stream, **self.kw)
        self.count = 0
        self.size = 0

    def write(self, obj):
        if (
            self.tarstream is None
            or self.count >= self.maxcount
            or self.size >= self.maxsize
        ):
            self.next_stream()
        size = self.tarstream.write(obj)
        self.count += 1
        self.total += 1
        self.size += size

    def finish(self):
        if self.tarstream is not None:
            self.tarstream.close()
            assert self.fname is not None
            if callable(self.post):
                self.post(self.fname)
            self.tarstream = None

    def close(self):
        self.finish()
        del self.tarstream
        del self.shard
        del self.count
        del self.size

    def __enter__(self):
        return self

    def __exit__(self, *args, **kw):
        self.close()

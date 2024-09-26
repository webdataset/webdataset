#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""Classes and functions for writing tar files and WebDataset files."""

import gzip
import io
import json
import pickle
import re
import tarfile
import time
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from . import gopen


def imageencoder(image: Any, format: str = "PNG"):  # skipcq: PYL-W0622
    """Compress an image using PIL and return it as a string.

    Can handle float or uint8 images.

    Args:
        image: ndarray representing an image
        format: compression format (PNG, JPEG, PPM)

    Returns:
        bytes: Compressed image data

    Raises:
        ValueError: If image values are out of range
    """
    import PIL
    import PIL.Image

    assert isinstance(image, (PIL.Image.Image, np.ndarray)), type(image)

    if isinstance(image, np.ndarray):
        if image.dtype in [np.dtype("f"), np.dtype("d")]:
            if not (np.amin(image) > -0.001 and np.amax(image) < 1.001):
                raise ValueError(
                    f"image values out of range {np.amin(image)} {np.amax(image)}"
                )
            image = np.clip(image, 0.0, 1.0)
            image = np.array(image * 255.0, "uint8")
        assert image.ndim in [2, 3]
        if image.ndim == 3:
            assert image.shape[2] in [1, 3]
        image = PIL.Image.fromarray(image)
    if format.upper() == "JPG":
        format = "JPEG"
    elif format.upper() in {"IMG", "IMAGE"}:
        format = "PPM"
    if format in {"JPEG", "tiff"}:
        opts = dict(quality=100)
    else:
        opts = {}
    with io.BytesIO() as result:
        image.save(result, format=format, **opts)
        return result.getvalue()


def bytestr(data: Any):
    """Convert data into a bytestring.

    Uses str and ASCII encoding for data that isn't already in string format.

    Args:
        data: Data to be converted

    Returns:
        bytes: Converted bytestring
    """
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("ascii")
    return str(data).encode("ascii")


def torch_dumps(data: Any):
    """Dump data into a bytestring using torch.dumps.

    This delays importing torch until needed.

    Args:
        data: Data to be dumped

    Returns:
        bytes: Dumped data as bytestring
    """
    import io

    import torch

    stream = io.BytesIO()
    torch.save(data, stream)
    return stream.getvalue()


def numpy_dumps(data: np.ndarray):
    """Dump data into a bytestring using numpy npy format.

    Args:
        data: Data to be dumped

    Returns:
        bytes: Dumped data as bytestring
    """
    import io

    import numpy.lib.format

    stream = io.BytesIO()
    numpy.lib.format.write_array(stream, data)
    return stream.getvalue()


def numpy_npz_dumps(data: Dict[str, np.ndarray]):
    """Dump data into a bytestring using numpy npz format.

    Args:
        data: Dictionary of numpy arrays to be dumped

    Returns:
        bytes: Dumped data as bytestring

    Raises:
        AssertionError: If input is not a dictionary of numpy arrays
    """
    import io

    assert isinstance(data, dict)
    for k, v in list(data.items()):
        assert isinstance(k, str)
        assert isinstance(v, np.ndarray)

    stream = io.BytesIO()
    np.savez_compressed(stream, **data)
    return stream.getvalue()


def tenbin_dumps(x):
    """Dump data into a bytestring using tenbin format.

    Args:
        x: Data to be dumped (list or single item)

    Returns:
        memoryview: Dumped data as memoryview
    """
    from . import tenbin

    if isinstance(x, list):
        return memoryview(tenbin.encode_buffer(x))
    else:
        return memoryview(tenbin.encode_buffer([x]))


def cbor_dumps(x):
    """Dump data into a bytestring using CBOR format.

    Args:
        x: Data to be dumped

    Returns:
        bytes: Dumped data as bytestring
    """
    import cbor

    return cbor.dumps(x)


def mp_dumps(x):
    """Dump data into a bytestring using MessagePack format.

    Args:
        x: Data to be dumped

    Returns:
        bytes: Dumped data as bytestring
    """
    import msgpack

    return msgpack.packb(x)


def add_handlers(d, keys, value):
    """Add handlers to a dictionary for given keys.

    Args:
        d: Dictionary to add handlers to
        keys: String of space-separated keys or list of keys
        value: Handler function to be added
    """
    if isinstance(keys, str):
        keys = keys.split()
    for k in keys:
        d[k] = value


def make_handlers():
    """Create a list of handlers for encoding data.

    Returns:
        dict: Dictionary of handlers for different data types
    """
    handlers = {}
    add_handlers(
        handlers, "cls cls2 class count index inx id", lambda x: str(x).encode("ascii")
    )
    add_handlers(handlers, "txt text transcript", lambda x: x.encode("utf-8"))
    add_handlers(handlers, "html htm", lambda x: x.encode("utf-8"))
    add_handlers(handlers, "pyd pickle", pickle.dumps)
    add_handlers(handlers, "pth", torch_dumps)
    add_handlers(handlers, "npy", numpy_dumps)
    add_handlers(handlers, "npz", numpy_npz_dumps)
    add_handlers(handlers, "ten tenbin tb", tenbin_dumps)
    add_handlers(handlers, "json jsn", lambda x: json.dumps(x).encode("utf-8"))
    add_handlers(handlers, "mp msgpack msg", mp_dumps)
    add_handlers(handlers, "cbor", cbor_dumps)
    add_handlers(handlers, "jpg jpeg img image", lambda data: imageencoder(data, "jpg"))
    add_handlers(handlers, "png", lambda data: imageencoder(data, "png"))
    add_handlers(handlers, "pbm", lambda data: imageencoder(data, "pbm"))
    add_handlers(handlers, "pgm", lambda data: imageencoder(data, "pgm"))
    add_handlers(handlers, "ppm", lambda data: imageencoder(data, "ppm"))
    add_handlers(handlers, "tiff tif", lambda data: imageencoder(data, "tiff"))
    return handlers


default_handlers = make_handlers()


def encode_based_on_extension1(data: Any, tname: str, handlers: dict):
    """Encode data based on its extension and a dict of handlers.

    Args:
        data: Data to be encoded
        tname: File extension
        handlers: Dictionary of handlers for different data types

    Raises:
        ValueError: If no handler is found for the given extension or if metadata values are not strings
    """
    if tname[0] == "_":
        if not isinstance(data, str):
            raise ValueError("the values of metadata must be of string type")
        return data
    compress = False
    if tname.endswith(".gz"):
        compress = True
        tname = tname[:-3]
    extension = re.sub(r".*\.", "", tname).lower()
    if isinstance(data, bytes):
        if compress:
            data = gzip.compress(data)
        return data
    if isinstance(data, str):
        data = data.encode("utf-8")
        if compress:
            data = gzip.compress(data)
        return data
    handler = handlers.get(extension)
    if handler is None:
        raise ValueError(f"no handler found for {extension}")
    result = handler(data)
    if compress:
        result = gzip.compress(result)
    return result


def encode_based_on_extension(sample: dict, handlers: dict):
    """Encode an entire sample with a collection of handlers.

    Args:
        sample: Data sample (a dict)
        handlers: Handlers for encoding

    Returns:
        dict: Encoded sample
    """
    return {
        k: encode_based_on_extension1(v, k, handlers) for k, v in list(sample.items())
    }


def make_encoder(spec: Union[bool, str, dict, Callable]):
    """Make an encoder function from a specification.

    Args:
        spec: Specification for the encoder

    Returns:
        Callable: Encoder function

    Raises:
        ValueError: If the specification is invalid or doesn't yield a callable encoder
    """
    if spec is False or spec is None:

        def encoder(x):
            """Do not encode at all."""
            return x

    elif callable(spec):
        encoder = spec
    elif isinstance(spec, dict):

        def f(sample):
            """Encode based on extension."""
            return encode_based_on_extension(sample, spec)

        encoder = f

    elif spec is True:
        handlers = default_handlers

        def g(sample):
            """Encode based on extension."""
            return encode_based_on_extension(sample, handlers)

        encoder = g

    else:
        raise ValueError(f"{spec}: unknown decoder spec")
    if not callable(encoder):
        raise ValueError(f"{spec} did not yield a callable encoder")
    return encoder


class TarWriter:
    """A class for writing dictionaries to tar files.

    Args:
        fileobj: File name for tar file (.tgz/.tar) or open file descriptor.
        encoder: Sample encoding. Defaults to True.
        compress: Compression flag. Defaults to None.
        user: User for tar files. Defaults to "bigdata".
        group: Group for tar files. Defaults to "bigdata".
        mode: Mode for tar files. Defaults to 0o0444.
        keep_meta: Flag to keep metadata (entries starting with "_"). Defaults to False.
        mtime: Modification time. Defaults to None.
        format: Tar format. Defaults to None.

    Returns:
        TarWriter object.

    Raises:
        ValueError: If the encoder doesn't yield bytes for a key.

    `True` will use an encoder that behaves similar to the automatic
    decoder for `Dataset`. `False` disables encoding and expects byte strings
    (except for metadata, which must be strings). The `encoder` argument can
    also be a `callable`, or a dictionary mapping extensions to encoders.

    The following code will add two file to the tar archive: `a/b.png` and
    `a/b.output.png`.


        tarwriter = TarWriter(stream)
        image = imread("b.jpg")
        image2 = imread("b.out.jpg")
        sample = {"__key__": "a/b", "png": image, "output.png": image2}
        tarwriter.write(sample)

    """

    def __init__(
        self,
        fileobj,
        user: str = "bigdata",
        group: str = "bigdata",
        mode: int = 0o0444,
        compress: Optional[Union[bool, str]] = None,
        encoder: Union[None, bool, Callable] = True,
        keep_meta: bool = False,
        mtime: Optional[float] = None,
        format: Any = None,
    ):  # sourcery skip: avoid-builtin-shadow
        """Create a tar writer.

        Args:
            fileobj: Stream to write data to.
            user: User for tar files.
            group: Group for tar files.
            mode: Mode for tar files.
            compress: Desired compression.
            encoder: Encoder function.
            keep_meta: Keep metadata (entries starting with "_").
            mtime: Modification time (set this to some fixed value to get reproducible tar files).
            format: Tar format.
        """
        format = getattr(tarfile, format, format) if format else tarfile.USTAR_FORMAT
        self.mtime = mtime
        tarmode = self.tarmode(fileobj, compress)
        if isinstance(fileobj, str):
            fileobj = gopen(fileobj, "wb")
            self.own_fileobj = fileobj
        else:
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
        """Enter context.

        Returns:
            self: The TarWriter object.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.close()

    def close(self):
        """Close the tar file."""
        self.tarstream.close()
        if self.own_fileobj is not None:
            self.own_fileobj.close()
            self.own_fileobj = None

    def write(self, obj):
        """Write a dictionary to the tar file.

        Args:
            obj: Dictionary of objects to be stored.

        Returns:
            int: Size of the entry.

        Raises:
            ValueError: If the object doesn't contain a __key__ or if a key doesn't map to bytes after encoding.
        """
        total = 0
        obj = self.encoder(obj)
        if "__key__" not in obj:
            raise ValueError("object must contain a __key__")
        for k, v in list(obj.items()):
            if k[0] == "_":
                continue
            if not isinstance(v, (bytes, bytearray, memoryview)):
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
            ti.mtime = self.mtime if self.mtime is not None else now
            ti.mode = self.mode
            ti.uname = self.user
            ti.gname = self.group
            if not isinstance(v, (bytes, bytearray, memoryview)):
                raise ValueError(f"converter didn't yield bytes: {k}, {type(v)}")
            stream = io.BytesIO(v)
            self.tarstream.addfile(ti, stream)
            total += ti.size

        return total

    @staticmethod
    def tarmode(fileobj, compress: Optional[Union[bool, str]] = None):
        if compress is False:
            return "w|"
        elif (
            compress is True
            or compress == "gz"
            or (isinstance(fileobj, str) and fileobj.endswith("gz"))
        ):
            return "w|gz"
        elif compress == "bz2" or (
            isinstance(fileobj, str) and fileobj.endswith("bz2")
        ):
            return "w|bz2"
        elif compress == "xz" or (isinstance(fileobj, str) and fileobj.endswith("xz")):
            return "w|xz"
        else:
            return "w|"


class ShardWriter:
    """Like TarWriter but splits into multiple shards.

    Args:
        pattern: Output file pattern.
        maxcount: Maximum number of records per shard. Defaults to 100000.
        maxsize: Maximum size of each shard. Defaults to 3e9.
        post: Optional callable to be executed after each shard is written. Defaults to None.
        start_shard: Starting shard number. Defaults to 0.
        verbose: Verbosity level. Defaults to 1.
        opener: Optional callable to open output files. Defaults to None.
        **kw: Other options passed to TarWriter.
    """

    def __init__(
        self,
        pattern: str,
        maxcount: int = 100000,
        maxsize: float = 3e9,
        post: Optional[Callable] = None,
        start_shard: int = 0,
        verbose: int = 1,
        opener: Optional[Callable] = None,
        **kw,
    ):
        """Create a ShardWriter.

        Args:
            pattern: Output file pattern.
            maxcount: Maximum number of records per shard.
            maxsize: Maximum size of each shard.
            post: Optional callable to be executed after each shard is written.
            start_shard: Starting shard number.
            verbose: Verbosity level.
            opener: Optional callable to open output files.
            **kw: Other options passed to TarWriter.
        """
        self.verbose = verbose
        self.kw = kw
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.post = post

        self.tarstream = None
        self.shard = start_shard
        self.pattern = pattern
        self.total = 0
        self.count = 0
        self.size = 0
        self.fname = None
        self.opener = opener
        self.next_stream()

    def next_stream(self):
        """Close the current stream and move to the next."""
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
        if self.opener:
            self.tarstream = TarWriter(self.opener(self.fname), **self.kw)
        else:
            self.tarstream = TarWriter(self.fname, **self.kw)
        self.count = 0
        self.size = 0

    def write(self, obj):
        """Write a sample.

        Args:
            obj: Sample to be written.
        """
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
        """Finish all writing (use close instead)."""
        if self.tarstream is not None:
            self.tarstream.close()
            assert self.fname is not None
            if callable(self.post):
                self.post(self.fname)
            self.tarstream = None

    def close(self):
        """Close the stream."""
        self.finish()
        del self.tarstream
        del self.shard
        del self.count
        del self.size

    def __enter__(self):
        """Enter context.

        Returns:
            self: The ShardWriter object.
        """
        return self

    def __exit__(self, *args, **kw):
        """Exit context."""
        self.close()

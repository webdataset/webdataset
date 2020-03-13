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

import pickle
import re

import numpy as np
import PIL
import PIL.Image
import simplejson
import six

from .checks import checkmember, checktype, checknotnone, checkcallable


class NoException(Exception):
    pass


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
    assert isinstance(sample, dict)
    for k, v in list(sample.items()):
        if k[0] == "_":
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            result[k] = v
            continue
        checknotnone(v)
        result[k] = decode_item_based_on_extension(v, k, handlers=handlers)
    return result


def make_decoder(spec):
    if spec is True:
        spec = "rgb"
    if spec is False or spec is None:

        def decoder(x):
            return x

    elif callable(spec):
        decoder = spec
    elif isinstance(spec, dict):

        def decoder(sample):
            return decode_sample_based_on_extensions(sample, spec)

    elif isinstance(spec, str):
        handlers = default_handlers.get(spec)
        checknotnone(handlers, spec)

        def decoder(sample):
            return decode_sample_based_on_extensions(sample, handlers)

    else:
        raise ValueError(f"{spec}: unknown decoder spec")
    checkcallable(decoder, "could not make a callable decoder")
    return decoder


def reraise_exception(exn):
    raise exn


def apply_decoder(decoder=make_decoder(True), handler=reraise_exception):
    """Decode samples by invoking the decoder with error handling.

        decode: decoder function
        errors: True, "warn", or False

    """

    def iterator(data):
        for sample in data:
            try:
                decoded = decoder(sample)
            except Exception as exn:  # skipcq: PYL-W0703
                if handler(exn):
                    continue
                else:
                    break
            yield decoded

    return iterator

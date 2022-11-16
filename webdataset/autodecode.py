#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""Automatically decode webdataset samples."""

import io, json, os, pickle, re, tempfile
from functools import partial

import numpy as np


# Obtained with:
# ```
# import PIL.Image
# IMAGE_EXTENSIONS = []
# PIL.Image.init()
# for ext, format in PIL.Image.EXTENSION.items():
#     if format in PIL.Image.OPEN:
#         IMAGE_EXTENSIONS.append(ext[1:])
# ```
"""Extensions passed on to the image decoder."""
IMAGE_EXTENSIONS = [
    "blp",
    "bmp",
    "dib",
    "bufr",
    "cur",
    "pcx",
    "dcx",
    "dds",
    "ps",
    "eps",
    "fit",
    "fits",
    "fli",
    "flc",
    "ftc",
    "ftu",
    "gbr",
    "gif",
    "grib",
    "h5",
    "hdf",
    "png",
    "apng",
    "jp2",
    "j2k",
    "jpc",
    "jpf",
    "jpx",
    "j2c",
    "icns",
    "ico",
    "im",
    "iim",
    "tif",
    "tiff",
    "jfif",
    "jpe",
    "jpg",
    "jpeg",
    "mpg",
    "mpeg",
    "msp",
    "pcd",
    "pxr",
    "pbm",
    "pgm",
    "ppm",
    "pnm",
    "psd",
    "bw",
    "rgb",
    "rgba",
    "sgi",
    "ras",
    "tga",
    "icb",
    "vda",
    "vst",
    "webp",
    "wmf",
    "emf",
    "xbm",
    "xpm",
]

################################################################
# handle basic datatypes
################################################################


def torch_loads(data):
    """Load data using torch.loads, importing torch only if needed.

    :param data: data to be decoded
    """
    import io

    import torch

    stream = io.BytesIO(data)
    return torch.load(stream)


def tenbin_loads(data):
    from . import tenbin

    return tenbin.decode_buffer(data)


def msgpack_loads(data):
    import msgpack

    return msgpack.unpackb(data)


def npy_loads(data):
    import numpy.lib.format

    stream = io.BytesIO(data)
    return numpy.lib.format.read_array(stream)


def cbor_loads(data):
    import cbor

    return cbor.loads(data)


decoders = {
    "txt": lambda data: data.decode("utf-8"),
    "text": lambda data: data.decode("utf-8"),
    "transcript": lambda data: data.decode("utf-8"),
    "cls": lambda data: int(data),
    "cls2": lambda data: int(data),
    "index": lambda data: int(data),
    "inx": lambda data: int(data),
    "id": lambda data: int(data),
    "json": lambda data: json.loads(data),
    "jsn": lambda data: json.loads(data),
    "pyd": lambda data: pickle.loads(data),
    "pickle": lambda data: pickle.loads(data),
    "pth": lambda data: torch_loads(data),
    "ten": tenbin_loads,
    "tb": tenbin_loads,
    "mp": msgpack_loads,
    "msg": msgpack_loads,
    "npy": npy_loads,
    "npz": lambda data: np.load(io.BytesIO(data)),
    "cbor": cbor_loads,
}


def basichandlers(key, data):
    """Handle basic file decoding.

    This function is usually part of the post= decoders.
    This handles the following forms of decoding:

    - txt -> unicode string
    - cls cls2 class count index inx id -> int
    - json jsn -> JSON decoding
    - pyd pickle -> pickle decoding
    - pth -> torch.loads
    - ten tenbin -> fast tensor loading
    - mp messagepack msg -> messagepack decoding
    - npy -> Python NPY decoding

    :param key: file name extension
    :param data: binary data to be decoded
    """
    extension = re.sub(r".*[.]", "", key)

    if extension in decoders:
        return decoders[extension](data)

    return None


################################################################
# Generic extension handler.
################################################################


def call_extension_handler(key, data, f, extensions):
    """Call the function f with the given data if the key matches the extensions.

    :param key: actual key found in the sample
    :param data: binary data
    :param f: decoder function
    :param extensions: list of matching extensions
    """
    extension = key.lower().split(".")
    for target in extensions:
        target = target.split(".")
        if len(target) > len(extension):
            continue
        if extension[-len(target) :] == target:
            return f(data)
    return None


def handle_extension(extensions, f):
    """Return a decoder function for the list of extensions.

    Extensions can be a space separated list of extensions.
    Extensions can contain dots, in which case the corresponding number
    of extension components must be present in the key given to f.
    Comparisons are case insensitive.

    Examples:
    handle_extension("jpg jpeg", my_decode_jpg)  # invoked for any file.jpg
    handle_extension("seg.jpg", special_case_jpg)  # invoked only for file.seg.jpg
    """
    extensions = extensions.lower().split()
    return partial(call_extension_handler, f=f, extensions=extensions)


################################################################
# handle images
################################################################

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


class ImageHandler:
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

    def __init__(self, imagespec, extensions=IMAGE_EXTENSIONS):
        """Create an image handler.

        :param imagespec: short string indicating the type of decoding
        :param extensions: list of extensions the image handler is invoked for
        """
        if imagespec not in list(imagespecs.keys()):
            raise ValueError("Unknown imagespec: %s" % imagespec)
        self.imagespec = imagespec.lower()
        self.extensions = extensions

    def __call__(self, key, data):
        """Perform image decoding.

        :param key: file name extension
        :param data: binary data
        """
        import PIL.Image

        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in self.extensions:
            return None
        imagespec = self.imagespec
        atype, etype, mode = imagespecs[imagespec]
        with io.BytesIO(data) as stream:
            img = PIL.Image.open(stream)
            img.load()
            img = img.convert(mode.upper())

        if atype == "pil":
            if mode == "l":
                img = img.convert("L")
                return img
            elif mode == "rgb":
                img = img.convert("RGB")
                return img
            elif mode == "rgba":
                img = img.convert("RGBA")
                return img
            else:
                raise ValueError("Unknown mode: %s" % mode)

        result = result = np.asarray(img)

        if etype == "float":
            result = result.astype(np.float32) / 255.0

        assert result.ndim in [2, 3], result.shape
        assert mode in ["l", "rgb", "rgba"], mode

        if mode == "l":
            if result.ndim == 3:
                result = np.mean(result[:, :, :3], axis=2)
        elif mode == "rgb":
            if result.ndim == 2:
                result = np.repeat(result[:, :, np.newaxis], 3, axis=2)
            elif result.shape[2] == 4:
                result = result[:, :, :3]
        elif mode == "rgba":
            if result.ndim == 2:
                result = np.repeat(result[:, :, np.newaxis], 4, axis=2)
                result[:, :, 3] = 255
            elif result.shape[2] == 3:
                result = np.concatenate([result, 255 * np.ones(result.shape[:2])], axis=2)

        assert atype in ["numpy", "torch"], atype

        if atype == "numpy":
            return result
        elif atype == "torch":
            import torch

            if result.ndim == 3:
                return torch.from_numpy(result.transpose(2, 0, 1))
            else:
                return torch.from_numpy(result)

        return None


def imagehandler(imagespec, extensions=IMAGE_EXTENSIONS):
    """Create an image handler.

    This is just a lower case alias for ImageHander.

    :param imagespec: textual image spec
    :param extensions: list of extensions the handler should be applied for
    """
    return ImageHandler(imagespec, extensions)


################################################################
# torch video
################################################################


def torch_video(key, data):
    """Decode video using the torchvideo library.

    :param key: file name extension
    :param data: data to be decoded
    """
    extension = re.sub(r".*[.]", "", key)
    if extension not in "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
        return None

    import torchvision.io

    with tempfile.TemporaryDirectory() as dirname:
        fname = os.path.join(dirname, f"file.{extension}")
        with open(fname, "wb") as stream:
            stream.write(data)
        return torchvision.io.read_video(fname, pts_unit="sec")


################################################################
# torchaudio
################################################################


def torch_audio(key, data):
    """Decode audio using the torchaudio library.

    :param key: file name extension
    :param data: data to be decoded
    """
    extension = re.sub(r".*[.]", "", key)
    if extension not in ["flac", "mp3", "sox", "wav", "m4a", "ogg", "wma"]:
        return None

    import torchaudio

    with tempfile.TemporaryDirectory() as dirname:
        fname = os.path.join(dirname, f"file.{extension}")
        with open(fname, "wb") as stream:
            stream.write(data)
        return torchaudio.load(fname)


################################################################
# special class for continuing decoding
################################################################


class Continue:
    """Special class for continuing decoding.

    This is mostly used for decompression, as in:

        def decompressor(key, data):
            if key.endswith(".gz"):
                return Continue(key[:-3], decompress(data))
            return None
    """

    def __init__(self, key, data):
        """__init__.

        :param key:
        :param data:
        """
        self.key, self.data = key, data


def gzfilter(key, data):
    """Decode .gz files.

    This decodes compressed files and the continues decoding.

    :param key: file name extension
    :param data: binary data
    """
    import gzip

    if not key.endswith(".gz"):
        return None
    decompressed = gzip.open(io.BytesIO(data)).read()
    return Continue(key[:-3], decompressed)


################################################################
# decode entire training amples
################################################################


default_pre_handlers = [gzfilter]
default_post_handlers = [basichandlers]


class Decoder:
    """Decode samples using a list of handlers.

    For each key/data item, this iterates through the list of
    handlers until some handler returns something other than None.
    """

    def __init__(self, handlers, pre=None, post=None, only=None, partial=False):
        """Create a Decoder.

        :param handlers: main list of handlers
        :param pre: handlers called before the main list (.gz handler by default)
        :param post: handlers called after the main list (default handlers by default)
        :param only: a list of extensions; when give, only ignores files with those extensions
        :param partial: allow partial decoding (i.e., don't decode fields that aren't of type bytes)
        """
        if isinstance(only, str):
            only = only.split()
        self.only = only if only is None else set(only)
        if pre is None:
            pre = default_pre_handlers
        if post is None:
            post = default_post_handlers
        assert all(callable(h) for h in handlers), f"one of {handlers} not callable"
        assert all(callable(h) for h in pre), f"one of {pre} not callable"
        assert all(callable(h) for h in post), f"one of {post} not callable"
        self.handlers = pre + handlers + post
        self.partial = partial

    def decode1(self, key, data):
        """Decode a single field of a sample.

        :param key: file name extension
        :param data: binary data
        """
        key = "." + key
        for f in self.handlers:
            result = f(key, data)
            if isinstance(result, Continue):
                key, data = result.key, result.data
                continue
            if result is not None:
                return result
        return data

    def decode(self, sample):
        """Decode an entire sample.

        :param sample: the sample, a dictionary of key value pairs
        """
        result = {}
        assert isinstance(sample, dict), sample
        for k, v in list(sample.items()):
            if k[0:2] == "__":
                if isinstance(v, bytes):
                    try:
                        v = v.decode("utf-8")
                    except:
                        print(f"Can't decode v of k = {k} as utf-8: v = {v}")
                result[k] = v
                continue
            if self.only is not None and k not in self.only:
                result[k] = v
                continue
            assert v is not None
            if self.partial:
                if isinstance(v, bytes):
                    result[k] = self.decode1(k, v)
                else:
                    result[k] = v
            else:
                assert isinstance(v, bytes), f"k,v = {k}, {v}"
                result[k] = self.decode1(k, v)
        return result

    def __call__(self, sample):
        """Decode an entire sample.

        :param sample: the sample
        """
        assert isinstance(sample, dict), (len(sample), sample)
        return self.decode(sample)

import gzip
import io
import warnings
from typing import Any, Dict, Optional, Union


def check_keys(sample: Dict[str, Any]):
    for k, _ in sample.items():
        if k.startswith("__"):
            continue
        assert k.startswith("."), f"Key {k} must start with a dot"


def decode_all_gz(
    sample: Dict[str, Any], format: Optional[Union[bool, str]] = True, update_key=True
):
    """Decode all keys that end in .gz and rename to the key without .gz."""

    check_keys(sample)

    for key, stream in list(sample.items()):
        extensions = key.split(".")
        if len(extensions) < 1:
            continue
        if len(extensions) < 2 and extensions[-1] == "gz":
            warnings.warn("Plain .gz extension in sample; not decompressed.")
            continue
        extension = extensions[-1]
        if extension in ["gz"]:
            decompressed = gzip.decompress(stream.read())
            new_key = ".".join(extensions[:-1])
            stream = io.BytesIO(decompressed)
            if update_key:
                sample[new_key] = stream
                del sample[key]
            else:
                sample[key] = stream

    check_keys(sample)

    return sample


def decode_basic(sample: Dict[str, Any], format: Optional[Union[bool, str]] = True):
    """Decode basic types in a sample.

    This decodes the following extensions and types:

    - .gz will be decompressed unconditionally and the key renamed without the .gz extension
    - .cls, .cls2 will be decoded as a class label (int)
    - .txt, .text will be decoded as a text string (str)
    - .safetensors will be decoded as a SafeTensors object
    - .npy will be decoded as a numpy array
    - .pkl will be decoded as a pickle object
    - .json will be decoded as a json object
    - .pth, .pt will be decoded as a torch.Tensor
    - .mp will be decoded as a msgpack data structure

    Libraries needed for decoding will be imported as needed.
    """
    check_keys(sample)

    for key, stream in sample.items():
        if key.startswith("__"):
            continue
        extensions = key.split(".")
        if len(extensions) < 1:
            continue
        extension = extensions[-1]
        if isinstance(stream, bytes):
            stream = io.BytesIO(stream)
        if extension in ["gz"] and len(extensions) >= 2:
            # we're assuming that .gz extensions are already decoded
            extension = extensions[-2]
        if extension in ["txt", "text"]:
            value = stream.read()
            sample[key] = value.decode("utf-8")
        elif extension in ["cls", "cls2"]:
            value = stream.read()
            sample[key] = int(value.decode("utf-8"))
        elif extension == "safetensors":
            import safetensors.torch

            sample[key] = safetensors.torch.load_file(stream)
        elif extension == "json":
            import json

            value = stream.read()
            sample[key] = json.loads(value)
        elif extension == "npy":
            import numpy as np

            sample[key] = np.load(stream)
        elif extension == "mp":
            import msgpack

            value = stream.read()
            sample[key] = msgpack.unpackb(value, raw=False)
        elif extension in ["pt", "pth"]:
            import torch

            sample[key] = torch.load(stream)
        elif extension in ["pickle", "pkl"]:
            import pickle

            sample[key] = pickle.load(stream)

    check_keys(sample)

    return sample


def decode_images_to_pil(
    sample: Dict[str, Any], format: Optional[Union[bool, str]] = True
):
    """ """
    import PIL.Image

    check_keys(sample)

    for key, stream in sample.items():
        if key.startswith("__"):
            continue
        extensions = key.split(".")
        if len(extensions) < 1:
            continue
        extension = extensions[-1]
        if isinstance(stream, bytes):
            stream = io.BytesIO(stream)
        if extension in ["jpg", "png", "ppm", "pgm", "pbm", "pnm"]:
            sample[key] = PIL.Image.open(stream)

    check_keys(sample)

    return sample


def decode_images_to_numpy(
    sample: Dict[str, Any], format: Optional[Union[bool, str]] = True
):
    import numpy as np
    import PIL.Image

    check_keys(sample)

    for key, stream in sample.items():
        if key.startswith("__"):
            continue
        extensions = key.split(".")
        if len(extensions) < 1:
            continue
        extension = extensions[-1]
        if isinstance(stream, bytes):
            stream = io.BytesIO(stream)
        if extension in ["jpg", "png", "ppm", "pgm", "pbm", "pnm"]:
            sample[key] = np.asarray(PIL.Image.open(stream))

    check_keys(sample)

    return sample


def default_decoder(
    sample: Dict[str, Any],
    format: Optional[Union[bool, str]] = True,
    gz_update_key=False,
):

    result = dict(sample)

    check_keys(result)

    decode_all_gz(result, update_key=gz_update_key)  # for backwards compatibility
    decode_basic(result)

    if format.lower() == "pil":
        decode_images_to_pil(result)
    elif format.lower() == "numpy":
        decode_images_to_numpy(result)
    else:
        raise ValueError(f"Unknown format: {format}")

    check_keys(result)

    return result

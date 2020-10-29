#!/usr/bin/python
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Train PyTorch models directly from POSIX tar archive, locally
or over HTTP connections.
"""

__all__ = """Dataset tariterator default_handlers imagehandler
reraise_exception ignore_and_continue warn_and_continue ignore_and_stop warn_and_stop
""".split()

import random
import re
import tarfile
import warnings

import braceexpand
from .utils import reraise_exception

from . import gopen


trace = False


meta_prefix = "__"
meta_suffix = "__"


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
    return (
        sample is not None
        and isinstance(sample, dict)
        and len(list(sample.keys())) > 0
        and not sample.get("__bad__", False)
    )


def shardlist(urls, *, shuffle=False):
    """Given a list of URLs, yields that list, possibly shuffled."""
    if isinstance(urls, str):
        urls = braceexpand.braceexpand(urls)
    else:
        urls = list(urls)
    if shuffle:
        random.shuffle(urls)
    for url in urls:
        yield dict(url=url)


def url_opener(data, handler=reraise_exception, **kw):
    """Given a stream of url names (packaged in `dict(url=url)`), yield opened streams."""
    for sample in data:
        assert isinstance(sample, dict)
        assert "url" in sample
        try:
            stream = gopen.gopen(sample["url"], **kw)
            sample.update(stream=stream)
            yield sample
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


def tar_file_iterator(fileobj, skip_meta=r"__[^/]*__($|/)", handler=reraise_exception):
    """Iterator yielding filename, content pairs for the given tar stream.

    :param fileobj: byte stream suitable for tarfile
    :param skip_meta: regexp for keys that are skipped entirely (Default value = r"__[^/]*__($|/)")

    """
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    for tarinfo in stream:
        try:
            if not tarinfo.isreg():
                continue
            fname = tarinfo.name
            if fname is None:
                continue
            if (
                "/" not in fname
                and fname.startswith(meta_prefix)
                and fname.endswith(meta_suffix)
            ):
                # skipping metadata for now
                continue
            if skip_meta is not None and re.match(skip_meta, fname):
                continue
            data = stream.extractfile(tarinfo).read()
            yield fname, data
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
    del stream


def tar_file_expander(data, handler=reraise_exception):
    """Expand a stream of open tar files into a stream of tar file contents.

    This returns an iterator over (filename, file_contents).
    """
    for source in data:
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for sample in tar_file_iterator(source["stream"]):
                assert isinstance(sample, tuple) and len(sample) == 2
                yield sample
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


def group_by_keys(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Returns function over iterator that groups key, value pairs into samples.

    keys: function that splits the key into key and extension (base_plus_ext)
    lcase: convert suffixes to lower case (Default value = True)

    """
    current_sample = None
    for fname, value in data:
        prefix, suffix = keys(fname)
        if trace:
            print(
                prefix,
                suffix,
                current_sample.keys() if isinstance(current_sample, dict) else None,
            )
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        if current_sample is None or prefix != current_sample["__key__"]:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix)
        if suffix in current_sample:
            raise ValueError(
                f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}"
            )
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample

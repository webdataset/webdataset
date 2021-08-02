#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Low level iteration functions for tar archives."""

import random
import re
import tarfile

import braceexpand
from .handlers import reraise_exception

from . import gopen


trace = False


meta_prefix = "__"
meta_suffix = "__"


def base_plus_ext(path):
    """Split off all file extensions.

    Returns base, allext.

    :param path: path with extensions
    :param returns: path with all extensions removed

    """
    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def valid_sample(sample):
    """Check whether a sample is valid.

    :param sample: sample to be checked
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
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        try:
            stream = gopen.gopen(url, **kw)
            sample.update(stream=stream)
            yield sample
        except Exception as exn:
            exn.args = exn.args + (url,)
            if handler(exn):
                continue
            else:
                break


def tar_file_iterator(fileobj, skip_meta=r"__[^/]*__($|/)", handler=reraise_exception, info={}):
    """Iterate over tar file, yielding filename, content pairs for the given tar stream.

    :param fileobj: byte stream suitable for tarfile
    :param skip_meta: regexp for keys that are skipped entirely (Default value = r"__[^/]*__($|/)")

    """
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    for tarinfo in stream:
        fname = tarinfo.name
        try:
            if not tarinfo.isreg():
                continue
            if fname is None:
                continue
            if "/" not in fname and fname.startswith(meta_prefix) and fname.endswith(meta_suffix):
                # skipping metadata for now
                continue
            if skip_meta is not None and re.match(skip_meta, fname):
                continue
            data = stream.extractfile(tarinfo).read()
            result = dict(fname=fname, data=data)
            result.update(info)
            yield result
            stream.members = []
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (exn.args[0] + " @ " + str(fileobj),) + exn.args[1:]
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
        info = None
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            info = {k: v for k, v in source.items() if k.startswith("_")}
            for sample in tar_file_iterator(source["stream"], info=info):
                assert isinstance(sample, dict) and "data" in sample and "fname" in sample
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), info)
            if handler(exn):
                continue
            else:
                break


def group_by_keys(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        info = {k: v for k, v in filesample.items() if k.startswith("__")}
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
            current_sample.update(info)
        if suffix in current_sample:
            raise ValueError(f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}")
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample

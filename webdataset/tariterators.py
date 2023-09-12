#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Low level iteration functions for tar archives."""

from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Set, Tuple

import random, re, tarfile

import braceexpand

from . import filters, gopen
from .handlers import reraise_exception

trace = False
meta_prefix = "__"
meta_suffix = "__"


def base_plus_ext(path):
    """Split off all file extensions.

    Returns base, allext.

    Args:
        path: path with extensions

    Returns:
        path with all extensions removed
    """
    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def valid_sample(sample: Dict[str, Any]) -> bool:
    """Check whether a sample is valid.

    Args:
        sample: a

    Returns:
        boolean indicating whether the sample is valid.
    """
    return (
        sample is not None
        and isinstance(sample, dict)
        and len(list(sample.keys())) > 0
        and not sample.get("__bad__", False)
    )


# FIXME: UNUSED
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


def url_opener(
    data: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    **kw: Dict[str, Any],
):
    """Open URLs and yield a stream of url+stream pairs.

    Args:
        data: iterator over dict(url=...)
        handler: exception handler.
        kw: keyword arguments for gopen.gopen.

    Yields:
        a stream of url+stream pairs.
    """
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


def tar_file_iterator(
    fileobj: tarfile.TarFile,
    skip_meta: Optional[str] = r"__[^/]*__($|/)",
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Iterate over tar file, yielding filename, content pairs for the given tar stream.

    Args:
        fileobj: the tar file stream.
        skip_meta: regexp for keys that are skipped entirely. Defaults to r"__[^/]*__($|/)".
        handler: exception handler. Defaults to reraise_exception.
        select: predicate for selecting files. Defaults to None.

    Yields:
        a stream of samples.
    """
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    for tarinfo in stream:
        fname = tarinfo.name
        try:
            if not tarinfo.isreg():
                continue
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
            if rename_files:
                fname = rename_files(fname)
            if select_files is not None and not select_files(fname):
                continue
            data = stream.extractfile(tarinfo).read()
            result = dict(fname=fname, data=data)
            yield result
            stream.members = []
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (str(exn.args[0]) + " @ " + str(fileobj),) + exn.args[1:]
            if handler(exn):
                continue
            else:
                break
    del stream


def tar_file_expander(
    data: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Expand tar files.

    Args:
        data: iterator over opened tar file streams.
        handler: exception handler.
        select_files: select files from tarfiles by name (permits skipping files).

    Yields:
        a stream of samples.
    """
    for source in data:
        url = source["url"]
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for sample in tar_file_iterator(
                source["stream"],
                handler=handler,
                select_files=select_files,
                rename_files=rename_files,
            ):
                assert (
                    isinstance(sample, dict) and "data" in sample and "fname" in sample
                )
                sample["__url__"] = url
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


def group_by_keys(
    data: Iterable[Dict[str, Any]],
    keys: Callable[[str], Tuple[str, str]] = base_plus_ext,
    lcase: bool = True,
    suffixes: Optional[Set[str]] = None,
    handler: Callable[[Exception], bool] = reraise_exception,
) -> Iterator[Dict[str, Any]]:
    """Group tarfile contents by keys and yield samples.

    Args:
        data: iterator over tarfile contents
        keys: function that takes a file name and returns a key and a suffix.
        lcase: whether to lowercase the suffix.
        suffixes: list of suffixes to keep.
        handler: exception handler.

    Raises:
        ValueError: raised if there are duplicate file names in the tar file.

    Yields:
        iterator over samples.
    """
    current_sample = None
    for filesample in data:
        try:
            assert isinstance(filesample, dict)
            fname, value = filesample["fname"], filesample["data"]
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
                current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
            if suffix in current_sample:
                raise ValueError(
                    f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}"
                )
            if suffixes is None or suffix in suffixes:
                current_sample[suffix] = value
        except Exception as exn:
            exn.args = exn.args + (filesample.get("stream"), filesample.get("url"))
            if handler(exn):
                continue
            else:
                break
    if valid_sample(current_sample):
        yield current_sample


def tarfile_samples(
    src: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
) -> Iterable[Dict[str, Any]]:
    """Given a stream of tar files, yield samples.

    Args:
        src: stream of tar files
        handler: exception handler
        select_files: function that selects files to be included

    Returns:
        stream of samples
    """
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(
        streams, handler=handler, select_files=select_files, rename_files=rename_files
    )
    samples = group_by_keys(files, handler=handler)
    return samples


tarfile_to_samples = filters.pipelinefilter(tarfile_samples)

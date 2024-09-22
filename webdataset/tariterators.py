#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Low level iteration functions for tar archives."""

import random
import re
import tarfile
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Set, Tuple

import braceexpand

from . import filters, gopen
from .handlers import reraise_exception

trace = False
meta_prefix = "__"
meta_suffix = "__"


def base_plus_ext(path):
    """Split off all file extensions.

    Args:
        path: Path with extensions.

    Returns:
        Tuple containing the base path and all extensions.
    """
    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def valid_sample(sample: Dict[str, Any]) -> bool:
    """Check whether a sample is valid.

    Args:
        sample: A dictionary representing a sample.

    Returns:
        Boolean indicating whether the sample is valid.
    """
    return (
        sample is not None
        and isinstance(sample, dict)
        and len(list(sample.keys())) > 0
        and not sample.get("__bad__", False)
    )


# FIXME: UNUSED
def shardlist(urls, *, shuffle=False):
    """Generate a list of URLs, possibly shuffled.

    Args:
        urls: A string or list of URLs.
        shuffle: Whether to shuffle the URLs.

    Yields:
        Dictionary containing the URL.
    """
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
        data: Iterator over dict(url=...).
        handler: Exception handler.
        **kw: Keyword arguments for gopen.gopen.

    Yields:
        A stream of url+stream pairs.
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
        fileobj: The tar file stream.
        skip_meta: Regexp for keys that are skipped entirely.
        handler: Exception handler.
        select_files: Predicate for selecting files.
        rename_files: Function to rename files.

    Yields:
        A stream of samples.
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
    eof_value: Optional[Any] = {},
) -> Iterator[Dict[str, Any]]:
    """Expand tar files.

    Args:
        data: Iterator over opened tar file streams.
        handler: Exception handler.
        select_files: Select files from tarfiles by name (permits skipping files).
        rename_files: Function to rename files.
        eof_value: Value to yield at the end of each shard.

    Yields:
        A stream of samples.
    """
    for source in data:
        url = source["url"]
        local_path = source.get("local_path")
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
                if local_path is not None:
                    sample["__local_path__"] = local_path
                yield sample
            # we yield an EOF marker at the end of each shard so that
            # samples from different shards don't get mixed up
            if eof_value is not None:
                yield eof_value
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
        data: Iterator over tarfile contents.
        keys: Function that takes a file name and returns a key and a suffix.
        lcase: Whether to lowercase the suffix.
        suffixes: List of suffixes to keep.
        handler: Exception handler.

    Raises:
        ValueError: If there are duplicate file names in the tar file.

    Yields:
        Iterator over samples.
    """
    current_sample = None
    for filesample in data:
        try:
            assert isinstance(filesample, dict)
            if filesample == {}:
                if valid_sample(current_sample):
                    yield current_sample
                current_sample = None
                continue
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
            local_path = filesample.get("__local_path__")
            if local_path is not None:
                current_sample["__local_path__"] = local_path
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
    """Generate samples from a stream of tar files.

    Args:
        src: Stream of tar files.
        handler: Exception handler.
        select_files: Function that selects files to be included.
        rename_files: Function to rename files.

    Returns:
        Stream of samples.
    """
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(
        streams, handler=handler, select_files=select_files, rename_files=rename_files
    )
    samples = group_by_keys(files, handler=handler)
    return samples


tarfile_to_samples = filters.pipelinefilter(tarfile_samples)

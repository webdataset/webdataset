"""Multi-modal WebDataset loading from separate tar files.

This module provides support for loading datasets where different modalities
(e.g., images, embeddings, text) are stored in separate tar file directories
with matching shard structures.
"""

import os
import random
import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Union

from . import filters, shardlists
from .compat import FluidInterface, check_empty
from .filters import pipelinefilter, reraise_exception
from .pipeline import DataPipeline
from .pytorch import IterableDataset
from .tariterators import group_by_keys, tar_file_iterator


class PairedShardList(IterableDataset):
    """An iterable dataset that yields paired shard URLs across modalities.

    Takes a dict mapping modality names to shard URL patterns, expands URLs,
    validates all modalities have the same shard count, and yields paired
    URL dictionaries.

    Args:
        modalities: Dict mapping modality names to shard URL patterns
            (strings with brace expansion or lists of URLs).
        seed: Random seed for shuffling; if None, no shuffling is done.
    """

    def __init__(self, modalities: Dict[str, Union[str, List[str]]], seed: Optional[int] = None):
        super().__init__()
        self.seed = seed

        # Expand URLs for each modality
        self.modality_urls: Dict[str, List[str]] = {}
        for name, urls in modalities.items():
            if isinstance(urls, str):
                expanded = shardlists.expand_urls(urls)
            else:
                expanded = list(urls)
            if len(expanded) == 0:
                raise ValueError(f"modality {name!r} has no shard URLs")
            self.modality_urls[name] = expanded

        # Validate all modalities have the same shard count
        counts = {name: len(urls) for name, urls in self.modality_urls.items()}
        unique_counts = set(counts.values())
        if len(unique_counts) > 1:
            raise ValueError(
                f"all modalities must have the same number of shards, got: "
                + ", ".join(f"{name}={count}" for name, count in counts.items())
            )

        self.nshards = next(iter(counts.values()))
        self.modality_names = list(self.modality_urls.keys())

    def __len__(self) -> int:
        return self.nshards

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        indices = list(range(self.nshards))
        if self.seed is not None:
            random.Random(self.seed).shuffle(indices)
        for idx in indices:
            urls = {name: self.modality_urls[name][idx] for name in self.modality_names}
            yield dict(urls=urls)


def _iter_tar_samples(
    url: str,
    handler: Callable = reraise_exception,
    select_files: Optional[Callable] = None,
    rename_files: Optional[Callable] = None,
) -> Iterator[Dict[str, Any]]:
    """Open a tar URL and yield grouped samples (key -> {suffix: data}).

    This is a helper that chains URL opening, tar iteration, and grouping
    for a single tar file.

    Args:
        url: URL of the tar file to open.
        handler: Exception handler.
        select_files: Predicate for selecting files.
        rename_files: Function to rename files.

    Yields:
        Grouped sample dicts with __key__ and extension keys.
    """
    from urllib.parse import urlparse

    from .gopen import gopen as _gopen

    parsed = urlparse(url)
    stream: Any = None
    try:
        if parsed.scheme in ["", "file"]:
            stream = open(parsed.path, "rb")
        else:
            stream = _gopen(url)
    except Exception as exn:
        exn.args = exn.args + (url,)
        if handler(exn):
            return
        else:
            return

    try:
        for tarinfo_sample in tar_file_iterator(
            stream,
            handler=handler,
            select_files=select_files,
            rename_files=rename_files,
        ):
            tarinfo_sample["__url__"] = url
            yield tarinfo_sample
    finally:
        if hasattr(stream, "close"):
            stream.close()


def _group_tar_samples(
    file_iter: Iterator[Dict[str, Any]],
    handler: Callable = reraise_exception,
) -> Iterator[Dict[str, Any]]:
    """Group raw file entries into samples by key.

    Args:
        file_iter: Iterator of dict(fname=..., data=..., __url__=...).
        handler: Exception handler.

    Yields:
        Grouped sample dicts with __key__, __url__, and extension keys.
    """
    yield from group_by_keys(file_iter, handler=handler)


def paired_tar_expander(
    data: Iterator[Dict[str, Any]],
    handler: Callable = reraise_exception,
    select_files: Optional[Callable] = None,
    rename_files: Optional[Callable] = None,
    missing_key_policy: str = "skip",
) -> Iterator[Dict[str, Any]]:
    """Expand paired shard URLs into merged multi-modal samples.

    For each shard group (dict with urls={name: url, ...}), opens all tar files
    simultaneously, groups files into samples by key within each modality, then
    merges samples across modalities by matching __key__ values.

    Uses a streaming sorted merge: maintains one buffered sample per modality.
    When all keys match, merges and yields. When keys differ, advances the
    modality with the minimum key.

    Args:
        data: Iterator of dict(urls={name: url, ...}).
        handler: Exception handler.
        select_files: Predicate for selecting files from tar archives.
        rename_files: Function to rename files from tar archives.
        missing_key_policy: How to handle samples missing from some modalities.
            "skip" (default): Drop samples not present in all modalities.
            "partial": Yield samples with whatever modalities are available.
            "error": Raise ValueError if a key is missing from any modality.

    Yields:
        Merged sample dicts with __key__ and keys from all modalities.

    Raises:
        ValueError: If missing_key_policy is "error" and keys don't align,
            or if two modalities produce the same extension key.
    """
    if missing_key_policy not in ("skip", "partial", "error"):
        raise ValueError(f"missing_key_policy must be 'skip', 'partial', or 'error', got {missing_key_policy!r}")

    for shard_group in data:
        try:
            assert isinstance(shard_group, dict) and "urls" in shard_group
            urls = shard_group["urls"]
            modality_names = list(urls.keys())

            # Create grouped sample iterators for each modality
            iters = {}
            for name in modality_names:
                file_stream = _iter_tar_samples(
                    urls[name],
                    handler=handler,
                    select_files=select_files,
                    rename_files=rename_files,
                )
                iters[name] = _group_tar_samples(file_stream, handler=handler)

            # Buffer one sample per modality
            buffers: Dict[str, Optional[Dict[str, Any]]] = {}
            exhausted: Set[str] = set()

            def advance(name: str) -> None:
                try:
                    buffers[name] = next(iters[name])
                except StopIteration:
                    buffers[name] = None
                    exhausted.add(name)

            # Initial fill
            for name in modality_names:
                advance(name)

            while len(exhausted) < len(modality_names):
                # Get current keys from non-exhausted modalities
                active_keys = {}
                for name in modality_names:
                    buf = buffers[name]
                    if name not in exhausted and buf is not None:
                        active_keys[name] = buf["__key__"]

                if not active_keys:
                    break

                min_key = min(active_keys.values())
                max_key = max(active_keys.values())

                if min_key == max_key:
                    # All active modalities have the same key - merge
                    merged = _merge_samples(buffers, modality_names, exhausted, missing_key_policy)
                    if merged is not None:
                        yield merged
                    # Advance all active modalities
                    for name in modality_names:
                        if name not in exhausted:
                            advance(name)
                else:
                    # Keys differ - handle based on policy
                    names_with_min = [n for n, k in active_keys.items() if k == min_key]

                    if missing_key_policy == "error":
                        raise ValueError(
                            f"key mismatch across modalities: "
                            + ", ".join(f"{n}={active_keys[n]!r}" for n in modality_names if n in active_keys)
                        )
                    elif missing_key_policy == "partial":
                        # Yield partial sample with the min key
                        partial = {"__key__": min_key}
                        for name in names_with_min:
                            sample = buffers[name]
                            assert sample is not None
                            _add_modality_to_sample(partial, name, sample)
                        yield partial
                        for name in names_with_min:
                            advance(name)
                    else:
                        # skip: advance the modality(ies) with the minimum key
                        for name in names_with_min:
                            advance(name)

            # Handle remaining buffered samples for partial policy
            if missing_key_policy == "partial":
                for name in modality_names:
                    buf = buffers[name]
                    if name not in exhausted and buf is not None:
                        partial = {"__key__": buf["__key__"]}
                        _add_modality_to_sample(partial, name, buf)
                        yield partial
                        # Drain remaining
                        for remaining in iters[name]:
                            partial = {"__key__": remaining["__key__"]}
                            _add_modality_to_sample(partial, name, remaining)
                            yield partial

        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


def _add_modality_to_sample(
    merged: Dict[str, Any],
    modality_name: str,
    sample: Dict[str, Any],
) -> None:
    """Add keys from a modality sample to the merged sample dict.

    Args:
        merged: The merged sample dict to update.
        modality_name: Name of the modality being added.
        sample: The modality sample dict.

    Raises:
        ValueError: If an extension key collides with an existing key.
    """
    url = sample.get("__url__", "")
    merged[f"__url_{modality_name}__"] = url
    if "__url__" not in merged:
        merged["__url__"] = url
    else:
        merged["__url__"] = merged["__url__"] + " " + url

    for key, value in sample.items():
        if key.startswith("__") and key.endswith("__"):
            continue
        if key in merged:
            raise ValueError(
                f"extension key {key!r} from modality {modality_name!r} "
                f"collides with an existing key in the merged sample"
            )
        merged[key] = value


def _merge_samples(
    buffers: Dict[str, Optional[Dict[str, Any]]],
    modality_names: List[str],
    exhausted: Set[str],
    missing_key_policy: str,
) -> Optional[Dict[str, Any]]:
    """Merge buffered samples from all modalities into a single sample.

    Args:
        buffers: Dict mapping modality name to current buffered sample.
        modality_names: List of all modality names.
        exhausted: Set of exhausted modality names.
        missing_key_policy: Policy for handling missing keys.

    Returns:
        Merged sample dict, or None if the sample should be skipped.
    """
    # Get the key from any active modality
    key = None
    for name in modality_names:
        buf = buffers[name]
        if name not in exhausted and buf is not None:
            key = buf["__key__"]
            break
    if key is None:
        return None

    # Check if any modality is missing this key
    missing = [name for name in modality_names if name in exhausted or buffers[name] is None]
    if missing:
        if missing_key_policy == "error":
            raise ValueError(f"key {key!r} missing from modalities: {missing}")
        elif missing_key_policy == "skip":
            return None

    merged: Dict[str, Any] = {"__key__": key}
    for name in modality_names:
        buf = buffers[name]
        if name not in exhausted and buf is not None:
            _add_modality_to_sample(merged, name, buf)

    return merged


class MultiModalWebDataset(DataPipeline, FluidInterface):
    """WebDataset pipeline for loading data from separate tar files per modality.

    Each modality is stored in its own set of tar shards with matching shard
    structure. Shards are paired across modalities before splitting/shuffling,
    ensuring alignment is maintained.

    Args:
        modalities: Dict mapping modality names to shard URL patterns.
        handler: Function to handle exceptions. Defaults to reraise_exception.
        shardshuffle: Number of shards to shuffle, or None/False.
        detshuffle: Whether to use deterministic shuffling.
        nodesplitter: Function to split data by node.
        workersplitter: Function to split data by worker.
        select_files: Function to select files from tar archives.
        rename_files: Function to rename files from tar archives.
        empty_check: Whether to check for empty datasets.
        seed: Random seed for shuffling.
        missing_key_policy: How to handle samples missing from some modalities.
            "skip" (default), "partial", or "error".

    Example::

        ds = wds.MultiModalWebDataset(
            modalities={
                "images": "s3://bucket/images/shard_{0000..0099}.tar",
                "embeddings": "s3://bucket/embeddings/shard_{0000..0099}.tar",
            },
            shardshuffle=100,
        ).decode("pil").to_tuple("jpg", "npy").batched(32)
    """

    def __init__(
        self,
        modalities: Dict[str, Union[str, List[str]]],
        handler: Callable = reraise_exception,
        shardshuffle: Optional[Union[int, bool]] = None,
        detshuffle: bool = False,
        nodesplitter: Optional[Callable] = shardlists.single_node_only,
        workersplitter: Optional[Callable] = shardlists.split_by_worker,
        select_files: Optional[Callable] = None,
        rename_files: Optional[Callable] = None,
        empty_check: bool = True,
        seed: Optional[int] = None,
        missing_key_policy: str = "skip",
    ):
        super().__init__()

        if shardshuffle is None:
            warnings.warn("MultiModalWebDataset(shardshuffle=...) is None; set explicitly to False or a number")
        if shardshuffle is True:
            warnings.warn("set MultiModalWebDataset(shardshuffle=...) to a positive integer or 0 or False")
            shardshuffle = 100

        self.seed = int(os.environ.get("WDS_SEED", random.randint(0, 1000000))) if seed is None else seed

        # 1. Paired shard URL generation
        self.append(PairedShardList(modalities))

        # 2. Node splitting (for distributed processing)
        if nodesplitter is not None:
            self.append(nodesplitter)

        # 3. Worker splitting (for DataLoader)
        if workersplitter is not None:
            self.append(workersplitter)

        # 4. Shard shuffling
        if shardshuffle is not None and shardshuffle is not False:
            if detshuffle:
                self.append(filters.detshuffle(shardshuffle, seed=self.seed))
            else:
                self.append(filters.shuffle(shardshuffle, seed=self.seed))

        # 5. Open paired tars and merge samples
        expander = pipelinefilter(paired_tar_expander)
        self.append(
            expander(
                handler=handler,
                select_files=select_files,
                rename_files=rename_files,
                missing_key_policy=missing_key_policy,
            )
        )

        # 6. Check for empty datasets
        if empty_check:
            self.append(check_empty)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Alias for export compatibility
PairedTarExpander = pipelinefilter(paired_tar_expander)

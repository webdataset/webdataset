# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#
# flake8: noqa


"""Exported globals for webdataset library."""

from . import tenbin
from .autodecode import (
    Continue,
    Decoder,
    DecodingError,
    gzfilter,
    handle_extension,
    imagehandler,
    pytorch_weights_only,
    torch_audio,
    torch_loads,
    torch_video,
)
from .cache import (
    LRUCleanup,
    cached_tarfile_samples,
    cached_tarfile_to_samples,
    pipe_cleaner,
)
from .compat import FluidInterface, FluidWrapper, WebDataset, WebLoader
from .extradatasets import MockDataset, with_epoch, with_length
from .filters import (
    Cached,
    LMDBCached,
    associate,
    batched,
    decode,
    detshuffle,
    extract_keys,
    getfirst,
    info,
    map,
    map_dict,
    map_tuple,
    pipelinefilter,
    rename,
    rename_keys,
    rsample,
    select,
    shuffle,
    slice,
    to_tuple,
    transform_with,
    unbatched,
    xdecode,
)
from .gopen import gopen, gopen_schemes
from .handlers import (
    ignore_and_continue,
    ignore_and_stop,
    reraise_exception,
    warn_and_continue,
    warn_and_stop,
)
from .mix import RandomMix, RoundRobin
from .pipeline import DataPipeline
from .shardlists import (
    MultiShardSample,
    ResampledShards,
    SimpleShardList,
    non_empty,
    resampled,
    shardspec,
    single_node_only,
    split_by_node,
    split_by_worker,
)
from .tariterators import tarfile_samples, tarfile_to_samples
from .utils import PipelineStage, repeatedly
from .writer import ShardWriter, TarWriter, numpy_dumps, torch_dumps

__version__ = "0.2.107"

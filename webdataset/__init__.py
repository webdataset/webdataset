# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#
# flake8: noqa


"""Exported globals for webdataset library."""

from . import tenbin
from .fluid import Dataset
from .dataset import (
    WebDataset,
    WebLoader,
)
from .pipeline import (
    DataPipeline,
    stage,
)
from .composable import (
    Composable,
    Shorthands,
    Processor,
)
from .shardlists import (
    SimpleShardList,
    PytorchShardList,
    ResampledShards,
    split_by_node,
    split_by_worker,
    non_empty,
    resampled,
)
from .extradatasets import (
    MockDataset
)
from .tariterators import (
    tarfile_samples,
    tarfile_to_samples,
)
from .handlers import (
    reraise_exception,
    ignore_and_continue,
    warn_and_continue,
    ignore_and_stop,
    warn_and_stop,
)
from .writer import ShardWriter, TarWriter, torch_dumps, numpy_dumps
from .autodecode import (
    Continue,
    handle_extension,
    torch_loads,
    imagehandler,
    torch_video,
    torch_audio,
    Decoder,
    gzfilter,
)
from .tariterators import (
    url_opener,
    tar_file_iterator,
    tar_file_expander,
    group_by_keys,
)
from .iterators import (
    transform_with,
    getfirst,
)
from .filters import (
    Curried,
    info,
    shuffle,
    select,
    decode,
    map,
    rename,
    associate,
    map_dict,
    to_tuple,
    map_tuple,
    batched,
    unbatched,
    slice,
)
from .dbcache import DBCache
from .dsspecs import RoundRobin

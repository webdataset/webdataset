# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#
# flake8: noqa


"""Exported globals for webdataset library."""

from . import tenbin
from .compat import (
    WebDataset,
    WebLoader,
)
from .pipeline import (
    DataPipeline,
)
from .shardlists import (
    SimpleShardList,
    ResampledShards,
    MultiShardSample,
    split_by_node,
    single_node_only,
    split_by_worker,
    non_empty,
    resampled,
    shardspec,
)
from .extradatasets import (
    MockDataset,
    with_epoch,
    with_length,
)
from .tariterators import (
    tarfile_samples,
    tarfile_to_samples,
)
from .cborsiterators import (
    cbors_to_samples,
    cbors2_to_samples,
)
from .handlers import (
    reraise_exception,
    ignore_and_continue,
    warn_and_continue,
    ignore_and_stop,
    warn_and_stop,
)
from .writer import (
    ShardWriter,
    TarWriter,
    torch_dumps,
    numpy_dumps,
)
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
from .cache import (
    cached_tarfile_samples,
    cached_tarfile_to_samples,
    lru_cleanup,
    pipe_cleaner,
)
from .filters import (
    transform_with,
    getfirst,
    pipelinefilter,
    info,
    shuffle,
    detshuffle,
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
    rsample,
    slice,
)
from .utils import (
    repeatedly,
    PipelineStage,
)

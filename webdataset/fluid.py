#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""A deprecated interface to WebDataset."""

from .dataset import (
    WebDataset,
    default_cache_dir,
    default_cache_name,
    default_cache_size,
    default_cache_verbose,
)
from .handlers import reraise_exception
import warnings

from .pytorch import IterableDataset


class Dataset(IterableDataset):
    """This class works almost identically to WebDataset but with internal state."""

    def __init__(
        self,
        urls,
        *,
        length=True,
        splitter=True,
        handler=reraise_exception,
        shuffle=False,
        cache_dir=default_cache_dir,
        cache_size=default_cache_size,
        cache_name=default_cache_name,
        cache_verbose=default_cache_verbose
    ):
        """Create a Dataset instance. See WebDataset for documentation."""
        super().__init__()
        raise Exception("Dataset is deprecated; use webdataset.WebDataset instead")
        self._dataset = WebDataset(
            urls,
            shardshuffle=shuffle,
            splitter=splitter,
            handler=handler,
            length=length,
            cache_dir=cache_dir,
            cache_size=cache_size,
            cache_name=cache_name,
            cache_verbose=cache_verbose,
        )

    def __getattr__(self, name):
        """Forward method calls to the underlying WebDataset and update the internal pipe."""
        if not hasattr(self._dataset, name):
            raise AttributeError()

        def f(*args, **kw):
            """Call the underlying method."""
            self._dataset = getattr(self._dataset, name)(*args, **kw)
            return self

        return f

    def __iter__(self):
        """Return an iterator over the underlying dataset."""
        return iter(self._dataset)

    def __len__(self):
        """Return the length of the underlying dataset."""
        return len(self._dataset)

#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""Mock implementations of torch interfaces when torch is not available."""


try:
    from torch.utils.data import DataLoader, IterableDataset
except ModuleNotFoundError:

    class IterableDataset:
        """Empty implementation of IterableDataset when torch is not available."""

        pass

    class DataLoader:
        """Empty implementation of DataLoader when torch is not available."""

        pass


try:
    from torch import Tensor as TorchTensor
except ModuleNotFoundError:

    class TorchTensor:
        """Empty implementation of TorchTensor when torch is not available."""

        pass

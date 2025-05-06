from typing import Any, Iterator, TypeVar

T = TypeVar('T')

# These are only mock implementations when torch is not available
# In real usage, they will be imported from torch.utils.data

class IterableDataset:
    def __iter__(self) -> Iterator[Any]: ...

class DataLoader:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __iter__(self) -> Iterator[Any]: ...

# Mock implementation of TorchTensor when torch is not available
class TorchTensor:
    shape: tuple
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

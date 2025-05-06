from typing import Any, Dict, Iterable, Iterator, Optional, TypeVar

from .pytorch import IterableDataset
from .utils import PipelineStage

T = TypeVar('T')
Sample = Dict[str, Any]

class MockDataset(IterableDataset):
    sample: Any
    length: int

    def __init__(self, sample: Any, length: int) -> None: ...
    def __iter__(self) -> Iterator[Any]: ...

class repeatedly(IterableDataset, PipelineStage):
    source: Any
    length: Optional[int]
    nbatches: Optional[int]
    nepochs: Optional[int]

    def __init__(self, source: Any, nepochs: Optional[int] = None,
                 nbatches: Optional[int] = None, length: Optional[int] = None) -> None: ...
    def invoke(self, source: Iterable[T]) -> Iterator[T]: ...

class with_epoch(IterableDataset):
    length: int
    source: Optional[Iterator[Any]]

    def __init__(self, dataset: Any, length: int) -> None: ...
    def __getstate__(self) -> Dict[str, Any]: ...
    def invoke(self, dataset: Iterable[T]) -> Iterator[T]: ...

class with_length(IterableDataset, PipelineStage):
    dataset: Any
    length: int

    def __init__(self, dataset: Any, length: int) -> None: ...
    def invoke(self, dataset: Iterable[T]) -> Iterator[T]: ...
    def __len__(self) -> int: ...

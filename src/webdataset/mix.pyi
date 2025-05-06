from typing import Any, Iterator, List, Optional, TypeVar


from .pytorch import IterableDataset

T = TypeVar('T')
Dataset = IterableDataset
Source = Iterator[T]

def round_robin_shortest(*sources: Source) -> Iterator[T]: ...
def round_robin_longest(*sources: Source) -> Iterator[T]: ...

class RoundRobin(IterableDataset):
    datasets: List[Dataset]
    longest: bool

    def __init__(self, datasets: List[Dataset], longest: bool = False) -> None: ...
    def __iter__(self) -> Iterator[Any]: ...

def random_samples(sources: List[Source], probs: Optional[List[float]] = None, longest: bool = False) -> Iterator[T]: ...

class RandomMix(IterableDataset):
    datasets: List[Dataset]
    probs: Optional[List[float]]
    longest: bool

    def __init__(self, datasets: List[Dataset], probs: Optional[List[float]] = None, longest: bool = False) -> None: ...
    def __iter__(self) -> Iterator[Any]: ...

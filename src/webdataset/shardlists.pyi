import random
import re
import sys
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

from .pytorch import IterableDataset

T = TypeVar('T')

def envlookup(m: re.Match) -> str: ...
def envsubst(s: str) -> str: ...
def split_by_node(src: Iterable[T], group: Any = None) -> Iterator[T]: ...
def single_node_only(src: Iterable[T], group: Any = None) -> Iterator[T]: ...
def split_by_worker(src: Iterable[T]) -> Iterator[T]: ...
def expand_urls(urls: str) -> List[str]: ...
def expand_source(source: Union[str, List[str], Iterable], max_urls: int = int(1e9)) -> List[str]: ...

class SimpleShardList(IterableDataset):
    urls: List[str]
    seed: Optional[Union[int, bool]]

    def __init__(self, urls: Union[str, List[str]], seed: Optional[Union[int, bool]] = None) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Dict[str, str]]: ...

def resampled_(src: Iterable[T], n: int = sys.maxsize) -> Iterator[T]: ...
resampled: Any  # pipelinefilter return type

def non_empty(src: Iterable[T]) -> Iterator[T]: ...

@dataclass
class MSSource:
    name: str = ""
    perepoch: int = -1
    resample: bool = False
    urls: List[str] = field(default_factory=list)

default_rng: random.Random

def expand(s: str) -> str: ...

class ResampledShards(IterableDataset):
    urls: List[str]
    nshards: int
    worker_seed: Callable
    deterministic: bool
    seed: int
    epoch: int
    rng: random.Random

    def __init__(
        self,
        urls: Union[str, List[str], Iterable],
        nshards: int = sys.maxsize,
        seed: int = 0,
        worker_seed: Optional[Callable] = None,
        deterministic: bool = False,
        max_urls: int = int(1e6),
        empty_check: bool = True,
    ) -> None: ...

    def __iter__(self) -> Iterator[Dict[str, str]]: ...

ResampledShardList = ResampledShards

def check_pid_is_running(pid: int) -> bool: ...
def without_last_extension(fname: str) -> str: ...
def get_pid_from_filename(fname: str) -> Optional[int]: ...

class DirectoryShardList(IterableDataset):
    path: str
    poll: int
    pattern: str
    mode: str
    select: str
    fate: Any
    timeout: float

    def __init__(
        self,
        path: str,
        pattern: str = "*.{tar,tgz,tar.tgz}",
        poll: int = 1,
        timeout: float = 1e12,
        mode: str = "resample",
        select: str = "random",
        fate: Any = None,
    ) -> None: ...

    def recycle(self, activename: str) -> None: ...
    def cleanup_files_without_processes(self) -> None: ...
    def __iter__(self) -> Iterator[Dict[str, str]]: ...

class MultiShardSample(IterableDataset):
    epoch: int
    rng: random.Random
    sources: List[MSSource]

    # Removed obsolete decorator to avoid type errors
    def __init__(self, fname: Union[str, Dict[str, Any]]) -> None: ...
    def parse_spec(self, fname: Union[str, Dict[str, Any]]) -> None: ...
    def set_epoch(self, seed: int) -> None: ...
    def get_shards_for_epoch(self) -> List[str]: ...
    def __iter__(self) -> Iterator[Dict[str, str]]: ...

def shardspec(spec: str) -> Union[MultiShardSample, SimpleShardList]: ...

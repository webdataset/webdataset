import multiprocessing as mp
import weakref
from typing import Any, Iterator, List, Optional, TypeVar

import zmq

T = TypeVar('T')

the_protocol: int
all_pids: weakref.WeakSet

class EOF:
    index: int

    def __init__(self, **kw: Any) -> None: ...

def reader(dataset: Any, sockname: str, index: int, num_workers: int) -> None: ...

class MultiLoader:
    dataset: Any
    workers: int
    verbose: bool
    pids: List[Optional[mp.Process]]
    socket: Optional[zmq.Socket]
    ctx: zmq.Context
    nokill: bool
    prefix: str
    sockname: str

    def __init__(
        self,
        dataset: Any,
        workers: int = 4,
        verbose: bool = False,
        nokill: bool = False,
        prefix: str = "/tmp/_multi-"
    ) -> None: ...

    def kill(self) -> None: ...
    def __iter__(self) -> Iterator[Any]: ...

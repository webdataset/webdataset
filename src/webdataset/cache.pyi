import os
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Union

from .handlers import reraise_exception
from .utils import obsolete

# Constants
default_cache_dir: str
default_cache_size: float
verbose_cache: int

# Functions
def islocal(url: str) -> bool: ...
def get_filetype(fname: str) -> str: ...
def check_tar_format(fname: str) -> bool: ...

@obsolete
def pipe_cleaner(spec: str) -> str: ...

def url_to_cache_name(url: str, ndir: int = 0) -> str: ...

# Classes
class LRUCleanup:
    cache_dir: Optional[str]
    cache_size: int
    keyfn: Callable[[str], float]
    verbose: bool
    interval: Optional[int]
    last_run: float

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        cache_size: int = int(1e12),
        keyfn: Callable[[str], float] = os.path.getctime,
        verbose: bool = False,
        interval: Optional[int] = 30,
    ) -> None: ...

    def set_cache_dir(self, cache_dir: str) -> None: ...
    def cleanup(self) -> None: ...

def download(url: str, dest: str, chunk_size: int = 1024**2, verbose: bool = False) -> None: ...

class StreamingOpen:
    verbose: bool
    handler: Callable[[Exception], bool]

    def __init__(self, verbose: bool = False, handler: Callable[[Exception], bool] = reraise_exception) -> None: ...
    def __call__(self, urls: Iterable[Union[str, Dict[str, str]]]) -> Iterator[Dict[str, Any]]: ...

class FileCache:
    url_to_name: Callable[[str], str]
    validator: Callable[[str], bool]
    handler: Callable[[Exception], bool]
    cache_dir: str
    verbose: bool
    cleaner: Optional[LRUCleanup]

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        *,
        url_to_name: Callable[[str], str] = url_to_cache_name,
        verbose: bool = False,
        validator: Callable[[str], bool] = check_tar_format,
        handler: Callable[[Exception], bool] = reraise_exception,
        cache_size: int = -1,
        cache_cleanup_interval: int = 30,
    ) -> None: ...

    def get_file(self, url: str) -> str: ...
    def __call__(self, urls: Iterable[Union[str, Dict[str, str]]]) -> Iterator[Dict[str, Any]]: ...

from subprocess import Popen
from typing import (
    IO,
    Any,
    BinaryIO,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

# Global variables
info: Dict[str, Any]
gopen_schemes: Dict[str, Callable]

T = TypeVar('T')

class Pipe:
    ignore_errors: bool
    ignore_status: List[int]
    timeout: float
    args: Tuple[Tuple, Dict[str, Any]]
    proc: Popen
    stream: Union[IO[bytes], BinaryIO]
    status: Optional[int]
    handler: Optional[Callable]

    def __init__(
        self,
        *args: Any,
        mode: Optional[str] = None,
        timeout: float = 7200.0,
        ignore_errors: bool = False,
        ignore_status: List[int] = [],
        **kw: Any
    ) -> None: ...

    def __str__(self) -> str: ...
    def check_status(self) -> None: ...
    def wait_for_child(self) -> None: ...
    def read(self, *args: Any, **kw: Any) -> bytes: ...
    def write(self, *args: Any, **kw: Any) -> int: ...
    def readLine(self, *args: Any, **kw: Any) -> bytes: ...
    def close(self) -> None: ...
    def __enter__(self) -> 'Pipe': ...
    def __exit__(self, etype: Any, value: Any, traceback: Any) -> None: ...
    def __del__(self) -> None: ...

def set_options(
    obj: Any,
    timeout: Optional[float] = None,
    ignore_errors: Optional[bool] = None,
    ignore_status: Optional[List[int]] = None,
    handler: Optional[Callable] = None
) -> bool: ...

def gopen_file(url: str, mode: str = "rb", bufsize: int = 8192) -> IO[bytes]: ...
def gopen_pipe(url: str, mode: str = "rb", bufsize: int = 8192) -> Pipe: ...
def gopen_curl(url: str, mode: str = "rb", bufsize: int = 8192) -> Pipe: ...
def gopen_htgs(url: str, mode: str = "rb", bufsize: int = 8192) -> Pipe: ...
def gopen_hf(url: str, mode: str = "rb", bufsize: int = 8192) -> Pipe: ...
def gopen_gsutil(url: str, mode: str = "rb", bufsize: int = 8192) -> Pipe: ...
def gopen_ais(url: str, mode: str = "rb", bufsize: int = 8192) -> Pipe: ...
def gopen_error(url: str, *args: Any, **kw: Any) -> None: ...
def rewrite_url(url: str) -> str: ...

def gopen(
    url: str,
    mode: str = "rb",
    bufsize: int = 8192,
    **kw: Any
) -> Union[BinaryIO, Pipe]: ...

def reader(url: str, **kw: Any) -> Union[BinaryIO, Pipe]: ...

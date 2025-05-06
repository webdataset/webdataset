import tarfile
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

import numpy as np
import PIL.Image

T = TypeVar('T')

# Type definitions
Handler = Callable[[Any], bytes]
Encoder = Callable[[Dict[str, Any]], Dict[str, Any]]
FileObj = Union[str, BinaryIO]

def imageencoder(image: Union[PIL.Image.Image, np.ndarray], format: str = "PNG") -> bytes: ...
def bytestr(data: Any) -> bytes: ...
def torch_dumps(data: Any) -> bytes: ...
def numpy_dumps(data: np.ndarray) -> bytes: ...
def numpy_npz_dumps(data: Dict[str, np.ndarray]) -> bytes: ...
def tenbin_dumps(x: Union[List[np.ndarray], np.ndarray]) -> memoryview: ...
def cbor_dumps(x: Any) -> bytes: ...
def mp_dumps(x: Any) -> bytes: ...

def add_handlers(d: Dict[str, Handler], keys: Union[str, List[str]], value: Handler) -> None: ...
def make_handlers() -> Dict[str, Handler]: ...

default_handlers: Dict[str, Handler]

def encode_based_on_extension1(data: Any, tname: str, handlers: Dict[str, Handler]) -> bytes: ...
def encode_based_on_extension(sample: Dict[str, Any], handlers: Dict[str, Handler]) -> Dict[str, Any]: ...
def make_encoder(spec: Union[bool, str, Dict[str, Handler], Callable]) -> Encoder: ...

class TarWriter:
    mtime: Optional[float]
    own_fileobj: Optional[BinaryIO]
    encoder: Encoder
    keep_meta: bool
    stream: Any
    tarstream: tarfile.TarFile
    user: str
    group: str
    mode: int
    compress: Optional[Union[bool, str]]

    def __init__(
        self,
        fileobj: FileObj,
        user: str = "bigdata",
        group: str = "bigdata",
        mode: int = 0o0444,
        compress: Optional[Union[bool, str]] = None,
        encoder: Union[None, bool, Callable] = True,
        keep_meta: bool = False,
        mtime: Optional[float] = None,
        format: Any = None,
    ) -> None: ...

    def __enter__(self) -> 'TarWriter': ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    def close(self) -> None: ...
    def write(self, obj: Dict[str, Any]) -> int: ...

    @staticmethod
    def tarmode(fileobj: FileObj, compress: Optional[Union[bool, str]] = None) -> str: ...

class ShardWriter:
    verbose: int
    kw: Dict[str, Any]
    maxcount: int
    maxsize: float
    post: Optional[Callable[[str], Any]]
    tarstream: Optional[TarWriter]
    shard: int
    pattern: str
    total: int
    count: int
    size: int
    fname: Optional[str]
    opener: Optional[Callable[[str], Any]]

    def __init__(
        self,
        pattern: str,
        maxcount: int = 100000,
        maxsize: float = 3e9,
        post: Optional[Callable[[str], Any]] = None,
        start_shard: int = 0,
        verbose: int = 1,
        opener: Optional[Callable[[str], Any]] = None,
        **kw: Any,
    ) -> None: ...

    def next_stream(self) -> None: ...
    def write(self, obj: Dict[str, Any]) -> None: ...
    def finish(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> 'ShardWriter': ...
    def __exit__(self, *args: Any, **kw: Any) -> None: ...

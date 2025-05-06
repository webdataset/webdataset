import tarfile
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)


from .handlers import reraise_exception

T = TypeVar('T')
Sample = Dict[str, Any]

trace: bool
meta_prefix: str
meta_suffix: str

def base_plus_ext(path: str) -> Tuple[Optional[str], Optional[str]]: ...
def valid_sample(sample: Dict[str, Any]) -> bool: ...
def shardlist(urls: Union[str, Iterable[str]], *, shuffle: bool = False) -> Iterator[Dict[str, str]]: ...

def url_opener(
    data: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    **kw: Any
) -> Iterator[Dict[str, Any]]: ...

def tar_file_iterator(
    fileobj: tarfile.TarFile,
    skip_meta: Optional[str] = r"__[^/]*__($|/)",
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
) -> Iterator[Dict[str, Any]]: ...

def tar_file_expander(
    data: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
    eof_value: Optional[Any] = {},
) -> Iterator[Dict[str, Any]]: ...

def group_by_keys(
    data: Iterable[Dict[str, Any]],
    keys: Callable[[str], Tuple[Optional[str], Optional[str]]] = base_plus_ext,
    lcase: bool = True,
    suffixes: Optional[Set[str]] = None,
    handler: Callable[[Exception], bool] = reraise_exception,
) -> Iterator[Dict[str, Any]]: ...

def tarfile_samples(
    src: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
) -> Iterable[Dict[str, Any]]: ...

tarfile_to_samples: Any  # pipelinefilter return type

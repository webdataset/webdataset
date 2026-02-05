from types import SimpleNamespace
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

from . import cache, filters, shardlists
from .filters import reraise_exception
from .pipeline import DataPipeline

T = TypeVar('T')
Sample = Dict[str, Any]
Handler = Callable[[Exception], bool]

class FluidInterface:
    def batched(self, batchsize: int, collation_fn: Optional[Callable] = filters.default_collation_fn, partial: bool = True) -> 'FluidInterface': ...
    def unbatched(self) -> 'FluidInterface': ...
    def listed(self, batchsize: int, partial: bool = True) -> 'FluidInterface': ...
    def unlisted(self) -> 'FluidInterface': ...
    def log_keys(self, logfile: Optional[str] = None) -> 'FluidInterface': ...
    def shuffle(self, size: int, **kw: Any) -> 'FluidInterface': ...
    def map(self, f: Callable, handler: Handler = reraise_exception) -> 'FluidInterface': ...
    def decode(
        self,
        *args: Union[str, Callable],
        pre: Optional[List[Callable]] = None,
        post: Optional[List[Callable]] = None,
        only: Optional[List[str]] = None,
        partial: bool = False,
        handler: Handler = reraise_exception,
    ) -> 'FluidInterface': ...
    def map_dict(self, handler: Handler = reraise_exception, **kw: Callable) -> 'FluidInterface': ...
    def select(self, predicate: Callable[[Any], bool], **kw: Any) -> 'FluidInterface': ...
    def to_tuple(self, *args: str, **kw: Any) -> 'FluidInterface': ...
    def map_tuple(self, *args: Callable, handler: Handler = reraise_exception) -> 'FluidInterface': ...
    def slice(self, *args: int) -> 'FluidInterface': ...
    def rename(self, **kw: str) -> 'FluidInterface': ...
    def rsample(self, p: float = 0.5) -> 'FluidInterface': ...
    def rename_keys(self, *args: Any, **kw: Any) -> 'FluidInterface': ...
    def extract_keys(self, *args: str, **kw: Any) -> 'FluidInterface': ...
    def xdecode(self, *args: Any, **kw: Any) -> 'FluidInterface': ...
    def mcached(self) -> 'FluidInterface': ...
    def lmdb_cached(self, *args: Any, **kw: Any) -> 'FluidInterface': ...
    def compose(self, other: Any) -> Any: ...

def check_empty(source: Iterable[Sample]) -> Iterator[Sample]: ...

class WebDataset(DataPipeline, FluidInterface):
    seed: int

    def __init__(
        self,
        urls: Union[str, Dict[str, Any], Iterable[str]],
        handler: Callable[[Exception], bool] = reraise_exception,
        mode: Optional[str] = None,
        resampled: bool = False,
        repeat: bool = False,
        shardshuffle: Optional[Union[bool, int]] = None,
        cache_size: int = -1,
        cache_dir: Optional[str] = None,
        url_to_name: Any = cache.pipe_cleaner,
        detshuffle: bool = False,
        nodesplitter: Optional[Callable] = shardlists.single_node_only,
        workersplitter: Optional[Callable] = shardlists.split_by_worker,
        select_files: Optional[Callable[[str], bool]] = None,
        rename_files: Optional[Callable[[str], str]] = None,
        empty_check: bool = True,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> None: ...

    def update_cache_info(self, args: SimpleNamespace) -> None: ...
    def create_url_iterator(self, args: SimpleNamespace) -> None: ...
    def __enter__(self) -> 'WebDataset': ...
    def __exit__(self, *args: Any) -> None: ...

    # Override DataPipeline methods to return WebDataset for fluent interface
    def compose(self, *args: Any) -> 'WebDataset': ...  # type: ignore[override]
    def with_length(self, n: int, silent: bool = False) -> 'WebDataset': ...  # type: ignore[override]
    def with_epoch(self, nsamples: int = -1, nbatches: int = -1) -> 'WebDataset': ...  # type: ignore[override]
    def repeat(self, nepochs: int = -1, nbatches: int = -1) -> 'WebDataset': ...  # type: ignore[override]

    # Override FluidInterface methods to return WebDataset for fluent interface
    def batched(self, batchsize: int, collation_fn: Optional[Callable] = filters.default_collation_fn, partial: bool = True) -> 'WebDataset': ...  # type: ignore[override]
    def unbatched(self) -> 'WebDataset': ...  # type: ignore[override]
    def listed(self, batchsize: int, partial: bool = True) -> 'WebDataset': ...  # type: ignore[override]
    def unlisted(self) -> 'WebDataset': ...  # type: ignore[override]
    def log_keys(self, logfile: Optional[str] = None) -> 'WebDataset': ...  # type: ignore[override]
    def shuffle(self, size: int, **kw: Any) -> 'WebDataset': ...  # type: ignore[override]
    def map(self, f: Callable, handler: Handler = reraise_exception) -> 'WebDataset': ...  # type: ignore[override]
    def decode(
        self,
        *args: Union[str, Callable],
        pre: Optional[List[Callable]] = None,
        post: Optional[List[Callable]] = None,
        only: Optional[List[str]] = None,
        partial: bool = False,
        handler: Handler = reraise_exception,
    ) -> 'WebDataset': ...  # type: ignore[override]
    def map_dict(self, handler: Handler = reraise_exception, **kw: Callable) -> 'WebDataset': ...  # type: ignore[override]
    def select(self, predicate: Callable[[Any], bool], **kw: Any) -> 'WebDataset': ...  # type: ignore[override]
    def to_tuple(self, *args: str, **kw: Any) -> 'WebDataset': ...  # type: ignore[override]
    def map_tuple(self, *args: Callable, handler: Handler = reraise_exception) -> 'WebDataset': ...  # type: ignore[override]
    def slice(self, *args: int) -> 'WebDataset': ...  # type: ignore[override]
    def rename(self, **kw: str) -> 'WebDataset': ...  # type: ignore[override]
    def rsample(self, p: float = 0.5) -> 'WebDataset': ...  # type: ignore[override]
    def rename_keys(self, *args: Any, **kw: Any) -> 'WebDataset': ...  # type: ignore[override]
    def extract_keys(self, *args: str, **kw: Any) -> 'WebDataset': ...  # type: ignore[override]
    def xdecode(self, *args: Any, **kw: Any) -> 'WebDataset': ...  # type: ignore[override]
    def mcached(self) -> 'WebDataset': ...  # type: ignore[override]
    def lmdb_cached(self, *args: Any, **kw: Any) -> 'WebDataset': ...  # type: ignore[override]

class FluidWrapper(DataPipeline, FluidInterface):
    def __init__(self, initial: Any) -> None: ...

    # Override DataPipeline methods to return FluidWrapper for fluent interface
    def compose(self, *args: Any) -> 'FluidWrapper': ...  # type: ignore[override]
    def with_length(self, n: int, silent: bool = False) -> 'FluidWrapper': ...  # type: ignore[override]
    def with_epoch(self, nsamples: int = -1, nbatches: int = -1) -> 'FluidWrapper': ...  # type: ignore[override]
    def repeat(self, nepochs: int = -1, nbatches: int = -1) -> 'FluidWrapper': ...  # type: ignore[override]

    # Override FluidInterface methods to return FluidWrapper for fluent interface
    def batched(self, batchsize: int, collation_fn: Optional[Callable] = filters.default_collation_fn, partial: bool = True) -> 'FluidWrapper': ...  # type: ignore[override]
    def unbatched(self) -> 'FluidWrapper': ...  # type: ignore[override]
    def listed(self, batchsize: int, partial: bool = True) -> 'FluidWrapper': ...  # type: ignore[override]
    def unlisted(self) -> 'FluidWrapper': ...  # type: ignore[override]
    def log_keys(self, logfile: Optional[str] = None) -> 'FluidWrapper': ...  # type: ignore[override]
    def shuffle(self, size: int, **kw: Any) -> 'FluidWrapper': ...  # type: ignore[override]
    def map(self, f: Callable, handler: Handler = reraise_exception) -> 'FluidWrapper': ...  # type: ignore[override]
    def decode(
        self,
        *args: Union[str, Callable],
        pre: Optional[List[Callable]] = None,
        post: Optional[List[Callable]] = None,
        only: Optional[List[str]] = None,
        partial: bool = False,
        handler: Handler = reraise_exception,
    ) -> 'FluidWrapper': ...  # type: ignore[override]
    def map_dict(self, handler: Handler = reraise_exception, **kw: Callable) -> 'FluidWrapper': ...  # type: ignore[override]
    def select(self, predicate: Callable[[Any], bool], **kw: Any) -> 'FluidWrapper': ...  # type: ignore[override]
    def to_tuple(self, *args: str, **kw: Any) -> 'FluidWrapper': ...  # type: ignore[override]
    def map_tuple(self, *args: Callable, handler: Handler = reraise_exception) -> 'FluidWrapper': ...  # type: ignore[override]
    def slice(self, *args: int) -> 'FluidWrapper': ...  # type: ignore[override]
    def rename(self, **kw: str) -> 'FluidWrapper': ...  # type: ignore[override]
    def rsample(self, p: float = 0.5) -> 'FluidWrapper': ...  # type: ignore[override]
    def rename_keys(self, *args: Any, **kw: Any) -> 'FluidWrapper': ...  # type: ignore[override]
    def extract_keys(self, *args: str, **kw: Any) -> 'FluidWrapper': ...  # type: ignore[override]
    def xdecode(self, *args: Any, **kw: Any) -> 'FluidWrapper': ...  # type: ignore[override]
    def mcached(self) -> 'FluidWrapper': ...  # type: ignore[override]
    def lmdb_cached(self, *args: Any, **kw: Any) -> 'FluidWrapper': ...  # type: ignore[override]

class WebLoader(DataPipeline, FluidInterface):
    def __init__(self, *args: Any, **kw: Any) -> None: ...

    # Override DataPipeline methods to return WebLoader for fluent interface
    def compose(self, *args: Any) -> 'WebLoader': ...  # type: ignore[override]
    def with_length(self, n: int, silent: bool = False) -> 'WebLoader': ...  # type: ignore[override]
    def with_epoch(self, nsamples: int = -1, nbatches: int = -1) -> 'WebLoader': ...  # type: ignore[override]
    def repeat(self, nepochs: int = -1, nbatches: int = -1) -> 'WebLoader': ...  # type: ignore[override]

    # Override FluidInterface methods to return WebLoader for fluent interface
    def batched(self, batchsize: int, collation_fn: Optional[Callable] = filters.default_collation_fn, partial: bool = True) -> 'WebLoader': ...  # type: ignore[override]
    def unbatched(self) -> 'WebLoader': ...  # type: ignore[override]
    def listed(self, batchsize: int, partial: bool = True) -> 'WebLoader': ...  # type: ignore[override]
    def unlisted(self) -> 'WebLoader': ...  # type: ignore[override]
    def log_keys(self, logfile: Optional[str] = None) -> 'WebLoader': ...  # type: ignore[override]
    def shuffle(self, size: int, **kw: Any) -> 'WebLoader': ...  # type: ignore[override]
    def map(self, f: Callable, handler: Handler = reraise_exception) -> 'WebLoader': ...  # type: ignore[override]
    def decode(
        self,
        *args: Union[str, Callable],
        pre: Optional[List[Callable]] = None,
        post: Optional[List[Callable]] = None,
        only: Optional[List[str]] = None,
        partial: bool = False,
        handler: Handler = reraise_exception,
    ) -> 'WebLoader': ...  # type: ignore[override]
    def map_dict(self, handler: Handler = reraise_exception, **kw: Callable) -> 'WebLoader': ...  # type: ignore[override]
    def select(self, predicate: Callable[[Any], bool], **kw: Any) -> 'WebLoader': ...  # type: ignore[override]
    def to_tuple(self, *args: str, **kw: Any) -> 'WebLoader': ...  # type: ignore[override]
    def map_tuple(self, *args: Callable, handler: Handler = reraise_exception) -> 'WebLoader': ...  # type: ignore[override]
    def slice(self, *args: int) -> 'WebLoader': ...  # type: ignore[override]
    def rename(self, **kw: str) -> 'WebLoader': ...  # type: ignore[override]
    def rsample(self, p: float = 0.5) -> 'WebLoader': ...  # type: ignore[override]
    def rename_keys(self, *args: Any, **kw: Any) -> 'WebLoader': ...  # type: ignore[override]
    def extract_keys(self, *args: str, **kw: Any) -> 'WebLoader': ...  # type: ignore[override]
    def xdecode(self, *args: Any, **kw: Any) -> 'WebLoader': ...  # type: ignore[override]
    def mcached(self) -> 'WebLoader': ...  # type: ignore[override]
    def lmdb_cached(self, *args: Any, **kw: Any) -> 'WebLoader': ...  # type: ignore[override]

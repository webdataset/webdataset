from . import iterators as iterators
from typing import Any

class Curried2:
    f: Any = ...
    args: Any = ...
    kw: Any = ...
    def __init__(self, f: Any, *args: Any, **kw: Any) -> None: ...
    def __call__(self, data: Any): ...

class Curried:
    f: Any = ...
    def __init__(self, f: Any) -> None: ...
    def __call__(self, *args: Any, **kw: Any): ...

map_stream: Any
info: Any
shuffle: Any
select: Any
decode: Any
map: Any
rename: Any
associate: Any
map_dict: Any
to_tuple: Any
map_tuple: Any
batched: Any
unbatched: Any

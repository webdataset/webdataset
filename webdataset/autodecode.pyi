from .checks import checkmember as checkmember, checknotnone as checknotnone
from typing import Any, Optional

check_present: Any
image_extensions: Any


def torch_loads(data: Any):
    ...


def basichandlers(key: Any, data: Any):
    ...


imagespecs: Any


def handle_extension(extensions: Any, f: Any):
    ...


class ImageHandler:
    imagespec: Any = ...
    extensions: Any = ...

    def __init__(self, imagespec: Any, extensions: Any = ...) -> None:
        ...

    def __call__(self, key: Any, data: Any):
        ...


def imagehandler(imagespec: Any):
    ...


def torch_video(key: Any, data: Any):
    ...


def torch_audio(key: Any, data: Any):
    ...


class Continue:
    def __init__(self, key: Any, data: Any) -> None:
        ...


def gzfilter(key: Any, data: Any):
    ...


default_pre_handlers: Any
default_post_handlers: Any


class Decoder:
    only: Any = ...
    handlers: Any = ...

    def __init__(
        self, handlers: Any, pre: Optional[Any] = ..., post: Optional[Any] = ..., only: Optional[Any] = ...
    ) -> None:
        ...

    def decode1(self, key: Any, data: Any):
        ...

    def decode(self, sample: Any):
        ...

    def __call__(self, sample: Any):
        ...

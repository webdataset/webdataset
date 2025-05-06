from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from PIL import Image as PILImage

# Constants
pytorch_weights_only: bool
IMAGE_EXTENSIONS: List[str]

# Type definitions
Sample = Dict[str, Any]
Handler = Callable[[str, bytes], Any]
DecoderList = List[Handler]

# Functions for basic data types
def torch_loads(data: bytes) -> Any: ...
def tenbin_loads(data: bytes) -> Any: ...
def msgpack_loads(data: bytes) -> Any: ...
def npy_loads(data: bytes) -> Any: ...
def npz_loads(data: bytes) -> Dict[str, np.ndarray]: ...
def cbor_loads(data: bytes) -> Any: ...

# Dictionary of decoders
decoders: Dict[str, Callable[[bytes], Any]]

# Basic handlers
def basichandlers(key: str, data: bytes) -> Optional[Any]: ...

# Extension handlers
def call_extension_handler(key: str, data: bytes, f: Callable[[bytes], Any], extensions: List[str]) -> Optional[Any]: ...
def handle_extension(extensions: str, f: Callable[[bytes], Any]) -> Handler: ...

# Image handling
imagespecs: Dict[str, Tuple[str, Optional[str], str]]

class ImageHandler:
    imagespec: str
    extensions: Set[str]

    def __init__(self, imagespec: str, extensions: List[str] = IMAGE_EXTENSIONS) -> None: ...
    def __call__(self, key: str, data: bytes) -> Optional[Union[PILImage.Image, np.ndarray, 'torch.Tensor']]: ...

def imagehandler(imagespec: str, extensions: List[str] = IMAGE_EXTENSIONS) -> ImageHandler: ...

# Video and audio handling
def torch_video(key: str, data: bytes) -> Optional[Tuple['torch.Tensor', 'torch.Tensor', Dict[str, Any]]]: ...
def torch_audio(key: str, data: bytes) -> Optional[Tuple['torch.Tensor', int]]: ...

# Special class for continuing decoding
class Continue:
    key: str
    data: bytes

    def __init__(self, key: str, data: bytes) -> None: ...

def gzfilter(key: str, data: bytes) -> Optional[Continue]: ...

# Default handlers
default_pre_handlers: List[Handler]
default_post_handlers: List[Handler]

# Decoder class and error
class DecodingError(Exception):
    url: Optional[str]
    key: Optional[str]
    k: Optional[str]
    sample: Optional[Sample]

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None,
                 k: Optional[str] = None, sample: Optional[Sample] = None) -> None: ...

class Decoder:
    handlers: List[Handler]
    only: Optional[Set[str]]
    partial: bool

    def __init__(self, handlers: List[Handler], pre: Optional[List[Handler]] = None,
                 post: Optional[List[Handler]] = None, only: Optional[Union[List[str], Set[str], str]] = None,
                 partial: bool = False) -> None: ...
    def decode1(self, key: str, data: bytes) -> Any: ...
    def decode(self, sample: Sample) -> Sample: ...
    def __call__(self, sample: Sample) -> Sample: ...

# Default decoder
default_decoder: Decoder

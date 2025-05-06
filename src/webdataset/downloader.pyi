from typing import Any, Callable, List, Optional, TypeVar, Union

import typer

# Import gopen via absolute import


# Type variables
T = TypeVar('T')

# Main Typer app
app: typer.Typer

# Function type for download functions
DownloadFunc = Callable[[str, str], Any]

def download_file(url: str, filename: str) -> None: ...

def download_with(command: str) -> DownloadFunc: ...

def total_file_size(files: List[str]) -> int: ...

def file_of_tempfile(tempfile: str) -> str: ...

def get_oldest_file(files: List[str]) -> str: ...

class RandomShardDownloader:
    shards: List[str]
    directory: Optional[str]
    nshards: int
    pattern: str
    increment: int
    errors: str
    maxsize: int
    verbose: bool
    download: DownloadFunc

    def __init__(
        self,
        shards: List[str],
        nshards: int,
        *,
        directory: Optional[str] = None,
        pattern: str = "*.{tar,tgz,tar.gz}",
        increment: int = 999999,
        maxsize: int = 999999999999,
        verbose: bool = False,
        download: Optional[Union[str, DownloadFunc]] = None,
        errors: str = "ignore",
    ) -> None: ...

    def list_files(self, inactive: bool = False) -> List[str]: ...
    def set_directory(self, directory: str) -> None: ...
    def update(self) -> None: ...
    def sleep(self, poll: float = 10) -> None: ...
    def update_every(self, poll: float = 10) -> None: ...
    def maybe_remove(self, strategy: str = "oldest") -> bool: ...
    def replace_every(self, poll: float = 60, strategy: str = "oldest") -> None: ...
    def run_job(self, poll: float = 10, mode: str = "update", strategy: str = "oldest") -> None: ...

@app.command()
def random_downloader(
    shards: List[str],
    *,
    directory: Optional[str] = None,
    nshards: int = 10,
    command: Optional[str] = None,
    pattern: str = "*.{tar,tgz,tar.gz}",
    increment: int = 999999,
    maxsize: int = 999999999999,
    njobs: int = 1,
    poll: float = 10,
    mode: str = "update",
    errors: str = "ignore",
    verbose: bool = False,
) -> None: ...

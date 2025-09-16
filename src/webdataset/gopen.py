# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""Open URLs by calling subcommands."""

import os
import re
import sys
from subprocess import PIPE, Popen
from urllib.parse import urlparse
from urllib.request import url2pathname

from . import utils

# global used for printing additional node information during verbose output
info = {}

# If set to True, allows:
# - opening of local files with gopen_file
# - opening of named pipes with gopen_pipe
# - URL rewriting with rewrite_url


class Pipe:
    """Wrapper class for subprocess.Pipe.

    This class looks like a stream from the outside, but it checks
    subprocess status and handles timeouts with exceptions.
    This way, clients of the class do not need to know that they are
    dealing with subprocesses.

    Args:
        *args: Passed to `subprocess.Pipe`
        mode: The mode for opening the pipe.
        timeout: Timeout for closing/waiting.
        ignore_errors: Don't raise exceptions on subprocess errors.
        ignore_status: List of status codes to ignore.
        **kw: Passed to `subprocess.Pipe`
    """

    def __init__(
        self,
        *args,
        mode=None,
        timeout=7200.0,
        ignore_errors=False,
        ignore_status=[],
        **kw,
    ):
        """Create an IO Pipe."""
        self.ignore_errors = ignore_errors
        self.ignore_status = [0] + ignore_status
        self.timeout = timeout
        self.args = (args, kw)
        if mode[0] == "r":
            self.proc = Popen(*args, stdout=PIPE, **kw)
            self.stream = self.proc.stdout
            if self.stream is None:
                raise ValueError(f"{args}: couldn't open")
        elif mode[0] == "w":
            self.proc = Popen(*args, stdin=PIPE, **kw)
            self.stream = self.proc.stdin
            if self.stream is None:
                raise ValueError(f"{args}: couldn't open")
        self.status = None

    def __str__(self):
        """Return a string representation of the Pipe object."""
        return f"<Pipe {self.args}>"

    def check_status(self):
        """Poll the process and handle any errors."""
        status = self.proc.poll()
        if status is not None:
            self.wait_for_child()

    def wait_for_child(self):
        """Check the status variable and raise an exception if necessary."""
        verbose = int(os.environ.get("GOPEN_VERBOSE", 0))
        if self.status is not None and verbose:
            # print(f"(waiting again [{self.status} {os.getpid()}:{self.proc.pid}])", file=sys.stderr)
            return
        self.status = self.proc.wait()
        if verbose:
            print(
                f"pipe exit [{self.status} {os.getpid()}:{self.proc.pid}] {self.args} {info}",
                file=sys.stderr,
            )
        if self.status not in self.ignore_status and not self.ignore_errors:
            raise IOError(f"{self.args}: exit {self.status} (read) {info}")

    def read(self, *args, **kw):
        """Wrap stream.read and checks status.

        Args:
            *args: Arguments to pass to stream.read
            **kw: Keyword arguments to pass to stream.read

        Returns:
            The result of stream.read
        """
        result = self.stream.read(*args, **kw)
        self.check_status()
        return result

    def write(self, *args, **kw):
        """Wrap stream.write and checks status.

        Args:
            *args: Arguments to pass to stream.write
            **kw: Keyword arguments to pass to stream.write

        Returns:
            The result of stream.write
        """
        result = self.stream.write(*args, **kw)
        self.check_status()
        return result

    def readLine(self, *args, **kw):
        """Wrap stream.readLine and checks status.

        Args:
            *args: Arguments to pass to stream.readLine
            **kw: Keyword arguments to pass to stream.readLine

        Returns:
            The result of stream.readLine
        """
        result = self.stream.readLine(*args, **kw)
        self.status = self.proc.poll()
        self.check_status()
        return result

    def close(self):
        """Wrap stream.close, wait for the subprocess, and handle errors."""
        if not self.stream.closed:
            self.stream.close()
            self.status = self.proc.wait(self.timeout)
            self.wait_for_child()

    def __enter__(self):
        """Context handler."""
        return self

    def __exit__(self, etype, value, traceback):
        """Context handler."""
        self.close()

    def __del__(self):
        """Close the stream upon delete.

        This is a fallback for when users can't use context managers.
        We catch all exceptions since __del__ should never raise exceptions
        during garbage collection.
        """
        try:
            self.close()
        except Exception:
            # Silently ignore exceptions in __del__ as per Python recommendations
            # We can't reliably log here during garbage collection
            pass


def set_options(obj, timeout=None, ignore_errors=None, ignore_status=None, handler=None):
    """Set options for Pipes.

    This function can be called on any stream. It will set pipe options only
    when its argument is a pipe.

    Args:
        obj: Any kind of stream
        timeout: Desired timeout
        ignore_errors: Desired ignore_errors setting
        ignore_status: Desired ignore_status setting
        handler: Desired error handler

    Returns:
        True if options were set, False otherwise
    """
    if not isinstance(obj, Pipe):
        return False
    if timeout is not None:
        obj.timeout = timeout
    if ignore_errors is not None:
        obj.ignore_errors = ignore_errors
    if ignore_status is not None:
        obj.ignore_status = ignore_status
    if handler is not None:
        obj.handler = handler
    return True


def gopen_file(url, mode="rb", bufsize=8192):
    """Open a file.

    This works for local files; path names only.

    Args:
        url: URL to be opened
        mode: Mode to open it with
        bufsize: Requested buffer size

    Returns:
        An opened file object
    """
    if url.startswith("file:"):
        url = re.sub(r"^file://?", "", url)
    return open(url, mode)


def gopen_pipe(url, mode="rb", bufsize=8192):
    """Use gopen to open a pipe.

    This function deliberately uses shell=True with the pipe URL to enable shell command
    execution directly from URLs. This is an intentional design feature that allows users
    to construct processing pipelines using shell commands via the pipe: URL scheme.
    The purpose is to enable flexible data processing directly within data loading pipelines.

    Note: This feature requires careful use with trusted input sources only, as it will
    execute arbitrary shell commands specified in the URL.

    Args:
        url: A pipe: URL
        mode: Desired mode
        bufsize: Desired buffer size

    Returns:
        A Pipe object

    Raises:
        ValueError: If the mode is unknown
    """
    assert url.startswith("pipe:")
    if utils.enforce_security:
        raise ValueError("gopen_pipe: unsafe_gopen is False, cannot open pipe URLs")
    cmd = url[5:]
    if mode[0] in ["r", "w"]:
        return Pipe(
            cmd,
            mode=mode,
            shell=True,
            bufsize=bufsize,
            ignore_status=[141],
        )  # skipcq: BAN-B604
    else:
        raise ValueError(f"{mode}: unknown mode")


def gopen_curl(url, mode="rb", bufsize=8192):
    """Open a URL with `curl`.

    Args:
        url: URL (usually, http:// etc.)
        mode: File mode
        bufsize: Buffer size

    Returns:
        A Pipe object

    Raises:
        ValueError: If the mode is unknown
    """
    if mode[0] == "r":
        cmd_args = ["curl", "--connect-timeout", "0.5", "--retry", "1", "--retry-delay", "1", "-f", "-s", "-L", url]
        return Pipe(
            cmd_args,
            mode=mode,
            bufsize=bufsize,
            ignore_status=[141, 23],
        )
    elif mode[0] == "w":
        cmd_args = ["curl", "-f", "-s", "-X", "PUT", "-L", "-T", "-", url]
        return Pipe(
            cmd_args,
            mode=mode,
            bufsize=bufsize,
            ignore_status=[141, 26],
        )
    else:
        raise ValueError(f"{mode}: unknown mode")


def gopen_htgs(url, mode="rb", bufsize=8192):
    """Open a URL with `curl`.

    Args:
        url: URL (usually, http:// etc.)
        mode: File mode
        bufsize: Buffer size

    Returns:
        A Pipe object

    Raises:
        ValueError: If the mode is write or unknown
    """
    if mode[0] == "r":
        url = re.sub(r"(?i)^htgs://", "gs://", url)
        cmd_args = ["curl", "-s", "-L", url]
        return Pipe(
            cmd_args,
            mode=mode,
            bufsize=bufsize,
            ignore_status=[141, 23],
        )
    elif mode[0] == "w":
        raise ValueError(f"{mode}: cannot write")
    else:
        raise ValueError(f"{mode}: unknown mode")


def gopen_hf(url, mode="rb", bufsize=8192):
    """Open a URL with `curl`.

    Args:
        url: URL (usually, hf:// etc.)
        mode: File mode
        bufsize: Buffer size

    Returns:
        A Pipe object

    Raises:
        ValueError: If the mode is write or unknown
    """
    from huggingface_hub import HfFileSystem, get_token, hf_hub_url

    if mode[0] == "r":
        resolved_path = HfFileSystem().resolve_path(url)
        http_url = hf_hub_url(
            repo_id=resolved_path.repo_id,
            filename=resolved_path.path_in_repo,
            repo_type=resolved_path.repo_type,
            revision=resolved_path.revision,
        )
        token = get_token()
        cmd_args = [
            "curl",
            "--connect-timeout",
            "30",
            "--retry",
            "30",
            "--retry-delay",
            "2",
            "-f",
            "-s",
            "-L",
            "-H",
            f"Authorization:Bearer {token}",
            http_url,
        ]
        return Pipe(
            cmd_args,
            mode=mode,
            bufsize=bufsize,
            ignore_status=[141, 23],
        )
    elif mode[0] == "w":
        raise ValueError(f"{mode}: cannot write")
    else:
        raise ValueError(f"{mode}: unknown mode")


def gopen_gsutil(url, mode="rb", bufsize=8192):
    """Open a URL with `gsutil`.

    Args:
        url: URL (usually, gs:// etc.)
        mode: File mode
        bufsize: Buffer size

    Returns:
        A Pipe object

    Raises:
        ValueError: If the mode is unknown
    """
    if mode[0] == "r":
        cmd_args = ["gsutil", "cat", url]
        return Pipe(
            cmd_args,
            mode=mode,
            bufsize=bufsize,
            ignore_status=[141, 23],
        )
    elif mode[0] == "w":
        cmd_args = ["gsutil", "cp", "-", url]
        return Pipe(
            cmd_args,
            mode=mode,
            bufsize=bufsize,
            ignore_status=[141, 26],
        )
    else:
        raise ValueError(f"{mode}: unknown mode")


def gopen_ais(url, mode="rb", bufsize=8192):
    """Open a URL with `ais`.

    Args:
        url: URL (usually, ais:// etc.)
        mode: File mode
        bufsize: Buffer size

    Returns:
        A Pipe object

    Raises:
        ValueError: If the mode is unknown
    """
    if mode[0] == "r":
        cmd_args = ["ais", "get", url, "-"]
        return Pipe(
            cmd_args,
            mode=mode,
            bufsize=bufsize,
            ignore_status=[141, 23],
        )
    elif mode[0] == "w":
        cmd_args = ["ais", "put", "-", url]
        return Pipe(
            cmd_args,
            mode=mode,
            bufsize=bufsize,
            ignore_status=[141, 26],
        )
    else:
        raise ValueError(f"{mode}: unknown mode")


def gopen_error(url, *args, **kw):
    """Raise a value error.

    Args:
        url: URL
        *args: Other arguments
        **kw: Other keywords

    Raises:
        ValueError: Always raised with the URL and a message
    """
    raise ValueError(f"{url}: no gopen handler defined")


"""A dispatch table mapping URL schemes to handlers."""
gopen_schemes = dict(
    __default__=gopen_error,
    pipe=gopen_pipe,
    http=gopen_curl,
    https=gopen_curl,
    ais=gopen_ais,
    sftp=gopen_curl,
    ftps=gopen_curl,
    scp=gopen_curl,
    gs=gopen_gsutil,
    htgs=gopen_htgs,
    hf=gopen_hf,
)

if "USE_AIS_FOR" in os.environ:
    for proto in os.environ["USE_AIS_FOR"].split(":"):
        gopen_schemes[proto] = gopen_ais


def rewrite_url(url):
    """Rewrite the URL based on environment variables.

    This function checks for URL rewrite rules defined in the GOPEN_REWRITE
    environment variable and applies them to the given URL. The rewrite rules
    allow for flexible modification of URLs before they are processed by the
    gopen system.

    The GOPEN_REWRITE environment variable should contain one or more rewrite
    rules separated by semicolons. Each rule consists of two parts separated
    by an equals sign: a pattern to match at the start of the URL, and a
    replacement string.

    Format of GOPEN_REWRITE:
    GOPEN_REWRITE="pattern1=replacement1;pattern2=replacement2;..."

    The function applies these rules in order, stopping at the first match.
    If a match is found, the pattern is replaced with the corresponding
    replacement at the start of the URL.

    The GOPEN_VERBOSE environment variable can be set to control logging.
    If GOPEN_VERBOSE is set to a non-zero value, the function will print
    information about any URL rewrites that occur.

    Note: This function performs basic URL rewriting without validation.
    Validation of the resulting URL for security concerns (such as path traversal)
    is the responsibility of the caller.

    Args:
        url (str): The original URL to potentially rewrite.

    Returns:
        str: The rewritten URL if a rewrite rule matches, otherwise the original URL.

    Example:
        If GOPEN_REWRITE is set to "http://old.com/=http://new.com/;ftp://=http://"
        and the input URL is "http://old.com/file.txt", the function will return
        "http://new.com/file.txt".
    """
    name = "GOPEN_REWRITE"
    verbose = int(os.environ.get("GOPEN_VERBOSE", 0))
    if name not in os.environ:
        return url
    if utils.enforce_security:
        raise ValueError("rewrite_url: unsafe_gopen is False, cannot rewrite URLs using environment variables")
    for r in os.environ[name].split(";"):
        k, v = r.split("=", 1)
        nurl = re.sub("^" + k, v, url)
        if nurl != url:
            if verbose:
                print(f"GOPEN REWRITE {url} -> {nurl}")
            return nurl
    return url


def gopen(url, mode="rb", bufsize=8192, **kw):
    """Open the URL using various schemes and protocols.

    This function provides a unified interface for opening resources specified by URLs,
    supporting multiple schemes and protocols. It uses the `gopen_schemes` dispatch table
    to handle different URL schemes.

    Built-in support is provided for the following schemes:
    - pipe: for opening named pipes
    - file: for local file system access
    - http, https: for web resources
    - sftp, ftps: for secure file transfer
    - scp: for secure copy protocol

    When no scheme is specified in the URL, it is treated as a local file path.

    Environment Variables:
    - GOPEN_VERBOSE: Set to a non-zero value to enable verbose logging of file operations.
      Format: GOPEN_VERBOSE=1
    - USE_AIS_FOR: Specifies which cloud storage services should use AIS (and its cache) for access.
      Format: USE_AIS_FOR=aws:gs:s3
    - GOPEN_BUFFER: Sets the buffer size for file operations (in bytes).
      Format: GOPEN_BUFFER=8192

    Args:
        url (str): The source URL or file path to open.
        mode (str): The mode for opening the resource. Only "rb" (read binary) and "wb" (write binary) are supported.
        bufsize (int): The buffer size for file operations. Default is 8192 bytes.
        **kw: Additional keyword arguments to pass to the underlying open function.

    Returns:
        file-like object: An opened file-like object for the specified resource.

    Raises:
        ValueError: If an unsupported mode is specified.
        Other exceptions may be raised depending on the specific handler used for the URL scheme.

    Note:
    - For stdin/stdout operations, use "-" as the URL.
    - The function applies URL rewriting based on the GOPEN_REWRITE environment variable before processing.
    """
    global fallback_gopen
    verbose = int(os.environ.get("GOPEN_VERBOSE", 0))
    if verbose:
        print("GOPEN", url, info, file=sys.stderr)
    assert mode in ["rb", "wb"], mode
    if url == "-":
        if mode == "rb":
            return sys.stdin.buffer
        elif mode == "wb":
            return sys.stdout.buffer
        else:
            raise ValueError(f"unknown mode {mode}")
    url = rewrite_url(url)
    pr = urlparse(url)
    if pr.scheme == "":
        if utils.enforce_security:
            raise ValueError("gopen: unsafe_gopen is False, cannot open local files")
        bufsize = int(os.environ.get("GOPEN_BUFFER", -1))
        return open(url, mode, buffering=bufsize)
    if pr.scheme == "file":
        if utils.enforce_security:
            raise ValueError("gopen: unsafe_gopen is False, cannot open local files")
        bufsize = int(os.environ.get("GOPEN_BUFFER", -1))
        return open(url2pathname(pr.path), mode, buffering=bufsize)
    handler = gopen_schemes["__default__"]
    handler = gopen_schemes.get(pr.scheme, handler)
    return handler(url, mode, bufsize, **kw)


def reader(url, **kw):
    """Open url with gopen and mode "rb".

    Args:
        url: Source URL
        **kw: Other keywords forwarded to gopen

    Returns:
        An opened file-like object in read mode
    """
    return gopen(url, "rb", **kw)

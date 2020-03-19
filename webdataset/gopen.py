#!/usr/bin/python
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""
Open URLs by calling subcommands.
"""

__all__ = "gopen scheme_to_command".split()

import sys
from subprocess import PIPE, Popen
from urllib.parse import urlparse


class Pipe:
    """Wrapper class for subprocess.Pipe.

    This class looks like a stream from the outside, but it checks
    subprocess status and handles timeouts with exceptions.
    This way, clients of the class do not need to know that they are
    dealing with subprocesses.

    :param *args: passed to `subprocess.Pipe`
    :param **kw: passed to `subprocess.Pipe`
    :param timeout: timeout for closing/waiting
    :param ignore_errors: don't raise exceptions on subprocess errors
    :param ignore_status: list of status codes to ignore
    """

    def __init__(
        self,
        *args,
        mode=None,
        timeout=3600.0,
        ignore_errors=False,
        ignore_status=[],
        **kw,
    ):
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

    def check_status(self):
        """Calls poll on the process and handles any errors."""
        self.status = self.proc.poll()
        self.handle_status()

    def handle_status(self):
        """Checks the status variable and raises an exception if necessary."""
        if self.status is not None:
            self.status = self.proc.wait()
            if self.status not in self.ignore_status and not self.ignore_errors:
                raise Exception(f"{self.args}: exit {self.status} (read)")

    def read(self, *args, **kw):
        """Wraps stream.read and checks status."""
        result = self.stream.read(*args, **kw)
        self.check_status()
        return result

    def write(self, *args, **kw):
        result = self.stream.write(*args, **kw)
        self.check_status()
        return result

    def readLine(self, *args, **kw):
        """Wraps stream.readLine and checks status."""
        result = self.stream.readLine(*args, **kw)
        self.status = self.proc.poll()
        self.check_status()
        return result

    def close(self):
        """Wraps stream.close, waits for the subprocess, and handles errors."""
        self.stream.close()
        self.status = self.proc.wait(self.timeout)
        self.handle_status()

    def __enter__(self):
        """Context handler."""
        return self

    def __exit__(self, etype, value, traceback):
        """Context handler."""
        self.close()


def gopen_objectio(url, mode="rb", bufsize=8192):
    import objectio

    return objectio.gopen(url, mode)


def gopen_file(url, mode="rb", bufsize=8192):
    return open(url, mode)


def gopen_pipe(url, mode="rb", bufsize=8192):
    assert url.startswith("pipe:")
    cmd = url[5:]
    if mode[0] == "r":
        return Pipe(
            cmd, mode=mode, shell=True, bufsize=bufsize, ignore_status=[141],
        )  # skipcq: BAN-B604
    elif mode[0] == "w":
        return Pipe(
            cmd, mode=mode, shell=True, bufsize=bufsize, ignore_status=[141],
        )  # skipcq: BAN-B604
    else:
        raise ValueError(f"{mode}: unknown mode")


def gopen_curl(url, mode="rb", bufsize=8192):
    if mode[0] == "r":
        cmd = f"curl -s -L '{url}'"
        return Pipe(
            cmd, mode=mode, shell=True, bufsize=bufsize, ignore_status=[141, 23],
        )  # skipcq: BAN-B604
    elif mode[0] == "w":
        cmd = f"curl -s -L -T - '{url}'"
        return Pipe(
            cmd, mode=mode, shell=True, bufsize=bufsize, ignore_status=[141, 26],
        )  # skipcq: BAN-B604
    else:
        raise ValueError(f"{mode}: unknown mode")


gopen_schemes = dict(
    __default__=gopen_objectio,
    pipe=gopen_pipe,
    http=gopen_curl,
    https=gopen_curl,
    sftp=gopen_curl,
    ftps=gopen_curl,
    scp=gopen_curl,
)


def gopen(url, mode="rb", bufsize=8192):
    global fallback_gopen
    assert mode in ["rb", "wb"], mode
    if url == "-":
        if mode == "rb":
            return sys.stdin.buffer
        elif mode == "wb":
            return sys.stdout.buffer
        else:
            raise ValueError(f"unknown mode {mode}")
    pr = urlparse(url)
    if pr.scheme == "":
        return open(url, mode)
    if pr.scheme == "file":
        return open(pr.path, mode)
    handler = gopen_schemes["__default__"]
    handler = gopen_schemes.get(pr.scheme, handler)
    return handler(url, mode, bufsize)


def reader(url):
    return gopen(url, "rb")

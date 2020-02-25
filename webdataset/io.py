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
    """

    def __init__(self, *args, timeout=3600.0, ignore_errors=False, **kw):
        self.ignore_errors = ignore_errors
        self.timeout = timeout
        self.proc = Popen(*args, **kw)
        self.args = (args, kw)
        self.stream = self.proc.stdout
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
            if self.status != 0 and not self.ignore_errors:
                raise Exception(f"{self.args}: exit {self.status} (read)")

    def read(self, *args, **kw):
        """Wraps stream.read and checks status."""
        result = self.stream.read(*args, **kw)
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


def gopen(url, mode="rb", handler=None, bufsize=8192):
    if mode[0] != "r":
        raise ValueError(f"{mode}: unsupported mode (only read supported)")
    pr = urlparse(url)
    if pr.scheme == "":
        return open(url, "rb")
    if pr.scheme == "file":
        return open(pr.path, "rb")
    if pr.scheme == "pipe":
        return Pipe(
            url[5:], stdout=PIPE, shell=True, bufsize=bufsize
        )  # skipcq: BAN-B604

    import objio

    return objio.gopen(url, "rb")


def command_pipe(handler):
    return lambda url: gopen(url, "rb", handler)


def reader(url):
    return gopen(url, "rb")

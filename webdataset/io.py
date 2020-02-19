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

import os
import sys
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import yaml

verbose = int(os.environ.get("WDS_GOPEN_VERBOSE", 0))

scheme_to_command = {
    "file": "dd if='{url}' bs=4M",
    "gs": "gsutil cat '{}'",
    "s3": "s3 cp '{url}' -",
    "az": "azure storage blob --container-name {netloc} " +
            "--name {filename} --file /dev/stdout",
    "http": "curl --fail -L -s '{url}' --output -",
    "https": "curl --fail -L -s '{url}' --output -"
}


def load_schemes():
    scheme_path = os.environ.get(
        "WDS_SCHEMES", "~/.wds-schemes.yml:./wds-schemes.yml").split(":")
    for fname in scheme_path:
        fname = os.path.expanduser(fname)
        if os.path.exists(fname):
            if verbose:
                print(f"# loading {fname}", file=sys.stderr)
            with open(fname) as stream:
                updates = yaml.load(stream)
                scheme_to_command.update(updates)


class Pipe(object):
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
        assert self.stream is not None
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

    def __exit__(self, type, value, traceback):
        """Context handler."""
        self.close()


def gopen(url, mode, handler=None, bufsize=8192):
    assert mode[0] == "r"
    pr = urlparse(url)
    if pr.scheme == "":
        pr = urlparse("file:" + url)
    if handler is None:
        handler = scheme_to_command.get(pr.scheme)
    if handler is None:
        raise ValueError(f"{url}: no handler found")
    variables = dict(pr._asdict(),
                     url=url,
                     filename=os.path.basename(pr.path),
                     dirname=os.path.dirname(pr.path),
                     abspath=os.path.abspath(pr.path) if pr.scheme == "file" else None)
    handler = handler.format(**variables)
    if verbose:
        sys.stderr.print(f"# {handler}", file=sys.stderr)
    return Pipe(handler, stdout=PIPE, shell=True, bufsize=bufsize)


def command_pipe(handler):
    return lambda url: gopen(url, "read", handler)


def reader(url):
    return gopen(url, "read")


load_schemes()

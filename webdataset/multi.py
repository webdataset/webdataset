#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""An alternative to DataLoader using ZMQ.

This implements MultiLoader, an alternative to DataLoader when torch
is not available. Subprocesses communicate with the loader through
ZMQ, provided for high performance multithreaded queueing.
"""

import os
import multiprocessing as mp
import pickle
import uuid
import weakref

import zmq

the_protocol = pickle.HIGHEST_PROTOCOL

all_pids = weakref.WeakSet()


class EOF:
    """A class that indicates that a data stream is finished."""

    def __init__(self, **kw):
        """Initialize the class with the kw as instance variables."""
        self.__dict__.update(kw)


def reader(dataset, sockname, index, num_workers):
    """Read samples from the dataset and send them over the socket.

    :param dataset: source dataset
    :param sockname: name for the socket to send data to
    :param index: index for this reader, using to indicate EOF
    """
    global the_protocol
    os.environ["WORKER"] = str(index)
    os.environ["NUM_WORKERS"] = str(num_workers)
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(sockname)
    for sample in dataset:
        data = pickle.dumps(sample, protocol=the_protocol)
        sock.send(data)
    sock.send(pickle.dumps(EOF(index=index)))
    sock.close()


class MultiLoader:
    """Alternative to PyTorch DataLoader based on ZMQ."""

    def __init__(
        self, dataset, workers=4, verbose=False, nokill=False, prefix="/tmp/_multi-"
    ):
        """Create a MultiLoader for a dataset.

        This creates ZMQ sockets, spawns `workers` subprocesses, and has them send data
        to the socket.

        :param dataset: source dataset
        :param workers: number of workers
        :param verbose: report progress verbosely
        :param nokill: don't kill old processes when restarting (allows multiple loaders)
        :param prefix: directory prefix for the ZMQ socket
        """
        self.dataset = dataset
        self.workers = workers
        self.verbose = verbose
        self.pids = []
        self.socket = None
        self.ctx = zmq.Context.instance()
        self.nokill = nokill
        self.prefix = prefix

    def kill(self):
        """kill."""
        for pid in self.pids:
            if pid is None:
                continue
            print("killing", pid)
            pid.kill()
            pid.join(1.0)
        self.pids = []
        if self.socket is not None:
            print("closing", self.socket)
            self.socket.close()
        self.socket = None

    def __iter__(self):
        """Return an iterator over this dataloader."""
        if not self.nokill:
            self.kill()
        self.sockname = "ipc://" + self.prefix + str(uuid.uuid4())
        self.socket = self.ctx.socket(zmq.PULL)
        self.socket.bind(self.sockname)
        if self.verbose:
            print("#", self.sockname)
        self.pids = [None] * self.workers
        for index in range(self.workers):
            args = (self.dataset, self.sockname, index, self.workers)
            self.pids[index] = mp.Process(target=reader, args=args)
        all_pids.update(self.pids)
        for pid in self.pids:
            pid.start()
        count = 0
        while self.pids.count(None) < len(self.pids):
            data = self.socket.recv()
            sample = pickle.loads(data)
            if isinstance(sample, EOF):
                if self.verbose:
                    print("# subprocess finished", sample.index)
                self.pids[sample.index].join(1.0)
                self.pids[sample.index] = None
            else:
                yield sample
            count += 1

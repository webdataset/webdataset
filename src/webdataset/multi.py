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

import multiprocessing as mp
import os
import pickle
import uuid
import weakref

import zmq

the_protocol = pickle.HIGHEST_PROTOCOL

all_pids = weakref.WeakSet()


class EOF:
    """Indicate that a data stream is finished.

    This class is used to signal the end of a data stream in the MultiLoader.

    Args:
        **kw: Arbitrary keyword arguments to be set as instance variables.
    """

    def __init__(self, **kw):
        """Initialize the EOF instance with keyword arguments.

        Args:
            **kw: Arbitrary keyword arguments to be set as instance variables.
        """
        self.__dict__.update(kw)


def reader(dataset, sockname, index, num_workers):
    """Read samples from the dataset and send them over the socket.

    This function is run in a separate process to read data from the dataset
    and send it to the main process through a ZMQ socket.

    Args:
        dataset: The source dataset to read samples from.
        sockname (str): The name of the ZMQ socket to send data to.
        index (int): The index of this reader process.
        num_workers (int): The total number of worker processes.

    Returns:
        None
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
    """Alternative to PyTorch DataLoader based on ZMQ.

    This class creates a multi-process data loader using ZMQ for inter-process
    communication, providing an alternative to PyTorch's DataLoader.

    Args:
        dataset: The source dataset to load data from.
        workers (int): Number of worker processes to spawn. Defaults to 4.
        verbose (bool): Whether to report progress verbosely. Defaults to False.
        nokill (bool): If True, don't kill old processes when restarting. Defaults to False.
        prefix (str): Directory prefix for the ZMQ socket. Defaults to "/tmp/_multi-".
    """

    def __init__(
        self, dataset, workers=4, verbose=False, nokill=False, prefix="/tmp/_multi-"
    ):
        """Initialize the MultiLoader instance.

        Args:
            dataset: The source dataset to load data from.
            workers (int): Number of worker processes to spawn. Defaults to 4.
            verbose (bool): Whether to report progress verbosely. Defaults to False.
            nokill (bool): If True, don't kill old processes when restarting. Defaults to False.
            prefix (str): Directory prefix for the ZMQ socket. Defaults to "/tmp/_multi-".
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
        """Kill all worker processes and close the ZMQ socket."""
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
        """Return an iterator over this dataloader.

        This method sets up the ZMQ socket, spawns worker processes, and yields
        samples from the dataset.

        Yields:
            Sample: A sample from the dataset.

        Raises:
            None
        """
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

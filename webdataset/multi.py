import os

import sys
import warnings

import torch.multiprocessing as mp
from torch.utils.data import IterableDataset

import webdataset as wds
import queue

from . import filters


verbose = int(os.environ.get("MULTIDATASET_VERBOSE", 0))


def D(*args):
    if verbose:
        print(" ".join([str(x) for x in args]), file=sys.stderr)


def _parallel_job(i, f, input_queue, output_queue, extra=(), kw={}):
    D("job", i, "start", os.environ.get("OMP_NUM_THREADS"))
    while True:
        try:
            arg = input_queue.get(False)
        except queue.Empty:
            break
        D("job", i, "url", arg)
        count = 0
        for sample in f(arg, *extra, **kw):
            assert isinstance(sample, (tuple, dict))
            output_queue.put(sample)
            count += 1
    D("job", i, "done")


class MultiDatasetIterator(IterableDataset):
    def __init__(
        self,
        shard_fn=None,
        factory=None,
        workers=4,
        input_size=0,
        output_size=100,
        daemon=True,
        threads=1,
    ):
        num_threads = int(os.environ.get("OMP_NUM_THREADS", "999999"))
        if num_threads >= 8:
            warnings.warn(
                f"you need to set environment variale OMP_NUM_THREADS to something small"
            )
        self.input_queue = mp.Queue(input_size)
        self.output_queue = mp.Queue(output_size)
        self.jobs = []
        urls = shard_fn()
        if len(urls) == 0:
            warnings.warn(f"no urls in MultiDatasetIterator")
            return
        D("urls", urls)
        for url in urls:
            self.input_queue.put(url)
        D("workers")
        for i in range(workers):
            job = mp.Process(
                target=_parallel_job,
                args=(i, factory, self.input_queue, self.output_queue),
                daemon=daemon,
            )
            self.jobs.append(job)
            job.start()
        D("started")

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.jobs) > 0:
            try:
                result = self.output_queue.get(True, timeout=0.1)
                assert isinstance(result, (tuple, dict))
                return result
            except (FileNotFoundError, queue.Empty):
                for job in self.jobs:
                    job.join(0.1)
                self.jobs = [job for job in self.jobs if job.is_alive()]
                D("timeout", len(self.jobs))
        raise StopIteration()

    def terminate(self):
        for job in self.jobs:
            job.terminate()
            job.join()
        self.jobs = []


class MultiDataset(IterableDataset, wds.Pipeline):
    def __init__(
        self,
        dataset,
        workers=4,
        input_size=0,
        output_size=10000,
        daemon=False,
        nominal=None,
    ):
        wds.Pipeline.__init__(self)
        self.kw = dict(
            shard_fn=dataset.shard_fn,
            factory=dataset.samples,
            input_size=input_size,
            output_size=output_size,
            daemon=daemon,
            workers=workers,
        )
        self.nominal = nominal

    def __iter__(self):
        D("iter called")
        src = MultiDatasetIterator(**self.kw)
        return filters.pipeline(src, *self.pipeline)

    def __len__(self):
        return self.nominal

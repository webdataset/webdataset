import os

import sys
import warnings
import time

import torch
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset

import webdataset as wds
import queue

from . import filters


verbose = int(os.environ.get("MULTIDATASET_VERBOSE", 0))

timeout = float(os.environ.get("MULTIDATASET_TIMEOUT", 0.1))


def omp_warning():
    num_threads = int(os.environ.get("OMP_NUM_THREADS", "999999"))
    if num_threads >= 8:
        warnings.warn(f"set environment variale OMP_NUM_THREADS to something small")


def D(*args):
    if verbose:
        print(" ".join([str(x) for x in args]), file=sys.stderr)


def maybe_copy(a, pin_memory):
    if isinstance(a, torch.Tensor):
        if pin_memory:
            return a.pin_memory()
        else:
            return torch.new_tensor(a)
    else:
        return a


def copy_and_delete_tensors(sample, pin_memory=True):
    if isinstance(sample, (list, tuple)):
        result = tuple(maybe_copy(a, pin_memory) for a in sample)
        for a in sample:
            del a
    else:
        result = {k: maybe_copy(a, pin_memory) for k, a in sample.items()}
        for _, a in sample.items():
            del a
    return result


def _parallel_job(dataset, i, n, output_queue):
    D("job", i, "started")
    dataset.shard_selection = lambda x: x[i::n]
    for sample in dataset:
        output_queue.put(sample)
    D("job", i, "waiting")
    while output_queue.qsize() > 0:
        time.sleep(1.0)
    output_queue.close()
    D("job", i, "done", dataset.sample_urls)


class MultiDatasetIterator(IterableDataset):
    def __init__(
        self, dataset=None, workers=4, output_size=100, pin_memory=True,
    ):
        IterableDataset.__init__(self)
        omp_warning()
        self.output_queue = mp.Queue(output_size)
        self.pin_memory = pin_memory
        self.jobs = []
        for i in range(workers):
            job = mp.Process(
                target=_parallel_job,
                args=(dataset, i, workers, self.output_queue),
                daemon=True,
            )
            self.jobs.append(job)
            job.start()
        D("started")

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.jobs) > 0:
            try:
                result = self.output_queue.get(True, timeout=timeout)
                assert isinstance(result, (tuple, list, dict))
                result = copy_and_delete_tensors(result)
                return result
            except queue.Empty:
                D("queue empty")
                if all(job.exitcode is not None for job in self.jobs):
                    break
        D("all done")
        for job in self.jobs:
            job.join()
        D("all joined")
        raise StopIteration()

    def terminate(self):
        for job in self.jobs:
            job.terminate()
            time.sleep(1.0)
            job.join()


class MultiDataset(IterableDataset, wds.Pipeline):
    def __init__(
        self, dataset, workers=4, output_size=10000, nominal=None, pin_memory=True,
    ):
        wds.Pipeline.__init__(self)
        D("dataset", dataset)
        self.kw = dict(dataset=dataset, workers=workers, output_size=output_size, pin_memory=pin_memory,)
        self.nominal = nominal

    def __iter__(self):
        D("iter called")
        src = MultiDatasetIterator(**self.kw)
        return filters.pipeline(src, *self.pipeline)

    def __len__(self):
        return self.nominal

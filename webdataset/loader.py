#!/usr/bin/python
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of webloader (see TBD).
# See the LICENSE file for licensing terms (BSD-style).
#

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import logging
import time
import collections
import math
from functools import wraps
import numpy as np
import warnings

#from past.utils import old_div

from torch.utils.data import Dataset

from webdataset import WebDataset

try:
    from torch.utils.data import IterableDataset
except:
    class IterableDataset(object): pass

try:
    from torch import Tensor as TorchTensor
except:
    class TorchTensor(object): pass

try:
    from numpy import ndarray
except:
    class ndarray(object): pass

def curried(f):
    """A decorator for currying functions in the first argument."""
    @wraps(f)
    def wrapper(*args, **kw):
        def g(x):
            return f(x, *args, **kw)
        return g
    return wrapper


def compose2(f, g):
    """Compose two functions, g(f(x))"""
    return lambda x: g(f(x))


def compose(*args):
    """Compose a sequence of functions (left-to-right)"""
    return reduce(compose2, args)


def pipeline(source, *args):
    """Write an input pipeline; first argument is source, rest are filters."""
    if len(args) == 0:
        return source
    return compose(*args)(source)

def getfirst(a, keys, default=None):
    assert isinstance(a, dict)
    if isinstance(keys, str):
        keys = keys.split(";")
    for k in keys:
        result = a.get(k)
        if result is not None:
            return result
    return default

def parse_field_spec(fields):
    if isinstance(fields, str):
        fields = fields.split()
    return [field.split(";") for field in fields]

@curried
def extract(data, *fields):
    """Extract the given fields and return a tuple.
    """
    for sample in data:
        yield [getfirst(sample, f) for f in fields]

@curried
def transform(data, f=None):
    """Map entire samples using the given function.

    :param data: iterator
    :param f: function from samples to samples
    :returns: iterator over transformed samples

    """

    if f is None:
        def f(x): return x
    for sample in data:
        result = f(sample)
        if isinstance(sample, dict) and isinstance(result, dict):
            result["__key__"] = sample.get("__key__")
        yield result

@curried
def shuffle(data, bufsize=1000, initial=100):
    """Shuffle the data in the stream.

    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.

    :param data: iterator
    :param bufsize: buffer size for shuffling
    :returns: iterator

    """
    assert initial <= bufsize
    buf = []
    startup = True
    for sample in data:
        if len(buf) < bufsize:
            buf.append(next(data))
        k = pyr.randint(0, len(buf) - 1)
        sample, buf[k] = buf[k], sample
        if startup and len(buf) < initial:
            buf.append(sample)
            continue
        startup = False
        yield sample
    for sample in buf:
        yield sample


def list2dict(l):
    return {i:v for i,v in enumerate(l)}

def dict2list(d):
    n = max(d.keys())+1
    return [d[i] for i in range(n)]

def samples_to_batch(samples, combine_tensors=True, combine_scalars=True, expand=False):
    """Take a collection of samples (dictionaries) and create a batch.

    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.

    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict

    """
    if len(samples) == 0:
        return None
    if isinstance(samples[0], (tuple, list)):
        samples = [list2dict(x) for x in samples]
        return dict2list(samples_to_batch(samples,
            combine_tensors=combine_tensors,
            combine_scalars=combine_scalars,
            expand=expand))
    if expand:
        return samples_to_batch_expanded(samples)
    result = {k: [] for k in list(samples[0].keys())}
    for i in range(len(samples)):
        for k in list(result.keys()):
            result[k].append(samples[i][k])
    for k in list(result.keys()):
        if combine_tensors and isinstance(result[k][0], np.ndarray):
            shapeset = set(tuple(a.shape) for a in result[k])
            assert len(shapeset) == 1, shapeset
            result[k] = np.array(result[k])
        elif combine_tensors and "torch" in str(type(result[k][0])):
            import torch
            shapeset = set(tuple(a.shape) for a in result[k])
            assert len(shapeset) == 1, shapeset
            if isinstance(result[k][0], torch.Tensor):
                result[k] = torch.stack(result[k])
        elif combine_scalars != False and isinstance(result[k][0], (int, float)):
            if combine_scalars == "torch":
                import torch
                result[k] = torch.as_tensor(result[k])
            else:
                result[k] = np.array(result[k])
    return result


@curried
def batched(data, batch_size=20, combine_tensors=True, combine_scalars=True, partial=True, expand=False):
    """Create batches of the given size.

    :param data: iterator
    :param batch_size: target batch size
    :param tensors: automatically batch lists of ndarrays into ndarrays
    :param partial: return partial batches
    :returns: iterator

    """
    batch = []
    for sample in data:
        if len(batch) >= batch_size:
            yield samples_to_batch(batch,
                                         combine_tensors=combine_tensors,
                                         combine_scalars=combine_scalars,
                                         expand=expand)
            batch = []
        batch.append(sample)
    if len(batch) == 0:
        return
    elif len(batch) == batch_size or partial:
        yield samples_to_batch(batch,
                                     combine_tensors=combine_tensors,
                                     combine_scalars=combine_scalars,
                                     expand=expand)


@curried
def batchedbuckets(data, batch_size=5, scale=1.8, seqkey="image", batchdim=1):
    """List-batch input samples into similar sized batches.

    :param data: iterator of samples
    :param batch_size: target batch size
    :param scale: spacing of bucket sizes
    :param seqkey: input key to use for batching
    :param batchdim: input dimension to use for bucketing
    :returns: batches consisting of lists of similar sequence lengths

    """
    buckets = {}
    for sample in data:
        seq = sample[seqkey]
        l = seq.shape[batchdim]
        r = int(math.floor(old_div(math.log(l), math.log(scale))))
        batched = buckets.get(r, {})
        for k, v in list(sample.items()):
            if k in batched:
                batched[k].append(v)
            else:
                batched[k] = [v]
        if len(batched[seqkey]) >= batch_size:
            batched["_bucket"] = r
            yield batched
            batched = {}
        buckets[r] = batched
    for r, batched in list(buckets.items()):
        if batched == {}:
            continue
        batched["_bucket"] = r
        yield batched

def concat(sources, maxepoch=1):
    """Concatenate multiple sources, usually for test sets.

    :param sources: list of iterators
    :param maxepochs: number of epochs (default=1)
    :returns: iterator

    """
    count = 0
    for source in sources:
        for sample in source:
            if maxepoch is not None and "__epoch__" in sample:
                if sample["__epoch__"] >= maxepoch:
                    break
            sample = dict(sample)
            sample["__count__"] = count
            yield sample
            count += 1


def objhash(obj):
    import hashlib
    obj = pickle.dumps(obj, -1)
    m = hashlib.md5()
    m.update(obj)
    return m.hexdigest()

@curried
def patched(data, patches, maxpatches=10000):
    """Patch a dataset with another dataset.

    Patches are stored in memory; for larger patch sizes, use diskpatched.

    :param patches: iterator yielding patch samples
    :param maxpatches: maximum number of patches to load
    :returns: iterator

    """
    patchdict = {}
    for i, sample in enumerate(patches):
        key = sample["__key__"]
        assert key not in patchdict, "{}: repeated key".format(key)
        assert i < maxpatches, "too many patches; increase maxpatches="
        patchdict[key] = sample
    for sample in data:
        key = sample["__key__"]
        return patchdict.get(key, sample)

@curried
def unique(data, key, rekey=False, skip_missing=False, error=True):
    """Ensure that data is unique in the given key.

    :param key: sample key to be made unique
    :param rekey: if True, use the hash value as the new key
    """
    finished = set()
    for sample in data:
        assert key in sample
        ident = objhash(sample.get(key))
        if ident in finished:
            if error:
                raise Exception("duplicate key")
            else:
                continue
        finished.add(ident)
        if rekey:
            sample["__key__"] = ident
        yield sample

def tonumpy(dtype=None, transpose=True):
    """Curried function to convert to NumPy.

    :param dtype: target dtype (Default value = None)
    :param transpose: transpose from PyTorch to NumPy conventions (Default value = True)

    """
    def f(a):
        """

        :param a: 

        """
        import numpy as np
        if isinstance(a, TorchTensor):
            if a.ndim == 3 and a.shape[0] in [3, 4]:
                a = a.permute(1, 2, 0)
            elif a.ndim == 4 and a.shape[1] in [3, 4]:
                a = a.transpose(0, 2, 3, 1)
            return a.detach().cpu().numpy()
        else:
            return a
    return f

def totorch(dtype=None, device="cpu", transpose=True):
    """Curried conversion to PyTorch.

    :param dtype: desired dtype, None=auto (Default value = None)
    :param device: desired target device (Default value = "cpu")
    :param transpose: transpose images to PyTorch conventions (Default value = True)

    """
    def f(a):
        import torch
        import numpy as np
        if isinstance(a, np.ndarray):
            dtype_ = dtype
            if dtype_ is None:
                if a.dtype in [np.float16, np.float32, np.float64]:
                    dtype_ = torch.float32
                elif a.dtype in [np.int16, np.uint16, np.int32, np.int64]:
                    dtype_ = torch.int64
            elif isinstance(dtype_, str):
                dtype_ = getattr(torch, dtype_)
            if a.ndim == 3 and a.shape[2] in [3, 4]:
                a = a.transpose(2, 0, 1)
            elif a.ndim == 4 and a.shape[3] in [3, 4]:
                a = a.transpose(0, 3, 1, 2)
            if device=="numpy":
                return a
            else:
                return torch.as_tensor(a, device=device, dtype=dtype_)
        else:
            return a
    return f

def transform_with(sample, transformers):
    """Transform a list of values using a list of functions.

    :param sample: list of values
    :param transformers: list of functions

    """
    assert not isinstance(sample, dict)
    assert isinstance(sample, (tuple, list))
    if transformers is None or len(transformers) == 0:
        return sample
    result = list(sample)
    ntransformers = len(transformers)
    for i in range(len(sample)):
        f = transformers[i%ntransformers]
        if f is not None:
            result[i] = f(sample[i])
    return result

def transformer(transformers):
    """Curried version of `transform_with`.

    :param transformers: 

    """
    def f(x): return transform_with(x, transformers)
    return f

def listify(x):
    """Turn a value into a list.

    Lists and tuples are turned into lists, everything else is turned
    into a one element list.

    :param x: value to be converted
    :param x): return transform_with(x: 
    :param transformers)return flistify(x: 

    """
    if x is None:
        return None
    elif isinstance(x, (list, tuple)):
        return x
    else:
        return [x]

def make_loader(args, kw, queue, index):
    kw["use_tracker"] = False
    data = WebLoader(*args, **kw)
    for sample in data:
        queue.put(sample)

def maybe_gpu(a, device=None, non_blocking=False):
    if isinstance(a, ndarray):
        import torch
        a = torch.from_numpy(a)
    if isinstance(a, TorchTensor):
        return a.contiguous().to(device=device, non_blocking=non_blocking)
    else:
        return a

def sync_gpu_transfer(device="cuda"):
    def f(source):
        for data in source:
            if isinstance(data, (tuple, list)):
                data = [maybe_gpu(a, device, True) for a in data]
            elif isinstance(data, dict):
                data = {k: maybe_gpu(a, device, True) for k, a in data.items()}
            yield data
    return f

def async_gpu_transfer(device="cuda", inflight=2):
    def f(source):
        q = collections.deque()
        done = False
        while not done:
            while not done and len(q) < inflight:
                try:
                    data = next(source)
                except StopIteration:
                    done = True
                    break
                if isinstance(data, (tuple, list)):
                    data = [maybe_gpu(a, device, True) for a in data]
                elif isinstance(data, dict):
                    data = {k: maybe_gpu(a, device, True) for k, a in data.items()}
                q.append(data)
            yield q.popleft()
            if done and len(q) == 0: break
    return f

def funlist(f):
    if f is None: return f
    assert callable(f) or callable(f[0])
    if callable(f):
        return [f]
    else:
        return f

converter_table = dict(
    torch=totorch(),
    torch_cuda=totorch(device="cuda"),
    torch_np=totorch(device="numpy"), # torch conventions, NumPy representation
    torch_numpy=totorch(device="numpy"), # torch conventions, NumPy representation
    numpy=tonumpy()
)

class WebLoader(object):
    """Iterate over sharded datasets."""
    def __init__(self, dataset,
                 max_batches=int(1e10), max_samples=int(1e10), epochs=1,
                 pipeline = None, fields=None, transforms=None, shuffle=0,
                 batcher=None, batch_size=None, tensor_batches=True, partial_batches=True,
                 batch_transforms=None, converters=None,
                 verbose=False, **kw):
        """Create a WebLoader
        """

        assert not isinstance(dataset, (str, int, list, tuple))
        if not isinstance(dataset, (Dataset, IterableDataset, WebDataset)):
            warnings.warn(f"dataset not instance of Dataset or IterableDataset")
        self.dataset = dataset
        self.max_batches = max_batches
        self.max_samples = max_samples
        self.num_batches = 0
        self.num_samples = 0
        self.num_epochs = 0
        self.epochs = epochs
        if isinstance(pipeline, (list, tuple)):
            pipeline = compose(*pipeline)
        self.pipeline = pipeline
        self.fields = fields
        self.transforms = funlist(transforms)
        self.shuffle = shuffle
        self.batcher = batcher
        self.batch_size = batch_size
        self.tensor_batches = tensor_batches
        self.partial_batches = partial_batches
        self.batch_transforms = funlist(batch_transforms)
        self.converters = funlist(converters)
        self.verbose = verbose
        if kw != {}:
            warnings.warn(f"WebLoader ignoring extra arguments {kw}")
        self.kw = kw

    def raw_iter(self):
        """Iterate over samples."""
        source = iter(self.dataset)
        if self.pipeline is not None:
            source = self.pipeline(source)
        if self.fields is not None:
            source = extract(*self.fields)(source)
        if self.transforms is not None:
            source = transform(transformer(self.transforms))(source)
        if self.shuffle > 1:
            source = shuffle(self.shuffle)(source)
        if self.batcher is not None:
            source = self.batcher(source)
        elif self.batch_size is not None:
            source = batched(self.batch_size,
                             combine_tensors=self.tensor_batches,
                             partial=self.partial_batches)(source)
        for sample in source:
            if self.num_batches >= self.max_batches: break
            if not math.isnan(self.num_samples) and self.num_samples >= self.max_samples: break
            if self.batch_transforms is not None:
                if isinstance(sample, dict):
                    raise ValueError("expect list for batch_transforms; did you specify fields= for WebLoader?")
                sample = transform_with(sample, self.batch_transforms)
            if self.converters is not None:
                if isinstance(sample, dict):
                    raise ValueError("expect list for batch_transforms; did you specify fields= for WebLoader?")
                sample = transform_with(sample, self.converters)
            self.last_sample = sample
            self.num_batches += 1
            try: self.num_samples += len(sample[0])
            except: self.num_samples = math.nan
            yield sample

    def __iter__(self):
        self.num_batches = 0
        self.num_samples = 0
        for epoch in range(self.epochs):
            for sample in self.raw_iter():
                yield sample
            self.num_epochs += 1

    def __len__(self):
        """Return the length of the dataset (the size argument passed on initialization)."""
        return self.batches

multi_pipes = dict(
    sync_gpu_transfer=sync_gpu_transfer(),
    async_gpu_transfer=async_gpu_transfer()
)

def enqueue_samples_from(dataset, queue, subset, loader_kw):
    if subset:
        dataset.subset = subset
    dl = WebLoader(dataset, **loader_kw)
    for sample in dl:
        queue.put(sample)

def joinall(jobs):
    result = []
    for job in jobs:
        if not job.is_alive():
            print(f"{job} finished", file=sys.stderr)
            job.join()
        else:
            result.append(job)
    return result

class MultiWebLoader(object):
    """Multiprocessing version of WebLoader """
    def __init__(self, dataset, num_workers=4, use_torch_mp=False, queue_size=10, multi_pipe=None, **kw):
        """Instantiate multiple WebLoaders in parallel.

        :param urls: input URLs
        :param num_workers: number of subnum_workers to use (Default value = 4)
        :param use_torch_mp: use the Torch version of multiprocessing (Default value = False)
        :param queue_size: number of samples buffered in the queue (Default value = 10)
        :param **kw: other keyword arguments are passed to WebLoader

        """
        self.dataset = dataset
        self.num_workers = num_workers
        self.use_torch_mp = use_torch_mp
        self.queue_size = queue_size
        self.multi_pipe = multi_pipes.get(multi_pipe, multi_pipe)
        assert self.multi_pipe is None or callable(self.multi_pipe)
        self.jobs = None
        self.sampler = None # for compatibility with DataLoader
        self.kw = kw

    def raw_iter(self):
        """Iterate over samples.

        Note that multiple iterators share the same input queue."""
        if self.use_torch_mp:
            import torch.multiprocessing as mp
        else:
            import multiprocessing as mp
        import queue as mpq
        total = 0
        jobs = []
        queue = mp.Queue(self.queue_size)
        try:
            for i in range(self.num_workers):
                subset = (i, self.num_workers)
                args=(self.dataset, queue,  subset, self.kw)
                process = mp.Process(target=enqueue_samples_from, args=args)
                jobs.append(process)
            for job in jobs:
                job.start()
            while len(jobs) > 0:
                try:
                    sample = queue.get(timeout=0.5)
                    total += 1
                    yield sample
                except mpq.Empty:
                    print("timeout")
                jobs = joinall(jobs)
        except FileNotFoundError as exn:
            pass
        finally:
            for job in jobs:
                job.terminate()
                time.sleep(0.01)
                os.kill(job.pid, 15)
                job.join()
            del queue

    def __iter__(self):
        result = self.raw_iter()
        if self.multi_pipe is not None:
            result = self.multi_pipe(result)
        return result

    def __len__(self):
        """ """
        return self.batches

def asdict(l):
    if isinstance(l, dict):
        return l
    return {i: v for i, v in enumerate(l)}

def loader_test(source, nbatches=10, skip=10):
    """Run a test against a loader."""
    for i, sample in enumerate(source):
        if i >= skip-1: break

    start = time.time()
    count = 0
    for i, sample in enumerate(source):
        sample = asdict(sample)
        for xs in sample.values():
            if isinstance(xs, (list, TorchTensor, ndarray)):
                count += len(xs)
                break
        if i >= nbatches-1: break
    finish = time.time()

    delta = finish-start
    print("{:.2f} samples/s {:.2f} batches/s".format(count/delta, nbatches/delta))

    print("Example:")
    sample = asdict(sample)
    for index, a in sorted(sample.items()):
        if isinstance(index, str) and index[0]=="_":
            if isinstance(a, list):
                print(index, a[0], "...")
            else:
                print(index, str(a)[:100], "...")
        elif isinstance(a, TorchTensor):
            print(index, ":", "Tensor", a.shape, a.device, a.dtype, a.min().item(), a.max().item())
        elif isinstance(a, ndarray):
            import numpy as np
            print(index, ":", "ndarray", a.shape, a.dtype, np.amin(a), np.amax(a))
        else:
            print(index, ":", type(a))

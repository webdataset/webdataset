#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of webloader (see TBD).
# See the LICENSE file for licensing terms (BSD-style).
#
from __future__ import unicode_literals

import glob
import pdb
from builtins import range
from imp import reload
from io import open

import numpy as np
from webdataset import WebDataset
from webdataset.loader import WebLoader, MultiWebLoader
from webdataset import loader
import itertools as itt


def test_webloader():
    ds = WebDataset("testdata/sample.tgz", extensions="png;jpg cls");
    dl = WebLoader(ds)
    for i, s in enumerate(ds):
        if i>3: break
        assert isinstance(s[0], np.ndarray)

def test_WebLoader_keys():
    ds = WebDataset("testdata/sample.tgz", extensions=None)
    dl = WebLoader(ds)
    for sample in dl:
        break
    print("sample", sample)
    assert set(["__key__", "png"]) < set(sample.keys()), list(sample.keys())
    assert isinstance(sample["png"], np.ndarray)
    the_key = sample["__key__"]
    for sample in dl :
        break
    assert the_key == sample["__key__"], (the_key, sample["__key__"])

def test_WebLoader_batch():
    import torch
    ds = WebDataset("testdata/sample.tgz", extensions="png cls")
    wl = WebLoader(ds, batch_size=32)
    for sample in wl:
        break
    assert len(sample) == 2
    assert isinstance(sample[0], np.ndarray), sample[0]
    assert isinstance(sample[1], np.ndarray), sample[1]
    assert sample[0].dtype == np.float32, sample[0].dtype
    assert sample[1].dtype == np.int64, sample[1].dtype
    assert len(sample[0].shape) == 4, sample[0].shape
    assert len(sample[1].shape) == 1, sample[1].shape
    assert sample[0].shape[0] == 32, sample[0].shape
    assert sample[1].shape[0] == 32, sample[1].shape
    assert sample[0].shape[3] == 3, sample[0].size()

def test_WebLoader_torch():
    import torch
    ds = WebDataset("testdata/sample.tgz", extensions="png cls")
    wl = WebLoader(ds, batch_size=32, converters=loader.totorch())
    for sample in wl :
        break
    assert len(sample) == 2
    assert isinstance(sample[0], torch.Tensor), sample[0]
    assert isinstance(sample[1], torch.Tensor), sample[1]
    assert sample[0].dtype == torch.float32, sample[0]
    assert sample[1].dtype == torch.int64, sample[1]
    assert len(sample[0].shape) == 4, sample[0].shape
    assert len(sample[1].shape) == 1, sample[1].shape
    assert sample[0].shape[0] == 32, sample[0].shape
    assert sample[1].shape[0] == 32, sample[1].shape
    assert sample[0].shape[1] == 3, sample[0].size()

def webloader(**kw):
    ds = WebDataset("testdata/imagenet-000000.tgz",
                    extensions="__key__ ppm;jpg;jpeg;png cls")
    dl = WebLoader(ds, tensor_batches=False, **kw)
    return dl

def count_samples(source, batches=False, verbose=False, max=100000):
    print("> count_samples")
    total = 0
    for i, sample in enumerate(source):
        print(i, "\t| ".join([repr(x)[:10] for x in sample]))
        if batches:
            total += 1
        else:
            total += len(sample[0])
        assert total < max
    print("< count_samples", total)
    return total

def test_WebLoader_batching_epochs1():
    wl = webloader(max_batches=5, epochs=1, batch_size=10)
    total = count_samples(wl)
    assert total==47, total

def test_WebLoader_batching_epochs2():
    wl = webloader(max_batches=5, epochs=1, batch_size=10, partial_batches=False)
    total = count_samples(wl)
    assert total==40, total

def test_WebLoader_batching_epochs3():
    wl = webloader(epochs=2, batch_size=10)
    total = count_samples(wl)
    assert total==94, total

def test_WebLoader_batching_epochs4():
    wl = webloader(max_batches=8, epochs=2, batch_size=10, partial_batches=False)
    total = count_samples(wl)
    assert total==80, (total, wl.num_samples, wl.num_batches, wl.num_epochs)

def test_WebLoader_batching_epochs5():
    wl = webloader(max_batches=8, epochs=2, batch_size=10, partial_batches=True)
    total = count_samples(wl)
    assert total==77, (total, wl.num_samples, wl.num_batches, wl.num_epochs)

def test_WebLoader_listarg():
    n = 3
    ds = WebDataset(["testdata/imagenet-000000.tgz"]*n, extensions="__key__ ppm;jpg;jpeg;png cls")
    wl = WebLoader(ds, tensor_batches=False)
    total = count_samples(wl, batches=True)
    assert total==n*47, total

def test_loader_test():
    wl = webloader(max_batches=8, epochs=2, batch_size=10, partial_batches=True)
    loader.loader_test(wl)

def test_MultiWebLoader_torch():
    import torch
    ds = WebDataset(["testdata/sample.tgz"]*4, extensions="png cls")
    wl = MultiWebLoader(ds, batch_size=32, converters=loader.totorch(), num_workers=4)
    for sample in wl :
        break
    assert len(sample) == 2
    assert isinstance(sample[0], torch.Tensor), sample[0]
    assert isinstance(sample[1], torch.Tensor), sample[1]
    assert sample[0].dtype == torch.float32, sample[0]
    assert sample[1].dtype == torch.int64, sample[1]
    assert len(sample[0].shape) == 4, sample[0].shape
    assert len(sample[1].shape) == 1, sample[1].shape
    assert sample[0].shape[0] == 32, sample[0].shape
    assert sample[1].shape[0] == 32, sample[1].shape
    assert sample[0].shape[1] == 3, sample[0].size()

def test_MultiWebLoader_epoch():
    import torch
    ds = WebDataset(["testdata/sample.tgz"]*4, extensions="png cls")
    wl = MultiWebLoader(ds, num_workers=4)
    count = 0
    for sample in wl :
        count += 1
    assert count == 4*90, count

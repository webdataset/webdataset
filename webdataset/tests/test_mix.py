import io
import os
import pickle
import time

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from io import StringIO

import webdataset as wds


def test_reader1():
    dataset = wds.DataPipeline(
        wds.SimpleShardList("testdata/imagenet-000000.tgz"),
        wds.tarfile_samples,
    )
    result = list(iter(dataset))
    assert len(result) == 47


def test_rr():
    dataset = wds.DataPipeline(
        wds.SimpleShardList("testdata/imagenet-000000.tgz"),
        wds.tarfile_samples,
    )
    mix = wds.RoundRobin([dataset, dataset])
    result = list(iter(mix))
    assert len(result) == 47 * 2
    dataset = wds.DataPipeline(
        wds.SimpleShardList("testdata/imagenet-000000.tgz"),
        wds.tarfile_samples,
    )
    mix = wds.RoundRobin([dataset, dataset, dataset])
    result = list(iter(mix))
    assert len(result) == 47 * 3
    mix = wds.FluidWrapper(wds.RoundRobin([dataset, dataset, dataset])).shuffle(10)
    result = list(iter(mix))
    assert len(result) == 47 * 3


def test_rs():
    dataset = wds.DataPipeline(
        wds.SimpleShardList("testdata/imagenet-000000.tgz"),
        wds.tarfile_samples,
    )
    mix = wds.RandomMix([dataset, dataset], longest=True)
    result = list(iter(mix))
    assert len(result) == 47 * 2

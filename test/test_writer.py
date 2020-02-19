import io
import os
import subprocess
import sys
from itertools import islice

import numpy as np
import PIL
import simplejson

import webdataset.dataset as wds
from webdataset import writer


def test_writer(tmpdir):
    with writer.TarWriter(f"{tmpdir}/writer.tar") as sink:
        sink.write(dict(__key__="a", txt="hello", cls="3"))
    os.system(f"ls -l {tmpdir}")
    ftype = os.popen(f"file {tmpdir}/writer.tar").read()
    assert "compress" not in ftype, ftype

    ds = wds.WebDataset(f"{tmpdir}/writer.tar")
    for sample in ds:
        break
    assert set(sample.keys()) == set("__key__ txt cls".split())


def test_writer2(tmpdir):
    with writer.TarWriter(f"{tmpdir}/writer.tgz") as sink:
        sink.write(dict(__key__="a", txt="hello", cls="3"))
    os.system(f"ls -l {tmpdir}")
    ftype = os.popen(f"file {tmpdir}/writer.tgz").read()
    assert "compress" in ftype, ftype

    ds = wds.WebDataset(f"{tmpdir}/writer.tgz")
    for sample in ds:
        break
    assert set(sample.keys()) == set("__key__ txt cls".split())

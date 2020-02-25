import os

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
        assert set(sample.keys()) == set("__key__ txt cls".split())
        break


def test_writer2(tmpdir):
    with writer.TarWriter(f"{tmpdir}/writer.tgz") as sink:
        sink.write(dict(__key__="a", txt="hello", cls="3"))
    os.system(f"ls -l {tmpdir}")
    ftype = os.popen(f"file {tmpdir}/writer.tgz").read()
    assert "compress" in ftype, ftype

    ds = wds.WebDataset(f"{tmpdir}/writer.tgz")
    for sample in ds:
        assert set(sample.keys()) == set("__key__ txt cls".split())
        break


def test_shardwriter(tmpdir):
    def post(fname):
        assert fname is not None

    with writer.ShardWriter(
        f"{tmpdir}/shards-%04d.tar", maxcount=5, post=post, encoder=False
    ) as sink:
        for i in range(50):
            sink.write(dict(__key__=str(i), txt=b"hello", cls=b"3"))

    os.system(f"ls -l {tmpdir}")
    ftype = os.popen(f"file {tmpdir}/shards-0000.tar").read()
    assert "compress" not in ftype, ftype

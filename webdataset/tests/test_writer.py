import os

import numpy as np

import webdataset as wds
from webdataset import writer


def getkeys(sample):
    return set(x for x in sample if not x.startswith("_"))

def test_writer(tmpdir):
    with writer.TarWriter(f"{tmpdir}/writer.tar") as sink:
        sink.write(dict(__key__="a", txt="hello", cls="3"))
    os.system(f"ls -l {tmpdir}")
    ftype = os.popen(f"file {tmpdir}/writer.tar").read()
    assert "compress" not in ftype, ftype

    ds = wds.DataPipeline(
        wds.SimpleShardList(f"{tmpdir}/writer.tar"),
        wds.tarfile_samples,
        wds.decode("rgb")
    )
    for sample in ds:
        assert getkeys(sample) == set("txt cls".split()), getkeys(sample)
        break


def test_writer2(tmpdir):
    with writer.TarWriter(f"{tmpdir}/writer2.tgz") as sink:
        sink.write(dict(__key__="a", txt="hello", cls="3"))
    os.system(f"ls -l {tmpdir}")
    ftype = os.popen(f"file {tmpdir}/writer2.tgz").read()
    assert "compress" in ftype, ftype

    ds = wds.DataPipeline(
        wds.SimpleShardList(f"{tmpdir}/writer2.tgz"),
        wds.tarfile_samples,
        wds.decode("rgb")
    )
    for sample in ds:
        assert getkeys(sample) == set("txt cls".split()), getkeys(sample)
        break


def test_writer3(tmpdir):
    with writer.TarWriter(f"{tmpdir}/writer3.tar") as sink:
        sink.write(dict(__key__="a", pth=["abc"], pyd=dict(x=0)))
    os.system(f"ls -l {tmpdir}")
    os.system(f"tar tvf {tmpdir}/writer3.tar")
    ftype = os.popen(f"file {tmpdir}/writer3.tar").read()
    assert "compress" not in ftype, ftype

    ds = wds.DataPipeline(
        wds.SimpleShardList(f"{tmpdir}/writer3.tar"),
        wds.tarfile_samples,
        wds.decode("rgb")
    )
    for sample in ds:
        assert getkeys(sample) == set("pth pyd".split())
        assert isinstance(sample["pyd"], dict)
        assert sample["pyd"] == dict(x=0)
        assert isinstance(sample["pth"], list)
        assert sample["pth"] == ["abc"]


def test_writer4(tmpdir):
    with writer.TarWriter(f"{tmpdir}/writer4.tar") as sink:
        sink.write(dict(__key__="a", ten=np.zeros((3, 3)), tb=[np.ones(1), np.ones(2)]))
    os.system(f"ls -l {tmpdir}")
    os.system(f"tar tvf {tmpdir}/writer4.tar")
    ftype = os.popen(f"file {tmpdir}/writer4.tar").read()
    assert "compress" not in ftype, ftype

    ds = wds.DataPipeline(
        wds.SimpleShardList(f"{tmpdir}/writer4.tar"),
        wds.tarfile_samples,
        wds.decode(),
    )
    for sample in ds:
        assert getkeys(sample) == set("tb ten".split())
        assert isinstance(sample["ten"], list)
        assert isinstance(sample["ten"][0], np.ndarray)
        assert sample["ten"][0].shape == (3, 3)
        assert isinstance(sample["tb"], list)
        assert len(sample["tb"]) == 2
        assert len(sample["tb"][0]) == 1
        assert len(sample["tb"][1]) == 2
        assert sample["tb"][0][0] == 1.0


def test_writer_pipe(tmpdir):
    with writer.TarWriter(f"pipe:cat > {tmpdir}/writer_pipe.tar") as sink:
        sink.write(dict(__key__="a", txt="hello", cls="3"))
    os.system(f"ls -l {tmpdir}")
    ds = wds.DataPipeline(
        wds.SimpleShardList(f"{tmpdir}/writer_pipe.tar"),
        wds.tarfile_samples,
        wds.decode("rgb")
    )
    for sample in ds:
        assert getkeys(sample) == set("txt cls".split())
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

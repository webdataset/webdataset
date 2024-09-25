import os

import numpy as np

import webdataset as wds
from webdataset import writer


def getkeys(sample):
    return set(x for x in sample if not x.startswith("_"))


def test_writer(tmpdir):
    with writer.TarWriter(f"{tmpdir}/writer.tar") as sink:
        sink.write(
            {
                "__key__": "a",
                "txt": "hello",
                "cls": "3",
                "json": {"a": 1},
                "mp": {"a": 1},
                "npy": np.zeros((3, 3)),
                "npz": dict(a=np.zeros((3, 3))),
                "ppm": np.zeros((3, 3, 3)),
                "jpg": np.zeros((3, 3, 3)),
                "png": np.zeros((3, 3, 3)),
            }
        )
    os.system(f"ls -l {tmpdir}")
    ftype = os.popen(f"file {tmpdir}/writer.tar").read()
    assert "compress" not in ftype, ftype

    ds = wds.DataPipeline(
        wds.SimpleShardList(f"{tmpdir}/writer.tar"),
        wds.tarfile_samples,
        wds.decode("rgb"),
    )
    for sample in ds:
        assert getkeys(sample) == set(
            "txt cls json mp npy npz ppm jpg png".split()
        ), getkeys(sample)
        assert isinstance(sample["json"], dict)
        assert sample["json"] == dict(a=1)
        assert isinstance(sample["mp"], dict)
        assert sample["mp"] == dict(a=1)
        assert isinstance(sample["npy"], np.ndarray)
        assert sample["npy"].shape == (3, 3)
        assert sample["npz"]["a"].shape == (3, 3)
        assert isinstance(sample["ppm"], np.ndarray)
        assert sample["ppm"].shape == (3, 3, 3)
        assert isinstance(sample["jpg"], np.ndarray)
        assert sample["jpg"].shape == (3, 3, 3)
        assert isinstance(sample["png"], np.ndarray)
        assert sample["png"].shape == (3, 3, 3)
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
        wds.decode("rgb"),
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
        wds.decode("rgb"),
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


def test_writer_gz(tmpdir):
    with writer.TarWriter(f"{tmpdir}/writer_gz.tar") as sink:
        sink.write({"__key__": "a", "txt.gz": "x" * 1000 + "\n"})
    os.system(f"tar tvf {tmpdir}/writer_gz.tar")
    assert os.system(f"tar tvf {tmpdir}/writer_gz.tar | grep a.txt.gz | grep 30") == 0

    ds = wds.DataPipeline(
        wds.SimpleShardList(f"{tmpdir}/writer_gz.tar"),
        wds.tarfile_samples,
        wds.decode(),
    )
    for sample in ds:
        print(sample)
        assert getkeys(sample) == set(["txt.gz"])
        assert isinstance(sample["txt.gz"], str)
        assert sample["txt.gz"] == "x" * 1000 + "\n"


def test_writer_pipe(tmpdir):
    with writer.TarWriter(f"pipe:cat > {tmpdir}/writer_pipe.tar") as sink:
        sink.write(dict(__key__="a", txt="hello", cls="3"))
    os.system(f"ls -l {tmpdir}")
    ds = wds.DataPipeline(
        wds.SimpleShardList(f"{tmpdir}/writer_pipe.tar"),
        wds.tarfile_samples,
        wds.decode("rgb"),
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

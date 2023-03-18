import webdataset as wds


def test_shuffle():
    dataset = wds.DataPipeline(
        wds.SimpleShardList(["testdata/imagenet-000000.tgz"] * 3),
        # wds.shuffle(10),
    )
    result = list(iter(dataset))
    assert len(result) == 3


def test_shuffle0():
    dataset = wds.DataPipeline(
        lambda: iter([]),
        wds.shuffle(10),
    )
    result = list(iter(dataset))
    assert len(result) == 0


def test_shuffle1():
    dataset = wds.DataPipeline(
        wds.SimpleShardList(["testdata/imagenet-000000.tgz"]),
        wds.shuffle(10),
    )
    result = list(iter(dataset))
    assert len(result) == 1


def test_detshuffle():
    dataset1 = wds.DataPipeline(
        wds.SimpleShardList("{000000..000999}"),
        wds.detshuffle(10),
    )
    result1 = list(iter(dataset1))
    dataset2 = wds.DataPipeline(
        wds.SimpleShardList("{000000..000999}"),
        wds.detshuffle(10),
    )
    result2 = list(iter(dataset2))
    assert result1 == result2
    result22 = list(iter(dataset2))
    assert result22 != result2
    result12 = list(iter(dataset1))
    assert result12 == result22
    assert dataset2.stage(1).epoch == 1

from webdataset import utils


def test_repeatedly():
    assert len(list(utils.repeatedly(range(3), nepochs=7))) == 21


def test_repeatedly2():
    assert len(list(utils.repeatedly(range(3), nbatches=10))) == 10


def test_repeatedly3():
    assert len(list(utils.repeatedly([[[1, 1], [2, 2]]] * 3, nsamples=10))) == 5

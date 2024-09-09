from webdataset import utils


def test_repeatedly():
    assert len(list(utils.repeatedly(range(3), nepochs=7))) == 21


def test_repeatedly2():
    assert len(list(utils.repeatedly(range(3), nbatches=10))) == 10


def test_repeatedly3():
    assert len(list(utils.repeatedly([[[1, 1], [2, 2]]] * 3, nsamples=10))) == 5


def test_is_iterable():

    sample_string = "sample_string"
    sample_bytes = b"sample_bytes"
    sample_list = [1, 2, 3, 4, 5]
    sample_iter = iter(sample_list)

    def gen_func():
        for i in range(5):
            yield i

    sample_generator = gen_func()

    assert utils.is_iterable(sample_string) is False, "String should not be iterable"
    assert utils.is_iterable(sample_bytes) is False, "Bytes should not be iterable"
    assert utils.is_iterable(sample_list) is True, "List should be iterable"
    assert utils.is_iterable(sample_iter) is True, "Iterator should be iterable"
    assert utils.is_iterable(sample_generator) is True, "Generator should be iterable"
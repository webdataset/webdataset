import random
from functools import partial

import pytest

from webdataset.filters import (
    _batched,
    _extract_keys,
    _map,
    _rename_keys,
    _rsample,
    _select,
    _shuffle,
    _to_tuple,
    _xdecode,
    default_collation_fn,
)


def test_shuffle():
    data = range(100)
    shuffled = list(_shuffle(iter(data), bufsize=20, initial=10))
    assert len(shuffled) == 100
    assert set(shuffled) == set(data)
    assert shuffled != data  # Very low probability of being equal


def test_select():
    data = range(10)
    even = list(_select(data, lambda x: x % 2 == 0))
    assert even == [0, 2, 4, 6, 8]


def test_map():
    data = range(5)
    squared = list(_map(data, lambda x: x**2))
    assert squared == [0, 1, 4, 9, 16]


def test_to_tuple():
    data = [{"a": 1, "b": 2, "c": 3}] * 3
    result = list(_to_tuple(data, "a", "c"))
    assert result == [(1, 3)] * 3


def test_batched():
    import numpy as np

    data = [(i, i) for i in range(10)]
    batches = list(_batched(iter(data), batchsize=3))
    assert len(batches) == 4
    assert len(batches[0]) == 2
    assert (batches[0][0] == np.array([0, 1, 2])).all()
    assert (batches[0][1] == np.array([0, 1, 2])).all()
    assert len(batches[-1]) == 2
    assert (batches[-1][0] == np.array([9])).all()
    assert (batches[-1][1] == np.array([9])).all()


def test_rsample():
    random.seed(42)
    data = range(1000)
    sampled = list(_rsample(data, p=0.1))
    assert 80 < len(sampled) < 120  # Approximately 10% of 1000


@pytest.mark.parametrize(
    "input_data,expected",
    [
        ([{"a": 1, "b": 2}, {"a": 3, "b": 4}], [(1,), (3,)]),
        ([{"x": 10, "y": 20}, {"x": 30, "y": 40}], [(10,), (30,)]),
    ],
)
def test_extract_keys(input_data, expected):
    result = list(_extract_keys(input_data, "a;x"))
    assert result == expected


def test_rename_keys():
    input_data = [{"old_key": "value"}]
    result = list(_rename_keys(input_data, new_key="old_key"))
    assert result == [{"new_key": "value"}]


def test_xdecode():
    input_data = [{"file.txt": b"Hello, World!"}]
    result = list(_xdecode(iter(input_data)))
    assert result == [{"file.txt": "Hello, World!"}]


def test_shuffle_empty_input():
    assert list(_shuffle([])) == []


def test_shuffle_single_item():
    assert list(_shuffle(iter([1]))) == [1]


def test_select_all_filtered():
    assert list(_select(range(10), lambda x: x > 100)) == []


def test_map_exception_handling():
    def raise_for_even(x):
        if x % 2 == 0:
            raise ValueError("Even number")
        return x

    result = list(_map(range(5), raise_for_even, handler=lambda _: True))
    assert result == [1, 3]


def test_to_tuple_missing_key():
    with pytest.raises(ValueError):
        list(_to_tuple([{"a": 1}], "b"))


def test_to_tuple_none_value():
    with pytest.raises(ValueError):
        list(_to_tuple([{"a": None}], "a", none_is_error=True))


def test_batched_exact_multiple():
    data = [(i,) for i in range(9)]
    batches = list(_batched(iter(data), batchsize=3))
    lengths = [len(batch[0]) for batch in batches]
    assert lengths == [3, 3, 3]


def test_batched_no_partial():
    data = [(i,) for i in range(10)]
    batches = list(_batched(data, batchsize=3, partial=False))
    lengths = [len(batch[0]) for batch in batches]
    assert lengths == [3, 3, 3]


def test_rsample_empty_input():
    assert list(_rsample([], p=0.5)) == []


def test_rsample_always_select():
    data = range(100)
    assert list(_rsample(data, p=1.0)) == list(data)


def test_rsample_never_select():
    data = range(100)
    assert list(_rsample(data, p=0.0)) == []


def test_extract_keys_ignore_missing():
    input_data = [{"a": 1}, {"b": 2}]
    result = list(_extract_keys(input_data, "a", "c", ignore_missing=True))
    assert result == [(1,), ()]


def test_extract_keys_with_dots():
    input_data = [{"a": 1}, {"b": 2}]
    result = list(_extract_keys(input_data, ".a", ".c", ignore_missing=True))
    assert result == [(1,), ()]


def test_extract_keys_duplicate_error():
    input_data = [{"a": 1, "a_extra": 2}]
    with pytest.raises(ValueError):
        list(_extract_keys(input_data, "a*", duplicate_is_error=True))


def test_rename_keys_must_match():
    input_data = [{"old_key": "value"}]
    with pytest.raises(ValueError):
        list(_rename_keys(input_data, new_key="non_existent_key", must_match=True))


def test_xdecode_must_decode():
    input_data = [{"unknown_extension.xyz": b"data"}]
    with pytest.raises(ValueError):
        list(_xdecode(input_data, must_decode=True))


def test_xdecode_custom_decoder():
    def custom_decode(stream):
        return stream.read().decode("utf-8").upper()

    input_data = [{"file.custom": b"hello"}]
    result = list(_xdecode(input_data, ("*.custom", custom_decode)))
    assert result == [{"file.custom": "HELLO"}]


def test_default_collation_fn_dict():
    data = [{"a": 1}, {"a": 2}, {"a": 3}]
    f = partial(default_collation_fn, combine_scalars=False, combine_tensors=False)
    result = _batched(iter(data), batchsize=2, collation_fn=f)
    assert list(result) == [{"a": [1, 2]}, {"a": [3]}]


def test_default_collation_fn_tuple():
    data = [(1,), (2,), (3,)]
    f = partial(default_collation_fn, combine_scalars=False, combine_tensors=False)
    result = _batched(iter(data), batchsize=2, collation_fn=f)
    assert list(result) == [([1, 2],), ([3],)]


def test_default_collation_fn_scalars():
    data = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
    f = partial(default_collation_fn, combine_scalars=True, combine_tensors=False)
    result = _batched(iter(data), batchsize=2, collation_fn=f)
    for r in result:
        assert set(r.keys()) == {"a"}
        assert len(r["a"]) == 2
        assert list(r["a"]) == [1, 2] or list(r["a"]) == [3, 4]


def test_default_collation_fn_np():
    import numpy as np

    data = [
        {"a": np.array([1])},
        {"a": np.array([1])},
        {"a": np.array([1])},
        {"a": np.array([1])},
    ]
    f = partial(default_collation_fn, combine_scalars=True, combine_tensors=True)
    result = _batched(iter(data), batchsize=2, collation_fn=f)
    for r in result:
        assert set(r.keys()) == {"a"}
        assert isinstance(r["a"], np.ndarray)
        assert r["a"].shape == (2, 1)
        assert list(r["a"][:, 0]) == [1, 1]


def test_default_collation_fn_torch():
    import torch

    data = [
        {"a": torch.tensor([1])},
        {"a": torch.tensor([1])},
        {"a": torch.tensor([1])},
        {"a": torch.tensor([1])},
    ]
    f = partial(default_collation_fn, combine_scalars=True, combine_tensors=True)
    result = _batched(iter(data), batchsize=2, collation_fn=f)
    for r in result:
        assert set(r.keys()) == {"a"}
        assert isinstance(r["a"], torch.Tensor)
        assert r["a"].shape == (2, 1)
        assert list(r["a"][:, 0]) == [1, 1]

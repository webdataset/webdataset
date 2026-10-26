"""Shard boundaries remain intact when a stream error is handled."""

import io
import tarfile

import pytest

from webdataset.tariterators import group_by_keys, tar_file_expander


def make_archive(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as output:
        for name, data in entries:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


@pytest.mark.quick
@pytest.mark.parametrize("truncated", [False, True])
def test_repeated_shard_keeps_boundary_after_read_error(truncated):
    complete = make_archive([("same.txt", b"first"), ("same.cls", b"x" * 2048)])
    first = complete[:1600] if truncated else complete
    errors = []

    def handle(error):
        errors.append(error)
        return True

    # Resampling can repeat a URL. The first read may have been interrupted.
    sources = [{"url": "synthetic:shard", "stream": io.BytesIO(data)} for data in [first, complete]]
    samples = list(group_by_keys(tar_file_expander(sources, handler=handle), handler=handle))
    full_sample = {"__key__": "same", "__url__": "synthetic:shard", "txt": b"first", "cls": b"x" * 2048}
    first_sample = {"__key__": "same", "__url__": "synthetic:shard", "txt": b"first"} if truncated else full_sample
    assert samples == [first_sample, full_sample]
    assert bool(errors) == truncated
    assert all(isinstance(error, tarfile.ReadError) for error in errors)


@pytest.mark.quick
@pytest.mark.parametrize("eof_value", [None, {"boundary": True}])
def test_handled_shard_error_respects_eof_value(eof_value):
    broken = make_archive([("same.txt", b"first"), ("broken.bin", b"x" * 2048)])[:1600]
    sources = [{"url": "synthetic:first", "stream": io.BytesIO(broken)}]
    result = list(tar_file_expander(sources, handler=lambda error: True, eof_value=eof_value))
    expected = [{"fname": "same.txt", "data": b"first", "__url__": "synthetic:first"}]
    if eof_value is not None:
        expected.append(eof_value)
    assert result == expected


@pytest.mark.quick
@pytest.mark.parametrize("raise_error", [False, True])
def test_shard_open_error_does_not_continue_when_handler_stops(raise_error):
    sources = [
        {"url": "synthetic:bad", "stream": io.BytesIO(b"invalid tar")},
        {"url": "synthetic:next", "stream": io.BytesIO(make_archive([("next.txt", b"next")]))},
    ]
    errors = []

    def handle(error):
        errors.append(error)
        if raise_error:
            raise error
        return False

    expanded = tar_file_expander(sources, handler=handle)
    if raise_error:
        with pytest.raises(tarfile.ReadError):
            list(expanded)
    else:
        assert list(expanded) == []
    assert len(errors) == 1

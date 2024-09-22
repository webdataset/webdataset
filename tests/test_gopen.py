import tempfile

import pytest

from webdataset import gopen


@pytest.mark.quick
def test_file():
    with tempfile.TemporaryDirectory() as work:
        with gopen(f"{work}/temp1", "wb") as stream:
            stream.write(b"abc")
        with gopen(f"{work}/temp1", "rb") as stream:
            assert stream.read() == b"abc"


@pytest.mark.quick
def test_pipe():
    with tempfile.TemporaryDirectory() as work:
        with gopen(f"pipe:cat > {work}/temp2", "wb") as stream:
            stream.write(b"abc")
        with gopen(f"pipe:cat {work}/temp2", "rb") as stream:
            assert stream.read() == b"abc"

import tempfile
from webdataset import gopen


def test_file():
    with tempfile.TemporaryDirectory() as work:
        with gopen.gopen(f"{work}/temp1", "wb") as stream:
            stream.write(b"abc")
        with gopen.gopen(f"{work}/temp1", "rb") as stream:
            assert stream.read() == b"abc"


def test_pipe():
    with tempfile.TemporaryDirectory() as work:
        with gopen.gopen(f"pipe:cat > {work}/temp2", "wb") as stream:
            stream.write(b"abc")
        with gopen.gopen(f"pipe:cat {work}/temp2", "rb") as stream:
            assert stream.read() == b"abc"

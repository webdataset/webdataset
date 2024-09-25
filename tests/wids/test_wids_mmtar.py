import io
import os
import tarfile
import time

import pytest

from wids.wids_mmtar import MMIndexedTar, keep_while_reading

# Define the different tarfile types
tarfile_types = [tarfile.USTAR_FORMAT, tarfile.PAX_FORMAT, tarfile.GNU_FORMAT]


# Create a fixture that creates a tarfile in each type
@pytest.fixture(params=tarfile_types)
def create_tarfile(tmpdir, request):
    tar_type = request.param
    tarfile_name = os.path.join(tmpdir, f"testfile.tar")
    with tarfile.open(tarfile_name, "w", format=tar_type) as tar:
        for i in range(5):
            file = io.BytesIO(f"content{i}".encode())
            tarinfo = tarfile.TarInfo(name=f"file{i}")
            tarinfo.size = len(file.getvalue())
            tar.addfile(tarinfo, fileobj=file)
    return tarfile_name


def test_MMIndexedTar(create_tarfile):
    # Test the MMIndexedTar class
    mmindexedtar = MMIndexedTar(create_tarfile)

    # Test __len__
    assert len(mmindexedtar) == 5

    # Test names
    assert set(mmindexedtar.names()) == {f"file{i}" for i in range(5)}

    # Test get_at_index and get_file
    for i in range(5):
        name, data = mmindexedtar.get_at_index(i)
        assert name == f"file{i}"
        assert data == f"content{i}".encode()
        fname, file = mmindexedtar.get_file(i)
        assert fname == name
        assert file.read() == data

    # Test get_by_name and __getitem__
    for i in range(5):
        name, data = mmindexedtar.get_by_name(f"file{i}")
        assert name == f"file{i}"
        assert data == f"content{i}".encode()
        name, data = mmindexedtar[f"file{i}"]
        assert name == f"file{i}"
        assert data == f"content{i}".encode()

    # Test __iter__
    for i, (name, data) in enumerate(mmindexedtar):
        assert name == f"file{i}"
        assert data == f"content{i}".encode()

    mmindexedtar.close()


def test_concurrent_access(tmpdir):
    # Create two temporary files
    temp_file1 = os.path.join(tmpdir, "tempfile1")
    temp_file2 = os.path.join(tmpdir, "tempfile2")
    with open(temp_file1, "w") as f:
        f.write("test")
    with open(temp_file2, "w") as f:
        f.write("test")

    # Open the files with multiple file descriptors
    fd1_1 = os.open(temp_file1, os.O_RDONLY)
    fd1_2 = os.open(temp_file1, os.O_RDONLY)
    fd2 = os.open(temp_file2, os.O_RDONLY)

    delay = 0.0

    # Simulate a sequence of calls to keep_while_reading
    keep_while_reading(temp_file1, fd1_1, "start")
    time.sleep(0.05)  # delay
    keep_while_reading(temp_file1, fd1_2, "start")
    time.sleep(0.05)  # delay
    keep_while_reading(temp_file2, fd2, "start")
    time.sleep(0.05)  # delay
    keep_while_reading(temp_file1, fd1_1, "end", delay)
    time.sleep(0.05)  # delay
    keep_while_reading(temp_file1, fd1_2, "end", delay)
    time.sleep(0.05)  # delay
    keep_while_reading(temp_file2, fd2, "end", delay)

    # Wait for the unlinking_worker_pool to delete the files
    time.sleep(1.0)

    # Check that all files have been deleted
    assert not os.path.exists(temp_file1)
    assert not os.path.exists(temp_file2)

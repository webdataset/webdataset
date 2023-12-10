import os
import tempfile
import time
from functools import partial
from multiprocessing import Pool

from wids.wids_dl import (
    download_and_open,
    download_file,
    keep_most_recent_files,
    recent_downloads,
)

test_download_url = "gs://webdataset/d-tokens/d-tokens-000000.tar"


def test_download_file():
    remote = test_download_url
    with tempfile.TemporaryDirectory() as tmpdir:
        local = os.path.join(tmpdir, "tempfile")

    try:
        download_file(remote, local)
        assert os.path.exists(local)
    finally:
        if os.path.exists(local):
            os.remove(local)


def test_download_and_open():
    remote = test_download_url
    with tempfile.TemporaryDirectory() as tmpdir:
        local = os.path.join(tmpdir, "tempfile")
        try:
            with download_and_open(remote, local) as f:
                data = f.read(10)
                assert data == b"././@PaxHe", data
        finally:
            if os.path.exists(local):
                os.remove(local)


def download_and_read(remote, local, *, n=10):
    stream = download_and_open(remote, local)
    data = stream.read(n)
    stream.close()
    return data


test_download_url2 = "gs://webdataset/d-tokens/d-tokens-000002.tar"


def test_concurrent_download_and_open():
    remote = test_download_url2
    num_processes = 10
    num_downloads = 500

    recent_downloads.clear()

    with Pool(num_processes) as p:
        with tempfile.TemporaryDirectory() as tmpdir:
            local = os.path.join(tmpdir, "test.tar")
            download_func = partial(download_and_read, remote)
            results = p.map(download_func, [local] * num_downloads)

            for result in results:
                assert result == b"././@PaxHe", result

    assert len(recent_downloads) == 0, recent_downloads


def test_keep_most_recent_files():
    nfiles = 50
    size = 73
    nremain = 17
    cutoff = size * nremain + 1
    print("nfiles", nfiles, "nremain", nremain, "cutoff", cutoff)
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(nfiles):
            with open(os.path.join(tmpdir, f"file{i:03d}"), "wb") as f:
                f.write(b"x" * size)
                time.sleep(0.01)

        # Trim to a maxsize of 5000 bytes
        keep_most_recent_files(tmpdir, maxsize=cutoff)

        # Verify that only 50 files remain
        remaining_files = sorted(os.listdir(tmpdir))
        assert len(remaining_files) == nremain

        # Verify that the remaining files are the last 50 files created
        expected_files = [f"file{i:03d}" for i in range(nfiles - nremain, nfiles)]
        assert sorted(remaining_files) == expected_files

import os
import tempfile
from functools import partial
from multiprocessing import Pool

from wids.wids_dl import download_and_open, download_file, recent_downloads

test_download_url = (
    "https://storage.googleapis.com/webdataset/d-tokens/d-tokens-000000.tar"
)


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


test_download_url2 = (
    "https://storage.googleapis.com/webdataset/d-tokens/d-tokens-000002.tar"
)


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

import glob
import multiprocessing
import os
import time

import pytest

from wids.wids_dl import ConcurrentDownloader, SimpleDownloader


class TestSimpleDownloader:
    @pytest.fixture(scope="class")
    def dlr(self):
        return SimpleDownloader(dldir="/tmp/dltest")

    def test_download(self, dlr):
        dlr.download("https://www.google.com", "google.html")
        dlr.download("pipe:cat google.html | grep google > {local}", "google.txt")
        with open("google.txt") as f:
            assert "google" in f.read()


class TestConcurrentDownloader:
    @pytest.fixture(autouse=True)
    def setup(self, tmpdir):
        self.dldir = str(tmpdir.mkdir("dltest"))
        self.dlr = ConcurrentDownloader(dldir=self.dldir)

    def test_download_and_release(self):
        destpath = os.path.join(self.dldir, "google.html")
        value1 = self.dlr.download("https://www.google.com", destpath)
        assert os.path.exists(destpath)

        destpath2 = os.path.join(self.dldir, "google.txt")
        value2 = self.dlr.download(
            "pipe:cat google.html | grep google > {local}", destpath2
        )
        assert os.path.exists(destpath2)

        self.dlr.release(value1)
        assert not os.path.exists(destpath)

        self.dlr.release(value2)
        assert not os.path.exists(destpath2)


class TestConcurrentDownloader:
    @pytest.fixture(autouse=True)
    def setup(self, tmpdir):
        self.dldir = str(tmpdir.mkdir("dltest"))

    def test_background_download(self):
        fname = "hello.txt"

        def background_download(step1, step2, sleep):
            destpath = os.path.join(self.dldir, fname)
            print(os.getpid(), "start", destpath)
            pid = os.getpid()
            dlr = ConcurrentDownloader()
            step1.wait()
            print(pid, "step1")
            resultpath = dlr.download(
                "pipe:sleep %d; echo 'hello world' > {local}" % sleep, destpath
            )
            print("files:", glob.glob(os.path.join(self.dldir, "*")))
            assert os.path.exists(resultpath)
            assert os.path.exists(destpath)
            step2.wait()
            print(pid, "step2")
            dlr.release(resultpath)
            print(pid, "done")

        proc1_step1 = multiprocessing.Event()
        proc1_step2 = multiprocessing.Event()
        proc2_step1 = multiprocessing.Event()
        proc2_step2 = multiprocessing.Event()

        proc1 = multiprocessing.Process(
            target=background_download, args=(proc1_step1, proc1_step2, 1)
        )
        proc1.start()
        proc2 = multiprocessing.Process(
            target=background_download, args=(proc2_step1, proc2_step2, 1)
        )
        proc2.start()

        try:
            destpath = os.path.join(self.dldir, fname)
            proc1_step1.set()
            assert not os.path.exists(destpath), destpath
            time.sleep(2)
            assert os.path.exists(destpath), destpath
            assert not os.path.exists(destpath + ".dl")
            assert os.stat(destpath).st_nlink == 2

            proc2_step1.set()
            assert os.path.exists(destpath)
            time.sleep(2)
            assert os.path.exists(destpath)
            assert not os.path.exists(destpath + ".dl")
            assert os.stat(destpath).st_nlink == 3

            proc1_step2.set()
            time.sleep(0.1)
            assert os.path.exists(destpath)
            assert not os.path.exists(destpath + ".dl")
            assert os.stat(destpath).st_nlink == 2

            proc2_step2.set()
            time.sleep(0.1)
            assert not os.path.exists(destpath + ".dl")
            assert not os.path.exists(destpath)

        finally:
            proc1.terminate()
            proc1.join()
            proc2.terminate()
            proc2.join()

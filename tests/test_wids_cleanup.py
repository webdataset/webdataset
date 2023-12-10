import os
import tempfile
import time

from wids.wids_cleanup import keep_most_recent_files


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
        keep_most_recent_files(tmpdir + "/*", maxsize=cutoff, debug=True)

        # Verify that only nremain files remain
        remaining_files = sorted(os.listdir(tmpdir))
        assert len(remaining_files) == nremain

        # Verify that the remaining files are the last 50 files created
        expected_files = [f"file{i:03d}" for i in range(nfiles - nremain, nfiles)]
        assert sorted(remaining_files) == expected_files

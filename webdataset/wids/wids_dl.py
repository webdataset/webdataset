import fcntl
import os
import urllib


class ULockFile:
    """A simple locking class. We don't need any of the third
    party libraries since we rely on POSIX semantics for linking
    below anyway."""

    def __init__(self, path):
        self.lockfile_path = path
        self.lockfile = None

    def __enter__(self):
        self.lockfile = open(self.lockfile_path, "w")
        fcntl.flock(self.lockfile.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        fcntl.flock(self.lockfile.fileno(), fcntl.LOCK_UN)
        self.lockfile.close()
        self.lockfile = None


def pipe_download(remote, local):
    """Perform a download for a pipe: url."""
    assert remote.startswith("pipe:")
    cmd = remote[5:]
    cmd = cmd.format(local=local)
    assert os.system(cmd) == 0, "Command failed: %s" % cmd


default_cmds = {
    "posixpath": "cp {url} {local}",
    "s3": "aws s3 cp {url} {local}",
    "gs": "gsutil cp {url} {local}",
    "http": "wget {url} -O {local}",
    "https": "wget {url} -O {local}",
    "ftp": "wget {url} -O {local}",
    "ftps": "wget {url} -O {local}",
    "file": "cp {url} {local}",
    "pipe": pipe_download,
}


def cleanup_downloads(dldir):
    """Iterate through the filenames in dldir, looking for file names of the form
    "something._1234_extra". Here, "something" is a base file name, 1234 is a PID,
    and "extra" is an optional extra string. If PID 1234 is not running, then the
    file is deleted. In addition, if there are no hardlinks remaining to the base
    ("something" in this example), then the base is also deleted."""

    # get the list of files in the download directory
    files = os.listdir(dldir)
    # iterate through the files
    for fname in files:
        # split the file name into base and pid
        base, rest = fname.rsplit("._", 1)
        pid, extra = rest.split("_", 1)
        # check if the pid is running
        try:
            os.kill(int(pid), 0)
        except OSError:
            # the pid is not running, so delete the file
            os.unlink(os.path.join(dldir, fname))
            # check if the base file has any remaining hardlinks
            if os.stat(os.path.join(dldir, base)).st_nlink == 0:
                # no hardlinks, so delete the base file
                os.unlink(os.path.join(dldir, base))


class SimpleDownloader:
    """A simple downloader class that can download files from a variety of
    sources. The downloader is configured with a set of handlers for different
    url schemas. The handlers can be either a string or a callable. If the
    handler is a string, it is treated as a command template and the url and
    local path are substituted into the template. If the handler is a callable,
    it is called with the url and local path as arguments."""

    def __init__(self, **kw):
        """Create a new downloader. The keyword arguments are used to configure
        the handlers."""
        self.handlers = dict(default_cmds)
        self.handlers.update(kw)

    def download(self, remote, local):
        """Download a file from a remote url to a local path. The remote url
        can be a pipe: url, in which case the remainder of the url is treated
        as a command template that is executed to perform the download."""

        # extract the url schema
        if remote.startswith("pipe:"):
            schema = "pipe"
        else:
            schema = urllib.parse.urlparse(remote).scheme
        if schema is None or schema == "":
            schema = "posixpath"
        # get the handler
        handler = self.handlers.get(schema)
        if handler is None:
            raise ValueError("Unknown schema: %s" % schema)
        # call the handler
        if callable(handler):
            handler(remote, local)
        else:
            assert isinstance(handler, str)
            cmd = handler.format(url=remote, local=local)
            assert os.system(cmd) == 0, "Command failed: %s" % cmd
        return local

    def release(self, local):
        """Release a downloaded file. The local is the name of the file to be
        released."""
        os.unlink(local)


def mkresult(fname, ident, extra):
    """Constructs a result path name from a file name, an identifier and an
    extra string. The result path is of the form fname._ident_extra."""
    return f"{fname}._{ident}_{extra}"


def splitresult(result):
    """Splits a result path into its components. The result path is of the form
    fname._ident_extra."""
    path, rest = result.rsplit("._", 1)
    ident, extra = rest.split("_", 1)
    return path, ident, extra


class ConcurrentDownloader:
    """A class that allows multiple processes on a single machine to download files
    concurrently, ensuring that there is only a single download of each file across
    the entire machine. Each client is given a resultpath and needs to call release
    on that path when it is finished with the downloaded file."""

    def __init__(self, keep=False):
        """Create a new concurrent downloader. The dldir is the directory where
        the downloads are stored. If keep is True, the downloaded files are not
        deleted when the caller releases them."""
        self.keep = keep
        self.downloader = SimpleDownloader()

    def download(self, url, destpath, ident=None, extra=""):
        """Download a file from a remote url to a local path. The remote url
        can be a pipe: url, in which case the remainder of the url is treated
        as a command template that is executed to perform the download. The
        dest is the name of the file to be downloaded. The ident is an
        identifier for the client. The extra string is appended to the local path
        as well. All result paths returned by this method must be released by
        calling release."""
        dirname = os.path.dirname(destpath)
        assert os.path.exists(dirname)
        assert os.access(dirname, os.W_OK)
        assert "." not in extra
        ident = ident or os.getpid()
        lockpath = destpath + ".lock"
        lock = ULockFile(lockpath)
        dlpath = destpath + ".dl"
        resultpath = mkresult(destpath, ident, extra)
        for _ in range(10):
            with lock:
                if not os.path.exists(destpath):
                    self.downloader.download(url, dlpath)
                    os.rename(dlpath, destpath)
        if os.path.exists(resultpath):
            os.unlink(resultpath)
        os.link(destpath, resultpath)
        return resultpath

    def release_dest(self, destpath, ident=None, extra=""):
        """Release a downloaded file. The dest is the name of the file to be
        released. The ident is an identifier for the client. The extra string
        is appended to the local path as well."""
        ident = ident or os.getpid()
        lockpath = destpath + ".lock"
        lock = ULockFile(lockpath)
        resultpath = mkresult(destpath, ident, extra)
        with lock:
            os.unlink(resultpath)
            nlink = os.stat(destpath).st_nlink
            if nlink == 1 and not self.keep:
                os.unlink(destpath)
                os.unlink(lockpath)

    def release(self, resultpath):
        """Release a downloaded file. The resultpath is the path returned by
        download."""
        destpath, ident, extra = splitresult(resultpath)
        self.release_dest(destpath, ident=ident, extra=extra)

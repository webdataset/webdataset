import errno
import fcntl
import json
import multiprocessing
import os
import sys
import time
from typing import Dict, List


class ExclusiveLock:
    """A simple non-blocking exclusive lock using fcntl."""

    def __init__(self, lockfile):
        self.lockfile = lockfile

    def try_lock(self):
        try:
            self.lock = open(self.lockfile, "w")
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            else:
                raise

    def release_lock(self):
        self.lock.close()
        os.unlink(self.lockfile)


def write_line_locked(fname, line):
    """Write a line to a file, locking it with fcntl."""
    stream = open(fname, "a")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(line + "\n")
        stream.flush()
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def read_lines_and_clear_locked(fname):
    """Read lines from a file, locking it with fcntl."""
    stream = open(fname, "r+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        lines = stream.readlines()
        stream.seek(0)
        stream.truncate()
        stream.flush()
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()
    return lines


def wait_for_file_contents(fname):
    """Wait for a file to change."""
    while True:
        if os.path.exists(fname) and os.stat(fname).st_size > 0:
            return True
        time.sleep(1.0)


def queue_processor(queue_file):
    """Infinitely process a queue file."""
    lockfile = queue_file + ".lock"
    lock = ExclusiveLock(lockfile)
    if not lock.try_lock():
        return
    while True:
        wait_for_file_contents(queue_file)
        lines = read_lines_and_clear_locked(queue_file)
        for line in lines:
            line = line.strip()
            print("line", line, file=sys.stderr)
            if line == "<<EOF>>":
                return
            yield json.loads(line)


def enqueue_eof(queue_file):
    """Enqueue an EOF."""
    write_line_locked(queue_file, "<<EOF>>\n")


def enqueue_task(queue_file, **kw):
    """Enqueue a task."""
    # make sure the json is single line
    line = json.dumps(kw)
    assert "\n" not in line
    write_line_locked(queue_file, line)


# Code for a file deletion agent.


def remove_dead_processes(pids: List[int]):
    """Remove dead processes from a list of pids."""
    return [pid for pid in pids if os.path.exists(f"/proc/{pid}")]


def file_deletion_job(queue):
    """Keep track of usage and deletion requests.

    This reads messages from a queue and keeps track of open/closed requests.
    Each request contains the PID of the process that opened the file.
    When the last process closes the file, the file is deleted.
    Processes that are not running anymore are also considered closed.
    """
    pids: Dict[str, List[int]] = {}
    for task in queue_processor(queue):
        if task["action"] == "open":
            pids.setdefault(task["fname"], []).append(task["pid"])
        elif task["action"] == "close":
            l = pids[task["fname"]]
            if task["pid"] in l:
                l.remove(task["pid"])
            l = remove_dead_processes(l)
            if len(l) == 0:
                os.unlink(task["fname"])
                del pids[task["fname"]]


def spawn_file_deletion_job(queue):
    """Spawn a file deletion job.

    This spawns a file deletion job and returns the process.
    This can be called from different processes; it uses a lock
    to ensure that only a single file deletion process runs.
    """
    p = multiprocessing.Process(target=file_deletion_job, args=(queue,))
    p.start()
    # wait briefly and collect it if it dies
    time.sleep(1.0)
    if not p.is_alive():
        p.join()
        return None
    else:
        return p


def notify_open_file(queue, fname):
    """Notify the file deletion job that a file is open."""
    enqueue_task(queue, action="open", fname=fname, pid=os.getpid())


def notify_close_file(queue, fname):
    """Notify the file deletion job that a file is closed."""
    enqueue_task(queue, action="close", fname=fname, pid=os.getpid())

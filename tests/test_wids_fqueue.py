import multiprocessing
import os
import random
import string
import tempfile
import time

import pytest

from wids.wids_fqueue import (
    ExclusiveLock,
    enqueue_eof,
    enqueue_task,
    notify_close_file,
    notify_open_file,
    queue_processor,
    read_lines_and_clear_locked,
    remove_dead_processes,
    spawn_file_deletion_job,
    write_line_locked,
)


@pytest.fixture
def temp_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        random_file_name = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=10)
        )
        file_path = os.path.join(temp_dir, random_file_name)
        with open(file_path, "w") as temp_file:
            yield temp_file.name


def test_exclusive_lock(temp_file):
    lock1 = ExclusiveLock(temp_file)
    lock2 = ExclusiveLock(temp_file)

    assert lock1.try_lock() == True
    assert lock2.try_lock() == False

    lock1.release_lock()

    assert lock2.try_lock() == True
    lock2.release_lock()


def test_write_read_line_locked(temp_file):
    write_line_locked(temp_file, "test")
    lines = read_lines_and_clear_locked(temp_file)
    assert lines == ["test\n"]
    assert os.path.exists(temp_file) and os.path.getsize(temp_file) == 0


def test_enqueue_task(temp_file):
    enqueue_task(temp_file, action="open", fname="file1", pid=1234)
    lines = read_lines_and_clear_locked(temp_file)
    assert lines == ['{"action": "open", "fname": "file1", "pid": 1234}\n']
    assert os.path.exists(temp_file) and os.path.getsize(temp_file) == 0


def test_remove_dead_processes():
    assert remove_dead_processes([1, 999999]) == [1]


def worker_enqueue(fname, task):
    time.sleep(
        random.uniform(0.01, 0.1)
    )  # Simulate random short delay before starting the job
    time.sleep(
        random.uniform(0.01, 0.1)
    )  # Simulate random short delay before processing
    enqueue_task(fname, **task)
    time.sleep(
        random.uniform(0.01, 0.1)
    )  # Simulate random short delay after processing


def worker_processor(fname, results):
    processor = queue_processor(fname)
    processed_tasks = list(processor)
    results["tasks"] = processed_tasks


def test_queue_processor_and_enqueue_task(temp_file):
    tasks = [{"action": "open", "fname": f"file{i}", "pid": i} for i in range(50)]
    processes = []

    manager = multiprocessing.Manager()
    results = manager.dict()

    p = multiprocessing.Process(target=worker_processor, args=(temp_file, results))
    p.start()
    processes.append(p)

    time.sleep(
        random.uniform(0.01, 0.1)
    )  # Simulate random short delay before starting the jobs

    for task in tasks:
        p = multiprocessing.Process(target=worker_enqueue, args=(temp_file, task))
        p.start()
        processes.append(p)

    # Enqueue EOF task
    time.sleep(2)
    p = multiprocessing.Process(target=enqueue_eof, args=(temp_file,))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()

    processed_tasks = results["tasks"]

    assert len(processed_tasks) == len(tasks)
    for task in tasks:
        assert task in processed_tasks


def test_file_deletion_job(tmpdir):
    queue_file = os.path.join(tmpdir, "__deletion_queue__")

    # Spawn the file deletion job
    p = spawn_file_deletion_job(queue_file)
    assert p is not None

    # Create a temporary file
    temp_file = os.path.join(tmpdir, "temp_file")
    with open(temp_file, "w") as f:
        f.write("test")

    # Notify the file deletion job that the file is open
    notify_open_file(queue_file, temp_file)

    # Notify the file deletion job that the file is closed
    notify_close_file(queue_file, temp_file)

    # Wait for the file deletion job to delete the file
    time.sleep(1.0)

    # Check that the file has been deleted
    assert not os.path.exists(temp_file)

    # Terminate the file deletion job
    p.terminate()
    p.join()


def worker_open_close_file(queue_file, file, delay):
    notify_open_file(queue_file, file)
    time.sleep(delay)
    notify_close_file(queue_file, file)


def test_multiple_file_deletion_job(tmpdir):
    queue_file = os.path.join(tmpdir, "__deletion_queue__")

    # Spawn the file deletion job
    p = spawn_file_deletion_job(queue_file)
    assert p is not None

    # Create multiple temporary files
    temp_files = [os.path.join(tmpdir, f"tempfile{i}") for i in range(1, 6)]
    for temp_file in temp_files:
        with open(temp_file, "w") as f:
            f.write("test")

    # Create multiple subprocesses that open and close the files
    processes = []
    for _ in range(20):
        temp_file = random.choice(temp_files)
        delay = random.uniform(0.01, 0.1)
        p = multiprocessing.Process(
            target=worker_open_close_file, args=(queue_file, temp_file, delay)
        )
        p.start()
        processes.append(p)

    # Wait for all subprocesses to finish
    for p in processes:
        p.join()

    # Wait for the file deletion job to delete the files
    time.sleep(1.0)

    # Check that all files have been deleted
    for temp_file in temp_files:
        assert not os.path.exists(temp_file)

    # Terminate the file deletion job
    p.terminate()
    p.join()

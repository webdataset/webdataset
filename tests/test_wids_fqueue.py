import pytest
import os
import json
from wids.wids_fqueue import ExclusiveLock, write_line_locked, read_lines_and_clear_locked, enqueue_task, remove_dead_processes

import pytest
import tempfile
from wids.wids_fqueue import ExclusiveLock

import os
import random
import string

@pytest.fixture
def temp_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        random_file_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
        file_path = os.path.join(temp_dir, random_file_name)
        with open(file_path, 'w') as temp_file:
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

import pytest
import multiprocessing
import time
import random
import json
from wids.wids_fqueue import queue_processor, enqueue_task, enqueue_eof

def worker_enqueue(fname, task):
    time.sleep(random.uniform(0.01, 0.1))  # Simulate random short delay before starting the job
    time.sleep(random.uniform(0.01, 0.1))  # Simulate random short delay before processing
    enqueue_task(fname, **task)
    time.sleep(random.uniform(0.01, 0.1))  # Simulate random short delay after processing

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

    time.sleep(random.uniform(0.01, 0.1))  # Simulate random short delay before starting the jobs

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
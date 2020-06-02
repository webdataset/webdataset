import sys
import os
import multiprocessing as mp
import queue


def _parallel_job(i, f, input_queue, output_queue, extra, kw):
    verbose = int(os.environ.get("MULTIEXPAND_VERBOSE", 0))
    while True:
        try:
            arg = input_queue.get(False)
        except queue.Empty:
            break
        if verbose:
            print(f"MULTIEXPAND JOB [{i}]", f, arg, extra, kw, file=sys.stderr)
        count = 0
        for sample in f(arg, *extra, **kw):
            if verbose and count % 1000 == 0:
                print(
                    f"MULTIEXPAND SAMPLE [{i}] @{count} {type(sample)}", file=sys.stderr
                )
            output_queue.put(sample)
            count += 1


def clean_jobs(jobs):
    result = []
    for j in jobs:
        if j.is_alive():
            result.append(j)
        else:
            j.join(0.1)
    return result


def multiexpand(
    inputs, f, extra=(), kw={}, nworkers=4, input_size=0, output_size=1024, daemon=False
):
    verbose = int(os.environ.get("MULTIEXPAND_VERBOSE", 0))
    input_queue = mp.Queue(input_size)
    output_queue = mp.Queue(output_size)
    jobs = [
        mp.Process(
            target=_parallel_job,
            args=(i, f, input_queue, output_queue, extra, kw),
            daemon=daemon,
        )
        for i in range(nworkers)
    ]
    for input in inputs:
        input_queue.put(input)
    for job in jobs:
        job.start()
    count = 0
    while len(jobs) > 0:
        if verbose > 9:
            print("MULTIEXPAND", count, jobs, file=sys.stderr)
        while True:
            try:
                sample = output_queue.get(True, 0.01)
                if verbose and count % 1000 == 0:
                    print("MULTIEXPAND GOT", count, repr(sample)[:50], file=sys.stderr)
            except queue.Empty:
                break
            yield sample
            count += 1
        jobs = clean_jobs(jobs)

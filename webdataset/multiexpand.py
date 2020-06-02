import multiprocessing as mp
import queue


def _parallel_job(i, f, input_queue, output_queue, extra, kw):
    while True:
        try:
            arg = input_queue.get(False)
        except queue.Empty:
            break
        for sample in f(arg, *extra, **kw):
            output_queue.put(sample)


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
    while len(jobs) > 0:
        while True:
            try:
                sample = output_queue.get(False)
            except queue.Empty:
                break
            yield sample
        jobs = clean_jobs(jobs)

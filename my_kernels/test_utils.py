import time

THREAD_RANGE = range(32, 1024, 32)


def timeit(f, args=[], kwargs={}, n=100):
    """
    Get the avg runtime over n runs.
    """
    res = 0
    for _ in range(n):
        t = time.time()
        f(*args, **kwargs)
        res += time.time() - t
    return res / n
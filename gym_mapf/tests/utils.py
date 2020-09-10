import time
import functools


def measure_time(func, *args, **kwargs):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        before = time.time()
        func(*args, **kwargs)
        after = time.time()

        print(f'took {after - before} seconds')

    return new_func

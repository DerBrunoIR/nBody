from numba.cuda import test
from vector import add
from utils import build_kernel
from vector import add_vector_arrays
import numpy as np
from test_utils import timeit, THREAD_RANGE
from numba import cuda


def test_add_vector_arrays():
    arr1 = np.random.randint(0, 10, (2, 2))
    arr2 = np.random.randint(0, 10, (2, 2))
    print(arr1)
    print(arr2)
    kf = build_kernel(add_vector_arrays, len(arr1), 32)
    kf(arr1, arr2)
    print(arr1)


def get_best_tpb():
    arr1 = np.random.randint(0, 10, (2, 2))
    arr2 = np.random.randint(0, 10, (2, 2))
    arr1 = cuda.to_device(arr1)
    arr2 = cuda.to_device(arr2)

    timings = []
    for tpb in THREAD_RANGE:
        kf = build_kernel(add_vector_arrays, len(arr1), tpb)
        dt = timeit(kf, [arr1, arr2])
        print(f"{tpb}: {dt}")
        timings.append((tpb, dt))
    
    best_tpb_dt = min(timings, key=lambda x: x[1])
    print(f"Best: {best_tpb_dt}")
    return best_tpb_dt[0]



if __name__ == "__main__":
    test_add_vector_arrays()
    get_best_tpb()

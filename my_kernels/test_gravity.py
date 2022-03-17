# Tests
from logging import log

from numba.core.errors import NumbaPendingDeprecationWarning, NumbaPerformanceWarning
from numpy.random import sample
from gravity import calculate_accelerations_with_mass
from utils import build_kernel
import numpy as np
from numba import cuda
from test_utils import timeit, THREAD_RANGE


DTYPE = np.float64
N = 100_0



def create_test_arrays(dtype=DTYPE, samples=N):
    """
    Create array full of random bodies.
    """
    positions = np.random.random((samples, 3)) * 100
    masses = np.random.random(samples)
    acc = np.zeros_like(positions, dtype=dtype)
    positions = cuda.to_device(positions)
    masses = cuda.to_device(masses)
    acc = cuda.to_device(acc)
    return positions, masses, acc

def create_specific_test_arrays(dtype=DTYPE):
    """
    Create arrays of 3 special bodies.
    """
    positions = np.array([
        (10,),
        (0,),
        (-10,),
    ], dtype)
    masses = np.full(3, 1, dtype)
    masses = np.array(masses, dtype)
    acc = np.full_like(positions, 0, dtype)

    # print("Positions:", positions)
    # print("Masses:", masses)
    # print("Accelerations:", acc)

    positions = cuda.to_device(positions)
    masses = cuda.to_device(masses)
    acc = cuda.to_device(acc)
    return positions, masses, acc

def find_best_tpb():
    """
    Get the best threads per block amount.
    """
    positions, masses, acc = create_test_arrays()
    timtings = []
    print(" Best TPB CONFIG ".center(100, "#"))

    for i in THREAD_RANGE:
        kf = build_kernel(calculate_accelerations_with_mass, len(positions), tpb=i)
        dt = timeit(kf, [positions, masses, acc])
        timtings.append((dt, i))
        print(f"{i:4}: {dt}")
    print()
    best_tpb = min(timtings, key=lambda e: e[0])
    print(f"Best tpb={best_tpb}")
    print()
    return best_tpb[1]

def speed_test(tpb, n):
    """
    Get the avg time needed by the gravity kernel
    """
    arrays = create_test_arrays(samples=n)
    print(" SPEED TEST ".center(100, "#"))
    kf = build_kernel(calculate_accelerations_with_mass, len(arrays[0]), tpb)
    t = timeit(kf, args=arrays, n=100)
    print(f"f(tpb={tpb}, samples={n}): needed {t}s")
    print()

def value_test(tpb):
    """
    Tests the gravity kernel with 3 Bodies and per hand calculated accelerations.
    """

    def mag(x):
        return np.sqrt(np.sum(x**2))
    
    def array2text(x):
        return "[" + ", ".join([f"{i}" for i in x]) + "]"
    
    positions, masses, accelerations = create_specific_test_arrays()
    print(" VALUE TEST ".center(100, "#"))
    kf = build_kernel(calculate_accelerations_with_mass, 3, tpb)
    print("TPB=", tpb)
    print()
    kf(positions, masses, accelerations)
    for pos, acc in zip(positions.copy_to_host(), accelerations.copy_to_host()):
        print(f"POS={array2text(pos):8}  ACC=", array2text(acc))
    print(f"Expected: ACC=[-0.05, 0, 0.05]")
    print()
        


if __name__ == "__main__":
    import warnings

    # warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
    
    best_tpb = 320
    speed_test(best_tpb, 1000)
    speed_test(best_tpb, 1000)
    # speed_test(best_tpb, 10_000)
    # speed_test(best_tpb, 100_000)
    # speed_test(best_tpb, 1_000_000)
    # value_test(best_tpb)

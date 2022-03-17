from numba import cuda
from render import convert_planets_to_image

if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    from utils import blocks_per_grid
    import time

    np.set_printoptions(threshold=np.inf)
    
    w_start = np.array([-400, -400], dtype=np.float64)
    w_end = np.array([400, 400], dtype=np.float64)
    positions = np.random.randint(w_start, w_end, (1_000, 2))
    masses = np.random.random(positions.shape[0]) ** 2 * 20
    im = np.zeros([2000, 2000, 3], dtype=np.float64)

    w_start = cuda.to_device(w_start)
    w_end = cuda.to_device(w_end)
    positions = cuda.to_device(positions)
    masses = cuda.to_device(masses)
    im = cuda.to_device(im)

    ts = []
    for tpb in range(32, 1025, 32):
        bpg = blocks_per_grid(2000**2, tpb)
        kf = convert_planets_to_image[bpg, tpb]
        avg = 0
        for _ in range(10):
            t = time.time()
            kf(positions, masses,  w_start, w_end, im)
            dt = time.time() - t
            avg += dt
        avg /= 10
        ts.append([tpb, avg])
        print(f"{tpb}: needed {avg}")
    
    mini = min(ts, key=lambda x: x[1])
    print(f"Best: {mini}")
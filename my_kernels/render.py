from itertools import count
from numba import cuda, types
import numpy as np
import math

MAX_DIM = 3
DEFAULT_DTYPE = types.float32
MINIMAL_DISTANCE = 1


@cuda.jit
def convert_planets_to_image(dt, global_positions, global_velocities, global_masses, global_window_start, global_window_end, global_image):
    # thread unic values
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    unique_x = tx + bx * cuda.blockDim.x
   
    # screen size
    width = global_image.shape[1]
    height = global_image.shape[0]

    dim = global_positions.shape[1]
    index = x_idx, y_idx = unique_x % width, unique_x // width
    count_of_bodies = global_positions.shape[0]

    if unique_x >= width * height:
        return

    local_pixel_position = cuda.local.array(MAX_DIM, DEFAULT_DTYPE)
    local_pixel_color = cuda.local.array(3, DEFAULT_DTYPE) # (r, g, b, a)

    for i in range(2):
        step = (global_window_end[i] - global_window_start[i]) / global_image.shape[i]
        local_pixel_position[i] = global_window_start[i] + index[i] * step
    for i in range(4):
        local_pixel_color[i] = 0

    # iterate over all positions
    for n in range(count_of_bodies):
        # load position
        pos = global_positions[n]
        vel = global_velocities[n]
        mass = global_masses[n]

        # DREHUNG
        angle = dt / 1000
        # turn around x axis
        # pos[1] = math.cos(angle) * y - math.sin(angle) * z
        # pos[2] = math.sin(angle) * y + math.cos(angle) * z
        # turn around y axis
        pos[0] = math.cos(angle) * pos[0] + math.sin(angle) * pos[2]
        pos[2] = -math.sin(angle) * pos[0] + math.cos(angle) * pos[2]
        # turn around z axis
        # pos[0] = math.cos(angle) * x - math.sin(angle) * y
        # pos[1] = math.sin(angle) * x + math.cos(angle) * y


        # calculate distance
        squared_rad = MINIMAL_DISTANCE
        for i in range(2):
            squared_rad += (pos[i] - local_pixel_position[i])**2
        squared_rad += 0.002 * pos[2]**2
        r = mass / (15 * squared_rad)

        local_pixel_color[0] += 24 * r
        local_pixel_color[1] += 8 * r
        local_pixel_color[2] += 4* r
        local_pixel_color[3] += 0


    for i in range(global_image.shape[2]):
        # local_pixel_color[i] = min(1, max(local_pixel_color[i], 0))
        local_pixel_color[i] = 1 - min(1, max(local_pixel_color[i], 0))
    
    for i in range(global_image.shape[2]):
        global_image[y_idx][x_idx][i] = local_pixel_color[i]


@cuda.jit
def turn_around_axis(global_positions: np.array, angle: float):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    idx = tx + bx * cuda.blockDim.x

    if idx > global_positions.shape[0]:
        return

    pos = global_positions[idx]

    # DREHUNG
    # turn around x axis
    # pos[1] = math.cos(angle) * y - math.sin(angle) * z
    # pos[2] = math.sin(angle) * y + math.cos(angle) * z
    # turn around y axis
    pos[0] = math.cos(angle) * pos[0] + math.sin(angle) * pos[2]
    pos[2] = -math.sin(angle) * pos[0] + math.cos(angle) * pos[2]
    # turn around z axis
    # pos[0] = math.cos(angle) * x - math.sin(angle) * y
    # pos[1] = math.sin(angle) * x + math.cos(angle) * y



if __name__ == "__main__":
    import numpy as np
    from utils import auto_build

    np.set_printoptions(threshold=np.inf)
    
    positions = [[1, 1, 1]]
    global_positions = cuda.to_device(np.array(positions, dtype=np.float64))
    f = auto_build(turn_around_axis, len(positions), 100)
    for _ in range(10):
        f(global_positions, 1)
        print(global_positions.copy_to_host())
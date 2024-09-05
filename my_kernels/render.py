from itertools import count
from numba import cuda, types
import numpy as np
import math

MAX_DIM = 3
DEFAULT_DTYPE = types.float32
MINIMAL_DISTANCE = 1
MAXIMAL_DISTANCE = 10000


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
        #angle = dt / 1000
        # turn around x axis
        # pos[1] = math.cos(angle) * y - math.sin(angle) * z
        # pos[2] = math.sin(angle) * y + math.cos(angle) * z
        # turn around y axis
        #pos[0] = math.cos(angle) * pos[0] + math.sin(angle) * pos[2]
        #pos[2] = -math.sin(angle) * pos[0] + math.cos(angle) * pos[2]
        # turn around z axis
        # pos[0] = math.cos(angle) * x - math.sin(angle) * y
        # pos[1] = math.sin(angle) * x + math.cos(angle) * y


        # calculate pixel color
        squared_rad = MINIMAL_DISTANCE
        background = (0x38, 0x3e, 0x70)
        color1 = (0xf0, 0x9e, 0x00)
        color2 = (0xf0, 0xd8, 0x00)
        color3 = (255, 0, 0x0b)
        wght = 15
        distr = (0, .5 * wght, 1 * wght, math.inf)

        for i in range(2):
            squared_rad += (pos[i] - local_pixel_position[i])**2
        #squared_rad += 0.002 * pos[2]**2
        w = mass / math.sqrt(squared_rad)

        local_pixel_color[0] += rclamp(background[0], color1[0], distr[0], distr[1], w) + rclamp(color1[0], color2[0], distr[1], distr[2], w) + rsclamp(color2[0], color3[0], distr[2], distr[3], w)
        local_pixel_color[1] += rclamp(background[1], color1[1], distr[0], distr[1], w) + rclamp(color1[1], color2[1], distr[1], distr[2], w) + rsclamp(color2[1], color3[1], distr[2], distr[3], w)
        local_pixel_color[2] += rclamp(background[2], color1[2], distr[0], distr[1], w) + rclamp(color1[2], color2[2], distr[1], distr[2], w) + rsclamp(color2[2], color3[2], distr[2], distr[3], w)

    local_pixel_color[3] = 255 * count_of_bodies

    for i in range(global_image.shape[2]):
        local_pixel_color[i] = min(1, max(local_pixel_color[i] / (255 * count_of_bodies), 0))
    
    for i in range(global_image.shape[2]):
        global_image[y_idx][x_idx][i] = local_pixel_color[i]


@cuda.jit
def clamp(a, b, t):
    # assert 0 <= t <= 1
    return a + (b - a) * min(max(t, 0), 1)


@cuda.jit
def rclamp(a, b, r1, r2, r):
    if r < r1 or r > r2:
        return 0
    rr = (r - r1) / (r2 - r1)
    return clamp(a, b, rr)

@cuda.jit
def rsclamp(a, b, r1, r2, r):
    if r < r1:
        return 0
    if r > r2:
        return b
    return clamp(a, b, (r - r1) / (r2 - r1))





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

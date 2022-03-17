from my_kernels import gravity, render, vector
from my_kernels.utils import auto_build
from PIL import Image
import numpy as np
from numba import cuda


DTYPE = np.float64

class Universum:

    @classmethod
    def from_np_arr(cls, positions, velocities, masses, image_buffer):
        return cls(
            cuda.to_device(positions.astype(DTYPE)),
            cuda.to_device(velocities.astype(DTYPE)),
            cuda.to_device(masses.astype(np.float16)),
            cuda.to_device(image_buffer.astype(np.float16)),
        )

    def __init__(self, positions, velocities, masses, image_buffer) -> None:
        self._positions = positions
        self._velocities = velocities
        self._masses = masses
        self._image_buffer = image_buffer

        self._update_accelerations = auto_build(gravity.calculate_accelerations_with_mass, positions.shape[0], 320)
        self._add_accelerations = auto_build(vector.add_vector_arrays, positions.shape[0], 640)
        self._draw_positions = auto_build(render.convert_planets_to_image, image_buffer.size, 64)
        self._turn_around_axis = auto_build(render.turn_around_axis, positions.shape[0], 128)

        size = np.array(image_buffer.shape[:2], DTYPE)
        start = np.array(-size, DTYPE)
        self._window_start = start
        end = np.array(size, DTYPE)
        self._window_end = end

    def simulate(self, dt):
        self._update_accelerations(self._positions, self._masses, self._velocities)
        self._add_accelerations(self._positions, self._velocities, dt)

    @property
    def positions(self):
        return self._positions.copy_to_host()

    @property
    def velocities(self):
        return self._velocities.copy_to_host()

    @property
    def masses(self):
        return self._masses.copy_to_host()
        
    def image_buffer(self, dt):
        self._draw_positions(dt, self._positions, self._velocities, self._masses, self._window_start, self._window_end, self._image_buffer)
        return (self._image_buffer.copy_to_host() * 255).astype(np.int8)

    def screenshot(self, fp) -> Image:
        Image.fromarray(self.image_buffer).save(fp)
        
    def set_window(self, start, end):
        self._window_start = np.array(start)
        self._window_end = np.array(end)


if __name__ == "__main__":
    
    positions = np.array([[1, 1, 1]])
    velocities = np.array([[0, 0, 0]])
    masses = np.array([1])
    image_buffer = np.full((100, 100), 0)
    uni = Universum.from_np_arr(positions, velocities, masses, image_buffer)
    uni.simulate(1)
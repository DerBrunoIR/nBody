from game import Game, GameObject
from universum import Universum
import pygame
import numpy as np


class ViewRect:
    def __init__(self, dim):
        self.dim = dim
        self.topleft = np.zeros(dim)
        self.botright = np.zeros(dim)

    def set_size(self, size):
        self.topleft += np.array(size)
    
    def move(self, offset):
        self.topleft += np.array(offset)
        self.botright += np.array(offset)
    
    def zoom(self, val):
        dif = (self.botright - self.topleft) / 2
        self.topleft -= dif * val
        self.botright += dif * val

    def size(self):
        return np.abs(self.botright - self.topleft)
    
    def diagonal(self):
        return np.sqrt(np.sum(self.size() ** 2))
    
    def __repr__(self) -> str:
        return f"<ViewRect start={self.topleft}, end={self.botright}>"


class UniverseGameObject(GameObject):
    def __init__(self, universum: Universum) -> None:
        self.universum = universum
        self.view_rect = ViewRect(2)
        self.view_rect.topleft = universum._window_start
        self.view_rect.botright = universum._window_end
        self.tmp_drag_offset = np.zeros(2)
        self.drag_start = np.zeros(2)
        self.offset = np.zeros(2)
        self.pause = False
        self.drag_start = np.zeros(2)
        self.zoom = 0.1
        self.inverse = 1

    def init_game(self, game):
        self.game = game
        

    def draw(self, surface: pygame.Surface):
        start = self.view_rect.topleft + self.tmp_drag_offset + self.offset
        end = self.view_rect.botright + self.tmp_drag_offset + self.offset
        self.universum.set_window(start, end)

        if not self.pause:
            for _ in range(1):
                self.universum.simulate(self.game.dt  * self.inverse)
                # self.universum.simulate(0.01  * self.inverse)
    
        image_buffer = self.universum.image_buffer(self.game.dt)
        image = pygame.image.frombuffer(image_buffer, image_buffer.shape[:2], "RGBA")
        image = pygame.transform.scale(image, self.game.screen_size)
        surface.blit(image, (0, 0))


    def process_event(self, event: pygame.event.Event):
        mouse = np.array(pygame.mouse.get_pos())

        if event.type == pygame.MOUSEBUTTONDOWN:
            # zoom
            if event.button == 4:
                self.view_rect.zoom(-self.zoom)
            elif event.button == 5:
                self.view_rect.zoom(self.zoom)
            
            # drag
            if event.button == 1:
                self.drag_start = mouse
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.offset += self.view_rect.diagonal() * 1/1000 * (self.drag_start - mouse)
                self.tmp_drag_offset = np.zeros(2)
                self.drag_start = np.zeros(2)
        elif event.type == pygame.MOUSEMOTION:
            if np.all(self.drag_start != 0):
                self.tmp_drag_offset = self.view_rect.diagonal() * 1/1000 * (self.drag_start - mouse)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.pause = not self.pause
            
            # screenshot
            if event.key == pygame.K_F2:
                self.universum.screenshot("screenshot.png")
            
            # restart
            if event.key == pygame.K_BACKSPACE:
                self.universum = universe_factory()
            
            # backwards
            if event.key == pygame.K_i:
                self.inverse *= -1


def random_universum_factory(width, height, n, dim):
    m_konst = np.sqrt(width**2+height**2)
    size = (n, dim)
    positions = (np.random.random(size) - 0.5) * 2
    positions *= (n + m_konst/2)
    velocities = (np.random.random(size) - 0.5) * 2
    velocities *= 0
    # velocities = np.zeros_like(positions)
    masses = (np.random.random(n) / 2 + 0.5) * 1000
    image_buffer = np.zeros((width, height, 4))
    return Universum.from_np_arr(positions, velocities, masses, image_buffer)

def universe_factory():
    return random_universum_factory(*[1000, 1000], 20, 2)

def two_body_factory(width, height):
    size = (2, 2)
    positions = np.array([[-500, 0], [500, 0]])
    velocities = np.array([[0, 5],[0, -10]])
    masses = np.array([5000, 1000])
    image_buffer = np.zeros((width, height, 4))
    return Universum.from_np_arr(positions, velocities, masses, image_buffer)

def three_body_factory(width, height):
    size = (3, 2)
    positions = np.array([[-500, 0], [500, 0], [0, 0]])
    velocities = np.array([[0, 16],[0, -16], [0, 0]])
    masses = np.array([20000, 20000, 20000])
    image_buffer = np.zeros((width, height, 4))
    return Universum.from_np_arr(positions, velocities, masses, image_buffer)

def three_body_3d_factory(width, height):
    size = (3, 3)
    positions = np.array([[-500, 0, 0], [500, 0, 0], [0, 0, 0]])
    velocities = np.array([[0, 16, 0],[0, -16, 0], [0, 0, 0]])
    masses = np.array([20000, 20000, 20000])
    image_buffer = np.zeros((width, height, 4))
    return Universum.from_np_arr(positions, velocities, masses, image_buffer)

# TODO not spiraling
def spiral_factory(width, height, n):
    size = (n, 3)
    positions = 10 * min(width, height) * (np.random.random(size) - 0.5) 
    velocities = []
    for pos in positions:
        vel = np.array([-pos[1], pos[0], 0])
        vel = 10 / np.linalg.norm(vel) * vel 
        velocities.append(vel)
    velocities = np.array(velocities)
    masses = np.array([50000] * n)
    image_buffer = np.zeros((width, height, 4))
    assert len(velocities) == len(positions) == len(masses) == n
    assert velocities.shape == positions.shape == size
    return Universum.from_np_arr(positions, velocities, masses, image_buffer)

if __name__ == "__main__":
    res = np.array([1000, 1000], dtype=int)
    game = Game(*res, record=True)
    # game.add_gameobject(UniverseGameObject(universe_factory()))
    # game.add_gameobject(UniverseGameObject(two_body_factory(1000, 1000)))
    game.add_gameobject(UniverseGameObject(three_body_3d_factory(1000, 1000)))
    # game.add_gameobject(UniverseGameObject(three_body_factory(1000, 1000)))
    # game.add_gameobject(UniverseGameObject(spiral_factory(1000, 1000, 500)))
    game.run()

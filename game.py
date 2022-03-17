import pygame
from time import time
import numpy as np
import cv2
from utils import generate_fp


pygame.font.init()
sysfont = pygame.font.SysFont(pygame.font.get_default_font(), 30)

class GameObject:
    def init_game(self, game):
        pass

    def draw(self, surface: pygame.Surface):
        pass

    def process_event(self, event: pygame.event.Event):
        pass


class Game():
    width = 1000
    height = 1000
    running = True
    gameObjects: list[GameObject] = []
    dt = 0
    fps = 0
    surface: pygame.Surface

    def __init__(self, width, height, record=False) -> None:
        self.width, self.height = width, height
        self.surface = pygame.display.set_mode((self.width, self.height))
        self.screen_size = np.array([width, height])
        if record:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vid_name = f"vid_{width}x{height}"
            self._recorder = cv2.VideoWriter(generate_fp("videos", vid_name, "mp4"), fourcc, 15, [width, height])

    def add_gameobject(self, go: GameObject):
        """
        f: function, f(pygame.Surface) -> None
        """
        go.init_game(self)
        self.gameObjects.append(go)

    def _draw_gameObjects(self):
        for go in self.gameObjects:
            go.draw(self.surface)
    
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                if hasattr(self, "_recorder"):
                    self._recorder.release()
                
            for go in self.gameObjects:
                go.process_event(event)

    def run(self):
        while self.running:
            loop_start = time()

            self._run()

            # calc fps
            self.dt = time() - loop_start
            self.fps = (1 / self.dt) if self.dt else float("inf")

            # draw frame time
            
            frame_time_text = sysfont.render(f"{self.dt} frame time", True, (255, 255, 255))
            # self.surface.blit(frame_time_text, frame_time_text.get_rect().topleft)
            
            # draw fps counter
            fps_text = sysfont.render(f"{int(self.fps)} fps", True, (255, 255, 255))
            x = fps_text.get_rect().left
            y = fps_text.get_rect().top + frame_time_text.get_rect().height
            self.surface.blit(fps_text, (x, y))

            if hasattr(self, "_recorder"):
                data = pygame.surfarray.array3d(self.surface)
                data = data.swapaxes(0, 1)
                data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                self._recorder.write(np.array(data))

            # update screen
            pygame.display.flip()
            self.surface.fill(0)

    def _run(self):
        self._handle_events()
        self._draw_gameObjects()

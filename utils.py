import os
import random
from time import time


def random_file(folder):
    files = os.listdir(folder)
    return folder + "\\" + random.choice(files)

def files(folder):
    files = os.listdir(folder)
    return [folder + "\\" + file for file in files]

def generate_fp(dir, name, ext):
    i = 0
    while True:
        if not os.path.exists(os.path.join(dir, f"{name}_{i}.{ext}")):
            return os.path.join(dir, f"{name}_{i}.{ext}")
        i += 1

class Timer:
    def __init__(self) -> None:
        self.timings = {}
    
    def start(self, name):
        self.timings[name] = time()

    def has_name(self, name):
        if name not in self.timings:
            raise ValueError(f"Name {name} doesn't exist!")
    
    def stop(self, name):
        self.has_name(name)
        self.timings[name] = time() - self.timings[name]
        print(self)
    
    def get_timing(self, name):
        self.has_name(name)
        return self.timings[name]
    
    def __repr__(self) -> str:
        s = " Timings ".center(100, "#")
        s += "\n"
        for name, dt in self.timings.items():
            s += f"{name}: {dt}s\n"
        return s


def func_runtime(func: callable):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        runtime = time() - start_time
        print(f"CALL '{func.__name__}' took {runtime} seconds.")
        return result
    return wrapper

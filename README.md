<image src="https://github.com/user-attachments/assets/cfeec7b8-e017-4ae9-92f6-8b68d121b37b" width="100%"/>

# N-Body simulation


This is the final version of the [nbody](https://en.wikipedia.org/wiki/N-body_simulation) simulation I made in my first year at my university.
The original write up of my group, that also covers the amazing unity implementation of my group buddy, can be found [here](https://www.mintgruen.tu-berlin.de/mathesisWiki/doku.php?id=ws2122:nbody:n-body-simulation).

<image src="https://github.com/user-attachments/assets/2f52cb1e-14fc-453a-8d95-3860b2867d7e" width="100%"/>

<image src="https://github.com/user-attachments/assets/48081a26-b385-4100-85cf-2f9ac5d421c1" width="100%"/>

Larger bodies are rendered larger whereas smaller bodies are rendered smaller.
Now let's check weather attraction works indeed!

https://github.com/user-attachments/assets/d1819951-c1f5-40de-b254-a3481bce092e

This kind of rendering is possible by calculating the color of each pixel individually.

With a mathematical function we can describe the shape and color of each body.

https://github.com/user-attachments/assets/2c64cb13-8efb-456c-bfa6-250179ef719f

The usage of thread shared gpu memory requires synchronization of all threads.

https://github.com/user-attachments/assets/f5f58a3d-42d9-480e-9ea8-3eed9565f8c7

The [three body problem](https://en.wikipedia.org/wiki/Three-body_problem) can only solved approximational. 

# How to run

First you need a CUDA capable GPU.

Then you can install dependencies and run the application as following:
```bash
cd ./nbody
pip install -r ./requirements.txt
python3 main.py
```
You can also change the run configuration inside of `main.py`.

Videos will be saved to the location `./nbody/videos/` if existing (see `game.py`).




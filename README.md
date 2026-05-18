<image src="https://github.com/user-attachments/assets/cfeec7b8-e017-4ae9-92f6-8b68d121b37b" width="100%"/>

# N-Body simulation

In 2022, Lennart and I decided to implement a solver for the [n body](https://en.wikipedia.org/wiki/N-body_simulation) problem, using Euler iteration, as our free project. 
Mathesis is a course for beginners to learn programming in Python and includes a project of choice as final homework. 
We both developed individually a N-body simulation using different technologies.

The original write-up of my group can be found [here](https://www.mintgruen.tu-berlin.de/mathesisWiki/doku.php?id=ws2122:nbody:n-body-simulation). There, you can see Lenart's simulation of 70k bodies in the background.

Without any prior-knowledge about GPU programming, I was able to implement an interactive viewer showing a stream of GPU computed images displaying a nBody simulation.
At that time, Python was my favorite programming language and I wanted to limit test the capabilities of the python ecosystem.
A newly discovered python library able to compile python functions to CUDA compute shaders emphasized this curiosity.
Only a basic subset of python features was actually supported by the compiler, i.e. programmers could only call a predifined subset of python functions within shaders. 
By subdividing needed functionality into separate shaders (image rendering, gravity, vector-add-mul) and call them in specific sequences, I was able to reduce shader code complexity.
The library `pygame` conviniently provides functionality for interactivity (mouse, keyboard) and rendering images stored in CPU memory.
Unfortunatly, sending data between CPU and GPU is rather expensive, as shader outputs are stored in GPU memory. 
By comparing the framerates of my simulation to Lennarts simulation (Unity render pipeline) and benchmarking specific components, I confirmed that the render pipeline has optimization potential.
Over the entiere project I experiemented a lot with the visualization of rendering point masses.

<image src="https://github.com/user-attachments/assets/2f52cb1e-14fc-453a-8d95-3860b2867d7e" width="100%"/>

<image src="https://github.com/user-attachments/assets/48081a26-b385-4100-85cf-2f9ac5d421c1" width="100%"/>

Larger bodies are rendered larger, whereas smaller bodies are rendered smaller.
Now let's check whether attraction works indeed!

https://github.com/user-attachments/assets/d1819951-c1f5-40de-b254-a3481bce092e

This kind of rendering is possible by calculating the color of each pixel individually.

With a mathematical function we can describe the shape and color of each body.
As there are no collisions, high density areas tend to disperse.

https://github.com/user-attachments/assets/2c64cb13-8efb-456c-bfa6-250179ef719f

(GitHub requires video compression.)

Here we can see the accumulating errors, as an initially stable system slowley moves towards a chaotic one.

https://github.com/user-attachments/assets/f5f58a3d-42d9-480e-9ea8-3eed9565f8c7

The [three-body problem](https://en.wikipedia.org/wiki/Three-body_problem) has no closed form solution. 

# How to run

First, you need a CUDA-capable GPU and install the requirements from the python package numba.cuda.

Then you can install dependencies and run the application as following:
```bash
cd ./nbody
pip install -r ./requirements.txt
python3 main.py
```
You can also change the initial configuration inside `main.py`.

Videos will be saved to the location `./nbody/videos/` if the directory `videos` exists (see `game.py`).




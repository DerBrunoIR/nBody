

# N-Body simulation
![image](https://github.com/user-attachments/assets/cfeec7b8-e017-4ae9-92f6-8b68d121b37b)

This is the final version of the [nbody](https://en.wikipedia.org/wiki/N-body_simulation) simulation I made in my first year at my university.

The original write up of my group, that also covers amazing unity implementation of my group buddy, can be found [here](https://www.mintgruen.tu-berlin.de/mathesisWiki/doku.php?id=ws2122:nbody:n-body-simulation).

The following will only summerize the simulation implemented in python.


![image](https://github.com/user-attachments/assets/2f52cb1e-14fc-453a-8d95-3860b2867d7e)

My focus was on achieving raw performance. 
However, back then, it didn't played out as I planed, due to missing knowledge of render pipelines.

At 1000 bodies, real time is merely reached.

Nevertheless I got an self written render pipeline I could experiement with!

![image](https://github.com/user-attachments/assets/48081a26-b385-4100-85cf-2f9ac5d421c1)

Larger bodies are rendered larger whereas smaller bodies are rendered smaller.

![two_bodies](https://github.com/user-attachments/assets/6c5251fc-fee2-4fc4-8802-f0b5f8e5a810)



#
At the end, the simulation has been fully implemented in python and includes the following features:
- simple 2d camera control, 
- supports 2d and 3d simulation space
- video recording, screenshots
- custom cuda shaders for rendering and force calculations (achieved in python via `numba.cuda`)





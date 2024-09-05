<image src="https://github.com/user-attachments/assets/cfeec7b8-e017-4ae9-92f6-8b68d121b37b" width="100%"/>

# N-Body simulation


This is the final version of the [nbody](https://en.wikipedia.org/wiki/N-body_simulation) simulation I made in my first year at my university.

The original write up of my group, that also covers the amazing unity implementation of my group buddy, can be found [here](https://www.mintgruen.tu-berlin.de/mathesisWiki/doku.php?id=ws2122:nbody:n-body-simulation).


<image src="https://github.com/user-attachments/assets/2f52cb1e-14fc-453a-8d95-3860b2867d7e" width="100%"/>

My focus was on achieving raw performance. 
However, back then, it didn't played out as I planed.

Instead I shifted focus to develope some nice looking shaders.

<image src="https://github.com/user-attachments/assets/48081a26-b385-4100-85cf-2f9ac5d421c1" width="100%"/>

Larger bodies are rendered larger whereas smaller bodies are rendered smaller.

<image src="https://github.com/user-attachments/assets/caf3472a-8707-4058-b4a3-2cd259ac8572" width="100%">

Now let's check weather attraction works indeed!

https://github.com/user-attachments/assets/d1819951-c1f5-40de-b254-a3481bce092e

This kind of rendering is possible by calculating the color of each pixel individually.

For `1000x1000` pixels we have `10**6` pixels per frame.
Gpu's are able to deal with that, however data takes some time to travle between those components.

https://github.com/user-attachments/assets/2c64cb13-8efb-456c-bfa6-250179ef719f

Shaders are executed in parallel, don't forget to synchronize your threads before accessing shared data.
Those bugs are painfully to debug.

https://github.com/user-attachments/assets/f5f58a3d-42d9-480e-9ea8-3eed9565f8c7

The [three body problem](https://en.wikipedia.org/wiki/Three-body_problem) has no analytical solutions. 

The End

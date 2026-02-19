# My Approach

I've heard it said that computer science is the art of building reliable systems out of unreliable parts. This project is me putting that maxim into practice. https://ieeexplore.ieee.org/abstract/document/1641372

The focus of this project is as a playground for me to learn basic skills associated with ML, and to understand some of the basic techniques used with transformers to produce usable outputs.

Although a core part of this project is my custom P2G/G2P transformer, I'm explicitly not trying to build the best G2P or P2G model ever. There are obviously better off the shelf solutions for that.

I want to explore building a "good enough" small model from scratch, and use what techniques I can to leverage good performance on a specific domain out of that model.

# Tricks I'm using

I've trained a single, multi-task model. Because it's bidirectional, I'm able to do some (I think) clever things to improve performance during my output beam search.

A frequent problem with small transformers in general is terminating early or entering strange repetitive tail loops (?TODO is this the right name?). Because my model is bidirectional, I am able to compute the loss associated with my output/input pair at inference time, which makes a reasonably good detector of these problems. A high loss implies that the model is very unlikely to recover the original input when run in reverse.

NOTE: This is called Bi-Directional Reranking, use correct terminology.

Specifically, I compute the loss when teacher forcing the actual output on the reversed task against the input (TODO, wording, unclear?).
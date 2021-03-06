# GPUContagion

Currently this is just a place for experimentation, I implement ideas from this repo into my actual projects, and hopefully PRs into the Julia GPU packages.

The basic functionality implemented is a simple CUDA kernel that implements an SIR model on a network, represented by a `BitMatrix`.

It's currently about an order of magnitude faster than the equivalent algorithm on a CPU. I am using a `BitMatrix` for it's fast edge-rewiring complexity, as the ultimate goal is a time-dependent network.


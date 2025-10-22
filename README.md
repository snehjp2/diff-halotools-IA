# Differentiable halotools-IA

|![](/src/demo/frame_first.png)<br>|![](/src/demo/scatter_evolution.gif)<br>| ![](/src/demo/frame_last.png)<br>
|:-:|:-:|:-:|

*Above: using gradients to update the galaxy field based on 1-pt statistics (galaxy number counts). In this case, the HOD parameterization was optimized towards producing a galaxy field with 500,000 galaxies.*


This github provides an implementation of differentiable halo-occuptation distribution (HOD), $\texttt{diff-halotools-IA}$, modeling that includes galaxy intrinsic alignment (IA) implementation.

The HOD implementation is standard (see [this paper](https://arxiv.org/abs/astro-ph/0703457)), and the IA modeling formalism follows [Van Alfen et al. 2023](https://arxiv.org/abs/2311.07374). The differentiable HOD implementation is largely inspired by [Horowitz et al. 2022](https://arxiv.org/abs/2211.03852), with subtle changes, and the IA modeling implementation is novel.

$\texttt{diff-halotools-IA}$ includes PyTorch and jax implementations. This repository is under active development.


# Generative diffusion models: theory + fundamentals

[a sample training run](diffusion/ddpm/progression.png)

I made this repo in an attempt to better fundamentally understand diffusion models. It's still a work-in-progress, and the structure isn't totally set. My main goal is two accomplish two things:

1. Give simple, **minimal code**, with a minimal toy dataset, that can show how to construct a generative diffusion model.
2. Give a longer explanation of how these models work, including filling in some gaps in theory.
3. Share some of my hard-earned intution on the problem.

Currently, I would suggest reading the repo in this order:

1. Go to diffusion/ddpm.
2. Read through the README.md. This contains all of the theory, math, notes and a little bit of code.
3. Run the code samples. The most important thing is the diffusers.py scheduler. The models, data, and training loop are all generic.

As usual, the requirements.txt file contains everything you would need to run this in a virtual env, conda env, or otherwise. `pip install -r requirements.txt`

You can run the ddpm demo by running `python -m diffusion.ddpm.main`. It will save as output a GIF giving the different sampled distributions of our DDPM over the course of training.

## Denoising Diffusion Probabilistic Models (DDPM)

In the diffusion/ddpm package, the diffusers.py module contains the Denoising Diffusion Probabilistic Models (DDPM) noise scheduler. Each diffusion technique can be thought of as a mixin or wrapper of a learnable model. It must satisfy the following methods:

- has a method to retrive the variance at either a) discrete time or b) continuous time
- defines a loss function given some input datapoint _x_ and a time in the diffusion process _t_
- defines a sample function, that allows you to generate samples from the distribution

Instead of learning the distribution $p_\theta(x)$, diffusion models learn the score $\nabla \ln p_\theta(x)$ of the distribution, which, through the magic of particle-wave duality (well, like the classical version) can be used to generate a sample from $p_\theta(x)$!

In the `diffusion/ddpm/main.py` module you'll find a toy script, that will train a diffusion model on the 2D swiss roll dataset. As output, it will generate an animated gif showing how the generated distributions for an ensemble of particles as training progresss. You'll see it gets closer and closer to the true distribution.

The `tutorial.ipynb` has all of the same stuff as main.py but in notebook form.

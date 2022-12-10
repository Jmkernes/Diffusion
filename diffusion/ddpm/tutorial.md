# Diffusion models, an introduction

What are diffusion models? They are essentially just generative models, which simply means instead of learning a target/class y given some data x, i.e. probability $p(y|x)$, you are learning the probability of a given distribution of data $p(x)$. Generative models are pretty common; in 1D, kernel density estimation is a common generative model, and can be thought of as fitting a smooth curve to a histogram (just like learning the probability density). A common 2D model is linear discriminant analysis, whereby you model the density of a distribution using gaussians. More complicated high dimensional examples include energy based models (EBMs), Generative Adversarial Networks (GANs), and Variational AutoEncoders (VAEs).

The main idea behind diffusion models is that instead of learning the distribution $p(x)$ directly, we learn a "Force" $F(x) = -\nabla  \log p(x)$ that, under certain conditions, has an equilibrium distribution which is exactly $p(x)$. We'll prove that connection, but the idea of diffusion models is this:

> Instead of learning p(x), learn how to make some q(x) closer to $p(x)$.

## Why can't we just learn $p(x)$?

This is the first question you should ask, and the answer is: because it's hard. It's hard for the same reasons that QFT, statistical mechanics, and modern condensed matter theory are hard. Normalization constants. The idea behind gradient based optimization problems is pretty simple. Pick a big function with lots of parameters, feed it a target, and optimize. The problem, is that an arbitrary function generator will not naturally spit out probability distributions. Distributions need to obey $\int p(x) dx = 1$, which is not invariant under gradient descent (at least naturally it isn't).

> If that's too abstract, let's say you wanted to train a model to learn the bias of a coin. You flip it 10 times and get 8 heads and 2 tails. The model can be specified by probability of heads, $\theta$, and written as $(\theta, 1 - \theta)$. Gradient updates would be symmetric and obey normalization, provided that $\theta$ stays within $[0, 1]$. Instead let's make this an _unconstrained_ problem, and also parametrize the redundant probability of tails such that the model is now $(\theta_1, \theta_2)$. If we start at $\theta_1 = \theta_2 = 0.5$, it's possible that after one update we might have $\theta_1 = 0.6$, $\theta_2 = 0.49$, which does not satisfy a normalized probability ($\theta_1 + \theta_2 != 1$).

### EBMs

You don't learn without trying, so let's **try** to compute an EBM objective. Given a Neural Net (or other function $E: \mathbb{R}^D \to \mathbb{R}$), the distribution is given by:

$$
p(x) = \frac{e^{-E_\theta(x)}}{\int dx e^{-E_\theta(x)}} = \frac{e^{-E_\theta(x)}}{Z(\theta)}
$$

where we've defined the partition function $Z(\theta) = \int dx e^{-E_\theta(x)}$. $\theta$ are some learnable model parameters that we're going to autodiff the hell out of. To learn the generative distribution, we want to maximize the log probability. Why? Because logs make things look good, and logarithms preserve optimization. Maxing out the probability of dataset $\mathcal{D}$, and the log-probability give the same answers.

$$
p(x) = \frac{e^{-E_\theta(x)}}{\int dx e^{-E_\theta(x)}} \implies  \ln p(x) = -E_\theta(x) - \ln  \int dx e^{-E_\theta(x)}
$$

The integral inside the logarithm is a real pain though. Continuing, the gradient update would be:

$$
\nabla_\theta  \ln p(x) = -\nabla_\theta E_\theta(x) + \mathbb{E}_{x\sim p(x)}[\nabla_\theta E_\theta(x)]
$$

where we've defined
$$\mathbb{E}_{x\sim p(x)}[\nabla_\theta E_\theta(x)] = \int dx \nabla_\theta E_\theta(x) e^{-E_\theta(x)} / Z = \int dx \nabla_\theta E_\theta(x) p(x)$$ If we think about maximizing over N i.i.d samples, we realize we're trying to push the gradient of the energy closer to the average gradient of the true distribution. This argument is kind of circular though. We need to know $p(x)$ to calculate the update to compute $p(x)$! One way to approach this, is iteratively sample from the current $p(x)$, compute $E(x)$, sample, compute, sample, compute, ... until convergence. In any rate, it's probably not ideal.

There is an alternative though. Notice what happens if we instead try to solve for $\nabla_x p(x)$. The normalization disappears! This leads us to define the notion of the _score_

$$
s_\theta(x) \equiv -\ln \nabla p(x) = \nabla E_\theta(x),
$$

which does not contain a pesky normalization constant. This is great, but how does it help us recover $p(x)$? Also, wouldn't optimizing this lead to some horrible Jacobian/second derivatives? Yes to the second. That's a big issue. That optimization objective also goes by the name of Fisher divergence (I think? might be getting the name wrong). Let's answer the first complaint though. It turns out, that there is an intimate relationship between the score and the distribution, which is related to the relationship between Langevin equations and Fokker-planck equations. The score $s(x)$ is like a Force that leads to a potential $p(x)$. We'll derive all of this later, but for now, let's leave the discussion here and think about a different approach to remove $Z$.

### Normalizing flow mentality

The reason $Z$ is hard to find, is because $p(x)$ is an arbitrary, complicated, unconstrained function (neural nets are universal approximators! blah blah blah...). In this small section, we're going to give a preview of the idea behind diffusion models.

Let's take a page out of the normalizing flows notebook, and try to make this problem tractable by making $p(x)$ simpler. Let's assume that $p(x) = N(x| \mu ,\Sigma)$. But wait... thats a pretty wild assumption right? Yeah, it is. It's a good thing we can take the usual physics approach though, and just find a limit where this isn't so crazy. It turns out that that's pretty easy to do actually. Let's introduce a new auxiliary parameter $t$, which we will call the "time". The function $p(x, t)$ has the following properties

1. $p(x, t \to  0)$ is something easy to understand. Let's call it $N(x|0, 1)$

2. $p(x, t \to  \infty)$ is exactly what we want, $p(x)$.

3. There exists some transition kernel $T$ such that $p(x, t + \delta t) = \int dx' T_{\delta t}(x, x', t)p(x', t)$ (this is just the [chapman-kolmogorov equation](https://en.wikipedia.org/wiki/Chapman%E2%80%93Kolmogorov_equation))

There's no loss of generality here, but you might question why make this problem harder. The reason, is that for small changes in times $\delta t$, we know that the transition kernel must become a delta function. And, since we can recover a delta function as the limit of a Gaussian with zero variance, there must some small window of time where the transition is basically Gaussian. And that's it. We'll need to iterate step (3) lots of times to get the answer that we want, but at least we've removed the normalization constant.

---

> We are making a trade. In exchange for removing one horrific normalization constant, I give you an integral over an infinite number of Gaussians with trivial normalization constants

---

## Langevin, Fokker Planck, and Path integrals

This is not directly needed to understand diffusion models, and can be skipped on a first read through. Although, you may want to refer back to this section periodically.

In this section, we will take an aside to show the connection between all three of these representations. They are each frameworks for understanding the dynamics of a probability distribution, and there exists a way to freely move between each of them. You can think of them using these sparknotes:

1. Langevin equations are ODEs that tell you how to generate one sample from a distribution, by simulating the motion of a particle.
2. Fokker-Planck equations are a PDE that tells you how the entire distribution changes over time
3. Path integrals are functional integrals that are designed to compute conditional probabilities to go from state A to state B in some amount of time $t$

Each has its strengths and weaknesses. We will find Langevins are good for generating samples and path integrals are good for making training objectives.

### Langevin

The Langevin equation is just an over-damped equation of motion plus some random, time-independent gaussian noise $\eta(t)$

$$
\frac{dx}{dt} + \nabla V = \eta(t), \\
p(\eta) \sim e^{-\eta^2 / 4D}
$$

In discrete form, it would look like:

$$
x_{t+1} = x_t + \epsilon F_t + \int_t^{t + \epsilon} \eta(\tau) d\tau
$$

We can convert the ODE to a probability density p(x) using the probability change of variable rule

$$
p(\eta) = p(x) \det\left|\frac{\delta  \eta}{\delta x}\right|^{-1},
$$

giving:

$$
p(x) \sim e^{-(\dot x + \nabla V)^2 / 4D} \det\left|\frac{\delta  \eta}{\delta x}\right|.
$$

There are two different ways to approach this determinant. If we make the Itô choice and discretize as

$x_{n+1} = x_n + F_n \delta t + \eta_n$, the Jacobian is:

$$
\det\left|\frac{\delta  \eta}{\delta x}\right| = 1
$$

If instead, we smear out the noise over the interval $\delta t$ instead of acting as an initial kick, we find a more complicated answer. The sketch for this case is as follows:

$$
\det\left|\frac{\delta  \eta}{\delta x}\right| = \det|\partial_t + \nabla^2 V| = \text{exp}\left( \ln (1 + \partial_t^{-1} \nabla^2 V)\right) = \frac{1}{2}\nabla^2 V
$$

(you'll need to know that the heaviside-step function is the inverse $\partial_t$, then taylor expand the logarithm)

In short, the Langevin equation tells us that the transition probability to a new state is given by a normal distribution:

$$
T_{\delta t}(x, t|x') = \mathcal{N}(x|x' + \delta t F(x, t), 2 D(x)).
$$

Notice, that this fits the form of our Gaussian transition kernel postulated earlier. That's nice, it turns out that if we know the transition kernel, we can use the Langevin equation to generate samples.

### Path integrals

Just to make things look familiar, here is the transition kernel equation written in the usual bra-ket notation:

$$
p(x, t+ \epsilon) = \int dx'' \langle x | \hat T(\epsilon) | x'' \rangle p(x'', t), \ \\ \\ \ \langle x | \hat T(\epsilon) | x'' \rangle  \sim e^{-(x_t - x_{t-1} - \epsilon F_{t-1})^2  \epsilon / 4D}
$$

This is in standard form, and so we can immediately write the path integral

$$
p(x_f, t) = \int_{x(0) = x_i}^{x(t) = x_f} \mathcal{D} x\ \ \text{exp}\left[ \frac{-1}{4D} \int_0^t(\dot x^2 + \nabla V(x))^2d\tau\right] p(x_i, 0)
$$

As is usual with path integrals, if we discretize them we get a chain of transition operators, which is equivalent to a Markov chain of conditional probabilities $q_t(x_t) = \int  \prod_{i=1}^{t} dx_{i-1} q_i(x_i | x_{i-1})$

You might think sampling from this is trivial, but it's not. That's because, like every 👏 single 👏 problem 👏 in 👏 physics 👏 there is an annoying boundary condition. You have to make damn sure (Taking Back Sunday style) that you end at your target distribution $p(x)$. Sampling from a path integral is like trying to walk blindfolded and hope you make it to where you want to go.

This section is not related to the literature, but just musing on what, if anything, can we do with the path integral representation. We can try to manipulate this path integral to do other stuff, but none of them are particularly helpful. The challenge is that $V(x)$ can be any arbitrary function, and we cannot form approximations for it either. You cannot formulate this as a scattering problem, since the assumption there is that $V(x \to  \pm  \infty) = 0$, which is not true here. That rules out some stuff. We can try to move into fourier space, but the boundary conditions on position make that difficult. We could try the trick of $x_\text{bd} + x_h$ where $x_\text{bd}$ satisfies the boundary conditions and the homogenous part does not. But, the action is not quadratic due to arbitrary $V(x)$, and there is no reason that we can truncate its expansion. So, not much we can do unfortunately.

### Fokker-Planck equations

We'll derive this in a hand-wavy manner by finding the generator function. Consider an arbitrary function $f(x)$, for which we want to find its time-dependent expectation. Just like in quantum mechanics, this can be expressed in a Schrodinger or Heisenberg representation, meaning the time dependence is either in the state or not. Let's adopt the Heisenberg approach, and consider a time-dependent probability density function $p(x, t)$ such that the expectation at any time can be computed as:

$$
\langle f \rangle = \int p(x, t) f(x) dx
$$

We will now compute this via an alternative method that makes use of the Langevin equation, thereby finding an equation that leads to the Fokker-Planck equation. Consider the standard Taylor series:

$$
df = dx_i \nabla_i f + \frac{1}{2} dx_i dx_j \nabla_i \nabla_j f + \mathcal{O}(dx^3).
$$

In order to make use of the Langevin equation, we'll need to make two assumptions about the noise. (1) impose that the second moment is proportional to the time difference between events, and (2) that events are time-independent, i.e. $\langle \eta_i \eta_j \rangle \sim \delta_{ij} dt^{>0}$. With this, we can can infer the averages:

$$
dx = -\nabla V dt + d\eta, \ \\ \ dx^2 = d \eta^2 + \mathcal{O}(dt^{>1})
$$

As a result, we can find the averaged time differential

$$
\langle \partial_t f \rangle = -\nabla f + \frac{1}{2} \langle \eta^2 \rangle \nabla^2 f
$$

We can replace the left-hand side with the previous definition for time averages using the density $p(x, t)$, leading to:

$$
\int \partial_t p(x, t) f(x) dx = \int p(x, t) \left[-\nabla f(x) + \frac{1}{2} \langle \eta^2 \rangle \nabla^2 f(x) \right] dx
$$

Let's now impose that $\langle \eta^2 \rangle = 2 D$, where $D$ is a constant known as the **diffusion coefficient**. Performing integration by parts, and noting that $p(x \to \pm \infty, t) = 0$, we find the following equation:

$$
\int f(x) \partial_t p(x, t) dx = \int f(x) \left( \nabla(p \nabla V) + D \nabla^2 p \right) dx
$$

Since this must hold for any arbitrary function $f(x)$, it implies the differential relation

$$
\partial_t p = \nabla(p \nabla V) + D \nabla^2 p
$$

which is the desired result. We've shown that for every Langevin equation, there is an equivalent Fokker-Planck PDE that describes the dynamics of the distribution p(x,t). We now have a variety of representations to make use of!

## The most important equation

Now that we know the connection between Langevin and Fokker-Planck equations, consider what happens if we set the potential energy to $V(x) = -\ln q_\theta(x)$, and the diffusion constant to $D=1$. The corresponding Langevin equation would be:

$$
x_t = x_{t-1} + \epsilon  \nabla_x  \ln q_\theta(x) + \sqrt{2\epsilon} z, \\
z \sim N(0, 1)
$$

The stationary solution $e^{-V}$ now becomes $q_\theta(x)$! We've shown we can generate samples from any distribution $q_\theta(x)$, by using the above Langevin equation.

This is the heart of what we're doing. We've shown how by just learning the **score** of the distribution, we can actually sample from it!

# Diffusion models

We have all of the math background we need to derive our results now, and so for the remainder of these notes, we will dive into three main papers, chronologically:

1. Sohl-Dickstein, Jascha, et al. "Deep unsupervised learning using nonequilibrium thermodynamics." _International Conference on Machine Learning_. PMLR, 2015. https://arxiv.org/abs/1503.03585
2. Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." _Advances in Neural Information Processing Systems_ 33 (2020): 6840-6851. https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf
3. Song, Yang, et al. "Score-based generative modeling through stochastic differential equations." _arXiv preprint arXiv:2011.13456_ (2020). https://arxiv.org/abs/2011.13456

The first paper lays the foundation. It gives the following formula for building diffusion models.

- Parametrize and represent some transition kernel $T_{\delta t}(x_t, t | x_{t-1}; \theta)$ via a neural network
- Pick some easy prior distribution like $p(x, t=0) = \mathcal{N}(0, 1)$ that will become our target distribution at long times $p_\text{data} = p(x, t\to\infty)$
- Minimize KL divergence of $p_\text{data}(x)$ and $p_\theta(x)$. Use importance sampling to approximate the integrals in the path integral objective we gave.
  - This is where differences appear. "Diffusion" is the surrogate distribution we use for importance sampling, hence the name!
- Generate samples using the Langevin equation plus transition kernel!

In the DDPM paper, they give an efficient method for importance sampling, and also find that the equations simplify drastically. It turns out that diffusion models in this limit are equivalent to removing random noise from samples of your data!

In the SDE paper, instead of learning parameters $\theta(t_i)$ at discrete time steps, they learn a continuous function $\theta(t)$. This is a more general framework with DDPM as a special case.

## Derivation

We'll start off with the paper Deep unsupervised learning using nonequilibrium thermodynamics (2015). Let's define two distributions:

**Forward (destructive) process**

$$q(x_0, \ldots, x_T) = q_0(x_0)\prod_{t=1}^T q(x_t | x_{t-1}), \ \\\ q_0(x) = p_\text{data}(x)$$

**Backard (generative) process**

$$p(x_0, \ldots, x_T; \theta) = p_T(x_T)\prod_{t=1}^T p(x_{t-1} | x_t; \theta), \ \\\ p_T(x) = \mathcal{N}(0, 1)$$

This says that the destructive process samples its initial condition from the observed data, and the constructive process samples its initial (long time) condition from pure noise. Our goal is to maximize the log-likelihood that the generative distribution produced the given data

$$
L = -\sum_{x \in  \mathcal{D}} \ln p_0(x) \approx -\int q_0(x_0) \ln p_0(x_0; \theta) dx_0,
$$

where the approximation is standard in ML. Equality is approached for sufficiently many i.i.d. samples from $\mathcal{D}$. At this stage, you may ask, "Why not just parametrize $p_0(x_0; \theta)$ and minimize this loss?" The answer is that doing so would require a normalizing constant, which means we are directly estimating the data distribution and we are right back where we started. Instead, we make insight (1): **The markov assumption allows us to escape the normalization issue, by rewriting the intractable distribution $p_0(x)$ as a product of normalized transition probabilities**. However, this will come at the expense of introducing additional complexity in the form of latent states $x_t, x_{t-1}, \ldots, x_1$ that will need to be integrated out. Let's get to work now on this and we will find that we are forced to make several choices.

$$
\ln p_0(x_0; \theta) = \ln\left\lbrace\int dx_{1:T} \ \p(x0, \ldots, x_T)\right\rbrace = \ln  \left\lbrace\int dx_{1:T}\ \ p_T(x_T)\prod_{t=1}^T p(x_{t-1} | x_t; \theta)\right\rbrace
$$

Clearly the integrals are a problem. We don't have any way to evaluate these. To circumvent this, we apply our usual trick of replacing integrals with Monte-Carlo estimates. But, we now need a distribution that gives a probability for all times $0 < t < T$ that are sampled! This is where the forward process comes into play. We need to engineer a distribution that is cheap and easy to sample for all of the data points, as well as analytically tractable. It turns out, that we can build this using the same method as for the backward process -- assume that the distribution is Markov. We will add a resolution of unity $q(x_{0:T}) / q(x_{0:T})$ and simplify a bit using the Markov representation

$$
\ln p_0(x_0; \theta) = \ln  \left\lbrace\int dx_{1:T}\ \ q(x_{1:T} | x_0) \left(p_T(x_T)\prod_{t=1}^T \frac{p(x_{t-1} | x_t; \theta)}{q(x_t | x_{t-1})}\right)\right\rbrace
$$

We have also made use of Bayes rule to write $q(x_{0:T}) = q(x_{1:T} | x_0)q(x_0)$ and omit the final $q_0(x_0)$ in our resolution of unity. Instead of directly estimating this with sampling though, we will instead apply a variational bound argument and optimize an approximate loss. In short, this means that we can move the $\ln$ inside the integral and past the distribution $q(x_{1:T} | x_0)$. For a small proof, see the note below.

> Note the following: Start with a distribution $\ln p(x)$, and add a spurious dependence on a latent variable $z$:

$$
\ln p(x) = \ln  \left(\frac{p(x, z)}{p(z | x)}\right)
$$

Now, take another normalized distribution, and multiply by the identity $\int q(z) dz = 1$

$$
\ln p(x) = \int q(z) \ln  \left(\frac{p(x, z)}{p(z | x)}\right) dz
$$

Next, add $q(z) / q(z)$ inside the logarithm, and use the product rule to expand this to:

$$
\ln p(x) = \int q(z) \ln  \left(\frac{p(x, z)}{q(z)}\right) dz + \int q(z) \ln  \left(\frac{q(z)}{p(z | x)}\right) dz
$$

We can make use of the fact that the second term is a KL-divergence, which is guaranteed to always be greater than or equal to zero. We now have the inequality:

$$
\ln p(x) \geq  \int q(z) \ln  \left(\frac{p(x, z)}{q(z)}\right) dz
$$

We're almost there. Now on the left side, add a dependence on a latent variable $z$ to get:

$$
\text{LHS} = \ln  \int p(x, z) dz = \ln  \int q(z) \left(\frac{p(x, z)}{ q(z)}\right) dz
$$

which gives:

$$
\ln  \int q(z) \left(\frac{p(x, z)}{ q(z)}\right) dz \geq  \int q(z) \ln  \left(\frac{p(x, z)}{q(z)}\right) dz
$$

Identifying the term inside the parenthesis as some function $g(z)$, we now have the desired inequality:

$$
\ln  \int q(z) f(z) dz \geq  \int q(z) \ln f(z) dz
$$

Moving on, we make the Evidence Lower Bound (ELBO) approximation:

$$
\ln p_0(x_0; \theta) \geq K = \int dx_{1:T}\ \ q(x_{1:T} | x_0) \ln  \left(p_T(x_T)\prod_{t=1}^T \frac{p(x_{t-1} | x_t; \theta)}{q(x_t | x_{t-1})}\right)
$$

Which is great because now we can tackle that product. What we want to do though, is reverse the arguments such that the forward process looks like the backward process, i.e. we want to somehow swap $q(t| t-1) \to q(t-1| t)$. This is not exactly easy, because from bayes rule we know $q(x_t | x_{t-1}) = q(x_{t-1} | x_t) q(x_t) / q(x_{t-1})$, which requires integrating the entire markov chain up to time $t$ (note, that also includes integrating out the $x_0$ component eek!). However, recall the other end of the chain is $p_0(x_0)$ which cannot be integrated over, since it is observed empirically! So it seems we are a little screwed. Thankfully, it's not that bad, since we can make use of the fact that we actually know $x_0$ for any given sample. The _conditional_ probability $p(x_{t} | x_{t-1}, x_0)$ _is_ invertible, since it would lead to $p(x_t | x_0)$ and $p(x_{t-1} | x_0)$ which is all gooood. Let's now go about our task of inverting all of the probabilities that we can. We need to pull out the 0th component though first, so we now have:

$$
K = \int dx_{1:T}\ \ q(x_{1:T} | x_0) \ln  \left(p_T(x_T)\prod_{t=2}^T \frac{p(x_{t-1} | x_t; \theta) q(x_{t-1} | x_0)}{q(x_{t-1} | x_t, x_0) q(x_t | x_0)}\right) + \ln\left(\frac{p(x_0 | x_1)}{q(x_1 | x_0)}\right)
$$

The term

$$
\ln  \left(p_T(x_T)\prod_{t=2}^T \frac{q(x_{t-1} | x_0)}{q(x_t | x_0)}\right)
$$

has some nice cancellations, similar to ((n-1)! / n!), and gives the simple result

$$
\ln  \left(p_T(x_T)\frac{q(x_{1} | x_0)}{q(x_T | x_0)}\right)
$$

The $q(x_{1} | x_0)$ term cancels with the denominator of the last term in the previous expression. We are now currently at:

$$
K = \int dx_{1:T}\ \ q(x_{1:T} | x_0) \left\lbrace\ln  \left(\frac{p_T(x_T)}{q(x_T | x_0)}\right) + \ln  \left(\prod_{t=2}^T \frac{p(x_{t-1} | x_t; \theta)}{q(x_{t-1} | x_t, x_0)}\right) + \ln p(x_0 | x_1)\right\rbrace
$$

At this point, the DDPM paper does a better job of simplication. If we recall, the loss was $\int q_0(x_0) K dx_0$, which amounts to just replacing the {1:T} with {0:T} in the integrand differential. Let's quickly do this and then we are at equation A.21 in the DDPM (Ho 2020) paper.

$$
\text{ELBO} = \int dx_{0:T}\ \ q(x_{0:T}) \left\lbrace\ln  \left(\frac{p_T(x_T)}{q(x_T | x_0)}\right) + \ln  \left(\prod_{t=2}^T \frac{p(x_{t-1} | x_t; \theta)}{q(x_{t-1} | x_t, x_0)}\right) + \ln p(x_0 | x_1)\right\rbrace
$$

Since integrating over an unused variable $x_t$ does nothing (i.e. it's just 1 since $q$ is a probability distribution), we can eliminate all the un-needed variables, which leads to the result

$$
\text{ELBO} = D_\text{KL}(p_T(x_T) || q(x_T | x_0)) + \sum_{t=2}^T D_\text{KL}(p(x_{t-1} | x_t; \theta)|| q(x_{t-1} | x_t, x_0)) + \int dx_1 dx_0\ \ q(x_1, x_0) \ln p(x_0 | x_1)
$$

### The forward path

By definition, the forward destructive process does not contain learnable parameters. We now have choice in how we parametrize it: continous or discrete.

#### Continuous

The nice thing about general solutions is they tend to be shorter, so we'll show the Yang answer first. We adopt the SDE method of Yang et. al., and describe the forward process via a stochastic equation, whose solution can be analytically computed. The key feature, is that we must be able to quickly and efficiently compute the probabilities $q(x_{t-1}|x_{t}, x_0)$. To that end, we shall consider all stochastic processes in the following family:

$$
dx = \alpha(t) \ \ dt + \beta(t) \ \ dw
$$

It obeys the corresponding Fokker-Planck equation

$$
\frac{\partial P}{\partial t} = -\alpha(t) \sum_i  \frac{\partial P}{\partial x_i} + \beta^2(t) \nabla^2 P
$$

$$
\frac{\partial P(q, t)}{\partial t} = -\left(\alpha(t)(\mathbb{1} \cdot q) + \beta^2(t) q^2\right) P(q, t)
$$

$$
P(x, t) = \int dq e^{iq\cdot x}e^{i\left(x - \mathbb{1}\int  \alpha(\tau) d\tau  \right) \cdot q - \int  \beta^2(\tau) d\tau q^2} = e^{ - \frac{\left(x - \mathbb{1}\int  \alpha(\tau) d\tau  \right)^2 }{4  \int  \beta^2(\tau) d\tau}}
$$

$$
= \mathcal{N}\left(\int  \alpha(\tau) d\tau, 2  \int  \beta^2(\tau) d\tau\right)
$$

To reproduce DDPM, we choose the stochastic process:

$$
dx = -\frac{1}{2}\beta(t) \ \ dt + \sqrt{\beta(t)} \ \ dw
$$

#### Discrete

This section follows the approach of DDPM. This is the IMO the most math heavy section. There's lots of algebra, but by the end we'll have a wonderfully simple loss function!

We're interested in minimizing:

$$
\text{ELBO} = D_\text{KL}(p_T(x_T) || q(x_T | x_0)) + \sum_{t=2}^T D_\text{KL}(p(x_{t-1} | x_t; \theta)|| q(x_{t-1} | x_t, x_0)) + \int dx_1 dx_0\ \ q(x_1, x_0) \ln p(x_0 | x_1),
$$

for some ordered schedule of times $0 < t_1 < \ldots < t_t < 1$, where time moves from ordered to disordered. We will assume (remember all the gaussian stuff we did in the opening part of these notes?) that we can represent the transition kernels as Gaussians

$$
q(x_t | x_{t-1}) = \mathcal{N}(\mu_t(x_{t-1}), \beta_t),
$$

with some position-dependent means $\mu_t(x_{t-1}),$, and time-dependent variances $\beta_t$.

Since there are no learnable parameters here, we need to choose the mean and variance of each forward kernel, at each time $t$. The paper suggests one particularly clever construction, which is analogous to a stick-breaking process. It ensures that the sum of variances is always less than 1. Choose the following:

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t| \sqrt{\alpha_t} x_{t-1}, 1 - \alpha_t).
$$

We identify the variances as $\beta_t = 1 - \alpha_t$. We can integrate out one intermediate state easily (i.e. finding $q(x_t | x_{t-2})$) by using the convolution formula for Gaussians. This amounts to summing the means and variances. Alternatively, we can solve for the random variable $x_t$ of these distribution, in terms of two normals $z_{t-1}, z_{t-2}$, then use that we know it's a Gaussian to recover the mean and variance. Set

$$
x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1 - \alpha_1} z_0 \\
x_2 = \sqrt{\alpha_2} x_1 + \sqrt{1 - \alpha_2} z_1 \\
\implies x_2 = \sqrt{\alpha_1 \alpha_2} x_0 + \sqrt{\alpha_2(1-\alpha_1)} z_0 + \sqrt{1-\alpha_2} z_1 \\
\implies  \langle x_2  \rangle = \sqrt{\alpha_1 \alpha_2} x_0, \ \\ \\ \ \langle x_2^2  \rangle_c = 1 - \alpha_1 \alpha_2
$$

We can be more formal and do a proof by induction, but it becomes clear that we have the following closed form:

$$
q(x_t | x_0) = \mathcal{N}(x_t | \sqrt{\alpha_1  \ldots  \alpha_t} x_0, (1 - \alpha_1  \ldots  \alpha_t)) \\
\bar  \alpha_t  \equiv  \prod_{i=1}^t \alpha_i
$$

Next we need to determine $q(x_{t-1} | x_t, x_0)$. We write:

$$
q(x_{t-1} | x_t, x_0) = q(x_{t} | x_{t-1})\frac{q(x_{t-1} | x_0)}{q(x_t | x_0)} \implies
$$

$$
\ln q(x_{t-1} | x_t, x_0) = \frac{1}{2}\left\lbrace\frac{(x_t - \sqrt{\alpha_{t}}x_{t-1})^2}{(1 - \alpha_{t})} + \frac{(x_{t-1} - \sqrt{\bar  \alpha_{t-1}}x_{0})^2}{(1 -\bar  \alpha_{t-1})} - \frac{(x_t - \sqrt{\bar\alpha_{t}}x_{0})^2}{(1 -\bar  \alpha_{t})}\right\rbrace
$$

Keeping only terms containing $x_{t-1}$ gives:

$$
\ln q(x_{t-1} | x_t, x_0) = \frac{1}{2}\left\lbrace
\left(
\frac{\alpha_{t}}{(1 - \alpha_{t})} +
\frac{1}{(1 - \bar  \alpha_{t-1})}
\right) x_{t-1}^2 +
2\left(
\frac{\sqrt{\alpha_{t}} x_t}{(1 - \alpha_{t})} +
\frac{\sqrt{\bar  \alpha_{t-1}} x_0}{(1 - \bar  \alpha_{t-1})}
\right) x_{t-1} \right\rbrace +
\text{const}
$$

Let's deal with simplifying the $x_{t-1}^2$ term first. This can be simplified to $\frac{1 - \bar  \alpha_t}{(1-\alpha_t)(1-\bar\alpha_{t-1})}$. Pulling this out gives:

$$
\ln q(x_{t-1} | x_t, x_0) = \frac{1}{2}\frac{1 - \bar  \alpha_t}{(1-\alpha_t)(1-\bar\alpha_{t-1})}\left\lbrace
x_{t-1}^2 +
2\left(
\frac{(1-\bar  \alpha_{t-1})\sqrt{\alpha_t}}{(1 - \bar  \alpha_{t})} x_t +
\frac{(1-\alpha_t)\sqrt{\bar  \alpha_{t-1}} x_0}{1 - \bar  \alpha_t}
x_{0}\right) \right\rbrace +
\text{const}
$$

We can now write down the solution

$$
q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1} | \mu_t(x_t, x_0), \sigma_t^2) \\ -- \\
\mu_t(x_t, x_0) = \left(
\frac{(1-\bar  \alpha_{t-1})\sqrt{\alpha_t}}{(1 - \bar  \alpha_{t})} x_t +
\frac{(1-\alpha_t)\sqrt{\bar  \alpha_{t-1}}}{1 - \bar  \alpha_t} x_0
\right) \\
\sigma_t^2 = \frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1 - \bar  \alpha_t}
$$

We're getting close. We now just need to deal with the main loss term which is a KL-divergence between two gaussians $D_\text{KL}(p(x_{t-1} | x_t; \theta)|| q(x_{t-1} | x_t, x_0))$ To make this explicit, let's use the parametrization $p(x_{t-1} | x_t; \theta) = \mathcal{N}(x_{t-1} | \mu_\theta(x_t, t), C_t)$, for some to-be-determined function $C_t$. From the previous equation, we now have:

$$
D_\text{KL}(\mathcal{N}(x_{t-1} | \mu_\theta(x_t, t), C_t)|| \mathcal{N}(x_{t-1} | \mu_t(x_t, x_0), \sigma_t^2))
$$

Using the formula for the KL-divergence of two gaussians with diagonal covariance:

$$
D_\text{KL}(q_1 || q_2) = \ln  \frac{\sigma_2}{\sigma_1} + \frac{\sigma_2^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

And keeping only terms with $\theta$ dependence, we find the per-time loss:

$$
L_{t-1} = \mathbb{E}_q\left[\frac{1}{2  \sigma_t^2}| \mu_\theta(x_t, t) - \mu_t(x_t, x_0) |^2\right]
$$

Note, it may have been confusing, but here we _integrated_ over $x_{t-1}$, but $x_t$ is considered a random variable, drawn from $q$. More explicitly, let's put all of the terms in here that we currently know of

$$
L_{t-1} = \mathbb{E}_q\left[\frac{1}{2  \sigma_t^2}\left|
\frac{(1-\bar  \alpha_{t-1})\sqrt{\alpha_t}}{(1 - \bar  \alpha_{t})} x_t +
\frac{(1-\alpha_t)\sqrt{\bar  \alpha_{t-1}}}{1 - \bar  \alpha_t} x_0 -
\mu_\theta(x_t, t)
\right|^2
\right]
$$

We can replace $x_0$ by using the relationship $x_0 = (x_t - \sqrt{1-\bar  \alpha_t} \epsilon) / \sqrt{\bar  \alpha_t}$. There more algebra to be done. The non $\theta$ parts are:

$$
\frac{(1 - \bar  \alpha_{t-1})\sqrt{\alpha_{t}}}{1 - \bar  \alpha_t} x_t+
\frac{(1 - \alpha_t)\sqrt{\bar  \alpha_{t-1}}}{1 - \bar  \alpha_t}\frac{1}{\sqrt{\bar  \alpha_t}} x_t-
\frac{(1 - \alpha_t)\sqrt{\bar  \alpha_{t-1}}}{1 - \bar  \alpha_t}\frac{\sqrt{1 - \bar  \alpha_t}}{\sqrt{\bar  \alpha_t}} \epsilon
$$

Now, we have to do a bunch of algebra.

$$
\left(
\frac{(1 - \bar  \alpha_{t-1})\sqrt{\alpha_{t}}}{1 - \bar  \alpha_t}+
\frac{(1 - \alpha_t)}{1 - \bar  \alpha_t}\frac{1}{\sqrt{\alpha_t}}
\right) x_t-
\frac{(1 - \alpha_t)\sqrt{\bar  \alpha_{t-1}}}{1 - \bar  \alpha_t}\frac{\sqrt{1 - \bar  \alpha_t}}{\sqrt{\bar  \alpha_t}} \epsilon
$$

Combine the $x_t$ into a single fraction

$$
\left(
\frac{(1 - \bar  \alpha_{t-1})\alpha_t + 1 - \alpha_t}{(1 - \bar  \alpha_t) \sqrt{\alpha_{t}}}
\right) x_t-
\frac{(1 - \alpha_t)\sqrt{\bar  \alpha_{t-1}}}{1 - \bar  \alpha_t}\frac{\sqrt{1 - \bar  \alpha_t}}{\sqrt{\bar  \alpha_t}} \epsilon
$$

Use that $\bar  \alpha_{t-1} \alpha_t = \bar  \alpha_t$ and eliminate some terms

$$
\left(
\frac{(1 - \bar  \alpha_{t})}{(1 - \bar  \alpha_t) \sqrt{\alpha_{t}}}
\right) x_t-
\frac{(1 - \alpha_t)\sqrt{\bar  \alpha_{t-1}}}{1 - \bar  \alpha_t}\frac{\sqrt{1 - \bar  \alpha_t}}{\sqrt{\bar  \alpha_t}} \epsilon
$$

eliminate the remaining parts on $x_t$

$$
\left(
\frac{1}{\sqrt{\alpha_{t}}}
\right) x_t-
\frac{(1 - \alpha_t)\sqrt{\bar  \alpha_{t-1}}}{1 - \bar  \alpha_t}\frac{\sqrt{1 - \bar  \alpha_t}}{\sqrt{\bar  \alpha_t}} \epsilon
$$

Use that $\bar  \alpha_{t-1} / \bar  \alpha_t = 1 / \alpha_t$

$$
\left(
\frac{1}{\sqrt{\alpha_{t}}}
\right) x_t-
\frac{(1 - \alpha_t)}{1 - \bar  \alpha_t}\frac{\sqrt{1 - \bar  \alpha_t}}{\sqrt{\alpha_t}} \epsilon
$$

elimination

$$
\left(
\frac{1}{\sqrt{\alpha_{t}}}
\right) x_t-
\frac{(1 - \alpha_t)}{\sqrt{1 - \bar  \alpha_t}\sqrt{\alpha_t}} \epsilon
$$

pull out the common factors and we're done

$$
\mu_t (x_t(x_0, \epsilon), t) = \frac{1}{\sqrt{\alpha_{t}}}
\left(
x_t-
\frac{\beta_t}{\sqrt{1 - \bar  \alpha_t}} \epsilon
\right)
$$

Why did we go through all of this trouble? Because through a clever parametrization of $\mu_\theta(x_t, t)$, we can rewrite it identically to the above, thereby forgetting about the $x_t$ term completely! We make the ansatz

$$
\mu_\theta(x_t, t) =
\frac{1}{\sqrt{\alpha_{t}}}
\left(
x_t-
\frac{\beta_t}{\sqrt{1 - \bar  \alpha_t}} \epsilon_\theta(x_t, t)
\right)
$$

Now, see how everything cancels

$$
L_{t-1} = \mathbb{E}_{x_0, \epsilon}\left[\frac{1}{2  \sigma_t^2}\left|
\frac{1}{\sqrt{\alpha_{t}}}
\left(
x_t-
\frac{\beta_t}{\sqrt{1 - \bar  \alpha_t}} \epsilon
\right)-
\frac{1}{\sqrt{\alpha_{t}}}
\left(
x_t-
\frac{\beta_t}{\sqrt{1 - \bar  \alpha_t}} \epsilon_\theta(x_t, t)
\right)
\right|^2
\right]
$$

$$
\begin{equation}
L_{t-1} = \mathbb{E}_{x_0, \epsilon}\left[\frac{\beta_t^2}{2  \sigma_t^2  \alpha_t (1 - \bar  \alpha_t)}\left|
\epsilon - \epsilon_\theta(x_t, t)
\right|^2
\right]
\end{equation}
$$

Given this form we can ignore the weighting coefficients as in Ho et. al. and we get the simple objective found therein. For convenience we also restate that one can sample from $x_t$ by using $x_t = \sqrt{\bar  \alpha_t} x_0 + \sqrt{1-\bar  \alpha_t} \epsilon$

### Intuition

We are essentially learning to denoise input, hence the name Denoising Diffusion Probabilistic Models. What this means in normal terms, is that you start with a sample data point, add some random noise to it, and then ask the model "hey can you recover the original input?".

There is one important difference though. We are also giving the model a hint about at what time the noise was applied. Intuitively, this allows the model to be fuzzy and reckless at early times (just point towards the center of probability mass!), yet be conservative and precise at later times (move slightly, the answer is almost correct!).

The schedule for DDPM has two limits of interest, small and large times, which back up our intuition. At small times, $\beta_0  \approx  0$ which means $\alpha_0  \approx  1$. The forward process transition kernel is then roughly $\mathcal{N}(x_{t-1}, 0) = \delta(x_{t-1} - x_t)$ In other words, as we get closer to our target, we decrease our uncertainty or "anneal" the sampling noise. In the other limit, we get roughty $\mathcal{N}(0, 1)$, meaning that our diffusion process makes almost no use of the memory of its last state, and we're essentially just hopping at random.

This separation of scales (which maybe doesn't sit so nicely with the path integral interpretation) was found to be crucial for the success of diffusion models. The differential equation extension allows us to formulate the noise as a continuous function of time, instead of a discrete one. Recall the previous derivation took a lot of painstaking algebra to ensure nice, closed forms for $q(x_{t-1}|x_t, x_0)$ and $q(x_t | x_0)$. This was a consequence of demanding discrete noise variances $0 < \beta_1 < \ldots < \beta_t$. The next insight, is to make this continuous. To do so though, we need to guarantee that we can still efficiently compute the forward transition probabilities, and that we can _reverse_ this process by sampling from our generative distribution.

### continuous

This is where the idea of stochastic differential equations comes into play. We need only define an intial (target) distribution $p_\text{data}(x)$ and a differential describing an infinitesmial forward step, then we can use a result by (cite some author... lol don't remember) to ensure that there exists a reverse sampling process with a learnable force function.

Specifically, given a forward process

$$
\begin{equation}
dx = f(x, t) dt + g(t) dB, \ \\ \\ \ \langle dB^2  \rangle = dt
\end{equation}
$$

There exists a backward process

$$
\begin{equation}
dx = \left[f(x, t) - g^2(t) \nabla_x  \log p(x|t)\right] dt + g(t)\ \ dB
\end{equation}
$$

Let's derive the backward process above. The Fokker Planck can be written as:

$$
\partial_t p = \nabla  \cdot (\mu(x, t) p) + \nabla^2 (D(x, t) p)
$$

There exists a stationary solution found by setting $\partial_t p = 0$, which gives the equation $\mu p = \nabla D p => p \sim e^{\int  \mu / D}$. If we suppose there is some function such that $\mu = -\nabla V$ and if $D(x, t) = D = 2k_\text{B}T$, then we find the familiar Boltzmann distribution from physics $p_\text{eq} \sim e^{-V / k_\text{B}T}$. In other words, by cleverly setting $\mu(x, t) = -D \nabla ln q(x)$, we have produced an equation whose equilibrium solution will be the target distribution $q(x)$ for any $q(x)$.

Why is this useful? After all, we can't easily parametrize a neural net to learn an arbitrary function $p(x, t)$, without makign some limiting assumptions (i.e. variational methods or normalizing flows). Well, it's useful because there is an equivalence between Fokker-Planck equations describing the time-evolution of the probability density $p(x,t)$, and the Langevin equation -- a stochastic ODE describing the trajectory $x(t)$ of a single particle, which when ensembled produces $p(x, t)$. A neural network can deal with a single realized trajectory, since that means we need only worry about one specific coordinate $x$ at any time $t$. And so, the langevin representation is conducive to generating samples from the Fokker-planck equation. Since we can also set the equilibrium solution, it means we can also sample from any distribution we want, provided we know how to take a forward step and we sample long enough.

#### Reverse diffusion

To briefly derive the FP equation from the Langevin equation, and thereby retrieve the reverse equation, we start with the differential relation

$$
dx = f(x, t) dt + g(t)dB
$$

where $\langle dB_i dB_j\rangle^2 = \delta_{ij} dt$. The expectation of any arbitrary function $F(x, t)$ can be written in two ways:

1. $\langle F(x, t) \rangle = \int F(x) p(x, t) dx$

2. $\langle F(x, t) \rangle = \int F(x, t) p(x) dx$

These are equivalent to the two views of quantum mechanics (schrodinger and heisenberg), where either the operator is time dependent and the states are stationary (option 2), or the states are time-dependent and the operator stationary (option 1). The relation is easy in bra-ket notation $\langle  \psi(t) | \hat O | \psi(t') \rangle = \langle  \psi| e^{i \hat H t} \hat O e^{-i \hat H t'} | \psi  \rangle = \langle  \psi| \hat O(t, t') | \psi  \rangle$. We'll use these two views to write the same thing, thereby deriving a new relation.

Let's go with view (2). We first taylor $F(x(t))$ to $\mathcal{O}(dx^2)$:

$$
\left  \langle dF \right  \rangle = \left  \langle  \frac{\partial F}{\partial x} dx \right  \rangle + \frac{1}{2} \left  \langle  \frac{\partial^2 F}{\partial x_i \partial x_j} dx_i dx_j \right  \rangle
$$

Substituting the differential relation and keeping only terms to $\mathcal{O}(dt)$ we find

$$
\left  \langle  \frac{\partial F}{\partial t} \right  \rangle = \frac{\partial F}{\partial x} f(x, t) + \frac{1}{2} \frac{\partial^2 F}{\partial x^2} g^2
$$

Taking a time derivative of view (1) and setting it equal to a time derivative of view (2) with the above substituted in for $\partial_t F$, we get the integral equation:

$$
\int F(x) \partial_t p(x, t) dx = \int dx\ \ p(x) \left[\frac{\partial F}{\partial x} f(x, t) + \frac{1}{2} \frac{\partial^2 F}{\partial x^2} g^2\right]
$$

Finally, we integrate by parts to find:

$$
\int F(x) \partial_t p(x, t) dx = \int dx\ \ F(x) \left[\partial_x(-f(x, t) p(x, t)) + \partial_x^2 (g^2 p(x, t)))\right]
$$

Since this must hold for any abitrary function $F(x)$, it must hold within the integrals as well, leading to the FP equation:

$$
\partial_t p(x, t) = \partial_x(-f(x, t) p(x, t)) + \partial_x^2 (g^2 p(x, t)))
$$

What's nice about this representation, is that it also allows us to compute the conjugate solution readily, whereby we instead integrate by parts on the left hand side. This gives the _reverse FP_ equation:

$$
\partial_t p_\text{rev}(x, t) = -f(x,t)\partial_x p_\text{rev}(x, t) - g^2  \partial_x^2 p_\text{rev}(x, t)
$$

## Equivalence of score-based models and diffusion models

Consider the simplest process we can write:

$$
dx = g(t) dB.
$$

These leads to a Fokker-Planck equation $\partial_t P = g^2(t) \nabla^2 P$, which has solution $\text{exp}(-(x-x')^2 / \int g^2(t)dt)$. Let's choose $g(t) = \sigma^t$. Integrating, we get variance $\text{Var}(t) = \frac{1}{2\ln \sigma}(\sigma^{2t} - 1)$ This defines a forward process transition probability:

$$
q(x(t) | x(0)) = \mathcal{N}\left(x(t)| x(0), \frac{\sigma^{2t} - 1}{2\ln \sigma}\right )
$$

Which can be evaluated at any time $t$. Note, that as $t \to \infty$, the variance explodes. Yang refers to these types of SDEs as **Variance Exploding (VE)** solutions. Yang then puts forth the following score matching objective

$$
L = \frac{1}{2} \mathbb{E}_{x \sim q(x|x_0), x_0 \sim p_\text{data}(x_0)}
\left[
  ||s_\theta(x) - \nabla \ln q(x | x_0) ||^2
\right]
$$

This is really similar to the one given by Jascha, Ho, etc., except that those guys use $q(x_{t-1} | x_t, x_0)$. Instead, we will show that the above is equivalent to the score-matching objective (i.e. $p(x)$ instead of $q(x|x')$ in the log). And, since we know there exists a langevin that can produce samples given the score, everything is just fine. The proof is actually not too bad, and comes from the appendix of

<cite>P. Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661–1674, 2011.</cite> [link]https://arxiv.org/pdf/1907.05600.pdf

Begin with the score matching objective, for some parametrizable function $s_\theta(x)$

$$
L = \frac{1}{2} \mathbb{E}_{x \sim p(x)} \left[||s_\theta(x) - \nabla \ln p(x)||^2\right] \\
L = \frac{1}{2}\int dx p(x) ||s_\theta(x) - \nabla \ln p(x)||^2 \\
L = \frac{1}{2}\int dx p(x) ||s_\theta(x)||^2  - 2 \int dx p(x) s_\theta(x) \nabla \ln p(x) + \text{const} \\
L = \frac{1}{2}\int dx p(x) ||s_\theta(x)||^2  - 2 \int dx s_\theta(x) \nabla p(x) + \text{const} \\
L = \frac{1}{2}\int dx p(x) ||s_\theta(x)||^2  - 2 \int dx dx' s_\theta(x) p(x')\nabla p(x|x') + \text{const} \\
L = \frac{1}{2}\int dx p(x) ||s_\theta(x)||^2  - 2 \int dx dx' s_\theta(x) p(x|x')p(x')\nabla \ln p(x|x') + \text{const} \\
L = \frac{1}{2}\int dx p(x) ||s_\theta(x)||^2  - 2 \int dx dx' s_\theta(x) p(x, x')\nabla \ln p(x|x') + \text{const} \\
L = \frac{1}{2}\int dx dx' p(x, x')||s_\theta(x) - \nabla \ln p(x|x')||^2 \\
L = \frac{1}{2}\mathbb{E}_{(x, x') \sim p(x, x')} \left[||s_\theta(x) - \nabla \ln p(x|x')||^2\right]
$$

when the transition kernel is gaussian, the final and initial positions cancel out after reparametrization, leaving only the noise for the score to match on $\approx \sigma_t^2 s_\theta(x, t) = \epsilon$

# The End

That wraps up diffusion models. There are lots of cool things to continue studying though. Supervised learning is one area, whereby we can push to reconstruct certain solutions. Class-conditioned generation is another, where we ask the model to generate samples from a given class. We can also consider multi-modality diffusion, which is used for making those really cool text-to-image generators.

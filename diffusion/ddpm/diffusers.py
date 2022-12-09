import torch
from torch import nn


class DDPM(nn.Module):
    """Dataclass to maintain the noise schedule in the DDPM procedure of discrete noise steps

    Mathematically, the transition kernel at time $t$ is defined by:
    $$
    q(x_t|x_{t-1}) = \mathcal{N}(x_t| \sqrt{\alpha_t} x_{t-1}, 1 - \alpha_t)
    $$

    We further define quantities $\beta$ and $\bar \alpha$ in terms $\alpha$:
    $$
    \beta_t \equiv 1 - \alpha_t
    $$
    $$
    \bar \alpha_t = \prod_{t' < t}\alpha_{t'}
    $$
    which will be useful later on when computing transitions between non adjacent times.
    """

    def __init__(self, n_steps: int, minval: float = 1e-5, maxval: float = 5e-3):
        super().__init__()
        assert 0 < minval < maxval <= 1
        assert n_steps > 0
        self.n_steps = n_steps
        self.minval = minval
        self.maxval = maxval
        self.register_buffer("beta", torch.linspace(minval, maxval, n_steps))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(0))

    def diffusion_loss(self, model: nn.Module, inp: torch.Tensor) -> torch.Tensor:
        device = inp.device
        batch_size = inp.shape[0]

        # create the noise perturbation
        eps = torch.randn_like(inp, device=device)

        # convert discrete time into a positional encoding embedding
        t = torch.randint(0, self.n_steps, (batch_size,), device=device)

        # compute the closed form sample x_noisy after t time steps
        a_t = self.alpha_bar[t][:, None]
        x_noisy = torch.sqrt(a_t) * inp + torch.sqrt(1 - a_t) * eps

        # predict the noise added given time t
        eps_pred = model(x_noisy, t)

        # Gaussian posterior, i.e. learn the Gaussian kernel.
        return nn.MSELoss()(eps_pred, eps)

    def sample(self, model: nn.Module, n_samples: int = 128):
        with torch.no_grad():
            device = next(model.parameters()).device

            # start off with an intial random ensemble of particles
            x = torch.randn(n_samples, 2, device=device)

            # the number of steps is fixed before beginning training. unfortunately.
            for t in reversed(range(self.n_steps)):
                # apply the same variance to all particles in the ensemble equally.
                a = self.alpha[t].repeat(n_samples)[:, None]
                abar = self.alpha_bar[t].repeat(n_samples)[:, None]

                # deterministic trajectory. eps_theta is similar to the Force on the particle
                eps_theta = model(x, torch.tensor([t] * n_samples, dtype=torch.long))
                x_mean = (x - eps_theta * (1 - a) / torch.sqrt(1 - abar)) / torch.sqrt(
                    a
                )
                sigma_t = torch.sqrt(1 - self.alpha[t])

                # sample a different realization of noise for each particle and propagate
                z = torch.randn_like(x)
                x = x_mean + sigma_t * z

            return x_mean  # clever way to skip the last noise addition

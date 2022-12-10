import torch
from torch import nn
import numpy as np


class PositionalEncoding(nn.Module):
    """The classic positional encoding from the original Attention papers"""

    def __init__(
        self,
        d_model: int = 128,
        maxlen: int = 1024,
        min_freq: float = 1e-4,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        """
        Args:
            d_model (int, optional): embedding dimension of each token. Defaults to 128.
            maxlen (int, optional): maximum sequence length. Defaults to 1024.
            min_freq (float, optional): use the magic 1/10,000 value! Defaults to 1e-4.
            device (str, optional): accelerator or nah. Defaults to "cpu".
            dtype (_type_, optional): torch dtype. Defaults to torch.float32.
        """
        super().__init__()
        pos_enc = self._get_pos_enc(d_model=d_model, maxlen=maxlen, min_freq=min_freq)
        self.register_buffer(
            "pos_enc", torch.tensor(pos_enc, dtype=dtype, device=device)
        )

    def _get_pos_enc(self, d_model: int, maxlen: int, min_freq: float):
        position = np.arange(maxlen)
        freqs = min_freq ** (2 * (np.arange(d_model) // 2) / d_model)
        pos_enc = position[:, None] * freqs[None]
        pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
        pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
        return pos_enc

    def forward(self, x):
        return self.pos_enc[x]


class GaussianFourierProjection(nn.Module):
    """Positional encoding for continuum states. Think how to embed
    functional dependence on a real-valued scalar, like f(x) -> f(x, t)
    for some scalar time variable t.

    This creates random Gaussian Fourier features. In fact, Random fourier
    Features have an interesting $N \to \infty$ limit for layer width $N$;
    They become Gaussian Processes!
    """

    def __init__(self, embed_dim: int, scale: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale
        self.W = torch.randn(self.embed_dim // 2) * self.scale

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * 3.1415927
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class DiscreteTimeResidualBlock(nn.Module):
    """Generic block to learn a nonlinear function f(x, t), where
    t is discrete and x is continuous."""

    def __init__(self, d_model: int, maxlen: int = 512):
        super().__init__()
        self.d_model = d_model
        self.emb = PositionalEncoding(d_model=d_model, maxlen=maxlen)
        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x, t):
        return self.norm(x + self.lin2(self.act(self.lin1(x + self.emb(t)))))


class BasicDiscreteTimeModel(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.lin_in = nn.Linear(2, d_model)
        self.lin_out = nn.Linear(d_model, 2)
        self.blocks = nn.ParameterList(
            [DiscreteTimeResidualBlock(d_model=d_model) for _ in range(n_layers)]
        )

    def forward(self, x, t):
        x = self.lin_in(x)
        for block in self.blocks:
            x = block(x, t)
        return self.lin_out(x)

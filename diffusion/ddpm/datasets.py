import torch
import numpy as np


class DataLoader:
    """A standard Dataloader for tensor datasets with fixed size. Yields only
    inputs, and not targets"""

    def __init__(
        self, x: np.ndarray, batch_size: int = 128, return_tensors: str = "pt"
    ):
        self.x = x
        self.bsz = batch_size
        self.return_tensors = return_tensors

    def __len__(self) -> int:
        return (len(self.x) // self.bsz) + 1 - (len(self.x) % self.bsz) // self.bsz

    def __iter__(self):
        ids = np.random.choice(len(self.x), len(self.x), replace=False)
        for i in range(0, len(self.x), self.bsz):
            curr = self.x[ids[i : i + self.bsz]]
            yield curr if self.return_tensors == "np" else torch.from_numpy(curr)


def load_N_gaussians(N: int = 3, n_per_class=100, eta: float = 2e-1):
    """Load a 2D dataset of Gaussian clusters located at angles of 2pi/N radians.

    Args:
        N (int, optional): number of clusters. Defaults to 3.
        n_per_class (int, optional): points per cluster. Defaults to 100.
        eta (float, optional): magnitude of noise about cluster center. Defaults to 2e-1.

    Returns:
        _type_: _description_
    """
    mu = np.concatenate(
        [
            np.array(
                [[np.cos(2 * np.pi * i / N), np.sin(2 * np.pi * i / N)]] * n_per_class
            )
            for i in range(1, N + 1)
        ]
    )
    data = mu + eta * np.random.randn(*mu.shape)
    return data.astype(np.float32)

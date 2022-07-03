import torch


def MSE(fake: torch.Tensor,
        image: torch.Tensor) -> torch.Tensor:
    return torch.mean((fake - image)**2)

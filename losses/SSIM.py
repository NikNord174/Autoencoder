import torch


def SSIM(x: torch.Tensor, y: torch.Tensor) -> float:
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x, var_x = torch.mean(x, dim=[1, 2, 3]), torch.var(x, dim=[1, 2, 3])
    mu_y, var_y = torch.mean(y, dim=[1, 2, 3]), torch.var(y, dim=[1, 2, 3])
    cov = torch.mean(x * y, dim=[1, 2, 3]) - mu_x * mu_y
    ssim_num = (2 * mu_x * mu_y + C1) * (2 * cov + C2)
    ssim_den = (mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2)
    ssim = 1 - (ssim_num / ssim_den + 1) / 2
    return torch.mean(ssim)

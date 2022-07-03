import torch


def SSIM(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    k1: float = 0.01
    k2: float = 0.03
    mean_fake = torch.mean(fake, dim=[1, 2, 3])
    mean_real = torch.mean(real, dim=[1, 2, 3])
    var_fake = torch.var(fake, dim=[1, 2, 3])
    var_real = torch.var(real, dim=[1, 2, 3])
    c1 = (k1) ** 2  # L=1, тк изображения нормализованы
    c2 = (k2) ** 2  # L=1, тк изображения нормализованы
    fake_dif = torch.sub(fake, mean_fake[:, None, None, None])
    real_dif = torch.sub(real, mean_real[:, None, None, None])
    covariance = (
        torch.sum(fake_dif * real_dif)
        / real.size()[-1] / real.size()[-2]
    )
    ssim_numerator = (
        (2 * mean_fake * mean_real + c1) * (2 * covariance + c2)
    )
    ssim_denominator = (
        ((mean_fake) ** 2 + (mean_real) ** 2 + c1)
        * ((var_fake) ** 2 + (var_real) ** 2 + c2)
    )
    ssim = ssim_numerator / ssim_denominator
    '''print('mean_fake:', mean_fake[0])
    print('mean_real:', mean_real[0])
    print('var_fake:', var_fake[0])
    print('var_real:', var_real[0])
    print('fake_dif:', fake_dif[0])
    print('real_dif:', real_dif[0])
    print('covariance:', covariance)
    print('ssim_numerator:', torch.mean(ssim_numerator))
    print('ssim_denominator:', torch.mean(ssim_denominator))
    print('ssim:', torch.mean(ssim))'''
    return torch.mean(ssim)

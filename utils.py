import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def np_and_save(image, image_name):
    image_cpu = image[0].detach().cpu()
    save_image(image_cpu, f'results/{image_name}.jpeg')
    image_np = image_cpu.numpy()
    return np.transpose(image_np, (1, 2, 0))


def illustration(fig, no, image, title, fontsize=28):
    ax = fig.add_subplot(1, 2, no)
    ax.set_title(title, fontsize=fontsize)
    ax.imshow(image)
    plt.axis('off')


def imshow(image, fake, image_name, fake_name):
    image_np = np_and_save(image, image_name)
    fake_np = np_and_save(fake, fake_name)
    fig = plt.figure(figsize=(15, 10))
    illustration(fig, 1, image_np, 'Real Image')
    illustration(fig, 2, fake_np, 'Fake Image')
    plt.show()

import os
from tqdm import tqdm

import hydra
import numpy as np
import torch
from torch import sqrt
from einops import rearrange
import matplotlib.pyplot as plt

from photo_gen.models.unet import UNET
from photo_gen.models.utils import DDPM_Scheduler
from timm.utils import ModelEmaV3
from utils import UNetPad, AttrDict, normalize
from train3 import display_reverse

from icecream import ic

bp = breakpoint


def guided_gen(init_image, cfg):
    checkpoint_path = os.path.expanduser(cfg.checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    image_shape = tuple(cfg.image_shape)
    num_time_steps = int(cfg.num_time_steps)
    if init_image.ndim > 2:
        n_images = init_image.shape[0]
    elif init_image.ndim == 2:
        n_images = 1
        init_image = init_image.unsqueeze(0)
    if isinstance(init_image, np.ndarray):
        init_image = torch.tensor(init_image)

    model = UNET().cuda()
    dtype = next(model.parameters()).dtype
    init_image = init_image.to(dtype)


    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=cfg.ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    images = []
    z = torch.randn((1, 1, ) + image_shape)
    padding_fn = UNetPad(z, depth=model.num_layers//2)
    padded_image_shape = padding_fn(z).shape[-2:]
    init_image = padding_fn(init_image)

    with torch.no_grad():
        samples = []
        model = ema.module.eval()
        for i in tqdm(range(n_images)):
            z = init_image[i].unsqueeze(0)
            ic(z.shape, z.min(), z.max())

            for t in reversed(range(1, num_time_steps)):
                t = [t]
                temp = (scheduler.beta[t]/((sqrt(1-scheduler.alpha[t]))
                                           * (sqrt(1-scheduler.beta[t]))))
                z = (
                    1/(sqrt(1-scheduler.beta[t])))*z
                - (temp*model(z.cuda(), t).cpu())
                if t[0] in times:
                    images.append(z)
                e = torch.randn((1, 1,) + padded_image_shape)
                z = z + (e*sqrt(scheduler.beta[t]))
            temp = scheduler.beta[0]/((sqrt(1-scheduler.alpha[0]))
                                      * (sqrt(1-scheduler.beta[0])))
            x = (1/(sqrt(1-scheduler.beta[0]))) * \
                z - (temp*model(z.cuda(), [0]).cpu())

            samples.append(x)
            images.append(x)
            x = rearrange(x.squeeze(0), 'c h w -> h w c').detach()
            x = x.numpy()
            display_reverse(images, cfg.savepath, i)
            images = []
    samples = torch.concat(samples, dim=0)
    samples = padding_fn.inverse(samples).squeeze()
    samples = samples.cpu().numpy()
    samples = (samples - samples.min()) / (samples.max() - samples.min())
    np.save(os.path.join(cfg.savepath, "images.npy"), samples)
    return samples


def comparison_test(guide, noisy_guide, guide_generated, random_generated, savepath, idx=""):
    print("Dot prod of guide with guided gen and random gen")
    gflat = guide.flatten()
    ggflat = guide_generated.flatten()
    rgflat = random_generated.flatten()
    p1 = np.dot(gflat, ggflat)
    p2 = np.dot(gflat, rgflat)
    print(p1, p2)
    print("ref dot(guide, guide)", np.dot(gflat, gflat))
    print("Distance of guide with guided gen and random gen")
    d1 = np.linalg.norm(gflat - ggflat)
    d2 = np.linalg.norm(gflat - rgflat)
    print(d1, d2)

    _, axes = plt.subplots(1, 4)
    axes[0].imshow(guide)
    axes[0].set_title("guide")
    axes[0].axis("off")
    axes[1].imshow(noisy_guide)
    axes[1].set_title("noisy guide")
    axes[1].axis("off")
    axes[2].imshow(guide_generated)
    axes[2].set_title("guide gen")
    axes[2].axis("off")
    axes[3].imshow(random_generated)
    axes[3].set_title("random gen")
    axes[3].axis("off")
    plt.savefig("gen_comparison" + idx + ".png")


# @hydra.main(config_path=".", config_name="config")
def main(cfg):
    global debug
    debug = cfg.debug
    guide_file = os.path.expanduser(cfg.guide_file)
    guide = torch.tensor(np.load(guide_file))
    for i in range(4):
        scale = 1 ** (-i)
        noise = torch.randn(1, 1, 101, 91)
        noisy_guide = scale * guide + noise
        # noisy_guide = normalize(noisy_guide)
        guided_gen_image = guided_gen(noisy_guide, cfg).squeeze()
        rand_guide_image = guided_gen(noise, cfg).squeeze()
        comparison_test(guide.squeeze(), noisy_guide.squeeze(),
                guided_gen_image, rand_guide_image, cfg.savepath, str(i))


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    config = AttrDict(config)
    main(config)

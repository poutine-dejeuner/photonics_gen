"""
Wasserstein GAN with Gradient Penalty (WGAN-GP) implementation for nanophotonics design generation.
"""
import os
import torch
from tqdm import tqdm
import random

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from photo_gen.utils.utils import set_seed
from photo_gen.utils.parameter_counting import count_parameters


class Generator(nn.Module):
    """WGAN Generator network."""

    def __init__(self, latent_dim: int = 100, img_channels: int = 1,
                 img_size: tuple[int, int] = (101, 91), hidden_dim: int = 64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels

        self.img_height, self.img_width = img_size

        self.init_size = max(4, min(self.img_height, self.img_width) // 16)
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 4 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(hidden_dim * 4),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim * 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim, img_channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.final_resize = nn.AdaptiveAvgPool2d(
            (self.img_height, self.img_width))

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        img = self.final_resize(img)
        return img


class Critic(nn.Module):
    """WGAN Critic (Discriminator) network."""

    def __init__(self, img_channels: int = 1, img_size: tuple = (101, 91), hidden_dim: int = 64):
        super(Critic, self).__init__()

        if isinstance(img_size, (tuple, list)):
            self.img_height, self.img_width = img_size
        else:
            self.img_height = self.img_width = img_size

        def critic_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers for the critic"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *critic_block(img_channels, hidden_dim, normalization=False),
            *critic_block(hidden_dim, hidden_dim * 2),
            *critic_block(hidden_dim * 2, hidden_dim * 4),
            *critic_block(hidden_dim * 4, hidden_dim * 8),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.adv_layer = nn.Sequential(
            nn.Linear(hidden_dim * 8, 1)
        )

    def forward(self, img):
        out = self.model(img)
        out = self.global_pool(out)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


class WGAN:
    """WGAN-GP model wrapper."""

    def __init__(self, latent_dim: int = 100, img_channels: int = 1,
                 img_size: tuple = (101, 91), hidden_dim: int = 64, device: str = 'cuda'):
        self.latent_dim = latent_dim
        self.device = device

        self.generator = Generator(
            latent_dim, img_channels, img_size, hidden_dim).to(device)
        self.critic = Critic(img_channels, img_size, hidden_dim).to(device)

    def compute_gradient_penalty(self, real_samples, fake_samples, batch_size):
        """Calculates the gradient penalty loss for WGAN GP"""

        if real_samples.shape != fake_samples.shape:
            print("Shape mismatch detected! Resizing fake samples to match real samples.")
            fake_samples = torch.nn.functional.interpolate(
                fake_samples, size=real_samples.shape[2:], mode='bilinear', align_corners=False
            )
            print(f"Resized fake samples shape: {fake_samples.shape}")

        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha)
                        * fake_samples)).requires_grad_(True)

        d_interpolates = self.critic(interpolates)
        fake = torch.ones(batch_size, 1).to(self.device)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


def train(data: np.ndarray, cfg, checkpoint_path: str, savedir: str, run=None):
    """Training function for WGAN-GP."""

    n_epochs = cfg.model.n_epochs
    batch_size = cfg.model.batch_size
    lr_g = cfg.model.lr_g
    lr_c = cfg.model.lr_c
    latent_dim = cfg.model.latent_dim
    n_critic = cfg.model.n_critic
    lambda_gp = cfg.model.lambda_gp
    seed = cfg.model.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training WGAN on {device}")
    print(f"{n_epochs} epochs total")

    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    data = torch.tensor(data, dtype=torch.float32)
    if len(data.shape) == 3:
        data = data.unsqueeze(1)

    print(f"Data tensor shape after processing: {data.shape}")

    if data.max() > 1.0:
        data = (data - data.min()) / (data.max() - data.min())

    img_height, img_width = data.shape[2], data.shape[3]
    img_channels = data.shape[1]

    print(f"Image dimensions: {img_channels}x{img_height}x{img_width}")

    wgan = WGAN(latent_dim=latent_dim, img_channels=img_channels,
                img_size=(img_height, img_width), device=device)

    N = cfg.n_model_parameters
    total_params = count_parameters(
        wgan.generator) + count_parameters(wgan.critic)
    assert (total_params - N) < 0.14 * N

    optimizer_G = optim.Adam(
        wgan.generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
    optimizer_C = optim.Adam(wgan.critic.parameters(),
                             lr=lr_c, betas=(0.5, 0.9))

    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=4)

    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device,
                                weights_only=False)
        wgan.generator.load_state_dict(checkpoint['generator'])
        wgan.critic.load_state_dict(checkpoint['critic'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_C.load_state_dict(checkpoint['optimizer_C'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, n_epochs):
        epoch_g_loss = 0
        epoch_c_loss = 0
        g_updates = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")

        for i, (real_imgs,) in enumerate(pbar):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.shape[0]

            optimizer_C.zero_grad()

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = wgan.generator(z)

            real_validity = wgan.critic(real_imgs)
            fake_validity = wgan.critic(fake_imgs)

            gradient_penalty = wgan.compute_gradient_penalty(
                real_imgs, fake_imgs, batch_size)

            c_loss = -torch.mean(real_validity) + \
                torch.mean(fake_validity) + lambda_gp * gradient_penalty

            c_loss.backward()
            optimizer_C.step()

            epoch_c_loss += c_loss.item()

            if i % n_critic == 0:
                optimizer_G.zero_grad()

                z = torch.randn(batch_size, latent_dim).to(device)
                fake_imgs = wgan.generator(z)

                g_loss = -torch.mean(wgan.critic(fake_imgs))

                g_loss.backward()
                optimizer_G.step()

                epoch_g_loss += g_loss.item()
                g_updates += 1

            pbar.set_postfix({
                'C_loss': f"{c_loss.item():.4f}",
                'G_loss': f"{g_loss.item():.4f}" if i % n_critic == 0 else "N/A"
            })

            if cfg.get('debug', False):
                break

        batch_count = max(1, len(dataloader))

        if run is not None:
            run.log({
                'epoch': epoch,
                'critic_loss': epoch_c_loss / batch_count,
                'generator_loss': epoch_g_loss / max(1, g_updates),
            })

        print(
            f"Epoch {epoch+1}: C_loss={epoch_c_loss / batch_count:.4f}, G_loss={epoch_g_loss / max(1, g_updates):.4f}")

        if cfg.get('debug', False):
            print("Debug mode: stopping after 1 epoch")
            # Save checkpoint in debug mode before breaking
            checkpoint = {
                'epoch': epoch,
                'generator': wgan.generator.state_dict(),
                'critic': wgan.critic.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_C': optimizer_C.state_dict(),
                'config': {
                    'latent_dim': latent_dim,
                    'img_channels': img_channels,
                    'img_size': (img_height, img_width),
                    'hidden_dim': getattr(cfg.model, 'hidden_dim', 64),
                    'n_critic': n_critic,
                    'lambda_gp': lambda_gp,
                    'lr_c': lr_c,
                    'lr_g': lr_g,
                    'batch_size': batch_size
                }
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'generator': wgan.generator.state_dict(),
                'critic': wgan.critic.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_C': optimizer_C.state_dict(),
                'config': {
                    'latent_dim': latent_dim,
                    'img_channels': img_channels,
                    'img_size': (img_height, img_width),
                    'hidden_dim': getattr(cfg.model, 'hidden_dim', 64),
                    'n_critic': n_critic,
                    'lambda_gp': lambda_gp,
                    'lr_c': lr_c,
                    'lr_g': lr_g,
                    'batch_size': batch_size
                }
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

        if (epoch + 1) % 20 == 0:
            wgan.generator.eval()
            with torch.no_grad():
                z = torch.randn(16, latent_dim).to(device)
                sample_imgs = wgan.generator(z)

                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for idx, ax in enumerate(axes.flat):
                    img = sample_imgs[idx].cpu().squeeze().numpy()
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(
                    savedir, f'samples_epoch_{epoch+1}.png'))
                plt.close()

            wgan.generator.train()

    print("Training completed!")


def inference(checkpoint_path: str, savepath: str, cfg):
    """Generate samples using trained WGAN."""

    n_samples = cfg.get('n_to_generate', 1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    model_cfg = checkpoint['config']

    img_size = model_cfg.get('img_size', 64)
    if hasattr(img_size, '_content'):
        img_size = tuple(img_size)
    elif isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        img_size = tuple(img_size)

    wgan = WGAN(latent_dim=model_cfg['latent_dim'],
                img_channels=model_cfg.get('img_channels', 1),
                img_size=img_size,
                device=device)

    wgan.generator.load_state_dict(checkpoint['generator'])
    wgan.generator.eval()

    print(f"Generating {n_samples} samples...")

    generated_images = []
    batch_size = 64

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size)):
            current_batch_size = min(batch_size, n_samples - i)
            z = torch.randn(current_batch_size,
                            model_cfg['latent_dim']).to(device)
            batch_imgs = wgan.generator(z)
            generated_images.append(batch_imgs.cpu().numpy())

    generated_images = np.concatenate(generated_images, axis=0)

    np.save(os.path.join(savepath, "images.npy"), generated_images)

    return generated_images


if __name__ == "__main__":
    train(np.random.rand(16, 101, 91), "debug", "debug")

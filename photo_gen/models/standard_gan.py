"""
Standard GAN implementation for nanophotonics design generation.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from typing import Optional
from photo_gen.utils.utils import set_seed
from photo_gen.utils.parameter_counting import count_parameters


class Generator(nn.Module):
    """Standard GAN Generator network."""
    
    def __init__(self, latent_dim: int = 100, img_channels: int = 1, 
                 img_size: tuple = (101,91), hidden_dim: int = 64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        
            
        self.img_size = img_size
        self.img_height, self.img_width = img_size
        
        self.init_size = max(4, min(self.img_height, self.img_width) // 16)
        self.l1 = nn.Sequential(nn.Linear(latent_dim, hidden_dim * 4 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(hidden_dim * 4),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim, img_channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        self.final_resize = nn.AdaptiveAvgPool2d((self.img_height, self.img_width))

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        img = self.final_resize(img)
        return img


class Discriminator(nn.Module):
    """Standard GAN Discriminator network."""
    
    def __init__(self, img_channels: int = 1, img_size:tuple = (101,91), hidden_dim: int = 64):
        super(Discriminator, self).__init__()
            
        self.img_size = img_size
        self.img_height, self.img_width = img_size
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(img_channels, hidden_dim, normalization=False),
            *discriminator_block(hidden_dim, hidden_dim * 2),
            *discriminator_block(hidden_dim * 2, hidden_dim * 4),
            *discriminator_block(hidden_dim * 4, hidden_dim * 8),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.adv_layer = nn.Sequential(
            nn.Linear(hidden_dim * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = self.global_pool(out)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


class StandardGAN:
    """Standard GAN model wrapper."""
    
    def __init__(self, latent_dim: int = 100, img_channels: int = 1, 
                 img_size:tuple = (101,91), hidden_dim: int = 64, device: str = 'cuda'):
        self.latent_dim = latent_dim
        self.device = device
        
        self.generator = Generator(latent_dim, img_channels, img_size, hidden_dim).to(device)
        self.discriminator = Discriminator(img_channels, img_size, hidden_dim).to(device)


def train(data: np.ndarray, cfg, checkpoint_path: str, savedir: str, run=None):
    """Training function for standard GAN."""
    
    n_epochs = cfg.model.n_epochs
    batch_size = cfg.model.batch_size
    lr_g = cfg.model.lr_g
    lr_c = cfg.model.lr_c
    latent_dim = cfg.model.latent_dim
    label_smoothing = cfg.model.get('label_smoothing', 0.0)
    seed = cfg.model.seed
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Standard GAN on {device}")
    print(f"{n_epochs} epochs total")
    
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)
    
    data = torch.tensor(data, dtype=torch.float32)
    if len(data.shape) == 3:
        data = data.unsqueeze(1)
    
    if data.max() > 1.0:
        data = (data - data.min()) / (data.max() - data.min())
    
    img_size = (data.shape[-2], data.shape[-1])
    img_channels = data.shape[1]
    
    gan = StandardGAN(latent_dim=latent_dim, img_channels=img_channels, 
                     img_size=img_size, device=device)
    
    N = cfg.n_model_parameters
    total_params = count_parameters(gan.generator) + count_parameters(gan.discriminator)
    assert (total_params - N) < 0.14 * N
    
    adversarial_loss = nn.BCELoss()
    
    optimizer_G = optim.Adam(gan.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(gan.discriminator.parameters(), lr=lr_c, betas=(0.5, 0.999))
    
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           drop_last=True, num_workers=4)
    
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        gan.generator.load_state_dict(checkpoint['generator'])
        gan.discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, n_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for i, (real_imgs,) in enumerate(pbar):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.shape[0]
            
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)
            
            if label_smoothing > 0:
                valid = valid - label_smoothing * torch.rand_like(valid)
            
            optimizer_G.zero_grad()
            
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_imgs = gan.generator(z)
            
            g_loss = adversarial_loss(gan.discriminator(gen_imgs), valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            optimizer_D.zero_grad()
            
            real_loss = adversarial_loss(gan.discriminator(real_imgs), valid)
            
            fake_loss = adversarial_loss(gan.discriminator(gen_imgs.detach()), fake)
            
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            pbar.set_postfix({
                'D_loss': f"{d_loss.item():.4f}",
                'G_loss': f"{g_loss.item():.4f}"
            })
            
            if cfg.get('debug', False):
                break
        
        if run is not None:
            batch_count = max(1, len(dataloader)) if not cfg.get('debug', False) else max(1, 1)
            run.log({
                'epoch': epoch,
                'discriminator_loss': epoch_d_loss / batch_count,
                'generator_loss': epoch_g_loss / batch_count,
            })
        
        batch_count = max(1, len(dataloader)) if not cfg.get('debug', False) else max(1, 1)
        print(f"Epoch {epoch+1}: D_loss={epoch_d_loss / batch_count:.4f}, G_loss={epoch_g_loss / batch_count:.4f}")
        
        if cfg.get('debug', False):
            print("Debug mode: stopping after 1 epoch")
            # Save checkpoint in debug mode before breaking
            checkpoint = {
                'epoch': epoch,
                'generator': gan.generator.state_dict(),
                'discriminator': gan.discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'config': {
                    'latent_dim': latent_dim,
                    'img_channels': img_channels,
                    'img_size': img_size,
                    'hidden_dim': cfg.model.hidden_dim,
                    'lr_g': lr_g,
                    'lr_c': lr_c,
                    'label_smoothing': label_smoothing,
                    'batch_size': batch_size
                }
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'generator': gan.generator.state_dict(),
                'discriminator': gan.discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'config': {
                    'latent_dim': latent_dim,
                    'img_channels': img_channels,
                    'img_size': img_size,
                    'hidden_dim': cfg.model.hidden_dim,
                    'lr_g': lr_g,
                    'lr_c': lr_c,
                    'label_smoothing': label_smoothing,
                    'batch_size': batch_size
                }
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        if (epoch + 1) % 20 == 0:
            gan.generator.eval()
            with torch.no_grad():
                z = torch.randn(16, latent_dim).to(device)
                sample_imgs = gan.generator(z)
                
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for idx, ax in enumerate(axes.flat):
                    img = sample_imgs[idx].cpu().squeeze().numpy()
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(savedir, f'samples_epoch_{epoch+1}.png'))
                plt.close()
            
            gan.generator.train()
    
    print("Training completed!")


def inference(checkpoint_path: str, savepath: str, cfg):
    """Generate samples using trained standard GAN."""
    n_samples = cfg.get('n_to_generate', 1000)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint['config']
    
    gan = StandardGAN(latent_dim=model_cfg['latent_dim'], 
                     img_channels=model_cfg.get('img_channels', 1),
                     img_size=model_cfg.get('img_size', 64), 
                     device=device)
    
    gan.generator.load_state_dict(checkpoint['generator'])
    gan.generator.eval()
    
    print(f"Generating {n_samples} samples...")
    
    generated_images = []
    batch_size = 64
    
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size)):
            current_batch_size = min(batch_size, n_samples - i)
            z = torch.randn(current_batch_size, model_cfg['latent_dim']).to(device)
            batch_imgs = gan.generator(z)
            generated_images.append(batch_imgs.cpu().numpy())
    
    generated_images = np.concatenate(generated_images, axis=0)
    
    np.save(os.path.join(savepath, "images.npy"), generated_images)
    
    return generated_images

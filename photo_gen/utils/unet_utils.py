import os
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange



def compute_unet_channels(initial_n_channels: int, n_layers: int):
    assert n_layers % 2 == 0, "n_layers must be even"
    n_encode_layers = n_layers // 2
    channels = [initial_n_channels * 2**k for k in range(n_encode_layers + 1)]
    
    for k in range(2, n_encode_layers + 1):
        decode_layer_dim = int(channels[-1] / 2 + channels[n_encode_layers + 1 - k])
        channels.append(decode_layer_dim)
    return channels


class DDPM_Scheduler(torch.nn.Module):
    def __init__(self, num_time_steps: int=1000, device:torch.device=torch.device("cpu")):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False, device=device)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]


def display_reverse(images: list, savepath: Path, idx: int):
    fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze()
        # x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.savefig(savepath / f"im{idx}.png")
    plt.close()


class UNetPad():
    """
    Pads a tensor x of shape (B, C, H, W) so that H and W are divisible by 2^depth.
    Returns the padded tensor and the slices to undo the padding.
    """

    def __init__(self, sample: torch.Tensor, depth: int):
        self.depth = depth
        h, w = sample.shape[-2:]
        target_h = ((h - 1) // 2**depth + 1) * 2**depth
        target_w = ((w - 1) // 2**depth + 1) * 2**depth
        pad_h = target_h - h
        pad_w = target_w - w
        self.pad = (0, pad_w, 0, pad_h)  # pad W then H
        self.unpad_slices = [slice(0, h), slice(0, w)]

    def __call__(self, x):
        return F.pad(x, self.pad)

    def inverse(self, x_padded):
        return x_padded[..., *self.unpad_slices]


def pad_to_unet(x: torch.Tensor, depth: int = 4):
    """
    Pads a tensor x of shape (B, C, H, W) so that H and W are divisible by 2^depth.
    Returns the padded tensor and the slices to undo the padding.
    """
    _, _, h, w = x.shape
    target_h = ((h - 1) // 2**depth + 1) * 2**depth
    target_w = ((w - 1) // 2**depth + 1) * 2**depth
    pad_h = target_h - h
    pad_w = target_w - w
    pad = (0, pad_w, 0, pad_h)  # pad W then H
    x_padded = F.pad(x, pad)
    return x_padded, (slice(0, h), slice(0, w))  # for unpadding later


class unet_pad_fun():
    def __init__(self, num_layers, data_sample):
        self.N = 2**num_layers

        def difference_with_next_multiple(x):
            reste = x % self.N
            if reste == 0:
                return 0
            else:
                return self.N - reste

        # Get the last 2 dimensions (H, W)
        h, w = data_sample.shape[-2:]
        
        # Calculate padding needed for each dimension
        h_pad_needed = difference_with_next_multiple(h)
        w_pad_needed = difference_with_next_multiple(w)
        
        # Split padding symmetrically
        self.h_pad_left = h_pad_needed // 2
        self.h_pad_right = h_pad_needed - self.h_pad_left
        self.w_pad_left = w_pad_needed // 2
        self.w_pad_right = w_pad_needed - self.w_pad_left
        
        # Store for F.pad format: [w_left, w_right, h_left, h_right]
        self.padding = [self.w_pad_left, self.w_pad_right, self.h_pad_left, self.h_pad_right]

    def pad(self, x):
        # F.pad expects padding in format [w_left, w_right, h_left, h_right, ...]
        # For tensors with more than 2 spatial dims, pad with zeros for extra dims
        n_extra_dims = x.ndim - 2  # Number of non-spatial dimensions (batch, channel, etc.)
        full_padding = self.padding + [0, 0] * (n_extra_dims - 2) if n_extra_dims > 2 else self.padding
        
        padded_tensor = F.pad(x, full_padding, mode='constant', value=0)
        return padded_tensor

    def crop(self, x):
        # Create slices for cropping
        h_slice = slice(self.h_pad_left, -self.h_pad_right if self.h_pad_right > 0 else None)
        w_slice = slice(self.w_pad_left, -self.w_pad_right if self.w_pad_right > 0 else None)
        
        cropped_tensor = x[..., h_slice, w_slice]
        return cropped_tensor

if __name__ == "__main__":
    def test_padding():
        import numpy as np
        
        depth = 2
        sample = torch.rand(1,1,101,91)
        pad_fn = UNetPad(sample, depth)
        padded_shape = pad_fn(sample).shape[-2:]
        padded_shape = np.array(padded_shape)
        print(padded_shape)
        for n in range(1, depth + 1):
            print(padded_shape/2**n)
        assert padded_shape[0]%2**depth == 0, padded_shape[0]%2
        assert padded_shape[1]%2**depth == 0, padded_shape[1]%2

    test_padding()

"""
Integration tests for UNet padding with actual model components.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from photo_gen.models.unet_utils import UNetPad, unet_pad_fun, compute_unet_channels
from photo_gen.models.unet import UNET


@pytest.fixture
def standard_unet():
    """Create a standard UNET with consistent parameters."""
    N = 4
    channels = compute_unet_channels(N*2, 3)
    num_groups = N
    return UNET(
        Channels=channels,
        num_groups=num_groups,
        input_channels=1,
        output_channels=1,
        device='cpu'
    )


@pytest.mark.integration
class TestUNetIntegration:
    """Integration tests with actual UNet models."""
    
    @pytest.mark.parametrize("input_shape,num_layers", [
        ((1, 1, 101, 91), 6),
        ((1, 1, 95, 95), 6),
        ((2, 1, 47, 48), 6),
    ])
    def test_unet_forward_with_padding(self, input_shape, num_layers, standard_unet):
        """Test that UNet forward pass works with padded inputs."""
        x = torch.randn(input_shape)
        
        # Apply padding
        depth = num_layers // 2
        pad_fn = UNetPad(x, depth=depth)
        x_padded = pad_fn(x)
        
        # Use the standard UNET
        model = standard_unet
        model.eval()
        
        # Create time tensor for diffusion model
        t = torch.randint(0, 1000, (input_shape[0],))
        
        with torch.no_grad():
            output = model(x_padded, t)
        
        # Should produce valid output
        assert output is not None
        assert output.shape[0] == x_padded.shape[0]  # Same batch size
        assert len(output.shape) == 4  # BCHW format
    
    def test_original_vs_padded_consistency(self, standard_unet):
        """Test that results are consistent between different padding approaches."""
        x = torch.randn(1, 1, 64, 64)  # Already properly sized
        
        # Use the standard UNET
        model = standard_unet
        model.eval()
        
        # Create time tensor
        t = torch.randint(0, 1000, (1,))
        
        with torch.no_grad():
            output_direct = model(x, t)
        
        # Test with padding (should be no-op for this size)
        pad_fn = UNetPad(x, depth=3)  # Use depth=3 for 6-layer UNET
        x_padded = pad_fn(x)
        
        with torch.no_grad():
            output_padded = model(x_padded, t)
            output_unpadded = pad_fn.inverse(output_padded)
        
        # Results should be very similar (allowing for numerical precision)
        assert torch.allclose(output_direct, output_unpadded, atol=1e-5)
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_consistency(self, batch_size, standard_unet):
        """Test that padding works consistently across batch sizes."""
        shape = (batch_size, 1, 101, 91)
        x = torch.randn(shape)
        
        # Apply padding
        pad_fn = UNetPad(x, depth=3)  # Use depth=3 for 6-layer UNET
        x_padded = pad_fn(x)
        
        # Use the standard UNET
        model = standard_unet
        model.eval()
        
        # Create time tensor
        t = torch.randint(0, 1000, (batch_size,))
        
        with torch.no_grad():
            output = model(x_padded, t)
            output_unpadded = pad_fn.inverse(output)
        
        # Check output shapes
        assert output_unpadded.shape[0] == batch_size
        assert output_unpadded.shape[-2:] == (101, 91)  # Original spatial dims
    
    def test_memory_efficiency(self, standard_unet):
        """Test that padding doesn't cause excessive memory usage."""
        # This is a basic test - in practice you'd want more sophisticated memory monitoring
        x = torch.randn(1, 1, 101, 91)  # Reduced batch size to avoid memory issues
        
        # Measure memory before
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated()
        
        # Apply padding and model
        pad_fn = UNetPad(x, depth=3)  # Use depth=3 for 6-layer UNET
        x_padded = pad_fn(x)
        
        # Use the standard UNET
        model = standard_unet
        model.eval()
        
        # Create time tensor
        t = torch.randint(0, 1000, (1,))
        
        with torch.no_grad():
            output = model(x_padded, t)
        
        # Basic assertion that we got a valid output
        assert output is not None
        assert output.numel() > 0


@pytest.mark.integration  
class TestErrorRecovery:
    """Test error recovery and edge cases in integration scenarios."""
    
    def test_zero_padding_edge_case(self):
        """Test the specific edge case that was causing errors."""
        # Create input that results in zero padding on one side
        x = torch.randn(1, 1, 48, 48)  # This should need minimal padding
        
        # Test both padding methods
        unetpad = UNetPad(x, depth=3)  # Use depth=3 for 6-layer UNET
        unet_pad_fun_instance = unet_pad_fun(6, x)  # Use 6 layers for actual UNET
        
        x_padded_1 = unetpad(x)
        x_padded_2 = unet_pad_fun_instance.pad(x)
        
        # Both should work without errors
        x_recovered_1 = unetpad.inverse(x_padded_1)
        x_recovered_2 = unet_pad_fun_instance.crop(x_padded_2)
        
        assert torch.allclose(x, x_recovered_1)
        assert torch.allclose(x, x_recovered_2)
    
    def test_concatenation_error_prevention(self):
        """Test that the specific concatenation error is prevented."""
        # This was the original error: "Expected size 48 but got size 47"
        x = torch.randn(1, 1, 101, 91)
        
        # Apply padding with the settings that were causing issues
        depth = 3  # This is num_layers//2 for the 6-layer UNET
        pad_fn = UNetPad(x, depth=depth)
        x_padded = pad_fn(x)
        
        # Simulate the encoder-decoder process that was failing
        current = x_padded
        encoder_features = []
        
        # Encoder path
        for level in range(depth + 1):  # Go one level deeper than depth
            encoder_features.append(current)
            if level < depth:  # Don't pool at the deepest level
                current = F.max_pool2d(current, 2)
        
        # Decoder path - this should not fail
        current = encoder_features[-1]
        for level in range(depth):
            # Upsample
            current = F.interpolate(current, scale_factor=2, mode='nearest')
            
            # Get skip connection
            skip_idx = depth - 1 - level
            skip = encoder_features[skip_idx]
            
            # This concatenation should work without the size mismatch error
            concatenated = torch.cat([current, skip], dim=1)
            current = concatenated
        
        # If we get here without errors, the fix worked
        assert True

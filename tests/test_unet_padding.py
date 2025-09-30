#!/usr/bin/env python3
"""
Pytest tests for UNet padding utilities.

This module contains comprehensive tests for the UNetPad and unet_pad_fun classes
to ensure proper padding and cropping functionality for UNet models.
"""

import pytest
import torch
import torch.nn.functional as F
import sys
import os

# Add the project directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from photo_gen.models.unet_utils import UNetPad, unet_pad_fun


class TestUNetPad:
    """Test suite for the UNetPad class."""
    
    @pytest.mark.parametrize("shape,depth", [
        ((1, 1, 101, 91), 2),   # Original problematic case
        ((1, 1, 101, 91), 3),   # Different depth
        ((1, 1, 101, 91), 5),   # Deeper
        ((2, 3, 64, 64), 4),    # Square, already divisible
        ((1, 1, 47, 48), 4),    # One divisible, one not
        ((1, 1, 100, 100), 4),  # Even numbers
        ((1, 1, 99, 97), 4),    # Odd numbers
    ])
    def test_divisibility(self, shape, depth):
        """Test if UNetPad ensures divisibility by 2^depth."""
        x = torch.randn(shape)
        pad_fn = UNetPad(x, depth=depth)
        x_padded = pad_fn(x)
        
        # Check divisibility
        h_padded, w_padded = x_padded.shape[-2:]
        required_divisor = 2**depth
        
        assert h_padded % required_divisor == 0, f"Height {h_padded} not divisible by {required_divisor}"
        assert w_padded % required_divisor == 0, f"Width {w_padded} not divisible by {required_divisor}"
    
    @pytest.mark.parametrize("shape,depth", [
        ((1, 1, 101, 91), 4),
        ((2, 3, 64, 64), 4),
        ((1, 1, 47, 48), 4),
        ((1, 1, 16, 16), 4),    # Already divisible
        ((1, 1, 1, 1), 4),      # Very small
    ])
    def test_inverse_operation(self, shape, depth):
        """Test if the inverse operation correctly recovers the original tensor."""
        x = torch.randn(shape)
        pad_fn = UNetPad(x, depth=depth)
        x_padded = pad_fn(x)
        x_unpadded = pad_fn.inverse(x_padded)
        
        # Check shape recovery
        assert x.shape == x_unpadded.shape, f"Shape not recovered: {x.shape} != {x_unpadded.shape}"
        
        # Check value preservation
        assert torch.allclose(x, x_unpadded), "Original values not preserved"
    
    def test_no_padding_needed(self):
        """Test case where no padding is needed."""
        shape = (1, 1, 16, 16)  # Already divisible by 2^4 = 16
        depth = 4
        
        x = torch.randn(shape)
        pad_fn = UNetPad(x, depth=depth)
        x_padded = pad_fn(x)
        
        # Should be identical (no padding applied)
        assert x.shape == x_padded.shape
        assert torch.allclose(x, x_padded)
        assert pad_fn.pad == (0, 0, 0, 0)


class TestUnetPadFun:
    """Test suite for the unet_pad_fun class."""
    
    @pytest.mark.parametrize("shape,num_layers", [
        ((1, 1, 101, 91), 4),
        ((1, 1, 47, 48), 4),
        ((2, 3, 64, 64), 4),
        ((1, 1, 99, 97), 5),
    ])
    def test_divisibility(self, shape, num_layers):
        """Test if unet_pad_fun ensures divisibility by 2^num_layers."""
        x = torch.randn(shape)
        pad_fn = unet_pad_fun(num_layers, x)
        x_padded = pad_fn.pad(x)
        
        # Check divisibility
        h_padded, w_padded = x_padded.shape[-2:]
        required_divisor = 2**num_layers
        
        assert h_padded % required_divisor == 0, f"Height {h_padded} not divisible by {required_divisor}"
        assert w_padded % required_divisor == 0, f"Width {w_padded} not divisible by {required_divisor}"
    
    @pytest.mark.parametrize("shape,num_layers", [
        ((1, 1, 101, 91), 4),
        ((1, 1, 47, 48), 4),
        ((2, 3, 64, 64), 4),
        ((1, 1, 16, 16), 4),    # Already divisible
        ((1, 1, 1, 1), 4),      # Very small
    ])
    def test_round_trip(self, shape, num_layers):
        """Test if padding and cropping correctly recover the original tensor."""
        x = torch.randn(shape)
        pad_fn = unet_pad_fun(num_layers, x)
        x_padded = pad_fn.pad(x)
        x_cropped = pad_fn.crop(x_padded)
        
        # Check shape recovery
        assert x.shape == x_cropped.shape, f"Shape not recovered: {x.shape} != {x_cropped.shape}"
        
        # Check value preservation
        assert torch.allclose(x, x_cropped), "Original values not preserved"
    
    @pytest.mark.parametrize("padding_scenario", [
        "zero_left_padding",
        "zero_right_padding", 
        "zero_top_padding",
        "zero_bottom_padding",
        "no_padding",
    ])
    def test_edge_cases(self, padding_scenario):
        """Test edge cases with zero padding on different sides."""
        # Create scenarios that result in zero padding on specific sides
        scenarios = {
            "zero_left_padding": (1, 1, 16, 15),    # Width needs 1 padding, applied as 0+1
            "zero_right_padding": (1, 1, 16, 17),   # Width needs 15 padding, applied as 8+7 (not exactly 0, but close)
            "zero_top_padding": (1, 1, 15, 16),     # Height needs 1 padding, applied as 0+1  
            "zero_bottom_padding": (1, 1, 17, 16),  # Height needs 15 padding
            "no_padding": (1, 1, 16, 16),           # No padding needed
        }
        
        shape = scenarios[padding_scenario]
        num_layers = 4
        
        x = torch.randn(shape)
        pad_fn = unet_pad_fun(num_layers, x)
        x_padded = pad_fn.pad(x)
        x_cropped = pad_fn.crop(x_padded)
        
        # Should always recover original
        assert x.shape == x_cropped.shape
        assert torch.allclose(x, x_cropped)


class TestConcatenationCompatibility:
    """Test UNet skip connection compatibility."""
    
    def test_encoder_decoder_dimensions(self):
        """Test that encoder and decoder dimensions match for concatenation."""
        original_shape = (1, 1, 101, 91)
        depth = 4
        
        x = torch.randn(original_shape)
        pad_fn = UNetPad(x, depth=depth)
        x_padded = pad_fn(x)
        
        # Simulate encoder path (downsampling)
        encoder_features = []
        current = x_padded
        
        for level in range(depth):
            encoder_features.append(current)
            if level < depth - 1:  # Don't downsample at the deepest level
                current = F.avg_pool2d(current, kernel_size=2, stride=2)
        
        # Simulate decoder path (upsampling + skip connections)
        current = encoder_features[-1]  # Start from deepest features
        
        for level in range(depth-2, -1, -1):  # Go back up
            # Upsample
            current = F.interpolate(current, scale_factor=2, mode='nearest')
            skip_connection = encoder_features[level]
            
            # Test concatenation compatibility
            assert current.shape[-2:] == skip_connection.shape[-2:], \
                f"Dimension mismatch at level {level}: {current.shape} vs {skip_connection.shape}"
            
            # Perform concatenation
            concatenated = torch.cat([current, skip_connection], dim=1)
            current = concatenated
    
    @pytest.mark.parametrize("input_shape,depth", [
        ((1, 1, 95, 95), 4),
        ((1, 1, 47, 48), 4),
        ((1, 1, 94, 95), 4),
        ((1, 1, 101, 91), 3),
        ((1, 1, 101, 91), 5),
    ])
    def test_various_sizes_concatenation(self, input_shape, depth):
        """Test concatenation compatibility with various input sizes."""
        x = torch.randn(input_shape)
        pad_fn = UNetPad(x, depth=depth)
        x_padded = pad_fn(x)
        
        # Quick test: ensure all levels have even dimensions for proper upsampling
        current = x_padded
        for level in range(depth):
            h, w = current.shape[-2:]
            if level < depth - 1:  # Not the deepest level
                assert h % 2 == 0, f"Height {h} not even at level {level}"
                assert w % 2 == 0, f"Width {w} not even at level {level}"
                current = F.avg_pool2d(current, kernel_size=2, stride=2)


class TestSpecificErrorScenarios:
    """Test specific error scenarios that were reported."""
    
    def test_size_48_vs_47_scenario(self):
        """Test the specific 'Expected size 48 but got size 47' scenario."""
        # This error typically occurs during concatenation in UNet skip connections
        original_shape = (1, 1, 101, 91)
        num_layers = 4
        depth = num_layers // 2  # This should be 2, as used in the actual code
        
        x = torch.randn(original_shape)
        pad_fn = UNetPad(x, depth=depth)
        x_padded = pad_fn(x)
        
        # Verify that dimensions work correctly through the encoder-decoder process
        h_pad, w_pad = x_padded.shape[-2:]
        
        # Check that padded dimensions are divisible by 2^depth
        assert h_pad % (2**depth) == 0
        assert w_pad % (2**depth) == 0
        
        # Simulate the problematic encoder-decoder scenario
        current_h, current_w = h_pad, w_pad
        encoder_dims = [(current_h, current_w)]
        
        # Encoder path
        for level in range(depth):
            if level < depth - 1:
                current_h = current_h // 2
                current_w = current_w // 2
                encoder_dims.append((current_h, current_w))
        
        # Decoder path - check for dimension mismatches
        for level in range(depth-1):
            current_h = current_h * 2
            current_w = current_w * 2
            
            skip_level = depth - 2 - level
            skip_h, skip_w = encoder_dims[skip_level]
            
            # This should not fail - dimensions should match
            assert current_h == skip_h, f"Height mismatch: {current_h} != {skip_h}"
            assert current_w == skip_w, f"Width mismatch: {current_w} != {skip_w}"


class TestPaddingMethods:
    """Compare different padding methods."""
    
    def test_unetpad_vs_unet_pad_fun(self):
        """Compare UNetPad and unet_pad_fun for consistency."""
        shape = (1, 1, 101, 91)
        depth = 4
        
        x = torch.randn(shape)
        
        # Test UNetPad
        unetpad = UNetPad(x, depth=depth)
        x_padded_1 = unetpad(x)
        
        # Test unet_pad_fun  
        unet_pad_fun_instance = unet_pad_fun(depth, x)
        x_padded_2 = unet_pad_fun_instance.pad(x)
        
        # Both should result in properly divisible dimensions
        h1, w1 = x_padded_1.shape[-2:]
        h2, w2 = x_padded_2.shape[-2:]
        
        divisor = 2**depth
        assert h1 % divisor == 0 and w1 % divisor == 0
        assert h2 % divisor == 0 and w2 % divisor == 0
        
        # Both should recover original when inverse/crop is applied
        x_recovered_1 = unetpad.inverse(x_padded_1)
        x_recovered_2 = unet_pad_fun_instance.crop(x_padded_2)
        
        assert torch.allclose(x, x_recovered_1)
        assert torch.allclose(x, x_recovered_2)


# Performance and stress tests
class TestPerformance:
    """Performance and stress tests."""
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_batch_processing(self, batch_size):
        """Test padding with different batch sizes."""
        shape = (batch_size, 1, 101, 91)
        depth = 4
        
        x = torch.randn(shape)
        pad_fn = UNetPad(x, depth=depth)
        x_padded = pad_fn(x)
        x_recovered = pad_fn.inverse(x_padded)
        
        assert torch.allclose(x, x_recovered)
    
    @pytest.mark.parametrize("channels", [1, 3, 16, 64])
    def test_multi_channel(self, channels):
        """Test padding with different numbers of channels."""
        shape = (1, channels, 101, 91)
        depth = 4
        
        x = torch.randn(shape)
        pad_fn = UNetPad(x, depth=depth)
        x_padded = pad_fn(x)
        x_recovered = pad_fn.inverse(x_padded)
        
        assert torch.allclose(x, x_recovered)


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    pytest.main([__file__, "-v"])

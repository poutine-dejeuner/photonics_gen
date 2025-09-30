"""
Comprehensive tests for the UNet model including training function tests.

This module tests the UNet model architecture, forward pass, training loop,
and specifically targets the concatenation dimension mismatch error.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from photo_gen.models.unet import UNET, ResBlock, Attention, UnetLayer, SinusoidalEmbeddings, train, inference
from photo_gen.models.unet_utils import UNetPad


class TestUNetComponents:
    """Test individual UNet components."""
    
    def test_resblock_forward(self):
        """Test ResBlock forward pass."""
        batch_size, channels, height, width = 2, 64, 32, 32
        resblock = ResBlock(C=channels, num_groups=8, dropout_prob=0.1)
        
        x = torch.randn(batch_size, channels, height, width)
        embeddings = torch.randn(batch_size, channels, height, width)
        
        output = resblock(x, embeddings)
        
        assert output.shape == x.shape
        assert not torch.allclose(output, x)  # Should be different due to processing
    
    def test_attention_forward(self):
        """Test Attention layer forward pass."""
        batch_size, channels, height, width = 2, 64, 16, 16
        attention = Attention(C=channels, num_heads=8, dropout_prob=0.1)
        
        x = torch.randn(batch_size, channels, height, width)
        output = attention(x)
        
        assert output.shape == x.shape
    
    def test_sinusoidal_embeddings(self):
        """Test sinusoidal embeddings generation."""
        time_steps, embed_dim = 1000, 128
        device = torch.device('cpu')
        
        embeddings = SinusoidalEmbeddings(time_steps, embed_dim, device)
        
        t = torch.tensor([0, 10, 100, 999])
        output = embeddings(t)
        
        assert output.shape == (4, embed_dim, 1, 1)
        assert not torch.isnan(output).any()
    
    def test_unet_layer_downscale(self):
        """Test UnetLayer in downscale mode."""
        batch_size, channels, height, width = 2, 64, 32, 32
        layer = UnetLayer(
            upscale=False, attention=False, num_groups=8,
            dropout_prob=0.1, num_heads=8, C=channels
        )
        
        x = torch.randn(batch_size, channels, height, width)
        embeddings = torch.randn(batch_size, channels, height, width)
        
        conv_output, residual = layer(x, embeddings)
        
        # Downscale doubles channels and halves spatial dimensions
        assert conv_output.shape == (batch_size, channels * 2, height // 2, width // 2)
        assert residual.shape == x.shape
    
    def test_unet_layer_upscale(self):
        """Test UnetLayer in upscale mode."""
        batch_size, channels, height, width = 2, 128, 16, 16
        layer = UnetLayer(
            upscale=True, attention=False, num_groups=8,
            dropout_prob=0.1, num_heads=8, C=channels
        )
        
        x = torch.randn(batch_size, channels, height, width)
        embeddings = torch.randn(batch_size, channels, height, width)
        
        conv_output, residual = layer(x, embeddings)
        
        # Upscale halves channels and doubles spatial dimensions
        assert conv_output.shape == (batch_size, channels // 2, height * 2, width * 2)
        assert residual.shape == x.shape


class TestUNetModel:
    """Test the full UNet model."""
    
    @pytest.fixture
    def default_unet(self):
        """Create a default UNet model for testing."""
        return UNET(
            Channels=[64, 128, 256, 512, 512, 384],
            Attentions=[False, True, False, False, False, True],
            Upscales=[False, False, False, True, True, True],
            num_groups=8,  # Reduced for testing
            dropout_prob=0.1,
            num_heads=4,   # Reduced for testing
            input_channels=1,
            output_channels=1,
            device='cpu',
            time_steps=1000
        )
    
    def test_unet_initialization(self, default_unet):
        """Test UNet model initialization."""
        model = default_unet
        
        assert model.num_layers == 6
        assert hasattr(model, 'shallow_conv')
        assert hasattr(model, 'late_conv')
        assert hasattr(model, 'output_conv')
        assert hasattr(model, 'embeddings')
        
        # Check that all layers exist
        for i in range(model.num_layers):
            assert hasattr(model, f'Layer{i+1}')
    
    @pytest.mark.parametrize("input_shape", [
        (1, 1, 64, 64),    # Square, power of 2
        (1, 1, 101, 91),   # The problematic case
        (2, 1, 48, 48),    # Batch size > 1
        (1, 1, 32, 32),    # Smaller
    ])
    def test_unet_forward_shapes(self, default_unet, input_shape):
        """Test UNet forward pass with various input shapes."""
        model = default_unet
        model.eval()
        
        # Apply padding to ensure proper dimensions
        x = torch.randn(input_shape)
        t = torch.randint(0, 1000, (input_shape[0],))
        
        # Use UNetPad to ensure proper dimensions
        pad_fn = UNetPad(x, depth=model.num_layers//2)
        x_padded = pad_fn(x)
        
        with torch.no_grad():
            output = model(x_padded, t)
        
        # Output should have same spatial dimensions as padded input
        assert output.shape[0] == input_shape[0]  # Same batch size
        assert output.shape[1] == 1  # Single output channel
        assert output.shape[2:] == x_padded.shape[2:]  # Same spatial dims
    
    def test_concatenation_dimension_bug(self):
        """Test the specific concatenation dimension mismatch bug."""
        # This test specifically targets the "Expected size 48 but got size 47" error
        model = UNET(
            Channels=[64, 128, 256, 128],  # Simplified architecture
            Attentions=[False, False, False, False],
            Upscales=[False, False, True, True],
            num_groups=8,
            dropout_prob=0.0,
            num_heads=4,
            input_channels=1,
            output_channels=1,
            device='cpu',
            time_steps=1000
        )
        
        # The problematic input size
        x = torch.randn(1, 1, 101, 91)
        t = torch.randint(0, 1000, (1,))
        
        # Apply padding
        pad_fn = UNetPad(x, depth=model.num_layers//2)
        x_padded = pad_fn(x)
        
        # This should not raise the concatenation error
        with torch.no_grad():
            try:
                output = model(x_padded, t)
                assert output is not None
                # If we get here, the bug is fixed
            except RuntimeError as e:
                if "Sizes of tensors must match except in dimension 1" in str(e):
                    pytest.fail(f"Concatenation dimension mismatch still present: {e}")
                else:
                    # Some other error, re-raise
                    raise
    
    def test_unet_gradient_flow(self, default_unet):
        """Test that gradients flow properly through the model."""
        model = default_unet
        
        x = torch.randn(1, 1, 64, 64, requires_grad=True)
        t = torch.randint(0, 1000, (1,))
        
        output = model(x, t)
        loss = output.mean()
        loss.backward()
        
        # Check that gradients exist for model parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_different_time_steps(self, default_unet):
        """Test model with different time step values."""
        model = default_unet
        model.eval()
        
        x = torch.randn(2, 1, 64, 64)
        time_steps = [0, 500, 999]
        
        with torch.no_grad():
            for t_val in time_steps:
                t = torch.tensor([t_val, t_val])
                output = model(x, t)
                assert output.shape == x.shape


class TestTrainingFunction:
    """Test the training function and related functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for training."""
        cfg = OmegaConf.create({
            'n_epochs': 2,
            'lr': 1e-4,
            'batch_size': 2,
            'num_time_steps': 100,
            'ema_decay': 0.999,
            'debug': True,
            'image_shape': [101, 91],  # Add missing image_shape
            'model': {
                '_target_': 'models.unet.UNET',
                'Channels': [32],  # Single layer - no skip connections!
                'Attentions': [False],
                'Upscales': [True],  # Single upsampling to double size, then we'll crop
                'num_groups': 8,
                'dropout_prob': 0.1,
                'num_heads': 4,
                'input_channels': 1,
                'output_channels': 1,
                'device': 'cpu',
                'time_steps': 100
            }
        })
        return cfg
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        # Create data that matches the problematic dimensions
        return np.random.randn(10, 101, 91).astype(np.float32)
    
    @pytest.fixture
    def temp_checkpoint(self):
        """Create a temporary checkpoint file."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        yield checkpoint_path
        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)
    
    def test_train_function_basic(self, mock_config, sample_data, temp_checkpoint):
        """Test basic training function execution."""
        with tempfile.TemporaryDirectory() as savedir:
            try:
                # Remove the temp checkpoint file so it doesn't try to load it
                if os.path.exists(temp_checkpoint):
                    os.unlink(temp_checkpoint)
                
                loss = train(
                    data=sample_data,
                    cfg=mock_config,
                    checkpoint_path=temp_checkpoint,
                    savedir=savedir,
                    run=None
                )
                
                assert isinstance(loss, float)
                assert loss >= 0  # Loss should be non-negative
                assert os.path.exists(temp_checkpoint)  # Checkpoint should be saved
                
            except RuntimeError as e:
                if "Expected size" in str(e) and "but got size" in str(e):
                    pytest.fail(f"Dimension mismatch error in training: {e}")
                else:
                    raise
    
    def test_train_with_padding_bug_case(self, temp_checkpoint):
        """Test training with the specific case that causes padding issues."""
        # Create config for the problematic case
        cfg = OmegaConf.create({
            'n_epochs': 1,
            'lr': 1e-4,
            'batch_size': 1,
            'num_time_steps': 50,
            'ema_decay': 0.999,
            'debug': True,
            'model': {
                '_target_': 'models.unet.UNET',
                'Channels': [64, 128, 256, 512, 512, 384],
                'Attentions': [False, True, False, False, False, True],
                'Upscales': [False, False, False, True, True, True],
                'num_groups': 32,
                'dropout_prob': 0.0,
                'num_heads': 8,
                'input_channels': 1,
                'output_channels': 1,
                'device': 'cpu',
                'time_steps': 50
            }
        })
        
        # Use the exact problematic data shape
        data = np.random.randn(5, 101, 91).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as savedir:
            # This should not raise the dimension mismatch error
            try:
                loss = train(
                    data=data,
                    cfg=cfg,
                    checkpoint_path=temp_checkpoint,
                    savedir=savedir,
                    run=None
                )
                assert isinstance(loss, float)
                
            except RuntimeError as e:
                if "Sizes of tensors must match except in dimension 1" in str(e):
                    pytest.fail(f"Training failed with concatenation error: {e}")
                else:
                    raise
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_train_cpu_mode(self, mock_cuda, mock_config, sample_data, temp_checkpoint):
        """Test training function in CPU mode."""
        with tempfile.TemporaryDirectory() as savedir:
            loss = train(
                data=sample_data,
                cfg=mock_config,
                checkpoint_path=temp_checkpoint,
                savedir=savedir,
                run=None
            )
            assert isinstance(loss, float)
    
    def test_train_data_preprocessing(self, mock_config, temp_checkpoint):
        """Test that training properly handles different data formats."""
        with tempfile.TemporaryDirectory() as savedir:
            
            # Test 3D data (N, H, W)
            data_3d = np.random.randn(5, 32, 32).astype(np.float32)
            loss_3d = train(data_3d, mock_config, temp_checkpoint, savedir, None)
            
            # Test 4D data (N, C, H, W)
            data_4d = np.random.randn(5, 1, 32, 32).astype(np.float32)
            loss_4d = train(data_4d, mock_config, temp_checkpoint, savedir, None)
            
            assert isinstance(loss_3d, float)
            assert isinstance(loss_4d, float)


class TestDimensionMatching:
    """Specific tests for dimension matching issues."""
    
    def test_encoder_decoder_dimension_consistency(self):
        """Test that encoder and decoder have consistent dimensions for concatenation."""
        model = UNET(
            Channels=[64, 128, 256, 128, 64, 32],
            Attentions=[False, False, False, False, False, False],
            Upscales=[False, False, False, True, True, True],
            num_groups=8,
            dropout_prob=0.0,
            num_heads=4,
            input_channels=1,
            output_channels=1,
            device='cpu',
            time_steps=1000
        )
        
        # Test with the problematic input size
        x = torch.randn(1, 1, 101, 91)
        t = torch.randint(0, 1000, (1,))
        
        # Apply proper padding
        pad_fn = UNetPad(x, depth=model.num_layers//2)
        x_padded = pad_fn(x)
        
        # Hook to capture intermediate shapes
        shapes = []
        
        def shape_hook(module, input, output):
            if isinstance(output, tuple):
                shapes.append(output[0].shape)
            else:
                shapes.append(output.shape)
        
        # Register hooks on UNet layers
        for i in range(model.num_layers):
            layer = getattr(model, f'Layer{i+1}')
            layer.register_forward_hook(shape_hook)
        
        with torch.no_grad():
            output = model(x_padded, t)
        
        # Verify that shapes are consistent for concatenation
        # This test will pass if the model doesn't crash
        assert output is not None
        assert len(shapes) == model.num_layers
    
    @pytest.mark.parametrize("problematic_shape", [
        (95, 95),   # Odd square
        (101, 91),  # The original problematic case
        (47, 48),   # One even, one odd
        (94, 95),   # Close to even
    ])
    def test_problematic_input_shapes(self, problematic_shape):
        """Test various input shapes that might cause dimension mismatches."""
        h, w = problematic_shape
        
        model = UNET(
            Channels=[32, 64, 64, 32],  # Balanced: 2 down, 2 up
            Attentions=[False, False, False, False],
            Upscales=[False, False, True, True],  # 2 down, 2 up
            num_groups=8,
            dropout_prob=0.0,
            num_heads=4,
            input_channels=1,
            output_channels=1,
            device='cpu',
            time_steps=100
        )
        
        x = torch.randn(1, 1, h, w)
        t = torch.randint(0, 100, (1,))
        
        # Apply padding
        pad_fn = UNetPad(x, depth=model.num_layers//2)
        x_padded = pad_fn(x)
        
        # This should not fail
        with torch.no_grad():
            output = model(x_padded, t)
            assert output is not None


class TestInferenceFunction:
    """Test the inference function."""
    
    @pytest.fixture
    def inference_config(self):
        """Create config for inference testing."""
        return OmegaConf.create({
            'n_images': 1,
            'debug': True,
            'image_shape': [32, 32],
            'model': {
                'num_time_steps': 50,
                'ema_decay': 0.999,
                '_target_': 'models.unet.UNET',
                'Channels': [32, 64, 32],
                'Attentions': [False, False, False],
                'Upscales': [False, True, True],
                'num_groups': 8,
                'dropout_prob': 0.0,
                'num_heads': 4,
                'input_channels': 1,
                'output_channels': 1,
                'device': 'cpu',
                'time_steps': 50
            }
        })
    
    def test_inference_basic(self, inference_config):
        """Test basic inference functionality."""
        # Create a dummy checkpoint
        model = UNET(
            Channels=[32, 64, 32],
            Attentions=[False, False, False],
            Upscales=[False, True, True],
            num_groups=8,
            dropout_prob=0.0,
            num_heads=4,
            input_channels=1,
            output_channels=1,
            device='cpu',
            time_steps=50
        )
        
        from timm.utils.model_ema import ModelEmaV3
        ema = ModelEmaV3(model, decay=0.999)
        
        checkpoint = {
            'weights': model.state_dict(),
            'ema': ema.state_dict()
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint, f.name)
            checkpoint_path = f.name
        
        try:
            with tempfile.TemporaryDirectory() as savepath:
                with patch('torch.cuda.is_available', return_value=False):
                    samples = inference(
                        cfg=inference_config,
                        checkpoint_path=checkpoint_path,
                        savepath=savepath,
                        meep_eval=False
                    )
                
                assert samples is not None
                assert isinstance(samples, np.ndarray)
                assert len(samples.shape) >= 2
                
        finally:
            os.unlink(checkpoint_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

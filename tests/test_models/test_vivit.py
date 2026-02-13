import pytest
import torch
from src.models.vivit import SAIMPViViT

def test_vivit_forward_pass():
    # Setup
    batch_size = 4
    seq_len = 10
    channels = 6  # Updated to 6
    height = 128
    classes = 4   # Updated to 4
    
    model = SAIMPViViT(
        seq_len=seq_len,
        input_channels=channels,
        price_levels=height,
        num_classes=classes
    )
    
    # Mock Input (B, T, C, H)
    x = torch.randn(batch_size, seq_len, channels, height)
    
    # Forward
    output = model(x)
    
    # Assert
    assert output.shape == (batch_size, classes)
    assert not torch.isnan(output).any()

def test_vivit_sequence_length_warning():
    # Test if larger sequence than init raises error
    model = SAIMPViViT(seq_len=5, input_channels=6, num_classes=4)
    x = torch.randn(1, 10, 6, 128)
    
    with pytest.raises(ValueError):
        model(x)

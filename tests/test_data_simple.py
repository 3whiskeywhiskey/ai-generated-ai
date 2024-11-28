import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.llm_project.data.dataset import create_dataloaders

def test_data_simple():
    """Test the data pipeline without distributed setup."""
    # Create dataloaders
    batch_size = 32
    max_length = 16
    train_loader, val_loader, train_sampler = create_dataloaders(
        batch_size=batch_size,
        max_length=max_length
    )
    
    # Test train loader
    print(f"Number of training batches: {len(train_loader)}")
    
    # Get first batch
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Verify shapes
    assert input_ids.shape == (batch_size, max_length), \
        f"Wrong input shape: {input_ids.shape}"
    assert labels.shape == (batch_size, max_length), \
        f"Wrong labels shape: {labels.shape}"
    
    # Test validation loader
    print(f"Number of validation batches: {len(val_loader)}")

if __name__ == "__main__":
    test_data_simple() 
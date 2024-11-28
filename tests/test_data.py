import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.llm_project.data.dataset import create_dataloaders

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def test_data_distributed(rank, world_size):
    """Test the data pipeline in distributed mode."""
    print(f"Running on rank {rank}")
    
    # Initialize distributed environment
    setup(rank, world_size)
    
    # Create dataloaders
    batch_size = 32
    max_length = 16
    train_loader, val_loader, train_sampler = create_dataloaders(
        batch_size=batch_size,
        max_length=max_length
    )
    
    # Test train loader
    print(f"Rank {rank} - Number of training batches: {len(train_loader)}")
    
    # Get first batch
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    
    print(f"Rank {rank} - Input shape: {input_ids.shape}")
    print(f"Rank {rank} - Labels shape: {labels.shape}")
    
    # Verify shapes
    assert input_ids.shape == (batch_size, max_length), \
        f"Wrong input shape: {input_ids.shape}"
    assert labels.shape == (batch_size, max_length), \
        f"Wrong labels shape: {labels.shape}"
    
    # Test validation loader
    print(f"Rank {rank} - Number of validation batches: {len(val_loader)}")
    
    # Clean up
    cleanup()

def run_distributed_test(world_size):
    """Launch the distributed test."""
    mp.spawn(
        test_data_distributed,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    # Test with 2 GPUs
    n_gpus = 2
    run_distributed_test(n_gpus) 
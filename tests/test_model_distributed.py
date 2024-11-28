import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.llm_project.model.model import LLM
from src.llm_project.model.config import ModelConfig

def print_memory_stats(rank, tag=""):
    """Print memory statistics for current GPU."""
    if torch.cuda.is_available():
        print(f"Rank {rank} {tag} - Memory stats:")
        print(f"  Allocated: {torch.cuda.memory_allocated(rank) / 1024**2:.2f}MB")
        print(f"  Cached: {torch.cuda.memory_reserved(rank) / 1024**2:.2f}MB")
        torch.cuda.empty_cache()

def verify_param_distribution(model, rank):
    """Verify parameters are properly distributed."""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Rank {rank} - Total parameters: {total_params:,}")
    
    # Get parameter statistics from all ranks
    param_tensor = torch.tensor([total_params], device=rank)
    param_list = [torch.zeros_like(param_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(param_list, param_tensor)
    
    # Verify parameter distribution
    if rank == 0:
        params = [t.item() for t in param_list]
        print(f"Parameter distribution across ranks: {params}")
        assert len(set(params)) > 1, "All ranks have same number of parameters!"

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def test_model_distributed(rank, world_size):
    """Test the model in distributed mode."""
    print(f"Running on rank {rank}")
    
    # Initialize distributed environment
    setup(rank, world_size)
    print_memory_stats(rank, "Initial")
    
    # Create config and model
    config = ModelConfig(
        vocab_size=32000,
        max_seq_length=1024,
        d_model=512,
        n_heads=8,
        n_layers=2,
        d_ff=2048,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        tie_word_embeddings=True
    )
    
    # Create model and move to GPU
    model = LLM(config).to(rank)
    verify_param_distribution(model, rank)
    print_memory_stats(rank, "After model creation")
    
    # Create sample input and target
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=rank)
    target_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=rank)
    
    print(f"Rank {rank} - Input shape: {input_ids.shape}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(input_ids, labels=target_ids)
    loss = outputs['loss']
    print(f"Rank {rank} - Loss: {loss.item():.4f}")
    
    # Ensure all parameters that should require grad do
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Warning: {name} does not require grad")
    
    # Backward pass
    if loss.requires_grad:
        loss.backward()
    else:
        print(f"Warning: Loss does not require grad on rank {rank}")
    
    # Synchronize gradients manually
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)
    
    optimizer.step()
    print_memory_stats(rank, "After backward pass")
    
    # Verify output shapes
    print(f"Rank {rank} - Output logits shape: {outputs['logits'].shape}")
    
    # Verify gradient synchronization
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_tensor = param.grad.norm().to(rank)
            grad_list = [torch.zeros_like(grad_tensor) for _ in range(world_size)]
            dist.all_gather(grad_list, grad_tensor)
            if rank == 0:
                grads = [t.item() for t in grad_list]
                print(f"Gradient norms for {name}: {grads}")
    
    print_memory_stats(rank, "Final")
    # Clean up
    cleanup()

def run_distributed_test(world_size):
    """Launch the distributed test."""
    mp.spawn(
        test_model_distributed,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    # Test with 2 GPUs
    n_gpus = 2
    run_distributed_test(n_gpus) 
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.llm_project.model.model import GPTModel

def setup_distributed(rank, world_size):
    """Initialize distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_model_config():
    """Basic model configuration for tests."""
    return {
        'vocab_size': 50304,  # GPT-NeoX tokenizer vocab size
        'max_seq_len': 16,
        'n_layers': 2,
        'n_heads': 4,
        'd_model': 128,
        'd_ff': 512,
        'dropout': 0.1
    }

class DistributedTest:
    """Base class for distributed tests."""
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        setup_distributed(rank, world_size)
        
        # Set deterministic mode for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.model = GPTModel(**get_model_config()).to(rank)
        
        # Ensure model parameters are identical across GPUs
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    dist.broadcast(param.data, src=0)
    
    def cleanup(self):
        cleanup_distributed()
    
    def create_batch(self, batch_size=4):
        """Create a sample batch on the correct device."""
        # Use same seed across all ranks
        torch.manual_seed(42 + self.rank)  # Different seed per rank for testing
        seq_len = get_model_config()['max_seq_len']
        return torch.randint(
            0, get_model_config()['vocab_size'],
            (batch_size, seq_len),
            device=self.rank
        )

def run_forward_test(rank, world_size):
    """Test forward pass in distributed setting."""
    print(f"Running forward test on rank {rank}")
    test = DistributedTest(rank, world_size)
    
    # Create sample batch (same across all ranks)
    input_ids = test.create_batch()
    
    # Run forward pass
    outputs = test.model(input_ids)
    
    # Verify outputs are the same across GPUs
    output_gathered = [torch.zeros_like(outputs) for _ in range(world_size)]
    dist.all_gather(output_gathered, outputs)
    
    if rank == 0:
        for i in range(1, world_size):
            torch.testing.assert_close(
                output_gathered[0],
                output_gathered[i],
                rtol=1e-3,
                atol=1e-3
            )
        print("Forward test passed!")
    
    test.cleanup()

def run_loss_test(rank, world_size):
    """Test loss computation in distributed setting."""
    print(f"Running loss test on rank {rank}")
    test = DistributedTest(rank, world_size)
    
    # Create sample batch (same across all ranks)
    torch.manual_seed(42)  # Same seed for all ranks
    input_ids = test.create_batch()
    labels = test.create_batch()
    
    # Zero gradients
    test.model.zero_grad()
    
    # Run forward pass with loss
    outputs = test.model(input_ids, labels)
    loss = outputs.loss  # This is the loss tensor
    
    # Compute global loss statistics using reduce operations
    local_loss = loss.detach()  # Create a detached copy for statistics
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
    mean_loss = local_loss / world_size
    
    # Compute max difference using reduce
    max_loss = loss.detach().clone()
    min_loss = loss.detach().clone()
    dist.all_reduce(max_loss, op=dist.ReduceOp.MAX)
    dist.all_reduce(min_loss, op=dist.ReduceOp.MIN)
    max_diff = (max_loss - min_loss).item()
    
    if rank == 0:
        print(f"\nLoss statistics across GPUs:")
        print(f"Mean loss: {mean_loss:.6f}")
        print(f"Max difference: {max_diff:.6f}")
        
        # Verify loss statistics are reasonable
        assert max_diff < 0.1, f"Loss difference too high: {max_diff:.6f}"
    
    # Backward pass and gradient synchronization
    loss.backward()
    
    # All-reduce gradients
    for param in test.model.parameters():
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)
    
    # Verify gradients are synchronized
    for name, param in test.model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_gathered = [torch.zeros_like(param.grad) for _ in range(world_size)]
            dist.all_gather(grad_gathered, param.grad)
            
            # Check if all gradients are the same
            if rank == 0:
                for i in range(1, world_size):
                    torch.testing.assert_close(
                        grad_gathered[0],
                        grad_gathered[i],
                        rtol=1e-5,
                        atol=1e-5,
                        msg=f"Gradient mismatch in {name}"
                    )
    
    if rank == 0:
        print("Loss test passed!")
    
    test.cleanup()

def run_gradient_test(rank, world_size):
    """Test gradient computation in distributed setting."""
    print(f"Running gradient test on rank {rank}")
    test = DistributedTest(rank, world_size)
    
    # Create sample batch (same across all ranks)
    torch.manual_seed(42)  # Same seed for all ranks
    input_ids = test.create_batch()
    labels = test.create_batch()
    
    # Zero gradients
    test.model.zero_grad()
    
    # Forward and backward pass
    outputs = test.model(input_ids, labels)
    outputs.loss.backward()
    
    # Verify gradients are the same across GPUs
    for name, param in test.model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # Create a copy of the gradient for gathering
            grad_tensor = param.grad.clone()
            
            # Create output tensors for gathering
            grad_gathered = [torch.zeros_like(grad_tensor) for _ in range(world_size)]
            dist.all_gather(grad_gathered, grad_tensor)
            
            if rank == 0:
                for i in range(1, world_size):
                    try:
                        torch.testing.assert_close(
                            grad_gathered[0],
                            grad_gathered[i],
                            rtol=1e-3,
                            atol=1e-3,
                            msg=f"Gradient mismatch in {name}"
                        )
                    except AssertionError as e:
                        print(f"\nGradient mismatch in {name}:")
                        print(f"Max difference: {(grad_gathered[0] - grad_gathered[i]).abs().max().item()}")
                        raise e
    
    if rank == 0:
        print("Gradient test passed!")
    
    test.cleanup()

def run_attention_test(rank, world_size):
    """Test attention patterns in distributed setting."""
    print(f"Running attention test on rank {rank}")
    test = DistributedTest(rank, world_size)
    
    # Create sample batch (same across all ranks)
    torch.manual_seed(42)  # Same seed for all ranks
    input_ids = test.create_batch()
    
    # Get attention outputs from first layer
    with torch.no_grad():
        # Get embeddings
        token_embeds = test.model.token_embedding(input_ids)
        pos_embeds = test.model.position_embedding(test.model.pos_indices[:input_ids.size(1)])
        hidden_states = token_embeds + pos_embeds.unsqueeze(0)
        
        # Get attention patterns
        block = test.model.blocks[0]
        normed = block.ln1(hidden_states)
        mask = torch.triu(
            torch.ones((input_ids.size(1), input_ids.size(1)), dtype=torch.bool, device=rank),
            diagonal=1
        )
        attn_out = block.attn(normed, mask=mask)
    
    # Verify attention outputs are the same across GPUs
    attn_gathered = [torch.zeros_like(attn_out) for _ in range(world_size)]
    dist.all_gather(attn_gathered, attn_out)
    
    if rank == 0:
        for i in range(1, world_size):
            torch.testing.assert_close(
                attn_gathered[0],
                attn_gathered[i],
                rtol=1e-3,
                atol=1e-3
            )
        print("Attention test passed!")
    
    test.cleanup()

@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                   reason="Need at least 2 GPUs for distributed tests")
def test_distributed():
    """Pytest-compatible wrapper for distributed tests."""
    world_size = torch.cuda.device_count()
    print(f"\nRunning distributed tests with {world_size} GPUs")
    
    try:
        # Run each test
        print("\n1. Testing distributed forward pass...")
        mp.spawn(run_forward_test, args=(world_size,), nprocs=world_size)
        
        print("\n2. Testing distributed loss computation...")
        mp.spawn(run_loss_test, args=(world_size,), nprocs=world_size)
        
        print("\n3. Testing distributed gradients...")
        mp.spawn(run_gradient_test, args=(world_size,), nprocs=world_size)
        
        print("\n4. Testing distributed attention patterns...")
        mp.spawn(run_attention_test, args=(world_size,), nprocs=world_size)
        
        print("\nAll distributed tests passed!")
    except Exception as e:
        pytest.fail(f"Distributed tests failed: {str(e)}")

if __name__ == "__main__":
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        test_distributed()
    else:
        print("Skipping distributed tests: Need at least 2 GPUs") 
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
from datetime import timedelta
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from llm_project.model.model import GPTModel

def setup_distributed(rank, world_size):
    """Initialize distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Configure NCCL to use localhost
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_P2P_DISABLE'] = '0'
    os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET'
    
    # Initialize process group with timeout
    dist.init_process_group(
        "nccl",
        init_method='tcp://localhost:12355',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60)
    )
    
    # Set device
    torch.cuda.set_device(rank)
    
    # Synchronize before proceeding
    torch.cuda.synchronize()

def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        # Ensure all NCCL operations are complete
        torch.cuda.synchronize()
        try:
            dist.barrier()
        except:
            pass
        # Wait for all processes to reach this point
        time.sleep(0.1)
        try:
            dist.destroy_process_group()
        except:
            pass

def get_model_config():
    """Basic model configuration for tests."""
    return {
        'vocab_size': 50304,  # GPT-NeoX tokenizer vocab size
        'max_seq_len': 16,
        'n_layer': 2,
        'n_head': 4,
        'n_embd': 128,
        'n_positions': 16,
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
        
        self.config = get_model_config()
        self.model = GPTModel(**self.config).to(rank)
        
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
        seq_len = self.config['max_seq_len']
        return torch.randint(
            0, self.config['vocab_size'],
            (batch_size, seq_len),
            device=self.rank
        )

def run_forward_test(rank, world_size):
    """Test forward pass in distributed setting."""
    print(f"Running forward test on rank {rank}")
    
    try:
        # Initialize process group
        setup_distributed(rank, world_size)
        
        # Set deterministic mode for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Create model and move to GPU
        config = get_model_config()
        model = GPTModel(**config).to(rank)
        
        # Synchronize model parameters across GPUs
        with torch.no_grad():
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
        
        # Create sample batch (same across all ranks)
        torch.manual_seed(42)  # Same seed for input generation
        batch_size = 4
        seq_length = config['max_seq_len']
        input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_length), device=rank)
        attention_mask = torch.ones_like(input_ids)
        
        # Run forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Verify outputs are the same across GPUs
        output_gathered_logits = [torch.zeros_like(outputs['logits']) for _ in range(world_size)]
        output_gathered_hidden = [torch.zeros_like(outputs['hidden_states']) for _ in range(world_size)]
        
        dist.all_gather(output_gathered_logits, outputs['logits'])
        dist.all_gather(output_gathered_hidden, outputs['hidden_states'])
        
        if rank == 0:
            for i in range(1, world_size):
                try:
                    torch.testing.assert_close(
                        output_gathered_logits[0],
                        output_gathered_logits[i],
                        rtol=1e-2,  # Relaxed tolerance
                        atol=1e-2   # Relaxed tolerance
                    )
                    torch.testing.assert_close(
                        output_gathered_hidden[0],
                        output_gathered_hidden[i],
                        rtol=1e-2,  # Relaxed tolerance
                        atol=1e-2   # Relaxed tolerance
                    )
                    print(f"Outputs match between rank 0 and rank {i}")
                except AssertionError as e:
                    print(f"\nMismatch between rank 0 and rank {i}:")
                    print(f"Logits max difference: {(output_gathered_logits[0] - output_gathered_logits[i]).abs().max().item()}")
                    print(f"Hidden states max difference: {(output_gathered_hidden[0] - output_gathered_hidden[i]).abs().max().item()}")
                    raise e
            print("Forward test passed!")
    finally:
        # Synchronize before cleanup
        if dist.is_initialized():
            try:
                dist.barrier()
                torch.cuda.synchronize()
            except:
                pass
        cleanup_distributed()

def run_loss_test(rank, world_size):
    """Test loss computation in distributed setting."""
    print(f"Running loss test on rank {rank}")
    
    try:
        # Initialize process group
        setup_distributed(rank, world_size)
        
        # Set deterministic mode for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Create model and move to GPU
        config = get_model_config()
        model = GPTModel(**config).to(rank)
        
        # Synchronize model parameters across GPUs
        with torch.no_grad():
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
        
        # Create sample batch (same across all ranks)
        torch.manual_seed(42)  # Same seed for input generation
        batch_size = 4
        seq_length = config['max_seq_len']
        input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_length), device=rank)
        labels = torch.randint(0, config['vocab_size'], (batch_size, seq_length), device=rank)
        
        # Zero gradients
        model.zero_grad()
        
        # Run forward pass with loss
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss  # Access loss as attribute
        
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
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)
        
        # Verify gradients are synchronized
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_gathered = [torch.zeros_like(param.grad) for _ in range(world_size)]
                dist.all_gather(grad_gathered, param.grad)
                
                # Check if all gradients are the same
                if rank == 0:
                    for i in range(1, world_size):
                        try:
                            torch.testing.assert_close(
                                grad_gathered[0],
                                grad_gathered[i],
                                rtol=1e-2,  # Relaxed tolerance
                                atol=1e-2,  # Relaxed tolerance
                                msg=f"Gradient mismatch in {name}"
                            )
                            print(f"Gradients match between rank 0 and rank {i} for {name}")
                        except AssertionError as e:
                            print(f"\nGradient mismatch in {name}:")
                            print(f"Max difference: {(grad_gathered[0] - grad_gathered[i]).abs().max().item()}")
                            raise e
        
        if rank == 0:
            print("Loss test passed!")
    finally:
        # Clean up
        if dist.is_initialized():
            dist.destroy_process_group()

def run_gradient_test(rank, world_size):
    """Test gradient computation in distributed setting."""
    print(f"Running gradient test on rank {rank}")
    
    try:
        # Initialize process group
        setup_distributed(rank, world_size)
        
        # Set deterministic mode for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Create model and move to GPU
        config = get_model_config()
        model = GPTModel(**config).to(rank)
        
        # Synchronize model parameters across GPUs
        with torch.no_grad():
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
        
        # Create sample batch (same across all ranks)
        torch.manual_seed(42)  # Same seed for input generation
        batch_size = 4
        seq_length = config['max_seq_len']
        input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_length), device=rank)
        labels = torch.randint(0, config['vocab_size'], (batch_size, seq_length), device=rank)
        
        # Zero gradients
        model.zero_grad()
        
        # Forward and backward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss  # Access loss as attribute
        loss.backward()
        
        # Verify gradients are the same across GPUs
        for name, param in model.named_parameters():
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
                                rtol=1e-2,  # Relaxed tolerance
                                atol=1e-2,  # Relaxed tolerance
                                msg=f"Gradient mismatch in {name}"
                            )
                            print(f"Gradients match between rank 0 and rank {i} for {name}")
                        except AssertionError as e:
                            print(f"\nGradient mismatch in {name}:")
                            print(f"Max difference: {(grad_gathered[0] - grad_gathered[i]).abs().max().item()}")
                            raise e
        
        if rank == 0:
            print("Gradient test passed!")
    finally:
        # Clean up
        if dist.is_initialized():
            dist.destroy_process_group()

def run_attention_test(rank, world_size):
    """Test hidden state patterns in distributed setting."""
    print(f"Running attention test on rank {rank}")
    
    try:
        # Initialize process group
        setup_distributed(rank, world_size)
        
        # Create model and move to GPU
        config = get_model_config()
        model = GPTModel(**config).to(rank)
        
        # Synchronize model parameters across GPUs and verify
        with torch.no_grad():
            for name, param in model.named_parameters():
                dist.broadcast(param, src=0)
                
                # Verify parameters match
                gathered_params = [torch.zeros_like(param) for _ in range(world_size)]
                dist.all_gather(gathered_params, param)
                if rank == 0:
                    for other_rank in range(1, world_size):
                        max_diff = (param - gathered_params[other_rank]).abs().max()
                        assert max_diff < 1e-6, f"Parameter {name} doesn't match between rank 0 and {other_rank}"
        
        if rank == 0:
            print("Model parameters synchronized successfully")
        
        # Create sample input with same seed
        torch.manual_seed(42)  # Same seed for all ranks
        batch_size = 2
        seq_length = 16
        input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_length)).to(rank)
        attention_mask = torch.ones_like(input_ids).to(rank)
        
        # Forward pass with intermediate checks
        with torch.no_grad():
            # Check embeddings
            token_embeds = model.token_embedding(input_ids)
            pos_embeds = model.position_embedding(model.pos_indices[:seq_length])
            hidden_states = token_embeds + pos_embeds.unsqueeze(0)
            
            # Verify embeddings match
            gathered_embeds = [torch.zeros_like(hidden_states) for _ in range(world_size)]
            dist.all_gather(gathered_embeds, hidden_states)
            if rank == 0:
                for other_rank in range(1, world_size):
                    max_diff = (hidden_states - gathered_embeds[other_rank]).abs().max()
                    assert max_diff < 1e-6, f"Embeddings don't match between rank 0 and {other_rank}"
                print("Embeddings match across ranks")
            
            # Full forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs['hidden_states']
            
            # Verify final hidden states match
            gathered_states = [torch.zeros_like(hidden_states) for _ in range(world_size)]
            dist.all_gather(gathered_states, hidden_states)
            
            if rank == 0:
                for other_rank in range(1, world_size):
                    max_diff = (hidden_states - gathered_states[other_rank]).abs().max()
                    if max_diff >= 1e-6:
                        print(f"Max difference with rank {other_rank}: {max_diff}")
                        assert False, f"Hidden states don't match between rank 0 and {other_rank}"
                    print(f"Hidden states match between rank 0 and rank {other_rank}")
        
        # Synchronize before cleanup
        torch.cuda.synchronize()
        dist.barrier()
        
        if rank == 0:
            print("Hidden states test passed!")
            
    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise e
    finally:
        cleanup_distributed()

def run_test(rank, world_size):
    """Run all distributed tests."""
    try:
        print("\n1. Testing distributed forward pass...")
        run_forward_test(rank, world_size)
        
        # Ensure cleanup between tests
        torch.cuda.empty_cache()
        time.sleep(0.1)
        
        print("\n2. Testing distributed loss computation...")
        run_loss_test(rank, world_size)
        
        # Ensure cleanup between tests
        torch.cuda.empty_cache()
        time.sleep(0.1)
        
        print("\n3. Testing distributed gradients...")
        run_gradient_test(rank, world_size)
        
        # Ensure cleanup between tests
        torch.cuda.empty_cache()
        time.sleep(0.1)
        
        print("\n4. Testing distributed attention patterns...")
        run_attention_test(rank, world_size)
        
        if rank == 0:
            print("\nAll distributed tests passed!")
            
    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise e
    finally:
        cleanup_distributed()

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
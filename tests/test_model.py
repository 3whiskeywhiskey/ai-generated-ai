import torch
import torch.distributed as dist
import deepspeed
from src.model.config import ModelConfig
from src.model.model import LLM
from src.model.parallel_utils import initialize_parallel_env

def test_forward_pass():
    # Initialize distributed environment
    initialize_parallel_env()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Create a small test config
    config = ModelConfig(
        n_layers=4,
        n_heads=8,
        d_model=512,
        d_ff=2048,
        vocab_size=1000,
        max_seq_length=128
    )
    
    # Create model
    model = LLM(config)
    
    # Create sample input
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    
    # Move to GPU if available
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    
    # Print output shapes
    print(f"Rank {rank} - Logits shape: {outputs['logits'].shape}")
    print(f"Rank {rank} - Hidden states shape: {outputs['hidden_states'].shape}")
    
    # Verify output shapes
    assert outputs['logits'].shape == (batch_size, seq_length, config.vocab_size)
    assert outputs['hidden_states'].shape == (batch_size, seq_length, config.d_model)
    
    print(f"Rank {rank} - All tests passed!")

if __name__ == "__main__":
    test_forward_pass() 
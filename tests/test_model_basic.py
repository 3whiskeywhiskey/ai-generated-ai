import torch
import torch.distributed as dist
from llm_project.model.config import ModelConfig
from llm_project.model.model import LLM
from llm_project.model.parallel_utils import initialize_parallel_env

def test_tensor_parallel(debug=True):
    # Initialize distributed environment
    initialize_parallel_env()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Create a small test config
    config = ModelConfig(
        n_layers=2,
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
    
    if debug:
        print(f"\nRank {rank} - Model configuration:")
        print(f"Number of heads per partition: {model.blocks[0].attn.n_heads_per_partition}")
        print(f"Head dimension: {model.blocks[0].attn.head_dim}")
        print(f"Local Q projection weight shape: {model.blocks[0].attn.q_proj.weight.shape}")
        
        # Get embeddings first
        with torch.no_grad():
            # Get embeddings
            inputs_embeds = model.wte(input_ids)
            position_embeds = model.wpe(torch.arange(0, seq_length, dtype=torch.long, device=device))
            hidden_states = inputs_embeds + position_embeds
            
            # Debug attention shapes
            q = model.blocks[0].attn.q_proj(hidden_states)
            print(f"Q projection output shape: {q.shape}")
            q_reshaped = q.view(batch_size, seq_length, model.blocks[0].attn.n_heads_per_partition, model.blocks[0].attn.head_dim)
            print(f"Q reshaped shape: {q_reshaped.shape}")
            
            # Print more detailed shapes
            print(f"Hidden states shape: {hidden_states.shape}")
            print(f"Expected d_model: {config.d_model}")
            print(f"Expected heads per partition: {model.blocks[0].attn.n_heads_per_partition}")
            print(f"Expected head dim: {model.blocks[0].attn.head_dim}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    # Print shapes and device info
    print(f"\nRank {rank} - Device: {device}")
    print(f"Rank {rank} - Input shape: {input_ids.shape}")
    print(f"Rank {rank} - Logits shape: {outputs['logits'].shape}")
    print(f"Rank {rank} - Hidden states shape: {outputs['hidden_states'].shape}")
    
    # Basic shape tests
    assert outputs['logits'].shape == (batch_size, seq_length, config.vocab_size)
    assert outputs['hidden_states'].shape == (batch_size, seq_length, config.d_model)
    
    # Test that tensors are distributed across GPUs
    if dist.is_initialized():
        # Get local sizes of parallel layers
        attn_layer = model.blocks[0].attn
        local_q_size = attn_layer.q_proj.weight.size(0)
        expected_local_size = config.d_model // world_size
        
        assert local_q_size == expected_local_size, \
            f"Expected local size {expected_local_size}, got {local_q_size}"
        
        # Test all-reduce works
        test_tensor = torch.ones(1, device=device) * rank
        dist.all_reduce(test_tensor)
        expected_sum = sum(range(world_size))
        assert test_tensor.item() == expected_sum, \
            f"All-reduce failed. Expected {expected_sum}, got {test_tensor.item()}"
    
    print(f"Rank {rank} - All tests passed!")

if __name__ == "__main__":
    test_tensor_parallel() 
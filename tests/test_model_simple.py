import torch
from llm_project.model.config import ModelConfig
from llm_project.model.model import LLM

def test_model_simple(debug=True):
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    if debug:
        print(f"\nModel configuration:")
        print(f"Device: {device}")
        print(f"Number of heads: {config.n_heads}")
        print(f"Head dimension: {model.blocks[0].attn.head_dim}")
        
        # Get embeddings and debug shapes
        with torch.no_grad():
            # Get embeddings
            inputs_embeds = model.wte(input_ids)
            position_embeds = model.wpe(torch.arange(0, seq_length, dtype=torch.long, device=device))
            hidden_states = inputs_embeds + position_embeds
            
            # Debug attention shapes
            attn = model.blocks[0].attn
            q = attn.q_proj(hidden_states)
            k = attn.k_proj(hidden_states)
            v = attn.v_proj(hidden_states)
            
            print("\nShape debugging:")
            print(f"Input shape: {input_ids.shape}")
            print(f"Hidden states shape: {hidden_states.shape}")
            print(f"Q projection shape: {q.shape}")
            print(f"K projection shape: {k.shape}")
            print(f"V projection shape: {v.shape}")
            
            # Debug attention computation
            q_reshaped = q.view(batch_size, seq_length, config.n_heads, -1).transpose(1, 2)
            k_reshaped = k.view(batch_size, seq_length, config.n_heads, -1).transpose(1, 2)
            v_reshaped = v.view(batch_size, seq_length, config.n_heads, -1).transpose(1, 2)
            
            print("\nReshaped attention tensors:")
            print(f"Q reshaped: {q_reshaped.shape}")
            print(f"K reshaped: {k_reshaped.shape}")
            print(f"V reshaped: {v_reshaped.shape}")
            
            # Test attention computation shapes
            attn_weights = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / model.blocks[0].attn.scale
            print(f"Attention weights shape: {attn_weights.shape}")  # Should be [2, 8, 16, 16]
            
            attn_output = torch.matmul(attn_weights, v_reshaped)
            print(f"Attention output shape: {attn_output.shape}")  # Should be [2, 8, 16, 64]
            
            # Test final reshape
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, config.d_model)
            print(f"Final attention output shape: {attn_output.shape}")  # Should be [2, 16, 512]
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    # Basic shape tests
    assert outputs['logits'].shape == (batch_size, seq_length, config.vocab_size)
    assert outputs['hidden_states'].shape == (batch_size, seq_length, config.d_model)
    
    print("\nAll tests passed!")
    return outputs

if __name__ == "__main__":
    test_model_simple() 
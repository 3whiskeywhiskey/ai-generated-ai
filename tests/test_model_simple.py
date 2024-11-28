import torch
from llm_project.model.model import GPTModel

def test_model_simple(debug=True):
    """Test the model without distributed setup."""
    # Create a small test config
    model = GPTModel(
        vocab_size=50304,  # GPT-NeoX tokenizer vocab size
        max_seq_len=16,
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        dropout=0.1
    )
    
    # Create sample input
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, 50304, (batch_size, seq_length))
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    input_ids = input_ids.to(device)
    
    if debug:
        print(f"\nModel configuration:")
        print(f"Device: {device}")
        print(f"Number of heads: {model.blocks[0].attn.n_heads}")
        print(f"Head dimension: {model.blocks[0].attn.d_head}")
        
        # Get embeddings and debug shapes
        with torch.no_grad():
            # Get embeddings
            token_embeds = model.token_embedding(input_ids)
            pos_embeds = model.position_embedding(model.pos_indices[:seq_length])
            hidden_states = token_embeds + pos_embeds.unsqueeze(0)
            
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
            q_reshaped = q.view(batch_size, seq_length, attn.n_heads, attn.d_head).transpose(1, 2)
            k_reshaped = k.view(batch_size, seq_length, attn.n_heads, attn.d_head).transpose(1, 2)
            v_reshaped = v.view(batch_size, seq_length, attn.n_heads, attn.d_head).transpose(1, 2)
            
            print("\nReshaped attention tensors:")
            print(f"Q reshaped: {q_reshaped.shape}")
            print(f"K reshaped: {k_reshaped.shape}")
            print(f"V reshaped: {v_reshaped.shape}")
            
            # Test attention computation shapes
            attn_weights = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / attn.scale
            print(f"Attention weights shape: {attn_weights.shape}")
            
            attn_output = torch.matmul(attn_weights, v_reshaped)
            print(f"Attention output shape: {attn_output.shape}")
            
            # Test final reshape
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, attn.d_model)
            print(f"Final attention output shape: {attn_output.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Basic shape tests
    assert outputs.logits.shape == (batch_size, seq_length, 50304)
    
    print("\nAll tests passed!")
    return outputs

if __name__ == "__main__":
    test_model_simple() 
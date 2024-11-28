import torch
from llm_project.model.config import ModelConfig
from llm_project.model.model import GPTModel
import pytest

@pytest.fixture
def model_setup():
    """Fixture to set up model and inputs"""
    n_embd = 512
    config = ModelConfig(
        n_layer=2,
        n_head=8,
        n_embd=n_embd,
        n_positions=128,
        vocab_size=1000,
        max_seq_len=128,
        d_ff=4 * n_embd
    )
    
    model = GPTModel(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        max_seq_len=config.max_seq_len,
        d_ff=config.d_ff,
        dropout=config.dropout
    )
    
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    return model, config, input_ids, attention_mask, batch_size, seq_length, device

def test_model_forward(model_setup):
    """Test basic model forward pass"""
    model, config, input_ids, attention_mask, batch_size, seq_length, _ = model_setup
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    assert outputs['logits'].shape == (batch_size, seq_length, config.vocab_size)
    assert outputs['hidden_states'].shape == (batch_size, seq_length, config.n_embd)

def test_embedding_shapes(model_setup):
    """Test embedding layer shapes"""
    model, config, input_ids, _, batch_size, seq_length, device = model_setup
    
    with torch.no_grad():
        inputs_embeds = model.token_embedding(input_ids)
        position_embeds = model.position_embedding(torch.arange(0, seq_length, dtype=torch.long, device=device))
        hidden_states = inputs_embeds + position_embeds
    
    assert inputs_embeds.shape == (batch_size, seq_length, config.n_embd)
    assert position_embeds.shape == (seq_length, config.n_embd)
    assert hidden_states.shape == (batch_size, seq_length, config.n_embd)

def test_attention_shapes(model_setup):
    """Test attention computation shapes"""
    model, config, input_ids, _, batch_size, seq_length, device = model_setup
    
    with torch.no_grad():
        # Get embeddings
        inputs_embeds = model.token_embedding(input_ids)
        position_embeds = model.position_embedding(torch.arange(0, seq_length, dtype=torch.long, device=device))
        hidden_states = inputs_embeds + position_embeds
        
        # Get attention projections
        attn = model.blocks[0].attn
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)
        
        # Check projection shapes
        assert q.shape == (batch_size, seq_length, config.n_embd)
        assert k.shape == (batch_size, seq_length, config.n_embd)
        assert v.shape == (batch_size, seq_length, config.n_embd)
        
        # Test attention computation shapes
        q_reshaped = q.view(batch_size, seq_length, config.n_head, -1).transpose(1, 2)
        k_reshaped = k.view(batch_size, seq_length, config.n_head, -1).transpose(1, 2)
        v_reshaped = v.view(batch_size, seq_length, config.n_head, -1).transpose(1, 2)
        
        head_dim = config.n_embd // config.n_head
        assert q_reshaped.shape == (batch_size, config.n_head, seq_length, head_dim)
        assert k_reshaped.shape == (batch_size, config.n_head, seq_length, head_dim)
        assert v_reshaped.shape == (batch_size, config.n_head, seq_length, head_dim)
        
        # Test attention weights and output shapes
        attn_weights = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / attn.scale
        assert attn_weights.shape == (batch_size, config.n_head, seq_length, seq_length)
        
        attn_output = torch.matmul(attn_weights, v_reshaped)
        assert attn_output.shape == (batch_size, config.n_head, seq_length, head_dim)
        
        # Test final reshape
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, config.n_embd)
        assert attn_output.shape == (batch_size, seq_length, config.n_embd)

def test_head_dimensions(model_setup):
    """Test attention head dimensions"""
    model, config, _, _, _, _, _ = model_setup
    
    head_dim = config.n_embd // config.n_head
    assert model.blocks[0].attn.head_dim == head_dim
    assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
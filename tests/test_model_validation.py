import os
import sys
import torch
import pytest

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.llm_project.model.model import GPTModel

class TestModelArchitecture:
    @pytest.fixture
    def model_config(self):
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
    
    @pytest.fixture
    def model(self, model_config):
        """Create model instance for tests."""
        return GPTModel(**model_config)
    
    @pytest.fixture
    def sample_batch(self, model_config):
        """Create sample batch for testing."""
        batch_size = 4
        seq_len = model_config['max_seq_len']
        return {
            'input_ids': torch.randint(0, model_config['vocab_size'], (batch_size, seq_len)),
            'labels': torch.randint(0, model_config['vocab_size'], (batch_size, seq_len))
        }

    def test_model_initialization(self, model, model_config):
        """Test model initialization and basic attributes."""
        # Test embedding layers
        assert model.token_embedding.num_embeddings == model_config['vocab_size']
        assert model.token_embedding.embedding_dim == model_config['d_model']
        assert model.position_embedding.num_embeddings == model_config['max_seq_len']
        assert model.position_embedding.embedding_dim == model_config['d_model']
        
        # Test transformer blocks
        assert len(model.blocks) == model_config['n_layers']
        for block in model.blocks:
            # Test attention
            assert block.attn.n_heads == model_config['n_heads']
            assert block.attn.d_model == model_config['d_model']
            assert block.attn.d_head == model_config['d_model'] // model_config['n_heads']
            
            # Test feed-forward
            assert isinstance(block.ff[0], torch.nn.Module)  # ParallelLinear
            assert isinstance(block.ff[1], torch.nn.GELU)
            assert isinstance(block.ff[2], torch.nn.Module)  # ParallelLinear
            assert isinstance(block.ff[3], torch.nn.Dropout)
    
    def test_embedding_shapes(self, model, sample_batch, model_config):
        """Test embedding layer output shapes."""
        batch_size = sample_batch['input_ids'].size(0)
        seq_len = sample_batch['input_ids'].size(1)
        
        # Test token embeddings
        token_embeds = model.token_embedding(sample_batch['input_ids'])
        assert token_embeds.shape == (batch_size, seq_len, model_config['d_model'])
        
        # Test position embeddings
        pos_embeds = model.position_embedding(model.pos_indices[:seq_len])
        assert pos_embeds.shape == (seq_len, model_config['d_model'])
        
        # Test combined embeddings
        combined_embeds = model.dropout(token_embeds + pos_embeds.unsqueeze(0))
        assert combined_embeds.shape == (batch_size, seq_len, model_config['d_model'])
    
    def test_attention_shapes(self, model, sample_batch, model_config):
        """Test attention mechanism shapes."""
        batch_size = sample_batch['input_ids'].size(0)
        seq_len = sample_batch['input_ids'].size(1)
        
        # Get embeddings
        token_embeds = model.token_embedding(sample_batch['input_ids'])
        pos_embeds = model.position_embedding(model.pos_indices[:seq_len])
        hidden_states = token_embeds + pos_embeds.unsqueeze(0)
        
        # Test attention block
        block = model.blocks[0]
        
        # Test layer norm
        normed = block.ln1(hidden_states)
        assert normed.shape == (batch_size, seq_len, model_config['d_model'])
        
        # Test attention
        mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool),
            diagonal=1
        )
        attn_out = block.attn(normed, mask=mask)
        assert attn_out.shape == (batch_size, seq_len, model_config['d_model'])
    
    def test_feedforward_shapes(self, model, sample_batch, model_config):
        """Test feed-forward layer shapes."""
        batch_size = sample_batch['input_ids'].size(0)
        seq_len = sample_batch['input_ids'].size(1)
        
        # Get embeddings
        token_embeds = model.token_embedding(sample_batch['input_ids'])
        pos_embeds = model.position_embedding(model.pos_indices[:seq_len])
        hidden_states = token_embeds + pos_embeds.unsqueeze(0)
        
        # Test feed-forward block
        block = model.blocks[0]
        
        # Layer norm
        normed = block.ln2(hidden_states)
        assert normed.shape == (batch_size, seq_len, model_config['d_model'])
        
        # Feed-forward
        ff_out = block.ff(normed)
        assert ff_out.shape == (batch_size, seq_len, model_config['d_model'])
    
    def test_full_forward_pass(self, model, sample_batch, model_config):
        """Test shapes throughout a full forward pass."""
        batch_size = sample_batch['input_ids'].size(0)
        seq_len = sample_batch['input_ids'].size(1)
        
        # Forward pass without labels
        outputs = model(sample_batch['input_ids'])
        assert outputs.shape == (batch_size, seq_len, model_config['vocab_size'])
        
        # Forward pass with labels
        outputs = model(sample_batch['input_ids'], sample_batch['labels'])
        assert outputs.logits.shape == (batch_size, seq_len, model_config['vocab_size'])
        assert isinstance(outputs.loss.item(), float)
        assert outputs.loss.requires_grad
    
    def test_parameter_count(self, model, model_config):
        """Test model parameter counts."""
        # Calculate expected parameter counts
        embedding_params = (
            model_config['vocab_size'] * model_config['d_model'] +  # Token embeddings
            model_config['max_seq_len'] * model_config['d_model']   # Position embeddings
        )
        
        attention_params_per_layer = (
            3 * model_config['d_model'] * model_config['d_model'] +  # Q, K, V projections
            model_config['d_model'] * model_config['d_model']        # Output projection
        )
        
        ff_params_per_layer = (
            model_config['d_model'] * model_config['d_ff'] +  # First linear
            model_config['d_ff'] +                            # First bias
            model_config['d_ff'] * model_config['d_model'] +  # Second linear
            model_config['d_model']                           # Second bias
        )
        
        layer_norm_params_per_layer = 4 * model_config['d_model']  # 2 layer norms per layer
        
        transformer_params = model_config['n_layers'] * (
            attention_params_per_layer +
            ff_params_per_layer +
            layer_norm_params_per_layer
        )
        
        output_params = model_config['d_model'] * model_config['vocab_size']
        
        total_expected_params = embedding_params + transformer_params + output_params
        
        # Count actual parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Print parameter counts for debugging
        print(f"\nParameter count validation:")
        print(f"Embeddings: {embedding_params:,}")
        print(f"Per-layer attention: {attention_params_per_layer:,}")
        print(f"Per-layer FF: {ff_params_per_layer:,}")
        print(f"Total transformer: {transformer_params:,}")
        print(f"Output layer: {output_params:,}")
        print(f"Expected total: {total_expected_params:,}")
        print(f"Actual total: {total_params:,}")
        
        # Assert with some tolerance for potential miscalculation
        assert abs(total_params - total_expected_params) / total_expected_params < 0.1
    
    def test_device_movement(self, model, sample_batch):
        """Test model can be moved to GPU if available."""
        if torch.cuda.is_available():
            model = model.cuda()
            input_ids = sample_batch['input_ids'].cuda()
            labels = sample_batch['labels'].cuda()
            
            # Test forward pass on GPU
            outputs = model(input_ids, labels)
            assert outputs.logits.device.type == 'cuda'
            assert outputs.loss.device.type == 'cuda'
    
    def test_gradient_flow(self, model, sample_batch):
        """Test gradient flow through the model."""
        # Forward pass
        outputs = model(sample_batch['input_ids'], sample_batch['labels'])
        loss = outputs.loss
        
        # Clear any existing gradients
        model.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Check that all parameters that require gradients have them
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"
                assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradient"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
import torch
import torch.nn as nn
import math
from .parallel_utils import ParallelLinear, RowParallelLinear

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Get world size for parallel layers
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        # Adjust head dimensions for parallel processing
        assert n_heads % self.world_size == 0, "n_heads must be divisible by world_size"
        self.n_heads_per_rank = n_heads // self.world_size
        
        # Each rank gets d_model // world_size output features
        self.d_model_per_rank = d_model // self.world_size
        
        # Create query, key, value projections that keep outputs distributed
        self.q_proj = ParallelLinear(d_model, d_model, gather_output=False)
        self.k_proj = ParallelLinear(d_model, d_model, gather_output=False)
        self.v_proj = ParallelLinear(d_model, d_model, gather_output=False)
        
        # Output projection uses row parallelism to handle the distributed input
        self.out_proj = RowParallelLinear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Scaling factor
        self.scale = math.sqrt(self.d_head)
        
    def _debug_shape(self, tensor, name, rank):
        """Helper to print tensor shape information."""
        print(f"Rank {rank} - {name} shape: {tensor.shape}")
        
    def _prepare_mask(self, mask, batch_size, n_heads, seq_len, rank):
        """Prepare attention mask for different input formats."""
        if mask is None:
            return None
            
        self._debug_shape(mask, "Raw mask", rank)
        
        # Handle different mask shapes
        if mask.dim() == 3:  # [batch, seq, seq]
            # Add head dimension
            mask = mask.unsqueeze(1)  # [batch, 1, seq, seq]
        elif mask.dim() == 2:  # [batch, seq]
            # Create causal mask
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=mask.device), diagonal=1).bool()
            mask = mask | causal_mask
        
        # Expand for all heads
        mask = mask.expand(batch_size, n_heads, seq_len, seq_len)
        
        self._debug_shape(mask, "Prepared mask", rank)
        return mask
        
    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        """Compute scaled dot-product attention with explicit dimensions."""
        # q, k, v shape: [batch, n_heads_per_rank, seq, d_head]
        batch_size, n_heads, seq_len, d_head = q.size()
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, n_heads_per_rank, seq, seq]
        scores = scores / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)  # [batch, n_heads_per_rank, seq, seq]
        attn_weights = self.dropout(attn_weights)
        
        # Make tensors contiguous and reshape for batch matmul
        attn_weights = attn_weights.contiguous().view(-1, seq_len, seq_len)
        v_reshaped = v.contiguous().view(-1, seq_len, d_head)
        
        # Compute attention
        output = torch.bmm(attn_weights, v_reshaped)  # [batch*n_heads, seq, d_head]
        
        # Restore original shape
        output = output.view(batch_size, n_heads, seq_len, d_head)
        
        return output
        
    def forward(self, x, mask=None):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        batch_size, seq_len, _ = x.size()
        
        # Project queries, keys, values
        q = self.q_proj(x)  # [batch, seq, d_model_per_rank]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        self._debug_shape(q, "After projection", rank)
        
        # Verify projection output dimensions
        assert q.size(-1) == self.d_model_per_rank, \
            f"Projection output size {q.size(-1)} != expected {self.d_model_per_rank}"
        
        # First reshape to separate heads
        q = q.reshape(batch_size, seq_len, self.n_heads_per_rank, self.d_head)
        k = k.reshape(batch_size, seq_len, self.n_heads_per_rank, self.d_head)
        v = v.reshape(batch_size, seq_len, self.n_heads_per_rank, self.d_head)
        
        self._debug_shape(q, "After first reshape", rank)
        
        # Then transpose seq_len and n_heads
        q = q.transpose(1, 2).contiguous()  # [batch, n_heads_per_rank, seq, d_head]
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        self._debug_shape(q, "After transpose", rank)
        
        # Prepare attention mask
        mask = self._prepare_mask(mask, batch_size, self.n_heads_per_rank, seq_len, rank)
        
        # Compute attention with explicit dimension handling
        attn_output = self._scaled_dot_product_attention(q, k, v, mask)
        self._debug_shape(attn_output, "After attention", rank)
        
        # Transpose back: [batch, n_heads_per_rank, seq, d_head] -> [batch, seq, n_heads_per_rank, d_head]
        attn_output = attn_output.transpose(1, 2).contiguous()
        self._debug_shape(attn_output, "After transpose back", rank)
        
        # Merge heads: [batch, seq, n_heads_per_rank, d_head] -> [batch, seq, d_model_per_rank]
        attn_output = attn_output.reshape(batch_size, seq_len, self.n_heads_per_rank * self.d_head)
        self._debug_shape(attn_output, "After final reshape", rank)
        
        # Verify final shape before projection
        expected_size = batch_size * seq_len * self.d_model_per_rank
        actual_size = attn_output.numel()
        assert expected_size == actual_size, \
            f"Size mismatch before projection: expected {expected_size}, got {actual_size}"
        
        # Final output projection - uses row parallelism to handle distributed input
        attn_output = self.out_proj(attn_output)  # [batch, seq, d_model]
        self._debug_shape(attn_output, "After projection", rank)
        
        return attn_output
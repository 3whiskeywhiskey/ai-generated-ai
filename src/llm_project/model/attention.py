import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .parallel_utils import ColumnParallelLinear, RowParallelLinear

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        # Get world size for model parallelism
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        # Model dimensions
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_heads_per_rank = n_heads // self.world_size
        self.d_head = d_model // n_heads
        
        # Ensure n_heads is divisible by world_size
        assert n_heads % self.world_size == 0, \
            f"n_heads ({n_heads}) must be divisible by world_size ({self.world_size})"
        
        # Linear layers for parallel processing
        # Each projection produces [batch, seq, d_model] after gathering
        self.q_proj = ColumnParallelLinear(d_model, d_model)
        self.k_proj = ColumnParallelLinear(d_model, d_model)
        self.v_proj = ColumnParallelLinear(d_model, d_model)
        self.out_proj = RowParallelLinear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V - each projection produces [batch, seq, d_model]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Split heads: [batch, seq, n_heads, d_head]
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # Transpose: [batch, n_heads, seq, d_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            # Expand mask for attention heads
            mask = mask.unsqueeze(1)  # [batch, 1, seq, seq]
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, n_heads, seq, d_head]
        
        # Reshape: [batch, seq, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq, n_heads, d_head]
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output
import torch
import torch.nn as nn
import math
from .parallel_utils import ParallelLinear

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Create query, key, value projections
        self.q_proj = ParallelLinear(d_model, d_model)
        self.k_proj = ParallelLinear(d_model, d_model)
        self.v_proj = ParallelLinear(d_model, d_model)
        
        # Output projection
        self.out_proj = ParallelLinear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Scaling factor
        self.scale = math.sqrt(self.d_head)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, n_heads, seq_len, d_head]
        k = k.transpose(1, 2)  # [batch_size, n_heads, seq_len, d_head]
        v = v.transpose(1, 2)  # [batch_size, n_heads, seq_len, d_head]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, n_heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for batch size and heads
            # mask shape: [seq_len, seq_len] -> [1, 1, seq_len, seq_len]
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Apply softmax and dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Compute output
        out = torch.matmul(attn, v)  # [batch_size, n_heads, seq_len, d_head]
        out = out.transpose(1, 2)  # [batch_size, seq_len, n_heads, d_head]
        out = out.reshape(batch_size, seq_len, self.d_model)
        
        # Final output projection
        out = self.out_proj(out)
        
        return out 
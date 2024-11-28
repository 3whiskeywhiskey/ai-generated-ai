import torch
import torch.nn as nn
import math
from .parallel_utils import ParallelLinear

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd {n_embd} must be divisible by n_head {n_head}"
        
        self.n_embd = n_embd
        self.n_head = n_head
        self._head_dim = n_embd // n_head
        
        # Create query, key, value projections
        self.q_proj = ParallelLinear(n_embd, n_embd)
        self.k_proj = ParallelLinear(n_embd, n_embd)
        self.v_proj = ParallelLinear(n_embd, n_embd)
        
        # Output projection
        self.out_proj = ParallelLinear(n_embd, n_embd)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Register buffer for attention scaling
        self.register_buffer(
            "scale",
            torch.tensor(1.0 / math.sqrt(self._head_dim))
        )
        
    @property
    def head_dim(self):
        """Dimension of each attention head."""
        return self._head_dim
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self._head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self._head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self._head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, n_head, seq_len, d_head]
        k = k.transpose(1, 2)  # [batch_size, n_head, seq_len, d_head]
        v = v.transpose(1, 2)  # [batch_size, n_head, seq_len, d_head]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, n_head, seq_len, seq_len]
        
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
        out = torch.matmul(attn, v)  # [batch_size, n_head, seq_len, d_head]
        out = out.transpose(1, 2)  # [batch_size, seq_len, n_head, d_head]
        out = out.reshape(batch_size, seq_len, self.n_embd)
        
        # Final output projection
        out = self.out_proj(out)
        
        return out 
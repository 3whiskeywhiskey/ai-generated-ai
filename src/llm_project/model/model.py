import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .parallel_utils import ParallelLinear
import math

class GPTBlock(nn.Module):
    def __init__(self, n_embd, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            ParallelLinear(n_embd, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            ParallelLinear(d_ff, n_embd),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Pre-LN architecture
        attn_ln = self.ln1(x)
        attn_out = self.attn(attn_ln, mask=mask)
        x = x + attn_out
        
        # Feed-forward with Pre-LN
        ff_ln = self.ln2(x)
        ff_out = self.ff(ff_ln)
        x = x + ff_out
        
        return x

class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_positions,
        n_embd,
        n_layer,
        n_head,
        max_seq_len,
        dropout=0.1
    ):
        super().__init__()
        
        # Store config
        self.n_embd = n_embd
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(max_seq_len, n_embd)
        
        # Register buffer for positional indices
        pos_indices = torch.arange(max_seq_len)
        self.register_buffer("pos_indices", pos_indices)
        
        # Calculate feed-forward dimension (4x embedding dim)
        d_ff = 4 * n_embd
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(n_embd, n_head, d_ff, dropout)
            for _ in range(n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.output = nn.Linear(n_embd, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Use scaled initialization for better gradient flow
            if isinstance(module, nn.Linear):
                # Scale linear layers by 1/sqrt(fan_in)
                fan_in = module.weight.shape[1]
                std = 0.02 / math.sqrt(fan_in)
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            # Use smaller std for embeddings
            else:
                std = 0.01 / math.sqrt(self.n_embd)
                nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get sequence length and create attention mask
        batch_size, seq_len = input_ids.size()
        
        # Create causal mask [seq_len, seq_len]
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device),
            diagonal=1
        )
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            # Convert attention_mask to float and unsqueeze for broadcasting
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
            
            # Combine with causal mask
            causal_mask = causal_mask.unsqueeze(0)  # [1, seq_len, seq_len]
            combined_mask = causal_mask | (attention_mask == torch.finfo(torch.float32).min)
        else:
            combined_mask = causal_mask
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        pos_embeds = self.position_embedding(self.pos_indices[:seq_len])  # [seq_len, d_model]
        
        # Combine embeddings
        x = self.dropout(token_embeds + pos_embeds.unsqueeze(0))  # [batch_size, seq_len, d_model]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=combined_mask)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.output(x)  # [batch_size, seq_len, vocab_size]
        
        # Calculate loss if labels are provided
        if labels is not None:
            # Ensure inputs to loss function require gradients
            logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
            labels = labels.view(-1)  # [batch_size * seq_len]
            
            # CrossEntropyLoss automatically creates gradients
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
            # Ensure loss requires gradients
            if not loss.requires_grad:
                loss.requires_grad_(True)
            
            return type('ModelOutput', (), {
                'loss': loss,
                'logits': logits.view(batch_size, seq_len, -1),
                'hidden_states': x
            })()
        
        return {
            'logits': logits,
            'hidden_states': x
        }
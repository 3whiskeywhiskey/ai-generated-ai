import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .parallel_utils import ParallelLinear

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            ParallelLinear(d_model, d_ff),
            nn.GELU(),
            ParallelLinear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_out = self.attn(self.ln1(x), mask=mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward
        ff_out = self.ff(self.ln2(x))
        x = x + ff_out
        
        return x

class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        dropout=0.1
    ):
        super().__init__()
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Register buffer for positional indices
        pos_indices = torch.arange(max_seq_len)
        self.register_buffer("pos_indices", pos_indices)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.output = ParallelLinear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, labels=None):
        # Get sequence length and create attention mask
        batch_size, seq_len = input_ids.size()
        
        # Create causal mask [seq_len, seq_len]
        mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device),
            diagonal=1
        )
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        pos_embeds = self.position_embedding(self.pos_indices[:seq_len])  # [seq_len, d_model]
        
        # Combine embeddings
        x = self.dropout(token_embeds + pos_embeds.unsqueeze(0))  # [batch_size, seq_len, d_model]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)
        
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
            
            return type('ModelOutput', (), {'loss': loss, 'logits': logits.view(batch_size, seq_len, -1)})()
        
        return logits 
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
        
        # Pad vocab size to be divisible by world size
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self.padded_vocab_size = ((vocab_size + world_size - 1) // world_size) * world_size
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(self.padded_vocab_size, d_model)
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
        self.output = ParallelLinear(d_model, self.padded_vocab_size)
        
        # Store original vocab size for masking
        self.vocab_size = vocab_size
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Loss function
        self.loss_fct = nn.CrossEntropyLoss()
        
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
        
        # Create causal mask [batch_size, seq_len, seq_len]
        mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device),
            diagonal=1
        ).expand(batch_size, seq_len, seq_len)
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        pos_embeds = self.position_embedding(self.pos_indices[:seq_len])  # [seq_len, d_model]
        
        # Combine embeddings
        hidden_states = self.dropout(token_embeds + pos_embeds.unsqueeze(0))  # [batch_size, seq_len, d_model]
        
        # Apply transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, mask=mask)
        
        # Final layer norm and output projection
        hidden_states = self.ln_f(hidden_states)
        logits = self.output(hidden_states)
        
        # Mask out padding tokens in vocab
        if self.vocab_size < self.padded_vocab_size:
            logits[..., self.vocab_size:] = float('-inf')
        
        outputs = {'logits': logits}
        
        # Calculate loss if labels are provided
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            outputs['loss'] = loss
        
        return type('ModelOutput', (), outputs)() 
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .parallel_utils import ParallelLinear
from .config import ModelConfig
import math

class GPTBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = MultiHeadAttention(config.n_embd, config.n_head)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ff = nn.Sequential(
            ParallelLinear(config.n_embd, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            ParallelLinear(config.d_ff, config.n_embd),
            nn.Dropout(config.dropout)
        )
        self.dropout = nn.Dropout(config.dropout)
        
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
        vocab_size=None,
        max_seq_len=None,
        n_embd=None,
        n_layer=None,
        n_head=None,
        d_ff=None,
        dropout=None,
        config: ModelConfig = None,
        gradient_checkpointing=True  # Enable by default
    ):
        super().__init__()
        
        # Use provided config or create one from parameters
        if config is None:
            config = ModelConfig(
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                d_ff=d_ff,
                dropout=dropout
            )
        self.config = config
        
        # Store config
        self.n_embd = config.n_embd
        self.gradient_checkpointing = gradient_checkpointing
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.n_embd)
        
        # Register buffer for positional indices
        pos_indices = torch.arange(config.max_seq_len)
        self.register_buffer("pos_indices", pos_indices)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(config)
            for _ in range(config.n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.output = nn.Linear(config.n_embd, config.vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
        # Scale output layer initialization
        if isinstance(module, nn.Linear) and module.weight.shape[0] == self.config.vocab_size:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get embeddings
        b, t = input_ids.size()
        pos = self.pos_indices[:t]
        
        # Token + positional embeddings
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)
        
        # Apply transformer blocks with optional gradient checkpointing
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                x = block(x, attention_mask)
        
        x = self.ln_f(x)
        logits = self.output(x)
        
        # Loss calculation
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return logits, loss
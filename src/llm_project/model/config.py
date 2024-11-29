from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    # Model architecture parameters
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50304  # GPT-NeoX tokenizer vocab size
    max_seq_len: int = 512
    d_ff: Optional[int] = 3072  # Feed-forward dimension
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    
    # Tokenizer parameters
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 10
    grad_acc_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    
    # System parameters
    distributed: bool = True
    num_gpus: int = 4
    checkpoint_dir: str = "checkpoints"
    log_every: int = 10
    use_cache: bool = True

    def __post_init__(self):
        # Set d_ff to 4 * n_embd if not specified
        if self.d_ff is None:
            self.d_ff = 4 * self.n_embd
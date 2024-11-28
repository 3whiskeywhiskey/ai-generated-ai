from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    n_layer: int = 32
    n_head: int = 24
    n_embd: int = 2560
    vocab_size: int = 50257
    max_seq_len: int = 2048
    n_positions: int = 2048
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True
    d_ff: Optional[int] = None  # Feed-forward dimension, defaults to 4 * n_embd if None 
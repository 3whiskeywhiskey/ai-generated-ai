import os
import sys
import torch
import torch.nn as nn
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.llm_project.model.model import GPTModel as BaseGPTModel
from src.llm_project.model.attention import MultiHeadAttention

class CustomLinear(nn.Module):
    """Custom linear layer that exposes weight and bias directly."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class GPTBlock(nn.Module):
    def __init__(self, n_embd, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        
        # Create feed-forward network with named submodules to match checkpoint
        self.ff = nn.ModuleDict({
            '0': CustomLinear(n_embd, d_ff),
            '3': CustomLinear(d_ff, n_embd)
        })
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Attention block
        attn_ln = self.ln1(x)
        attn_out = self.attn(attn_ln, mask=mask)
        x = x + attn_out
        
        # Feed-forward block
        ff_ln = self.ln2(x)
        ff_mid = self.dropout(self.gelu(self.ff['0'](ff_ln)))
        ff_out = self.dropout(self.ff['3'](ff_mid))
        x = x + ff_out
        
        return x

class GPTModel(BaseGPTModel):
    def __init__(self, vocab_size, n_positions, n_embd, n_layer, n_head, max_seq_len, d_ff=None, dropout=0.1):
        # Call parent's init without creating blocks
        nn.Module.__init__(self)
        
        # Store config
        self.n_embd = n_embd
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(max_seq_len, n_embd)
        
        # Register buffer for positional indices
        pos_indices = torch.arange(max_seq_len)
        self.register_buffer("pos_indices", pos_indices)
        
        # Use provided d_ff or calculate default
        if d_ff is None:
            d_ff = 4 * n_embd
        
        # Transformer blocks with custom d_ff
        self.blocks = nn.ModuleList([
            GPTBlock(n_embd, n_head, d_ff, dropout)
            for _ in range(n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.output = nn.Linear(n_embd, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

def remap_state_dict(state_dict):
    """Remap state dict keys to match model structure."""
    new_state_dict = {}
    
    for k, v in state_dict.items():
        # Handle feed-forward layers
        if '.ff.' in k:
            if 'linear.weight' in k:
                new_k = k.replace('.linear.weight', '.weight')
            elif 'linear.bias' in k:
                new_k = k.replace('.linear.bias', '.bias')
            else:
                new_k = k
        else:
            new_k = k
        new_state_dict[new_k] = v
    
    return new_state_dict

def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Get model config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("Found model config in checkpoint")
    else:
        print("Inferring model config from state dict...")
        # Try to infer config from state dict
        state_dict = checkpoint['model_state_dict']
        n_embd = state_dict['token_embedding.weight'].shape[1]
        vocab_size = state_dict['token_embedding.weight'].shape[0]
        n_layer = max([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('blocks.')]) + 1
        
        # Print available keys for debugging
        print("\nAvailable keys in state dict:")
        for k in sorted(state_dict.keys()):
            print(f"  {k}")
        
        # Find attention weight key
        attn_key = next(k for k in state_dict.keys() if 'attn.q_proj.weight' in k)
        n_head = state_dict[attn_key].shape[0] // n_embd
        max_seq_len = state_dict['position_embedding.weight'].shape[0]
        
        # Get feed-forward dimensions from checkpoint
        ff_in_key = next(k for k in state_dict.keys() if k.startswith('blocks.0.ff.0.linear.weight'))
        ff_out_key = next(k for k in state_dict.keys() if k.startswith('blocks.0.ff.3.linear.weight'))
        
        # Get dimensions from both layers
        d_ff_in = state_dict[ff_in_key].shape[0]  # First layer expands
        d_ff_out = state_dict[ff_out_key].shape[1]  # Last layer contracts
        
        config = {
            'vocab_size': vocab_size,
            'max_seq_len': max_seq_len,
            'n_positions': max_seq_len,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
            'd_ff': d_ff_in,  # Use the expansion dimension
            'dropout': 0.1
        }
        
        print("\nInferred model config:")
        for k, v in config.items():
            print(f"{k}: {v}")
    
    # Create model and load state
    print("\nCreating model with inferred config...")
    model = GPTModel(**config)
    
    # Remap state dict keys and load
    print("Remapping state dict keys...")
    remapped_state_dict = remap_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(remapped_state_dict)
    
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model

def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=32,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    device='cuda'
):
    """Generate text from a prompt."""
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Initialize generation tracking
    generated = input_ids
    
    # Get max sequence length from config
    max_seq_len = 32  # Same as training config
    
    # Generate tokens
    with torch.no_grad():
        for _ in tqdm(range(max_new_tokens), desc="Generating"):
            # Get model outputs
            outputs = model(generated[:, -max_seq_len:])  # Only use last max_seq_len tokens
            
            # Get logits from outputs
            if isinstance(outputs, dict):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Get next token logits (last token only)
            next_token_logits = logits[:, -1, :].float()
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))  # Safety check
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[..., indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Stop if we generate an EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Generate text using trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt to start generation')
    parser.add_argument('--max_new_tokens', type=int, default=32, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k filtering value')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) filtering value')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID to use for generation')
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        if args.gpu_id >= torch.cuda.device_count():
            print(f"Warning: GPU {args.gpu_id} not available, defaulting to GPU 1")
            args.gpu_id = 1
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully")
    
    # Generate text
    print(f"\nPrompt: {args.prompt}")
    print("\nGenerating...")
    
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device
    )
    
    print("\nGenerated text:")
    print("=" * 50)
    print(generated_text)
    print("=" * 50)

if __name__ == "__main__":
    main() 
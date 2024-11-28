import os
import sys
import torch
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.llm_project.model.model import GPTModel

def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    config = {
        'vocab_size': 50304,  # GPT-NeoX tokenizer vocab size
        'max_seq_len': 32,    # Must match training config
        'n_layers': 2,
        'n_heads': 2,
        'd_model': 32,
        'd_ff': 128,
        'dropout': 0.1
    }
    
    # Create model and load state
    model = GPTModel(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
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
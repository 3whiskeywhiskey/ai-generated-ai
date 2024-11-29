import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.llm_project.model.model import GPTModel

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
        
        # Get FF layer dimensions from checkpoint
        ff_second_key = next(k for k in state_dict.keys() if k.startswith('blocks.0.ff.3.linear.weight'))
        ff_second_weight = state_dict[ff_second_key]
        d_ff = ff_second_weight.shape[1]  # Get d_ff from second layer input dimension
        
        config = {
            'vocab_size': vocab_size,
            'max_seq_len': max_seq_len,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
            'd_ff': d_ff,
            'dropout': 0.1
        }
    
    print("\nInferred model config:")
    for k, v in config.items():
        print(f"{k}: {v}")
    
    # Create model and load state
    print("\nCreating model with inferred config...")
    model = GPTModel(**config)
    
    # Reconstruct full weights from DDP-split weights
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    
    for k, v in state_dict.items():
        if 'ff.0.linear.weight' in k:
            # First FF layer was split across GPUs (output dim was split)
            # Original shape: [32, 32] x 4 GPUs = [128, 32]
            new_shape = (d_ff, v.shape[1])
            new_v = torch.zeros(new_shape, device=v.device)
            new_v[:v.shape[0]] = v
            new_state_dict[k] = new_v
        elif 'ff.0.linear.bias' in k:
            # Bias was also split
            new_shape = (d_ff,)
            new_v = torch.zeros(new_shape, device=v.device)
            new_v[:v.shape[0]] = v
            new_state_dict[k] = new_v
        elif 'ff.3.linear.weight' in k:
            # Second FF layer was split across GPUs (input dim was split)
            # Original shape: [8, 128] x 4 GPUs = [32, 128]
            new_shape = (n_embd, v.shape[1])
            new_v = torch.zeros(new_shape, device=v.device)
            new_v[:v.shape[0]] = v
            new_state_dict[k] = new_v
        elif 'ff.3.linear.bias' in k:
            # Bias was also split
            new_shape = (n_embd,)
            new_v = torch.zeros(new_shape, device=v.device)
            new_v[:v.shape[0]] = v
            new_state_dict[k] = new_v
        else:
            new_state_dict[k] = v
    
    print("Loading state dict...")
    model.load_state_dict(new_state_dict)
    
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
    min_tokens=10,  # Minimum tokens to generate before allowing EOS
    device='cuda'
):
    """Generate text from a prompt."""
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    print(f"Input shape: {input_ids.shape}")
    print(f"Input tokens: {[tokenizer.decode([t]) for t in input_ids[0].tolist()]}")
    
    # Initialize generation tracking
    generated = input_ids
    num_generated = 0
    
    # Get max sequence length from model config
    max_seq_len = model.config.max_seq_len if hasattr(model, 'config') else 1024
    print(f"Using max_seq_len: {max_seq_len}")
    
    # Generate tokens
    with torch.no_grad():
        for i in tqdm(range(max_new_tokens), desc="Generating"):
            # Get model outputs - model returns (logits, cache)
            current_input = generated[:, -max_seq_len:]
            print(f"\nStep {i}:")
            print(f"Current input shape: {current_input.shape}")
            
            outputs = model(current_input)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            print(f"Logits shape: {logits.shape}")
            
            # Get next token logits from last position
            next_token_logits = logits[:, -1, :].float()
            print(f"Next token logits shape: {next_token_logits.shape}")
            
            # Print logits statistics before temperature
            print(f"Logits mean: {next_token_logits.mean():.2f}, std: {next_token_logits.std():.2f}")
            print(f"Top 5 logits before temperature:")
            top_logits, top_indices = next_token_logits.topk(5)
            for logit, idx in zip(top_logits[0], top_indices[0]):
                print(f"  {tokenizer.decode([idx])} ({idx}): {logit:.2f}")
            
            # Prevent EOS before minimum length
            if num_generated < min_tokens:
                next_token_logits[0, tokenizer.eos_token_id] = float('-inf')
                print("Prevented EOS token (minimum length not reached)")
            
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
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[..., indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Print top probabilities after all filtering
            print(f"\nTop 5 tokens after filtering:")
            top_probs, top_indices = probs.topk(5)
            for prob, idx in zip(top_probs[0], top_indices[0]):
                print(f"  {tokenizer.decode([idx])} ({idx}): {prob:.4f}")
            
            next_token = torch.multinomial(probs, num_samples=1)
            print(f"\nSampled token: {next_token.item()} -> {tokenizer.decode([next_token.item()])}")
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=-1)
            num_generated += 1
            
            # Stop if we generate an EOS token
            if next_token.item() == tokenizer.eos_token_id and num_generated >= min_tokens:
                print("Generated EOS token after minimum length, stopping")
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
    parser.add_argument('--min_tokens', type=int, default=10, help='Minimum tokens to generate before allowing EOS')
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
        min_tokens=args.min_tokens,
        device=device
    )
    
    print("\nGenerated text:")
    print("=" * 50)
    print(generated_text)
    print("=" * 50)

if __name__ == "__main__":
    main() 
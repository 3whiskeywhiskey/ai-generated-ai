import os
import torch
import torch.multiprocessing as mp
from train_distributed import train
import math

def launch_distributed():
    # Get the number of available GPUs
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("No CUDA devices available")
    
    # Configuration - keep original small model
    model_config = {
        'vocab_size': 50304,  # GPT-NeoX tokenizer vocab size
        'max_seq_len': 32,    
        'n_positions': 32,    
        'n_layer': 2,
        'n_head': 2,         
        'n_embd': 32,        
        'dropout': 0.0  # Remove dropout initially
    }
    
    # Much more conservative learning rate
    base_lr = 1e-5  # Even smaller learning rate
    scaled_lr = base_lr  # No scaling for now
    
    # Smaller effective batch size for initial training
    global_batch_size = 32  # Reduced from 64
    grad_acc_steps = 2  # More frequent updates
    per_gpu_batch_size = max(1, global_batch_size // (world_size * grad_acc_steps))
    
    train_config = {
        'batch_size': per_gpu_batch_size,
        'learning_rate': scaled_lr,
        'weight_decay': 0.0,  # Remove weight decay initially
        'max_epochs': 10,
        'grad_acc_steps': grad_acc_steps,
        'save_steps': 100,
        'log_every': 1,
        'checkpoint_dir': 'checkpoints',
        'warmup_steps': 100,  # Longer warmup
        'max_grad_norm': 0.5  # Add gradient clipping
    }
    
    print(f"Starting distributed training on {world_size} GPUs")
    print(f"Global batch size: {per_gpu_batch_size * world_size * grad_acc_steps}")
    print(f"Per-GPU batch size: {per_gpu_batch_size}")
    print(f"Learning rate: {scaled_lr:.2e} (no scaling)")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(train_config['checkpoint_dir'], exist_ok=True)
    
    # Launch processes
    mp.spawn(
        train,
        args=(world_size, model_config, train_config),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    launch_distributed() 
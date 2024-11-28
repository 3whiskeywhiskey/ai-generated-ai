import os
import sys
import socket
import torch
import torch.distributed as dist
import wandb
import argparse
from datetime import datetime, timedelta

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.llm_project.model.model import GPTModel
from src.llm_project.data.dataset import create_dataloaders
from src.llm_project.training.trainer import Trainer

def setup_distributed(local_rank, world_size):
    """Initialize distributed training."""
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=5)
        )
        
    except Exception as e:
        print(f"Rank {local_rank}: Error during setup: {str(e)}")
        raise e

def cleanup():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    
    # Model configuration
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=128)
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--grad_acc_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    
    # System configuration
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--local-rank", type=int, default=0)
    
    args = parser.parse_args()
    
    try:
        local_rank = args.local_rank
        world_size = torch.cuda.device_count()
        
        if args.distributed:
            setup_distributed(local_rank, world_size)
        
        print(f"Rank {local_rank}: Creating model...")
        model = GPTModel(
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_model=args.d_model,
            d_ff=args.d_ff,
            max_seq_len=args.max_length,
            dropout=args.dropout,
            vocab_size=50257  # GPT-2 vocab size
        ).cuda(local_rank)
        print(f"Rank {local_rank}: Model created")
        
        print(f"Rank {local_rank}: Loading dataset...")
        train_loader, val_loader, train_sampler = create_dataloaders(
            batch_size=args.batch_size,
            max_length=args.max_length,
            cache_dir=None,
            seed=42
        )
        print(f"Rank {local_rank}: Dataset loaded")
        
        if local_rank == 0:
            print("Rank 0: Initializing wandb...")
            wandb.init(
                project="llm-training-small",
                config=vars(args),
                name=f"gpt-{args.d_model}d-{args.n_layers}l"
            )
        
        print(f"Rank {local_rank}: Creating trainer...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            grad_acc_steps=args.grad_acc_steps,
            max_grad_norm=args.max_grad_norm,
            warmup_steps=args.warmup_steps,
            local_rank=local_rank,
            checkpoint_dir=args.checkpoint_dir,
            log_every=args.log_every
        )
        print(f"Rank {local_rank}: Trainer created")
        
        print(f"Rank {local_rank}: Starting training...")
        trainer.train()
        print(f"Rank {local_rank}: Training complete")
        
    except Exception as e:
        print(f"Rank {local_rank}: Error during training: {str(e)}")
        raise e
        
    finally:
        cleanup()

if __name__ == "__main__":
    main() 
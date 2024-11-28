import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
import argparse
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.llm_project.model.model import GPTModel
from src.llm_project.data.dataset import create_dataloaders
from src.llm_project.training.trainer import Trainer

def setup(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()

def train(rank, world_size, args):
    """Main training function."""
    # Set up distributed training
    if args.distributed:
        setup(rank, world_size)
    
    # Create model
    model = GPTModel(
        vocab_size=50304,  # GPT-NeoX tokenizer vocab size
        max_seq_len=args.max_length,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout
    )
    
    # Create dataloaders
    train_loader, val_loader, train_sampler = create_dataloaders(
        batch_size=args.batch_size,
        max_length=args.max_length,
        cache_dir=args.cache_dir
    )
    
    # Initialize wandb
    if rank == 0:
        run_name = f"gpt-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="llm-project",
            name=run_name,
            config=vars(args)
        )
    
    # Create trainer
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
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        distributed=args.distributed,
        local_rank=rank
    )
    
    # Load checkpoint if provided
    if args.checkpoint_path:
        start_epoch, _ = trainer.load_checkpoint(args.checkpoint_path)
    else:
        start_epoch = 0
    
    # Train
    trainer.train(start_epoch=start_epoch)
    
    # Clean up
    if args.distributed:
        cleanup()
    
    if rank == 0:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train GPT model on TinyStories dataset")
    
    # Model arguments
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--max_length", type=int, default=16, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--grad_acc_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    
    # System arguments
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--cache_dir", type=str, help="Directory for caching datasets")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    
    args = parser.parse_args()
    
    if args.distributed:
        mp.spawn(
            train,
            args=(args.num_gpus, args),
            nprocs=args.num_gpus,
            join=True
        )
    else:
        train(0, 1, args)

if __name__ == "__main__":
    main() 
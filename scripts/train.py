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
from src.llm_project.model.config import ModelConfig
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
    # Create config from args
    config = ModelConfig(
        n_layer=args.n_layers,
        n_head=args.n_heads,
        n_embd=args.d_model,
        max_seq_len=args.max_length,
        n_positions=args.max_length,
        d_ff=args.d_ff,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        grad_acc_steps=args.grad_acc_steps,
        max_grad_norm=args.max_grad_norm,
        distributed=args.distributed,
        num_gpus=args.num_gpus,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every
    )
    
    # Set up distributed training
    if config.distributed:
        setup(rank, world_size)
    
    # Create model using config directly
    model = GPTModel(config=config)
    
    # Create dataloaders
    train_loader, val_loader, train_sampler = create_dataloaders(
        batch_size=config.batch_size,
        max_length=config.max_seq_len,
        cache_dir=args.cache_dir
    )
    
    # Initialize wandb
    if rank == 0:
        run_name = f"gpt-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="llm-project",
            name=run_name,
            config=vars(config)  # Log the full config
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
        grad_acc_steps=config.grad_acc_steps,
        max_grad_norm=config.max_grad_norm,
        checkpoint_dir=config.checkpoint_dir,
        log_every=config.log_every,
        distributed=config.distributed,
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
    if config.distributed:
        cleanup()
    
    if rank == 0:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train GPT model on TinyStories dataset")
    
    # Model arguments
    parser.add_argument("--n_layers", type=int, default=ModelConfig.n_layer, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=ModelConfig.n_head, help="Number of attention heads")
    parser.add_argument("--d_model", type=int, default=ModelConfig.n_embd, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=ModelConfig.d_ff, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=ModelConfig.dropout, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=ModelConfig.batch_size, help="Batch size per GPU")
    parser.add_argument("--max_length", type=int, default=ModelConfig.max_seq_len, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=ModelConfig.learning_rate, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=ModelConfig.weight_decay, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=ModelConfig.max_epochs, help="Number of epochs")
    parser.add_argument("--grad_acc_steps", type=int, default=ModelConfig.grad_acc_steps, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=ModelConfig.max_grad_norm, help="Maximum gradient norm")
    
    # System arguments
    parser.add_argument("--distributed", action="store_true", default=ModelConfig.distributed, help="Use distributed training")
    parser.add_argument("--num_gpus", type=int, default=ModelConfig.num_gpus, help="Number of GPUs to use")
    parser.add_argument("--checkpoint_dir", type=str, default=ModelConfig.checkpoint_dir, help="Directory for checkpoints")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--cache_dir", type=str, help="Directory for caching datasets")
    parser.add_argument("--log_every", type=int, default=ModelConfig.log_every, help="Log every N steps")
    
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
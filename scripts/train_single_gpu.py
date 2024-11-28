import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.llm_project.model.model import GPTModel
from src.llm_project.data.dataset import create_dataloaders

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, checkpoint_dir, is_best=False, is_epoch=False):
    """Save model checkpoint."""
    print(f"Saving checkpoint - Best: {is_best}, Epoch: {is_epoch}, Step: {step}")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'step': step,
        'loss': loss
    }
    
    # Create checkpoint filename based on type
    if is_best:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_best.pt')
    elif is_epoch:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
    
    print(f"Saving to path: {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)
    print(f"Successfully saved checkpoint to {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    max_epochs=10,
    checkpoint_dir='checkpoints',
    checkpoint_path=None,
    log_every=10,
    grad_acc_steps=8
):
    """Training loop for single GPU."""
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Initialize wandb
    wandb.init(
        project="llm-project",
        config={
            "model_config": model.config if hasattr(model, 'config') else None,
            "max_epochs": max_epochs,
            "batch_size": train_loader.batch_size if hasattr(train_loader, 'batch_size') else None,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "grad_acc_steps": grad_acc_steps
        }
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    global_step = 0
    steps_this_epoch = 0
    best_val_loss = float('inf')
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        start_epoch, global_step, _ = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
        
        # Calculate how many steps we've done in the current epoch
        steps_per_epoch = len(train_loader)
        steps_this_epoch = global_step % steps_per_epoch
        
        print(f"Resuming from epoch {start_epoch}, global step {global_step}")
        print(f"Starting at step {steps_this_epoch}/{steps_per_epoch} of epoch {start_epoch}")
        
        # Adjust scheduler steps to match global step
        for _ in range(global_step):
            scheduler.step()
    
    # Training loop
    for epoch in range(start_epoch, max_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        # Create progress bar starting from the correct step
        progress_bar = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch}",
            initial=steps_this_epoch,  # Start from where we left off
            total=len(train_loader)
        )
        
        # Skip steps we've already done in this epoch
        for step, batch in enumerate(train_loader):
            if step < steps_this_epoch:
                continue
                
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss / grad_acc_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every grad_acc_steps or at the end of epoch
            if (step + 1) % grad_acc_steps == 0 or (step + 1) == len(train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler:
                    scheduler.step()
                
                # Increment global step
                global_step += 1
                
                # Save periodic checkpoint
                if global_step > 0 and global_step % 1000 == 0:
                    print(f"\nSaving checkpoint at step {global_step}")
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=global_step,
                        loss=loss.item() * grad_acc_steps,
                        checkpoint_dir=checkpoint_dir,
                        is_best=False
                    )
            
            # Update metrics (use unscaled loss for logging)
            total_loss += (loss.item() * grad_acc_steps)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item() * grad_acc_steps,
                "lr": optimizer.param_groups[0]['lr'],
                "global_step": global_step,
                "epoch_step": step
            })
            progress_bar.update(1)
            
            # Log metrics
            if step % log_every == 0:
                wandb.log({
                    "train/loss": loss.item() * grad_acc_steps,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/epoch": epoch,
                    "train/step": global_step,
                    "train/epoch_progress": step / len(train_loader)
                })
        
        # Reset steps_this_epoch for next epoch
        steps_this_epoch = 0
        progress_bar.close()
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)
        
        # Save epoch checkpoint
        print(f"\nSaving epoch {epoch} checkpoint")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=global_step,
            loss=avg_train_loss,
            checkpoint_dir=checkpoint_dir,
            is_best=False,
            is_epoch=True
        )
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Log epoch metrics
        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": avg_val_loss,
            "epoch": epoch,
            "epoch_complete": True
        })
        
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"\nNew best validation loss: {avg_val_loss:.4f}, saving checkpoint")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                loss=avg_val_loss,
                checkpoint_dir=checkpoint_dir,
                is_best=True
            )

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith('checkpoint_step_') and f.endswith('.pt'):
            try:
                step = int(f.split('_')[2].split('.')[0])  # Extract step number before .pt
                checkpoints.append((step, os.path.join(checkpoint_dir, f)))
            except:
                continue
    
    if not checkpoints:
        return None
        
    # Sort by step number and return the latest
    latest_checkpoint = sorted(checkpoints, key=lambda x: x[0])[-1]
    print(f"Found latest checkpoint at step {latest_checkpoint[0]}")
    return latest_checkpoint[1]

def main():
    # Model configuration
    model_config = {
        'vocab_size': 50304,  # GPT-NeoX tokenizer vocab size
        'max_seq_len': 32,    # Reduced from 64
        'n_positions': 32,    # Same as max_seq_len
        'n_layer': 2,
        'n_head': 2,         # Reduced from 4
        'n_embd': 32,        # Reduced from 64
        'dropout': 0.1
    }
    
    # Training configuration
    train_config = {
        'batch_size': 4,      # Reduced from 16
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'max_epochs': 10,
        'checkpoint_dir': 'checkpoints',
        'log_every': 10,
        'grad_acc_steps': 16  # Increased from 8
    }
    
    # Find latest checkpoint first
    latest_checkpoint = find_latest_checkpoint(train_config['checkpoint_dir'])
    if latest_checkpoint:
        print(f"Found checkpoint: {latest_checkpoint}")
        print("Resuming training from checkpoint...")
    else:
        print("No checkpoint found, starting training from scratch...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = GPTModel(**model_config)
    model = model.to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params:,} parameters")
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=train_config['batch_size'],
        max_length=model_config['max_seq_len']
    )
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * train_config['max_epochs'] // train_config['grad_acc_steps']
    )
    
    # Train model
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_epochs=train_config['max_epochs'],
        checkpoint_dir=train_config['checkpoint_dir'],
        checkpoint_path=latest_checkpoint,
        log_every=train_config['log_every'],
        grad_acc_steps=train_config['grad_acc_steps']
    )

if __name__ == "__main__":
    main() 
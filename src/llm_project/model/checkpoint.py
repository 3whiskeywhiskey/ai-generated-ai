import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.pt') and f.startswith('checkpoint_epoch_'):
            try:
                checkpoint_path = os.path.join(checkpoint_dir, f)
                # Load checkpoint to get actual step number, using weights_only for safety
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                epoch = checkpoint['epoch']
                step = checkpoint['step']
                checkpoints.append((epoch, step, checkpoint_path))
            except:
                continue
    
    if not checkpoints:
        return None
    
    # Sort by epoch first, then by step
    latest_checkpoint = sorted(checkpoints, key=lambda x: (x[0], x[1]))[-1]
    print(f"Found latest checkpoint at epoch {latest_checkpoint[0]}, step {latest_checkpoint[1]}: {latest_checkpoint[2]}")
    return latest_checkpoint[2]

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, checkpoint_dir, is_main_process=False, is_best=False, scaler=None):
    """Save model checkpoint."""
    if not is_main_process:
        return
        
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get model state dict
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    # Get optimizer state dict and ensure it's on CPU
    optimizer_state = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v
        for k, v in optimizer.state_dict().items()
    }
    
    # Get scheduler state dict
    scheduler_state = scheduler.state_dict() if scheduler else None
    
    # Get scaler state dict
    scaler_state = scaler.state_dict() if scaler else None
    
    checkpoint = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'scheduler_state_dict': scheduler_state,
        'scaler_state_dict': scaler_state,
        'epoch': epoch,
        'step': step,
        'loss': loss
    }
    
    # Save checkpoint with both epoch and step
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best checkpoint if specified
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path, _use_new_zipfile_serialization=True)
        print(f"Saved best checkpoint to {best_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, rank, scaler=None):
    """Load model checkpoint."""
    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint to CPU first, using weights_only for safety
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # Load model state
    if rank == 0:
        print("Loading model state...")
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move optimizer states to correct device and load
    if rank == 0:
        print("Loading optimizer state...")
    optimizer_state = checkpoint['optimizer_state_dict']
    
    # Handle nested optimizer state
    def _move_to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cuda(rank)
        elif isinstance(obj, dict):
            return {k: _move_to_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_move_to_device(v) for v in obj]
        else:
            return obj
    
    # Move optimizer state to correct device
    optimizer_state = _move_to_device(optimizer_state)
    optimizer.load_state_dict(optimizer_state)
    
    # Load scheduler state
    if scheduler and checkpoint['scheduler_state_dict']:
        if rank == 0:
            print("Loading scheduler state...")
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load scaler state if it exists
    if scaler and 'scaler_state_dict' in checkpoint:
        if rank == 0:
            print("Loading gradient scaler state...")
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    if rank == 0:
        print(f"Successfully loaded checkpoint from step {checkpoint['step']} (epoch {checkpoint['epoch']})")
    
    return checkpoint['epoch'], checkpoint['step'] 
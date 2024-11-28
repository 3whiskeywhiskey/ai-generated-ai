import os
import sys
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb
import argparse
from datetime import datetime, timedelta
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.llm_project.model.model import GPTModel
from src.llm_project.data.dataset import create_dataloaders

class ResumeDistributedSampler(DistributedSampler):
    """DistributedSampler that supports starting from a specific index."""
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        self.start_index = 0
    
    def set_start_index(self, start_index):
        """Set the starting index for the sampler."""
        self.start_index = start_index
    
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample based on rank and start index
        start_idx = self.start_index * self.num_replicas + self.rank
        indices = indices[start_idx:self.total_size:self.num_replicas]
        
        return iter(indices)

def print_gpu_topology():
    """Print detailed GPU topology information."""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        print("\nGPU Topology:")
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            print(f"\nGPU {i}: {name}")
            
            # Check NVLink connections
            try:
                for j in range(device_count):
                    if i != j:
                        nvlink_status = pynvml.nvmlDeviceGetNvLinkState(handle, j)
                        if nvlink_status == pynvml.NVML_NVLINK_STATUS_NOT_SUPPORTED:
                            print(f"  -> GPU {j}: No NVLink support")
                        else:
                            nvlink_version = pynvml.nvmlDeviceGetNvLinkVersion(handle, j)
                            print(f"  -> GPU {j}: NVLink v{nvlink_version}")
            except pynvml.NVMLError:
                print(f"  NVLink information not available")
            
            # Check P2P capabilities
            for j in range(device_count):
                if i != j:
                    can_access = torch.cuda.can_device_access_peer(i, j)
                    print(f"  -> GPU {j}: P2P {'Enabled' if can_access else 'Disabled'}")
        
        print("\nPCI Topology:")
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
            print(f"GPU {i}: Bus ID {pci_info.busId.decode('utf-8')}")
            
    except Exception as e:
        print(f"Could not detect GPU topology: {str(e)}")
    print()

def setup_distributed(rank, world_size):
    """Initialize distributed environment."""
    # Use fixed port and localhost
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'  # Fixed port
    
    # Configure NCCL for optimal GPU communication
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_P2P_DISABLE'] = '0'
    os.environ['NCCL_P2P_LEVEL'] = '5'
    
    # Force local communication
    os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET'  # Force IPv4
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo'  # Use loopback for Gloo backup
    
    if rank == 0:
        print("\nDistributed Configuration:")
        print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
        print(f"WORLD_SIZE: {world_size}")
        print(f"RANK: {rank}")
        print("\nNCCL Configuration:")
        for k, v in sorted(os.environ.items()):
            if k.startswith('NCCL_'):
                print(f"{k}: {v}")
        print()
    
    # Initialize process group with both NCCL and Gloo backends
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:29500",
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)
    if rank == 0:
        print(f"Process {rank}: CUDA device set to {torch.cuda.current_device()}")
        print(f"Process group initialized successfully!")

def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.pt') and (f.startswith('checkpoint_step_') or f.startswith('checkpoint_epoch_')):
            try:
                checkpoint_path = os.path.join(checkpoint_dir, f)
                # Load checkpoint to get actual step number
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                actual_step = checkpoint['step']
                checkpoints.append((actual_step, checkpoint_path))
            except:
                continue
    
    if not checkpoints:
        return None
    
    # Sort by actual step number and return the latest
    latest_checkpoint = sorted(checkpoints, key=lambda x: x[0])[-1]
    print(f"Found latest checkpoint at step {latest_checkpoint[0]}: {latest_checkpoint[1]}")
    return latest_checkpoint[1]

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, checkpoint_dir, is_main_process=False, is_best=False, is_epoch=False):
    """Save model checkpoint."""
    if not is_main_process:
        return
        
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get model state dict
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'step': step,
        'loss': loss
    }
    
    # Save step checkpoint
    if is_epoch:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best checkpoint if specified
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint to {best_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, rank):
    """Load model checkpoint."""
    if rank == 0:
        print(f"  Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint to CPU first, with weights_only=True for security
    if rank == 0:
        print("  Loading state dict...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # Remap state dict keys if needed
    if rank == 0:
        print("  Remapping state dict keys...")
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'ff.' in k and '.linear.' not in k:
            parts = k.split('.')
            # Insert 'linear' before 'weight' or 'bias'
            new_parts = parts[:-1] + ['linear'] + [parts[-1]]
            new_k = '.'.join(new_parts)
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    
    # Load model state
    if rank == 0:
        print("  Loading model state...")
    if isinstance(model, DDP):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    
    # Move optimizer states to correct device
    if rank == 0:
        print("  Loading optimizer state...")
    optimizer_state = checkpoint['optimizer_state_dict']
    for state in optimizer_state['state'].values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(rank)
    optimizer.load_state_dict(optimizer_state)
    
    if scheduler and checkpoint['scheduler_state_dict']:
        if rank == 0:
            print("  Loading scheduler state...")
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Calculate correct epoch from step number
    steps_per_epoch = len(train_loader) // train_config['grad_acc_steps']
    epoch = checkpoint['step'] // steps_per_epoch
    
    if rank == 0:
        print(f"  Loaded checkpoint from step {checkpoint['step']} (epoch {epoch})")
    
    return epoch, checkpoint['step']

def train(rank, world_size, model_config, train_config):
    """Training function for each process."""
    setup_distributed(rank, world_size)
    torch.manual_seed(42 + rank)
    
    # Set debug level for distributed training
    if rank == 0:
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    
    # Create model and move to GPU
    model = GPTModel(**model_config)
    device = f'cuda:{rank}'
    model = model.to(device)
    
    # Initialize wandb on rank 0 only
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        wandb.init(
            project="llm-project",
            config={
                "model_config": model_config,
                "train_config": train_config,
                "world_size": world_size,
                "effective_batch_size": train_config['batch_size'] * world_size * train_config['grad_acc_steps'],
                "total_params": total_params,
            },
            name=f"distributed_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            resume=True  # Allow wandb to resume from previous run
        )
    
    # Print model size and initial stats on rank 0
    if rank == 0:
        print(f"\nModel size: {total_params:,} parameters")
    
    # Wrap model with DDP, using static graph optimization
    model = DDP(
        model,
        device_ids=[rank],
        find_unused_parameters=False,
        broadcast_buffers=False
    )
    model._set_static_graph()
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    
    # Create dataloaders with fixed batch size
    train_loader, val_loader, train_sampler = create_dataloaders(
        batch_size=train_config['batch_size'],
        max_length=model_config['max_seq_len']
    )
    
    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * train_config['max_epochs'] // train_config['grad_acc_steps']
    )
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_loader) // train_config['grad_acc_steps']
    
    # Load checkpoint if available
    start_epoch = 0
    global_step = 0
    steps_this_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = find_latest_checkpoint(train_config['checkpoint_dir'])
    
    if checkpoint_path:
        if rank == 0:
            print(f"\nLoading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        # Calculate correct epoch from step number
        global_step = checkpoint['step']
        steps_per_epoch = len(train_loader) // train_config['grad_acc_steps']
        start_epoch = (global_step // steps_per_epoch) - 1  # Subtract 1 to make it 0-based
        steps_this_epoch = (global_step % steps_per_epoch) * train_config['grad_acc_steps']
        
        # Load model state
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer_state = checkpoint['optimizer_state_dict']
        for state in optimizer_state['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(rank)
        optimizer.load_state_dict(optimizer_state)
        
        # Load scheduler state
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Adjust scheduler steps
        for _ in range(global_step):
            scheduler.step()
            
        if rank == 0:
            print(f"Resuming from step {global_step} (epoch {start_epoch})")
            print(f"Starting at step {steps_this_epoch}/{len(train_loader)} in current epoch")
    elif rank == 0:
        print("\nStarting training from scratch")
    
    if rank == 0:
        print(f"\nTraining Configuration:")
        print(f"Learning Rate: {train_config['learning_rate']}")
        print(f"Weight Decay: {train_config['weight_decay']}")
        print(f"Batch Size per GPU: {train_config['batch_size']}")
        print(f"Effective Batch Size: {train_config['batch_size'] * world_size * train_config['grad_acc_steps']}")
        print(f"Gradient Accumulation Steps: {train_config['grad_acc_steps']}")
        print(f"Starting from epoch {start_epoch}, step {global_step}")
        print(f"Using static graph optimization")
    
    # Training loop
    for epoch in range(start_epoch, train_config['max_epochs']):
        if rank == 0:
            print(f"\nProcess {rank}: Starting epoch {epoch}")
        
        # Set epoch and start index for sampler
        train_sampler.set_epoch(epoch)
        if steps_this_epoch > 0:
            if rank == 0:
                print(f"Process {rank}: Setting start index to {steps_this_epoch}")
            train_sampler.set_start_index(steps_this_epoch)
        else:
            train_sampler.set_start_index(0)
        
        if rank == 0:
            print(f"Process {rank}: Creating progress bar...")
        progress = tqdm(
            total=len(train_loader),
            initial=steps_this_epoch,
            desc=f"Training Epoch {epoch}",
            disable=rank != 0
        )
        
        model.train()
        running_loss = 0
        optimizer.zero_grad()
        
        # Start training from the correct step
        for batch_idx, batch in enumerate(train_loader, start=steps_this_epoch):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / train_config['grad_acc_steps']
            
            # Debug first batch of first epoch
            if rank == 0 and global_step == 0:
                logits = outputs.logits
                print("\nFirst batch analysis:")
                print(f"Input shape: {input_ids.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Logits shape: {logits.shape}")
                print(f"Logits mean/std: {logits.mean().item():.6f}/{logits.std().item():.6f}")
                print(f"Loss: {loss.item() * train_config['grad_acc_steps']:.6f}")
                
                print("\nSample logits from first sequence:")
                print(logits[0, 0, :10].tolist())
                
                print("\nLabel statistics:")
                print(f"Label range: {labels.min().item()} to {labels.max().item()}")
                print(f"Unique labels: {torch.unique(labels).tolist()}")
            
            # Backward pass
            loss.backward()
            
            # Update weights every grad_acc_steps or at the end of epoch
            if (batch_idx + 1) % train_config['grad_acc_steps'] == 0 or (batch_idx + 1) == len(train_loader):
                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer and scheduler step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Increment global step
                global_step += 1
                
                # Save checkpoint every N steps
                if rank == 0 and global_step % train_config['save_steps'] == 0:
                    save_checkpoint(
                        model=model.module,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=global_step,
                        loss=loss.item() * train_config['grad_acc_steps'],
                        checkpoint_dir=train_config['checkpoint_dir'],
                        is_main_process=True
                    )
                
                # Log training metrics on rank 0
                if rank == 0:
                    # Update progress bar
                    progress.update(train_config['grad_acc_steps'])
                    progress.set_description(
                        f"Training Epoch {epoch}: {batch_idx}/{len(train_loader)} "
                        f"[loss={loss.item() * train_config['grad_acc_steps']:.3f}, "
                        f"lr={scheduler.get_last_lr()[0]:.6f}, "
                        f"grad_norm={grad_norm:.3f}]"
                    )
                    
                    # Log to wandb
                    wandb.log({
                        'train/loss': loss.item() * train_config['grad_acc_steps'],
                        'train/learning_rate': scheduler.get_last_lr()[0],
                        'train/grad_norm': grad_norm,
                        'train/epoch': epoch,
                        'train/global_step': global_step,
                        'train/epoch_progress': batch_idx / len(train_loader)
                    }, step=global_step)
            
            running_loss += (loss.item() * train_config['grad_acc_steps'])
        
        # Reset steps_this_epoch for next epoch
        steps_this_epoch = 0
        progress.close()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for val_batch in val_loader:
                val_input_ids = val_batch['input_ids'].to(device)
                val_labels = val_batch['labels'].to(device)
                
                val_outputs = model(val_input_ids, labels=val_labels)
                val_loss = val_outputs.loss
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        if rank == 0:
            print(f"\nEpoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Save epoch checkpoint
            save_checkpoint(
                model=model.module,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                loss=avg_train_loss,
                checkpoint_dir=train_config['checkpoint_dir'],
                is_main_process=True,
                is_epoch=True
            )
            
            # Log epoch metrics to wandb
            wandb.log({
                'epoch/train_loss': avg_train_loss,
                'epoch/val_loss': avg_val_loss,
                'epoch': epoch,
            }, step=global_step)
            
            # Save if validation improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(
                    model=model.module,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    loss=avg_val_loss,
                    checkpoint_dir=train_config['checkpoint_dir'],
                    is_main_process=True,
                    is_best=True
                )
                
                # Log best metrics to wandb
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch
    
    # Cleanup
    if rank == 0:
        wandb.finish()
    cleanup_distributed()

def main():
    # Model configuration - match single GPU version exactly
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
        'batch_size': 4,      # Per GPU batch size
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'max_epochs': 10,
        'checkpoint_dir': 'checkpoints',
        'log_every': 10,
        'grad_acc_steps': 16,  # Same as single GPU
        'save_steps': 1000
    }
    
    # Launch distributed training
    world_size = torch.cuda.device_count()
    mp.spawn(
        train,
        args=(world_size, model_config, train_config),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main() 
import os
import sys
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb
import argparse
from datetime import datetime, timedelta
import random

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.llm_project.model.model import GPTModel
from src.llm_project.data.dataset import create_dataloaders

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
        if f.startswith('checkpoint_step_') and f.endswith('.pt'):
            try:
                step = int(f.split('_')[2].split('.')[0])
                checkpoints.append((step, os.path.join(checkpoint_dir, f)))
            except:
                continue
    
    if not checkpoints:
        return None
        
    latest_checkpoint = sorted(checkpoints, key=lambda x: x[0])[-1]
    return latest_checkpoint[1]

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, checkpoint_dir, is_main_process=False):
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
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, rank):
    """Load model checkpoint."""
    # Load checkpoint to CPU first to avoid GPU RAM issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer and scheduler states on the appropriate device
    optimizer_state = checkpoint['optimizer_state_dict']
    for state in optimizer_state['state'].values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(rank)
    optimizer.load_state_dict(optimizer_state)
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['step']

def train(rank, world_size, model_config, train_config):
    """Training function for each process."""
    setup_distributed(rank, world_size)
    
    # Create model and move to GPU
    model = GPTModel(**model_config)
    device = f'cuda:{rank}'
    model = model.to(device)
    
    if rank == 0:
        print("\nModel Configuration:")
        print(f"Vocab Size: {model_config['vocab_size']}")
        print(f"Embedding Dim: {model_config['n_embd']}")
        print(f"Num Layers: {model_config['n_layer']}")
        print(f"Num Heads: {model_config['n_head']}")
        print(f"Max Sequence Length: {model_config['max_seq_len']}")
        
        # Check initial weights
        print("\nInitial weight stats:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: mean={param.data.mean().item():.3f}, std={param.data.std().item():.3f}")
    
    # Wrap model for distributed training
    model = DDP(model, device_ids=[rank])
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    
    # Create dataloaders
    train_loader, val_loader, train_sampler = create_dataloaders(
        batch_size=train_config['batch_size'],
        max_length=model_config['max_seq_len']
    )
    
    # Training loop
    global_step = 0
    optimizer.zero_grad()
    
    for epoch in range(train_config['max_epochs']):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        model.train()
        progress_bar = tqdm(total=len(train_loader), disable=rank != 0)
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Debug first batch
            if rank == 0 and global_step == 0:
                print("\nFirst batch debug:")
                print(f"Input shape: {input_ids.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Labels min/max: {labels.min().item()}, {labels.max().item()}")
                print(f"Output type: {type(outputs)}")
                print(f"Loss: {loss.item():.3f}")
                print(f"Loss requires grad: {loss.requires_grad}")
                
                # Check logits
                logits = outputs.logits
                print(f"\nLogits stats:")
                print(f"Logits shape: {logits.shape}")
                print(f"Logits mean/std: {logits.mean().item():.3f}, {logits.std().item():.3f}")
                probs = torch.softmax(logits[:, 0], dim=-1)  # Look at first position
                print(f"Probs mean/std: {probs.mean().item():.3f}, {probs.std().item():.3f}")
                print(f"Max prob: {probs.max().item():.3f}")
                
                # Check parameters
                print("\nParameter stats:")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(f"{name}: mean={param.data.mean().item():.3f}, std={param.data.std().item():.3f}")
            
            # Scale loss for gradient accumulation
            loss = loss / train_config['grad_acc_steps']
            
            # Backward pass
            loss.backward()
            
            # Monitor gradients
            if rank == 0 and (global_step == 0 or global_step % 100 == 99):
                total_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"\nStep {global_step} gradient norm: {total_norm:.3f}")
            
            # Gradient accumulation and optimization step
            if (batch_idx + 1) % train_config['grad_acc_steps'] == 0:
                if train_config.get('max_grad_norm', None):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_config['max_grad_norm']
                    )
                
                optimizer.step()
                optimizer.zero_grad()
                
                # Update progress bar
                if rank == 0:
                    progress_bar.set_postfix({
                        'loss': loss.item() * train_config['grad_acc_steps'],
                        'lr': train_config['learning_rate'],
                        'global_step': global_step
                    })
                    progress_bar.update(train_config['grad_acc_steps'])
                
                global_step += 1
                
                # Save checkpoint
                if rank == 0 and global_step % train_config['save_steps'] == 0:
                    checkpoint_path = os.path.join(
                        train_config['checkpoint_dir'],
                        f"checkpoint_step_{global_step}.pt"
                    )
                    torch.save({
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
                    print(f"\nSaved checkpoint to {checkpoint_path}")
    
    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description='Distributed training for LLM')
    parser.add_argument('--world_size', type=int, default=None,
                      help='Number of GPUs to use (default: all available)')
    args = parser.parse_args()
    
    # Model configuration
    model_config = {
        'vocab_size': 50304,  # GPT-NeoX tokenizer vocab size
        'n_positions': 32,
        'max_seq_len': 32,
        'n_embd': 32,
        'n_layer': 2,
        'n_head': 2,
        'dropout': 0.1
    }
    
    # Training configuration
    train_config = {
        'batch_size': 4,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'max_epochs': 10,
        'checkpoint_dir': 'checkpoints',
        'log_every': 10,
        'grad_acc_steps': 16,
        'save_steps': 1000
    }
    
    # Get world size
    if args.world_size is None:
        world_size = torch.cuda.device_count()
    else:
        world_size = min(args.world_size, torch.cuda.device_count())
    
    if world_size < 2:
        print("Need at least 2 GPUs for distributed training")
        return
    
    # Launch processes
    mp.spawn(
        train,
        args=(world_size, model_config, train_config),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main() 
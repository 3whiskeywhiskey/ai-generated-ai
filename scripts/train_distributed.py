import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
import argparse
from contextlib import nullcontext
from torch.amp import autocast, GradScaler
from datetime import datetime
import signal
import atexit

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.llm_project.model.model import GPTModel
from src.llm_project.data.dataset import create_dataloaders, show_dataset_examples
from src.llm_project.model.parallel_utils import (
    setup_distributed, cleanup_distributed, print_gpu_topology, ResumeDistributedSampler
)
from src.llm_project.training.trainer import (
    clear_cuda_cache, log_gpu_memory, track_memory
)
from src.llm_project.model.checkpoint import (
    find_latest_checkpoint, save_checkpoint, load_checkpoint
)

def cleanup_handler(signum, frame):
    """Signal handler for cleanup."""
    print(f"\nReceived signal {signum}, cleaning up...")
    cleanup_distributed()
    sys.exit(0)

def train(rank, world_size, model_config, train_config, use_wandb=True):
    """Main training function."""
    # Set up signal handlers for cleanup
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    atexit.register(cleanup_distributed)
    
    try:
        setup_distributed(rank, world_size)
        torch.manual_seed(42 + rank)
        
        if rank == 0:
            os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
        
        device = f'cuda:{rank}'
        torch.cuda.set_device(device)
        clear_cuda_cache(rank)
        
        scaler = GradScaler()
        
        try:
            train_loader, val_loader, train_sampler = create_dataloaders(
                batch_size=train_config['batch_size'],
                max_length=model_config['max_seq_len']
            )
            
            if rank == 0:
                show_dataset_examples(
                    train_loader.dataset.dataset,
                    train_loader.dataset.tokenizer
                )
            
            with track_memory(rank, "Model Creation"):
                model = GPTModel(**model_config)
                model = model.to(device)
                model.gradient_checkpointing = True
            
            if rank == 0:
                total_params = sum(p.numel() for p in model.parameters())
                print(f"\nModel size: {total_params:,} parameters")
                log_gpu_memory(rank, "After model creation")
                
                if use_wandb:
                    wandb.init(
                        project="openwebtext-500m",
                        config={
                            "model_config": model_config,
                            "train_config": train_config,
                            "world_size": world_size,
                            "effective_batch_size": train_config['batch_size'] * world_size * train_config['grad_acc_steps'],
                            "total_params": total_params,
                        },
                        name=f"distributed_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        resume=True
                    )
            
            model = DDP(
                model,
                device_ids=[rank],
                find_unused_parameters=False,
                broadcast_buffers=False
            )
            model._set_static_graph()
            
            optimizer = AdamW(
                model.parameters(),
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay']
            )
            
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=len(train_loader) * train_config['max_epochs'] // train_config['grad_acc_steps']
            )
            
            steps_per_epoch = len(train_loader) // train_config['grad_acc_steps']
            start_epoch = 0
            global_step = 0
            steps_this_epoch = 0
            best_val_loss = float('inf')
            
            checkpoint_path = find_latest_checkpoint(train_config['checkpoint_dir'])
            if checkpoint_path:
                start_epoch, global_step = load_checkpoint(
                    model, optimizer, scheduler, checkpoint_path, rank, scaler
                )
                steps_this_epoch = (global_step % steps_per_epoch) * train_config['grad_acc_steps']
            
            if rank == 0:
                print(f"\nTraining Configuration:")
                print(f"Learning Rate: {train_config['learning_rate']}")
                print(f"Weight Decay: {train_config['weight_decay']}")
                print(f"Batch Size per GPU: {train_config['batch_size']}")
                print(f"Effective Batch Size: {train_config['batch_size'] * world_size * train_config['grad_acc_steps']}")
                print(f"Gradient Accumulation Steps: {train_config['grad_acc_steps']}")
                print(f"Max Sequence Length: {model_config['max_seq_len']}")
                print(f"Starting from epoch {start_epoch}, step {global_step}")
            
            # Training loop
            for epoch in range(start_epoch, train_config['max_epochs']):
                if rank == 0:
                    print(f"\nProcess {rank}: Starting epoch {epoch}")
                    log_gpu_memory(rank, "Start of epoch")
                
                train_sampler.set_epoch(epoch)
                train_sampler.set_start_index(steps_this_epoch if steps_this_epoch > 0 else 0)
                
                if rank == 0:
                    progress = tqdm(
                        total=len(train_loader),
                        initial=steps_this_epoch,
                        desc=f"Training Epoch {epoch}",
                        disable=rank != 0
                    )
                
                model.train()
                running_loss = 0
                optimizer.zero_grad()
                clear_cuda_cache(rank)
                
                for batch_idx, batch in enumerate(train_loader, start=steps_this_epoch):
                    memory_tracker = track_memory(rank, f"Training Step {global_step}") if batch_idx == 0 else nullcontext()
                    with memory_tracker:
                        input_ids = batch['input_ids'].to(device)
                        labels = batch['labels'].to(device)
                        
                        with autocast(device_type='cuda', dtype=torch.float16):
                            logits, loss = model(input_ids, labels=labels)
                            loss = loss / train_config['grad_acc_steps']
                        
                        scaler.scale(loss).backward()
                        running_loss += loss.item() * train_config['grad_acc_steps']
                        
                        acc_step = (batch_idx + 1) % train_config['grad_acc_steps']
                        
                        if acc_step == 0:
                            scaler.unscale_(optimizer)
                            
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                train_config['max_grad_norm']
                            )
                            
                            if torch.isfinite(grad_norm):
                                scaler.step(optimizer)
                                scaler.update()
                                scheduler.step()
                            else:
                                if rank == 0:
                                    print(f"\nSkipping step due to NaN gradients (grad_norm={grad_norm})")
                                scaler._scale = scaler._scale / 2.0
                                scaler.update()
                            
                            optimizer.zero_grad(set_to_none=True)
                            
                            if rank == 0:
                                avg_loss = running_loss / train_config['grad_acc_steps']
                                lr = scheduler.get_last_lr()[0]
                                progress.set_description(
                                    f"Training Epoch {epoch} [{batch_idx}/{len(train_loader)} "
                                    f"loss={avg_loss:.3f}, lr={lr:.6f}, "
                                    f"grad_norm={grad_norm:.3f}, scale={scaler.get_scale():.1f}]"
                                )
                                progress.update(train_config['grad_acc_steps'])
                                
                                if use_wandb:
                                    wandb.log({
                                        'train/loss': avg_loss,
                                        'train/learning_rate': lr,
                                        'train/gradient_norm': grad_norm,
                                        'train/scale': scaler.get_scale()
                                    }, step=global_step)
                            
                            running_loss = 0
                            global_step += 1
                            
                            if global_step % train_config['checkpoint_interval'] == 0 and rank == 0:
                                save_checkpoint(
                                    model=model.module,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    epoch=epoch,
                                    step=global_step,
                                    loss=avg_loss,
                                    checkpoint_dir=train_config['checkpoint_dir'],
                                    is_main_process=True,
                                    scaler=scaler
                                )
                
                steps_this_epoch = 0
                if rank == 0:
                    progress.close()
                
                avg_train_loss = running_loss / len(train_loader)
                
                model.eval()
                total_val_loss = 0
                
                with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                    for val_batch in val_loader:
                        val_input_ids = val_batch['input_ids'].to(device)
                        val_labels = val_batch['labels'].to(device)
                        
                        val_outputs = model(val_input_ids, labels=val_labels)
                        val_loss = val_outputs.loss
                        total_val_loss += val_loss.item()
                
                avg_val_loss = total_val_loss / len(val_loader)
                
                if rank == 0:
                    print(f"\nEpoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
                    log_gpu_memory(rank, f"End of epoch {epoch}")
                    
                    save_checkpoint(
                        model=model.module,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=global_step,
                        loss=avg_train_loss,
                        checkpoint_dir=train_config['checkpoint_dir'],
                        is_main_process=True,
                        is_epoch=True,
                        scaler=scaler
                    )
                    
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
                            is_best=True,
                            scaler=scaler
                        )
            
        except Exception as e:
            print(f"Rank {rank} encountered error: {str(e)}")
            raise e
        finally:
            if rank == 0 and use_wandb:
                wandb.finish()
    
    finally:
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="Distributed training for GPT model")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    model_config = {
        'vocab_size': 50304,
        'max_seq_len': 1024,
        'n_layer': 24,
        'n_head': 24,
        'n_embd': 1536,
        'dropout': 0.1,
        'gradient_checkpointing': True
    }
    
    train_config = {
        'batch_size': 1,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'max_epochs': 10,
        'grad_acc_steps': 32,
        'max_grad_norm': 5.0,
        'warmup_steps': 2000,
        'checkpoint_dir': 'checkpoints',
        'checkpoint_interval': 16,
        'log_every': 10,
        'distributed': True,
        'num_gpus': 4
    }
    
    print("\nEffective batch size:", train_config['batch_size'] * 4 * train_config['grad_acc_steps'])
    
    try:
        world_size = torch.cuda.device_count()
        mp.spawn(
            train,
            args=(world_size, model_config, train_config, not args.no_wandb),
            nprocs=world_size,
            join=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main() 
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import wandb
import time
from typing import Optional
from transformers import get_linear_schedule_with_warmup

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        train_sampler: Optional[torch.utils.data.Sampler],
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_epochs: int = 10,
        grad_acc_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        log_every: int = 10,
        use_wandb: bool = True,
        warmup_steps: int = 0,
        total_steps: Optional[int] = None,
        local_rank: int = 0,
        distributed: bool = False
    ):
        self.local_rank = local_rank
        self.distributed = distributed
        
        # Use local_rank if distributed, otherwise use 0
        self.rank = local_rank if distributed else 0
        self.world_size = dist.get_world_size() if distributed else 1
        
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            self.device = 'cpu'
            print(f"[Rank {self.rank}] Warning: No GPUs available, using CPU")
        else:
            # Assign GPU based on local rank, wrapping around if needed
            gpu_id = self.local_rank % num_gpus
            self.device = f'cuda:{gpu_id}'
            print(f"[Rank {self.rank}] Using GPU {gpu_id} out of {num_gpus} GPUs")
        
        # Move model to device
        self.model = model.to(self.device)
        if distributed:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[gpu_id] if torch.cuda.is_available() else None,
                output_device=gpu_id if torch.cuda.is_available() else None
            )
        
        # Setup data
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_sampler = train_sampler
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.grad_acc_steps = grad_acc_steps
        self.max_grad_norm = max_grad_norm
        
        # Calculate total steps if not provided
        if total_steps is None:
            total_steps = len(train_loader) * max_epochs // grad_acc_steps
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Logging and checkpointing
        self.checkpoint_dir = checkpoint_dir
        self.log_every = log_every
        self.use_wandb = use_wandb and self.rank == 0
        
        # Initialize step counters
        self.global_step = 0
        self.epoch = 0
        
        # Initialize timing metrics
        self.last_log_time = time.time()
        self.batch_times = []
        
    def _log_metrics(self, metrics: dict, step: int, prefix: str = ""):
        """Log metrics to console and wandb."""
        if self.rank == 0:  # Only log on main process
            # Add timing metrics
            current_time = time.time()
            batch_time = current_time - self.last_log_time
            self.batch_times.append(batch_time)
            if len(self.batch_times) > 100:  # Keep a moving window
                self.batch_times.pop(0)
            avg_batch_time = sum(self.batch_times) / len(self.batch_times)
            
            # Calculate throughput
            samples_per_second = self.train_loader.batch_size * self.world_size / avg_batch_time
            tokens_per_second = samples_per_second * self.train_loader.dataset.max_length
            
            # Add performance metrics
            metrics.update({
                f"{prefix}batch_time": batch_time,
                f"{prefix}samples_per_second": samples_per_second,
                f"{prefix}tokens_per_second": tokens_per_second,
            })
            
            # Log to console
            metrics_str = [f"{k}: {v:.4f}" for k, v in metrics.items()]
            #print(f"[Rank {self.rank}] Step {step} - " + " | ".join(metrics_str))
            
            # Log to wandb
            if self.use_wandb:
                wandb.log(metrics, step=step)
            
            self.last_log_time = current_time
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        
        if self.rank == 0:
            print(f"\nStarting epoch {epoch}")
        
        for batch_idx, batch in enumerate(self.train_loader):
            start_time = time.time()
            
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss / self.grad_acc_steps  # Access loss directly from ModelOutput
            
            # Backward pass
            loss.backward()
            
            # Update weights if gradient accumulation is done
            if (batch_idx + 1) % self.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()  # Step the scheduler
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * self.grad_acc_steps
            
            # Log progress
            if (batch_idx + 1) % self.log_every == 0:
                current_loss = total_loss / (batch_idx + 1)
                metrics = {
                    "train/loss": current_loss,
                    "train/epoch": epoch,
                    "train/progress": batch_idx / num_batches,
                    "train/learning_rate": self.scheduler.get_last_lr()[0],  # Get current learning rate
                }
                
                # Add GPU memory metrics
                if torch.cuda.is_available():
                    metrics.update({
                        "gpu/memory_allocated": torch.cuda.memory_allocated(self.device) / 1024**3,  # GB
                        "gpu/memory_cached": torch.cuda.memory_reserved(self.device) / 1024**3,  # GB
                    })
                
                self._log_metrics(metrics, self.global_step)
            
            self.global_step += 1
        
        epoch_loss = total_loss / num_batches
        if self.rank == 0:
            print(f"\nEpoch {epoch} complete - Average loss: {epoch_loss:.4f}")
        
        return epoch_loss
    
    def validate(self, epoch: int) -> float:
        """Run validation."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        if self.rank == 0:
            print(f"\nStarting validation for epoch {epoch}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, labels=labels)
                loss = outputs.loss  # Access loss directly from ModelOutput
                
                # Update metrics
                total_loss += loss.item()
                
                # Log progress
                if (batch_idx + 1) % self.log_every == 0:
                    metrics = {
                        "val/loss": total_loss / (batch_idx + 1),
                        "val/epoch": epoch,
                        "val/progress": batch_idx / num_batches,
                    }
                    self._log_metrics(metrics, self.global_step, prefix="val/")
        
        val_loss = total_loss / num_batches
        if self.rank == 0:
            print(f"\nValidation complete - Average loss: {val_loss:.4f}")
        
        return val_loss
    
    def train(self):
        """Train the model for the specified number of epochs."""
        if self.rank == 0:
            print("\nStarting training...")
        
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate(epoch)
            
            # Log epoch metrics
            if self.rank == 0:
                metrics = {
                    "epoch": epoch,
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": val_loss,
                }
                self._log_metrics(metrics, self.global_step, prefix="epoch/")
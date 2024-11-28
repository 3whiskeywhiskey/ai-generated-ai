import os
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        train_sampler=None,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_epochs=10,
        grad_acc_steps=1,
        max_grad_norm=1.0,
        checkpoint_dir="checkpoints",
        log_every=10,
        device=None,
        distributed=False,
        local_rank=0,
        use_wandb=True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_sampler = train_sampler
        self.grad_acc_steps = grad_acc_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.log_every = log_every
        self.max_epochs = max_epochs
        self.distributed = distributed
        self.local_rank = local_rank
        self.use_wandb = use_wandb
        
        # Set up device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Wrap model in DDP if distributed
        if self.distributed:
            self.model = DDP(self.model, device_ids=[local_rank])
        
        # Create optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Cosine learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.train_loader) * max_epochs // grad_acc_steps
        )
        
        # Create checkpoint directory
        if local_rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, epoch, step, loss):
        """Save model checkpoint."""
        if self.local_rank != 0:
            return
            
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss
        }
        
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}_step_{step}.pt')
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['step']
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        step = 0
        
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Training Epoch {epoch}",
            disable=self.local_rank != 0
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss / self.grad_acc_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if gradient accumulation is done
            if (batch_idx + 1) % self.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                step += 1
                
                # Log metrics
                if step % self.log_every == 0 and self.local_rank == 0 and self.use_wandb:
                    wandb.log({
                        "train/loss": loss.item() * self.grad_acc_steps,
                        "train/lr": self.scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/step": step
                    })
            
            total_loss += loss.item() * self.grad_acc_steps
            
            # Update progress bar
            if self.local_rank == 0:
                progress_bar.set_postfix({"loss": loss.item() * self.grad_acc_steps})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self, epoch):
        """Run validation."""
        self.model.eval()
        total_loss = 0
        
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Validation Epoch {epoch}",
            disable=self.local_rank != 0
        )
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            
            # Update progress bar
            if self.local_rank == 0:
                progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Log validation metrics
        if self.local_rank == 0 and self.use_wandb:
            wandb.log({
                "val/loss": avg_loss,
                "val/epoch": epoch
            })
        
        return avg_loss
    
    def train(self, start_epoch=0):
        """Main training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch, self.max_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Save checkpoint if validation loss improved
            if self.local_rank == 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(
                    epoch=epoch,
                    step=epoch * len(self.train_loader),
                    loss=val_loss
                )
            
            # Log epoch metrics
            if self.local_rank == 0 and self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss
                }) 
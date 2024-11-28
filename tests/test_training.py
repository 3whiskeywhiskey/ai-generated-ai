import os
import sys
import torch

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.llm_project.model.model import GPTModel
from src.llm_project.data.dataset import create_dataloaders
from src.llm_project.training.trainer import Trainer

def test_training_setup():
    """Test the training setup with a small model and dataset."""
    # Create a tiny model for testing
    model = GPTModel(
        vocab_size=50304,  # GPT-NeoX tokenizer vocab size
        max_seq_len=16,
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        dropout=0.1
    )
    
    # Create dataloaders with small batch size
    train_loader, val_loader, train_sampler = create_dataloaders(
        batch_size=4,
        max_length=16
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_epochs=1,
        grad_acc_steps=1,
        max_grad_norm=1.0,
        checkpoint_dir="test_checkpoints",
        log_every=1,
        use_wandb=False  # Disable wandb for testing
    )
    
    # Run one epoch
    train_loss = trainer.train_epoch(epoch=0)
    print(f"Training loss: {train_loss}")
    
    # Run validation
    val_loss = trainer.validate(epoch=0)
    print(f"Validation loss: {val_loss}")
    
    # Check that losses are reasonable
    assert not torch.isnan(torch.tensor(train_loss)), "Training loss is NaN"
    assert not torch.isnan(torch.tensor(val_loss)), "Validation loss is NaN"
    assert train_loss > 0, "Training loss should be positive"
    assert val_loss > 0, "Validation loss should be positive"

if __name__ == "__main__":
    test_training_setup() 
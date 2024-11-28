import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import torch.distributed as dist

class TinyStoriesDataset(Dataset):
    def __init__(self, split="train", max_length=16, cache_dir=None):
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b",  # Using a standard tokenizer
            cache_dir=cache_dir
        )
        # Set padding token to eos token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        self.dataset = load_dataset(
            "roneneldan/TinyStories",
            split=split,
            cache_dir=cache_dir
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get story text
        text = self.dataset[idx]["text"]
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.max_length + 1,  # +1 for labels shift
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Get input_ids and create shifted labels
        input_ids = tokens["input_ids"].squeeze(0)[:-1]
        labels = tokens["input_ids"].squeeze(0)[1:]
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }

def create_dataloaders(
    batch_size: int,
    max_length: int = 16,
    cache_dir: str = None,
    seed: int = 42
):
    """Create train and validation dataloaders."""
    # Set up datasets
    train_dataset = TinyStoriesDataset("train", max_length, cache_dir)
    val_dataset = TinyStoriesDataset("validation", max_length, cache_dir)
    
    # Get world size and rank for distributed training
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Create samplers for distributed training
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed
    ) if dist.is_initialized() else None
    
    val_sampler = torch.utils.data.DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if dist.is_initialized() else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler 
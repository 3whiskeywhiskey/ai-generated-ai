import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer
import torch.distributed as dist

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

def create_dataloaders(batch_size, max_length):
    """Create train and validation dataloaders."""
    # Create datasets
    train_dataset = TinyStoriesDataset(split="train", max_length=max_length)
    val_dataset = TinyStoriesDataset(split="validation", max_length=max_length)
    
    # Get world size and rank for distributed training
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Create samplers
    train_sampler = ResumeDistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler

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
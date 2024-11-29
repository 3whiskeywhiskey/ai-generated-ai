import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer
import torch.distributed as dist
import os
import time
from typing import Optional, Tuple
import json
import hashlib

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

class OpenWebTextDataset(Dataset):
    def __init__(self, split="train", max_length=1024, cache_dir=None, val_size: Optional[int] = None, seed: int = 42):
        """
        Args:
            split: Which split to load ("train" or "validation")
            max_length: Maximum sequence length
            cache_dir: Directory to cache the dataset
            val_size: Number of examples to use for validation
            seed: Random seed for validation split
        """
        self.max_length = max_length
        self.cache_dir = cache_dir or ".cache/huggingface/datasets"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load tokenizer first - we need this regardless of cache hit/miss
        self.tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        # Set padding token to eos token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Generate a unique cache key based on split parameters
        cache_params = {
            "split": split,
            "val_size": val_size,
            "seed": seed
        }
        cache_key = hashlib.md5(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()
        self.cache_path = os.path.join(self.cache_dir, f"openwebtext_{cache_key}.arrow")

        # Try to load from cache first
        dataset_loaded = False
        if os.path.exists(self.cache_path):
            try:
                print(f"Loading cached dataset from {self.cache_path}...")
                self.dataset = HFDataset.load_from_disk(self.cache_path)
                print(f"Loaded {len(self.dataset):,} examples from cache")
                dataset_loaded = True
            except Exception as e:
                print(f"Failed to load cache, regenerating: {e}")

        # If cache loading failed or cache doesn't exist, load from scratch
        if not dataset_loaded:
            try:
                # First load the full dataset to get total size
                full_dataset = load_dataset(
                    "openwebtext",
                    split="train",
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                total_size = len(full_dataset)
                print(f"Total dataset size: {total_size:,} examples")
                
                # Calculate validation size if not provided
                if val_size is None:
                    val_size = min(10000, int(total_size * 0.001))  # 0.1% or 10k examples, whichever is smaller
                
                # Determine dataset split
                if split == "validation":
                    actual_split = f"train[:{val_size}]"
                elif split == "train":
                    actual_split = f"train[{val_size}:]"
                else:
                    actual_split = split

                # Load the actual split
                print(f"Loading OpenWebText dataset (split={split}, actual_split={actual_split})...")
                self.dataset = load_dataset(
                    "openwebtext",
                    split=actual_split,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                print(f"Loaded {len(self.dataset):,} examples")
                
                # Cache the processed dataset
                print(f"Caching processed dataset to {self.cache_path}...")
                self.dataset.save_to_disk(self.cache_path)
                print("Dataset cached successfully")
                
            except Exception as e:
                print(f"Error loading dataset: {e}")
                raise e

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get text
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

def create_dataloaders(batch_size: int, max_length: int) -> Tuple[DataLoader, DataLoader, ResumeDistributedSampler]:
    """Create train and validation dataloaders."""
    # Get world size and rank for distributed training
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Use shared cache dir to avoid duplicating downloads
    cache_dir = ".cache/huggingface/datasets"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create datasets with fixed validation size
    val_size = 10000  # Use 10k examples for validation
    train_dataset = OpenWebTextDataset(
        split="train",
        max_length=max_length,
        cache_dir=cache_dir,
        val_size=val_size
    )
    
    val_dataset = OpenWebTextDataset(
        split="validation",
        max_length=max_length,
        cache_dir=cache_dir,
        val_size=val_size
    )
    
    # Create samplers
    train_sampler = ResumeDistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
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
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler

def show_dataset_examples(dataset, tokenizer=None, num_examples=2):
    """Show a few examples from the dataset with tokenization details."""
    # If tokenizer not provided, try to get it from dataset
    if tokenizer is None:
        if not hasattr(dataset, 'tokenizer'):
            raise AttributeError("Dataset must have a tokenizer attribute or tokenizer must be provided")
        tokenizer = dataset.tokenizer
    
    # Get max_length from dataset if available
    max_length = getattr(dataset, 'max_length', 1024)  # default to 1024 if not found
        
    print("\nDataset Examples:")
    print(f"Tokenizer: {tokenizer.name_or_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print(f"Padding token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"Max sequence length: {max_length}")
    
    # Handle both raw datasets and our custom dataset class
    if hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    
    for i in range(min(num_examples, len(dataset))):
        text = dataset[i]["text"]
        tokens = tokenizer(text)["input_ids"]
        
        print(f"\nExample {i+1}:")
        print(f"Text: {text[:200]}...")
        print(f"Length: {len(tokens)} tokens")
        print(f"First 5 tokens: {tokens[:5]} -> {tokenizer.decode(tokens[:5])}")
        print(f"Last 5 tokens: {tokens[-5:]} -> {tokenizer.decode(tokens[-5:])}")
        
        model_tokens = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"][0]
        
        print(f"Model input length (with padding): {len(model_tokens)}")
        print(f"Last 5 model tokens: {model_tokens[-5:].tolist()} -> {tokenizer.decode(model_tokens[-5:])}") 
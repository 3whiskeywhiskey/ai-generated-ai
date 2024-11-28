import torch
import torch.distributed as dist
from typing import Optional
import math
import torch.nn as nn

def initialize_parallel_env():
    """Initialize the distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
def get_parallel_ranks():
    """Get world size and rank for distributed training."""
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
    return world_size, rank

def split_tensor_along_last_dim(tensor: torch.Tensor, num_partitions: int):
    """Split a tensor along its last dimension."""
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    return torch.split(tensor, last_dim_size, dim=last_dim)

class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.
    The linear layer is divided along the output dimension."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        # Initialize process group if not already done
        if not dist.is_initialized():
            self.world_size = 1
            self.rank = 0
        else:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            
        # Split the output features across GPUs
        self.output_per_rank = out_features // self.world_size
        assert out_features % self.world_size == 0, \
            f"Output features ({out_features}) must be divisible by world size ({self.world_size})"
        
        # Create local linear layer
        self.linear = nn.Linear(in_features, self.output_per_rank, bias=bias)
        
    def forward(self, x):
        # Local forward pass
        local_out = self.linear(x)  # [*, local_dim]
        
        if self.world_size == 1:
            return local_out
            
        # Gather outputs from all ranks
        gather_list = [torch.zeros_like(local_out) for _ in range(self.world_size)]
        dist.all_gather(gather_list, local_out)
        
        # Concatenate along feature dimension
        output = torch.cat(gather_list, dim=-1)
        return output

class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.
    The linear layer is divided along the input dimension."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        # Initialize process group if not already done
        if not dist.is_initialized():
            self.world_size = 1
            self.rank = 0
        else:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            
        # Split the input features across GPUs
        self.input_per_rank = in_features // self.world_size
        assert in_features % self.world_size == 0, \
            f"Input features ({in_features}) must be divisible by world size ({self.world_size})"
        
        # Create local linear layer
        self.linear = nn.Linear(self.input_per_rank, out_features, bias=bias and self.rank == 0)
        
    def forward(self, x):
        # Split input along feature dimension
        input_list = list(x.chunk(self.world_size, dim=-1))
        local_input = input_list[self.rank]
        
        # Local forward pass
        local_out = self.linear(local_input)
        
        if self.world_size == 1:
            return local_out
            
        # All-reduce across GPUs
        dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
        return local_out

class ParallelLinear(nn.Module):
    """Linear layer with model parallelism."""
    def __init__(self, in_features, out_features, gather_output=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        
        # Initialize process group if not already done
        if not dist.is_initialized():
            self.world_size = 1
            self.rank = 0
        else:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            
        # Split the output features across GPUs
        self.output_per_rank = out_features // self.world_size
        assert out_features % self.world_size == 0, \
            f"Output features ({out_features}) must be divisible by world size ({self.world_size})"
        
        # Create local linear layer
        self.linear = nn.Linear(in_features, self.output_per_rank, bias=True)
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1) if len(x.shape) > 2 else 1
        
        # Reshape input if needed
        if len(x.shape) > 2:
            x = x.reshape(-1, self.in_features)
        
        # Local forward pass
        local_out = self.linear(x)  # [batch*seq, local_dim]
        
        if not self.gather_output or self.world_size == 1:
            # Reshape back if needed
            if seq_len > 1:
                local_out = local_out.reshape(batch_size, seq_len, -1)
            return local_out
        
        # For multi-GPU case, use all_gather
        if self.world_size > 1:
            # Create list of tensors for gathering
            gather_list = [torch.zeros_like(local_out) for _ in range(self.world_size)]
            
            # All-gather operation
            dist.all_gather(gather_list, local_out)
            
            # Concatenate along feature dimension
            output = torch.cat(gather_list, dim=-1)
            
            # Reshape back if needed
            if seq_len > 1:
                output = output.reshape(batch_size, seq_len, -1)
            
            return output
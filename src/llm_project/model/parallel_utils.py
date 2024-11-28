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

class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism."""
    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.world_size, self.rank = get_parallel_ranks()
        
        # Split output size across GPUs
        self.output_size_per_partition = output_size // self.world_size
        assert output_size % self.world_size == 0, \
            f"Output size {output_size} must be divisible by world size {self.world_size}"
        
        # Create local linear layer
        self.linear = torch.nn.Linear(input_size, self.output_size_per_partition, bias=bias)
        
        # Initialize with different seeds per rank
        if dist.is_initialized():
            torch.manual_seed(42 + self.rank)
            self.linear.weight.data.normal_(mean=0.0, std=0.02)
            if bias:
                self.linear.bias.data.zero_()
        
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Local computation
        local_output = self.linear(input_)
        
        if dist.is_initialized():
            # Gather outputs from all GPUs
            output_list = [torch.empty_like(local_output) for _ in range(self.world_size)]
            dist.all_gather(output_list, local_output)
            output = torch.cat(output_list, dim=-1)
        else:
            output = local_output
            
        return output

class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism."""
    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.output_size = output_size
        self.world_size, self.rank = get_parallel_ranks()
        
        # Split input size across GPUs
        self.input_size_per_partition = input_size // self.world_size
        assert input_size % self.world_size == 0, \
            f"Input size {input_size} must be divisible by world size {self.world_size}"
        
        # Create local linear layer
        self.linear = torch.nn.Linear(self.input_size_per_partition, output_size, bias=bias and self.rank == 0)
        
        # Initialize with different seeds per rank
        if dist.is_initialized():
            torch.manual_seed(42 + self.rank)
            self.linear.weight.data.normal_(mean=0.0, std=0.02)
            if bias and self.rank == 0:
                self.linear.bias.data.zero_()
        
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if dist.is_initialized():
            # Split input across GPUs
            input_list = input_.chunk(self.world_size, dim=-1)
            local_input = input_list[self.rank]
        else:
            local_input = input_
            
        # Local computation
        local_output = self.linear(local_input)
        
        if dist.is_initialized():
            # Reduce outputs across GPUs
            dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
            
        return local_output

class ParallelLinear(nn.Module):
    """Linear layer with model parallelism."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.world_size, self.rank = get_parallel_ranks()
        
        # Split output features across GPUs
        self.out_features_per_rank = out_features // self.world_size
        self.out_features_start = self.rank * self.out_features_per_rank
        self.out_features_end = (self.rank + 1) * self.out_features_per_rank
        
        # Create local linear layer
        self.linear = nn.Linear(
            in_features,
            self.out_features_per_rank,
            bias=bias
        )
        
    def forward(self, x):
        # Get local output
        local_out = self.linear(x)
        
        if dist.is_initialized():
            # Gather outputs from all GPUs
            output_list = [torch.empty_like(local_out) for _ in range(self.world_size)]
            dist.all_gather(output_list, local_out)
            output = torch.cat(output_list, dim=-1)
            
            # Make sure output requires grad
            if self.training:
                output.requires_grad_(True)
        else:
            output = local_out
            
        return output
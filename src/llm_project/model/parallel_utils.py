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
        self.input_size = input_size
        self.output_size = output_size
        self.world_size, self.rank = get_parallel_ranks()
        
        # Split input size across GPUs
        self.input_size_per_partition = input_size // self.world_size
        assert input_size % self.world_size == 0, \
            f"Input size {input_size} must be divisible by world size {self.world_size}"
        
        # Create local linear layer - note transposed dimensions
        self.linear = torch.nn.Linear(self.input_size_per_partition, output_size, bias=bias and self.rank == 0)
        
        # Initialize with different seeds per rank
        if dist.is_initialized():
            torch.manual_seed(42 + self.rank)
            self.linear.weight.data.normal_(mean=0.0, std=0.02)
            if bias and self.rank == 0:
                self.linear.bias.data.zero_()
        
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, seq_len, input_size_per_partition]
        batch_size, seq_len, _ = input_.size()
        
        # Verify input size
        assert input_.size(-1) == self.input_size_per_partition, \
            f"Expected input size {self.input_size_per_partition}, got {input_.size(-1)}"
        
        # Reshape for linear layer
        input_2d = input_.reshape(-1, self.input_size_per_partition)
        
        # Local computation
        local_output = self.linear(input_2d)  # [-1, output_size]
        
        if dist.is_initialized():
            # All-reduce across GPUs
            dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
        
        # Restore batch dimension
        output = local_output.reshape(batch_size, seq_len, self.output_size)
        
        return output

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
        assert out_features % self.world_size == 0, f"Output features ({out_features}) must be divisible by world size ({self.world_size})"
        
        # Create local linear layer
        self.linear = nn.Linear(in_features, self.output_per_rank, bias=True)
        
    def forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()
        orig_shape = x.shape
        
        # Reshape input if needed
        if len(orig_shape) > 2:
            x = x.view(-1, self.in_features)
            
        # Local forward pass
        local_out = self.linear(x)  # [*, local_dim]
        
        if not self.gather_output or self.world_size == 1:
            # Reshape back if needed
            if len(orig_shape) > 2:
                local_out = local_out.view(*orig_shape[:-1], self.output_per_rank)
            return local_out
            
        # Parallel reduction for multi-GPU case
        if self.world_size > 1:
            # Allocate full output tensor
            full_output = torch.zeros(
                (*local_out.shape[:-1], self.out_features),
                dtype=local_out.dtype,
                device=local_out.device
            )
            
            # Place local output in the correct slice
            start_idx = self.rank * self.output_per_rank
            end_idx = start_idx + self.output_per_rank
            full_output[..., start_idx:end_idx] = local_out
            
            # All-reduce to combine results
            dist.all_reduce(full_output, op=dist.ReduceOp.SUM)
            
            # Reshape if needed
            if len(orig_shape) > 2:
                full_output = full_output.view(*orig_shape[:-1], self.out_features)
                
            return full_output
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism."""
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
        local_out = self.linear(x)
        
        if self.world_size == 1:
            return local_out
            
        # Create list of tensors for gathering
        gather_list = [torch.empty_like(local_out) for _ in range(self.world_size)]
        
        # All-gather operation
        dist.all_gather(gather_list, local_out)
        
        # Concatenate along feature dimension
        return torch.cat(gather_list, dim=-1)

class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism."""
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
        input_chunks = x.chunk(self.world_size, dim=-1)
        local_input = input_chunks[self.rank]
        
        # Local forward pass
        local_out = self.linear(local_input)
        
        if self.world_size == 1:
            return local_out
            
        # All-reduce to combine results
        dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
        return local_out

class ParallelLinear(nn.Module):
    """Linear layer with model parallelism."""
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
        local_out = self.linear(x)
        
        if self.world_size == 1:
            return local_out
            
        # Create output tensor
        output_shape = list(local_out.shape)
        output_shape[-1] = output_shape[-1] * self.world_size
        output = local_out.new_zeros(output_shape)
        
        # Place local output in the correct slice
        start_idx = self.rank * self.output_per_rank
        end_idx = start_idx + self.output_per_rank
        output[..., start_idx:end_idx] = local_out
        
        # All-reduce to combine results
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output
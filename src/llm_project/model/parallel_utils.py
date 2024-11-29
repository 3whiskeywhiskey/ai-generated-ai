import torch
import torch.distributed as dist
from typing import Optional
import math
import torch.nn as nn
import os
import pynvml
from torch.utils.data.distributed import DistributedSampler

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
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        start_idx = self.start_index * self.num_replicas + self.rank
        indices = indices[start_idx:self.total_size:self.num_replicas]
        
        return iter(indices)

def print_gpu_topology():
    """Print detailed GPU topology information."""
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        print("\nGPU Topology:")
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            print(f"\nGPU {i}: {name}")
            
            # Check NVLink connections
            try:
                for j in range(device_count):
                    if i != j:
                        nvlink_status = pynvml.nvmlDeviceGetNvLinkState(handle, j)
                        if nvlink_status == pynvml.NVML_NVLINK_STATUS_NOT_SUPPORTED:
                            print(f"  -> GPU {j}: No NVLink support")
                        else:
                            nvlink_version = pynvml.nvmlDeviceGetNvLinkVersion(handle, j)
                            print(f"  -> GPU {j}: NVLink v{nvlink_version}")
            except pynvml.NVMLError:
                print(f"  NVLink information not available")
            
            # Check P2P capabilities
            for j in range(device_count):
                if i != j:
                    can_access = torch.cuda.can_device_access_peer(i, j)
                    print(f"  -> GPU {j}: P2P {'Enabled' if can_access else 'Disabled'}")
        
        print("\nPCI Topology:")
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
            print(f"GPU {i}: Bus ID {pci_info.busId.decode('utf-8')}")
            
    except Exception as e:
        print(f"Could not detect GPU topology: {str(e)}")
    print()

def setup_distributed(rank, world_size):
    """Initialize distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_P2P_DISABLE'] = '0'
    os.environ['NCCL_P2P_LEVEL'] = '5'
    os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET'
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
    
    if rank == 0:
        print("\nDistributed Configuration:")
        print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
        print(f"WORLD_SIZE: {world_size}")
        print(f"RANK: {rank}")
        print("\nNCCL Configuration:")
        for k, v in sorted(os.environ.items()):
            if k.startswith('NCCL_'):
                print(f"{k}: {v}")
        print()
    
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:29500",
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)
    if rank == 0:
        print(f"Process {rank}: CUDA device set to {torch.cuda.current_device()}")
        print(f"Process group initialized successfully!")

def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
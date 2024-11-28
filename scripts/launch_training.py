import os
import deepspeed
import torch
from src.model.model import LLM
from src.model.config import ModelConfig
from src.model.parallel_utils import initialize_parallel_env

def main():
    # Initialize distributed environment
    initialize_parallel_env()
    
    # Create model config
    model_config = ModelConfig()
    
    # Create model
    model = LLM(model_config)
    
    # Initialize DeepSpeed
    ds_config = "configs/deepspeed_config.json"
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config
    )
    
    # Set NCCL parameters for NVLink
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "0"
    os.environ["NCCL_NET_GDR_LEVEL"] = "5"

if __name__ == "__main__":
    main() 
#!/bin/bash

# Set NCCL environment variables
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# Launch distributed test across 4 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    tests/test_model.py 
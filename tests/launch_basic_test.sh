#!/bin/bash

# Set NCCL environment variables
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_DEBUG=INFO

# Add the project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Launch distributed test across 4 GPUs
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    tests/test_model_basic.py 
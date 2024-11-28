#!/bin/bash

# Default values
NUM_GPUS=4

# Model configuration
MODEL_ARGS="
    --n_layers 12 \
    --n_heads 12 \
    --d_model 768 \
    --d_ff 3072 \
    --dropout 0.1 \
    --max_length 128"

# Training configuration
TRAIN_ARGS="
    --batch_size 32 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --max_epochs 10 \
    --grad_acc_steps 4 \
    --max_grad_norm 1.0 \
    --warmup_steps 2000"

# System configuration
SYS_ARGS="
    --distributed \
    --num_gpus $NUM_GPUS \
    --checkpoint_dir checkpoints \
    --log_every 10"

# Combine all arguments
ALL_ARGS="$MODEL_ARGS $TRAIN_ARGS $SYS_ARGS"

# Clean up any stale coordination files
rm -f /tmp/torch_distributed_*

# Set environment variables
export NCCL_DEBUG=WARN
#export NCCL_DEBUG_SUBSYS=ALL

# Simplified NCCL configuration
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# Use file-based coordination
export TORCHELASTIC_USE_AGENT=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=INFO

# Coordination settings
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Thread settings
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Launch training
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=localhost \
    --master_port=29500 \
    --nnodes=1 \
    --node_rank=0 \
    scripts/train.py $ALL_ARGS 

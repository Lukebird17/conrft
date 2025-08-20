#!/bin/bash
# Simple conversion script that doesn't require full environment setup
conda run -n conrft2 python convert_demos_simple.py \
    --input_file=/home/ziyu/Project/VLA-RL/hil-serl/examples/demo_data/twist_20_demos_2025-08-09_15-41-07.pkl \
    --reward_scale=1.0 \
    --reward_bias=0.0 \
    --discount=0.99

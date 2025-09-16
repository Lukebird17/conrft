export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_PLATFORMS=cuda
export WANDB_API_KEY=afb9ea51b9652834c2840be8f209de8b38bebc34 && \
export HF_ENDPOINT=https://hf-mirror.com && \
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=/Disk2/lhl/anaconda3/envs/conrft/lib:$LD_LIBRARY_PATH

python ../../train_conrft_octo.py "$@" \
    --exp_name=pp \
    --checkpoint_path=/home/hongliang/Project/VLA-RL/conrft/examples/experiments/pp/ckpt\
    --actor \
    --ip=192.168.0.119 \
    --port=5555 \
    # --eval_checkpoint_step=20000 \
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_PLATFORMS=cuda
export WANDB_API_KEY=afb9ea51b9652834c2840be8f209de8b38bebc34 && \
export HF_ENDPOINT=https://hf-mirror.com && \
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
nvcc -V

python ../../train_conrft_octo.py "$@" \
    --exp_name=pp \
    --checkpoint_path=/home/ziyu/Project/VLA_RL/conrft/examples/experiments/pp/ckpt \
    --q_weight=1.0 \
    --bc_weight=0.1 \
    --demo_path=/home/ziyu/Project/VLA_RL/conrft/examples/demo_data/pp_30_demos_2025-09-15_16-49-43.pkl \
    --ip=192.168.0.119 \
    --host=0.0.0.0 \
    --port=5556 \
    --debug=False \
    --learner \



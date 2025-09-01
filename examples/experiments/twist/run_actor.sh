export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export WANDB_API_KEY=afb9ea51b9652834c2840be8f209de8b38bebc34 && \

python ../../train_conrft_octo.py "$@" \
    --exp_name=twist \
    --checkpoint_path=../../experiments/twist/ckpt \
    --actor \
    --ip=192.168.1.23 \
    --port=5555 \
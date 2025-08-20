export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export WANDB_API_KEY=afb9ea51b9652834c2840be8f209de8b38bebc34 && \

python ../../train_rlpd.py "$@" \
    --exp_name=pin \
    --checkpoint_path=../../experiments/pin/ckpt \
    --actor \
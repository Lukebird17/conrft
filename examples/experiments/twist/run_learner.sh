export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export WANDB_API_KEY=afb9ea51b9652834c2840be8f209de8b38bebc34 && \

name=twist

python ../../train_rlpd.py "$@" \
    --exp_name=twist \
    --checkpoint_path=../../experiments/twist/ckpt \
    --demo_path=../../demo_data/twist_20_demos_2025-08-09_15-41-07.pkl \
    --learner \

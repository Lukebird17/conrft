export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8 && \
export PYTHONPATH=/home/ziyu/Project/VLA-RL/conrft/serl_robot_infra:$PYTHONPATH && \
export HF_ENDPOINT=https://hf-mirror.com && \
export WANDB_API_KEY=afb9ea51b9652834c2840be8f209de8b38bebc34 && \
source ~/.bashrc   

python ../../train_conrft_octo.py "$@" \
    --exp_name=pp \
    --checkpoint_path=/home/hongliang/Project/VLA-RL/conrft/examples/experiments/pp/ckpt\
    --q_weight=0.1 \
    --bc_weight=1.0 \
    --demo_path=/home/hongliang/Project/VLA-RL/conrft/examples/demo_data/demo-path-to-be-determined.pkl \
    --pretrain_steps=20000 \
    --debug=False \
    --learner \


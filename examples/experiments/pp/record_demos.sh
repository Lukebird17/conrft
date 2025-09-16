export PYTHONPATH=/home/ziyu/Project/VLA_RL/conrft/serl_robot_infra:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
python ../../record_demos_octo.py \
    --exp_name pp \
    --successes_needed 30 \
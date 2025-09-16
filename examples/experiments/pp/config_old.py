import os
import sys

import jax
import numpy as np
import jax.numpy as jnp
from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    KeyboardIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func
import numpy as np
from experiments.config import DefaultTrainingConfig
from experiments.twist.wrapper import PPEnv, GripperPenaltyWrapper

from PIL import Image
import requests
import io
import base64
# from torchvision.transforms import ToPILImage
from transformers import AutoModelForCausalLM, AutoProcessor
import re
from io import BytesIO
import threading
import time

from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from omegaconf import DictConfig

class EnvConfig(DefaultEnvConfig):
    SERVER_URL: str = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "wrist": {
            "serial_number": "233722071807",
            "dim": (1280, 720),
            "exposure": 16500,
        },
        # "side": {
        #     "serial_number": "233522075695",
        #     "dim": (1280, 720),
        #     "exposure": 10500,
        # },
        "side": {
            "serial_number": "233522075695",
            "dim": (1280, 720),
            "exposure": 16500,
        },
        "side_policy_256": { #web
            "cam_id": 0,
            "dim": (1280, 720),
        },
        "side_classifier": {
            "cam_id": 0,
            "dim": (1280, 720),
        },
    }
    
    IMAGE_CROP = {  "wrist" : lambda img: img[5:714,290:1080],
                    "side" : lambda img: img[210:605,610:1040],
                    "side_policy_256" : lambda img: img[140:720,270:1150],
                    "side_classifier" : lambda img: img[269:397,670:798]}


    TARGET_POSE = np.array([0.6590301394462585,-0.04521015286445618,0.1423424333333969,0,0,np.pi])
    
    RESET_POSE = np.array([0.6960667700767517,-0.03699609637260437,0.21825205087661743,0,0,np.pi])

    ACTION_SCALE = np.array([0.012, 0.5, 1]) # keyboard

    RANDOM_RESET = True
    DISPLAY_IMAGE = True
    RANDOM_XY_RANGE = 0.01
    RANDOM_RZ_RANGE = 0.1

    DELTA = np.array([0.3, 0.1, 0.12, 1.1, 0, 0])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + DELTA
    ABS_POSE_LIMIT_LOW = TARGET_POSE - DELTA

    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.006,
        "translational_clip_y": 0.0059,
        "translational_clip_z": 0.0035,
        "translational_clip_neg_x": 0.005,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.0035,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.015,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.015,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0.0,
    }
    MAX_EPISODE_LENGTH = 200


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class TrainConfig(DefaultTrainingConfig):

    image_keys = ["wrist", "side", "side_policy_256"]
    classifier_keys = ["side_classifier"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    reward_neg = -0.05
    task_desc = "pick up and place"
    octo_path = "/home/hongliang/Project/VLA_RL/conrft/octo-small"


    def get_environment(self, fake_env=False, save_video=False, classifier=False, stack_obs_num=1):
        env = PPEnv(
            fake_env=fake_env, save_video=save_video, config=EnvConfig()
        )
        if not fake_env:
            env = KeyboardIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),  
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path="/home/hongliang/Project/VLA_RL/conrft/examples/classifier_ckpt/classifier_ckpt_pp",
            )

            def reward_func(obs):

                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                print("sigmoid",(sigmoid(classifier(obs))))

                return int(np.array(sigmoid(classifier(obs)) > 0.95))
            
            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env
import os
import jax
import numpy as np
import jax.numpy as jnp

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.task1_pick_banana.wrapper import PickBananaEnv, GripperPenaltyWrapper


class EnvConfig(DefaultEnvConfig):
    SERVER_URL: str = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "115222071051",
            "dim": (1280, 720),
            "exposure": 10500,
        },
        "side_policy_256": {
            "serial_number": "242422305075",
            "dim": (1280, 720),
            "exposure": 13000,
        },
        "side_classifier": {
            "serial_number": "242422305075", 
            "dim": (1280, 720),
            "exposure": 13000,
        },
        "demo": {
            "serial_number": "242422305075", 
            "dim": (1280, 720),
            "exposure": 13000,
        },
    }
    IMAGE_CROP = {"wrist_1": lambda img: img,
                  "side_policy_256": lambda img: img[250:-150, 400:-500],
                  "side_classifier": lambda img: img[390:-150, 420:-700],
                  "demo": lambda img: img[50:-150, 400:-400]}

    TARGET_POSE = np.array([0.33, -0.15, 0.20, np.pi, 0, 0])
    RESET_POSE = np.array([0.61, -0.17, 0.22, np.pi, 0, 0])
    ACTION_SCALE = np.array([0.08, 0.2, 1])
    RANDOM_RESET = True
    DISPLAY_IMAGE = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.03
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.3, 0.03, 0.02, 0.01, 0.01, 0.3])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.05, 0.05, 0.01, 0.01, 0.3])
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.008,
        "translational_clip_y": 0.005, 
        "translational_clip_z": 0.005,
        "translational_clip_neg_x": 0.008,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.005, 
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.02, 
        "rotational_Ki": 0,
    }  # for normal operation other than reset procedure
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
    }  # only for reset procedure
    MAX_EPISODE_LENGTH = 100
 

class TrainConfig(DefaultTrainingConfig):
    image_keys = ["side_policy_256", "wrist_1"]
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
    task_desc = "Put the yellow banana to the green plate"
    octo_path = "/root/online_rl/octo_model/octo-small"

    def get_environment(self, fake_env=False, save_video=False, classifier=False, stack_obs_num=1):
        env = PickBananaEnv( fake_env=fake_env, save_video=save_video, config=EnvConfig())
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)  # Converts rotation from quaternion to Euler angles in observations.
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)  # Restructures observations to the SERL/Octo standard format.
        """
        convert:
        {
            "state": {
                "tcp_pose": np.array([...]),      # 机器人姿态 (一个数组)
                "tcp_vel": np.array([...]),       # 机器人速度 (一个数组)
                "gripper_pose": np.array([...])   # 夹爪位置 (一个数组)
                "tcp_force": np.array([...]),     # 机器人力 (一个数组)
                "tcp_torque": np.array([...]),    # 机器人力矩 (一个数组)
            },
            "images": {
                "wrist_1": np.array([...]),       # 手腕摄像头图像 (一个图像数组)
                "side_policy_256": np.array([...]) # 侧面摄像头图像 (一个图像数组)
            }
        }
        to:
        {
            "state": np.array([...]),             # 一个被压平的、包含所有状态信息的长向量
            "wrist_1": np.array([...]),           # 手腕摄像头图像 (现在位于顶层)
            "side_policy_256": np.array([...])     # 侧面摄像头图像 (现在位于顶层)
        }
        """
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)  # Stacks consecutive observations to provide a history to the policy.
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                def sigmoid(x): return 1 / (1 + jnp.exp(-x))
                # Should open the gripper and pull up after putting the banana
                if sigmoid(classifier(obs)[0]) > 0.9 and env.curr_gripper_pos > 0.5 and env.currpos[2] > 0.16:  # sigmoid/gripper state/z
                    return 10.0
                else:
                    return self.reward_neg

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)  # Overrides reward calculation using a learned visual classifier.
        env = GripperPenaltyWrapper(env, penalty=-0.2)  # Adds a penalty for inefficient or incorrect gripper actions.
        return env

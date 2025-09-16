import os
import sys
from gymnasium import Env, spaces
import gymnasium as gym
from scipy.spatial.transform import Rotation as R


# Add reactive_diffusion_policy to Python path
rdp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "reactive_diffusion_policy"))
if rdp_path not in sys.path:
    sys.path.insert(0, rdp_path)

import jax
import numpy as np
import jax.numpy as jnp
# from franka_env.envs.wrappers import (
#     Quat2EulerWrapper,
#     MultiCameraBinaryRewardClassifierWrapper,
# )
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func
import numpy as np
from experiments.config import DefaultTrainingConfig
from experiments.pp.wrapper import PPEnv, GripperPenaltyWrapper

from PIL import Image
import requests
import io
import base64
# from torchvision.transforms import ToPILImage
from transformers import AutoModelForCausalLM, AutoProcessor
import re
from io import BytesIO
# from openai import OpenAI
import threading
import time

# from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from omegaconf import DictConfig

class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation
    
class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz = None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        # print("=============================")
        rew = self.compute_reward(obs)
        done = done or rew
        info['succeed'] = bool(rew)
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info

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
    # IMAGE_CROP = {"wrist": lambda img: img[50:-200, 200:-200], #250 400
                #   "side": lambda img: img[275:620, 135:365], #345 230
                #   "side_policy": lambda img: img[250:500, 350:650], #250 300
                #   "side_classifier": lambda img: img[270:398, 500:628]} #128, 128
    
    IMAGE_CROP = {  "wrist" : lambda img: img[5:714,290:1080],
                    "side" : lambda img: img[210:605,610:1040],
                    # "side_policy" : lambda img: img[130:635,230:915],
                    # "side_classifier" : lambda img: img[280:408,650:778]}
                    "side_policy_256" : lambda img: img[140:720,270:1150],
                    "side_classifier" : lambda img: img[269:397,670:798]}


    # TARGET_POSE = np.array([0.5882624983787537,-0.0555502250790596,0.04414592310786247,-0.045765915684659264,-0.011763434853535815,3.1374591797682516])
    TARGET_POSE = np.array([0.6590301394462585,-0.04521015286445618,0.1423424333333969,0,0,np.pi])
    # RESET_POSE = TARGET_POSE + np.array([0, 0.03, 0.05, 0, 0, 0])
    # RESET_POSE = np.array([0.42907944321632385,-0.05901959165930748,0.06817488372325897,-0.020562794658465222,-0.03109255683347323,3.1313006298638424])
    # RESET_POSE = np.array([0.6848835349082947,-0.06681141257286072,0.11950448155403137,0.006532105740336025,-0.013204312608933177,3.123098962678037])
    
    RESET_POSE = np.array([0.6960667700767517,-0.03699609637260437,0.21825205087661743,0,0,np.pi])
    # RESET_POSE = np.array([0.715574324131012,-0.04581369087100029,0.14184178411960602,0,0,np.pi])
    # RESET_POSE = np.array([0.6960667700767517,-0.03699609637260437,0.,2.212594154116232e-05,1.1625573961282498e-06,1.0,4.295046892366372e-06])
    # #target
    # [0.6590301394462585,-0.04521015286445618,0.1423424333333969,-6.615063284698408e-06,-3.8057482925069053e-06,1.0,1.328332564298762e-05]
    # #before_target
    # [0.6776853799819946,-0.04482879862189293,0.14285042881965637,2.8629054213524796e-05,-8.168333806679584e-06,1.0,-1.4746319720870815e-05]
    # #grasp_place
    # [0.856775164604187,-0.04361021891236305,0.08989010006189346,-4.311542397772428e-06,5.085386760583788e-07,1.0,-4.779576102009742e-06]


    ACTION_SCALE = np.array([0.012, 0.5, 1]) # keyboard
    # ACTION_SCALE = np.array([1, 1, 1]) # vr teleop
    RANDOM_RESET = True
    DISPLAY_IMAGE = True
    RANDOM_XY_RANGE = 0.01
    RANDOM_RZ_RANGE = 0.1
    # ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.06, 0.05, 0.1, 0.1, 0.3])
    # ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.01, 0.03, 0.1, 0.1, 0.3])
    DELTA = np.array([0.3, 0.1, 0.12, 1.1, 0, 0])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + DELTA
    ABS_POSE_LIMIT_LOW = TARGET_POSE - DELTA
    # ABS_POSE_LIMIT_HIGH = np.array([0.8,0.2,0.5,2,2,4])
    # ABS_POSE_LIMIT_LOW = np.array([0.2,-0.2, 0.02, -2, -2, -4])

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

def pil_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# 异步执行 Qwen 请求的函数
# def async_reward_func(obs: dict, result: dict) -> None:
#     """
#     异步请求 Qwen API 并将结果存入共享字典中
#     """
#     TASK_PROMPT = (
#         "请判断图中机械臂是否已经将USB成功插入端口。\n"
#         "如果完成，请仅返回一个表示完成概率的数值（0到1之间的小数），不要多余解释。"
#     )

#     # 初始化 OpenAI 客户端
#     # client = OpenAI(
#     #     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     # )

#     # 获取图像数据并转换为 base64
#     image = obs["side_classifier"]
#     if isinstance(image, np.ndarray):
#         if image.ndim == 4:
#             image = image[0]  # 如果是4D数组，选择第一个batch
#         image = Image.fromarray(image)
#     image_base64 = pil_image_to_base64(image)

#     try:
#         # 请求 Qwen API 获取模型响应
#         response = client.chat.completions.create(
#             model="qwen-vl-plus",
#             messages=[{
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": TASK_PROMPT},
#                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
#                 ]
#             }]
#         )
#         reply = response.choices[0].message.content.strip()
#         prob = float(reply)
#         prob = max(0.0, min(1.0, prob))  # 保证结果在 [0, 1] 范围内
#     except Exception as e:
#         print(f"[Qwen API Error] {e}")
#         prob = 0.5  # 出错时使用一个默认值

#     result['prob'] = prob  # 将计算的概率保存到共享字典中

# def reward_func(obs: dict) -> float:
#     """
#     异步计算并返回 reward（概率值）
#     """
#     result = {'prob': 0.0}  # 用于存储计算结果
#     # 启动新线程进行异步计算
#     thread = threading.Thread(target=async_reward_func, args=(obs, result))
#     thread.start()
    
#     # 等待线程执行完成
#     thread.join()
    
#     # 返回计算出的概率值
#     return result['prob']

class TrainConfig(DefaultTrainingConfig):
    # image_keys = ["left", "side_policy"]
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
    task_desc = "Twist the red cap off the blue bottle and lift the cap up"
    octo_path = "/home/ziyu/Project/VLA_RL/conrft/octo-small"
    cfg = DictConfig({
        "task": {
            "transforms": {
                "calibration_path": '/home/ziyu/Project/VLA_RL/reactive_diffusion_policy/data/calibration/single_arm'
            }
        }
    })

    # transforms = RealWorldTransforms(option=cfg.task.transforms)

    def get_environment(self, fake_env=False, save_video=False, classifier=False, stack_obs_num=1):
        env = PPEnv(
            fake_env=fake_env, save_video=save_video, config=EnvConfig()
        )
        if not fake_env:
            from franka_env.envs.wrappers import KeyboardIntervention
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
                checkpoint_path=os.path.abspath("classifier_ckpt/classifier_ckpt_pp"),
            )

            def reward_func(obs):
                # print(obs.keys())
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # print((sigmoid(classifier(obs)) > 0.7 and obs["state"][0, 0] > 0.4))
                # print(int([False]))
                print("sigmoid",(sigmoid(classifier(obs))))
                # print("state", obs["state"][0, 0])
                # print("obs",obs["state"])
                # print("obs",obs["state"][0, 4])
                # return int(np.array(sigmoid(classifier(obs)) > 0.95 and obs["state"][0, 4] > 0.04))
                return int(np.array(sigmoid(classifier(obs)) > 0.95))
                # finished pose: {"pose":[0.6655418872833252,-0.04682869836688042,0.1418590247631073,3.603585281575228e-05,1.9406897666662815e-07,-3.1414877518272055]}
            
            # TASK_PROMPT = (
            #     "请判断图中机械臂是否已经将USB成功插入端口。\n"
            #     "如果完成，请仅返回一个表示完成概率的数值（0到1之间的小数），不要多余解释。"
            # )
            # client = OpenAI(
            #     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            #     api_key=os.getenv("DASHSCOPE_API_KEY"),
            #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            # )
            # # reward_fn：输入为 obs 字典，输出为概率
            # def pil_image_to_base64(image: Image.Image) -> str:
            #     buffered = io.BytesIO()
            #     image.save(buffered, format="JPEG")
            #     return base64.b64encode(buffered.getvalue()).decode("utf-8")

            # # Reward function：输入 obs 字典，输出 float 概率
            # def reward_func(obs: dict) -> float:
            #     image = obs["side_classifier"]
            #     # print(obs.keys())
            #     # print(image.shape)
            #     # 如果是 numpy 数组，转为 PIL.Image
            #     if isinstance(image, np.ndarray):
            #         if image.ndim == 4:
            #         # 如果还剩 4D，就只取第一个batch
            #             image = image[0]
            #         image = Image.fromarray(image)
            #     image_base64 = pil_image_to_base64(image)
            #     start_time = time.time()
            #     try:
            #         response = client.chat.completions.create(
            #             model="qwen-vl-plus",
            #             messages=[
            #                 {
            #                     "role": "user",
            #                     "content": [
            #                         {"type": "text", "text": TASK_PROMPT},
            #                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            #                     ],
            #                 }
            #             ],
            #         )
            #         # 提取模型的文字回答
            #         reply = response.choices[0].message.content.strip()

            #         # 尝试从回复中解析概率数字
            #         prob = float(reply)
            #         prob = max(0.0, min(1.0, prob))  # 限制范围在 [0, 1]
            #     except Exception as e:
            #         print(f"[Qwen API Error] {e}")
            #         prob = 0.5  # fallback 值：模型无法响应时使用中性值
            #     print(prob)
            #     end_time = time.time()
            #     request_duration = end_time - start_time
            #     print(f"API请求的延迟时间: {request_duration:.2f}秒")
            #     return 0

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
            # print("env initialize succ")
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env
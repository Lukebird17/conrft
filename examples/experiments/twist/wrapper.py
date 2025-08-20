from typing import OrderedDict
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.web_capture import WebcamCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.utils.rotations import euler_2_quat
import numpy as np
import requests
import copy
import gymnasium as gym
import time
from franka_env.envs.franka_env import FrankaEnv

class USBEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            if cam_name == "side_classifier":
                self.cap["side_classifier"] = self.cap["side_policy_256"]
            elif cam_name == "side_policy_256":
                cap = VideoCapture(
                    WebcamCapture(name=cam_name, **kwargs)
                )
                self.cap[cam_name] = cap
            else:
                cap = VideoCapture(
                    RSCapture(name=cam_name, **kwargs)
                )
                self.cap[cam_name] = cap



    def reset_old(self, **kwargs):
        self._recover()
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.1)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        self._send_gripper_command(0.1)
        
        # Move above the target pose
        target = copy.deepcopy(self.currpos)
        target[2] = self.config.TARGET_POSE[2] + 0.1
        self.interpolate_move(target, timeout=0.5)
        time.sleep(0.5)
        self.interpolate_move(self.config.TARGET_POSE, timeout=0.5)
        time.sleep(0.5)
        self._send_gripper_command(0.1)

        self._update_currpos()
        reset_pose = copy.deepcopy(self.config.TARGET_POSE)
        reset_pose[1] += 0.02
        self.interpolate_move(reset_pose, timeout=0.5)

        obs, info = super().reset(**kwargs)
        self._send_gripper_command(0.1)
        time.sleep(1)
        self.success = False
        self._update_currpos()
        obs = self._get_obs()

        return obs, info

    def reset(self, **kwargs):
        self._recover()
        self._update_currpos()
        # self._send_pos_command(self.currpos)
        time.sleep(0.1)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)


        # currpose = self.currpos
        # reset_pose = currpose
        # reset_pose[0] += 0.05
        # self.interpolate_move(reset_pose, timeout=0.5)
        # time.sleep(0.5)
        # print("reset low point")
        # # reset_pos = np.array([0.856775164604187,-0.04361021891236305,0.08989010006189346,-4.311542397772428e-06,5.085386760583788e-07,1.0,-4.779576102009742e-06])
        # reset_pos = np.array([0.7395889163017273,-0.04663907364010811,0.0313747692108154,-4.311542397772428e-06,5.085386760583788e-07,1.0,-4.779576102009742e-06])
        # reset_pos = np.array([0.7479987740516663,-0.03063167631626129,0.16586297750473022,-0.00018006890604738146,2.0702129859273555e-06,1.0,2.430021959298756e-05])
        reset_pos = self._RESET_POSE
        # print("end reseting")
        # reset_pos[2] + 0.05
        # self.interpolate_move(reset_pos, timeout=0.5)
        # time.sleep(0.5)

        obs, info = super().reset(**kwargs)

        self.success = False
        self._send_gripper_command(0.05)
        self.interpolate_move(reset_pos, timeout=0.5)
        time.sleep(1)
        print("##################this is the reset function########################")
        self._send_gripper_command(0.1)
        time.sleep(1)
        self._send_gripper_command(0.05)
        time.sleep(5)
        self._update_currpos()
        obs = self._get_obs()

        return obs, info

    
    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        self._send_pos_command(goal)
        time.sleep(timeout)
        self._update_currpos()
    
    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)



class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.045) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.02
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info
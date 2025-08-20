import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "twist", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("failure_needed", 600, "Number of successful transitions to collect.")
flags.DEFINE_string("resume_success_file", "", "Path to previous success demo file.")
flags.DEFINE_string("resume_failure_file", "", "Path to previous failure demo file.")

# 全局按键标志位
key_flags = {
    "success": False,
    "pause": False,
    "failure": False,
}

def on_press(key):
    try:
        if key == keyboard.Key.space:
            key_flags["failure"] = True
        elif key == keyboard.KeyCode.from_char('p'):
            key_flags["pause"] = True
            print("⏸️ Pause requested. Saving current demos...")

    except AttributeError:
        pass

def load_demo_file(file_path):
    if file_path and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pkl.load(f)
    return []

def save_demos(failures, exp_name, failure_needed):
    os.makedirs(f"./classifier_data/classifier_data_{exp_name}", exist_ok=True)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # success_file = f"./classifier_data/{exp_name}_{success_needed}_success_images_{uuid}.pkl"
    failure_file = f"./classifier_data/classifier_data_{exp_name}/{exp_name}_failure_images_{uuid}.pkl"

    with open(failure_file, "wb") as f:
        pkl.dump(failures, f)
        print(f"✅ Saved {len(failures)} failure transitions to {failure_file}")

def main(_):
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print(f"[Start] Experiment: {FLAGS.exp_name}")
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, _ = env.reset()
    print("[Init] Environment reset finished")

    # successes = load_demo_file(FLAGS.resume_success_file)
    failures = load_demo_file(FLAGS.resume_failure_file)
    print(f"[Info] Loaded {len(failures)} failures from previous session.")

    failure_needed = FLAGS.failure_needed
    pbar = tqdm(total=failure_needed)
    pbar.update(len(failures))
    prev_actions = np.zeros(env.action_space.sample().shape)
    while len(failures) < failure_needed:
        if key_flags["pause"]:
            break

        actions = np.zeros(env.action_space.sample().shape)
        actions[6] = prev_actions[6]
        next_obs, rew, done, truncated, info = env.step(actions)

        if "intervene_action" in info:
            actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        obs = next_obs
        prev_actions = actions
        if key_flags["failure"]:
            failures.append(transition)
            print("✨ failures plus one ✨")
            pbar.update(1)
            key_flags["failure"] = False  # 重置按键状态


    save_demos(failures, FLAGS.exp_name, failure_needed)

if __name__ == "__main__":
    app.run(main)


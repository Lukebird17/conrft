#!/usr/bin/env python3
"""
Batch conversion utility for converting multiple demo files
"""

import os
import glob
import subprocess
import sys
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("demo_dir", "/home/ziyu/Project/VLA-RL/hil-serl/examples/demo_data", "Directory containing demo files")
flags.DEFINE_string("output_dir", "./demo_data", "Output directory for converted files")
flags.DEFINE_float("reward_scale", 1.0, "Reward scale")
flags.DEFINE_float("reward_bias", 0.0, "Reward bias")
flags.DEFINE_float("discount", 0.99, "Discount factor")
flags.DEFINE_string("conda_env", "conrft2", "Conda environment to use")

def find_demo_files(demo_dir):
    """Find all demo pickle files"""
    pattern = os.path.join(demo_dir, "*.pkl")
    files = glob.glob(pattern)
    # Filter out already converted files
    files = [f for f in files if "converted" not in os.path.basename(f)]
    return files

def convert_file(input_file, conda_env, reward_scale, reward_bias, discount, output_dir):
    """Convert a single demo file"""
    print(f"\n=== Converting {os.path.basename(input_file)} ===")
    
    cmd = [
        "conda", "run", "-n", conda_env, "python", "convert_demos_simple.py",
        f"--input_file={input_file}",
        f"--reward_scale={reward_scale}",
        f"--reward_bias={reward_bias}",
        f"--discount={discount}",
        f"--output_dir={output_dir}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print(f"✅ Successfully converted {os.path.basename(input_file)}")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ Failed to convert {os.path.basename(input_file)}")
            if result.stderr:
                print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"❌ Exception converting {os.path.basename(input_file)}: {e}")

def main(_):
    demo_files = find_demo_files(FLAGS.demo_dir)
    
    if not demo_files:
        print(f"No demo files found in {FLAGS.demo_dir}")
        return
    
    print(f"Found {len(demo_files)} demo files:")
    for f in demo_files:
        print(f"  - {os.path.basename(f)}")
    
    print(f"\nUsing conda environment: {FLAGS.conda_env}")
    print(f"Output directory: {FLAGS.output_dir}")
    print(f"Parameters: reward_scale={FLAGS.reward_scale}, reward_bias={FLAGS.reward_bias}, discount={FLAGS.discount}")
    
    # Create output directory if it doesn't exist
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    # Convert each file
    for demo_file in demo_files:
        convert_file(
            demo_file, 
            FLAGS.conda_env,
            FLAGS.reward_scale,
            FLAGS.reward_bias,
            FLAGS.discount,
            FLAGS.output_dir
        )
    
    print(f"\n=== Batch Conversion Complete ===")
    print(f"Processed {len(demo_files)} files")
    
    # Show converted files
    converted_files = glob.glob(os.path.join(FLAGS.output_dir, "*converted*.pkl"))
    if converted_files:
        print(f"\nConverted files ({len(converted_files)}):")
        for f in converted_files:
            print(f"  - {os.path.basename(f)}")

if __name__ == "__main__":
    app.run(main)

#!/usr/bin/env python3
"""
Convert demos recorded with record_demos.py to the format expected by record_demos_octo.py
This script adds the missing data processing steps that are required by the octo project.
"""

import os
import pickle as pkl
import copy
import datetime
import numpy as np
from absl import app, flags
from tqdm import tqdm

from experiments.mappings import CONFIG_MAPPING
from data_util import add_mc_returns_to_trajectory, add_embeddings_to_trajectory, add_next_embeddings_to_trajectory
from octo.model.octo_model import OctoModel

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "Path to the input demo file (recorded with record_demos.py)")
flags.DEFINE_string("output_dir", "./demo_data", "Directory to save the converted demos")
flags.DEFINE_string("exp_name", "twist", "Experiment name (must match the original demo)")
flags.DEFINE_float("reward_scale", 1.0, "Reward scale factor")
flags.DEFINE_float("reward_bias", 0.0, "Reward bias")
flags.DEFINE_float("discount", 0.99, "Discount factor for MC returns")

def load_original_demos(file_path):
    """Load demos recorded with record_demos.py"""
    print(f"Loading original demos from: {file_path}")
    with open(file_path, "rb") as f:
        transitions = pkl.load(f)
    print(f"Loaded {len(transitions)} transitions")
    return transitions

def group_transitions_into_trajectories(transitions):
    """Group flat transitions back into trajectories based on episode boundaries"""
    trajectories = []
    current_trajectory = []
    
    for transition in transitions:
        current_trajectory.append(transition)
        
        # If this transition ends an episode, save the trajectory
        if transition["dones"]:
            trajectories.append(current_trajectory)
            current_trajectory = []
    
    # Add the last trajectory if it doesn't end with done=True
    if current_trajectory:
        trajectories.append(current_trajectory)
    
    print(f"Grouped into {len(trajectories)} trajectories")
    return trajectories

def convert_demos(input_file, exp_name, reward_scale, reward_bias, discount, output_dir):
    """Convert demos from record_demos.py format to record_demos_octo.py format"""
    
    # Load configuration and model
    assert exp_name in CONFIG_MAPPING, f'Experiment {exp_name} not found in CONFIG_MAPPING.'
    config = CONFIG_MAPPING[exp_name]()
    
    print("Loading Octo model...")
    model = OctoModel.load_pretrained(config.octo_path)
    tasks = model.create_tasks(texts=[config.task_desc])
    print("Model loaded successfully")
    
    # Load original transitions
    original_transitions = load_original_demos(input_file)
    
    # Group transitions into trajectories
    trajectories = group_transitions_into_trajectories(original_transitions)
    
    # Process each trajectory
    converted_transitions = []
    print("Converting trajectories...")
    
    for i, trajectory in enumerate(tqdm(trajectories, desc="Processing trajectories")):
        try:
            # Add MC returns
            processed_trajectory = add_mc_returns_to_trajectory(
                trajectory, 
                discount, 
                reward_scale, 
                reward_bias, 
                config.reward_neg, 
                is_sparse_reward=True
            )
            
            # Add embeddings
            processed_trajectory = add_embeddings_to_trajectory(
                processed_trajectory, 
                model, 
                tasks=tasks
            )
            
            # Add next embeddings
            processed_trajectory = add_next_embeddings_to_trajectory(processed_trajectory)
            
            # Add processed transitions to the final list
            for transition in processed_trajectory:
                converted_transitions.append(copy.deepcopy(transition))
                
        except Exception as e:
            print(f"Error processing trajectory {i}: {e}")
            print("Skipping this trajectory...")
            continue
    
    print(f"Successfully converted {len(converted_transitions)} transitions from {len(trajectories)} trajectories")
    
    # Save converted demos
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract original filename info
    input_basename = os.path.basename(input_file)
    if input_basename.endswith('.pkl'):
        input_basename = input_basename[:-4]
    
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_dir, f"{input_basename}_converted_octo_{uuid}.pkl")
    
    with open(output_file, "wb") as f:
        pkl.dump(converted_transitions, f)
    
    print(f"Converted demos saved to: {output_file}")
    return output_file

def validate_converted_demos(file_path):
    """Validate that the converted demos have the expected format"""
    print(f"\nValidating converted demos: {file_path}")
    
    with open(file_path, "rb") as f:
        transitions = pkl.load(f)
    
    if not transitions:
        print("❌ No transitions found!")
        return False
    
    sample_transition = transitions[0]
    required_keys = ["observations", "actions", "next_observations", "rewards", "masks", "dones", "infos"]
    
    print(f"✅ Found {len(transitions)} transitions")
    print("✅ Checking required keys...")
    
    for key in required_keys:
        if key not in sample_transition:
            print(f"❌ Missing key: {key}")
            return False
        print(f"  ✅ {key}: {type(sample_transition[key])}")
    
    # Check if octo-specific fields were added
    if "mc_returns" in sample_transition:
        print("  ✅ mc_returns: Added successfully")
    else:
        print("  ⚠️  mc_returns: Not found (might be in infos)")
    
    # Check for embeddings in observations
    if "embeddings" in sample_transition["observations"]:
        print("  ✅ embeddings: Added successfully")
    else:
        print("  ⚠️  embeddings: Not found in observations")
    
    print("✅ Validation completed!")
    return True

def main(_):
    if FLAGS.input_file is None:
        print("Please specify --input_file")
        return
    
    if FLAGS.exp_name is None:
        print("Please specify --exp_name")
        return
    
    if not os.path.exists(FLAGS.input_file):
        print(f"Input file not found: {FLAGS.input_file}")
        return
    
    print("=== Demo Conversion Tool ===")
    print(f"Input file: {FLAGS.input_file}")
    print(f"Experiment: {FLAGS.exp_name}")
    print(f"Reward scale: {FLAGS.reward_scale}")
    print(f"Reward bias: {FLAGS.reward_bias}")
    print(f"Discount: {FLAGS.discount}")
    print(f"Output directory: {FLAGS.output_dir}")
    print()
    
    try:
        output_file = convert_demos(
            FLAGS.input_file,
            FLAGS.exp_name,
            FLAGS.reward_scale,
            FLAGS.reward_bias,
            FLAGS.discount,
            FLAGS.output_dir
        )
        
        # Validate the converted demos
        validate_converted_demos(output_file)
        
        print("\n=== Conversion Complete ===")
        print(f"Converted demos saved to: {output_file}")
        print("You can now use this file with the octo project!")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    app.run(main)

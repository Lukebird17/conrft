#!/usr/bin/env python3
"""
Simplified demo conversion script that doesn't require full environment setup.
This script only performs the basic data structure conversion without loading models or environments.
"""

import os
import pickle as pkl
import copy
import datetime
import numpy as np
from absl import app, flags
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "Path to the input demo file (recorded with record_demos.py)")
flags.DEFINE_string("output_dir", "./demo_data", "Directory to save the converted demos")
flags.DEFINE_string("output_suffix", "converted_simple", "Suffix for the output file")
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

def add_mc_returns_simple(trajectory, discount=0.99, reward_scale=1.0, reward_bias=0.0):
    """
    Simple implementation of Monte Carlo returns calculation
    without requiring external dependencies
    """
    # Calculate returns from the end backwards
    returns = []
    G = 0
    
    # Go backwards through trajectory
    for i in reversed(range(len(trajectory))):
        reward = trajectory[i]["rewards"]
        # Apply reward scaling and bias
        scaled_reward = reward * reward_scale + reward_bias
        G = scaled_reward + discount * G
        returns.append(G)
    
    # Reverse to get forward order
    returns.reverse()
    
    # Add returns to each transition
    for i, transition in enumerate(trajectory):
        if "infos" not in transition:
            transition["infos"] = {}
        transition["infos"]["mc_returns"] = returns[i]
        # Also add as a separate key for compatibility
        transition["mc_returns"] = returns[i]
    
    return trajectory

def add_dummy_embeddings(trajectory):
    """
    Add dummy embeddings to make the data structure compatible
    Real embeddings would require loading the Octo model
    """
    embedding_dim = 512  # Common embedding dimension
    
    for transition in trajectory:
        # Add dummy embeddings to observations
        if "embeddings" not in transition["observations"]:
            # Create dummy embedding based on observation shape
            dummy_embedding = np.zeros(embedding_dim, dtype=np.float32)
            transition["observations"]["embeddings"] = dummy_embedding
        
        # Add dummy embeddings to next_observations
        if "embeddings" not in transition["next_observations"]:
            dummy_embedding = np.zeros(embedding_dim, dtype=np.float32)
            transition["next_observations"]["embeddings"] = dummy_embedding
    
    return trajectory

def add_next_embeddings_simple(trajectory):
    """
    Simple implementation to add next embeddings
    """
    for i in range(len(trajectory)):
        if i < len(trajectory) - 1:
            # Use next observation's embedding
            next_embedding = trajectory[i + 1]["observations"]["embeddings"]
        else:
            # For last transition, use current embedding
            next_embedding = trajectory[i]["observations"]["embeddings"]
        
        trajectory[i]["observations"]["next_embeddings"] = next_embedding.copy()
    
    return trajectory

def convert_demos_simple(input_file, reward_scale, reward_bias, discount, output_dir, output_suffix):
    """Convert demos without requiring full environment setup"""
    
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
            processed_trajectory = add_mc_returns_simple(
                trajectory, 
                discount=discount,
                reward_scale=reward_scale, 
                reward_bias=reward_bias
            )
            
            # Add dummy embeddings (placeholder for real Octo embeddings)
            processed_trajectory = add_dummy_embeddings(processed_trajectory)
            
            # Add next embeddings
            processed_trajectory = add_next_embeddings_simple(processed_trajectory)
            
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
    output_file = os.path.join(output_dir, f"{input_basename}_{output_suffix}_{uuid}.pkl")
    
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
    
    # Check if new fields were added
    if "mc_returns" in sample_transition:
        print("  ✅ mc_returns: Added successfully")
    else:
        print("  ⚠️  mc_returns: Not found")
    
    # Check for embeddings in observations
    if isinstance(sample_transition["observations"], dict) and "embeddings" in sample_transition["observations"]:
        print("  ✅ embeddings: Added successfully")
        print(f"     Shape: {sample_transition['observations']['embeddings'].shape}")
    else:
        print("  ⚠️  embeddings: Not found in observations")
    
    print("✅ Validation completed!")
    return True

def main(_):
    if FLAGS.input_file is None:
        print("Please specify --input_file")
        return
    
    if not os.path.exists(FLAGS.input_file):
        print(f"Input file not found: {FLAGS.input_file}")
        return
    
    print("=== Simple Demo Conversion Tool ===")
    print(f"Input file: {FLAGS.input_file}")
    print(f"Reward scale: {FLAGS.reward_scale}")
    print(f"Reward bias: {FLAGS.reward_bias}")
    print(f"Discount: {FLAGS.discount}")
    print(f"Output directory: {FLAGS.output_dir}")
    print()
    print("Note: This creates dummy embeddings. For real embeddings, use the full conversion script.")
    print()
    
    try:
        output_file = convert_demos_simple(
            FLAGS.input_file,
            FLAGS.reward_scale,
            FLAGS.reward_bias,
            FLAGS.discount,
            FLAGS.output_dir,
            FLAGS.output_suffix
        )
        
        # Validate the converted demos
        validate_converted_demos(output_file)
        
        print("\n=== Conversion Complete ===")
        print(f"Converted demos saved to: {output_file}")
        print("\nIMPORTANT: This file contains dummy embeddings!")
        print("For real Octo embeddings, you'll need to run the full conversion script")
        print("in an environment with proper X server setup.")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    app.run(main)

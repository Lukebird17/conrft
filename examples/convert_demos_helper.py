#!/usr/bin/env python3
"""
Simple conversion script usage example and batch processing utility
"""

import os
import glob
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("demo_dir", "./demo_data", "Directory containing original demo files")
flags.DEFINE_string("exp_name", None, "Experiment name")
flags.DEFINE_boolean("batch", False, "Process all .pkl files in the directory")

def main(_):
    if FLAGS.exp_name is None:
        print("Please specify --exp_name")
        return
    
    if FLAGS.batch:
        # Process all pkl files in the directory
        demo_files = glob.glob(os.path.join(FLAGS.demo_dir, "*.pkl"))
        demo_files = [f for f in demo_files if "converted_octo" not in f]  # Skip already converted files
        
        if not demo_files:
            print(f"No demo files found in {FLAGS.demo_dir}")
            return
        
        print(f"Found {len(demo_files)} demo files to convert:")
        for f in demo_files:
            print(f"  - {f}")
        
        print("\nConverting all files...")
        for demo_file in demo_files:
            print(f"\n=== Converting {demo_file} ===")
            cmd = f"python convert_demos.py --input_file={demo_file} --exp_name={FLAGS.exp_name}"
            print(f"Running: {cmd}")
            os.system(cmd)
    else:
        # Interactive mode
        print("=== Demo Conversion Helper ===")
        print(f"Looking for demo files in: {FLAGS.demo_dir}")
        
        demo_files = glob.glob(os.path.join(FLAGS.demo_dir, "*.pkl"))
        demo_files = [f for f in demo_files if "converted_octo" not in f]
        
        if not demo_files:
            print(f"No demo files found in {FLAGS.demo_dir}")
            return
        
        print("\nAvailable demo files:")
        for i, f in enumerate(demo_files):
            print(f"  {i+1}. {os.path.basename(f)}")
        
        try:
            choice = int(input(f"\nSelect file to convert (1-{len(demo_files)}): ")) - 1
            if 0 <= choice < len(demo_files):
                selected_file = demo_files[choice]
                cmd = f"python convert_demos.py --input_file={selected_file} --exp_name={FLAGS.exp_name}"
                print(f"\nRunning: {cmd}")
                os.system(cmd)
            else:
                print("Invalid selection!")
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")

if __name__ == "__main__":
    app.run(main)

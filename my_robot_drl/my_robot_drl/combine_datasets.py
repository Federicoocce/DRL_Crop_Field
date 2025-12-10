import os
import pickle
import sys
import gc
import joblib
import numpy as np

HOME_DIR = os.path.expanduser('~')
DATASET_DIR = os.path.join(HOME_DIR, 'ros2_ws', 'drl_datasets', 'imitation_learning')
OUTPUT_DIR = os.path.join(DATASET_DIR, 'sharded_dataset')
CHUNK_SIZE = 1000  # Number of samples per file

def combine_and_shard_datasets():
    print("="*60)
    print("--- Starting Dataset Chunking ---")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        all_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.pkl') and 'sharded' not in f])
    except FileNotFoundError:
        print(f"Error: Directory {DATASET_DIR} does not exist.")
        sys.exit(1)

    if not all_files:
        print("No .pkl datasets found.")
        sys.exit(0)

    print(f"Found {len(all_files)} files. Merging and splitting into chunks of {CHUNK_SIZE}...")

    buffer = []
    chunk_index = 0
    total_samples = 0

    for i, filename in enumerate(all_files):
        file_path = os.path.join(DATASET_DIR, filename)
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                buffer.extend(data)
                
            # While we have enough data to make full chunks
            while len(buffer) >= CHUNK_SIZE:
                chunk_data = buffer[:CHUNK_SIZE]
                buffer = buffer[CHUNK_SIZE:]
                
                output_name = os.path.join(OUTPUT_DIR, f"chunk_{chunk_index:05d}.pkl")
                joblib.dump(chunk_data, output_name, compress=0)
                print(f"  -> Saved {output_name} ({len(chunk_data)} samples)")
                
                chunk_index += 1
                total_samples += len(chunk_data)

        except Exception as e:
            print(f"WARNING: Failed to load {filename}: {e}")
            continue
        
        # Force garbage collection
        gc.collect()

    # Save remaining data
    if buffer:
        output_name = os.path.join(OUTPUT_DIR, f"chunk_{chunk_index:05d}.pkl")
        joblib.dump(buffer, output_name, compress=0)
        print(f"  -> Saved {output_name} ({len(buffer)} samples)")
        total_samples += len(buffer)

    print("="*60)
    print(f"Done. Total samples: {total_samples}")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    combine_and_shard_datasets()